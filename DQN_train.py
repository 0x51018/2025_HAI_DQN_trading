# 파일명: dqn_trading_agent.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# ─── 0. 데이터 로드 ─────────────────────────────────────────
df = pd.read_csv("prices/AAPL.csv", index_col=0)
open_prices  = df["open"].values.tolist()
close_prices = df["close"].values.tolist()

# ─── 1. 디바이스 설정 ───────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ─── 2. 네트워크 정의 ───────────────────────────────────────
class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers=[256]):
        """
        state_size: 입력 특성 크기
        action_size: 가능한 행동 수 (보통 3: 0=Hold, 1=Buy, 2=Sell)
        hidden_layers: 은닉층 노드 수 리스트
        """
        super().__init__()
        layers = []
        in_dim = state_size
        # 은닉층 구성
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        # 출력층
        layers.append(nn.Linear(in_dim, action_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# ─── 3. 에이전트 정의 ───────────────────────────────────────
class Agent:
    def __init__(self,
                 state_size, window_size, trend, skip, batch_size,
                 open_list, close_list,
                 reward_fn=None,
                 network_class=DQNetwork, network_kwargs=None):
        """
        reward_fn: (agent, t, action) -> float 형태의 함수. None이면 기본 보상 함수 사용
        network_class: DQN 네트워크 클래스로, DQNetwork 외 다른 클래스도 주입 가능
        network_kwargs: 네트워크 생성자에 넘길 추가 키워드 인자 dict
        """
        # 원래 인자들
        self.state_size  = state_size
        self.window_size = window_size
        self.half_window = window_size // 2
        self.trend       = trend
        self.open        = open_list
        self.close       = close_list
        self.skip        = skip
        self.action_size = 3
        self.batch_size  = batch_size
        self.memory      = deque(maxlen=1000)
        self.inventory   = []

        # 하이퍼파라미터
        self.gamma         = 0.95
        self.epsilon       = 0.5
        self.epsilon_min   = 0.01
        self.epsilon_decay = 0.999

        # 보상 함수 주입
        if reward_fn is not None:
            self.reward_fn = reward_fn
        else:
            # 기본 보상 함수: 기존 로직 그대로
            self.reward_fn = self._default_reward

        # 디바이스 및 네트워크 초기화
        self.device = torch.device(DEVICE)
        net_kwargs = network_kwargs if network_kwargs else {}
        self.model     = network_class(state_size, self.action_size, **net_kwargs).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        self.criterion = nn.MSELoss()

    def _default_reward(self, agent, t, action):
        """원본 코드의 보상 계산 로직을 함수로 분리"""
        reward = 0.0
        # 매수
        if action == 1 and t < len(self.trend) - self.half_window:
            # 매수 시점 기록
            self.inventory.append(self.trend[t])
            # 기준가: 직전 시가/종가/현재가 중 최대
            ref = (max(self.open[t-1], self.close[t-1], self.trend[t])
                   if t > 0 else self.trend[t])
            reward = (ref - self.trend[t]) / self.trend[t]
        # 매도
        elif action == 2 and self.inventory:
            bought = self.inventory.pop(0)
            reward = (self.trend[t] - bought) / self.trend[t]
        return reward

    def act(self, state):
        """ε-그리디 정책으로 행동 선택"""
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def get_state(self, t):
        ws = self.window_size + 1
        d = t - ws + 1
        if d >= 0:
            block = self.trend[d:t + 1]
        else:
            pad_len = -d
            block = [self.trend[0]] * pad_len + self.trend[0:t + 1]
        # 🔒 안전 장치 추가
        if len(block) != ws:
            block = block[:ws]  # 잘못되면 자르기

        diffs = [block[i + 1] - block[i] for i in range(len(block) - 1)]
        return np.array(diffs, dtype=np.float32)
    

    def replay(self):
        """경험 재플레이 및 모델 업데이트"""
        if len(self.memory) < self.batch_size:
            return
        mini = random.sample(self.memory, self.batch_size)
        s, a, r, ns, done = zip(*mini)

        states      = torch.tensor(s, dtype=torch.float32).to(self.device)
        actions     = torch.tensor(a, dtype=torch.long).to(self.device)
        rewards     = torch.tensor(r, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(ns, dtype=torch.float32).to(self.device)
        dones       = torch.tensor(done, dtype=torch.float32).to(self.device)

        curr_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q = self.model(next_states).max(1)[0].detach()
        target = rewards + (1 - dones) * self.gamma * next_q

        loss = self.criterion(curr_q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, episodes):
        """에피소드 단위 학습"""
        for ep in range(episodes):
            state = self.get_state(0)
            total_profit = 0.0

            for t in range(0, len(self.trend)-1, self.skip):
                action = self.act(state)
                nxt    = self.get_state(t+1)
                # 기본 보상 계산 (inventory 조작은 _default_reward 내부에서 처리)
                default_reward = self._default_reward(self, t, action)
                # 주입된 보상 함수 호출
                reward = self.reward_fn(self, t, action)
                # total_profit 집계 (발표용 로그 기준)
                if action == 2 and default_reward != 0:
                    total_profit += default_reward

                self.memory.append((state, action, reward, nxt, (t == len(self.trend)-2)))
                state = nxt
                self.replay()

            print(f"Episode {ep+1}/{episodes}, Total Profit: {total_profit:.4f}")

    def buy(self, initial_money):
        """훈련된 정책으로 매매 시뮬레이션"""
        state   = self.get_state(0)
        balance = initial_money
        buys, sells = [], []

        for t in range(0, len(self.trend)-1, self.skip):
            action = self.act(state)
            state  = self.get_state(t+1)
            price  = self.trend[t]

            if action == 1 and balance >= price:
                self.inventory.append(price)
                balance -= price
                buys.append(t)
            elif action == 2 and self.inventory:
                bought = self.inventory.pop(0)
                balance += price
                sells.append(t)

        gain    = balance - initial_money
        invest  = gain / initial_money * 100
        return buys, sells, gain, invest

# ─── 4. 학습 및 트레이드 시뮬레이션 ─────────────────────────
initial_money = 10000
window_size   = 30
skip          = 1
batch_size    = 32

agent = Agent(
    state_size=window_size,
    window_size=window_size,
    trend=close_prices,
    skip=skip,
    batch_size=batch_size,
    open_list=open_prices,
    close_list=close_prices,
    # reward_fn=None,                  # 기본 보상 사용
    # network_class=DQNetwork,         # 기본 네트워크 사용
    # network_kwargs={"hidden_layers":[256]}  # 기본 은닉 256
)

agent.train(episodes=20)
buys, sells, total_gain, invest_pct = agent.buy(initial_money)

# ─── 5. 결과 시각화 및 파일 저장 ─────────────────────────────
import matplotlib.pyplot as plt
plt.figure(figsize=(15,5))
plt.plot(close_prices, color='black', lw=2)
plt.plot(close_prices, '^', markersize=10, color='blue',
         label='Buy Signal',  markevery=buys)
plt.plot(close_prices, 'v', markersize=10, color='red',
         label='Sell Signal', markevery=sells)
plt.title(f"Total Gain: {total_gain:.4f}, ROI: {invest_pct:.2f}%")
plt.legend()
plt.savefig("trade_signals.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved Png")
# ─── 6. 결과 요약 파일 저장 ───────────────────────────────