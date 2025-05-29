import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# ─── 0. 데이터 로드 ─────────────────────────────────────────
df = pd.read_csv("prices/AAPL.csv", index_col=0)
# Normalize column names to lowercase so 'Open'/'Close' become 'open'/'close'
df.columns = df.columns.str.lower()
open_prices  = df["open"].values.tolist()
close_prices = df["close"].values.tolist()

# ─── 1. 디바이스 설정 ───────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ─── 2. 네트워크 정의 ───────────────────────────────────────
class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1  = nn.Linear(state_size, 256)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(256, action_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# ─── 3. 에이전트 정의 ───────────────────────────────────────
class Agent:
    def __init__(self, state_size, window_size, trend, skip, batch_size, open_list, close_list):
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

        self.gamma         = 0.95
        self.epsilon       = 0.5
        self.epsilon_min   = 0.01
        self.epsilon_decay = 0.999

        self.device    = torch.device(DEVICE)
        self.model     = DQNetwork(state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        self.criterion = nn.MSELoss()

    def act(self, state):
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
        for ep in range(episodes):
            state = self.get_state(0)
            total_profit = 0.0
            for t in range(0, len(self.trend)-1, self.skip):
                action = self.act(state)
                nxt   = self.get_state(t+1)
                reward = 0.0
                done   = (t == len(self.trend)-2)

                if action == 1 and t < len(self.trend) - self.half_window:
                    self.inventory.append(self.trend[t])
                    ref = (max(self.open[t-1], self.close[t-1], self.trend[t]) 
                           if t>0 else self.trend[t])
                    reward = (ref - self.trend[t]) / self.trend[t]
                elif action == 2 and self.inventory:
                    bought = self.inventory.pop(0)
                    reward = (self.trend[t] - bought) / self.trend[t]
                    total_profit += reward

                self.memory.append((state, action, reward, nxt, done))
                state = nxt
                self.replay()

            print(f"Episode {ep+1}/{episodes}, Total Profit: {total_profit:.4f}")

    def buy(self, initial_money):
        state = self.get_state(0)
        balance = initial_money
        buys, sells = [], []
        for t in range(0, len(self.trend)-1, self.skip):
            action = self.act(state)
            state  = self.get_state(t+1)
            if action == 1 and balance >= self.trend[t]:
                self.inventory.append(self.trend[t])
                balance -= self.trend[t]
                buys.append(t)
            elif action == 2 and self.inventory:
                bought = self.inventory.pop(0)
                balance += self.trend[t]
                sells.append(t)
        gain    = balance - initial_money
        invest  = gain / initial_money * 100
        return buys, sells, gain, invest

# ─── 4. 학습 및 트레이드 시뮬레이션 ─────────────────────────
initial_money = 10000
window_size   = 30
skip          = 1
batch_size    = 32

agent = Agent(state_size=window_size,
              window_size=window_size,
              trend=close_prices,
              skip=skip,
              batch_size=batch_size,
              open_list=open_prices,
              close_list=close_prices)

agent.train(episodes=20)
buys, sells, total_gain, invest_pct = agent.buy(initial_money)

# ─── 5. 결과 시각화 및 파일 저장 ─────────────────────────────
plt.figure(figsize=(15,5))
plt.plot(close_prices, color='black', lw=2)
plt.plot(close_prices, '^', markersize=10, color='blue',
         label='Buy Signal',  markevery=buys)
plt.plot(close_prices, 'v', markersize=10, color='red',
         label='Sell Signal', markevery=sells)
plt.title(f"Total Gain: {total_gain:.4f}, ROI: {invest_pct:.2f}%")
plt.legend()

# 파일로 저장
plt.savefig("trade_signals.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved Png")
