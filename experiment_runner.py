# 파일명: experiment_runner.py

import numpy as np
import matplotlib.pyplot as plt
import random
from DQN_train import Agent, DQNetwork

# ─── 커스텀 보상 함수 정의 ─────────────────────────────────
def reward_log_return(agent, t, action):
    """로그 수익률 보상: log(curr_val / prev_val)"""
    # 이전 포트폴리오 가치는 기본 보상 로직에서 알기 어려우므로,
    # 단순히 가격 비율로 대체 예시
    prev = agent.trend[t]
    curr = agent.trend[t+1]
    return np.log(curr / prev)

def reward_with_cost(agent, t, action):
    """거래 비용 반영 보상: 기본 보상 -0.1%"""
    base = agent._default_reward(agent, t, action)
    cost = 0.001 * agent.trend[t] if action in [1,2] else 0
    return base - cost

def reward_risk_penalty(agent, t, action):
    """변동성 페널티 보상: 기본 보상 - (최근 5일 표준편차 * 0.1)"""
    base = agent._default_reward(agent, t, action)
    if t >= 5:
        vol = np.std(agent.trend[t-5:t])
        return base - 0.1 * vol
    else:
        return base

# ─── 실험 설정 리스트 ───────────────────────────────────────
experiments = [
    {
        "label": "baseline",
        "reward_fn": None,
        "network_kwargs": {"hidden_layers":[256]},
        "description": "기본 보상, 은닉256"
    },
    {
        "label": "log_return",
        "reward_fn": reward_log_return,
        "network_kwargs": {"hidden_layers":[256]},
        "description": "로그 수익률 보상, 은닉256"
    },
    {
        "label": "with_cost",
        "reward_fn": reward_with_cost,
        "network_kwargs": {"hidden_layers":[256]},
        "description": "거래 비용 페널티 보상, 은닉256"
    },
    {
        "label": "risk_penalty",
        "reward_fn": reward_risk_penalty,
        "network_kwargs": {"hidden_layers":[256]},
        "description": "리스크 페널티 보상, 은닉256"
    },
    {
        "label": "deep_net",
        "reward_fn": None,
        "network_kwargs": {"hidden_layers":[256,128,64]},
        "description": "기본 보상, 은닉256-128-64"
    },
    # 하이퍼파라미터 추가 실험 등….
]

# ─── 가격 데이터 로드 ───────────────────────────────────────
import pandas as pd
df = pd.read_csv("prices/AAPL.csv", index_col=0)
price_series = df["Close"].values.tolist()

# ─── 실험 루프 ─────────────────────────────────────────────
for idx, cfg in enumerate(experiments, start=1):
    print(f"\n=== Experiment {idx}: {cfg['label']} ===")
    agent = Agent(
        state_size=30,
        window_size=30,
        trend=price_series,
        skip=1,
        batch_size=32,
        open_list=df["open"].values.tolist(),
        close_list=df["close"].values.tolist(),
        reward_fn=cfg["reward_fn"],
        network_class=DQNetwork,
        network_kwargs=cfg["network_kwargs"]
    )
    # 학습 및 평가
    agent.train(episodes=20)
    buys, sells, gain, roi = agent.buy(10000)

    # 결과 시각화
    plt.figure(figsize=(10,4))
    plt.plot(price_series, label="Close Price")
    plt.plot(price_series, '^', markevery=buys, label="Buy", markersize=8)
    plt.plot(price_series, 'v', markevery=sells, label="Sell", markersize=8)
    title = f"Exp{idx}_{cfg['label']} | Gain: {gain:.2f}, ROI: {roi:.2f}%"
    plt.suptitle(title, fontsize=10)
    plt.legend()
    img_name = f"exp{idx}_{cfg['label']}.png"
    plt.savefig(img_name, dpi=150, bbox_inches="tight")
    plt.close()

    # 설정 설명 파일 저장
    txt_name = f"exp{idx}_{cfg['label']}.txt"
    with open(txt_name, "w") as f:
        f.write(f"Experiment {idx}: {cfg['description']}\n")
        f.write(f"Reward Function: {cfg['reward_fn'].__name__ if cfg['reward_fn'] else 'default'}\n")
        f.write(f"Network Hidden Layers: {cfg['network_kwargs']['hidden_layers']}\n")
        f.write(f"Final Gain: {gain:.2f}, ROI: {roi:.2f}%\n")

    print(f"저장 완료 → {img_name}, {txt_name}")
