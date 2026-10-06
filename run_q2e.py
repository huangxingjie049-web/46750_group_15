"""Q2(e).iv: price structure experiment."""
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data_loader import load_question
from src.model_q2_quadratic import QuadraticConsumerModel
from src.q2e_nonseparable import NonseparableConsumerModel

# 加载基础数据
d = load_question('Q2_quadratic')
hours = d.hours
T = len(hours)

# 场景A：交替价格（每小时切换）
price_alt = np.array([1.0, 2.0] * 12)

# 场景B：块状价格（连续6小时同方向）
price_block = np.array([1.0] * 6 + [2.0] * 6 + [1.0] * 6 + [2.0] * 6)

scenarios = {
    'alternating': price_alt,
    'block': price_block,
}

results = {}

for name, price in scenarios.items():
    # 改价格
    d_mod = d.__class__(**{**d.__dict__, 'energy_price': price})

    # 跑二次模型
    r_q = QuadraticConsumerModel(d_mod).build().solve()

    # 跑 non-separable 模型
    r_ns = NonseparableConsumerModel(d_mod).build().solve()

    results[name] = {
        'quadratic': r_q,
        'nonseparable': r_ns,
    }

    print(f"\n{name}:")
    print(
        f"  Quadratic: A={r_q.hourly['absolute_deviation'].sum():.2f}, E={r_q.hourly['load'].sum():.2f}, C={r_q.meta['procurement_cost']:.2f}")
    print(
        f"  Non-separable: A={r_ns.hourly['absolute_deviation'].sum():.2f}, E={r_ns.hourly['load'].sum():.2f}, C={r_ns.meta['procurement_cost']:.2f}")

# 画价格对比图
fig, ax = plt.subplots(figsize=(11, 3))
ax.step(hours, price_alt, where='mid', label='Alternating')
ax.step(hours, price_block, where='mid', label='Block')
ax.set(xlabel='Hour', ylabel='Price (DKK/kWh)', title='Price scenarios')
ax.legend()
ax.grid(alpha=.25)
ax.set_xticks(hours)
fig.tight_layout()

out = Path('results/Q2_nonseparable')
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / 'q2e_prices.png', dpi=180)
plt.close(fig)

print(f"\nSaved to {out / 'q2e_prices.png'}")
