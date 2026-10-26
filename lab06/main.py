import math
import random
from pathlib import Path

import matplotlib.pyplot as plt


folder = Path(__file__).resolve().parent
plots = folder / "plots"
plots.mkdir(exist_ok=True)

random.seed(5)
digits = list(range(10))
sample_sizes = [50, 100, 200, 500, 1000, 2000, 3000]


def probabilities(numbers):
    counts = [numbers.count(i) for i in digits]
    probs = [count / len(numbers) for count in counts]
    return counts, probs


def entropy(probs):
    return -sum(p * math.log2(p) for p in probs if p > 0)


def save_plot(name):
    plt.tight_layout()
    plt.savefig(plots / name, dpi=150)
    plt.close()


numbers = [random.randint(0, 9) for _ in range(100)]
counts, probs = probabilities(numbers)
infos = [-math.log2(p) if p > 0 else None for p in probs]

rows = []
for i in digits:
    info_text = f"{infos[i]:.4f}" if infos[i] is not None else "не визначено"
    row = f"{i} | {counts[i]} | {probs[i]:.4f} | {info_text}"
    rows.append(f"| {row} |")

plt.figure(figsize=(8, 5))
plt.bar(digits, probs, color="lightgreen", edgecolor="black")
plt.axhline(0.1, color="red", linestyle="--", label="Теоретична ймовірність 0,1")
plt.title("Імовірності появи цифр, N = 100")
plt.xlabel("Цифра")
plt.ylabel("Імовірність")
plt.xticks(digits)
plt.legend()
plt.grid(axis="y", alpha=0.3)
save_plot("probabilities.png")

plt.figure(figsize=(8, 5))
present = [i for i in digits if infos[i] is not None]
plt.bar(present, [infos[i] for i in present], color="skyblue", edgecolor="black")
plt.title("Кількість інформації при появі цифри, N = 100")
plt.xlabel("Цифра")
plt.ylabel("Інформація, біт")
plt.xticks(digits)
plt.grid(axis="y", alpha=0.3)
save_plot("information.png")

entropies = []
entropy_rows = []
for n in sample_sizes:
    sample = numbers if n == 100 else [random.randint(0, 9) for _ in range(n)]
    _, sample_probs = probabilities(sample)
    h = entropy(sample_probs)
    entropies.append(h)
    entropy_rows.append(f"| {n} | {h:.4f} |")

h_max = math.log2(10)
plt.figure(figsize=(8, 5))
plt.plot(sample_sizes, entropies, "o-", label="Оцінка ентропії")
plt.axhline(h_max, color="red", linestyle="--", label=f"Максимум: {h_max:.4f}")
plt.title("Залежність ентропії від розміру вибірки")
plt.xlabel("Розмір вибірки N")
plt.ylabel("Ентропія, біт/символ")
plt.legend()
plt.grid(alpha=0.3)
save_plot("entropy.png")

report = f"""# Лабораторна робота №6 — Кількісна оцінка інформації

## Вибірка N = 100

```text
{numbers}
```

| Цифра | Кількість появ | Імовірність | Інформація, біт |
|---|---|---|---|
{chr(10).join(rows)}

![Імовірності появи цифр](plots/probabilities.png)

![Кількість інформації при появі цифри](plots/information.png)

## Ентропія для різних розмірів вибірки

| N | Ентропія, біт/символ |
|---|---|
{chr(10).join(entropy_rows)}

![Залежність ентропії від розміру вибірки](plots/entropy.png)
"""
(folder / "README.md").write_text(report, encoding="utf-8")
print("\nГрафіки збережено в", plots)
print("Результати з графіками:", folder / "README.md")
