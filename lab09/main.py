from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def discrete_convolution(x, y):
    result = [0] * (len(x) + len(y) - 1)
    for m in range(len(result)):
        for k in range(len(x)):
            if 0 <= m - k < len(y):
                result[m] += x[k] * y[m - k]
    return result


def plot_signals(x, y, result, pair, path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle(f"Варіант 5 — пара {pair}")
    for ax, values, label, color in zip(
        axes, (x, y, result), ("x[k]", "y[k]", "f[m]"), ("b", "g", "r")
    ):
        ax.stem(range(len(values)), values, linefmt=f"{color}-",
                markerfmt=f"{color}o", basefmt="k-")
        ax.set_title(label)
        ax.set_xlabel("m" if label == "f[m]" else "k")
        ax.set_ylabel(label)
        ax.set_xticks(range(len(values)))
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    datasets = [
        ([6, 7, 1], [5, 6, 7]),
        ([6, 4, 7, 9, 2, 3, 6, 4, 9], [8, 2, 7, 3, 6, 4, 0]),
        ([2, 3, 6, 4, 2, 4, 8, 2], [1, 2, 8, 6, 4, 0, 8, 3, 7, 6, 4, 0]),
    ]
    folder = Path(__file__).resolve().parent
    (folder / "plots").mkdir(exist_ok=True)
    lines = ["# Лабораторна робота №9 — результати, варіант 5", ""]

    for pair, (x, y) in enumerate(datasets, 1):
        result = discrete_convolution(x, y)
        matches = np.array_equal(result, np.convolve(x, y))
        assert matches, f"Пара {pair}: результат не збігається з NumPy"
        plot_signals(x, y, result, pair, folder / "plots" / f"pair_{pair}.png")
        lines.extend([
            f"## Пара {pair}", "",
            f"- x[k] = {x}",
            f"- y[k] = {y}",
            f"- f[m] = {result}",
            f"- Довжини: x — {len(x)}, y — {len(y)}, f — {len(result)}.",
            "- Перевірка з numpy.convolve: збігається.", "",
            f"![Сигнали та згортка, пара {pair}](plots/pair_{pair}.png)", "",
        ])
        print(f"Пара {pair}: f[m] = {result}; перевірка: {matches}")

    (folder / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
