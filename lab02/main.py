import os
import matplotlib.pyplot as plt
import numpy as np


class DiscreteSignalOperations:

    @staticmethod
    def scale_amplitude(n, x, a):
        """Масштабування амплітуди: y[n] = a * x[n]"""
        return n.copy(), a * x

    @staticmethod
    def time_reversal(n, x):
        """Реверс у часі: y[n] = x[-n]"""
        return -n[::-1], x[::-1]

    @staticmethod
    def time_shift(n, x, N):
        """Зсув у часі: y[n] = x[n - N]"""
        return n + N, x.copy()

    @staticmethod
    def decimation(n, x, a):
        """Розширення/стиснення у часі (децимація): y[n] = x[a * n]"""
        if not isinstance(a, int) or a <= 0:
            raise ValueError(
                "Коефіцієнт децимації a повинен бути додатним цілим числом!"
            )

        mask = n % a == 0
        new_n = (n[mask] / a).astype(int)
        new_x = x[mask]
        return new_n, new_x

    @staticmethod
    def add(n1, x1, n2, x2):
        """Додавання двох сигналів: y[n] = x1[n] + x2[n]"""
        min_n = min(n1.min(), n2.min())
        max_n = max(n1.max(), n2.max())
        common_n = np.arange(min_n, max_n + 1)

        y1 = np.zeros_like(common_n, dtype=float)
        y2 = np.zeros_like(common_n, dtype=float)

        y1[np.isin(common_n, n1)] = x1
        y2[np.isin(common_n, n2)] = x2

        return common_n, y1 + y2

    @staticmethod
    def multiply(n1, x1, n2, x2):
        """Множення двох сигналів: y[n] = x1[n] * x2[n]"""
        min_n = min(n1.min(), n2.min())
        max_n = max(n1.max(), n2.max())
        common_n = np.arange(min_n, max_n + 1)

        y1 = np.zeros_like(common_n, dtype=float)
        y2 = np.zeros_like(common_n, dtype=float)

        y1[np.isin(common_n, n1)] = x1
        y2[np.isin(common_n, n2)] = x2

        return common_n, y1 * y2


OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_plot(filename, title):
    """Налаштовує підписи, зберігає поточну фігуру в папку plots і закриває її."""
    plt.title(title, fontsize=12)
    plt.xlabel("n", fontsize=10)
    plt.ylabel("Амплітуда", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Збережено: {filepath}")


# ==========================================================
# ОБЧИСЛЕННЯ ТА ЗБЕРЕЖЕННЯ ГРАФІКІВ
# ==========================================================

# Генерація сигналів
n = np.arange(-10, 11)
x = np.sin(0.4 * np.pi * n)
x2 = np.where(n >= 0, 1.0, 0.0)

ops = DiscreteSignalOperations()

# 1. Масштабування амплітуди
n_scaled, x_scaled = ops.scale_amplitude(n, x, a=2.5)
plt.figure(figsize=(8, 4.5))
plt.stem(
    n, x, linefmt="b-", markerfmt="bo", basefmt="k-", label="Оригінал x[n]"
)
plt.stem(
    n_scaled,
    x_scaled,
    linefmt="r--",
    markerfmt="ro",
    basefmt="k-",
    label="Масштабований y[n] = 2.5*x[n]",
)
save_plot("1_scaling.png", "1. Масштабування амплітуди")

# 2. Реверс у часі
n_rev, x_rev = ops.time_reversal(n, x)
plt.figure(figsize=(8, 4.5))
plt.stem(
    n, x, linefmt="b-", markerfmt="bo", basefmt="k-", label="Оригінал x[n]"
)
plt.stem(
    n_rev,
    x_rev,
    linefmt="g--",
    markerfmt="go",
    basefmt="k-",
    label="Реверс y[n] = x[-n]",
)
save_plot("2_time_reversal.png", "2. Реверс у часі")

# 3. Зсув у часі
n_shift, x_shift = ops.time_shift(n, x, N=4)
plt.figure(figsize=(8, 4.5))
plt.stem(
    n, x, linefmt="b-", markerfmt="bo", basefmt="k-", label="Оригінал x[n]"
)
plt.stem(
    n_shift,
    x_shift,
    linefmt="m--",
    markerfmt="mo",
    basefmt="k-",
    label="Зсув y[n] = x[n - 4] (затримка)",
)
save_plot("3_time_shift.png", "3. Зсув у часі")

# 4. Децимація (стиснення за часом)
n_dec, x_dec = ops.decimation(n, x, a=2)
plt.figure(figsize=(8, 4.5))
plt.stem(
    n, x, linefmt="b-", markerfmt="bo", basefmt="k-", label="Оригінал x[n]"
)
plt.stem(
    n_dec,
    x_dec,
    linefmt="c--",
    markerfmt="co",
    basefmt="k-",
    label="Децимація y[n] = x[2n] (a=2)",
)
save_plot("4_decimation.png", "4. Масштабування часу (децимація)")

# 5. Додавання сигналів
n_add, x_add = ops.add(n, x, n, x2)
plt.figure(figsize=(8, 4.5))
plt.stem(
    n_add,
    x_add,
    linefmt="darkorange",
    markerfmt="o",
    basefmt="k-",
    label="y[n] = x1[n] + x2[n]",
)
save_plot(
    "5_addition.png", "5. Додавання сигналів (синусоїда + сходинка Хевісайда)"
)

# 6. Множення сигналів
n_mul, x_mul = ops.multiply(n, x, n, x2)
plt.figure(figsize=(8, 4.5))
plt.stem(
    n_mul,
    x_mul,
    linefmt="purple",
    markerfmt="o",
    basefmt="k-",
    label="y[n] = x1[n] * x2[n]",
)
save_plot(
    "6_multiplication.png", "6. Множення сигналів (стробування/вирізання)"
)

print("\nУсі графіки успішно згенеровано та збережено в папку 'plots/'!")