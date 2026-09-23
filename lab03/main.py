from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


N_RANDOM = 200
MAX_LAG = 20
RANDOM_SEED = 5
OUTPUT_DIR = Path(__file__).resolve().parent / "plots"


def correlation_at_lag(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    """Кореляція r_xy(j) з нульовим доповненням за межами сигналу."""
    if x.shape != y.shape:
        raise ValueError("Сигнали повинні мати однакову довжину.")
    if lag < 0:
        return correlation_at_lag(y, x, -lag)

    n = len(x)
    if lag >= n:
        return 0.0

    # y[n + lag] існує лише для n = 0, ..., N-lag-1.
    # Відсутні відліки вважаємо нулями, а результат ділимо на повне N.
    return float(np.dot(x[: n - lag], y[lag:]) / n)


def correlation_vector(
    x: np.ndarray, y: np.ndarray, lags: np.ndarray
) -> np.ndarray:
    """Обчислює взаємну кореляцію для заданого набору зсувів."""
    return np.array([correlation_at_lag(x, y, int(lag)) for lag in lags])


def normalized_correlation_at_lag(
    x: np.ndarray, y: np.ndarray, lag: int = 0
) -> float:
    """Нормований коефіцієнт кореляції rho_xy(j) з методички."""
    denominator = np.sqrt(np.sum(x**2) * np.sum(y**2)) / len(x)
    if np.isclose(denominator, 0.0):
        raise ValueError("Неможливо нормувати сигнал із нульовою енергією.")
    return correlation_at_lag(x, y, lag) / denominator


def normalized_energy(x: np.ndarray) -> float:
    """Нормована енергія S = (1/N) * sum(x[n]^2)."""
    return float(np.mean(x**2))


def configure_plot(title: str, xlabel: str, ylabel: str) -> None:
    plt.title(title, fontsize=12)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle=":", alpha=0.65)
    plt.legend()
    plt.tight_layout()


def save_plot(filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()


def calculate_table_results() -> dict[str, float]:
    """Виконує розрахунки для таблиць 1 і 2 з методички."""
    # Порожні клітинки x2 у таблиці 1 доповнено нулями.
    table1_x1 = np.array([4, 2, -1, 3, -2, -6, -5, 4, 5], dtype=float)
    table1_x2 = np.array([7, 4, -2, -8, -2, -1, 0, 0, 0], dtype=float)
    table1_r12 = correlation_at_lag(table1_x1, table1_x2, lag=0)

    table2_x1 = np.array([0, 3, 5, 5, 5, 2, 0.5, 0.25, 0], dtype=float)
    table2_x2 = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=float)
    table2_x3 = np.array([0, 9, 15, 15, 15, 6, 1.5, 0.75, 0], dtype=float)
    table2_x4 = np.array([2, 2, 2, 2, 2, 0, 0, 0, 0], dtype=float)

    r12 = correlation_at_lag(table2_x1, table2_x2, lag=0)
    r34 = correlation_at_lag(table2_x3, table2_x4, lag=0)
    norm12 = np.sqrt(np.sum(table2_x1**2) * np.sum(table2_x2**2)) / len(
        table2_x1
    )
    norm34 = np.sqrt(np.sum(table2_x3**2) * np.sum(table2_x4**2)) / len(
        table2_x3
    )
    rho12 = normalized_correlation_at_lag(table2_x1, table2_x2)
    rho34 = normalized_correlation_at_lag(table2_x3, table2_x4)

    n = np.arange(len(table2_x1))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(n, table2_x1, "o-", label="x1[n]")
    axes[0].plot(n, table2_x3, "s--", label="x3[n] = 3x1[n]")
    axes[0].set_title("Сигнали однакової форми з різними амплітудами")
    axes[0].set_ylabel("Амплітуда")
    axes[0].grid(True, linestyle=":", alpha=0.65)
    axes[0].legend()

    axes[1].plot(n, table2_x2, "o-", label="x2[n]")
    axes[1].plot(n, table2_x4, "s--", label="x4[n] = 2x2[n]")
    axes[1].set_xlabel("n")
    axes[1].set_ylabel("Амплітуда")
    axes[1].grid(True, linestyle=":", alpha=0.65)
    axes[1].legend()
    fig.tight_layout()
    save_plot("1_table2_signals.png")

    return {
        "table1_r12": table1_r12,
        "table2_r12": r12,
        "table2_r34": r34,
        "norm12": norm12,
        "norm34": norm34,
        "rho12": rho12,
        "rho34": rho34,
    }


def analyze_random_signals() -> dict[str, float | int]:
    """Генерує два випадкові сигнали та виконує основну частину роботи."""
    rng = np.random.default_rng(RANDOM_SEED)
    x1 = rng.normal(loc=0.0, scale=1.0, size=N_RANDOM)
    x2 = rng.normal(loc=0.0, scale=1.0, size=N_RANDOM)
    n = np.arange(N_RANDOM)
    lags = np.arange(MAX_LAG + 1)

    cross_correlation = correlation_vector(x1, x2, lags)
    autocorrelation_x1 = correlation_vector(x1, x1, lags)
    autocorrelation_x2 = correlation_vector(x2, x2, lags)
    energy_x1 = normalized_energy(x1)
    energy_x2 = normalized_energy(x2)

    max_cross_index = int(np.argmax(np.abs(cross_correlation)))

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(n, x1, color="royalblue", linewidth=1.0, label="x1[n]")
    axes[0].set_title("Перший випадковий сигнал")
    axes[0].set_ylabel("Амплітуда")
    axes[0].grid(True, linestyle=":", alpha=0.65)
    axes[0].legend()

    axes[1].plot(n, x2, color="darkorange", linewidth=1.0, label="x2[n]")
    axes[1].set_title("Другий випадковий сигнал")
    axes[1].set_xlabel("Номер відліку n")
    axes[1].set_ylabel("Амплітуда")
    axes[1].grid(True, linestyle=":", alpha=0.65)
    axes[1].legend()
    fig.tight_layout()
    save_plot("2_random_signals.png")

    plt.figure(figsize=(9, 4.8))
    plt.stem(lags, cross_correlation, linefmt="g-", markerfmt="go", basefmt="k-")
    plt.plot([], [], "go-", label="r12(j)")
    configure_plot(
        "Взаємна кореляція випадкових сигналів",
        "Зсув j",
        "r12(j)",
    )
    save_plot("3_cross_correlation.png")

    plt.figure(figsize=(9, 4.8))
    plt.stem(
        lags,
        autocorrelation_x1,
        linefmt="b-",
        markerfmt="bo",
        basefmt="k-",
    )
    plt.plot([], [], "bo-", label="r11(j)")
    configure_plot("Автокореляція першого сигналу", "Зсув j", "r11(j)")
    save_plot("4_autocorrelation_x1.png")

    plt.figure(figsize=(9, 4.8))
    plt.stem(
        lags,
        autocorrelation_x2,
        linefmt="m-",
        markerfmt="mo",
        basefmt="k-",
    )
    plt.plot([], [], "mo-", label="r22(j)")
    configure_plot("Автокореляція другого сигналу", "Зсув j", "r22(j)")
    save_plot("5_autocorrelation_x2.png")

    plt.figure(figsize=(7, 4.8))
    bars = plt.bar(
        ["S1", "S2"],
        [energy_x1, energy_x2],
        color=["royalblue", "darkorange"],
        width=0.55,
        label="Нормована енергія",
    )
    for bar, value in zip(bars, [energy_x1, energy_x2]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )
    configure_plot("Порівняння енергій сигналів", "Сигнал", "S")
    save_plot("6_signal_energies.png")

    return {
        "energy_x1": energy_x1,
        "energy_x2": energy_x2,
        "max_cross_lag": int(lags[max_cross_index]),
        "max_cross_value": float(cross_correlation[max_cross_index]),
        "autocorrelation_x1_zero": float(autocorrelation_x1[0]),
        "autocorrelation_x2_zero": float(autocorrelation_x2[0]),
    }


def write_markdown_report(
    table_results: dict[str, float], random_results: dict[str, float | int]
) -> None:
    """Створює Markdown-звіт із результатами та вбудованими графіками."""
    report_path = Path(__file__).resolve().parent / "README.md"
    report = rf"""# Лабораторна робота №3

## Кореляція та автокореляція сигналів

Програма виконує розрахунки за методичкою, генерує два незалежні випадкові
сигнали та будує графіки їх взаємної кореляції й автокореляції.

## Результати для таблиці 1

Порожні клітинки другого сигналу доповнено нулями.

| Величина | Значення |
|---|---:|
| Взаємна кореляція $r_{{12}}(0)$ | {table_results['table1_r12']:.6f} |

## Результати для таблиці 2

| Величина | Пара $x_1, x_2$ | Пара $x_3, x_4$ |
|---|---:|---:|
| Звичайна кореляція | {table_results['table2_r12']:.6f} | {table_results['table2_r34']:.6f} |
| Нормуючий коефіцієнт | {table_results['norm12']:.6f} | {table_results['norm34']:.6f} |
| Нормована кореляція | {table_results['rho12']:.6f} | {table_results['rho34']:.6f} |

Звичайні кореляції різняться через різні амплітуди сигналів. Після нормування
отримуємо однакове значення, тому що форми обох пар сигналів однакові.

![Сигнали з таблиці 2](plots/1_table2_signals.png)

## Аналіз випадкових сигналів

- Кількість відліків: **{N_RANDOM}**.
- Діапазон зсувів: **$j=0\ldots{MAX_LAG}$**.
- Нормована енергія першого сигналу: **{random_results['energy_x1']:.6f}**.
- Нормована енергія другого сигналу: **{random_results['energy_x2']:.6f}**.
- Найбільше за модулем значення взаємної кореляції отримано при
  **$j={random_results['max_cross_lag']}$**:
  **{random_results['max_cross_value']:.6f}**.
- $r_{{11}}(0)={random_results['autocorrelation_x1_zero']:.6f}$, що дорівнює $S_1$.
- $r_{{22}}(0)={random_results['autocorrelation_x2_zero']:.6f}$, що дорівнює $S_2$.

### Початкові сигнали

![Два випадкові сигнали](plots/2_random_signals.png)

### Взаємна кореляція

![Взаємна кореляція](plots/3_cross_correlation.png)

Значення коливаються біля нуля, тому між незалежно згенерованими сигналами
немає помітного лінійного зв'язку.

### Автокореляція першого сигналу

![Автокореляція першого сигналу](plots/4_autocorrelation_x1.png)

### Автокореляція другого сигналу

![Автокореляція другого сигналу](plots/5_autocorrelation_x2.png)

Обидві автокореляційні функції мають виражений максимум при нульовому зсуві.
Для інших зсувів значення знаходяться біля нуля, що характерно для шумових
випадкових сигналів.

### Порівняння енергій

![Порівняння енергій](plots/6_signal_energies.png)
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table_results = calculate_table_results()
    random_results = analyze_random_signals()
    write_markdown_report(table_results, random_results)


if __name__ == "__main__":
    main()
