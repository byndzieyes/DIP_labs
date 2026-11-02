from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def dft(values, inverse=False):
    """Пряма сума ДПФ; для оберненого змінюємо знак і ділимо на N."""
    size = len(values)
    indices = np.arange(size)
    result = np.zeros(size, dtype=complex)
    sign = 1 if inverse else -1
    for k in range(size):
        result[k] = np.sum(values * np.exp(sign * 2j * np.pi * k * indices / size))
    return result / size if inverse else result


def save_plot(plots, name):
    plt.tight_layout()
    plt.savefig(plots / name, dpi=150)
    plt.close()


def create_signal(dt, t_max):
    t = np.arange(int(t_max / dt) + 1) * dt
    x = np.sin(2 * np.pi * t / 20 + 1) + np.cos(2 * np.pi * t / 20 + np.pi / 2)
    return t, x


def calculate_spectra(X, dt):
    N = len(X)
    fs = 1 / dt
    freq = np.arange(N // 2 + 1) * fs / N
    amplitude = np.abs(X[:len(freq)]) / N
    amplitude[1:] *= 2
    if N % 2 == 0:
        amplitude[-1] /= 2
    phase = np.angle(X)
    phase[np.abs(X) < np.max(np.abs(X)) * 1e-8] = np.nan

    power = np.abs(X[:len(freq)]) ** 2 / (fs * N)
    power[1:] *= 2
    if N % 2 == 0:
        power[-1] /= 2
    return freq, amplitude, phase, power


def plot_signal(plots, t, x):
    plt.figure(figsize=(11, 4))
    plt.plot(t, x, linewidth=1)
    plt.title("Варіант 5 — початковий сигнал")
    plt.xlabel("Час t")
    plt.ylabel("x(t)")
    plt.grid(alpha=0.3)
    save_plot(plots, "signal.png")


def plot_amplitude(plots, X, freq, amplitude):
    N = len(X)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    axes[0].plot(np.arange(N), np.abs(X), linewidth=1)
    axes[0].set(title="Повний спектр ДПФ", xlabel="Індекс k", ylabel="|X[k]|")
    axes[1].plot(freq, amplitude, ".-", markersize=3)
    axes[1].axvline(1 / 20, color="red", linestyle="--", label="Теоретична частота 0,05")
    axes[1].set(xlim=(0, 0.15), title="Однобічний амплітудний спектр (збільшено)",
                xlabel="Частота, циклів / одиницю часу", ylabel="Амплітуда")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    save_plot(plots, "amplitude.png")


def plot_phase(plots, phase):
    N = len(phase)
    plt.figure(figsize=(11, 4))
    plt.plot(np.arange(N), phase, ".", markersize=2)
    plt.title("Фазовий спектр ДПФ")
    plt.xlabel("Індекс k")
    plt.ylabel("Фаза, рад")
    plt.grid(alpha=0.3)
    save_plot(plots, "phase.png")


def plot_periodogram(plots, freq, power):
    plt.figure(figsize=(11, 4))
    plt.plot(freq, power, ".-", markersize=3)
    plt.xlim(0, 0.15)
    plt.title("Періодограма — однобічна спектральна густина потужності")
    plt.xlabel("Частота, циклів / одиницю часу")
    plt.ylabel("Спектральна густина потужності")
    plt.grid(alpha=0.3)
    save_plot(plots, "periodogram.png")


def plot_reconstruction(plots, t, x, restored):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    axes[0].plot(t, x, label="Початковий", linewidth=2)
    axes[0].plot(t, restored.real, "--", label="Відновлений", linewidth=1)
    axes[0].set(title="Початковий і відновлений сигнали (перші 60 одиниць часу)",
                xlim=(0, 60), ylabel="Амплітуда")
    axes[0].legend()
    axes[1].plot(t, restored.real - x, linewidth=0.7)
    axes[1].set(title="Похибка відновлення на всьому інтервалі", xlabel="Час t", ylabel="Похибка")
    for ax in axes:
        ax.grid(alpha=0.3)
    save_plot(plots, "reconstruction.png")


def save_results(folder, dt, t, x, X, restored, freq, amplitude, direct_time, inverse_time):
    N = len(t)
    fs = 1 / dt
    peak = np.argmax(amplitude[1:]) + 1
    error = np.max(np.abs(restored.real - x))
    imaginary_error = np.max(np.abs(restored.imag))
    report = f"""# Лабораторна робота №7 — Результати

| Параметр | Значення |
|---|---:|
| Крок часу | {dt} |
| Кількість відліків | {N} |
| Початковий час | {t[0]:.2f} |
| Останній відлік часу | {t[-1]:.2f} |
| Частота дискретизації | {fs:.8f} |
| Крок частот | {fs / N:.8f} |

## Числові результати

| Величина | Значення |
|---|---:|
| Індекс основного піка | {peak} |
| Частота основного піка | {freq[peak]:.8f} |
| Однобічна амплітуда основного піка | {amplitude[peak]:.8f} |
| Фаза основного піка, рад | {np.angle(X[peak]):.8f} |
| Максимальна похибка відновлення | {error:.3e} |
| Максимальний залишок уявної частини | {imaginary_error:.3e} |
| Час прямого ДПФ, с | {direct_time:.4f} |
| Час оберненого ДПФ, с | {inverse_time:.4f} |

## Графіки

![Початковий сигнал](plots/signal.png)

![Амплітудний спектр](plots/amplitude.png)

![Фазовий спектр](plots/phase.png)

![Періодограма](plots/periodogram.png)

![Відновлення сигналу](plots/reconstruction.png)
    """
    (folder / "README.md").write_text(report, encoding="utf-8")


def main():
    folder = Path(__file__).resolve().parent
    plots = folder / "plots"
    plots.mkdir(exist_ok=True)
    dt = 0.07
    t, x = create_signal(dt, t_max=480)

    started = perf_counter()
    X = dft(x)
    direct_time = perf_counter() - started
    started = perf_counter()
    restored = dft(X, inverse=True)
    inverse_time = perf_counter() - started

    freq, amplitude, phase, power = calculate_spectra(X, dt)
    plot_signal(plots, t, x)
    plot_amplitude(plots, X, freq, amplitude)
    plot_phase(plots, phase)
    plot_periodogram(plots, freq, power)
    plot_reconstruction(plots, t, x, restored)
    save_results(folder, dt, t, x, X, restored, freq, amplitude, direct_time, inverse_time)
    print(f"Результати: {folder / 'README.md'}")


if __name__ == "__main__":
    main()
