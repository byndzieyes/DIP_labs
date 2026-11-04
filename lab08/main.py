from cmath import exp
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def dft(values, inverse=False):
    """Безпосередня сума ДПФ, як у лабораторній №7."""
    size = len(values)
    indices = np.arange(size)
    result = np.zeros(size, dtype=complex)
    sign = 1 if inverse else -1
    for k in range(size):
        result[k] = np.sum(values * np.exp(sign * 2j * np.pi * k * indices / size))
    return result / size if inverse else result


def fft(values):
    """Алгоритм Кулі–Тюкі для кількості відліків, що є степенем двійки."""
    size = len(values)
    # Один відлік уже є своїм одноточковим спектром. Тут поділ закінчується.
    if size == 1:
        return np.array(values, dtype=complex)

    # Ділимо за номерами позицій і рахуємо два менші спектри.
    even = fft(values[::2])
    odd = fft(values[1::2])
    result = np.zeros(size, dtype=complex)
    for k in range(size // 2):
        # Метелик: один добуток використовується для двох коефіцієнтів.
        term = exp(-2j * np.pi * k / size) * odd[k]
        # Суму записуємо в першу половину спектра, різницю — в другу.
        result[k] = even[k] + term
        result[k + size // 2] = even[k] - term
    return result


def ifft(values):
    """Обернене ШПФ через власне пряме ШПФ і комплексне спряження."""
    return np.conjugate(fft(np.conjugate(values))) / len(values)


def create_signal(size, t_max):
    t = np.linspace(0, t_max, size)
    x = np.cos(2 * np.pi * t / 10 + 1) + np.cos(2 * np.pi * t / 40 + np.pi / 2)
    return t, x


def measure_transform(transform, values):
    times = []
    for _ in range(3):
        started = perf_counter()
        result = transform(values)
        times.append(perf_counter() - started)
    return result, float(np.median(times))


def calculate_spectra(X, dt):
    size = len(X)
    fs = 1 / dt
    freq = np.arange(size // 2 + 1) * fs / size
    amplitude = np.abs(X[:len(freq)]) / size
    power = np.abs(X[:len(freq)]) ** 2 / (fs * size)
    amplitude[1:-1] *= 2
    power[1:-1] *= 2
    phase = np.angle(X[:len(freq)])
    phase[amplitude < np.max(amplitude) * 1e-8] = np.nan
    return freq, amplitude, phase, power


def benchmark(durations, base_duration, base_size):
    rows = []
    for duration in durations:
        size = int(base_size * duration / base_duration)
        _, x = create_signal(size, duration)
        X_dft, dft_time = measure_transform(dft, x)
        X_fft, fft_time = measure_transform(fft, x)
        difference = np.max(np.abs(X_dft - X_fft))
        rows.append((duration, size, dft_time, fft_time, dft_time / fft_time, difference))
    return rows


def save_plot(plots, name):
    plt.tight_layout()
    plt.savefig(plots / name, dpi=150)
    plt.close()


def plot_signal(plots, t, x):
    plt.figure(figsize=(11, 4))
    plt.plot(t, x, linewidth=1)
    plt.title("Початковий сигнал — лабораторна №8")
    plt.xlabel("Час t")
    plt.ylabel("x(t)")
    plt.grid(alpha=0.3)
    save_plot(plots, "signal.png")


def plot_spectra(plots, freq, dft_spectra, fft_spectra):
    labels = ["Амплітуда", "Фаза, рад", "Спектральна густина потужності"]
    titles = ["Амплітудні спектри", "Фазові спектри", "Періодограми"]
    names = ["amplitude.png", "phase.png", "periodogram.png"]
    for i in range(3):
        plt.figure(figsize=(11, 4))
        plt.plot(freq, dft_spectra[i], label="ДПФ", linewidth=2)
        plt.plot(freq, fft_spectra[i], "--", label="ШПФ", linewidth=1)
        plt.xlim(0, 0.15)
        plt.title(titles[i] + " — порівняння ДПФ і ШПФ")
        plt.xlabel("Частота, циклів / одиницю часу")
        plt.ylabel(labels[i])
        plt.legend()
        plt.grid(alpha=0.3)
        save_plot(plots, names[i])


def plot_difference(plots, X_dft, X_fft):
    plt.figure(figsize=(11, 4))
    plt.plot(np.arange(len(X_dft)), np.abs(X_dft - X_fft), linewidth=1)
    plt.title("Різниця комплексних коефіцієнтів ДПФ і ШПФ")
    plt.xlabel("Індекс k")
    plt.ylabel("|ДПФ[k] − ШПФ[k]|")
    plt.grid(alpha=0.3)
    save_plot(plots, "difference.png")


def plot_reconstruction(plots, t, x, restored):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    axes[0].plot(t, x, label="Початковий", linewidth=2)
    axes[0].plot(t, restored.real, "--", label="Відновлений ШПФ", linewidth=1)
    axes[0].set(xlim=(0, 100), title="Початковий і відновлений сигнали", ylabel="Амплітуда")
    axes[0].legend()
    axes[1].plot(t, restored.real - x, linewidth=1)
    axes[1].set(title="Похибка відновлення", xlabel="Час t", ylabel="Похибка")
    for ax in axes:
        ax.grid(alpha=0.3)
    save_plot(plots, "reconstruction.png")


def plot_timing(plots, timings):
    durations, _, dft_times, fft_times, _, _ = zip(*timings)
    plt.figure(figsize=(9, 5))
    plt.loglog(durations, dft_times, "o-", label="ДПФ")
    plt.loglog(durations, fft_times, "o-", label="ШПФ")
    plt.xticks(durations, durations)
    plt.gca().xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    plt.title("Час виконання — медіана трьох запусків")
    plt.xlabel("Кінцевий час сигналу")
    plt.ylabel("Час, с")
    plt.legend()
    plt.grid(which="both", alpha=0.3)
    save_plot(plots, "timing.png")


def save_results(folder, dt, t, x, X_dft, X_fft, restored, spectra, times, timings):
    freq, amplitude, phase, _ = spectra
    dft_time, fft_time, inverse_time = times
    peak_rows = []
    for expected in [1 / 40, 1 / 10]:
        # Шукаємо локальний основний пік навколо кожної заданої частоти.
        nearby = np.flatnonzero(np.abs(freq - expected) < 0.01)
        peak = nearby[np.argmax(amplitude[nearby])]
        peak_rows.append(f"| {peak} | {freq[peak]:.8f} | {amplitude[peak]:.8f} | {phase[peak]:.8f} |")
    timing_rows = [
        f"| {duration} | {n} | {duration / (n - 1):.8f} | {direct:.6f} | {fast:.6f} | {speedup:.2f} | {difference:.3e} |"
        for duration, n, direct, fast, speedup, difference in timings
    ]
    report = f"""# Лабораторна робота №8 — Результати

| Параметр | Значення |
|---|---:|
| Крок часу | {dt:.8f} |
| Кількість відліків | {len(t)} |
| Початковий час | {t[0]:.2f} |
| Останній відлік часу | {t[-1]:.2f} |
| Частота дискретизації | {1 / dt:.8f} |
| Крок частот | {1 / (dt * len(t)):.8f} |
| Кількість запусків для вимірювання часу (медіана) | 3 |

## Основні піки ШПФ

| Індекс k | Частота | Однобічна амплітуда | Фаза, рад |
|---|---:|---:|---:|
{chr(10).join(peak_rows)}

## Порівняння та відновлення

| Величина | Значення |
|---|---:|
| Максимальна різниця комплексних коефіцієнтів ДПФ і ШПФ | {np.max(np.abs(X_dft - X_fft)):.3e} |
| Максимальна похибка відновлення | {np.max(np.abs(restored.real - x)):.3e} |
| Максимальний залишок уявної частини | {np.max(np.abs(restored.imag)):.3e} |
| Час прямого ДПФ, с | {dft_time:.6f} |
| Час прямого ШПФ, с | {fft_time:.6f} |
| Час оберненого ШПФ, с | {inverse_time:.6f} |
| Прискорення прямого ШПФ відносно ДПФ, разів | {dft_time / fft_time:.2f} |

## Час виконання для різних часових інтервалів

| Кінцевий час | N | Крок часу | ДПФ, с | ШПФ, с | Прискорення, разів | Максимальна різниця коефіцієнтів |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(timing_rows)}

## Графіки

![Початковий сигнал](plots/signal.png)

![Амплітудні спектри](plots/amplitude.png)

![Фазові спектри](plots/phase.png)

![Періодограми](plots/periodogram.png)

![Різниця коефіцієнтів](plots/difference.png)

![Відновлення сигналу](plots/reconstruction.png)

![Час виконання](plots/timing.png)
"""
    (folder / "README.md").write_text(report, encoding="utf-8")


def main():
    folder = Path(__file__).resolve().parent
    plots = folder / "plots"
    plots.mkdir(exist_ok=True)
    # Формула (12), рівномірні відліки на замкненому інтервалі [0, 480].
    t_max = 480
    size = 1024
    t, x = create_signal(size, t_max)
    dt = t[1] - t[0]
    # Порівнюємо алгоритми на тих самих відліках x.
    X_dft, dft_time = measure_transform(dft, x)
    X_fft, fft_time = measure_transform(fft, x)
    # Для відновлення передаємо весь комплексний спектр, з амплітудами й фазами.
    restored, inverse_time = measure_transform(ifft, X_fft)

    freq, *dft_spectra = calculate_spectra(X_dft, dt)
    _, *fft_spectra = calculate_spectra(X_fft, dt)
    durations = [t_max // 4, t_max // 2, t_max * 2, t_max * 4, t_max * 8]
    timings = benchmark(durations, t_max, size)
    timings.append((t_max, len(x), dft_time, fft_time, dft_time / fft_time,
                    np.max(np.abs(X_dft - X_fft))))
    timings.sort()

    plot_signal(plots, t, x)
    plot_spectra(plots, freq, dft_spectra, fft_spectra)
    plot_difference(plots, X_dft, X_fft)
    plot_reconstruction(plots, t, x, restored)
    plot_timing(plots, timings)
    save_results(folder, dt, t, x, X_dft, X_fft, restored, (freq, *fft_spectra),
                 (dft_time, fft_time, inverse_time), timings)
    print(f"Різниця ДПФ і ШПФ: {np.max(np.abs(X_dft - X_fft)):.3e}")
    print(f"Похибка відновлення: {np.max(np.abs(restored.real - x)):.3e}")
    print(f"Прискорення при N = {len(x)}: {dft_time / fft_time:.2f}")
    print(f"Результати: {folder / 'README.md'}")


if __name__ == "__main__":
    main()
