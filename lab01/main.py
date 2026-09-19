import os
import numpy as np
import matplotlib.pyplot as plt

output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)


# Допоміжна функція збереження
def save_pair(filename, t, x, t_label, x_label, title_time,
              u, X, u_label, X_label, title_freq,
              is_stem=False, ylim_t=None, ylim_u=None, xlim_u=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Часова область
    ax1.plot(t, x, 'b', lw=2)
    ax1.set_title(title_time)
    ax1.set_xlabel(t_label)
    ax1.set_ylabel(x_label)
    if ylim_t:
        ax1.set_ylim(ylim_t)
    ax1.grid(True)

    # Частотна область
    if is_stem:
        ax2.stem(u, X, linefmt='r-', markerfmt='ro', basefmt='k-')
    else:
        ax2.plot(u, X, 'r', lw=2)

    ax2.set_title(title_freq)
    ax2.set_xlabel(u_label)
    ax2.set_ylabel(X_label)
    if ylim_u:
        ax2.set_ylim(ylim_u)
    if xlim_u:
        ax2.set_xlim(xlim_u)
    ax2.grid(True)

    plt.tight_layout()
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close(fig)
    print(f"Збережено: {filepath}")


# -------------------------------------------------------------
# 1. Гармонічний сигнал: cos(pi * t)
# -------------------------------------------------------------
t1 = np.linspace(-3, 3, 1000)
x1 = np.cos(np.pi * t1)
u1 = [-0.5, 0.5]
X1 = [0.5, 0.5]

save_pair(
    filename='1_cosine.png',
    t=t1, x=x1, t_label='Час (t)', x_label='x(t)',
    title_time='Косинусоїдальний сигнал: cos(πt)',
    u=u1, X=X1, u_label='Частота (u)', X_label='|X(u)|',
    title_freq='Спектр: 0.5·δ(u + 0.5) + 0.5·δ(u - 0.5)',
    is_stem=True, ylim_u=(0, 0.7), xlim_u=(-2, 2)
)

# -------------------------------------------------------------
# 2. Дельта-функція Дірака
# -------------------------------------------------------------
t2 = np.linspace(-2, 2, 1001)
dt2 = t2[1] - t2[0]
x2 = np.zeros_like(t2)
x2[len(t2) // 2] = 1.0 / dt2
u2 = np.linspace(-10, 10, 500)
X2 = np.ones_like(u2)

save_pair(
    filename='2_dirac_delta.png',
    t=t2, x=x2, t_label='Час (t)', x_label='δ(t)',
    title_time='δ-функція Дірака (апроксимація)',
    u=u2, X=X2, u_label='Частота (u)', X_label='|X(u)|',
    title_freq='Спектр дельта-функції: X(u) = 1',
    ylim_t=(-10, x2.max() * 1.1), ylim_u=(0, 1.5)
)

# -------------------------------------------------------------
# 3. Прямокутний імпульс (rect)
# -------------------------------------------------------------
t3 = np.linspace(-2.5, 2.5, 1000)
rect = np.where(np.abs(t3) <= 0.5, 1.0, 0.0)
u3 = np.linspace(-4, 4, 1000)
X3 = np.sinc(u3)

save_pair(
    filename='3_rect.png',
    t=t3, x=rect, t_label='Час (t)', x_label='rect(t)',
    title_time='Прямокутний імпульс: rect(t)',
    u=u3, X=X3, u_label='Частота (u)', X_label='X(u)',
    title_freq='Спектр прямокутного імпульсу: sinc(u)',
    ylim_t=(-0.2, 1.2)
)

# -------------------------------------------------------------
# 4. Функція відліків (sinc)
# -------------------------------------------------------------
t4 = np.linspace(-6, 6, 1000)
x4 = np.sinc(t4)
u4 = np.linspace(-2, 2, 1000)
X4 = np.where(np.abs(u4) <= 0.5, 1.0, 0.0)

save_pair(
    filename='4_sinc.png',
    t=t4, x=x4, t_label='Час (t)', x_label='sinc(t)',
    title_time='Функція відліків: sinc(t)',
    u=u4, X=X4, u_label='Частота (u)', X_label='X(u)',
    title_freq='Спектр sinc(t): rect(u)',
    ylim_u=(-0.2, 1.2)
)

# -------------------------------------------------------------
# 5. Функція знаку: signum (sgn)
# -------------------------------------------------------------
t5 = np.linspace(-3, 3, 1000)
x5 = np.sign(t5)
u5 = np.linspace(-4, 4, 1001)
u5 = u5[u5 != 0]
abs_X5 = 1.0 / (np.pi * np.abs(u5))

save_pair(
    filename='5_sgn.png',
    t=t5, x=x5, t_label='Час (t)', x_label='sgn(t)',
    title_time='Функція знаку: sgn(t)',
    u=u5, X=abs_X5, u_label='Частота (u)', X_label='|X(u)|',
    title_freq='Модуль спектра sgn: |1 / (j·π·u)|',
    ylim_t=(-1.3, 1.3), ylim_u=(0, 5)
)

# -------------------------------------------------------------
# 6. Трикутна функція
# -------------------------------------------------------------
t6 = np.linspace(-2.5, 2.5, 1000)
tri = np.maximum(0, 1 - np.abs(t6))
u6 = np.linspace(-4, 4, 1000)
X6 = np.sinc(u6) ** 2

save_pair(
    filename='6_triangle.png',
    t=t6, x=tri, t_label='Час (t)', x_label='Λ(t)',
    title_time='Трикутна функція: Λ(t)',
    u=u6, X=X6, u_label='Частота (u)', X_label='X(u)',
    title_freq='Спектр трикутного імпульсу: sinc²(u)',
    ylim_t=(-0.2, 1.2)
)