import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# 1. Задаємо підінтегральну функцію навантаження на сервер
def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2)

a, b = 0.0, 24.0

# 2. Знаходження точного значення інтегралу
exact_integral, _ = quad(f, a, b)
print(f"Точне значення інтегралу I_0: {exact_integral:.12f}")

# 3. Функція складового методу Сімпсона для N розбиттів
def simpson_composite(f_func, a, b, N):
    if N % 2 != 0:
        N += 1  # гарантуємо парність
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f_func(x)
    integral = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2])
    return integral * h / 3

# 4. Дослідження точності залежно від N
N_values = np.arange(10, 1001, 2)
errors = []
N_opt = None

for N in N_values:
    val = simpson_composite(f, a, b, N)
    err = abs(val - exact_integral)
    errors.append(err)
    if err <= 1e-12 and N_opt is None:
        N_opt = N

if N_opt is None:
    N_opt = 1000

eps_opt = abs(simpson_composite(f, a, b, N_opt) - exact_integral)
print(f"Оптимальне N для точності 1e-12: N_opt = {N_opt}, похибка: {eps_opt:.2e}")

# 5. Вибір базового N_0
N_0 = int(N_opt / 10)
if N_0 % 8 != 0:
    N_0 = (N_0 // 8) * 8
if N_0 < 8: 
    N_0 = 8

eps_0 = abs(simpson_composite(f, a, b, N_0) - exact_integral)
print(f"Базове N_0 = {N_0}, похибка eps_0 = {eps_0:.2e}")

# 6. Метод Рунге-Ромберга
I_N0 = simpson_composite(f, a, b, N_0)
I_N0_half = simpson_composite(f, a, b, N_0 // 2)
I_R = I_N0 + (I_N0 - I_N0_half) / 15.0
eps_R = abs(I_R - exact_integral)
print(f"Метод Рунге-Ромберга: I_R = {I_R:.12f}, похибка eps_R = {eps_R:.2e}")

# 7. Метод Ейткена
I_N0_quarter = simpson_composite(f, a, b, N_0 // 4)
I_E = (I_N0_half**2 - I_N0 * I_N0_quarter) / (2 * I_N0_half - (I_N0 + I_N0_quarter))
p_aitken = (1 / np.log(2)) * np.log(abs((I_N0_quarter - I_N0_half) / (I_N0_half - I_N0)))
eps_E = abs(I_E - exact_integral)
print(f"Метод Ейткена: I_E = {I_E:.12f}, порядок p = {p_aitken:.2f}, похибка eps_E = {eps_E:.2e}")

# ==================== ПОБУДОВА ГРАФІКІВ ====================

# Графік 1: Залежність похибки від кількості розбиттів N
plt.figure(figsize=(10, 5))
plt.plot(N_values, errors, label='Похибка $\\epsilon(N)$', color='blue')
plt.axvline(N_opt, color='red', linestyle='--', label=f'Оптимальне N = {N_opt}')
plt.yscale('log')
plt.xlabel('Кількість розбиттів N')
plt.ylabel('Абсолютна похибка (лог. шкала)')
plt.title('Залежність точності складової формули Сімпсона від N')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig('error_vs_n.png')  # Зберігає графік у файл
print("Графік похибки збережено у файл 'error_vs_n.png'.")

# Графік 2: Функція навантаження на сервер f(x)
x_plot = np.linspace(a, b, 500)
y_plot = f(x_plot)

plt.figure(figsize=(10, 5))
plt.plot(x_plot, y_plot, label='$f(x) = 50 + 20\\sin(\\frac{\\pi x}{12}) + 5e^{-0.2(x-12)^2}$', color='purple')
plt.xlabel('Час, x (год)')
plt.ylabel('Навантаження, f(x)')
plt.title('Модель навантаження на сервер M(t)')
plt.legend()
plt.grid(True)
plt.savefig('server_load_model.png')  # Зберігає графік у файл
print("Графік моделі навантаження збережено у файл 'server_load_model.png'.")

# Відображення графіків на екрані (якщо працюєте у середовищі з підтримкою GUI)
plt.show()