import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1. Funkcje przynależności
# =========================

def triangular(x, a, b, c):
    """
    Trójkątna funkcja przynależności.
    a - lewy punkt
    b - szczyt
    c - prawy punkt
    """
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=float)

    # lewa strona
    left = (a < x) & (x <= b)
    result[left] = (x[left] - a) / (b - a) if b != a else 0.0

    # prawa strona
    right = (b < x) & (x < c)
    result[right] = (c - x[right]) / (c - b) if c != b else 0.0

    # szczyt
    result[x == b] = 1.0

    return np.clip(result, 0.0, 1.0)


def trapezoidal(x, a, b, c, d):
    """
    Trapezowa funkcja przynależności.
    a,b - początek i koniec narastania
    c,d - początek i koniec opadania
    """
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=float)

    # narastanie
    rising = (a < x) & (x < b)
    result[rising] = (x[rising] - a) / (b - a) if b != a else 0.0

    # plateau
    plateau = (b <= x) & (x <= c)
    result[plateau] = 1.0

    # opadanie
    falling = (c < x) & (x < d)
    result[falling] = (d - x[falling]) / (d - c) if d != c else 0.0

    return np.clip(result, 0.0, 1.0)


# ======================================
# 2. Uniwersa zmiennych i zbiory rozmyte
# ======================================

temp_universe = np.linspace(0, 40, 1000)       # temperatura [°C]
humid_universe = np.linspace(0, 100, 1000)     # wilgotność [%]
fan_universe = np.linspace(0, 100, 1000)       # prędkość wentylatora [%]

# Temperatura
temp_cold = trapezoidal(temp_universe, 0, 0, 10, 18)
temp_warm = triangular(temp_universe, 15, 22, 30)
temp_hot  = trapezoidal(temp_universe, 26, 32, 40, 40)

# Wilgotność
humid_low  = trapezoidal(humid_universe, 0, 0, 30, 45)
humid_mid  = triangular(humid_universe, 35, 50, 65)
humid_high = trapezoidal(humid_universe, 55, 70, 100, 100)

# Wentylator
fan_low    = trapezoidal(fan_universe, 0, 0, 20, 40)
fan_medium = triangular(fan_universe, 30, 50, 70)
fan_high   = trapezoidal(fan_universe, 60, 80, 100, 100)


# =====================================
# 3. Obliczanie stopni przynależności
# =====================================

def fuzzify_temperature(value):
    return {
        "cold": trapezoidal(np.array([value]), 0, 0, 10, 18)[0],
        "warm": triangular(np.array([value]), 15, 22, 30)[0],
        "hot":  trapezoidal(np.array([value]), 26, 32, 40, 40)[0]
    }


def fuzzify_humidity(value):
    return {
        "low":  trapezoidal(np.array([value]), 0, 0, 30, 45)[0],
        "mid":  triangular(np.array([value]), 35, 50, 65)[0],
        "high": trapezoidal(np.array([value]), 55, 70, 100, 100)[0]
    }


# ==========================================
# 4. Reguły rozmyte typu Mamdani
# ==========================================
#
# Reguły:
# R1: IF temp is cold AND humidity is low  THEN fan is low
# R2: IF temp is warm AND humidity is low  THEN fan is medium
# R3: IF temp is warm AND humidity is high THEN fan is high
# R4: IF temp is hot                       THEN fan is high
# R5: IF humidity is high                 THEN fan is high
# R6: IF temp is cold AND humidity is high THEN fan is medium
#

def fuzzy_inference(temp_value, humid_value):
    t = fuzzify_temperature(temp_value)
    h = fuzzify_humidity(humid_value)

    print("Stopnie przynależności temperatury:", t)
    print("Stopnie przynależności wilgotności:", h)

    # Siła aktywacji reguł
    r1 = min(t["cold"], h["low"])
    r2 = min(t["warm"], h["low"])
    r3 = min(t["warm"], h["high"])
    r4 = t["hot"]
    r5 = h["high"]
    r6 = min(t["cold"], h["high"])

    print("\nAktywacje reguł:")
    print(f"R1 (cold AND low -> low):     {r1:.3f}")
    print(f"R2 (warm AND low -> medium):  {r2:.3f}")
    print(f"R3 (warm AND high -> high):   {r3:.3f}")
    print(f"R4 (hot -> high):             {r4:.3f}")
    print(f"R5 (high humidity -> high):   {r5:.3f}")
    print(f"R6 (cold AND high -> medium): {r6:.3f}")

    # Implikacja: przycięcie zbiorów wyjściowych
    fan_rule1 = np.minimum(r1, fan_low)
    fan_rule2 = np.minimum(r2, fan_medium)
    fan_rule3 = np.minimum(r3, fan_high)
    fan_rule4 = np.minimum(r4, fan_high)
    fan_rule5 = np.minimum(r5, fan_high)
    fan_rule6 = np.minimum(r6, fan_medium)

    # Agregacja wszystkich reguł
    aggregated = np.maximum.reduce([
        fan_rule1, fan_rule2, fan_rule3,
        fan_rule4, fan_rule5, fan_rule6
    ])

    return aggregated


# =====================================
# 5. Defuzyfikacja - środek ciężkości
# =====================================

def defuzzify(universe, membership_values):
    numerator = np.sum(universe * membership_values)
    denominator = np.sum(membership_values)

    if denominator == 0:
        return 0.0

    return numerator / denominator


# =====================================
# 6. Przykład działania
# =====================================

temperature_input = 28
humidity_input = 78

aggregated_output = fuzzy_inference(temperature_input, humidity_input)
fan_speed = defuzzify(fan_universe, aggregated_output)

print(f"\nDla temperatury {temperature_input}°C i wilgotności {humidity_input}%")
print(f"Zalecana prędkość wentylatora: {fan_speed:.2f}%")


# =====================================
# 7. Wizualizacja
# =====================================

fig, axes = plt.subplots(4, 1, figsize=(10, 14))

# Temperatura
axes[0].plot(temp_universe, temp_cold, label="cold")
axes[0].plot(temp_universe, temp_warm, label="warm")
axes[0].plot(temp_universe, temp_hot, label="hot")
axes[0].axvline(temperature_input, color="black", linestyle="--", label="input")
axes[0].set_title("Zbiory rozmyte: temperatura")
axes[0].legend()
axes[0].grid(True)

# Wilgotność
axes[1].plot(humid_universe, humid_low, label="low")
axes[1].plot(humid_universe, humid_mid, label="mid")
axes[1].plot(humid_universe, humid_high, label="high")
axes[1].axvline(humidity_input, color="black", linestyle="--", label="input")
axes[1].set_title("Zbiory rozmyte: wilgotność")
axes[1].legend()
axes[1].grid(True)

# Wyjście wentylatora
axes[2].plot(fan_universe, fan_low, label="low")
axes[2].plot(fan_universe, fan_medium, label="medium")
axes[2].plot(fan_universe, fan_high, label="high")
axes[2].set_title("Zbiory rozmyte: prędkość wentylatora")
axes[2].legend()
axes[2].grid(True)

# Agregacja wyniku
axes[3].plot(fan_universe, aggregated_output, label="aggregated output")
axes[3].fill_between(fan_universe, aggregated_output, alpha=0.3)
axes[3].axvline(fan_speed, color="red", linestyle="--", label=f"defuzzified = {fan_speed:.2f}%")
axes[3].set_title("Wynik po agregacji i defuzyfikacji")
axes[3].legend()
axes[3].grid(True)

plt.tight_layout()
plt.show()
