import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Sabitler
mu_0 = 4 * np.pi * 1e-7  # Serbest uzayın manyetik geçirgenliği
n_turns = 100 # Bobin sarım sayısı
I = 1  # Bobinden geçen akım (Amper)
radius = 0.05  # Bobin yarıçapı (metre)
C = 1e-12  # Kapasitans (Farad)
frequency_range = np.linspace(400e3, 1100e3, 100)  # Frekans aralığı (Hz)

# Fonksiyonlar
def calculate_inductance(radius, n_turns):
    """Bobinin indüktansını hesapla (yaklaşık formül)."""
    return mu_0 * n_turns**2 * np.pi * radius**2 / radius

def calculate_mutual_inductance(radius, d):
    """Bobinler arasındaki karşılıklı indüktansı hesapla."""
    return mu_0 * n_turns**2 * np.pi * radius**2 / (2 * d)

def calculate_resonant_frequency(L, M, C):
    """Rezonans frekansını hesapla."""
    return 1 / (2 * np.pi * np.sqrt((L + M) * C))

def calculate_magnetic_field(radius, I, n_turns):
    """Bobinin merkezindeki manyetik alanı hesapla."""
    return (mu_0 * n_turns * I) / (2 * radius)

import numpy as np

# Sabitler
# mu_0 = 4 * np.pi * 1e-7  # Serbest uzayın manyetik geçirgenliği (H/m)

I1 = 1  # Bobin 1'den geçen akım (Amper)
I2 = 1  # Bobin 2'den geçen akım (Amper)
n1 = 100  # Bobin 1'in sarım sayısı
n2 = 100  # Bobin 2'nin sarım sayısı
d = 0.02# Bobinler arasındaki mesafe (metre)

def calculate_mutual_magnetic_field(I1, I2, n1, n2, d):
    """Bobinler arasındaki karşılıklı manyetik alanı hesapla."""
    return (mu_0 * n1 * n2 * I1 * I2) / (2 * np.pi * d)

# Örnek kullanım:

# Karşılıklı manyetik alanı hesapla
B = calculate_mutual_magnetic_field(I1, I2, n1, n2, d)
print(f"Bobinler arasındaki karşılıklı manyetik alan: {B} Tesla")

# # Bobinler arası uzaklık d = 0.20 cm
# d = 0.15e-2  # cm'yi metreye çeviriyoruz

# Hesaplamalar
L = calculate_inductance(radius, n_turns)
M = calculate_mutual_inductance(radius, d)
f_res = calculate_resonant_frequency(L, M, C)
B1 = calculate_magnetic_field(radius, I, n_turns)  # Bobin 1'in oluşturduğu manyetik alan
B_mutual = calculate_mutual_magnetic_field(I, I, n_turns, n_turns, d)  # Bobinler arasındaki karşılıklı manyetik alan

# Frekans ve uzaklık grafiği
plt.figure(figsize=(10, 6))
plt.plot([d * 100], [f_res / 1e3], 'ro', label="Rezonans Frekansı (kHz)")
plt.xlabel("Bobinler Arası Uzaklık d (cm)")
plt.ylabel("Frekans (kHz)")
plt.title("Bobinler Arası Uzaklık ile Frekans İlişkisi")
plt.legend()
plt.grid()
plt.show()

# Bobinlerin 3D Görselleştirmesi
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

def plot_coil(ax, center, radius, n_turns):
    """Bir bobinin 3D görselleştirmesini çiz."""
    theta = np.linspace(0, 2 * np.pi, 100)
    z = np.linspace(0, 0.01, n_turns)
    for zi in z:
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        ax.plot(x, y, zi + center[2], color='b')

# Bobin 1 ve Bobin 2'nin çizimi
plot_coil(ax, [0, 0, 0], radius, n_turns)
plot_coil(ax, [0.3, 0, 0], radius, n_turns)  # 0.30 cm uzaklık

ax.set_title("Bobinlerin 3D Görselleştirmesi")
ax.set_xlabel("X Ekseni (m)")
ax.set_ylabel("Y Ekseni (m)")
ax.set_zlabel("Z Ekseni (m)")
plt.show()

# Sonuçları yazdırma
print(f"Bobinler Arası Uzaklık d = 0.20 cm için rezonans frekansı: {f_res / 1e3:.2f} kHz")
print(f"Bobin 1'in merkezindeki manyetik alan: {B1:.2e} Tesla")
print(f"Bobinler Arasındaki Karşılıklı Manyetik Alan: {B_mutual:.2e} Tesla")
