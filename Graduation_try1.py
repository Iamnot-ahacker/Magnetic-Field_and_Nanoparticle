# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D 
# # Sabitler
# mu_0 = 4 * np.pi * 1e-7  # Serbest uzayın manyetik geçirgenliği
# n_turns = 500 # Bobin sarım sayısı
# I = 1  # Bobinden geçen akım (Amper)
# radius = 0.0012  # Bobin yarıçapı (metre)
# C = 1e-12  # Kapasitans (Farad)

# frequency_range = np.linspace(400e3, 1100e3, 100)  # Frekans aralığı (Hz)
# # Fonksiyonlar
# def calculate_inductance(radius, n_turns):
#     """Bobinin indüktansını hesapla (yaklaşık formül)."""
#     return mu_0 * n_turns**2 * np.pi * radius**2 / radius

# def calculate_mutual_inductance(radius, d):
#     """Bobinler arasındaki karşılıklı indüktansı hesapla."""
#     return mu_0 * n_turns**2 * np.pi * radius**2 / (2 * d)

# def calculate_resonant_frequency(L, M, C):
#     """Rezonans frekansını hesapla."""
#     return 1 / (2 * np.pi * np.sqrt((L + M) * C))

# def calculate_magnetic_field(radius, I, n_turns):
#     """Bobinin merkezindeki manyetik alanı hesapla."""
#     return (mu_0 * n_turns * I) / (2 * radius)

# # Sabitler
# # mu_0 = 4 * np.pi * 1e-7  # Serbest uzayın manyetik geçirgenliği (H/m)

# I1 = 1  # Bobin 1'den geçen akım (Amper)
# I2 = 1  # Bobin 2'den geçen akım (Amper)
# n1 = 100  # Bobin 1'in sarım sayısı
# n2 = 100  # Bobin 2'nin sarım sayısı
# d = 0.0026 # Bobinler arasındaki mesafe (metre)

# def calculate_mutual_magnetic_field(I1, I2, n1, n2, d):
#     """Bobinler arasındaki karşılıklı manyetik alanı hesapla."""
#     return (mu_0 * n1 * n2 * I1 * I2) / (2 * np.pi * d)

# # Örnek kullanım:

# # Karşılıklı manyetik alanı hesapla
# B = calculate_mutual_magnetic_field(I1, I2, n1, n2, d)
# print(f"Bobinler arasındaki karşılıklı manyetik alan: {B} Tesla")

# # # Bobinler arası uzaklık d = 0.20 cm
# # d = 0.15e-2  # cm'yi metreye çeviriyoruz

# # Hesaplamalar
# L = calculate_inductance(radius, n_turns)
# M = calculate_mutual_inductance(radius, d)
# f_res = calculate_resonant_frequency(L, M, C)
# B1 = calculate_magnetic_field(radius, I, n_turns)  # Bobin 1'in oluşturduğu manyetik alan
# B_mutual = calculate_mutual_magnetic_field(I, I, n_turns, n_turns, d)  # Bobinler arasındaki karşılıklı manyetik alan

# # Frekans ve uzaklık grafiği
# plt.figure(figsize=(10, 6))
# plt.plot([d * 100], [f_res / 1e3], 'ro', label="Rezonans Frekansı (kHz)")
# plt.xlabel("Bobinler Arası Uzaklık d (cm)")
# plt.ylabel("Frekans (kHz)")
# plt.title("Bobinler Arası Uzaklık ile Frekans İlişkisi")
# plt.legend()
# plt.grid()
# plt.show()
# # Bobinlerin 3D Görselleştirmesi
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# def plot_coil(ax, center, radius, n_turns):
#     """Bir bobinin 3D görselleştirmesini çiz."""
#     theta = np.linspace(0, 2 * np.pi, 100)
#     z = np.linspace(0, 0.01, n_turns)
#     for zi in z:
#         x = center[0] + radius * np.cos(theta)
#         y = center[1] + radius * np.sin(theta)
#         ax.plot(x, y, zi + center[2], color='b')

# # Bobin 1 ve Bobin 2'nin çizimi
# plot_coil(ax, [0, 0, 0], radius, n_turns)
# plot_coil(ax, [0.02, 0, 0], radius, n_turns)  

# ax.set_title("Bobinlerin 3D Görselleştirmesi")
# ax.set_xlabel("X Ekseni (m)")
# ax.set_ylabel("Y Ekseni (m)")
# ax.set_zlabel("Z Ekseni (m)")
# plt.show()

# # Sonuçları yazdırma
# print(f"Bobinler Arası Uzaklık d = 0.26 cm için rezonans frekansı: {f_res / 1e3:.2f} kHz")
# print(f"Bobin 1'in merkezindeki manyetik alan: {B1:.2e} Tesla")
# print(f"Bobinler Arasındaki Karşılıklı Manyetik Alan: {B_mutual:.2e} Tesla")

# import numpy as np
# import matplotlib.pyplot as plt

# # Sabitler
# mu_0 = 4 * np.pi * 1e-7  # Serbest uzayın manyetik geçirgenliği
# n_turns = 100  # Bobin sarım sayısı
# I = 1  # Bobinden geçen akım (Amper)
# radius = 0.05  # Bobin yarıçapı (metre)
# C = 1e-12  # Kapasitans (Farad)

# # Fonksiyonlar
# def calculate_inductance(radius, n_turns):
#     """Bobinin indüktansını hesapla (yaklaşık formül)."""
#     return mu_0 * n_turns**2 * np.pi * radius**2 / radius

# def calculate_mutual_inductance(radius, d):
#     """Bobinler arasındaki karşılıklı indüktansı hesapla."""
#     return mu_0 * n_turns**2 * np.pi * radius**2 / (2 * d)

# def calculate_resonant_frequency(L, M, C):
#     """Rezonans frekansını hesapla."""
#     return 1 / (2 * np.pi * np.sqrt((L + M) * C))

# # Bobinler arası uzaklık d aralığı (0 ile 0.5 cm arasında)
# d_values = np.linspace(0.0, 0.5e-2, 100)  # 0 cm ile 0.5 cm arasında

# # Hesaplamalar
# optimal_frequencies = []
# for d in d_values:
#     L = calculate_inductance(radius, n_turns)
#     M = calculate_mutual_inductance(radius, d)
#     f_res = calculate_resonant_frequency(L, M, C)
#     optimal_frequencies.append(f_res)

# # Frekans ve uzaklık grafiği
# plt.figure(figsize=(10, 6))
# plt.plot(d_values * 100, np.array(optimal_frequencies) / 1e3, label="Rezonans Frekansı (kHz)")
# plt.xlabel("Bobinler Arası Uzaklık d (cm)")
# plt.ylabel("Frekans (kHz)")
# plt.title("Bobinler Arası Uzaklık ile Frekans İlişkisi")
# plt.legend()
# plt.grid()
# plt.show()

# # 1100 kHz'e en yakın frekansı bul
# target_frequency = 1100e3  # 1100 kHz
# frequency_differences = np.abs(np.array(optimal_frequencies) - target_frequency)
# closest_index = np.argmin(frequency_differences)
# closest_d = d_values[closest_index] * 100  # cm cinsinden

# print(f"1100 kHz'e en yakın frekans, bobinler arası mesafe d = {closest_d:.2f} cm'dir.")


# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Constants
# mu_0 = 4 * np.pi * 1e-7  # Magnetic permeability (H/m)
# n_turns = 100             # Number of turns in the coil
# I = 1.0                   # Current through the coil (A)
# radius = 0.05             # Radius of the coil (m)
# C = 1e-9                  # Capacitance (F)
# frequency_range = np.linspace(1e3, 1e6, 1000)  # Frequency range (Hz)

# # Nanoparticle parameters
# nanoparticle_radius = 5e-9  # Nanoparticle radius (m)
# magnetic_susceptibility = 1e-5  # Magnetic susceptibility (dimensionless)
# nanoparticle_volume = (4 / 3) * np.pi * nanoparticle_radius**3
# magnetic_moment = magnetic_susceptibility * nanoparticle_volume / mu_0  # Magnetic moment (A*m^2)

# # Coil inductance calculation
# def calculate_inductance(radius, n_turns):
#     return mu_0 * n_turns**2 * np.pi * radius**2 / radius

# # Mutual inductance calculation
# def calculate_mutual_inductance(radius, d):
#     return mu_0 * n_turns**2 * np.pi * radius**2 / (2 * d)

# # Resonant frequency calculation
# def calculate_resonant_frequency(L, M, C):
#     return 1 / (2 * np.pi * np.sqrt((L + M) * C))

# # Magnetic field calculation
# def calculate_magnetic_field(radius, I, n_turns):
#     return (mu_0 * n_turns * I) / (2 * radius)

# # Magnetic torque on the nanoparticle
# def calculate_magnetic_torque(magnetic_moment, magnetic_field):
#     return magnetic_moment * magnetic_field

# # Coil parameters
# L = calculate_inductance(radius, n_turns)
# M = calculate_mutual_inductance(radius, 0.02)  # Distance between coils is 0.02 m

# # Resonant frequency
# resonant_frequency = calculate_resonant_frequency(L, M, C)

# # Magnetic field at the center of the coil
# magnetic_field = calculate_magnetic_field(radius, I, n_turns)

# # Magnetic torque on the nanoparticle
# torque = calculate_magnetic_torque(magnetic_moment, magnetic_field)

# # Results
# print(f"Resonant Frequency: {resonant_frequency:.2f} Hz")
# print(f"Magnetic Field at Coil Center: {magnetic_field:.2e} T")
# print(f"Nanoparticle Magnetic Moment: {magnetic_moment:.2e} A*m^2")
# print(f"Magnetic Torque on Nanoparticle: {torque:.2e} N*m")

# # Visualization
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')

# def plot_coil(ax, radius, n_turns, offset):
#     theta = np.linspace(0, 2 * np.pi, 100)
#     z = np.linspace(0, 0.01, n_turns)
#     for i in range(n_turns):
#         x = radius * np.cos(theta)
#         y = radius * np.sin(theta)
#         ax.plot(x, y, z[i] + offset, color='b')

# # Plot coil 1
# plot_coil(ax, radius, 10, 0)

# # Plot nanoparticle
# ax.scatter(0, 0, 0.02, color='r', s=100, label='Nanoparticle')

# ax.set_xlabel('X (m)')
# ax.set_ylabel('Y (m)')
# ax.set_zlabel('Z (m)')
# ax.legend()
# plt.title("3D Visualization of Coil and Nanoparticle")
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Constants
# mu_0 = 4 * np.pi * 1e-7  # Magnetic permeability (H/m)
# n_turns = 100             # Number of turns in each coil
# I = 1.0                   # Current through the coils (A)
# radius = 0.05             # Radius of the coils (m)
# C = 1e-9                  # Capacitance (F)
# d = 0.1                   # Distance between the centers of the two coils (m)

# # Coil inductance calculation
# def calculate_inductance(radius, n_turns):
#     return mu_0 * n_turns**2 * np.pi * radius**2 / radius

# # Mutual inductance calculation
# def calculate_mutual_inductance(radius, d):
#     return mu_0 * n_turns**2 * np.pi * radius**2 / (2 * d)

# # Resonant frequency calculation
# def calculate_resonant_frequency(L, M, C):
#     return 1 / (2 * np.pi * np.sqrt((L + M) * C))

# # Magnetic field calculation
# def calculate_magnetic_field(radius, I, n_turns):
#     return (mu_0 * n_turns * I) / (2 * radius)

# # Coil parameters
# L = calculate_inductance(radius, n_turns)
# M = calculate_mutual_inductance(radius, d)  # Distance between coils

# # Resonant frequency
# resonant_frequency = calculate_resonant_frequency(L, M, C)

# # Magnetic field at the center of the first coil
# magnetic_field_coil1 = calculate_magnetic_field(radius, I, n_turns)

# # Results
# print(f"Resonant Frequency: {resonant_frequency:.2f} Hz")
# print(f"Magnetic Field at Coil 1 Center: {magnetic_field_coil1:.2e} T")
# print(f"Mutual Inductance Between Coils: {M:.2e} H")

# # Visualization
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')

# def plot_coil(ax, center, radius, n_turns):
#     theta = np.linspace(0, 2 * np.pi, 100)
#     z = np.linspace(0, 0.01, n_turns)
#     for zi in z:
#         x = center[0] + radius * np.cos(theta)
#         y = center[1] + radius * np.sin(theta)
#         ax.plot(x, y, zi + center[2], color='b')

# # Plot coil 1
# plot_coil(ax, [0, 0, 0], radius, 10)

# # Plot coil 2
# plot_coil(ax, [d, 0, 0], radius, 10)

# ax.set_xlabel('X (m)')
# ax.set_ylabel('Y (m)')
# ax.set_zlabel('Z (m)')
# plt.title("3D Visualization of Two Coils")
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Constants
# mu_0 = 4 * np.pi * 1e-7  # Magnetic permeability (H/m)
# n_turns = 100             # Number of turns in each coil
# I = 1.0                   # Current through the coils (A)
# radius = 0.05             # Radius of the coils (m)
# C = 1e-9                  # Capacitance (F)
# d = 0.1                   # Distance between the centers of the two coils (m)
# nanoparticle_radius = 5e-9  # Radius of the nanoparticle (m)

# # Coil inductance calculation
# def calculate_inductance(radius, n_turns):
#     return mu_0 * n_turns**2 * np.pi * radius**2 / radius

# # Mutual inductance calculation
# def calculate_mutual_inductance(radius, d):
#     return mu_0 * n_turns**2 * np.pi * radius**2 / (2 * d)

# # Resonant frequency calculation
# def calculate_resonant_frequency(L, M, C):
#     return 1 / (2 * np.pi * np.sqrt((L + M) * C))

# # Magnetic field calculation
# def calculate_magnetic_field(radius, I, n_turns):
#     return (mu_0 * n_turns * I) / (2 * radius)

# # Coil parameters
# L = calculate_inductance(radius, n_turns)
# M = calculate_mutual_inductance(radius, d)  # Distance between coils

# # Resonant frequency
# resonant_frequency = calculate_resonant_frequency(L, M, C)

# # Magnetic field at the center of the first coil
# magnetic_field_coil1 = calculate_magnetic_field(radius, I, n_turns)

# # Results
# print(f"Resonant Frequency: {resonant_frequency:.2f} Hz")
# print(f"Magnetic Field at Coil 1 Center: {magnetic_field_coil1:.2e} T")
# print(f"Mutual Inductance Between Coils: {M:.2e} H")

# # Visualization
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')

# def plot_coil(ax, center, radius, n_turns):
#     theta = np.linspace(0, 2 * np.pi, 100)
#     z = np.linspace(0, 0.01, n_turns)
#     for zi in z:
#         x = center[0] + radius * np.cos(theta)
#         y = center[1] + radius * np.sin(theta)
#         ax.plot(x, y, zi + center[2], color='b')

# # Plot coil 1
# plot_coil(ax, [0, 0, 0], radius, 10)

# # Plot coil 2
# plot_coil(ax, [d, 0, 0], radius, 10)

# # Add nanoparticle
# def plot_nanoparticle(ax, center, radius):
#     u = np.linspace(0, 2 * np.pi, 100)
#     v = np.linspace(0, np.pi, 100)
#     x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
#     y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
#     z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
#     ax.plot_surface(x, y, z, color='r', alpha=0.6)

# # Nanoparticle position (center between the coils)
# nanoparticle_position = [d / 2, 0, 0]

# # Increase the radius for visualization purposes
# # visual_nanoparticle_radius = 0.002  # 2 mm for better visibility
# plot_nanoparticle(ax, nanoparticle_position, nanoparticle_radius)

# # plot_nanoparticle(ax, nanoparticle_position, visual_nanoparticle_radius)

# ax.set_xlabel('X (m)')
# ax.set_ylabel('Y (m)')
# ax.set_zlabel('Z (m)')
# plt.title("3D Visualization of Two Coils with Nanoparticle")
# plt.show()


# # Compute magnetic field at a given point
# def magnetic_field(x, y, z, coil_center, radius, n_turns, I):
#     r = np.sqrt((x - coil_center[0])**2 + (y - coil_center[1])**2 + (z - coil_center[2])**2)
#     if r == 0:
#         return np.array([0, 0, 0])
#     B_magnitude = (mu_0 * n_turns * I * radius**2) / (2 * (r**3))
#     return B_magnitude * np.array([x - coil_center[0], y - coil_center[1], z - coil_center[2]])

# # Compute magnetic field gradient numerically
# def magnetic_field_gradient(pos, coil_center, radius, n_turns, I, delta=1e-6):
#     grad = np.zeros(3)
#     for i in range(3):
#         pos_forward = pos.copy()
#         pos_forward[i] += delta
#         pos_backward = pos.copy()
#         pos_backward[i] -= delta
#         B_forward = magnetic_field(*pos_forward, coil_center, radius, n_turns, I)
#         B_backward = magnetic_field(*pos_backward, coil_center, radius, n_turns, I)
#         grad[i] = (np.linalg.norm(B_forward) - np.linalg.norm(B_backward)) / (2 * delta)
#     return grad

# # Position of the nanoparticle
# nanoparticle_position = [d / 2, 0, 0]

# # Magnetic moment of the nanoparticle (aligned along z-axis)
# magnetic_moment = np.array([0, 0, 1e-18])  # Example magnetic moment in A·m²

# # Calculate the net magnetic field and gradient
# B_net = magnetic_field(*nanoparticle_position, [0, 0, 0], radius, n_turns, I) + \
#         magnetic_field(*nanoparticle_position, [d, 0, 0], radius, n_turns, I)

# grad_B = magnetic_field_gradient(nanoparticle_position, [0, 0, 0], radius, n_turns, I) + \
#          magnetic_field_gradient(nanoparticle_position, [d, 0, 0], radius, n_turns, I)

# # Calculate the force
# force = np.dot(magnetic_moment, grad_B)
# print(f"Magnetic Force on Nanoparticle: {force:.2e} N")



























































# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.constants import mu_0, pi

# # Sabitler ve Parametreler
# N1, N2 = 100, 100  # Bobin sarım sayısı
# r1, r2 = 0.05, 0.05  # Bobin yarıçapları (metre)
# d = 0.1  # Bobinler arası mesafe (metre)
# l = 0.1  # Bobin uzunluğu (metre)
# mu_r = 1  # Göreceli manyetik geçirgenlik
# d_particle = 50e-9  # Nanoparçacık çapı (metre)
# chi = 1.5  # Manyetik duyarlılık

# # Bobinlerin İndüktansı
# def inductance(N, r, l):
#     """Bobin indüktansı hesaplama"""
#     return (mu_0 * N**2 * pi * r**2) / l

# L1 = inductance(N1, r1, l)
# L2 = inductance(N2, r2, l)

# # Karşılıklı İndüktans
# def mutual_inductance(N1, N2, r1, r2, d):
#     """Karşılıklı indüktans hesaplama"""
#     if d == 0:
#         raise ValueError("Bobinler arası mesafe sıfır olamaz.")
#     return mu_0 * N1 * N2 * pi * (r1 * r2)**2 / (2 * d**3)

# M = mutual_inductance(N1, N2, r1, r2, d)

# # Rezonans Frekansı
# def resonance_frequency(L, C):
#     """Rezonans frekansı hesaplama"""
#     return 1 / (2 * pi * np.sqrt(L * C))

# C = 1e-9  # Kapasitans (Farad)
# f_res = resonance_frequency(L1, C)

# # Manyetik Alan Hesaplama
# def magnetic_field(I, N, r, l):
#     """Manyetik alan hesaplama"""
#     return (mu_0 * N * I) / l

# I = 1  # Akım (Amper)
# B = magnetic_field(I, N1, r1, l)

# # Nanoparçacık Manyetik Momenti
# def magnetic_moment(B, r_particle, chi):
#     """Nanoparçacık manyetik momenti hesaplama"""
#     V = (4/3) * pi * (r_particle**3)
#     return chi * V * (B / mu_0)

# m_particle = magnetic_moment(B, d_particle / 2, chi)

# # Nanoparçacık Tork Hesabı
# def magnetic_torque(m, B):
#     """Manyetik tork hesaplama"""
#     return np.abs(m * B)

# torque = magnetic_torque(m_particle, B)

# # Sonuçlar
# print(f"Bobin 1 İndüktansı (L1): {L1:.2e} H")
# print(f"Bobin 2 İndüktansı (L2): {L2:.2e} H")
# print(f"Karşılıklı İndüktans (M): {M:.2e} H")
# print(f"Rezonans Frekansı (f_res): {f_res:.2e} Hz")
# print(f"Manyetik Alan (B): {B:.2e} T")
# print(f"Nanoparçacık Manyetik Momenti: {m_particle:.2e} Am^2")
# print(f"Nanoparçacık Manyetik Tork: {torque:.2e} Nm")

# # Görselleştirme
# fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
# u = np.linspace(0, 2 * pi, 100)
# v = np.linspace(0, pi, 100)
# x = r1 * np.outer(np.cos(u), np.sin(v))
# y = r1 * np.outer(np.sin(u), np.sin(v))
# z = l * np.outer(np.ones(np.size(u)), np.cos(v))
# ax.plot_surface(x, y, z, color='b', alpha=0.5)
# ax.set_title("Bobin 1 Görselleştirmesi")
# plt.show()


### Bitirme Deneme 3 diyelim: Bobinlerin yarıçapı 0.5 ve N= 10 olacak.
import numpy as np
import matplotlib.pyplot as plt

# Bobin parametreleri
radius = 0.5  # cm
turns = 10
current = 1.0  # Amper
mu_0 = 4 * np.pi * 1e-7  # Manyetik geçirgenlik sabiti (T·m/A)

# Pozisyonlar
bobin1_center = np.array([-1.0, 0])  # cm
bobin2_center = np.array([1.0, 0])   # cm

# Manyetik alanı hesapla
def magnetic_field(x, y, center, radius, turns, current):
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    # Eğer r sıfırsa manyetik alan sonsuza gider, bunu önlemek için bir sınır koyuyoruz
    r = np.maximum(r, 1e-6)
    B_magnitude = (mu_0 * turns * current * radius**2) / (2 * (radius**2 + r**2)**(3/2))
    Bx = B_magnitude * (x - center[0]) / r
    By = B_magnitude * (y - center[1]) / r
    return Bx, By

x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

# Manyetik alan
Bx1, By1 = np.zeros_like(X), np.zeros_like(Y)
Bx2, By2 = np.zeros_like(X), np.zeros_like(Y)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Bx1[i, j], By1[i, j] = magnetic_field(X[i, j], Y[i, j], bobin1_center, radius, turns, current)
        Bx2[i, j], By2[i, j] = magnetic_field(X[i, j], Y[i, j], bobin2_center, radius, turns, current)

# Toplam manyetik alan
Bx = Bx1 + Bx2
By = By1 + By2

plt.figure(figsize=(8, 8))
plt.streamplot(X, Y, Bx, By, color=np.sqrt(Bx**2 + By**2), cmap='plasma', density=1.5)
plt.scatter(*bobin1_center, color='blue', label='Bobin 1')
plt.scatter(*bobin2_center, color='red', label='Bobin 2')
plt.title("Yan Yana İki Bobinin Manyetik Alanı")
plt.xlabel("x (cm)")
plt.ylabel("y (cm)")
plt.legend()
plt.colorbar(label="Manyetik Alan Şiddeti (T)")
plt.axis('equal')
plt.grid()
plt.show()




import numpy as np
import matplotlib.pyplot as plt

# Bobin parametreleri
radius = 0.5  # cm
turns = 10
current = 1.0  # Amper
mu_0 = 4 * np.pi * 1e-7  # Manyetik geçirgenlik sabiti (T·m/A)

# Manyetik alanı hesaplayan fonksiyon
def magnetic_field_at_center(distance, radius, turns, current):
    r = distance / 2
    B = (mu_0 * turns * current * radius**2) / (2 * (radius**2 + r**2)**(3/2))
    return B

# Kuvvet hesaplama fonksiyonu
def magnetic_force(distance, radius, turns, current):
    B = magnetic_field_at_center(distance, radius, turns, current)
    force = turns * current * B * (2 * np.pi * radius)  # Kuvvet = N * I * B * uzunluk
    return force

# Mesafe aralığı (d)
distances = np.linspace(0.5, 5.0, 100)  # 0.5 cm'den 5 cm'e kadar
forces = np.array([magnetic_force(d, radius, turns, current) for d in distances])

# En optimum mesafe (maksimum kuvvet)
optimal_distance = distances[np.argmax(forces)]
optimal_force = np.max(forces)

# Grafik
plt.figure(figsize=(8, 6))
plt.plot(distances, forces, label="Manyetik Kuvvet", color="blue")
plt.axvline(optimal_distance, color="red", linestyle="--", label=f"Optimum Mesafe: {optimal_distance:.2f} cm")
plt.title("İki Bobin Arasındaki Mesafeye Göre Manyetik Kuvvet")
plt.xlabel("Mesafe (d) [cm]")
plt.ylabel("Kuvvet (N)")
plt.legend()
plt.grid()
plt.show()

# Optimum mesafeyi yazdır
print(f"Manyetik kuvvetin en optimum olduğu mesafe: {optimal_distance:.2f} cm")




import numpy as np
import matplotlib.pyplot as plt

# Bobin parametreleri
radius = 0.5  # cm
turns = 10
current = 1.0  # Amper
mu_0 = 4 * np.pi * 1e-7  # Manyetik geçirgenlik sabiti (T·m/A)

# Manyetik alanın zamana bağlı fonksiyonu
def magnetic_field_time(distance, radius, turns, current, t, f):
    r = distance / 2
    B_max = (mu_0 * turns * current * radius**2) / (2 * (radius**2 + r**2)**(3/2))
    return B_max * np.sin(2 * np.pi * f * t)

# Nanoparçacıkların hareketi (x ve y ekseninde)
def nanoparticle_motion(distance, radius, turns, current, size, f, time_steps):
    x, y = 0, 0  # Başlangıç koordinatları
    positions = []
    for t in time_steps:
        B_t = magnetic_field_time(distance, radius, turns, current, t, f)
        Fx = size * B_t * np.cos(2 * np.pi * f * t)  # x yönünde kuvvet
        Fy = size * B_t * np.sin(2 * np.pi * f * t)  # y yönünde kuvvet
        x += Fx * 1e-3  # Küçük bir zaman adımı (ör: ms)
        y += Fy * 1e-3
        positions.append((x, y))
    return np.array(positions)

# Frekanslar için optimal mesafe hesaplamak
frequencies = np.linspace(1, 100, 100)  # 1 Hz'den 100 Hz'e kadar
optimal_distances = []

for f in frequencies:
    time_steps = np.linspace(0, 0.1, 1000)  # 0.1 saniye boyunca
    nanoparticle_sizes = np.linspace(5, 60, 12)  # 5 nm'den 60 nm'e
    optimal_distance_for_f = []

    # Her frekans için optimum mesafe hesapla
    for size in nanoparticle_sizes:
        positions = nanoparticle_motion(2.0, radius, turns, current, size, f, time_steps)  # Başlangıçta 2.0 cm mesafesi
        distance = np.linalg.norm(positions[-1])  # Nanoparçacığın son pozisyonu
        optimal_distance_for_f.append(distance)

    optimal_distances.append(np.mean(optimal_distance_for_f))  # Her frekans için ortalama optimum mesafe

# Grafik
plt.figure(figsize=(10, 6))
plt.plot(frequencies, optimal_distances, label="Optimum Mesafe", color="blue")
plt.title("Optimum Mesafe ile Frekans Arasındaki İlişki")
plt.xlabel("Frekans (Hz)")
plt.ylabel("Optimum Mesafe (cm)")
plt.legend()
plt.grid(True)
plt.show()




import numpy as np
import matplotlib.pyplot as plt

# Bobin parametreleri
radius = 0.5  # Bobin yarıçapı (cm)
turns = 10  # Sarım sayısı
current = 1.0  # Akım (Amper)
mu_0 = 4 * np.pi * 1e-7  # Manyetik geçirgenlik (T·m/A)
f = 50  # Frekans (Hz)
optimal_distance = 2.0  # Daha önce hesaplanan optimal mesafe (cm)

# Manyetik alanın zamana bağlı fonksiyonu
def magnetic_field_time(distance, radius, turns, current, t, f):
    r = distance / 2
    B_max = (mu_0 * turns * current * radius**2) / (2 * (radius**2 + r**2)**(3/2))
    return B_max * np.sin(2 * np.pi * f * t)

# Nanoparçacıkların hareketi (x ve y ekseninde)
def nanoparticle_motion(distance, radius, turns, current, size, f, time_steps):
    x, y = 0, 0  # Başlangıç koordinatları
    positions = []
    for t in time_steps:
        B_t = magnetic_field_time(distance, radius, turns, current, t, f)
        Fx = size * B_t * np.cos(2 * np.pi * f * t)  # x yönünde kuvvet
        Fy = size * B_t * np.sin(2 * np.pi * f * t)  # y yönünde kuvvet
        x += Fx * 1e-3  # Küçük bir zaman adımı (ör: ms)
        y += Fy * 1e-3
        positions.append((x, y))
    return np.array(positions)

# Simülasyon parametreleri
time_steps = np.linspace(0, 0.1, 1000)  # 0.1 saniye boyunca
nanoparticle_sizes = np.linspace(5, 60, 12)  # 5 nm'den 60 nm'e

# Her nanoparçacık boyutu için hareketi hesapla
motions = []
for size in nanoparticle_sizes:
    positions = nanoparticle_motion(optimal_distance, radius, turns, current, size, f, time_steps)
    motions.append((size, positions))

# Grafik düzeni (3x4, toplam 12 grafik)
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

# Her nanoparçacık için grafik çizimi
for i, (size, positions) in enumerate(motions):
    ax = axes[i]
    ax.plot(positions[:, 0], positions[:, 1], label=f"{size:.1f} nm", color="blue")
    ax.set_title(f"Boyut: {size:.1f} nm")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("x ekseni (cm)")
    ax.set_ylabel("y ekseni (cm)")
    ax.legend()
    ax.grid()

# Boş kalan alt grafikler için
for j in range(len(motions), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()




import numpy as np
import matplotlib.pyplot as plt

# Bobin parametreleri
radius = 0.5  # Bobin yarıçapı (cm)
turns = 10  # Sarım sayısı
current = 1.0  # Akım (Amper)
mu_0 = 4 * np.pi * 1e-7  # Manyetik geçirgenlik (T·m/A)
f = 50  # Frekans (Hz)
optimal_distance = 2.0  # Daha önce hesaplanan optimal mesafe (cm)

# Manyetik alanın zamana bağlı fonksiyonu
def magnetic_field_time(distance, radius, turns, current, t, f):
    r = distance / 2
    B_max = (mu_0 * turns * current * radius**2) / (2 * (radius**2 + r**2)**(3/2))
    return B_max * np.sin(2 * np.pi * f * t)

# Nanoparçacıkların hareketi (x ve y ekseninde)
def nanoparticle_motion(distance, radius, turns, current, size, f, time_steps):
    x, y = 0, 0  # Başlangıç koordinatları
    positions = []
    for t in time_steps:
        B_t = magnetic_field_time(distance, radius, turns, current, t, f)
        Fx = size * B_t * np.cos(2 * np.pi * f * t)  # x yönünde kuvvet
        Fy = size * B_t * np.sin(2 * np.pi * f * t)  # y yönünde kuvvet
        x += Fx * 1e-3  # Küçük bir zaman adımı (ör: ms)
        y += Fy * 1e-3
        positions.append((x, y))
    return np.array(positions)

# Simülasyon parametreleri
time_steps = np.linspace(0, 0.1, 1000)  # 0.1 saniye boyunca
nanoparticle_sizes = np.linspace(5, 60, 12)  # 5 nm'den 60 nm'e

# Her nanoparçacık boyutu için hareketi hesapla
motions = []
for size in nanoparticle_sizes:
    positions = nanoparticle_motion(optimal_distance, radius, turns, current, size, f, time_steps)
    motions.append((size, positions))

# Her nanoparçacık için ayrı grafikler
for i, (size, positions) in enumerate(motions):
    plt.figure(figsize=(8, 6))
    plt.plot(positions[:, 0], positions[:, 1], label=f"Nanoparçacık ({size:.1f} nm)", color="blue")
    plt.title(f"Nanoparçacık Hareketi (Boyut: {size:.1f} nm)")
    plt.xlabel("x ekseni (cm)")
    plt.ylabel("y ekseni (cm)")
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
    plt.legend()
    plt.grid()
    plt.show()




import numpy as np
import matplotlib.pyplot as plt

# Bobin parametreleri
radius = 0.5  # Bobin yarıçapı (cm)
turns = 10  # Bobin sarım sayısı
current = 1.0  # Akım (Amper)
mu_0 = 4 * np.pi * 1e-7  # Manyetik geçirgenlik sabiti (T·m/A)

# Manyetik alanı hesaplayan fonksiyon
def magnetic_field_at_center(distance, radius, turns, current):
    r = distance / 2
    B = (mu_0 * turns * current * radius**2) / (2 * (radius**2 + r**2)**(3/2))
    return B

# Kuvvet hesaplama fonksiyonu
def magnetic_force(distance, radius, turns, current, size):
    B = magnetic_field_at_center(distance, radius, turns, current)
    force = turns * current * B * (2 * np.pi * radius) * (size / 10)  # Kuvvet = N * I * B * uzunluk * boyut oranı
    return force

# Optimum mesafeyi belirleme
distances = np.linspace(0.5, 5.0, 100)  # 0.5 cm ile 5.0 cm arasında mesafeler
forces_at_distances = np.array([magnetic_force(d, radius, turns, current, size=1) for d in distances])  # Varsayılan nanoparçacık boyutu: 1 nm
optimal_distance = distances[np.argmax(forces_at_distances)]
optimal_force = np.max(forces_at_distances)

# Nanoparçacık boyutları ve hareketleri
nanoparticle_sizes = np.linspace(5, 60, 12)  # 5 nm'den 60 nm'e kadar 12 farklı boyut
movement_range = 0.1 * optimal_distance  # Hareket ±%10 sapma

# Nanoparçacık için manyetik alan ve kuvvet hesaplama
nanoparticle_data = []
for size in nanoparticle_sizes:
    distances_with_movement = np.linspace(optimal_distance - movement_range, optimal_distance + movement_range, 10)
    forces = [magnetic_force(d, radius, turns, current, size) for d in distances_with_movement]
    max_force = max(forces)
    nanoparticle_data.append((size, max_force))

# En optimum nanoparçacık boyutu
optimal_nanoparticle = max(nanoparticle_data, key=lambda x: x[1])  # Kuvvete göre sıralama
optimal_size, optimal_F = optimal_nanoparticle

# Görselleştirme
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 1. Grafik: Bobinler
axs[0, 0].add_artist(plt.Circle((-optimal_distance/2, 0), radius, color="blue", fill=False, lw=2, label="Bobin 1"))
axs[0, 0].add_artist(plt.Circle((optimal_distance/2, 0), radius, color="green", fill=False, lw=2, label="Bobin 2"))
axs[0, 0].set_xlim(-optimal_distance, optimal_distance)
axs[0, 0].set_ylim(-1, 1)
axs[0, 0].set_aspect('equal', adjustable='datalim')
axs[0, 0].set_title("Bobinlerin Görseli")
axs[0, 0].legend()
axs[0, 0].grid()

# 2. Grafik: Mesafe ve Kuvvet İlişkisi
axs[0, 1].plot(distances, forces_at_distances, label="Manyetik Kuvvet", color="blue")
axs[0, 1].axvline(optimal_distance, color="red", linestyle="--", label=f"Optimum Mesafe: {optimal_distance:.2f} cm")
axs[0, 1].set_title("Bobinler Arasındaki Mesafeye Göre Manyetik Kuvvet")
axs[0, 1].set_xlabel("Mesafe (cm)")
axs[0, 1].set_ylabel("Kuvvet (N)")
axs[0, 1].legend()
axs[0, 1].grid()

# 3. Grafik: Nanoparçacık Boyutu ve Kuvvet İlişkisi
sizes, max_forces = zip(*nanoparticle_data)
axs[1, 0].plot(sizes, max_forces, label="Nanoparçacık Boyutu", marker='o', color="green")
axs[1, 0].axvline(optimal_size, color="red", linestyle="--", label=f"Optimum Boyut: {optimal_size:.2f} nm")
axs[1, 0].set_title("Nanoparçacık Boyutuna Göre Manyetik Kuvvet")
axs[1, 0].set_xlabel("Nanoparçacık Boyutu (nm)")
axs[1, 0].set_ylabel("Kuvvet (N)")
axs[1, 0].legend()
axs[1, 0].grid()

# 4. Grafik: Manyetik Alan (Bobinler Arasında)
x = np.linspace(-optimal_distance, optimal_distance, 500)
B_field = [magnetic_field_at_center(abs(i), radius, turns, current) for i in x]
axs[1, 1].plot(x, B_field, color="purple", label="Manyetik Alan")
axs[1, 1].set_title("Bobinler Arasındaki Manyetik Alan")
axs[1, 1].set_xlabel("Mesafe (cm)")
axs[1, 1].set_ylabel("Manyetik Alan (T)")
axs[1, 1].legend()
axs[1, 1].grid()

plt.tight_layout()
plt.show()

# Çıktılar
print(f"Bobinler arasındaki optimum mesafe: {optimal_distance:.2f} cm")
print(f"Manyetik alan ve kuvveti en optimum şekilde kullanan nanoparçacık boyutu: {optimal_size:.2f} nm")




import numpy as np
import matplotlib.pyplot as plt

# Bobin parametreleri
radius = 0.5  # Bobin yarıçapı (cm)
turns = 10  # Sarım sayısı
current = 1.0  # Akım (Amper)
mu_0 = 4 * np.pi * 1e-7  # Manyetik geçirgenlik (T·m/A)
f = 50  # Frekans (Hz)

# Manyetik alanın zamana bağlı fonksiyonu
def magnetic_field_time(distance, radius, turns, current, t, f):
    r = distance / 2
    B_max = (mu_0 * turns * current * radius**2) / (2 * (radius**2 + r**2)**(3/2))
    return B_max * np.sin(2 * np.pi * f * t)

# Kuvvetin zamana bağlı fonksiyonu
def magnetic_force_time(distance, radius, turns, current, size, t, f):
    B = magnetic_field_time(distance, radius, turns, current, t, f)
    force = turns * current * B * (2 * np.pi * radius) * (size / 10)
    return force

# Nanoparçacıkların hareketi (x ve y ekseninde)
def nanoparticle_motion(distance, radius, turns, current, size, f, time_steps):
    x, y = 0, 0  # Başlangıç koordinatları
    positions = []
    for t in time_steps:
        B_t = magnetic_field_time(distance, radius, turns, current, t, f)
        Fx = size * B_t * np.cos(2 * np.pi * f * t)  # x yönünde kuvvet
        Fy = size * B_t * np.sin(2 * np.pi * f * t)  # y yönünde kuvvet
        x += Fx * 1e-3  # Küçük bir zaman adımı (ör: ms)
        y += Fy * 1e-3
        positions.append((x, y))
    return np.array(positions)

# Simülasyon parametreleri
time_steps = np.linspace(0, 0.1, 1000)  # 0.1 saniye boyunca
nanoparticle_sizes = np.linspace(5, 60, 12)  # 5 nm'den 60 nm'e

# Her nanoparçacık boyutu için hareketi hesapla
motions = []
for size in nanoparticle_sizes:
    positions = nanoparticle_motion(optimal_distance, radius, turns, current, size, f, time_steps)
    motions.append((size, positions))

# En uygun nanoparçacığı seç
optimal_motion = max(motions, key=lambda m: np.max(np.linalg.norm(m[1], axis=1)))
optimal_size = optimal_motion[0]

# Grafikler
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# 1. Grafik: Bobinlerin Görseli
axs[0].add_artist(plt.Circle((-optimal_distance/2, 0), radius, color="blue", fill=False, lw=2, label="Bobin 1"))
axs[0].add_artist(plt.Circle((optimal_distance/2, 0), radius, color="green", fill=False, lw=2, label="Bobin 2"))
axs[0].set_xlim(-optimal_distance, optimal_distance)
axs[0].set_ylim(-1, 1)
axs[0].set_title("Bobinlerin Görseli ve Hareket Alanı")
axs[0].legend()
axs[0].grid()

# Nanoparçacık Hareketi (En uygun boyut)
positions = optimal_motion[1]
axs[0].plot(positions[:, 0], positions[:, 1], color="red", label=f"Optimum Nanoparçacık Hareketi ({optimal_size:.1f} nm)")
axs[0].legend()

# 2. Grafik: Hareketlerin Boyutlara Göre Karşılaştırılması
for size, pos in motions:
    axs[1].plot(pos[:, 0], pos[:, 1], label=f"{size:.1f} nm")
axs[1].set_title("Nanoparçacık Hareketi (Boyutlara Göre)")
axs[1].set_xlabel("x ekseni (cm)")
axs[1].set_ylabel("y ekseni (cm)")
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.show()

# Çıktılar
print(f"Manyetik alan ve kuvveti en uygun şekilde kullanan nanoparçacık boyutu: {optimal_size:.1f} nm")




import numpy as np
import matplotlib.pyplot as plt

# Bobin parametreleri
radius = 0.5  # Bobin yarıçapı (cm)
turns = 10  # Bobin sarım sayısı
current = 1.0  # Akım (Amper)
mu_0 = 4 * np.pi * 1e-7  # Manyetik geçirgenlik sabiti (T·m/A)

# Manyetik alanı hesaplayan fonksiyon
def magnetic_field_at_center(distance, radius, turns, current):
    r = distance / 2
    B = (mu_0 * turns * current * radius**2) / (2 * (radius**2 + r**2)**(3/2))
    return B

# Kuvvet hesaplama fonksiyonu
def magnetic_force(distance, radius, turns, current, size):
    B = magnetic_field_at_center(distance, radius, turns, current)
    force = turns * current * B * (2 * np.pi * radius) * (size / 10)  # Kuvvet = N * I * B * uzunluk * boyut oranı
    return force

# Optimum mesafeyi belirleme
distances = np.linspace(0.5, 5.0, 100)  # 0.5 cm ile 5.0 cm arasında mesafeler
forces_at_distances = np.array([magnetic_force(d, radius, turns, current, size=1) for d in distances])  # Varsayılan nanoparçacık boyutu: 1 nm
optimal_distance = distances[np.argmax(forces_at_distances)]
optimal_force = np.max(forces_at_distances)

# Nanoparçacık boyutları ve hareketleri
nanoparticle_sizes = np.linspace(5, 60, 12)  # 5 nm'den 60 nm'e kadar 12 farklı boyut
movement_range = 0.1 * optimal_distance  # Hareket ±%10 sapma

# Nanoparçacık için manyetik alan ve kuvvet hesaplama
nanoparticle_data = []
for size in nanoparticle_sizes:
    distances_with_movement = np.linspace(optimal_distance - movement_range, optimal_distance + movement_range, 10)
    forces = [magnetic_force(d, radius, turns, current, size) for d in distances_with_movement]
    max_force = max(forces)
    nanoparticle_data.append((size, max_force))

# En optimum nanoparçacık boyutu
optimal_nanoparticle = max(nanoparticle_data, key=lambda x: x[1])  # Kuvvete göre sıralama
optimal_size, optimal_F = optimal_nanoparticle

# Görselleştirme
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 1. Grafik: Bobinler
axs[0, 0].add_artist(plt.Circle((-optimal_distance/2, 0), radius, color="blue", fill=False, lw=2, label="Bobin 1"))
axs[0, 0].add_artist(plt.Circle((optimal_distance/2, 0), radius, color="green", fill=False, lw=2, label="Bobin 2"))
axs[0, 0].set_xlim(-optimal_distance, optimal_distance)
axs[0, 0].set_ylim(-1, 1)
axs[0, 0].set_aspect('equal', adjustable='datalim')
axs[0, 0].set_title("Bobinlerin Görseli")
axs[0, 0].legend()
axs[0, 0].grid()

# 2. Grafik: Mesafe ve Kuvvet İlişkisi
axs[0, 1].plot(distances, forces_at_distances, label="Manyetik Kuvvet", color="blue")
axs[0, 1].axvline(optimal_distance, color="red", linestyle="--", label=f"Optimum Mesafe: {optimal_distance:.2f} cm")
axs[0, 1].set_title("Bobinler Arasındaki Mesafeye Göre Manyetik Kuvvet")
axs[0, 1].set_xlabel("Mesafe (cm)")
axs[0, 1].set_ylabel("Kuvvet (N)")
axs[0, 1].legend()
axs[0, 1].grid()

# 3. Grafik: Nanoparçacık Boyutu ve Kuvvet İlişkisi
sizes, max_forces = zip(*nanoparticle_data)
axs[1, 0].plot(sizes, max_forces, label="Nanoparçacık Boyutu", marker='o', color="green")
axs[1, 0].axvline(optimal_size, color="red", linestyle="--", label=f"Optimum Boyut: {optimal_size:.2f} nm")
axs[1, 0].set_title("Nanoparçacık Boyutuna Göre Manyetik Kuvvet")
axs[1, 0].set_xlabel("Nanoparçacık Boyutu (nm)")
axs[1, 0].set_ylabel("Kuvvet (N)")
axs[1, 0].legend()
axs[1, 0].grid()

# 4. Grafik: Manyetik Alan (Bobinler Arasında)
x = np.linspace(-optimal_distance, optimal_distance, 500)
B_field = [magnetic_field_at_center(abs(i), radius, turns, current) for i in x]
axs[1, 1].plot(x, B_field, color="purple", label="Manyetik Alan")
axs[1, 1].set_title("Bobinler Arasındaki Manyetik Alan")
axs[1, 1].set_xlabel("Mesafe (cm)")
axs[1, 1].set_ylabel("Manyetik Alan (T)")
axs[1, 1].legend()
axs[1, 1].grid()

plt.tight_layout()
plt.show()

# Çıktılar
print(f"Bobinler arasındaki optimum mesafe: {optimal_distance:.2f} cm")
print(f"Manyetik alan ve kuvveti en optimum şekilde kullanan nanoparçacık boyutu: {optimal_size:.2f} nm")

