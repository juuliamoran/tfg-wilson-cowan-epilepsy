import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from matplotlib.lines import Line2D

# ============================================================
# PARÀMETRES (els de l'article)
# ============================================================
a_e, a_i = 1.2, 1.0
theta_e, theta_i = 2.8, 4.0

w_ee, w_ei = 12.0, 4.0
w_ie, w_ii = 13.0, 11.0

Q = 0.0
P = 0.0

# ============================================================
# SIGMOIDES
# ============================================================
def S_e(x):
    return 1/(1 + np.exp(-a_e*(x - theta_e))) - 1/(1 + np.exp(a_e*theta_e))

def S_i(x):
    return 1/(1 + np.exp(-a_i*(x - theta_i))) - 1/(1 + np.exp(a_i*theta_i))

# derivades
def dS_e(x):
    sig = 1/(1 + np.exp(-a_e*(x - theta_e)))
    return a_e * sig * (1 - sig)

def dS_i(x):
    sig = 1/(1 + np.exp(-a_i*(x - theta_i)))
    return a_i * sig * (1 - sig)

# inverses
def S_inv_e(y):
    c = 1/(1 + np.exp(a_e*theta_e))
    return theta_e - (1/a_e)*np.log((1 - c - y)/(y + c))

def S_inv_i(y):
    c = 1/(1 + np.exp(a_i*theta_i))
    return theta_i - (1/a_i)*np.log((1 - c - y)/(y + c))

# ============================================================
# SISTEMA
# ============================================================
def system(z):
    E, I = z

    dE = -E + (1 - E) * S_e(w_ee*E - w_ei*I + P)
    dI = -I + (1 - I) * S_i(w_ie*E - w_ii*I + Q)

    return np.array([dE, dI])

# ============================================================
# JACOBIANA
# ============================================================
def jacobian(z):
    E, I = z

    xe = w_ee*E - w_ei*I + P
    xi = w_ie*E - w_ii*I + Q

    Se = S_e(xe)
    Si = S_i(xi)

    dSe = dS_e(xe)
    dSi = dS_i(xi)

    J11 = -1 - Se + (1 - E)*dSe*w_ee
    J12 = (1 - I)*dSi*w_ie
    J21 = (1 - E)*dSe*(-w_ei)
    J22 = -1 - Si + (1 - I)*dSi*(-w_ii)

    return np.array([[J11, J12],
                     [J21, J22]])

# ============================================================
# ISOCLINES
# ============================================================
def I_nullcline(E):
    arg = E / (1 - E)
    return (w_ee*E - S_inv_e(arg) + P) / w_ei

def E_nullcline(I):
    arg = I / (1 - I)
    return (w_ii*I + S_inv_i(arg) - Q) / w_ie

E_vals = np.linspace(-0.1, 0.999, 500)
I_vals = np.linspace(-0.1, 0.999, 500)

I_nc = np.array([I_nullcline(E) for E in E_vals])
E_nc = np.array([E_nullcline(I) for I in I_vals])

mask_I = np.isfinite(I_nc)
mask_E = np.isfinite(E_nc)

# ============================================================
# PUNTS FIXOS (grid + root finding)
# ============================================================
fixed_points = []

def add_if_new(p):
    if np.any(np.isnan(p)):
        return
    for q in fixed_points:
        if np.linalg.norm(p - q) < 1e-4:
            return
    fixed_points.append(p)

grid = np.linspace(0, 1, 20)

for E0 in grid:
    for I0 in grid:
        sol = root(system, [E0, I0]).x

        if np.linalg.norm(system(sol)) < 1e-8:
            add_if_new(sol)

fixed_points = sorted(fixed_points, key=lambda x: x[0])

# ============================================================
# CLASSIFICACIÓ
# ============================================================
print("\nPUNTS FIXOS\n" + "="*40)

info = []

for i, pf in enumerate(fixed_points):
    J = jacobian(pf)
    eig = np.linalg.eigvals(J)

    stable = np.all(np.real(eig) < 0)

    if np.sign(np.real(eig[0])) * np.sign(np.real(eig[1])) < 0:
        tipo = "sella"
    elif np.any(np.imag(eig) != 0):
        tipo = "focus estable" if stable else "focus inestable"
    else:
        tipo = "node estable" if stable else "node inestable"

    info.append((pf, stable, tipo))

    print(f"PF {i+1}: E*={pf[0]:.4f}, I*={pf[1]:.4f}, {tipo}")
    print(f"        λ = {eig}\n")

# ============================================================
# FIGURA
# ============================================================
plt.figure(figsize=(8, 8))

# isoclines
plt.plot(I_nc[mask_I], E_vals[mask_I], 'orange', lw=2, zorder = 1)#, label=r'dE/dt=0')
plt.plot(I_vals[mask_E], E_nc[mask_E], 'blue', lw=2, zorder = 1)#, label=r'dI/dt=0')

# punts fixos
for i, (pf, stable, tipo) in enumerate(info, start=1):
    color = "green" if stable else "black"

    plt.scatter(
        pf[1], pf[0],
        c=color,
        s=120,
        edgecolor='k',
        zorder=10
    )

    plt.annotate(
        f"PF{i}",
        (pf[1], pf[0]),
        xytext=(10, -15),
        textcoords="offset points",
        fontsize=11,
        fontweight="bold"
    )

legend_elements = [
    Line2D([0], [0], color='orange', lw=2, label=r'dE/dt=0'),
    Line2D([0], [0], color='blue', lw=2, label=r'dI/dt=0'),
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor='green', markeredgecolor='k',
           markersize=8, label='Estable'),
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor='black', markeredgecolor='k',
           markersize=8, label='Inestable (sella)')
]

plt.xlabel("I (inhibidora)")
plt.ylabel("E (excitadora)")
#plt.title(f"Isoclines Wilson–Cowan (P={P})")
plt.legend(handles=legend_elements)
plt.grid()
plt.xlim(-0.2, 0.5)
plt.ylim(-0.05, 0.5)
plt.show()