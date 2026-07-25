import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# 1. PARÀMETRES
a_e = 1.3; a_i = 2.0
theta_e = 4.0; theta_i = 3.7
w_ee = 16; w_ei = 12.0
w_ie = 15.0; w_ii = 3.0
Q = 0.0; P = 1.25

# 2. SIGMOIDALS
def S_e(x):
    return 1/(1+np.exp(-a_e*(x-theta_e))) - 1/(1+np.exp(a_e*theta_e))
def S_i(x):
    return 1/(1+np.exp(-a_i*(x-theta_i))) - 1/(1+np.exp(a_i*theta_i))
def dS_e(x):
    sig = 1/(1+np.exp(-a_e*(x-theta_e))); return a_e*sig*(1-sig)
def dS_i(x):
    sig = 1/(1+np.exp(-a_i*(x-theta_i))); return a_i*sig*(1-sig)

def S_inv_e(y):
    c_val = 1/(1+np.exp(a_e*theta_e))
    with np.errstate(invalid='ignore', divide='ignore'):
        res = theta_e - (1/a_e)*np.log((1-c_val-y)/(y+c_val))
    return np.where((y > S_e(-100)) & (y < S_e(100)), res, np.nan)

def S_inv_i(y):
    c_val = 1/(1+np.exp(a_i*theta_i))
    with np.errstate(invalid='ignore', divide='ignore'):
        res = theta_i - (1/a_i)*np.log((1-c_val-y)/(y+c_val))
    return np.where((y > S_i(-100)) & (y < S_i(100)), res, np.nan)

# 3. ISOCLINES
def I_nullcline(E):
    arg = E / (1 - E)
    return (w_ee*E - S_inv_e(arg) + P) / w_ei

def E_nullcline(I):
    arg = I / (1 - I)
    return (w_ii*I + S_inv_i(arg) - Q) / w_ie

E_vals = np.linspace(-0.1, 0.999, 10000)
I_vals = np.linspace(-0.1, 0.999, 10000)

I_null_vals = I_nullcline(E_vals)
E_null_vals = E_nullcline(I_vals)

mask_e = ~np.isnan(I_null_vals)
mask_i = ~np.isnan(E_null_vals)

# 4. ODE
def WC_ode(y, t):
    E, I = y
    return [-E + (1-E)*S_e(w_ee*E - w_ei*I + P),
            -I + (1-I)*S_i(w_ie*E - w_ii*I + Q)]

# 5. JACOBIANA I PUNTS FIXOS
def jacobiana(pf):
    E, I = pf
    xe = w_ee*E - w_ei*I + P; xi = w_ie*E - w_ii*I + Q
    Se = S_e(xe); Si = S_i(xi); dSe = dS_e(xe); dSi = dS_i(xi)
    return np.array([
        [-1-Se+(1-E)*dSe*w_ee,  -(1-E)*dSe*w_ei],
        [ (1-I)*dSi*w_ie,       -1-Si-(1-I)*dSi*w_ii]
    ])

def system_eq(y):
    E, I = y
    return [-E+(1-E)*S_e(w_ee*E-w_ei*I+P),
            -I+(1-I)*S_i(w_ie*E-w_ii*I+Q)]

punts_fixos = []
for E0 in np.linspace(0, 1, 20):
    for I0 in np.linspace(0, 1, 20):
        try:
            sol = fsolve(system_eq, [E0, I0], full_output=True)
            if sol[2] == 1:
                pf = sol[0]
                if np.sqrt(sum(np.array(system_eq(pf))**2)) < 1e-9:
                    if not any(np.sqrt(sum((pf-p)**2)) < 1e-4
                               for p in punts_fixos):
                        punts_fixos.append(pf)
        except: pass

punts_fixos = sorted(punts_fixos, key=lambda p: p[0])

pf_info = []
for i, pf in enumerate(punts_fixos):
    J    = jacobiana(pf)
    eigv = np.linalg.eigvals(J)
    estable = all(np.real(eigv) < 0)
    if np.real(eigv[0])*np.real(eigv[1]) < 0:
        tipus = "sella"
    elif any(np.imag(eigv) != 0):
        tipus = "espiral estable" if estable else "espiral inestable"
    else:
        tipus = "node estable" if estable else "node inestable"
    pf_info.append({"E": pf[0], "I": pf[1], "estable": estable, "tipus": tipus})
    print(f"PF{i+1}: E*={pf[0]:.5f}  I*={pf[1]:.5f}  {tipus.upper()}")
    print(f"     lambda = {np.real(eigv[0]):.4f}+{np.imag(eigv[0]):.4f}i,"
          f"  {np.real(eigv[1]):.4f}+{np.imag(eigv[1]):.4f}i")

# 6. INTEGRACIÓ
times_long = np.linspace(0, 600, 12000)

sol_inner = odeint(WC_ode, [0.02, 0.02], times_long)
sol_outer = odeint(WC_ode, [0.45, 0.45], times_long)

# Cicle límit convergit (últim 15%)
n_start = int(0.85 * len(times_long))
cycle_E = sol_inner[n_start:, 0]
cycle_I = sol_inner[n_start:, 1]

# 7. GRÀFIC
fig, ax = plt.subplots(figsize=(7, 6))

# Isoclines
ax.plot(I_null_vals[mask_e], E_vals[mask_e],
        color="darkorange", lw=2.5, label="dE/dt = 0")
ax.plot(I_vals[mask_i], E_null_vals[mask_i],
        color="steelblue",  lw=2.5, label="dI/dt = 0")

# Trajectòries transients (difuminades)
n_trans = int(0.85 * len(times_long))
ax.plot(sol_inner[:n_trans, 1], sol_inner[:n_trans, 0],
        color="grey", alpha=0.5, lw=1.2, ls="--")
ax.plot(sol_outer[:n_trans, 1], sol_outer[:n_trans, 0],
        color="grey", alpha=0.5, lw=1.2, ls="--")

# Cicle límit
ax.plot(cycle_I, cycle_E, color="#7B2D8B", lw=3, label="Cicle límit")

# Fletxes al cicle límit
n_arrows  = 2
cycle_len = len(cycle_E)
arrow_idx = np.round(np.linspace(0, cycle_len-1,
                                  n_arrows+1)).astype(int)[:-1]
#gap = 10

for k in arrow_idx:
    #k2 = min(k + gap, cycle_len - 1)
    ax.annotate("",
        xy    =(cycle_I[k+10], cycle_E[k+10]),
        xytext=(cycle_I[k],   cycle_E[k]),
        arrowprops=dict(arrowstyle="-|>", lw=2, color= "black",  mutation_scale=20)) # 7B2D8B

# Punts fixos
for i, pf in enumerate(pf_info):
    col = "limegreen" if pf["estable"] else "black"
    mk  = "s" if pf["tipus"] == "sella" else "o"
    ax.plot(pf["I"], pf["E"], mk, ms=12,
            markerfacecolor=col, markeredgecolor="black",
            markeredgewidth=1.5, zorder=5)
    ax.text(pf["I"]-0.035, pf["E"], f"PF{i+1}",
            fontweight="bold", fontsize=10, zorder=6)

ax.set_xlim(-0.01, 0.5)
ax.set_ylim(-0.01, 0.5)
ax.set_xlabel("I  (activitat inhibidora)", fontsize=12)
ax.set_ylabel("E  (activitat excitadora)", fontsize=12)
ax.grid(alpha=0.3)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0], color="darkorange", lw=2.5, label="dE/dt = 0"),
    Line2D([0],[0], color="steelblue",  lw=2.5, label="dI/dt = 0"),
    Line2D([0],[0], color="#7B2D8B",    lw=3,   label="Cicle límit"),
    Line2D([0],[0], color="grey",       lw=1.2, ls="--",
           label="Trajectòries"),
]
ax.legend(handles=legend_elements, frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig("isoclines_cicle.png", dpi=150, bbox_inches="tight")
plt.show()