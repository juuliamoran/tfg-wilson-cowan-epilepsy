import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve

# ── 1. PARÀMETRES ────────────────────────────────────────────────
a_e     = 1.2;  a_i     = 1.0
theta_e = 2.8;  theta_i = 4.0
w_ee    = 12.0; w_ei    = 4.0
w_ie    = 13.0; w_ii    = 11.0
Q       = 0.0;  P       = 0.0

# ── 2. SIGMOIDALS I DERIVADES ────────────────────────────────────
def S_e(x):
    return 1/(1 + np.exp(-a_e*(x - theta_e))) - 1/(1 + np.exp(a_e*theta_e))

def S_i(x):
    return 1/(1 + np.exp(-a_i*(x - theta_i))) - 1/(1 + np.exp(a_i*theta_i))

def dS_e(x):
    sig = 1/(1 + np.exp(-a_e*(x - theta_e)))
    return a_e * sig * (1 - sig)

def dS_i(x):
    sig = 1/(1 + np.exp(-a_i*(x - theta_i)))
    return a_i * sig * (1 - sig)

def S_inv_e(y):
    c_val = 1/(1 + np.exp(a_e*theta_e))
    s_min = S_e(-100); s_max = S_e(100)
    with np.errstate(invalid='ignore', divide='ignore'):
        result = theta_e - (1/a_e)*np.log((1 - c_val - y)/(y + c_val))
    result = np.where((y > s_min) & (y < s_max), result, np.nan)
    return result

def S_inv_i(y):
    c_val = 1/(1 + np.exp(a_i*theta_i))
    s_min = S_i(-100); s_max = S_i(100)
    with np.errstate(invalid='ignore', divide='ignore'):
        result = theta_i - (1/a_i)*np.log((1 - c_val - y)/(y + c_val))
    result = np.where((y > s_min) & (y < s_max), result, np.nan)
    return result

# ── 3. ISOCLINES ─────────────────────────────────────────────────
def I_nullcline(E):
    arg = E / (1 - E)
    return (w_ee*E - S_inv_e(arg) + P) / w_ei

def E_nullcline(I):
    arg = I / (1 - I)
    return (w_ii*I + S_inv_i(arg) - Q) / w_ie

E_vals = np.linspace(-0.1, 0.999, 500)
I_vals = np.linspace(-0.1, 0.999, 500)

I_null_vals = I_nullcline(E_vals)
E_null_vals = E_nullcline(I_vals)

mask_e = ~np.isnan(I_null_vals)
mask_i = ~np.isnan(E_null_vals)

# ── 4. SISTEMA ODE ───────────────────────────────────────────────
def WC_ode(y, t):
    E, I = y
    dE = -E + (1 - E) * S_e(w_ee*E - w_ei*I + P)
    dI = -I + (1 - I) * S_i(w_ie*E - w_ii*I + Q)
    return [dE, dI]

def WC_back(y, t):
    E, I = y
    dE = E - (1 - E) * S_e(w_ee*E - w_ei*I + P)
    dI = I - (1 - I) * S_i(w_ie*E - w_ii*I + Q)
    return [dE, dI]

# ── 5. JACOBIANA ─────────────────────────────────────────────────
def jacobiana(pf):
    E, I = pf
    xe  = w_ee*E - w_ei*I + P
    xi  = w_ie*E - w_ii*I + Q
    Se  = S_e(xe);  Si  = S_i(xi)
    dSe = dS_e(xe); dSi = dS_i(xi)
    return np.array([
        [-1 - Se + (1-E)*dSe*w_ee,  -(1-E)*dSe*w_ei  ],
        [ (1-I)*dSi*w_ie,           -1 - Si - (1-I)*dSi*w_ii]
    ])

# ── 6. PUNTS FIXOS ───────────────────────────────────────────────
def system_eq(y):
    E, I = y
    return [
        -E + (1 - E) * S_e(w_ee*E - w_ei*I + P),
        -I + (1 - I) * S_i(w_ie*E - w_ii*I + Q)
    ]


punts_fixos = []
# Graella més densa i amb valors inicials variats
for E0 in np.linspace(0.001, 0.999, 40):
    for I0 in np.linspace(0.001, 0.999, 40):
        try:
            sol = fsolve(system_eq, [E0, I0], full_output=True)
            if sol[2] == 1:
                pf = np.array(sol[0])
                resid = np.sqrt(sum(np.array(system_eq(pf))**2))
                if resid < 1e-8:
                    if not any(np.sqrt(sum((pf - p)**2)) < 1e-4
                               for p in punts_fixos):
                        punts_fixos.append(pf)
        except:
            pass

punts_fixos = sorted(punts_fixos, key=lambda p: p[0])
print(f"Trobats {len(punts_fixos)} punts fixos")
for p in punts_fixos:
    print(f"  E={p[0]:.4f}, I={p[1]:.4f}, resid={np.sqrt(sum(np.array(system_eq(p))**2)):.2e}")

# Classificació
pf_info = []
print(f"PUNTS FIXOS (P = {P})")
print("=" * 55)
for i, pf in enumerate(punts_fixos):
    J    = jacobiana(pf)
    eigv = np.linalg.eigvals(J)
    estable = all(np.real(eigv) < 0)

    if np.real(eigv[0]) * np.real(eigv[1]) < 0:
        tipus = "sella"
    elif any(np.imag(eigv) != 0):
        tipus = "espiral estable" if estable else "espiral inestable"
    else:
        tipus = "node estable" if estable else "node inestable"

    pf_info.append({"E": pf[0], "I": pf[1], "estable": estable, "tipus": tipus})
    print(f"PF{i+1}: E*={pf[0]:.5f}  I*={pf[1]:.5f}  {tipus.upper()}")
    print(f"     lambda = {np.real(eigv[0]):.4f} + {np.imag(eigv[0]):.4f}i,  "
          f"{np.real(eigv[1]):.4f} + {np.imag(eigv[1]):.4f}i")

# ── 7. FIGURA ────────────────────────────────────────────────────
# afegir fletxes
def afegir_fletxa(ax, x, y, pos=0.01):
    n = len(x)

    i = int(pos*(n-1))
    j = min(i+20, n-1)

    

    ax.annotate(
        "",
        xy=(x[j], y[j]),
        xytext=(x[i], y[i]),
        arrowprops=dict(
            arrowstyle="->",
            lw=1.5
        ),
        zorder=20
    )



fig, ax = plt.subplots(figsize=(7, 6))

# Isoclines
ax.plot(I_null_vals[mask_e], E_vals[mask_e], color="darkorange", lw=2.5, zorder = 1)
ax.plot(I_vals[mask_i],      E_null_vals[mask_i], color="steelblue",  lw=2.5, zorder = 1)
ax.grid()
#ax.grid(color = "#e5e5e5"", linestyle="-", linewidth=0.5)

# Punts fixos
colors_pf = {"espiral estable": "limegreen", "node estable": "limegreen",
             "sella": "black", "espiral inestable": "white", "node inestable": "white"}
markers_pf = {"sella": "o", "espiral estable": "o", "node estable": "o",
              "espiral inestable": "o", "node inestable": "o"}


for i, pf in enumerate(pf_info):
    col = colors_pf[pf["tipus"]]
    mk  = markers_pf[pf["tipus"]]
    ax.plot(pf["I"], pf["E"], mk, markersize=12,
            markerfacecolor=col, markeredgecolor="black", markeredgewidth=1.5, zorder = 5)
    ax.text(pf["I"] - 0.07, pf["E"], f"PF{i+1}",
        color="black",
        fontweight="bold",
        fontsize=11,
        zorder=6)
    


# ── 8. ÒRBITES LOCALS PER PF1 i PF3 ─────────────────────────────
times_short = np.linspace(0, 40, 3000)
r = 0.05

for idx, color in [(0, "royalblue"), (2, "darkgreen")]:
    pf = pf_info[idx]
    J  = jacobiana([pf["E"], pf["I"]])

    vecs = np.linalg.eig(J)[1]
    for j in range(2):
        v = np.real(vecs[:, j])
        v = v / np.linalg.norm(v)
        for signe in [1, -1]:
            ci = [pf["E"] + signe*r*v[0], pf["I"] + signe*r*v[1]]
            sol = odeint(WC_ode, ci, times_short)
            ax.plot(sol[:, 1], sol[:, 0], color=color, lw=1.8, zorder=3)
            afegir_fletxa(ax,sol[:,1], sol[:,0])
            ax.plot(ci[1], ci[0], "o", color=color, ms=5, zorder=3)



# ── 9. VARIETATS DE LA SELLA (PF2) ───────────────────────────────
pf2 = pf_info[1]
J2  = jacobiana([pf2["E"], pf2["I"]])
eigv2, vecs2 = np.linalg.eig(J2)

idx_est   = np.argmin(np.real(eigv2))
idx_inest = np.argmax(np.real(eigv2))

v_est   = np.real(vecs2[:, idx_est]);   v_est   /= np.linalg.norm(v_est)
v_inest = np.real(vecs2[:, idx_inest]); v_inest /= np.linalg.norm(v_inest)

eps = 0.005 #0.02
times_long = np.linspace(0, 8, 3000)

print(f"\nPF2 valors propis: {eigv2}")
print(f"v_est:   {v_est}")
print(f"v_inest: {v_inest}")

# Varietat INESTABLE: integrem cap endavant
for signe in [1, -1]:
    ci = [pf2["E"] + signe*eps*v_inest[0],
          pf2["I"] + signe*eps*v_inest[1]]
    sol = odeint(WC_ode, ci, times_long)
    ax.plot(sol[:, 1], sol[:, 0], color="red3" if False else "#cc0000", lw=2, ls="-", zorder=15)
    afegir_fletxa(ax,
              sol[:,1],
              sol[:,0],
              pos = 0.5)
    ax.plot(ci[1], ci[0], "o", color="#cc0000", ms=5, zorder=15)

# Varietat ESTABLE: integrem cap enrere
for k, signe in enumerate ([1, -1]):
    ci = [pf2["E"] + signe*eps*v_est[0],
          pf2["I"] + signe*eps*v_est[1]]
    sol = odeint(WC_back, ci, times_long)


    # Talla si surt del domini
    mask = ((sol[:,1] >= -0.2) & (sol[:,1] <= 0.5) &
            (sol[:,0] >= -0.05) & (sol[:,0] <= 0.5))
    idx_stop = np.where(~mask)[0]
    if len(idx_stop) > 0:
        sol = sol[:idx_stop[0]]


    ax.plot(sol[:, 1], sol[:, 0], color="purple", lw=2, ls="--")
    ax.plot(ci[1], ci[0], "o", color="purple", ms=5)

    n = len(sol)
    if n < 20:
        continue

    # Posició diferent per a cada branca
    frac = 0.7 if k == 0 else 0.9
    i = int(frac * n)
    j = max(i - 10, 0)

    if j >= 0 and i < n:
        ax.annotate("",
            xy    = (sol[j, 1], sol[j, 0]),
            xytext= (sol[i, 1], sol[i, 0]),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="black"),
            zorder=20)

    #afegir_fletxa(ax,
              #sol[::-1, 1],   # invertim per tenir sentit cap a PF2
              #sol[::-1, 0],
              #pos=0.1)



# ── 9.5. TRAJECTÒRIA GLOBAL D'EXEMPLE ────────────────────────────
# Una condició inicial allunyada dels equilibris per veure
# com el sistema evoluciona globalment cap a un dels atractors

cis_globals = [
    {"ci": [0.15, -0.15], "color": "black"},
    {"ci": [0.1, -0.15], "color": "dimgray"},
    {"ci": [0.28, 0.45], "color": "black"},
    {"ci": [0.32, 0.45], "color": "dimgray"},
]

times_global = np.linspace(0, 60, 4000)

for cg in cis_globals:
    ci = cg["ci"]
    sol = odeint(WC_ode, ci, times_global)

    ax.plot(sol[:, 1], sol[:, 0],
            color=cg["color"], lw=1.3, ls="-", alpha=0.8, zorder=8)
    ax.plot(ci[1], ci[0], color=cg["color"], ms=12, zorder=9)

    n_g = len(sol)
    i_g = int(0.05 * n_g)
    j_g = min(i_g + 15, n_g - 1)
    ax.annotate("",
        xy    = (sol[j_g, 1], sol[j_g, 0]),
        xytext= (sol[i_g, 1], sol[i_g, 0]),
        arrowprops=dict(arrowstyle="->", lw=1.5, color=cg["color"]),
        zorder=20)

    
# ── 10. LLEGENDA I EIXOS ─────────────────────────────────────────
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

legend_elements = [
    Line2D([0],[0], color="darkorange",  lw=2.5, label="dE/dt = 0"),
    Line2D([0],[0], color="steelblue",   lw=2.5, label="dI/dt = 0"),
    Line2D([0],[0], color="royalblue",   lw=1.8, label="Trajectòries PF1"),
    Line2D([0],[0], color="darkgreen",   lw=1.8, label="Trajectòries PF3"),
    Line2D([0],[0], color="#cc0000",     lw=2,   label="Var. inestable PF2"),
    Line2D([0],[0], color="purple",      lw=2, ls="--", label="Var. estable PF2 (separatriu)"),
]

ax.legend(handles=legend_elements, loc="upper left",
          frameon=False, fontsize=9)

ax.set_xlim(-0.2, 0.5)
ax.set_ylim(-0.05, 0.5)
ax.set_xlabel("I  (activitat inhibidora)", fontsize=12)
ax.set_ylabel("E  (activitat excitadora)", fontsize=12)
#ax.set_title(f"Retrat de fase – Model de Wilson-Cowan  (P = {P})", fontsize=12)

plt.tight_layout()
plt.savefig("retrat_fase_WC.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 11. DIAGNÒSTIC FINAL ─────────────────────────────────────────
print("\nVarietat inestable (+):")
ci = [pf2["E"] + eps*v_inest[0], pf2["I"] + eps*v_inest[1]]
sol = odeint(WC_ode, ci, times_long)
print(f"  Final: E={sol[-1,0]:.4f}, I={sol[-1,1]:.4f}")

print("Varietat inestable (-):")
ci = [pf2["E"] - eps*v_inest[0], pf2["I"] - eps*v_inest[1]]
sol = odeint(WC_ode, ci, times_long)
print(f"  Final: E={sol[-1,0]:.4f}, I={sol[-1,1]:.4f}")