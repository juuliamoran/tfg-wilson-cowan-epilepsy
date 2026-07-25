
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Parametres
a_e = 1.2
a_i = 1
theta_e  = 2.8
theta_i = 4
w_ee = 12
w_ei = 4
w_ie = 13
w_ii = 11
Q = 0

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

def F(z):
    E, I, P = z
    xe = w_ee*E - w_ei*I + P
    xi = w_ie*E - w_ii*I + Q
    return np.array([
        -E + (1 - E) * S_e(xe),
        -I + (1 - I) * S_i(xi)
    ])

def J_EI(z):
    E, I, P = z
    xe = w_ee*E - w_ei*I + P
    xi = w_ie*E - w_ii*I + Q
    Se = S_e(xe)
    Si = S_i(xi)
    dSe = dS_e(xe)
    dSi = dS_i(xi)
    return np.array([
        [-1 - Se + (1-E)*dSe*w_ee,   -(1-E)*dSe*w_ei          ],
        [ (1-I)*dSi*w_ie,            -1 - Si - (1-I)*dSi*w_ii ]
    ])


def dF_dP(z):
    E, I, P = z
    xe = w_ee*E - w_ei*I + P
    return np.array([(1 - E) * dS_e(xe), 0.0])

def stability(branch):
    stabs = []
    for z in branch:
        evs = np.linalg.eigvals(J_EI(z))
        re  = np.real(evs)
        if np.all(re < 0):
            stabs.append("stable")
        else:
            stabs.append("unstable")  # sella o focus repulsor, tots inestables
    return stabs



def tangent(z, t_prev=None):
    A = np.column_stack([J_EI(z), dF_dP(z)])
    _, _, vt = np.linalg.svd(A)
    t = vt[-1]
    t = t / np.linalg.norm(t)
    if t_prev is not None and np.dot(t, t_prev) < 0:
        t = -t
    return t


def corrector(z_pred, t_pred, tol=1e-10, maxit=30):
    z = z_pred.copy()
    for _ in range(maxit):
        Fval = F(z)
        G = np.array([Fval[0], Fval[1], np.dot(z - z_pred, t_pred)])
        if np.linalg.norm(G) < tol:
            return z, True
        E, I, P = z
        xe = w_ee*E - w_ei*I + P
        xi = w_ie*E - w_ii*I + Q
        dSe = dS_e(xe)
        dSi = dS_i(xi)
        JF = np.array([
            [-1 - S_e(xe) + (1-E)*dSe*w_ee,  -(1-E)*dSe*w_ei,           (1-E)*dSe],
            [ (1-I)*dSi*w_ie,                 -1 - S_i(xi) - (1-I)*dSi*w_ii, 0.0  ]
        ])
        JG = np.vstack([JF, t_pred])
        try:
            delta = np.linalg.solve(JG, -G)
        except np.linalg.LinAlgError:
            return z, False
        z += delta
        if np.linalg.norm(delta) < tol:
            return z, True
    return z, False
    

def continuation(z_start, n_steps=300, h=0.02):
    """
    BUCLE PRINCIPAL
    Cada iteració:
        calcula tangent,
        predictor,
        corrector,
        desa nou punt.

    Així anem recorrent la branca
    """
    pts    = [np.array(z_start, dtype=float)]
    t_prev = None
    for _ in range(n_steps):
        z0     = pts[-1]
        t      = tangent(z0, t_prev)
        z_pred = z0 + h*t
        z_new, ok = corrector(z_pred, t)
        if not ok:
            break
        pts.append(z_new)
        t_prev = t
    return np.array(pts)


def find_fixed_points(P_val):
    def F_fixed(EI):
        E, I = EI
        xe = w_ee*E - w_ei*I + P_val
        xi = w_ie*E - w_ii*I + Q
        return [
            -E + (1-E)*S_e(xe),
            -I + (1-I)*S_i(xi)
        ]
    punts = []
    for E0 in np.linspace(0.01, 0.99, 10):
        for I0 in np.linspace(0.01, 0.99, 10):
            sol = fsolve(F_fixed, [E0, I0], full_output=True)
            if sol[2] == 1:
                pf = np.round(sol[0], 6)
                if not any(np.allclose(pf, p, atol=1e-4) for p in punts):
                    punts.append(pf)
    return punts

def detect_folds(branch, tol=1e-3):
    P = branch[:,2]
    E = branch[:,0]

    dEdP = np.gradient(E, P)

    folds = []

    for i in range(1, len(dEdP)-1):
        if dEdP[i-1] * dEdP[i] < 0:
            folds.append((P[i], E[i]))

    return folds


# Punts fixos inicials
print("\n=== Punts fixos per a P=0 ===")
pf_list = find_fixed_points(0.0)
for pf in pf_list:
    print(f"  E={pf[0]:.4f}, I={pf[1]:.4f}")

col_map = {"stable": "#2166ac", "unstable": "#d73027"}
lty_map = {"stable": "-",       "unstable": "--"}

z0 = [pf_list[1][0], pf_list[1][1], 0.0] # des del punt mig

b_fwd = continuation(z0, n_steps=500, h=0.005)
b_bwd = continuation(z0, n_steps=500, h=-0.005)

branch = np.vstack([b_bwd[::-1], b_fwd[1:]])

plt.figure(figsize=(8,5))

stabs = stability(branch)

for k in range(len(branch)-1):
    s = stabs[k]
    plt.plot(
        branch[k:k+2,2],
        branch[k:k+2,0],
        color=col_map[s],
        linestyle=lty_map[s],
        lw=2
    )

folds = detect_folds(branch)

folds = sorted(folds, key=lambda x: x[0])  # ordena per P

for i, (p, e) in enumerate(folds[:2], start=1):
    plt.plot(p, e, 'ko', ms=8)
    plt.text(p, e, rf"$P_{i}^*={p:.3f}$",
             fontsize=11, ha='left', va='bottom')
    


P1, E1 = folds[0]
P2, E2 = folds[1]

# salt cap amunt
plt.annotate(
    "",
    xy=(P2, 0.47),
    xytext=(P2, 0.05),
    arrowprops=dict(arrowstyle="->", lw=2, color="black")
)

# salt cap avall
plt.annotate(
    "",
    xy=(P1, -0.02),
    xytext=(P1, 0.38),
    arrowprops=dict(arrowstyle="->", lw=2, color="black")
)

plt.grid(True, alpha=0.3)

legend_elements = [
    Line2D([0],[0], color="#2166ac", lw=2, ls="-",  label="Estable"),
    Line2D([0],[0], color="#d73027", lw=2, ls="-", label="Inestable")
]

plt.xlim(-1, 1)
plt.legend(handles=legend_elements, frameon=False)
plt.xlabel("$P$ (input extern)", fontsize=12)
plt.ylabel("$E^*$", fontsize=12)
plt.show()


"""
Estudiem la resposta dinàmica del sistema excitació–inhibició quan el 
paràmetre P varia lentament en el temps. En lloc de calcular directament les branques 
d’equilibris amb continuació numèrica, integrem les equacions diferencials i permetem 
que el sistema relaxi cap a un estat estacionari per a cada valor de P.


El sistema no “salta instantàniament” entre equilibris matemàtics, sinó que:
evoluciona segons la seva dinàmica i queda atrapat en l’atractor on ja es trobava.
Això fa que el resultat depengui de la història del sistema.


Quan augmentem i després disminuïm P, observem que:
- el sistema no segueix el mateix camí en pujada i baixada
- apareix una diferència entre les trajectòries
- hi ha transicions abruptes entre estats
Això es coneix com a histèresi dinàmica.



Simulem:
- fixem P
- deixem evolucionar el sistema fins a equilibri (o atractor)
- usem aquest estat com a “memòria”
- canviem P -> P+ΔP
repeteim

Això és:
“adiabatic parameter sweep amb histèresi”
"""

def run_to_steady(E0, I0, P, T=200, dt=0.05):
    E, I = E0, I0

    for _ in range(int(T/dt)):
        xe = w_ee*E - w_ei*I + P
        xi = w_ie*E - w_ii*I + Q

        dE = -E + (1 - E)*S_e(xe)
        dI = -I + (1 - I)*S_i(xi)

        E += dt * dE
        I += dt * dI

    return E, I

def sweep_P(P_vals, E0, I0):
    E, I = E0, I0
    path = []

    for P in P_vals:
        E, I = run_to_steady(E, I, P)
        path.append((P, E, I))

    return np.array(path)

def sweep_P_backward(P_vals, E0, I0):
    E, I = E0, I0
    path = []

    for P in P_vals[::-1]:
        E, I = run_to_steady(E, I, P)
        path.append((P, E, I))

    return np.array(path)

P_vals = np.linspace(-0.6, 0.8, 80)

up = sweep_P(P_vals, 0.1, 0.1)
down = sweep_P_backward(P_vals, up[-1,1], up[-1,2])

plt.figure(figsize=(8,5))

plt.plot(up[:,0], up[:,1], 'b', label="P pujant")
plt.plot(down[:,0], down[:,1], 'r', label="P baixant")

plt.xlabel("P")
plt.ylabel("E*")
plt.title("Histèresi via dinàmica temporal (sweep de P)")
plt.legend()
plt.grid()
plt.show()


## SÈRIE TEMPORAL
import numpy as np
import matplotlib.pyplot as plt

# Paràmetres
a_e = 1.2; a_i = 1.0
theta_e = 2.8; theta_i = 4.0
w_ee = 12.0; w_ei = 4.0
w_ie = 13.0; w_ii = 11.0
Q = 0.0

def S_e(x):
    return 1/(1+np.exp(-a_e*(x-theta_e))) - 1/(1+np.exp(a_e*theta_e))
def S_i(x):
    return 1/(1+np.exp(-a_i*(x-theta_i))) - 1/(1+np.exp(a_i*theta_i))

# Simulació amb P variable en el temps
dt = 0.05
T  = 4000
n  = int(T/dt)

t_vals = np.arange(n) * dt
P_up   = np.linspace(-0.6, 0.8, n//2)
P_down = np.linspace(0.8, -0.6, n//2)
P_vals = np.concatenate([P_up, P_down])

E, I = 0.01, 0.01
E_traj = np.zeros(n)

for k in range(n):
    P  = P_vals[k]
    xe = w_ee*E - w_ei*I + P
    xi = w_ie*E - w_ii*I + Q
    E += dt * (-E + (1-E)*S_e(xe))
    I += dt * (-I + (1-I)*S_i(xi))
    E_traj[k] = E


jump_up_idx = np.argmax(np.diff(E_traj))
jump_down_idx = np.argmin(np.diff(E_traj))

t_up = t_vals[jump_up_idx]
t_down = t_vals[jump_down_idx]



# Gràfic
fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(t_vals, E_traj, color="#2166ac", lw=1.5)
ax.axvline(T/2, color="black", lw=0.8, ls=":",
           label="Inici baixada de $P$")
ax.axvline(t_up, color="red", ls="--", lw=1.2,
           label=r"$P=P_2^*$")

ax.axvline(t_down, color="darkgreen", ls="--", lw=1.2,
           label=r"$P=P_1^*$")
ax.set_ylabel("$E(t)$", fontsize=11)
ax.set_xlabel("Temps (ms)", fontsize=11)
#ax.set_title("Histèresi dinàmica – Model de Wilson-Cowan", fontsize=12)
ax.legend(frameon=False, fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("histeresi_temporal.png", dpi=150, bbox_inches="tight")
plt.show()