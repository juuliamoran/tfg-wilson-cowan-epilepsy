import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# paràmetres del cicle límit
a_e=1.3; a_i=2.0; theta_e=4.0; theta_i=3.7
w_ee=16; w_ei=12.0; w_ie=15.0; w_ii=3.0; Q=0.0

def S_e(x): return 1/(1+np.exp(-a_e*(x-theta_e))) - 1/(1+np.exp(a_e*theta_e))
def S_i(x): return 1/(1+np.exp(-a_i*(x-theta_i))) - 1/(1+np.exp(a_i*theta_i))
def dS_e(x):
    sig = 1/(1+np.exp(-a_e*(x-theta_e))); return a_e*sig*(1-sig)
def dS_i(x):
    sig = 1/(1+np.exp(-a_i*(x-theta_i))); return a_i*sig*(1-sig)

def jacobiana(E, I, P):
    xe = w_ee*E - w_ei*I + P
    xi = w_ie*E - w_ii*I + Q
    Se=S_e(xe); Si=S_i(xi); dSe=dS_e(xe); dSi=dS_i(xi)
    return np.array([
        [-1-Se+(1-E)*dSe*w_ee,  -(1-E)*dSe*w_ei],
        [ (1-I)*dSi*w_ie,       -1-Si-(1-I)*dSi*w_ii]
    ])

P_vals = np.linspace(0.5, 2.0, 300)
re_max = []
P_hopf = None

for Pv in P_vals:
    def sys(y):
        E,I = y
        return [-E+(1-E)*S_e(w_ee*E-w_ei*I+Pv),
                -I+(1-I)*S_i(w_ie*E-w_ii*I+Q)]
    # Busca el punt fix (en aquest règim n'hi ha un sol)
    pf = None
    for E0,I0 in [(0.1,0.1),(0.2,0.15),(0.15,0.1),(0.3,0.2)]:
        sol = fsolve(sys, [E0,I0], full_output=True)
        if sol[2]==1 and np.sqrt(sum(np.array(sys(sol[0]))**2))<1e-8:
            pf = sol[0]; break
    if pf is None:
        re_max.append(np.nan); continue
    evs = np.linalg.eigvals(jacobiana(pf[0], pf[1], Pv))
    re_max.append(max(np.real(evs)))

re_max = np.array(re_max)

# Troba P*
idx = np.where(np.diff(np.sign(re_max)))[0]
if len(idx) > 0:
    P_hopf = P_vals[idx[0]]
    print(f"Bifurcació de Hopf: P* ≈ {P_hopf:.3f}")

# Gràfic
fig, ax = plt.subplots(figsize=(7,4))
ax.plot(P_vals, re_max, color="#2166ac", lw=2.5)
ax.axhline(0, color="black", lw=1, ls="--")
if P_hopf:
    ax.axvline(P_hopf, color="#d73027", lw=1.5, ls="--",
               label=f"$P^* \\approx {P_hopf:.3f}$")
ax.fill_between(P_vals, re_max, 0,
                where=(re_max < 0), alpha=0.15, color="#2166ac",
                label="Focus estable")
ax.fill_between(P_vals, re_max, 0,
                where=(re_max > 0), alpha=0.15, color="#d73027",
                label="Focus inestable (cicle límit)")
ax.set_xlabel("$P$ (input extern)", fontsize=12)
ax.set_ylabel(r"$\max\,\mathrm{Re}(\lambda)$", fontsize=12)
#ax.set_title("Bifurcació de Hopf", fontsize=12)
ax.legend(frameon=False, fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("hopf_remax.png", dpi=150)
plt.show()



from scipy.integrate import odeint

P_cycle = np.linspace(0.80, 1.8, 60)
amplituds = []
times = np.linspace(0, 500, 10000)

for Pv in P_cycle:
    def ode(y, t):
        E,I = y
        return [-E+(1-E)*S_e(w_ee*E-w_ei*I+Pv),
                -I+(1-I)*S_i(w_ie*E-w_ii*I+Q)]
    sol = odeint(ode, [0.1, 0.1], times)
    # Agafa l'últim 20% (cicle convergit)
    E_conv = sol[int(0.8*len(times)):, 0]
    amplituds.append((max(E_conv) - min(E_conv)) / 2)

fig, ax = plt.subplots(figsize=(7,4))
ax.plot(P_cycle, amplituds, "o-", color="#7B2D8B", lw=2, ms=4)
if P_hopf:
    ax.axvline(P_hopf, color="#d73027", lw=1.5, ls="--",
               label=f"$P^* \\approx {P_hopf:.3f}$")
ax.set_xlabel("$P$ (input extern)", fontsize=12)
ax.set_ylabel("Amplitud de $E(t)$", fontsize=12)
#ax.set_title("Amplitud del cicle límit – Bifurcació de Hopf", fontsize=12)
ax.legend(frameon=False, fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("hopf_amplitud.png", dpi=150)
plt.show()



P_plot = 1.25   # després de la Hopf
P_plot = P_hopf-0.1

def ode(y, t):
    E, I = y
    return [
        -E + (1-E)*S_e(w_ee*E - w_ei*I + P_plot),
        -I + (1-I)*S_i(w_ie*E - w_ii*I + Q)
    ]

times = np.linspace(0, 50, 6000)

sol = odeint(ode, [0.1, 0.1], times)

E = sol[:,0]
I = sol[:,1]

plt.figure(figsize=(8,4))
plt.plot(times, E, lw=2, color="#7B2D8B")
plt.xlabel("Temps")
plt.ylabel("$E(t)$")
plt.title(f"Oscil·lacions de l'activitat excitadora ($P={P_plot}$)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ELS DOS EN UN

fig, axes = plt.subplots(2, 1, sharex=True)

escenaris = [
    {"P": P_hopf - 0.1, "label": f"Abans de la bifurcació  ($P = {P_hopf-0.1:.2f} < P^*$)",
     "col": "#2166ac"},
    {"P": 1.25,          "label": f"Després de la bifurcació  ($P = 1.25 > P^*$)",
     "col": "#7B2D8B"},
]

times = np.linspace(0, 150, 10000)

for ax, sc in zip(axes, escenaris):
    Pv = sc["P"]
    def ode(y, t):
        E, I = y
        return [-E + (1-E)*S_e(w_ee*E - w_ei*I + Pv),
                -I + (1-I)*S_i(w_ie*E - w_ii*I + Q)]
    
    sol = odeint(ode, [0.1, 0.1], times)
    
    ax.plot(times, sol[:, 0], lw=2, color=sc["col"])
    ax.set_ylabel("$E(t)$", fontsize=11)
    ax.set_title(sc["label"], fontsize=11)
    ax.grid(alpha=0.3)
    
    if sc["P"] < P_hopf:
        ax.axhline(sol[-1, 0], color="grey", lw=0.8, ls="--",
                   label="Equilibri")
        ax.legend(frameon=False, fontsize=9)


axes[-1].set_xlabel("Temps (ms)", fontsize=11)
plt.suptitle("Transició per bifurcació de Hopf – Model de Wilson-Cowan",
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("hopf_timeseries.png", dpi=150, bbox_inches="tight")
plt.show()