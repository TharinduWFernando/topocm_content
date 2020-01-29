```python
import sys
sys.path.append('../code')
from init_mooc_nb import *
init_notebook()
from holoviews.core.options import Cycle
%output size=120
pi_ticks = [(-np.pi, r'$-\pi$'), (0, '0'), (np.pi, r'$\pi$')]


def ts_modulated_wire(L=50):
    """Create an infinite wire with a periodic potential

    Chain lattice, one orbital per site.
    Returns kwant system.

    Arguments required in onsite/hoppings: 
        t, mu, mu_lead, A, phase

    The period of the potential is 2*pi/L.
    """
    omega = 2 * np.pi / L

    def onsite(site, p):
        x = site.pos[0]
        return 2 * p.t - p.mu + p.A * (np.cos(omega * x + p.phase) + 1)

    def hopping(site1, site2, p):
        return -p.t

    sym_lead = kwant.TranslationalSymmetry([-L])

    lat = kwant.lattice.chain()
    syst = kwant.Builder(sym_lead)

    syst[(lat(x) for x in range(L))] = onsite
    syst[lat.neighbors()] = hopping

    return syst


def modulated_wire(L=50, dL=10):
    """Create a pump. 

    Chain lattice, one orbital per site.
    Returns kwant system.

    L is the length of the pump,
    dL is the length of the clean regions next to the pump,
            useful for demonstration purposes.

    Arguments required in onsite/hoppings: 
        t, mu, mu_lead, A, omega, phase
    """
    def onsite(site, p):
        x = site.pos[0]
        return 2 * p.t - p.mu + p.A * (np.cos(p.omega * x + p.phase) + 1)

    lead_onsite = lambda site, p: 2 * p.t - p.mu_lead

    def hopping(site1, site2, p):
        return -p.t

    lat = kwant.lattice.chain()
    syst = kwant.Builder()

    syst[(lat(x) for x in range(L))] = onsite
    syst[lat.neighbors()] = hopping

    sym_lead = kwant.TranslationalSymmetry([-1])
    lead = kwant.Builder(sym_lead)
    lead[lat(0)] = lead_onsite
    lead[lat.neighbors()] = hopping

    syst.attach_lead(lead, add_cells=dL)
    syst.attach_lead(lead.reversed(), add_cells=dL)

    return syst


def total_charge(value_array):
    """Calculate the pumped charge from the list of reflection matrices."""
    determinants = [np.linalg.det(r) for r in value_array]
    charge = np.cumsum(np.angle(np.roll(determinants, -1) / determinants))
    charge = charge - charge[0]
    return charge / (2 * np.pi)

```

# Thouless pumps

Dganit Meidan from Ben Gurion University will introduce Thouless pumps,.


```python
MoocVideo("gKZK9IGY9wo", src_location='3.1-intro', res='360')
```

# Hamiltonians with parameters

Previously, when studying the topology of systems supporting Majoranas (both the Kitaev chain and the nanowire), we were able to calculate topological properties by studying the bulk Hamiltonian $H(k)$.

There are two points of view on this Hamiltonian. We could either consider it a Hamiltonian of an infinite system with momentum conservation

$$H = H(k) |k\rangle\langle k|,$$

or we could equivalently study a finite system with only a small number of degrees of freedom (corresponding to a single unit cell), and a Hamiltonian which depends on some continuous periodic parameter $k$.

Of course, without specifying that $k$ is the real space momentum, there is no meaning in bulk-edge correspondence (since the edge is an edge in real space), but the topological properties are still well-defined.

Sometimes we want to know how a physical system changes if we slowly vary some parameters of the system, for example a bias voltage or a magnetic field. Because the parameters change with time, the Hamiltonian becomes time-dependent, namely

$$H = H(t).$$

The slow [adiabatic](https://en.wikipedia.org/wiki/Adiabatic_theorem) change of parameters ensures that if the system was initially in the ground state, it will stay in the ground state, so that the topological properties are useful.

A further requirement for topology to be useful is the *periodicity* of time evolution:

$$H(t) = H(t+T).$$

The period can even go to $\infty$, in which case $H(-\infty) = H(+\infty)$. The reasons for the requirement of periodicity are somewhat abstract. If the Hamiltonian has parameters, we're studying the topology of a *mapping* from the space of parameter values to the space of all possible gapped Hamiltonians. This mapping has nontrivial topological properties only if the space of parameter values is compact.

For us, this simply means that the Hamiltonian has to be periodic in time.

Of course, if we want systems with bulk-edge correspondence, then in addition to $t$ our Hamiltonian must still depend on the real space coordinate, or the momentum $k$.

# Quantum pumps

In the image below (source: Chambers's Encyclopedia, 1875, via Wikipedia) you see a very simple periodic time-dependent system, an Archimedes screw pump.

![](figures/Archimedes_screw.jpg)

The changes to the system are clearly periodic, and the pump works the same no matter how slowly we use it (that is, change the parameters), so it is an adiabatic tool.

What about a quantum analog of this pump? 

Let's take a one-dimensional region, coupled to two electrodes on both sides, and apply a strong sine-shaped confining potential in this region. As we move the confining potential, we drag the electrons captured in it.

So our system now looks like this:


```python
# Plot of the potential in the pumping system as a function of coordinate.
# Some part of the leads is shown with a constant potential.
# Regions with E < 0 should be shaded to emulate Fermi sea.
A = 0.6
L = 10
lamb = (10 / 5.3) / (2 * np.pi)
mu = -0.4
mu_lead = -0.8


def f(x):
    if x < 0.0:
        return mu_lead
    if x >= 0.0 and x <= L:
        return mu + A * (1.0 - np.cos(x / lamb))
    if x > L:
        return mu_lead

x = np.linspace(-5, 15, 1000)
y = [f(i) for i in x]

plt.figure(figsize=(6, 4))
plt.plot(x, y, 'k', lw=1.2)

plt.xlim(-2.5, 12.5)
plt.ylim(-2, 2)

y = [i if i <= 0 else 0 for i in y]
plt.fill_between(x, y, 0, color='r', where=np.array(y) <
                 0.0, alpha=0.5, edgecolor='k', lw=1.5)

plt.arrow(2.0, 1.25, 5, 0, head_width=0.15, head_length=1.0, fc='k', ec='k')

plt.xlabel('$x$')
plt.ylabel('$U(x)$')
plt.xticks([])
plt.yticks([])
plt.show()
```

It is described by the Hamiltonian

$$H(t) = \frac{k^2}{2m} + A [1 - \cos(x/\lambda + 2\pi t/T)].$$

As we discussed, if we change $t$ very slowly, the solution will not depend on how fast $t$ varies.

When $A \gg 1 /m \lambda^2$ the confining potential is strong, and additionally if the chemical potential $\mu \ll A$, the states bound in the separate minima of the potential have very small overlap.

The potential near the bottom of each minimum is approximately quadratic, so the Hamiltonian is that of a simple Harmonic oscillator. This gives us discrete levels of the electrons with energies $E_n = (n + \tfrac{1}{2})\omega_c$, with $\omega_c = \sqrt{A/m\lambda^2}$ the oscillator frequency. In the large A limit, the states in the different minima are completely isolated so that the energy bands are flat with vanishing (group) velocity $d E_n(k)/d k=0$ of propagation.

We can numerically check how continuous bands in the wire become discrete evenly spaced bands as we increase $A$:


```python
p = SimpleNamespace(t=1, mu=0.0, phase=0.0, A=None)
syst = ts_modulated_wire(L=17)

def title(p):
    return "Band structure, $A={:.2}$".format(p.A)

kwargs = {'ylims': [-0.2, 1.3],
          'xticks': pi_ticks,
          'yticks': [0, 0.5, 1.0],
          'xdim': r'$k$',
          'ydim': r'$E$',
          'k_x': np.linspace(-np.pi, np.pi, 101),
          'title': title}


holoviews.HoloMap({p.A: spectrum(syst, p, **kwargs) for p.A in np.linspace(0, 0.8, 10)}, kdims=[r'$A$'])
```

So unless $\mu = E_n$ for some $n$, each minimum of the potential contains an integer number of electrons $N$.There are a large number of states at this energy and almost no states at $\mu$ away from $E_n$.

Since electrons do not move between neighboring potential minima, so when we change the potential by one time period, we move exactly $N$ electrons.


```python
question = "Why are some levels in the band structure flat while some are not?"
answers = ["The flat levels are the ones whose energies are not sensitive to the offset of confining potential.",
           "Destructive interference of the wave functions in neighboring minima suppresses the dispersion.",
           "The flat levels are localized deep in the potential minima, "
           "so the bandwidth is exponentially small.",
           "The flat levels correspond to filled states, and the rest to empty states."]
explanation = ("The dispersion of the bands in a perodic potential appears "
               "when the wave functions from neighboring minima overlap.")

MoocMultipleChoiceAssessment(question=question, answers=answers, correct_answer=2, explanation=explanation)
```

# Quantization of pumped charge

As we already learned, integers are important, and they could indicate that something topological is happening.

At this point we should ask ourselves these questions: Is the discreteness of the number of electrons $N$ pumped per cycle limited to the deep potential limit, or is the discreteness a more general consequence of topology? 

### Thought experiment

Let us consider the reservoirs to be closed finite (but large) boxes. When the potential in the wire is shifted the electrons clearly move from the left to the right reservoir. How do the reservoirs accomodate these electrons?

Since the Hamiltonian is periodic in time, the Hamiltonian together with all its eigenstates return to the initial values at the end of the period. The adiabatic theorem guarantees that when the Hamiltonian changes slowly the eigenstates can evolve to an eigenstate that is adjacent in energy. 

As we see in the figure below states in the left reservoir move down in energy and states in the right reservoir move up: 


```python
p = SimpleNamespace(t=1, mu=0.0, mu_lead=0, A=0.1, omega=0.3, phase=None)
syst = modulated_wire(L=110, dL=40).finalized()
phases = np.linspace(0, 2*np.pi, 251)
en = [np.linalg.eigvals(syst.hamiltonian_submatrix(args=[p])) for p.phase in phases]
en = np.array(en)
coord = kwant.operator.Density(syst, onsite=(lambda site, p: site.pos[0]))
ticks = {'xticks': [0, 1], 'yticks': [0, 0.5, 1]}
kdims = [r'$t/T$', r'$E$']
holoviews.Path((phases / (2*np.pi), en), kdims=kdims)[:, 0:.4](plot=ticks)
```

While dynamics in the relatively flat dense mess of states (from the wire) is complicated, the situation in the energy gaps (i.e. ranges with few states) is clearer. States in the gap belong to either one reservoir or the other. States in the left reservoir turn out to move down in energy and ones in the right reservoir move up in energy (right now this is numerical - we will see why later ). 

Let us imagine we park our Fermi energy (i.e. the energy that separates completely occupied and completely empty states) in the gap where there are few states.

With time, an empty level in the left reservoir moves from above the Fermi level to a state below the Fermi level. The occupation of this state cannot  change in this process because of the adiabatic theorem. Therefore after the pumping cycle is over the left reservoir has an empty state below the Fermi level i.e. one less electron. The reverse process happens in the right lead so that it has one extra electron.

So the electron that was transferred in the wire from the left reservoir to the right came from emptying the highest energy occupied state one the left and occupying the lowest energy unoccupied state on the right.

More generally, while levels do not have to move as shown in the figure, the energy level structure of the periodic in time Hamiltonian has to return to itself. The number (possibly negative) of levels in the left reservoir that crossed from being above the Fermi level to below  must exactly be the change in charge of the left reservoir. This is an INTEGER. 

Therefore, our thoughts about "where the electrons came from" leads us to an interesting conclusion: The number of charges pumped in an adiabatic pumping cycle (independent of the strength $A$) is an integer (possibly $0$).

Furthermore, it is an integer as long as the wire is gapped at the chosen Fermi level ( so as to isolate the reservoirs).


So without doing any calculations, we can conclude that:

> The number of electrons pumped per cycle of a quantum pump is an integer as long as
> the bulk of the pump is gapped. Therefore it is a **topological invariant**.

# Counting electrons through reflection.

The expression for the pumped charge in terms of the bulk Hamiltonian $H(k, t)$ is complicated.

It's an integral over both $k$ and $t$, called a **Chern number** or in other sources a TKNN integer. Its complexity is beyond the scope of our course, but is extremely important, so we will have to study it... next week.

There is a much simpler way to calculate the same quantity using scattering formalism. This follows from returning to understanding how energy levels in the reservoir move as a function of time.

Consider levels in the energy gap of the wire so that the levels near the Fermi energy are confined to the reservoir. Now all the levels in the reservoir are quantized, and are standing waves, so they are equal weight superpositions of waves going to the left $\psi_L$ and to the right $\psi_R$,

$$
\psi_n = \psi_L(x) + \psi_R(x) \propto \exp(ik_n x) + \exp(-ik_n x + i\phi),
$$

where the wave number $k_n$ is of course a function of energy. The relative phase shift $\phi$ is necessary to satisfy the boundary condition at $x=0$, where $\psi_L = r \psi_R$, and so $\exp(i \phi) = r$. The energies of the levels are determined by requiring that the phases of $\psi_L$ and $\psi_R$ also match at $x = -L$. These boundary conditions lead to the relation 

$$
2 k_n L = 2n \pi +\phi.
$$

Changing the phase continuously in time $t$ by $2\pi$ evolves $k_n$ from $(2 L)^{-1}[2n\pi+\phi]$ to $(2L)^{-1}[2n\pi+2\pi+\phi]=2k_{n+1}L$. This means that the change in phase $\phi$ results in the state with index $n$ evolving 
into an index  $n+1$.
Thus, the winding of phase $\phi$ by $2\pi$ also changes the wave-function $\psi_n \rightarrow \psi_{n+1}$.

As discussed in the previous paragraph, such a movement of energy level in the reservoir is associated with transfer of charge between the reservoir through the wire.

> We conclude that if the reflection phase $\phi$ from the wire advances by $2\pi$, this corresponds to a unit charge pumped across the wire.

It's very easy to generalize our argument to many modes. For that we just need to sum all of the reflection phase shifts, which means we need to look at the phase of $\det r$.

We conclude that there's a very compact relation between charge $dq$ pumped by an infinitesimal change of an external parameter and the change in reflection matrix $dr$:

$$
dq = \frac{d \log \det r}{2\pi i} = \operatorname{Tr}\frac{r^\dagger dr }{ 2 \pi i}.
$$

While we derived this relation only for the case when all incoming particles reflect, and $r$ is unitary, written in form of trace it also has physical implications even if there is transmission.[¹](https://arxiv.org/abs/cond-mat/9808347)

Let's check if this expression holds to our expectations. If $||r||=1$, this is just the number of times the phase of $\det r$ winds around zero, and it is certainly an integer, as we expected.

This provides an independent argument (in addition to the movement of energy levels from the last section ) for why the pumped charge is quantized as long as the gap is preserved.

# Applying the topological invariant

We know now how to calculate the pumped charge during one cycle, so let's just see how it works in practice.

The scattering problem in 1D can be solved quickly, so let's calculate the pumped charge as a function of time for different values of the chemical potential in the pump.


```python
%%opts Path.Q (color=Cycle(values=['r', 'g', 'b', 'y']))
%%opts HLine (color=Cycle(values=['r', 'g', 'b', 'y']) linestyle='--')

def plot_charge(mu):
    energy = 0.0
    phases = np.linspace(0, 2*np.pi, 100)
    p = SimpleNamespace(t=1, mu=mu, mu_lead=mu, A=0.6, omega= .3)
    syst = modulated_wire(L=100).finalized()
    rs = [kwant.smatrix(syst, energy, args=[p]).submatrix(0, 0) for p.phase in phases]
    wn = -total_charge(rs)
    title = '$\mu={:.2}$'.format(mu)
    kdims = [r'$t/T$', r'$q/e$']
    plot = holoviews.Path((phases / (2 * np.pi), wn), kdims=kdims, label=title, group='Q')
    return plot[:, -0.5:3.5](plot={'xticks': [0, 1], 'yticks': [0, 1, 2, 3]})


kwargs = {'ylims': [-0.2, 1.3],
          'xticks': pi_ticks,
          'yticks': [0, 0.5, 1.0],
          'xdim': r'$k$',
          'ydim': r'$E$',
          'k_x': np.linspace(-np.pi, np.pi, 101),
          'title': lambda p: "Band structure, $A={:.2}$".format(p.A)}

p = SimpleNamespace(t=1, mu=0.0, phase=0.0, A=0.6)
syst = ts_modulated_wire(L=17)
mus = [0.1, 0.3, 0.6, 0.9]
HLines = holoviews.Overlay([holoviews.HLine(mu) for mu in mus])
spectrum(syst, p, **kwargs) * HLines + holoviews.Overlay([plot_charge(mu) for mu in mus]).relabel('Pumped charge')
```

In the left plot, we show the band structure, where the different colors correspond to different chemical potentials. The right plot shows the corresponding pumped charge. During the pumping cycle the charge may change, and the relation between the offset $\phi$ of the potential isn't always linear. However we see that after a full cycle, the pumped charge exactly matches the number of filled levels in a single potential well.


```python
question = ("What happens to the dependence of the reflection phase shift on time if we "
            "remove one of the reservoirs and leave the other one?")
answers = ["It becomes constant.",
           "For most of the cycle it stays the same, but there appear "
           "sharp jumps such that the total winding becomes zero.",
           "Nothing changes, since the two ends of the pump are "
           "far apart from each other, and the pump is not conducting.",
           "The reflection phase gets a new time dependence with zero winding, unrelated to the original one."]
explanation = ("The total pumped charge must become equal to zero since there's nowhere to place the charge, but "
               "since the pump is insulating, the phase cannot change "
               "for most of the cycle unless a sharp resonance appears")

MoocMultipleChoiceAssessment(question=question, answers=answers, correct_answer=1, explanation=explanation)
```

# Quantized charge and scattering invariant


```python
MoocVideo("6lXRAZ7hv7E", src_location='3.1-summary', res='360')
```

**Questions about what you learned? Ask them below**


```python
MoocDiscussion('Questions', 'Quantum pumps')
```
