Import Simulator from library:


```python
from python_kinetic_simulator.simulator import Simulator
```

Set up a simulation and run it:


```python
# initialize simulator and its temperature (in °C)
sim = Simulator(T_C=50)

# add species, concentrations (M) and relative energies (kcal/mol).
# Default values for energy and concentration are 0.
sim.add_species("A", conc=1)
sim.add_species("B", energy=-2)
sim.add_species("B(adsorbed)", energy=-2.5)
sim.add_species("C", energy=-30)

# add reactions by specifying their activation
# energy (in kcal/mol)
sim.add_reaction(["A"], ["B", "B"], ts_energy=23.0)

# or their rate constant (in M^n * s^-1)
sim.add_reaction(["B"], ["C"], rate=5e-4)

# it is also possible to enforce a K_eq (overriding
# the energy value assigned to the involved species).
# The rate of the reaction will be assumed as fast,
# to ensure equilibration at any point in time.
sim.add_reaction(["B"], ["B(adsorbed)"], enforced_K_eq=2.18)

# specify the timeframe of the simulation and run it
sim.run(time=10, t_units="h")

# running simulations with time=0 is also possible
# to get equilibrium concentrations
# sim.run(time=0)
```

    +------------------+-----------------+----------+-----------------------------+
    |     Reaction     | Faster k (s^-1) |   K_eq   |          Speed Rank         |
    +------------------+-----------------+----------+-----------------------------+
    |    A -> B + B    |     1.83e-03    | 5.09e+02 | SLOW:  evolved step-by-step |
    |      B -> C      |     5.00e-04    | 8.89e+18 | SLOW:  evolved step-by-step |
    | B -> B(adsorbed) |      N / A      | 2.18e+00 | K_EQ: always at equilibrium |
    +------------------+-----------------+----------+-----------------------------+

    --> Running simulation for 10 h with the Backwards Euler method (3.6 s increments, 10000 iterations)
    Iterations  |##################################################| 100.0%

    --> Simulation complete (3.0 s)

Show results:


```python
sim.show()
```


    Final Concentrations:
    A           : 0.00 M (0.0 % of initial conc., 100.00 % consumed)
    B           : 0.00 M (0.12 % total molar fraction)
    B(adsorbed) : 0.01 M (0.26 % total molar fraction)
    C           : 1.99 M (99.62 % total molar fraction)





![png](assets/README_files/README_5_1.png)



Only visualize the evolution of some species:


```python
sim.show(species=("B", "B(adsorbed)"))
```


    Final Concentrations:
    B           : 0.00 M (0.12 % total molar fraction)
    B(adsorbed) : 0.01 M (0.26 % total molar fraction)





![png](assets/README_files/README_7_1.png)



Simulator data is available for further manipulation:


```python
import matplotlib.pyplot as plt
import numpy as np

# plot [C] vs. time in a loglog plot
C_concs = sim.results["C"]
plt.loglog(sim.time_data, C_concs, label="Conc. of C over time")
plt.xlabel("ln(time (s))")
plt.ylabel("ln(conc. (M))")

# define and plot an interpolation time range
start, end = 5e1, 5e2  # 1E3
plt.axvspan(start, end, color="red", alpha=0.25, label="interpolation area")

# get boundary indices of arrays
start_idx = next(index + 1 for index, value in enumerate(sim.time_data[1:]) if value > start)
end_idx = (
    len(sim.time_data)
    - 2
    - next(index for index, value in enumerate(reversed(sim.time_data[:-1])) if value < end)
)

# fit line through logarithms, in the defined range
log_time_interp = np.log(sim.time_data[start_idx:end_idx])
log_conc_interp = np.log(C_concs[start_idx:end_idx])
X = np.vstack([log_time_interp, np.ones(log_time_interp.shape[0])]).T
m, c = np.linalg.lstsq(X, log_conc_interp, rcond=None)[0]

# draw an interpolation line
x_fit = (start, end)
y_fit = (start**m * np.exp(c), end**m * np.exp(c))
plt.plot(
    x_fit,
    y_fit,
    label=f"linear fit (slope = {m:.3f})",
    color="black",
    linestyle="dashed",
    markersize=4,
)
_ = plt.legend()
```



![png](assets/README_files/README_9_0.png)
