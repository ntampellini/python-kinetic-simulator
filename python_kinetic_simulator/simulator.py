"""Python Kinetic Simulator: A flexible kinetic simulator for chemical reactions."""

from collections import Counter
from time import perf_counter
from typing import Any, Iterable, cast

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

K_BOLTZMANN = 1.380649e-23  # J/K
H_PLANCK = 6.62607015e-34  # J*s
R = 0.001985877534  # kcal/(mol*K)

# number of points to save to generate final plot
PLOT_NUM_POINTS = 1000

# maximum number of computed steps
MAX_AUTO_STEPS = 1e4


def time_to_string(total_time: float, verbose: bool = False, digits: int = 1) -> str:
    """Convert totaltime (float) to a timestring with hours, minutes and seconds."""
    timestring = ""

    names = ("days", "hours", "minutes", "seconds") if verbose else ("d", "h", "m", "s")

    if total_time > 24 * 3600:
        d = total_time // (24 * 3600)
        timestring += f"{int(d)} {names[0]} "
        total_time %= 24 * 3600

    if total_time > 3600:
        h = total_time // 3600
        timestring += f"{int(h)} {names[1]} "
        total_time %= 3600

    if total_time > 60:
        m = total_time // 60
        timestring += f"{int(m)} {names[2]} "
        total_time %= 60

    timestring += f"{round(total_time, digits):{2 + digits}} {names[3]}"

    return timestring


class FastEquilibriumSolver:
    """
    Solver for multiple coupled fast reactions assumed to be at equilibrium.

    Each reaction is defined by:
    - Stoichiometric matrix (reactants negative, products positive)
    - Equilibrium constant
    """

    def __init__(self, species_names: list[str], reactions: dict[str, dict[str, Any]]) -> None:
        """
        Initialize the fast equilibrium solver.

        Parameters
        ----------
        species_names : list of str
            Names of all species involved
        reactions : dict of dict
            Dict mapping reaction names to reaction dicts. Each reaction dict contains:
            - 'K_eq': equilibrium constant
            - 'speed_rank': 'instantaneous' or 'normal'
                (only 'instantaneous' reactions are equilibrated)
            - 'name': reaction name (optional)

        Example:
        --------
        reactions = {
            'rxn1': {
                'reagents': ['A', 'B'], 'products': ['C'],
                'K_eq': 10.0, 'speed_rank': 'instantaneous', 'name': 'A+B<->C'
            },
            'rxn2': {
                'reagents': ['C'], 'products': ['D'],
                'K_eq': 5.0, 'speed_rank': 'instantaneous', 'name': 'C<->D'
            },
            'rxn3': {
                'reagents': ['D'], 'products': ['E'],
                'K_eq': 2.0, 'speed_rank': 'normal', 'name': 'D->E (slow)'
            }
        }
        """
        self.species_names = species_names
        self.n_species = len(species_names)

        # Filter for only instantaneous reactions
        self.fast_reactions = [
            r
            for r in reactions.values()
            if r.get("speed_rank") in ("instantaneous", "enforced_K_eq")
        ]
        self.all_reactions = reactions  # Keep reference to all reactions
        self.n_reactions = len(self.fast_reactions)

        # calculate stoichiometry
        for reaction in self.fast_reactions:
            reaction["stoichiometry"] = {
                **{name: -reaction["reagents"].count(name) for name in reaction["reagents"]},
                **{name: +reaction["products"].count(name) for name in reaction["products"]},
            }

        # Create species name to index mapping
        self.species_to_idx = {name: i for i, name in enumerate(species_names)}

        # Build stoichiometric matrix (n_species x n_reactions) for fast reactions only
        self.stoich_matrix = np.zeros((self.n_species, self.n_reactions))

        self.K_eq_values = np.zeros(self.n_reactions)

        for j, reaction in enumerate(self.fast_reactions):
            K_eq_name = "K_eq" if reaction["speed_rank"] == "instantaneous" else "enforced_K_eq"
            self.K_eq_values[j] = reaction[K_eq_name]
            for species, coeff in reaction["stoichiometry"].items():
                if species in self.species_to_idx:
                    i = self.species_to_idx[species]
                    self.stoich_matrix[i, j] = coeff

        # Precompute log(K_eq) once — avoids recomputing it on every solver call.
        self.log_K_eq = np.log(np.maximum(self.K_eq_values, 1e-15))

    def calculate_equilibrium_concentrations(
        self,
        initial_concentrations: dict[str, float],
        bounds_check: bool = True,
        verbose: bool = False,
        max_iterations: int = 3,
    ) -> tuple[np.ndarray, np.ndarray, float, bool]:
        """
        Calculate equilibrium concentrations for the coupled fast reactions.

        Parameters
        ----------
        initial_concentrations : dict
            Initial concentrations, keys are species names.
        bounds_check : bool
            Whether to enforce positive concentrations
        verbose : bool
            Print convergence information
        max_iterations : int
            Maximum number of solver iterations if convergence fails

        Returns
        -------
        equilibrium_concentrations : np.array
            Equilibrium concentrations in same order as species_names
        extents : np.array
            Extents of reaction for each fast reaction
        success : bool
            Whether the solver converged
        """
        # Convert initial concentrations to array
        if isinstance(initial_concentrations, dict):
            C0 = np.array([initial_concentrations.get(name, 0.0) for name in self.species_names])
        else:
            C0 = np.array(initial_concentrations)

        # make sure we should change any concentrations
        if np.sum(C0) < 1e-15:
            return C0, np.zeros(self.n_reactions), np.inf, False

        def equilibrium_equations(xi: np.ndarray) -> np.ndarray:
            concentrations = C0 + self.stoich_matrix @ xi

            # 1. C1-Continuous smooth log extension to prevent stalls at 0.0
            epsilon = 1e-12
            is_small = concentrations < epsilon

            log_conc = np.empty_like(concentrations)
            # Normal log for valid concentrations
            log_conc[~is_small] = np.log(concentrations[~is_small])
            # Linear tangent extension for zero or negative values
            log_conc[is_small] = np.log(epsilon) + (concentrations[is_small] - epsilon) / epsilon

            log_Q = self.stoich_matrix.T @ log_conc
            equations = log_Q - self.log_K_eq

            # 2. Smooth, differentiable penalty for coupled negative concentrations
            negative_mask = concentrations < 0
            if np.any(negative_mask):
                # A strong quadratic penalty based on the magnitude of violation
                violation = concentrations[negative_mask]
                penalty = 1e8 * np.sum(violation**2)
                # Distribute the penalty to all equations
                equations += penalty

            return cast("np.ndarray", equations)

        def equilibrium_jacobian(xi: np.ndarray) -> np.ndarray:
            concentrations = C0 + self.stoich_matrix @ xi

            # Exact analytical derivative of the C1-continuous log extension
            epsilon = 1e-12
            is_small = concentrations < epsilon

            inv_C = np.empty_like(concentrations)
            inv_C[~is_small] = 1.0 / concentrations[~is_small]
            inv_C[is_small] = 1.0 / epsilon  # Constant slope matches the linear extension

            jac = (self.stoich_matrix.T * inv_C) @ self.stoich_matrix

            # 3. Provide the exact analytical derivative of our penalty
            negative_mask = concentrations < 0
            if np.any(negative_mask):
                dP_dC = np.zeros_like(concentrations)
                # Derivative of (1e8 * C^2) is (2e8 * C)
                dP_dC[negative_mask] = 2e8 * concentrations[negative_mask]

                # Chain rule: dP/dxi = dC/dxi * dP/dC
                dP_dxi = self.stoich_matrix.T @ dP_dC

                # Broadcast the penalty derivative across all rows
                jac += dP_dxi

            return cast("np.ndarray", jac)

        # Calculate bounds for extents
        bounds = None
        if bounds_check:
            bounds = []
            for j in range(self.n_reactions):
                max_forward = np.inf
                max_backward = np.inf

                for i in range(self.n_species):
                    stoich_coeff = self.stoich_matrix[i, j]
                    if stoich_coeff < 0:  # Reactant consumed in forward direction
                        if C0[i] > 1e-12:
                            max_forward = min(max_forward, C0[i] / abs(stoich_coeff))
                    elif stoich_coeff > 0:  # Product consumed in reverse direction
                        if C0[i] > 1e-12:
                            max_backward = min(max_backward, C0[i] / stoich_coeff)

                # Set conservative bounds
                lower = -max_backward if max_backward != np.inf else -1e3
                upper = max_forward if max_forward != np.inf else 1e3
                bounds.append((lower, upper))

        # Iterative solver: always start from 0 for step-wise updates
        for iteration in range(max_iterations):
            try:
                if iteration == 0:
                    # The perfect guess for a delta step is always 0
                    xi_guess = np.zeros(self.n_reactions)
                else:
                    # If we failed and are retrying, perturb slightly around 0
                    xi_guess = np.random.normal(0, 1e-6, self.n_reactions)

                # Solve the system
                if bounds:
                    bounds_array = np.array(bounds).T
                    result = least_squares(
                        equilibrium_equations,
                        xi_guess,
                        jac=equilibrium_jacobian,
                        bounds=bounds_array,
                        method="trf",
                        ftol=1e-12,
                        xtol=1e-12,
                    )
                else:
                    result = least_squares(
                        equilibrium_equations,
                        xi_guess,
                        jac=equilibrium_jacobian,
                        method="lm",
                        ftol=1e-12,
                        xtol=1e-12,
                    )

                success: bool = result.success
                xi_final = result.x
                fun_norm = float(np.linalg.norm(result.fun))

                # Check convergence quality
                if success and fun_norm < 1e-8:
                    break
                elif success and fun_norm < 1e-6:
                    # Acceptable convergence, but check if we can do better
                    if iteration < max_iterations - 1:
                        continue
                    else:
                        break
                elif not success and iteration == max_iterations - 1:
                    if verbose:
                        print("Failed to converge after maximum iterations")
                    break

            except Exception as e:
                if verbose:
                    print(f"Iteration {iteration + 1} failed: {e}")
                if iteration == max_iterations - 1:
                    return C0, np.zeros(self.n_reactions), np.inf, False
                continue

        # Calculate final concentrations
        C_eq = C0 + self.stoich_matrix @ xi_final

        # Final bounds check and cleanup
        if np.any(C_eq < -1e-10):
            if verbose:
                print("Warning: Some equilibrium concentrations are significantly negative.")
            # Try to resolve by reducing problematic extents
            for _attempt in range(3):
                xi_final *= 0.9
                C_eq = C0 + self.stoich_matrix @ xi_final
                if np.all(C_eq >= -1e-15):
                    break

        # Calculate final concentrations before cleanup
        C_eq = C0 + self.stoich_matrix @ xi_final

        # Exact Mass-Conserving Projection
        # If coupled reactions overdrew a species, find the exact multiplier (alpha)
        # to scale back the extents so the limiting species hits exactly 0.
        if np.any(C_eq < 0):
            alpha = 1.0
            for i in range(self.n_species):
                if C_eq[i] < 0:
                    delta = C_eq[i] - C0[i]  # Total change for this species
                    if delta < 0:
                        # We need C0 + alpha * delta >= 0
                        max_alpha = -C0[i] / delta
                        alpha = min(alpha, max_alpha)

            # Scale back all extents uniformly to conserve stoichiometry
            xi_final *= alpha
            C_eq = C0 + self.stoich_matrix @ xi_final

        # Any remaining negatives are pure floating-point noise (e.g., -1e-18).
        # Setting these to exactly 0.0 is safe and adds negligible mass.
        C_eq[C_eq < 0] = 0.0

        return C_eq, xi_final, fun_norm, success

    def get_equilibrium_dict(
        self, initial_concentrations: dict[str, float], **kwargs: Any
    ) -> tuple[dict[str, float], float]:
        """Return equilibrium concentrations as a dict."""
        if len(self.fast_reactions) > 0:
            C_eq, xi, fun_norm, success = self.calculate_equilibrium_concentrations(
                initial_concentrations, **kwargs
            )

            result = {name: C_eq[i] for i, name in enumerate(self.species_names)}
            result["_extents"] = xi
            result["_success"] = success

        else:
            result = {name: initial_concentrations[name] for name in self.species_names}
            result["_extents"] = "N/A"
            result["_success"] = True
            fun_norm = 0

        return result, fun_norm


class Simulator:
    """Simulator for kinetic schemes with flexible reactions and equilibria definitions."""

    def __init__(self, T_C: float = 25.0) -> None:
        """Initialize the simulator with a given temperature in Celsius."""
        self.T: float = T_C + 273.15
        self.T_C = T_C
        self.species: dict[str, dict[str, float]] = {}
        self.reactions: dict[str, dict[str, Any]] = {}

    def add_species(
        self, name: str, energy: float | None = None, conc: float | None = None
    ) -> None:
        """Add a state with name, energy (kcal/mol), and initial concentration (mol/L)."""
        if conc is None:
            conc = 0.0
        if energy is None:
            energy = 0.0

        assert conc > -1e-10

        self.species[name] = {
            "energy": float(energy),
            "conc": float(conc),
        }

        # update current conc dict
        self.current_conc_dict = {name: state["conc"] for name, state in self.species.items()}
        self.species_id_dict = {name: i for i, name in enumerate(self.species)}

    def add_reaction(
        self,
        reagents: list[str],
        products: list[str],
        ts_energy: float | None = None,
        rate: float | None = None,
        inv_rate: float | None = None,
        throughput_tgt: str | None = None,
        enforced_K_eq: float | None = None,
    ) -> None:
        """Add a reaction with reagents, products, and either a TS energy or a rate constant.

        Reagents and product: lists of strings
        ts_energy: absolute value relative to the
            whole PES, in kcal/mol (overrides rate)
        rate: forward reaction rate, in M^n * s^-1 (overridden by ts_energy)
        inv_rate: reverse reaction rate, in M^n * s^-1 (overridden by ts_energy)
        throughput_tgt: string with species name - will keep track of how
            much of this species is generated through this reaction
        enforced_K_eq: will consider the reaction instantaneous and always at
            equilibrium obeying the provided constant.
        """
        # check if we know all names
        for name in set(reagents + products):
            if name not in self.species.keys():
                raise NameError(f'State name "{name}" not defined.')

        # create a reaction hash string
        hash_name = " + ".join(reagents) + " -> " + " + ".join(products)

        if enforced_K_eq is not None:
            self.reactions[hash_name] = {
                "reagents": reagents,
                "products": products,
                "enforced_K_eq": enforced_K_eq,
            }

        else:
            if ts_energy is not None:
                # calculate the activation energy relative to the reagents
                activation_energy = ts_energy - np.sum(
                    [self.species[name]["energy"] for name in reagents]
                )
                inverse_act_energy = ts_energy - np.sum(
                    [self.species[name]["energy"] for name in products]
                )

                # calculate reaction rates
                k_rate = get_eyring_k(activation_energy, self.T)
                k_inv = get_eyring_k(inverse_act_energy, self.T)

            elif rate is not None:
                k_rate = rate
                k_inv = inv_rate or 0.0

            else:
                raise RuntimeError("Please provide either the reaction rate or absolute ts_energy")

            # add the reaction to the self.reactions attribute
            self.reactions[hash_name] = {
                "reagents": reagents,
                "products": products,
                "activation_energy": self.get_ts_energy(reagents, k_rate),
                "k_rate": k_rate,
                "k_inv": k_inv,
                "faster_k": k_rate if k_rate > k_inv else k_inv,
                "K_eq": self.get_K_eq(reagents, products),
            }

        if throughput_tgt:
            self.reactions[hash_name]["cumulative_throughput"] = 0.0
            self.reactions[hash_name]["throughput_tgt"] = throughput_tgt

        # We no longer calculate speed_rank here unless it is strictly
        # enforced by the user. Dynamic kinetic ranking (instantaneous vs normal)
        # will be evaluated in run() based on dt_s.
        if "enforced_K_eq" in self.reactions[hash_name]:
            self.reactions[hash_name]["speed_rank"] = "enforced_K_eq"
            self.reactions[hash_name]["description"] = (
                f'--> "{hash_name}" will be enforced at the provided equilibrium constant '
                f"(K_eq = {self.reactions[hash_name]['enforced_K_eq']:.2e})."
            )

    def get_K_eq(self, reagents: list[str], products: list[str]) -> float:
        """Return the equilibrium constant for a reaction from reagents and products energies."""
        dG: float = 0.0
        for product in products:
            dG += self.species[product]["energy"]
        for reagent in reagents:
            dG -= self.species[reagent]["energy"]
        return cast("float", np.exp(-dG / (R * self.T)))

    def evaluate_dynamic_kinetic_ranking(self) -> None:
        """Evaluate the speed rank (instantaneous vs normal) of each reaction based on dt_s.

        Define the threshold for instantaneous reactions.
        If the rate constant k > (FAST_THRESHOLD_MULTIPLIER / dt_s), the reaction will essentially
        reach equilibrium within a single time step, so we flag it as instantaneous.

        From Gemini:

        The Math Behind the Multiplier

        In chemical kinetics, the progression of a first-order (or pseudo-first-order) reaction
        follows an exponential decay based on its rate constant k and time t:

            Fraction Remaining = e^{-k * delta_t}

        By setting your threshold to 10.0 / dt_s, you are saying the algorithm should switch
        to the instant equilibrium solver when k * delta_t >= 10.If we plug 10 into that equation:

            e^{-10} ~ 0.000045

        This means that within a single time step, the reaction will reach
        ~99.995% of its equilibrium state.

        """
        FAST_THRESHOLD_MULTIPLIER = 10.0
        thr_abs_fast_rxn = FAST_THRESHOLD_MULTIPLIER / self.dt_s

        print("\nReaction Classifications for this run:")
        for hash_name, reaction in self.reactions.items():
            # Skip reactions that were explicitly enforced
            if reaction.get("speed_rank") == "enforced_K_eq":
                pass
            elif reaction["faster_k"] > thr_abs_fast_rxn:
                reaction["speed_rank"] = "instantaneous"
                reaction["description"] = (
                    f'--> "{hash_name}" is very fast relative to dt: will be considered always at '
                    + f"equilibrium (Rate = {reaction['faster_k']:.2e} s^-1, "
                    + f"dt = {self.dt_s:.2e} s)."
                )
            else:
                reaction["speed_rank"] = "normal"
                reaction["description"] = (
                    f'--> "{hash_name}" will be evolved step-by-step'
                    f" (Rate = {reaction['faster_k']:.2e} s^-1)."
                )

            print(reaction["description"])

    def run(
        self,
        time: float = 1,
        t_units: str = "s",
        dt_s: float | None = None,
        max_equilib_iters: int = 5,
    ) -> None:
        """Run the simulation for a given time, with an optional custom time step (in s)."""
        self.run_t_units = t_units
        self.max_equilib_iters = max_equilib_iters

        # Reset cached solver so it is rebuilt with the current species/reactions.
        if hasattr(self, "equilibrium_solver"):
            del self.equilibrium_solver

        self.multiplier = {
            "s": 1,
            "m": 60,
            "h": 3600,
            "d": 3600 * 24,
        }[t_units]

        if dt_s is None:
            dt_s = time * self.multiplier / MAX_AUTO_STEPS

        self.dt_s = dt_s

        self.evaluate_dynamic_kinetic_ranking()

        iterations = int(np.ceil(time * self.multiplier / self.dt_s))
        print(
            f"\n--> Running simulation for {time} {t_units} with the Backwards Euler method "
            f"({self.dt_s:.1{'f' if self.dt_s > 0.1 else 'e'}} "
            f"s increments, {iterations} iterations)"
        )
        plot_num_points = min(PLOT_NUM_POINTS, iterations)
        save_every = max(1, int(iterations / plot_num_points)) if plot_num_points > 0 else 1

        # equilibrate before the main loop
        self._equilibrate_instantaneous_reactions()

        # Calculate total number of data points
        total_points = 1 + (iterations // save_every)

        # Preallocate arrays
        self.time_data = np.zeros(total_points)
        self.conc_data = np.zeros((total_points, len(self.species)))
        self.data_idx = 0

        # Set initial data point
        self.time_data[self.data_idx] = 0.0
        self.conc_data[self.data_idx] = np.array(list(self.current_conc_dict.values()))
        self.data_idx += 1

        t_start = perf_counter()
        for i in range(1, iterations + 1):
            self.current_time_s = self.dt_s * i
            loadbar(i, iterations, prefix="Iterations ")

            # first, iteratively evolve all normal reactions
            for rxn in self.reactions.values():
                if rxn["speed_rank"] == "normal":
                    self._normal_reaction_step(rxn)

            # then, equilibrate all instantaneous reactions together
            self._equilibrate_instantaneous_reactions()

            # if it's time, collect a datapoint
            if i % save_every == 0:
                self._add_status_to_results()

            if min(self.current_conc_dict.values()) < -1e-6:
                s = (
                    "-> Something blew up and we got a negative concentration. "
                    "(< -1E-6 M, {i} iterations)"
                )
                print(s)
                break

        self.results = dict(zip(self.species.keys(), self.conc_data.T, strict=True))

        print(f"\n--> Simulation complete ({time_to_string(perf_counter() - t_start)})")

    def _normal_reaction_step(self, rxn: dict[str, Any]) -> None:
        """Evolve normal (slow) reaction by one iteration."""
        deltaC = self._get_deltaC_step(rxn)

        # Bckw Euler + mass balance
        for reagent in rxn["reagents"]:
            self.current_conc_dict[reagent] -= deltaC

        for product in rxn["products"]:
            self.current_conc_dict[product] += deltaC

        if "cumulative_throughput" in rxn:
            rxn["cumulative_throughput"] += deltaC * rxn["products"].count(rxn["throughput_tgt"])

    def _get_deltaC_step(self, rxn: dict[str, Any]) -> float:
        """Return implicit Newton iteration step on reaction extent (robust for stiff kinetics)."""
        dt = self.dt_s

        # Stoichiometry
        reagents = rxn["reagents"]
        products = rxn["products"]

        # Initial concentrations
        C0 = self.current_conc_dict

        # Precompute multiplicities (handles duplicates correctly)
        nu_reac = Counter(reagents)
        nu_prod = Counter(products)

        kf = rxn["k_rate"]
        kb = rxn["k_inv"]

        # Bounds: cannot consume more than available
        max_forward = min(C0[r] / nu_reac[r] for r in nu_reac) if nu_reac else np.inf
        max_backward = min(C0[p] / nu_prod[p] for p in nu_prod) if nu_prod else np.inf

        # Initial guess: explicit Euler
        def rate(C: dict[str, float]) -> float:
            forward = kf * np.prod([C[r] ** nu_reac[r] for r in nu_reac])
            backward = kb * np.prod([C[p] ** nu_prod[p] for p in nu_prod])
            return cast("float", forward - backward)

        delta = dt * rate(C0)

        # Clamp initial guess
        delta = max(-max_backward, min(delta, max_forward))

        # Newton iterations
        for _ in range(8):
            # Build updated concentrations
            C = {}
            for s in C0:
                C[s] = C0[s]

            for r in nu_reac:
                C[r] -= nu_reac[r] * delta
            for p in nu_prod:
                C[p] += nu_prod[p] * delta

            # Prevent negative concentrations during iteration
            for s, val in C.items():
                C[s] = max(val, 1e-15)

            # f(Δ)
            f = delta - dt * rate(C)

            # Numerical derivative df/dΔ
            eps = 1e-8 * max(1.0, abs(delta))
            delta_eps = delta + eps

            C_eps: dict[str, float] = {}
            for s in C0:
                C_eps[s] = C0[s]
            for r in nu_reac:
                C_eps[r] -= nu_reac[r] * delta_eps
            for p in nu_prod:
                C_eps[p] += nu_prod[p] * delta_eps

            # Prevent negative concentrations again
            for s, val in C_eps.items():
                C_eps[s] = max(val, 1e-15)

            f_eps = delta_eps - dt * rate(C_eps)

            df = (f_eps - f) / eps

            if abs(df) < 1e-12:
                break

            step = f / df
            delta -= step

            # Clamp to physical bounds
            delta = max(-max_backward, min(delta, max_forward))

            if abs(step) < 1e-12:
                break

        return delta

    def _equilibrate_instantaneous_reactions(self, **kwargs: Any) -> None:
        """Equilibrate all instantaneous reactions together using the FastEquilibriumSolver.

        This method is called before the main loop to equilibrate the initial state,
        and after each normal reaction step to re-equilibrate the fast reactions.
        """
        if not hasattr(self, "equilibrium_solver"):
            self.equilibrium_solver = FastEquilibriumSolver(
                list(self.species.keys()), self.reactions
            )

        for _ in range(self.max_equilib_iters):
            # Solve for equilibrium
            new_conc_dict, fun_norm = self.equilibrium_solver.get_equilibrium_dict(
                self.current_conc_dict, **kwargs
            )

            for species in self.species.keys():
                self.current_conc_dict[species] = new_conc_dict[species]

            if fun_norm < 1e-2 or fun_norm == np.inf:
                break

    def _add_status_to_results(self) -> None:
        """Add the current concentrations and time to the results arrays for plotting."""
        current_concs = np.array(list(self.current_conc_dict.values()))
        self.time_data[self.data_idx] = self.current_time_s
        self.conc_data[self.data_idx] = current_concs
        self.data_idx += 1

    def show(self, species: Iterable[str] | None = None) -> None:
        """Show the concentration profiles of the species over time.

        species: iterable of strings of species to show, defaulting to all species.
        """
        species_to_plot = species or self.species

        x = np.array(self.time_data) / self.multiplier

        plt.figure()
        print("\nFinal Concentrations:")
        sum_of_final_concs = np.sum([concs[-1] for concs in self.results.values()])
        longest = max(len(name) for name in self.species.keys())

        for name, concs in self.results.items():
            if name in species_to_plot:
                plt.plot(x, concs, label=name)
                s = f"{name:{longest}s} : {concs[-1]:.2f} M"
                if self.species[name]["conc"] > 0:
                    final_percentage = concs[-1] / self.species[name]["conc"] * 100
                    s += (
                        f" ({final_percentage:.1f} % of initial conc., "
                        f"{100 - final_percentage:.2f} % consumed)"
                    )

                else:
                    final_fraction = concs[-1] / sum_of_final_concs
                    s += f" ({final_fraction * 100:.2f} % total molar fraction)"
                print(s)

        print()

        for hash_name, reaction in self.reactions.items():
            if "cumulative_throughput" in reaction:
                fraction = (
                    reaction["cumulative_throughput"]
                    / self.current_conc_dict[reaction["throughput_tgt"]]
                )
                print(
                    f'Reaction "{hash_name}" throughput is '
                    f"{reaction['cumulative_throughput']:.3f} M, "
                    f'{fraction * 100:.2f} % of final "{reaction["throughput_tgt"]}" conc.'
                )

        plt.legend()
        plt.title(f"Concentrations over time (T={self.T_C} °C)")
        plt.xlabel(f"Time ({self.run_t_units})")
        plt.ylabel("Concentration (M)")
        plt.show()

    def get_ts_energy(self, reagents: list[str], rate: float) -> float:
        """Return the absolute TS energy in kcal/mol from reagents and a rate constant.

        reagents: list of strings
        rate: forward reaction rate in M^n * s^-1
        """
        activation_energy = -np.log(rate * H_PLANCK / K_BOLTZMANN / self.T) * (R * self.T)
        reagents_energy = np.sum([self.species[r]["energy"] for r in reagents])
        return cast("float", activation_energy + reagents_energy)


def get_eyring_k(activation_energy: float, T: float = 298.15) -> float:
    """Return a rate constant in s^-1 given an activation energy in kcal/mol and a temperature."""
    return K_BOLTZMANN / H_PLANCK * T * cast("float", np.exp(-activation_energy / (R * T)))


def loadbar(
    iteration: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    decimals: int = 1,
    length: int = 50,
    fill: str = "#",
) -> None:
    """Print a progress bar to the console."""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end="\r")
    if iteration == total:
        print()
