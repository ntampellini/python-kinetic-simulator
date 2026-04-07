"""Python Kinetic Simulator: A flexible kinetic simulator for chemical reactions."""

from time import perf_counter
from typing import Any, Iterable, cast

import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
from scipy.optimize import least_squares, root

K_BOLTZMANN = 1.380649e-23  # J/K
H_PLANCK = 6.62607015e-34  # J*s
R = 0.001985877534  # kcal/(mol*K)

# number of points to save to generate final plot
PLOT_NUM_POINTS = 1000

# maximum number of computed steps
MAX_AUTO_STEPS = 1e4


class tcolors:
    """Terminal colors for pretty printing.

    Use like: print(tcolors.RED + "This is red" + tcolors.ENDC)
    """

    GREY = "\033[90m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


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
    """Solver for multiple coupled fast reactions assumed to be at equilibrium."""

    def __init__(self, species_names: list[str], reactions: dict[str, dict[str, Any]]) -> None:
        self.species_names = species_names
        self.n_species = len(species_names)

        # Filter for only instantaneous reactions
        self.fast_reactions = [
            r
            for r in reactions.values()
            if r.get("speed_rank") in ("instantaneous", "enforced_K_eq")
        ]
        self.n_reactions = len(self.fast_reactions)

        # calculate stoichiometry
        for reaction in self.fast_reactions:
            reaction["stoichiometry"] = {
                **{name: -reaction["reagents"].count(name) for name in reaction["reagents"]},
                **{name: +reaction["products"].count(name) for name in reaction["products"]},
            }

        self.species_to_idx = {name: i for i, name in enumerate(species_names)}
        self.stoich_matrix = np.zeros((self.n_species, self.n_reactions))
        self.K_eq_values = np.zeros(self.n_reactions)

        for j, reaction in enumerate(self.fast_reactions):
            K_eq_name = "K_eq" if reaction["speed_rank"] == "instantaneous" else "enforced_K_eq"
            self.K_eq_values[j] = reaction[K_eq_name]
            for species, coeff in reaction["stoichiometry"].items():
                if species in self.species_to_idx:
                    i = self.species_to_idx[species]
                    self.stoich_matrix[i, j] = coeff

        self.log_K_eq = np.log(np.maximum(self.K_eq_values, 1e-15))

        # Cache for warm starting the solver
        self.last_xi = np.zeros(self.n_reactions)

    def calculate_equilibrium_concentrations(
        self,
        C0: np.ndarray,
        verbose: bool = False,
        max_iterations: int = 3,
    ) -> tuple[np.ndarray, np.ndarray, float, bool]:
        """Calculate equilibrium using native NumPy arrays natively."""
        # make sure we should change any concentrations
        if np.sum(C0) < 1e-15:
            return C0, np.zeros(self.n_reactions), np.inf, False

        def equilibrium_equations(xi: np.ndarray) -> np.ndarray:
            concentrations = C0 + self.stoich_matrix @ xi

            # 1. C1-Continuous smooth log extension to prevent stalls at 0.0
            epsilon = 1e-12
            is_small = concentrations < epsilon

            log_conc = np.empty_like(concentrations)
            log_conc[~is_small] = np.log(concentrations[~is_small])
            log_conc[is_small] = np.log(epsilon) + (concentrations[is_small] - epsilon) / epsilon

            log_Q = self.stoich_matrix.T @ log_conc
            equations = log_Q - self.log_K_eq

            # 2. Smooth, differentiable penalty for coupled negative concentrations
            negative_mask = concentrations < 0
            if np.any(negative_mask):
                violation = concentrations[negative_mask]
                penalty = 1e8 * np.sum(violation**2)
                equations += penalty

            return cast("np.ndarray", equations)

        def equilibrium_jacobian(xi: np.ndarray) -> np.ndarray:
            concentrations = C0 + self.stoich_matrix @ xi

            epsilon = 1e-12
            is_small = concentrations < epsilon

            inv_C = np.empty_like(concentrations)
            inv_C[~is_small] = 1.0 / concentrations[~is_small]
            inv_C[is_small] = 1.0 / epsilon

            jac = (self.stoich_matrix.T * inv_C) @ self.stoich_matrix

            negative_mask = concentrations < 0
            if np.any(negative_mask):
                dP_dC = np.zeros_like(concentrations)
                dP_dC[negative_mask] = 2e8 * concentrations[negative_mask]
                dP_dxi = self.stoich_matrix.T @ dP_dC
                jac += dP_dxi

            return cast("np.ndarray", jac)

        # Iterative solver block
        for iteration in range(max_iterations):
            try:
                if iteration == 0:
                    # --- SAFE WARM START ---
                    xi_guess = self.last_xi.copy()

                    # Project guess to strictly obey C >= 0 bounds
                    C_guess = C0 + self.stoich_matrix @ xi_guess
                    if np.any(C_guess < 0):
                        alpha = 1.0
                        for i in range(self.n_species):
                            if C_guess[i] < 0:
                                delta = C_guess[i] - C0[i]
                                if delta < 0:
                                    # Scale back the guess to prevent going negative.
                                    # We use 0.95 to leave a tiny safety margin above exactly 0.
                                    max_alpha = -C0[i] / delta
                                    alpha = min(alpha, max_alpha * 0.95)

                        # Uniformly scale down the guess to preserve stoichiometry
                        xi_guess *= alpha

                elif iteration == 1:
                    # --- FALLBACK 1: COLD START ---
                    # If warm start failed, perfect zero is the mathematically safest fallback
                    xi_guess = np.zeros(self.n_reactions)
                else:
                    # --- FALLBACK 2: JITTERED START ---
                    # If both failed, perturb slightly to escape saddle points
                    xi_guess = np.random.normal(0, 1e-6, self.n_reactions)

                # Solve the system
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
                elif success and fun_norm < 1e-6 and iteration == max_iterations - 1:
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

        # --- UPDATE CACHE ---
        if success:
            self.last_xi = xi_final.copy()
        else:
            self.last_xi = np.zeros(self.n_reactions)

        # Calculate final concentrations...
        C_eq = C0 + self.stoich_matrix @ xi_final

        # Bounds check and cleanup
        if np.any(C_eq < -1e-10):
            if verbose:
                print("Warning: Some equilibrium concentrations are significantly negative.")
            for _attempt in range(3):
                xi_final *= 0.9
                C_eq = C0 + self.stoich_matrix @ xi_final
                if np.all(C_eq >= -1e-15):
                    break

        C_eq = C0 + self.stoich_matrix @ xi_final

        # Exact Mass-Conserving Projection
        if np.any(C_eq < 0):
            alpha = 1.0
            for i in range(self.n_species):
                if C_eq[i] < 0:
                    delta = C_eq[i] - C0[i]
                    if delta < 0:
                        max_alpha = -C0[i] / delta
                        alpha = min(alpha, max_alpha)
            xi_final *= alpha
            C_eq = C0 + self.stoich_matrix @ xi_final

        C_eq[C_eq < 0] = 0.0

        return C_eq, xi_final, fun_norm, success


class Simulator:
    """Simulator for kinetic schemes with flexible reactions and equilibria definitions."""

    def __init__(self, T_C: float = 25.0) -> None:
        """Initialize the simulator with a given temperature in Celsius."""
        self.T: float = T_C + 273.15
        self.T_C = T_C
        self.species: dict[str, dict[str, float]] = {}
        self.reactions: dict[str, dict[str, Any]] = {}
        # self.track_throughput: list[dict[str, Any]] = []

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

        self.species_names = list(self.species.keys())

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
            if name not in self.species_names:
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
            raise NotImplementedError("Willi try to re-implement this...")
            # self.reactions[hash_name]["cumulative_throughput"] = 0.0
            # self.reactions[hash_name]["throughput_tgt"] = throughput_tgt
            # self.track_throughput.append(self.reactions[hash_name])

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
        if self.dt_s == 0.0:  # type: ignore[has-type]
            return

        FAST_THRESHOLD_MULTIPLIER = 10.0
        thr_abs_fast_rxn = FAST_THRESHOLD_MULTIPLIER / self.dt_s  # type: ignore[has-type]

        table = PrettyTable()
        table.field_names = [
            tcolors.BOLD + "Reaction" + tcolors.ENDC,
            tcolors.BOLD + "Faster k (s^-1)" + tcolors.ENDC,
            tcolors.BOLD + "K_eq" + tcolors.ENDC,
            tcolors.BOLD + "Speed Rank" + tcolors.ENDC,
        ]

        # loop 1: set unspecified speed rank attributes
        for reaction in self.reactions.values():
            if reaction.get("speed_rank") is not None:
                pass
            elif reaction["faster_k"] > thr_abs_fast_rxn:
                reaction["speed_rank"] = "instantaneous"
            else:
                reaction["speed_rank"] = "normal"

        # loop 2: print table
        for hash_name, reaction in self.reactions.items():
            match reaction["speed_rank"]:
                case "enforced_K_eq":
                    table.add_row(
                        [
                            hash_name,
                            tcolors.GREY + "N / A" + tcolors.ENDC,
                            tcolors.BOLD + f"{reaction['enforced_K_eq']:.2e}" + tcolors.ENDC,
                            tcolors.YELLOW + "K_EQ: always at equilibrium" + tcolors.ENDC,
                        ]
                    )
                case "instantaneous":
                    table.add_row(
                        [
                            hash_name,
                            tcolors.GREY + f"{reaction['faster_k']:.2e}" + tcolors.ENDC,
                            tcolors.BOLD + f"{reaction['K_eq']:.2e}" + tcolors.ENDC,
                            tcolors.GREEN + "FAST: always at equilibrium" + tcolors.ENDC,
                        ]
                    )
                case _:
                    table.add_row(
                        [
                            hash_name,
                            tcolors.BOLD + f"{reaction['faster_k']:.2e}" + tcolors.ENDC,
                            tcolors.GREY + f"{reaction['K_eq']:.2e}" + tcolors.ENDC,
                            tcolors.RED + "SLOW:  evolved step-by-step" + tcolors.ENDC,
                        ]
                    )

        # sort rows by the faster_k value in descending order
        table._sort_key = lambda row: float(row[1].split(" ")[0])

        # lef-align speed rank and color FAST green and SLOW red
        table.custom_format[tcolors.BOLD + "Speed Rank" + tcolors.ENDC] = lambda _, x: (
            tcolors.GREEN + f"{x:<s}" + tcolors.ENDC
            if "FAST" in x
            else tcolors.RED + f"{x:<s}" + tcolors.ENDC
        )

        print(table.get_string())

    def _compile_normal_reactions(self) -> None:
        """Build NumPy arrays for fully vectorized normal reaction rates."""
        self.normal_rxns = [r for r in self.reactions.values() if r.get("speed_rank") == "normal"]
        self.n_normal = len(self.normal_rxns)
        self.n_species = len(self.species_names)

        # N_species x M_reactions matrices
        self.nu_reactants = np.zeros((self.n_species, self.n_normal))
        self.nu_products = np.zeros((self.n_species, self.n_normal))

        self.kf_vec = np.zeros(self.n_normal)
        self.kb_vec = np.zeros(self.n_normal)

        for j, rxn in enumerate(self.normal_rxns):
            self.kf_vec[j] = rxn["k_rate"]
            self.kb_vec[j] = rxn["k_inv"]

            for r in rxn["reagents"]:
                self.nu_reactants[self.species_id_dict[r], j] += 1
            for p in rxn["products"]:
                self.nu_products[self.species_id_dict[p], j] += 1

        # Net stoichiometry matrix (Delta)
        self.nu_net = self.nu_products - self.nu_reactants

    def _calculate_rates(self, C_array: np.ndarray) -> np.ndarray:
        """Calculate rates for all normal reactions simultaneously."""
        # Add epsilon to prevent log(0) warnings
        C_safe = np.maximum(C_array, 1e-15)

        # Log-space dot product for better performance
        log_C = np.log(C_safe)
        forward_rates = self.kf_vec * np.exp(self.nu_reactants.T @ log_C)
        backward_rates = self.kb_vec * np.exp(self.nu_products.T @ log_C)

        return cast("np.ndarray", forward_rates - backward_rates)

    def _normal_reactions_step(self) -> None:
        """Evolve all normal reactions using an implicit Backward Euler step via SciPy."""
        if self.n_normal == 0:
            return

        C0 = self.C_array.copy()  # type: ignore[has-type]

        def residual(C_next: np.ndarray) -> np.ndarray:
            """Residual for Backward Euler: F(C_{n+1}) = C_{n+1} - C_n - dt * f(C_{n+1}) = 0."""
            # Prevent negative concentrations during solver steps
            C_safe = np.maximum(C_next, 1e-15)
            rates = self._calculate_rates(C_safe)

            # The rate of change for each species is nu_net @ rates
            return cast("np.ndarray", C_next - C0 - self.dt_s * (self.nu_net @ rates))  # type: ignore[has-type]

        # 'hybr' is MINPACK's Powell hybrid method, excellent for systems of nonlinear equations
        sol = root(residual, C0, method="hybr")

        # if self.track_throughput:
        #     for rxn_data in self.track_throughput:
        #         tgt_index = self.species_id_dict[rxn_data["throughput_tgt"]]
        #         delta = sol.x[tgt_index] - self.C_array[tgt_index]
        #         frac_contrib = 1
        #         rxn_data["cumulative_throughput"] += delta * frac_contrib

        # Update the main state array, clamping to 0 to eliminate floating-point noise
        self.C_array = np.maximum(sol.x, 0.0)

        # to reimplement ...
        # if "cumulative_throughput" in rxn:
        #     rxn["cumulative_throughput"] += deltaC * rxn["products"].count(rxn["throughput_tgt"])

    def get_sim_time(self) -> tuple[float, str]:
        """Return an estimated simulation time and unit of measure."""
        # 5 half lives should lead to >95% progress for the slowest reaction
        N_HALF_LIVES = 5

        slowest_k_fwd = min([rxn["k_rate"] for rxn in self.reactions.values() if "k_rate" in rxn])

        assert slowest_k_fwd > 0, (
            "The smallest non-zero forward rate constant must "
            "be greater than zero to estimate simulation time."
        )

        t_half_life = np.log(2) / slowest_k_fwd
        t_s = N_HALF_LIVES * t_half_life
        t_m = t_s // 60
        t_h = t_m // 60

        match True:
            case _ if t_s < 60:
                output = (round(t_s, 3), "s")
            case _ if t_m < 60:
                output = (t_m, "m")
            case _:
                output = (t_h, "h")

        print(f"--> Estimated simulation time: {output[0]:.1f} {output[1]}")

        return output

    def run(
        self,
        time: float | None = None,
        t_units: str = "s",
        dt_s: float | None = None,
        max_equilib_iters: int = 5,
    ) -> None:
        """Run the simulation for a given time, with an optional custom time step (in s)."""
        self.run_t_units = t_units
        self.max_equilib_iters = max_equilib_iters

        # Reset cached solver so it is rebuilt with the current species/reactions.
        if hasattr(self, "equilibrium_solver"):
            del self.equilibrium_solver  # type: ignore[has-type]

        self.multiplier = {
            "s": 1,
            "m": 60,
            "h": 3600,
            "d": 3600 * 24,
        }[t_units]

        if time is None:
            time, t_units = self.get_sim_time()

        time_s = time * self.multiplier

        if dt_s is None:
            dt_s = time_s / MAX_AUTO_STEPS

        self.dt_s = dt_s

        # evaluate which reactions are instantaneous
        # vs normal based on the provided or default dt_s
        self.evaluate_dynamic_kinetic_ranking()

        # with that, precompile the normal reactions for
        # fast vectorized stepping in the main loop
        self._compile_normal_reactions()

        # Initialize the working concentration array
        self.C_array = np.array([self.species[name]["conc"] for name in self.species_names])

        if self.dt_s != 0:
            iterations = int(np.ceil(time_s / self.dt_s))
            print(
                f"\n--> Running simulation for {time} {t_units} with the Backwards Euler method "
                f"({self.dt_s:.1{'f' if self.dt_s > 0.1 else 'e'}} "
                f"s increments, {iterations} iterations)"
            )
        else:
            iterations = 0

        plot_num_points = min(PLOT_NUM_POINTS, iterations)
        save_every = max(1, int(iterations / plot_num_points)) if plot_num_points > 0 else 1

        # Calculate total number of data points
        total_points = 1 + (iterations // save_every)

        # Preallocate arrays
        self.time_data = np.zeros(total_points)
        self.conc_data = np.zeros((total_points, len(self.species)))
        self.data_idx = 0

        t_start = perf_counter()

        # set solver and equilibrate before the main loop
        self.equilibrium_solver = FastEquilibriumSolver(list(self.species_names), self.reactions)
        self._equilibrate_instantaneous_reactions()

        # Set initial data point
        self.time_data[self.data_idx] = 0.0
        self.conc_data[self.data_idx] = self.C_array[:]
        self.data_idx += 1

        for i in range(1, iterations + 1):
            self.current_time_s = self.dt_s * i
            loadbar(i, iterations, prefix="Iterations ")

            # first, evolve all normal reactions
            self._normal_reactions_step()

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

        self.results = dict(zip(self.species_names, self.conc_data.T, strict=True))

        print(f"\n--> Simulation complete ({time_to_string(perf_counter() - t_start)})")

    def _equilibrate_instantaneous_reactions(self, **kwargs: Any) -> None:
        """Equilibrate all instantaneous reactions together using the FastEquilibriumSolver.

        This method is called before the main loop to equilibrate the initial state,
        and after each normal reaction step to re-equilibrate the fast reactions.
        """
        if self.equilibrium_solver.n_reactions == 0:
            return

        for _ in range(self.max_equilib_iters):
            # Pass C_array directly to the solver
            C_eq, _, fun_norm, _ = self.equilibrium_solver.calculate_equilibrium_concentrations(
                self.C_array, **kwargs
            )

            # Update array in place
            self.C_array = C_eq

            if fun_norm < 1e-2 or fun_norm == np.inf:
                break

    def _add_status_to_results(self) -> None:
        """Add the current concentrations and time to the results arrays."""
        self.time_data[self.data_idx] = self.current_time_s
        self.conc_data[self.data_idx] = self.C_array.copy()
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
        longest = max(len(name) for name in self.species_names)

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
                    reaction["cumulative_throughput"] / self.results[reaction["throughput_tgt"]][-1]
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
