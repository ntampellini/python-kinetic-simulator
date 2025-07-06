from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

K_BOLTZMANN = 1.380649E-23 # J/K
H_PLANCK = 6.62607015E-34 # J*s
R = 0.001985877534 # kcal/(mol*K)

# A reaction is considered instantaneous if
# more than THR_REL_FAST_RXN times faster than
# the slowest reaction of the run
THR_REL_FAST_RXN = 1e3

# steps must bring about a realtive change to the
# concentration of a species that is less than
# MAX_REL_CHANGE_PER_STEP (0.01 = 1%)
MAX_REL_CHANGE_PER_STEP = 0.01

# number of points to save to generate final plot
PLOT_NUM_POINTS = 1000

# maximum number of computed steps
MAX_AUTO_STEPS = 1e4

def time_to_string(total_time: float, verbose=False, digits=1):
    '''
    Converts totaltime (float) to a timestring
    with hours, minutes and seconds.
    '''
    timestring = ''

    names = ('days', 'hours', 'minutes', 'seconds') if verbose else ('d', 'h', 'm', 's')

    if total_time > 24*3600:
        d = total_time // (24*3600)
        timestring += f'{int(d)} {names[0]} '
        total_time %= (24*3600)

    if total_time > 3600:
        h = total_time // 3600
        timestring += f'{int(h)} {names[1]} '
        total_time %= 3600

    if total_time > 60:
        m = total_time // 60
        timestring += f'{int(m)} {names[2]} '
        total_time %= 60

    timestring += f'{round(total_time, digits):{2+digits}} {names[3]}'

    return timestring

class FastEquilibriumSolver:
    """
    Solver for multiple coupled fast reactions assumed to be at equilibrium.
    
    Each reaction is defined by:
    - Stoichiometric matrix (reactants negative, products positive)
    - Equilibrium constant
    """
    
    def __init__(self, species_names, reactions):
        """
        Initialize the fast equilibrium solver.
        
        Parameters:
        -----------
        species_names : list of str
            Names of all species involved
        reactions : list of dict
            Each reaction dict contains:
            - 'K_eq': equilibrium constant
            - 'speed_rank': 'instantaneous' or 'normal' (only 'instantaneous' reactions are equilibrated)
            - 'name': reaction name (optional)
        
        Example:
        --------
        reactions = [
            {'stoichiometry': {'A': -1, 'B': -1, 'C': 1}, 'K_eq': 10.0, 'speed_rank': 'instantaneous', 'name': 'A+B<->C'},
            {'stoichiometry': {'C': -1, 'D': 1}, 'K_eq': 5.0, 'speed_rank': 'instantaneous', 'name': 'C<->D'},
            {'stoichiometry': {'D': -1, 'E': 1}, 'K_eq': 2.0, 'speed_rank': 'normal', 'name': 'D->E (slow)'}
        ]
        """
        self.species_names = species_names
        self.n_species = len(species_names)
        
        # Filter for only instantaneous reactions
        self.fast_reactions = [r for r in reactions.values() if r.get('speed_rank') in ('instantaneous', 'enforced_K_eq')]
        self.all_reactions = reactions  # Keep reference to all reactions
        self.n_reactions = len(self.fast_reactions)

        # calculate stoichiometry
        for reaction in self.fast_reactions:
            reaction["stoichiometry"] = {
                **{name : -reaction["reagents"].count(name) for name in reaction["reagents"]},
                **{name : +reaction["products"].count(name) for name in reaction["products"]},
            }
        
        # Create species name to index mapping
        self.species_to_idx = {name: i for i, name in enumerate(species_names)}
        
        # Build stoichiometric matrix (n_species x n_reactions) for fast reactions only
        self.stoich_matrix = np.zeros((self.n_species, self.n_reactions))
        self.K_eq_values = np.zeros(self.n_reactions)
        
        for j, reaction in enumerate(self.fast_reactions):
            K_eq_name = 'K_eq' if reaction['speed_rank'] == "instantaneous" else 'enforced_K_eq'
            self.K_eq_values[j] = reaction[K_eq_name]
            for species, coeff in reaction['stoichiometry'].items():
                if species in self.species_to_idx:
                    i = self.species_to_idx[species]
                    self.stoich_matrix[i, j] = coeff

    def calculate_equilibrium_concentrations(self, initial_concentrations, 
                                           bounds_check=False, verbose=False, 
                                           max_iterations=3):
        """
        Calculate equilibrium concentrations for the coupled fast reactions.
        
        Parameters:
        -----------
        initial_concentrations : dict or array
            Initial concentrations. If dict, keys are species names.
            If array, order matches species_names.
        bounds_check : bool
            Whether to enforce positive concentrations
        verbose : bool
            Print convergence information
        max_iterations : int
            Maximum number of solver iterations if convergence fails
            
        Returns:
        --------
        equilibrium_concentrations : np.array
            Equilibrium concentrations in same order as species_names
        extents : np.array
            Extents of reaction for each fast reaction
        success : bool
            Whether the solver converged
        """
        

        # Convert initial concentrations to array
        if isinstance(initial_concentrations, dict):
            C0 = np.array([initial_concentrations.get(name, 0.0) 
                          for name in self.species_names])
        else:
            C0 = np.array(initial_concentrations)
        
        # make sure we should change any concentrations
        if np.sum(C0) < 1e-15:
            return C0, np.zeros(self.n_reactions), np.inf, False

        # Better initial guess for extents of reaction
        def get_initial_guess():
            """Generate a reasonable initial guess for extents"""
            xi_guess = np.zeros(self.n_reactions)
            
            # For each reaction, estimate a reasonable extent based on
            # equilibrium constant and available reactants
            for j in range(self.n_reactions):
                K_eq = self.K_eq_values[j]
                
                # Find limiting reactant and estimate extent
                min_reactant_ratio = np.inf
                for i in range(self.n_species):
                    stoich_coeff = self.stoich_matrix[i, j]
                    if stoich_coeff < 0:  # Reactant
                        if C0[i] > 1e-12:
                            reactant_ratio = C0[i] / abs(stoich_coeff)
                            min_reactant_ratio = min(min_reactant_ratio, reactant_ratio)
                
                if min_reactant_ratio != np.inf:
                    # For large K_eq, assume reaction goes mostly to completion
                    # For small K_eq, assume minimal progress
                    if K_eq > 100:
                        xi_guess[j] = 0.8 * min_reactant_ratio
                    elif K_eq > 1:
                        xi_guess[j] = 0.5 * min_reactant_ratio
                    else:
                        xi_guess[j] = 0.1 * min_reactant_ratio
            
            return xi_guess
        
        # Define the system of equations to solve
        def equilibrium_equations(xi):
            """
            System of equations: equilibrium constraints for each reaction
            """
            # Calculate concentrations from extents
            concentrations = C0 + self.stoich_matrix @ xi
            
            # Soft penalty for negative concentrations
            penalty = 0.0
            if bounds_check:
                negative_mask = concentrations < 0
                if np.any(negative_mask):
                    penalty = 1e6 * np.sum(concentrations[negative_mask]**2)
            
            equations = np.zeros(self.n_reactions)
            
            for j in range(self.n_reactions):
                # Calculate reaction quotient for reaction j
                # Q = [products]^stoich / [reactants]^stoich
                numerator = 1.0
                denominator = 1.0
                
                for i in range(self.n_species):
                    stoich_coeff = self.stoich_matrix[i, j]
                    if stoich_coeff != 0:
                        # Better handling of small concentrations
                        # Use a concentration floor that's proportional to the initial total
                        conc_floor = max(1e-12, 1e-6 * np.sum(C0))
                        conc = max(concentrations[i], conc_floor)
                        
                        if stoich_coeff > 0:  # Product
                            numerator *= conc ** stoich_coeff
                        else:  # Reactant (stoich_coeff < 0)
                            denominator *= conc ** abs(stoich_coeff)
                
                Q = numerator / denominator
                
                # Equilibrium constraint: log(Q) - log(K_eq) = 0
                # Better numerical stability for extreme K values
                log_Q = np.log(max(Q, 1e-15))
                log_K = np.log(max(self.K_eq_values[j], 1e-15))
                equations[j] = log_Q - log_K
            
            # Add penalty to equations if there are negative concentrations
            if penalty > 0:
                equations += penalty / self.n_reactions
            
            return equations
        
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
        
        # Iterative solver with improving initial guesses
        for iteration in range(max_iterations):
            try:
                if iteration == 0:
                    xi_guess = get_initial_guess()
                else:
                    # Use previous result as starting point, but perturb slightly
                    xi_guess = xi_final * (1 + 0.1 * np.random.normal(0, 1, self.n_reactions))
                
                # Solve the system
                if bounds_check and bounds:
                    bounds_array = np.array(bounds).T
                    result = least_squares(equilibrium_equations, xi_guess, 
                                         bounds=bounds_array,
                                         method='trf',
                                         ftol=1e-12,
                                         xtol=1e-12)
                    success = result.success
                    xi_final = result.x
                    fun_norm = np.linalg.norm(result.fun)
                    nfev = result.nfev

                else:
                    result = least_squares(equilibrium_equations, xi_guess, 
                                         method='lm',
                                         ftol=1e-12,
                                         xtol=1e-12)
                    success = result.success
                    xi_final = result.x
                    fun_norm = np.linalg.norm(result.fun)
                    nfev = result.nfev
                
                if verbose:
                    print(f"Iteration {iteration + 1}: Convergence: {success}, "
                          f"Residual norm: {fun_norm:.2e}, Function evals: {nfev}")
                
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
            for attempt in range(3):
                xi_final *= 0.9
                C_eq = C0 + self.stoich_matrix @ xi_final
                if np.all(C_eq >= -1e-15):
                    break
        
        # Clamp slightly negative values to zero
        C_eq = np.maximum(C_eq, 0.0)
        
        return C_eq, xi_final, fun_norm, success
    
    def get_equilibrium_dict(self, initial_concentrations, **kwargs):
        """
        Convenience method that returns equilibrium concentrations as a dict.
        """

        if len(self.fast_reactions) > 0:

            C_eq, xi, fun_norm, success = self.calculate_equilibrium_concentrations(
                initial_concentrations, **kwargs)
            
            result = {name: C_eq[i] for i, name in enumerate(self.species_names)}
            result['_extents'] = xi
            result['_success'] = success

        else:
            result = {name: initial_concentrations[name] for name in self.species_names}
            result['_extents'] = 'N/A'
            result['_success'] = True 
            fun_norm = 0
        
        return result, fun_norm

class Simulator:
    def __init__(self, T_C=25):
        self.T = T_C + 273.15
        self.T_C = T_C
        self.species = dict()
        self.reactions = dict()
    
    def add_species(self, name, energy=None, conc=None):
        '''
        Add a state with a given name and energy, in kcal/mol,
        and initial concentration, in mol/L
        
        '''
        conc = conc or 0.
        energy = energy or 0.

        assert conc > -1E-10

        self.species[name] = {
            "energy" : float(energy),
            "conc" : float(conc),
        }

        # update current conc dict
        self.current_conc_dict = {name: state["conc"] for name, state in self.species.items()}
        self.species_id_dict = {name : i for i, name in enumerate(self.species)}

    def add_reaction(self, reagents, products, ts_energy=None,
                     rate=None, throughput_target=None, enforced_K_eq=None):
        '''
        reagents and product: lists of strings 
        ts_energy: absoolute value relative to the
            whole PES, in kcal/mol (overrides rate)
        rate: forward reaction rate, in M^n * s^-1 (overridden by ts_energy)
        throughput_target: string with species name - will keep track of how
            much of this species is generated through this reaction
        enforced_K_eq: will consider the reaction instantaneous and always at
            equilibrium obeying the provided constant. 
        '''

        # check if we know all names
        for name in set(reagents + products):
            if name not in self.species.keys():
                raise NameError(f'State name \"{name}\" not defined.')
        
        # create a reaction hash string
        hash_name = " + ".join(reagents) + " -> " + " + ".join(products)

        if enforced_K_eq is not None:
            self.reactions[hash_name] = {
                "reagents" : reagents,
                "products" : products,
                "enforced_K_eq" : enforced_K_eq,
            }

        else:    
            if ts_energy is None:
                if rate is not None:
                    ts_energy = self.get_ts_energy(reagents, rate)
                else:
                    raise RuntimeError('Please provide either the reaction rate or absolute ts_energy')

            # calculate the activation energy relative to the reagents
            activation_energy = ts_energy - np.sum([self.species[name]["energy"] for name in reagents])
            inverse_act_energy = ts_energy - np.sum([self.species[name]["energy"] for name in products])


            # calculate reaction rates
            k_rate = get_eyring_k(activation_energy, self.T)
            k_inv = get_eyring_k(inverse_act_energy, self.T)

            # add the reaction to the self.reactions attribute
            self.reactions[hash_name] = {
                "reagents" : reagents,
                "products" : products,
                "activation_energy" : activation_energy,
                "inverse_act_energy" : inverse_act_energy,
                "smaller_act_energy" : activation_energy if activation_energy < inverse_act_energy else inverse_act_energy,

                "k_rate" : k_rate,
                "k_inv" : k_inv,
                "faster_k" : k_rate if k_rate > k_inv else k_inv,
                "K_eq" : self.get_K_eq(reagents, products),

            }

        if throughput_target:
            self.reactions[hash_name]["cumulative_throughput"] = 0.0
            self.reactions[hash_name]["throughput_target"] = throughput_target

        # label the reactions based on their relative rate:
        # the ones that are much faster than the others and 
        # are below a given activation energy are considered
        # always at equilibrium
        rates = [reaction["faster_k"] for reaction in self.reactions.values() if reaction.get("enforced_K_eq", None) is None]
        if rates:
            slowest_k_rate = sorted(rates)[0]
            for hash_name, reaction in self.reactions.items():

                if reaction.get("enforced_K_eq", None):
                    reaction["speed_rank"] = "enforced_K_eq"
                    reaction["description"] = f"--> \"{hash_name}\" will be enforced at the provided equilibrium constant (K_eq = {reaction['enforced_K_eq']:.2{'f' if abs(np.log10(reaction['enforced_K_eq']))<2 else 'e'}})."

                else:
                    rel_rate = reaction["faster_k"] / slowest_k_rate

                    if rel_rate > THR_REL_FAST_RXN:
                        reaction["speed_rank"] = "instantaneous"
                        reaction["description"] = (f"--> \"{hash_name}\" is very fast: will be considered always at equilibrium " + 
                            f"(Rel. rate = {rel_rate:.2{'f' if rel_rate<100 else 'e'}}, " +
                            f"K_eq = {reaction['K_eq']:.2{'f' if abs(np.log10(reaction['K_eq']))<2 else 'e'}}).")
                    

                    else:
                        reaction["speed_rank"] = "normal"
                        reaction["description"] = f"--> \"{hash_name}\" will be evolved step-by-step (Rel. rate = {rel_rate:.2{'f' if rel_rate<100 else 'e'}})."

        else:
            for hash_name, reaction in self.reactions.items():
                reaction["speed_rank"] = "enforced_K_eq"
                reaction["description"] = f"--> \"{hash_name}\" will be enforced at the provided equilibrium constant (K_eq = {reaction['enforced_K_eq']:.2{'f' if abs(np.log10(reaction['enforced_K_eq']))<2 else 'e'}})."

    def get_K_eq(self, reagents, products):
        dG = 0
        for product in products:
            dG += self.species[product]["energy"]
        for reagent in reagents:
            dG -= self.species[reagent]["energy"]
        return np.exp(-dG/(R*self.T))

    def run(
            self,
            time=1,
            t_units="s",
            dt_s=None,
            method="backwards_euler",
            max_equilib_iters=5,
        ):
        
        self.run_t_units = t_units
        self.method = method
        self.max_equilib_iters = max_equilib_iters

        print()
        for reaction in self.reactions.values():
            print(reaction["description"])
        
        self.multiplier = {
            "s":1,
            "m":60,
            "h":3600,
            "d":3600*24,
            }[t_units]

        if dt_s is None:
            dt_s = time * self.multiplier / MAX_AUTO_STEPS


        self.dt_s = dt_s
        
        iterations = int(np.ceil(time*self.multiplier/self.dt_s))
        print(f"\n-> Running simulation for {time} {t_units} with the {self.method} method ({self.dt_s:.1{'f' if self.dt_s>0.1 else 'e'}} s increments, {iterations} iterations)")
        plot_num_points = min(PLOT_NUM_POINTS, iterations)

        # equilibrate before the main loop
        self._equilibrate_instantaneous_reactions()
        
        # init results array and time list
        self.conc_data = [np.array(list(self.current_conc_dict.values()))]
        self.current_time_s = 0.0
        self.time_data = [0.0]

        # do a quick check on the stepsize relative to the initial quantity
        # but only for the non-instantaneous rxns
        # self._check_stepsize()

        t_start = perf_counter()
        for i in range(1,iterations+1):
            self.current_time_s = self.dt_s * i
            loadbar(i, iterations, prefix='Iterations ')

            # first, iteratively evolve all normal reactions
            for rxn in self.reactions.values():
                if rxn["speed_rank"] == "normal":
                    self._normal_reaction_step(rxn)

            # then, equilibrate all instantaneous reactions together
            self._equilibrate_instantaneous_reactions()

            # if it's time, collect a datapoint
            if i % int(iterations/plot_num_points) == 0:
                self._add_status_to_results()

            if min(self.current_conc_dict.values()) < -1E-6:
                s = f'-> Something blew up and we got a negative concentration. (< -1E-6 M, {i} iterations)'
                print(s)
                break

        self.time_data = np.array(self.time_data)

        print(f"\n--> Simulation complete ({time_to_string(perf_counter()-t_start)})")

    def _normal_reaction_step(self, rxn):
        '''
        Evolve normal (slow) reaction by one iteration.
        '''

        deltaC = self._get_deltaC_step(rxn)
        
        # Bckw Euler + mass balance
        for reagent in rxn["reagents"]:
            self.current_conc_dict[reagent] -= deltaC

        for product in rxn["products"]:
            self.current_conc_dict[product] += deltaC

        if "cumulative_throughput" in rxn:
            rxn["cumulative_throughput"] += deltaC * rxn["products"].count(rxn["throughput_target"])

    def _get_deltaC_step(self, rxn):
        '''
        Get the change in concentration that a single reagent
        in the given reaction will undergo in one time step.

        '''
        direct_delta = self.dt_s * rxn["k_rate"] * np.prod([self.current_conc_dict[reagent] for reagent in rxn["reagents"]])
        inverse_delta = self.dt_s * rxn["k_inv"] * np.prod([self.current_conc_dict[product] for product in rxn["products"]])
        delta = direct_delta - inverse_delta

        if self.method == "backwards_euler":
            return 1/(1-delta) - 1
        
        elif self.method == "euler":
            return delta

    def _equilibrate_instantaneous_reactions(self, **kwargs):

            if not hasattr(self, "equilibrium_solver"):
                self.equilibrium_solver = FastEquilibriumSolver(self.species.keys(), self.reactions)
            
            for _ in range(self.max_equilib_iters):
                # Solve for equilibrium
                new_conc_dict, fun_norm = self.equilibrium_solver.get_equilibrium_dict(self.current_conc_dict, **kwargs)
                
                for species in self.species.keys():
                    self.current_conc_dict[species] = new_conc_dict[species]

                if fun_norm < 1e-2 or fun_norm == np.inf:
                    break

    def _add_status_to_results(self):
        current_concs = np.array(list(self.current_conc_dict.values()))
        self.conc_data = np.concatenate((self.conc_data, [current_concs]))
        self.time_data.append(self.current_time_s)

    def show(self, species=None):
        '''
        species: iterable of strings of species to show,
            default will show all.
        '''

        species_to_plot = species or self.species

        x = np.array(self.time_data)/self.multiplier

        plt.figure()
        print("\nFinal Concentrations:")
        sum_of_final_concs = np.sum([concs[-1] for concs in self.conc_data.T])
        longest = max(len(name) for name in self.species.keys())

        for name, concs in zip(self.species.keys(), self.conc_data.T):
            if name in species_to_plot:
                plt.plot(x, concs, label=name)
                s = f"{name:{longest}s} : {concs[-1]:.2f} M"
                if self.species[name]["conc"] > 0:
                    final_percentage = concs[-1]/self.species[name]['conc']*100
                    s += f" ({final_percentage:.1f} % of initial conc., {100-final_percentage:.2f} % consumed)"

                else:
                    final_fraction = concs[-1]/sum_of_final_concs
                    s += f" ({final_fraction*100:.2f} % total molar fraction)"
                print(s)

        print()
        # sum_C0 = sum([species["conc"] for species in sim.species.values()])
        # print(f'Safety check: sum of C(start) = {sum_C0:.4f} M, sum of C(end) = {sum_of_final_concs:.4f} M\n')

        for hash_name, reaction in self.reactions.items():
            if "cumulative_throughput" in reaction:
                fraction = reaction["cumulative_throughput"]/self.current_conc_dict[reaction["throughput_target"]]
                print(f'Reaction \"{hash_name}\" throughput is {reaction["cumulative_throughput"]:.3f} M, {fraction*100:.2f} % of final \"{reaction["throughput_target"]}\" conc. ')

        plt.legend()
        plt.title(f'Concentrations over time (T={self.T_C} °C)')
        plt.xlabel(f"Time ({self.run_t_units})")
        plt.ylabel("Concentration (M)")
        plt.show()

    def get_ts_energy(self, reagents, rate):
        '''
        reagents: list of strings
        rate: forward reaction rate in M^n * s^-1
        '''
        activation_energy = -np.log(rate*H_PLANCK/K_BOLTZMANN/self.T) * (R*self.T)
        reagents_energy = np.sum([self.species[r]["energy"] for r in reagents])
        return activation_energy + reagents_energy

def get_eyring_k(activation_energy, T=298.15):
    '''
    Returns a rate constant in s^-1 given an
    activation energy in kcal/mol and a temperature.
    
    '''
    return K_BOLTZMANN/H_PLANCK*T*np.exp(-activation_energy/(R*T))

def loadbar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='#'):
    percent = ('{0:.' + str(decimals) + 'f}').format(100 * (iteration/float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()