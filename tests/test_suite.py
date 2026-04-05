"""
Tests for the python_kinetic_simulator package.

All hardcoded numerical values coming from
equivalent simulations run with COPASI 4.45.
"""

from python_kinetic_simulator.simulator import Simulator

def test_simple() -> None:
    """Test a simple kinetic scheme."""
    sim = Simulator()
    sim.add_species("A", conc=1)
    sim.add_species("B", energy=-2)
    sim.add_species("C", energy=-30)
    sim.add_reaction(["A"], ["B", "B"], ts_energy=23.0)
    sim.add_reaction(["B"], ["C"], rate=5e-4)
    sim.run(time=10, t_units="h")
    assert abs(sim.results["C"][-1] - 1.88208) < 1e-3

def test_pre_eq() -> None:
    """Test a simple pre-equilibrium scheme."""
    sim = Simulator()
    sim.add_species("A", conc=1)
    sim.add_species("B", energy=0.2)
    sim.add_species("C", energy=-30)
    
    sim.add_reaction(["A"], ["B"], ts_energy=1)
    sim.add_reaction(["B"], ["C"], ts_energy=23)

    sim.run(time=10, t_units="h")

    assert abs(sim.results["C"][-1] - 0.82783) < 1e-3

def test_stiff_pre_eq() -> None:
    """Test a more complex pre-equilibrium scheme."""
    sim = Simulator(T_C=139)

    # species present from the start
    sim.add_species("A", conc= 1)
    sim.add_species("B", conc= 1)
    sim.add_species("cat", conc= 0.1, energy=1.8)

    # conc=0 at start
    sim.add_species("A.cat", energy= 0.0)
    sim.add_species("C", energy= -3)
    sim.add_species("D", energy= -6)

    sim.add_reaction(["A", "cat"], ["A.cat"], ts_energy=1)
    sim.add_reaction(["A.cat", "B"], ["C", "cat"], ts_energy=23)
    sim.add_reaction(["C", "cat"], ["D", "cat"], ts_energy=28)

    sim.run(time=10, t_units="h")

    assert abs(sim.results["D"][-1] - .93726) < 1e-3