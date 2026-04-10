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

    # check adherence to COPASI reference
    assert abs(sim.results["C"][-1] - 0.82783) < 1e-3


def test_stiff_pre_eq() -> None:
    """Test a more complex pre-equilibrium scheme."""
    sim = Simulator(T_C=139)

    # species present from the start
    sim.add_species("A", conc=1)
    sim.add_species("B", conc=1)
    sim.add_species("cat", conc=0.1, energy=1.8)

    # conc=0 at start
    sim.add_species("A.cat", energy=0.0)
    sim.add_species("C", energy=-3)
    sim.add_species("D", energy=-6)

    sim.add_reaction(["A", "cat"], ["A.cat"])
    sim.add_reaction(["A.cat", "B"], ["C", "cat"], ts_energy=23)
    sim.add_reaction(["C", "cat"], ["D", "cat"], ts_energy=28)

    sim.run(time=10, t_units="h")

    # check adherence to COPASI reference
    assert abs(sim.results["D"][-1] - 0.93726) < 1e-3


def test_complex_pre_eq() -> None:
    """Test a complex pre-equilibrium involving many equations."""
    sim = Simulator(T_C=25)
    EH_TO_KCAL = 627.5096080305927
    C = 0.2

    sim.add_species("SM", conc=1 * C, energy=-2318.10145867 * EH_TO_KCAL)
    sim.add_species("PNP", conc=1.1 * C, energy=-511.96570431 * EH_TO_KCAL)
    sim.add_species("cat", conc=0.05 * C, energy=-2414.35416366 * EH_TO_KCAL)
    sim.add_species("btz", energy=-797.91082415 * EH_TO_KCAL)
    sim.add_species("prod_S", energy=-2032.17956641 * EH_TO_KCAL)
    sim.add_species("prod_R", energy=-2032.17956641 * EH_TO_KCAL)

    # non-covalent dimers
    sim.add_species("cat.SM", energy=-4732.466223 * EH_TO_KCAL)
    sim.add_reaction("SM cat", "cat.SM")

    sim.add_species("SM.PNP", energy=-2830.06952852 * EH_TO_KCAL)
    sim.add_reaction("SM PNP", "SM.PNP")

    sim.add_species("cat.PNP", energy=-2926.31837359 * EH_TO_KCAL)
    sim.add_reaction("cat PNP", "cat.PNP")

    sim.add_species("cat.cat", energy=-4828.72621977 * EH_TO_KCAL)
    sim.add_reaction("cat cat", "cat.cat")

    sim.add_species("cat.btz", energy=-3212.27880121 * EH_TO_KCAL)
    sim.add_reaction("cat btz", "cat.btz")

    # non-covalent pre-reaction complexes
    sim.add_species("cat.SM.PNP", energy=-5244.4428431 * EH_TO_KCAL)
    sim.add_reaction("cat.SM PNP", "cat.SM.PNP")

    sim.add_species("cat.cat.SM.PNP", energy=-7658.81181265 * EH_TO_KCAL)
    sim.add_reaction("cat cat.SM.PNP", "cat.cat.SM.PNP")

    # this is estimated assuming the same binding energy for prods as for SM
    sim.add_species("cat.prod_S", energy=-2790249.28805103)
    sim.add_species("cat.prod_R", energy=-2790249.28805103)
    sim.add_reaction("cat prod_S", "cat.prod_S")
    sim.add_reaction("cat prod_R", "cat.prod_R")

    # genuine reactions

    # monomolecular in catalyst
    sim.add_reaction(
        "cat.SM.PNP",
        "cat.prod_S btz",
        ts_energy=-5244.42284508 * EH_TO_KCAL,
        force_slow=True,
        throughput_tgt="prod_S",
    )
    sim.add_reaction(
        "cat.SM.PNP",
        "cat.prod_R btz",
        ts_energy=-5244.42874492 * EH_TO_KCAL,
        force_slow=True,
        throughput_tgt="prod_R",
    )

    # bimolecular in catalyst
    sim.add_reaction(
        "cat.cat.SM.PNP",
        "cat.prod_S cat.btz",
        ts_energy=-7658.79719711 * EH_TO_KCAL,
        force_slow=True,
        throughput_tgt="prod_S",
    )
    sim.add_reaction(
        "cat.cat.SM.PNP",
        "cat.prod_R cat.btz",
        ts_energy=-7658.80165587 * EH_TO_KCAL,
        force_slow=True,
        throughput_tgt="prod_R",
    )

    # Run for zero seconds to equilibrate all fast reactions
    sim.run(time=0)

    # check equilibration (from a verified run reference)
    assert abs(sim.results["SM.PNP"][-1] - 0.106364) < 1e-4
    assert abs(sim.results["SM"][-1] - 0.08364536) < 1e-4
    assert abs(sim.results["PNP"][-1] - 0.10364627) < 1e-4

    # run for 0.05 s
    sim.run(time=0.05)

    # check final concentration of product (COPASI reference)
    assert abs(sim.results["prod_R"][-1] - 0.199258) < 1e-4
