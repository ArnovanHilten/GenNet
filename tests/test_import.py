def test_import_gennet_utils():
    import GenNet_utils
    from GenNet_utils import Convert, Topology, Interpret

    assert GenNet_utils is not None
    assert Convert.convert is not None
    assert Topology.topology is not None
    assert Interpret.interpret is not None
