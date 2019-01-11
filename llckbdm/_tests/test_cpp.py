from llckbdm.bindings import kbdm


def test_hello():
    assert kbdm('world') == 'Hello, world'