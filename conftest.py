from pathlib import Path

import pytest

pytest_plugins = ['llckbdm._tests.fixtures']


@pytest.fixture(scope="session")
def data_path():
    root_path = Path(__file__).parents[0]
    data_path = f'{root_path}/data'

    return data_path
