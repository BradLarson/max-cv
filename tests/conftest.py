from pathlib import Path
import pytest
from max.driver import CPU
from max.engine import InferenceSession

@pytest.fixture(scope="session")
def session() -> InferenceSession:
    device = CPU()
    # TODO: Pull this from environment variable.
    path = Path("operations.mojopkg")
    return InferenceSession(
        devices=[device],
        custom_extensions=path,
    )
