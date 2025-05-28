from pathlib import Path
import pytest
from max.driver import CPU
from max.engine import InferenceSession

@pytest.fixture(scope="session")
def session() -> InferenceSession:
    device = CPU()
    return InferenceSession(
        devices=[device],
    )
