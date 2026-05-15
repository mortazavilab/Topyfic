import matplotlib
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData


matplotlib.use("Agg")


@pytest.fixture()
def synthetic_adata():
    counts = np.array(
        [
            [8, 1, 0, 3],
            [7, 1, 1, 2],
            [6, 2, 0, 2],
            [1, 7, 3, 0],
            [0, 8, 2, 1],
            [1, 6, 4, 0],
        ],
        dtype=np.float64,
    )

    obs = pd.DataFrame(index=[f"cell_{index}" for index in range(counts.shape[0])])
    var = pd.DataFrame(index=[f"gene_{index}" for index in range(counts.shape[1])])

    return AnnData(counts, obs=obs, var=var)