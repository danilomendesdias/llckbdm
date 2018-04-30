import logging

import numpy as np

from llckbdm.metrics import calculate_freq_domain_rmse
from llckbdm.sampling import sample_kbdm

logger = logging.getLogger(__name__)


def min_rmse_kbdm(data, dwell, m_range, l, samples=None):
    if samples is None:
        samples, _ = sample_kbdm(
            data=data,
            dwell=dwell,
            m_range=m_range,
            l=l,
            q=0,
            p=1,
            gep_solver='svd',
            filter_invalid_features=True
        )

    rmses = []

    for i, line_list in enumerate(samples):
        rmse = calculate_freq_domain_rmse(data=data, params_est=line_list, dwell=dwell)

        rmses.append(rmse)
        logger.debug(f'RMSE for sample #{i}')

    min_index = np.argmin(rmses)
    min_rmse = rmses[min_index]
    min_rmse_params_est = samples[min_index]

    return min_rmse_params_est, min_rmse, min_index, samples
