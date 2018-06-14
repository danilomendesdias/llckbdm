import logging

import attr
import numpy as np

from llckbdm.metrics import calculate_freq_domain_rmse
from llckbdm.sampling import sample_kbdm

logger = logging.getLogger(__name__)


@attr.s
class MinRmseKbdmResult:
    line_list = attr.ib()
    min_rmse = attr.ib()
    min_index = attr.ib()
    samples = attr.ib()
    rmses_list = attr.ib()


def min_rmse_kbdm(data, dwell, m_range=None, l=None, samples=None):
    if samples is None:
        samples, _ = sample_kbdm(
            data=data,
            dwell=dwell,
            m_range=m_range,
            l=l,
            q=0,
            p=1,
            filter_invalid_features=True
        )

    rmses = []

    for i, line_list in enumerate(samples):
        if len(line_list) > 0:
            rmse = calculate_freq_domain_rmse(data=data, params_est=line_list, dwell=dwell)
        else:
            rmse = np.inf

        rmses.append(rmse)
        logger.debug('RMSE for sample #%d: %f', i, rmse)

    if len(rmses) > 0:
        min_index = np.argmin(rmses)
        min_rmse = rmses[min_index]
        min_rmse_params_est = samples[min_index]

        return MinRmseKbdmResult(
            line_list=min_rmse_params_est,
            min_rmse=min_rmse,
            min_index=min_index,
            samples=samples,
            rmses_list=rmses
        )
    return None
