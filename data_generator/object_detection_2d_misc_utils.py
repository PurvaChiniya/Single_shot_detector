
from __future__ import division
import numpy as np

def apply_inverse_transforms(y_pred_decoded, inverse_transforms):
    

    if isinstance(y_pred_decoded, list):

        y_pred_decoded_inv = []

        for i in range(len(y_pred_decoded)):
            y_pred_decoded_inv.append(np.copy(y_pred_decoded[i]))
            if y_pred_decoded_inv[i].size > 0: # If there are any predictions for this batch item.
                for inverter in inverse_transforms[i]:
                    if not (inverter is None):
                        y_pred_decoded_inv[i] = inverter(y_pred_decoded_inv[i])

    elif isinstance(y_pred_decoded, np.ndarray):

        y_pred_decoded_inv = np.copy(y_pred_decoded)

        for i in range(len(y_pred_decoded)):
            if y_pred_decoded_inv[i].size > 0: # If there are any predictions for this batch item.
                for inverter in inverse_transforms[i]:
                    if not (inverter is None):
                        y_pred_decoded_inv[i] = inverter(y_pred_decoded_inv[i])

    else:
        raise ValueError("`y_pred_decoded` must be either a list or a Numpy array.")

    return y_pred_decoded_inv
