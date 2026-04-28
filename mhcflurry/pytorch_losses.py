"""
PyTorch loss functions for mhcflurry.

Supports inequality constraints where training data includes (=), (<), and (>)
relationships. For inequality constraints, penalization is applied only when
predictions violate the constraint.
"""
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F


class MSEWithInequalities(nn.Module):
    """
    MSE loss with inequality support.

    y_true is encoded as follows:
      - [0, 1]: equality constraint, standard MSE
      - [2, 3]: greater-than constraint (value = y_true - 2), penalize if pred < value
      - [4, 5]: less-than constraint (value = y_true - 4), penalize if pred > value
    """
    supports_inequalities = True
    supports_multiple_outputs = False

    @staticmethod
    def encode_y(y, inequalities=None):
        """
        Encode targets with inequality information.

        Parameters
        ----------
        y : array-like
            Target values in [0, 1]
        inequalities : array-like of str, optional
            One of "=", ">", "<" for each target

        Returns
        -------
        numpy.ndarray
        """
        y = numpy.array(y, dtype=numpy.float32)
        if numpy.isnan(y).any():
            raise ValueError("y contains NaN")
        if (y < 0).any() or (y > 1).any():
            raise ValueError("Targets must be in [0, 1] for MSEWithInequalities")
        if inequalities is None:
            return y
        if len(inequalities) != len(y):
            raise ValueError("inequalities must have same length as y")
        for ineq in inequalities:
            if ineq not in {"=", ">", "<"}:
                raise ValueError("Inequalities must be one of '=', '>', '<'")
        offsets = numpy.array([
            {'=': 0, '>': 2, '<': 4}[ineq] for ineq in inequalities
        ], dtype=numpy.float32)
        encoded = y + offsets
        assert not numpy.isnan(encoded).any()
        return encoded

    def forward(self, y_pred, y_true, sample_weights=None):
        """
        Compute loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predictions, shape (batch,) or (batch, 1)
        y_true : torch.Tensor
            Encoded targets, shape (batch,) or (batch, 1)

        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        pass


class MSEWithInequalitiesAndMultipleOutputs(nn.Module):
    """
    MSE loss with inequality and multiple output support.

    Extends MSEWithInequalities by encoding the output index into the target:
    encoded_target = inequality_encoded_value + output_index * 10
    """
    supports_inequalities = True
    supports_multiple_outputs = True

    @staticmethod
    def encode_y(y, inequalities=None, output_indices=None):
        """
        Encode targets with inequality and output index information.

        Parameters
        ----------
        y : array-like
            Target values in [0, 1]
        inequalities : array-like of str, optional
            One of "=", ">", "<" for each target
        output_indices : array-like of int, optional
            Output index for each target

        Returns
        -------
        numpy.ndarray
        """
        encoded = MSEWithInequalities.encode_y(y, inequalities)
        if output_indices is not None:
            output_indices = numpy.array(output_indices)
            if output_indices.shape != (len(encoded),):
                raise ValueError(
                    "Expected output_indices to have shape %s not %s"
                    % ((len(encoded),), output_indices.shape)
                )
            if (output_indices < 0).any():
                raise ValueError("Invalid output indices: %s" % output_indices)
            encoded = encoded + output_indices.astype(numpy.float32) * 10
        return encoded

    def forward(self, y_pred, y_true, sample_weights=None):
        """
        Compute loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predictions, shape (batch, num_outputs)
        y_true : torch.Tensor
            Encoded targets, shape (batch,) or (batch, 1)

        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        pass


class MultiallelicMassSpecLoss(nn.Module):
    """
    Loss function for multiallelic mass spectrometry data.

    For each (hit, decoy) pair, penalizes when any decoy allele prediction
    exceeds the best hit allele prediction by more than delta.

    y_true encoding:
      - 1.0: hit (positive)
      - 0.0: decoy (negative)
      - -1.0: ignored
    """
    supports_inequalities = True
    supports_multiple_outputs = False

    def __init__(self, delta=0.2, multiplier=1.0):
        super(MultiallelicMassSpecLoss, self).__init__()
        self.delta = delta
        self.multiplier = multiplier

    @staticmethod
    def encode_y(y):
        """Encode y (no-op for this loss)."""
        y = numpy.array(y, dtype=numpy.float32)
        assert numpy.isin(y, [-1.0, 0.0, 1.0]).all()
        return y

    def forward(self, y_pred, y_true, sample_weights=None):
        """
        Compute loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predictions, shape (batch, num_alleles)
        y_true : torch.Tensor
            Labels, shape (batch,) or (batch, 1)

        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        pass


class StandardLoss(nn.Module):
    """
    Wrapper for standard PyTorch loss functions (MSE, MAE, etc).
    """
    supports_inequalities = False
    supports_multiple_outputs = False

    def __init__(self, loss_name="mse"):
        super(StandardLoss, self).__init__()
        self.loss_name = loss_name
        if loss_name == "mse":
            self._loss_fn = nn.MSELoss()
        elif loss_name == "mae":
            self._loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unknown standard loss: {loss_name}")

    @staticmethod
    def encode_y(y):
        """Encode y (simple cast to float32)."""
        return numpy.array(y, dtype=numpy.float32)

    def forward(self, y_pred, y_true, sample_weights=None):
        """
        Compute loss.

        Parameters
        ----------
        y_pred : torch.Tensor
        y_true : torch.Tensor
        sample_weights : torch.Tensor | None
            Optional per-example weights.

        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        pass


# Registry of custom losses
_CUSTOM_LOSSES = {
    'mse_with_inequalities': MSEWithInequalities,
    'mse_with_inequalities_and_multiple_outputs': MSEWithInequalitiesAndMultipleOutputs,
    'multiallelic_mass_spec_loss': MultiallelicMassSpecLoss,
}


def get_pytorch_loss(name):
    """
    Get a PyTorch loss object by name.

    Parameters
    ----------
    name : str
        Loss name. Prefix with "custom:" for custom losses,
        otherwise a standard loss name like "mse".

    Returns
    -------
    nn.Module
        Loss module with encode_y, supports_inequalities,
        and supports_multiple_outputs attributes.
    """
    if name.startswith("custom:"):
        custom_name = name.replace("custom:", "")
        if custom_name not in _CUSTOM_LOSSES:
            raise ValueError(
                f"No such custom loss: {name}. "
                f"Supported: {', '.join('custom:' + k for k in _CUSTOM_LOSSES)}"
            )
        return _CUSTOM_LOSSES[custom_name]()
    return StandardLoss(name)
