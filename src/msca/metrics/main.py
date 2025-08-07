from enum import StrEnum, auto
import numpy as np
import pandas as pd

from sklearn import metrics


class Metric(StrEnum):
    """
    A metric enum that can be instantiated with string names and supports both error and skill calculations.

    Examples
    --------
    >>> # Simple usage
    >>> metric = Metric("mean_absolute_error")
    >>> score = metric.eval(df, "obs", "pred", "weights")
    >>> # Grouped calculation
    >>> grouped_scores = metric.eval(
    ...     df, "obs", "pred", "weights", groupby=["region"]
    ... )
    >>>
    >>> # Skill calculation
    >>> skill_score = metric.eval(
    ...     df, "obs", "pred_ref", "weights", pred_alt="pred_alt"
    ... )
    """

    MEAN_ABSOLUTE_ERROR = auto()
    MEAN_ABSOLUTE_PERCENTAGE_ERROR = auto()
    MEAN_SQUARED_ERROR = auto()
    MEDIAN_ABSOLUTE_ERROR = auto()
    ROOT_MEAN_SQUARED_ERROR = auto()

    def eval(
        self,
        data: pd.DataFrame,
        obs: str,
        pred_ref: str,
        weights: str,
        pred_alt: str | None = None,
        groupby: list[str] | None = None,
    ) -> float | pd.DataFrame:
        """
        Evaluate the error metric or skill score on the provided data.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame containing all required columns
        obs : str
            Column name for observed/actual values
        pred_ref : str
            Column name for reference predicted values (used as "pred" for error calculation)
        weights : str
            Column name for sample weights
        pred_alt : str, optional
            Column name for alternative predicted values. If provided, calculates skill score
            comparing pred_alt vs pred_ref. If None, calculates error metric for pred_ref.
        groupby : list[str], optional
            Column names to group by for grouped calculations

        Returns
        -------
        Union[float, pd.DataFrame]
            Single metric/skill value if no groupby, DataFrame with grouped results if groupby specified
        """
        if groupby is not None:
            return self._eval_grouped(
                data, obs, pred_ref, weights, groupby, pred_alt
            )

        if pred_alt is None:
            # Error calculation mode
            obs_values = data[obs].to_numpy()
            pred_values = data[pred_ref].to_numpy()
            weight_values = data[weights].to_numpy()

            # Calculate single metric value
            return self._eval_single(obs_values, pred_values, weight_values)
        else:
            # Skill calculation mode
            obs_values = data[obs].to_numpy()
            pred_ref_values = data[pred_ref].to_numpy()
            pred_alt_values = data[pred_alt].to_numpy()
            weight_values = data[weights].to_numpy()

            ref_score = self._eval_single(
                obs_values, pred_ref_values, weight_values
            )
            alt_score = self._eval_single(
                obs_values, pred_alt_values, weight_values
            )

            if ref_score == 0:
                raise ZeroDivisionError(
                    "Reference score is zero, cannot calculate skill score"
                )

            return 1.0 - (alt_score / ref_score)

    def _eval_single(
        self,
        obs_values: np.ndarray,
        pred_values: np.ndarray,
        weight_values: np.ndarray,
    ) -> float:
        """
        Calculate metric for single set of values.

        Parameters
        ----------
        obs_values : np.ndarray
            Observed values
        pred_values : np.ndarray
            Predicted values
        weight_values : np.ndarray
            Sample weights

        Returns
        -------
        float
            Calculated metric value
        """
        match self:
            case Metric.ROOT_MEAN_SQUARED_ERROR:
                mse_value = metrics.mean_squared_error(
                    y_true=obs_values,
                    y_pred=pred_values,
                    sample_weight=weight_values,
                )
                return np.sqrt(mse_value)
            case Metric.MEAN_ABSOLUTE_ERROR:
                return metrics.mean_absolute_error(
                    y_true=obs_values,
                    y_pred=pred_values,
                    sample_weight=weight_values,
                )
            case Metric.MEAN_SQUARED_ERROR:
                return metrics.mean_squared_error(
                    y_true=obs_values,
                    y_pred=pred_values,
                    sample_weight=weight_values,
                )
            case Metric.MEAN_ABSOLUTE_PERCENTAGE_ERROR:
                return metrics.mean_absolute_percentage_error(
                    y_true=obs_values,
                    y_pred=pred_values,
                    sample_weight=weight_values,
                )
            case Metric.MEDIAN_ABSOLUTE_ERROR:
                return metrics.median_absolute_error(
                    y_true=obs_values,
                    y_pred=pred_values,
                    sample_weight=weight_values,
                )
            case _:
                raise ValueError(f"Unsupported metric type: {self}")

    def _eval_grouped(
        self,
        data: pd.DataFrame,
        obs: str,
        pred_ref: str,
        weights: str,
        groupby: list[str],
        pred_alt: str | None = None,
    ) -> pd.DataFrame:
        """
        Calculate error metrics or skill scores for each group in the DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame
        obs : str
            Observed values column name
        pred_ref : str
            Reference predicted values column name
        weights : str
            Weights column name
        groupby : list[str]
            Grouping column names
        pred_alt : str, optional
            Alternative predicted values column name. If provided, calculates skill scores.

        Returns
        -------
        pd.DataFrame
            DataFrame with groupby columns and calculated metric/skill column
        """
        df = data.copy()

        if pred_alt is None:
            result_column_name = f"{pred_ref}_{self.value}"
        else:
            result_column_name = f"{pred_alt}_{self.value}_skill"

        grouped_results = (
            df.groupby(groupby)
            .apply(
                self._calculate_group_result,
                obs,
                pred_ref,
                weights,
                pred_alt,
                include_groups=False,
            )
            .reset_index()
        )
        grouped_results = grouped_results.rename(
            columns={grouped_results.columns[-1]: result_column_name}
        )

        return grouped_results

    def _calculate_group_result(
        self,
        group: pd.DataFrame,
        obs: str,
        pred_ref: str,
        weights: str,
        pred_alt: str | None = None,
    ) -> float:
        """Calculate metric or skill score for a single group."""
        obs_values = group[obs].to_numpy()
        weight_values = group[weights].to_numpy()

        if pred_alt is None:
            # Error calculation mode
            pred_values = group[pred_ref].to_numpy()
            return self._eval_single(obs_values, pred_values, weight_values)
        else:
            # Skill calculation mode
            pred_ref_values = group[pred_ref].to_numpy()
            pred_alt_values = group[pred_alt].to_numpy()

            ref_score = self._eval_single(
                obs_values, pred_ref_values, weight_values
            )
            alt_score = self._eval_single(
                obs_values, pred_alt_values, weight_values
            )

            if ref_score == 0:
                raise ZeroDivisionError(
                    "Reference score is zero, cannot calculate skill score"
                )

            return 1.0 - (alt_score / ref_score)
