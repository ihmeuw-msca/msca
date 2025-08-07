from enum import StrEnum, auto
import numpy as np
import pandas as pd

from sklearn import metrics


class Metric(StrEnum):
    """
    A metric enum that can be instantiated with string names and supports both error and skill calculations.

    Examples
    --------
    >>> # Simple error metric calculation
    >>> metric = Metric("mean_absolute_error")
    >>> score = metric.eval(df, "obs", "pred", "weights")
    >>> # Grouped error calculation
    >>> grouped_scores = metric.eval(
    ...     df, "obs", "pred", "weights", groupby=["region"]
    ... )
    >>>
    >>> # Skill calculation
    >>> skill_score = metric.eval_skill(
    ...     df, "obs", "pred_alt", "pred_ref", "weights"
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
        pred: str,
        weights: str,
        groupby: list[str] | None = None,
    ) -> float | pd.DataFrame:
        """
        Evaluate the error metric on the provided data.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame containing all required columns
        obs : str
            Column name for observed/actual values
        pred : str
            Column name for predicted values
        weights : str
            Column name for sample weights
        groupby : list[str], optional
            Column names to group by for grouped calculations

        Returns
        -------
        Union[float, pd.DataFrame]
            Single metric value if no groupby, DataFrame with grouped results if groupby specified
        """
        if groupby is not None:
            return self._eval_grouped(data, obs, pred, weights, groupby)

        return self._eval_single(data, obs, pred, weights)

    def eval_skill(
        self,
        data: pd.DataFrame,
        obs: str,
        pred_alt: str,
        pred_ref: str,
        weights: str,
        groupby: list[str] | None = None,
    ) -> float | pd.DataFrame:
        """
        Calculate skill score by comparing pred_alt performance against pred_ref.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame containing all required columns
        obs : str
            Column name for observed/actual values
        pred_alt : str
            Column name for alternative predicted values to evaluate
        pred_ref : str
            Column name for reference predicted values to compare against
        weights : str
            Column name for sample weights
        groupby : list[str], optional
            Column names to group by for grouped calculations

        Returns
        -------
        Union[float, pd.DataFrame]
            Single skill score if no groupby, DataFrame with grouped skill scores if groupby specified
        """
        if groupby is not None:
            return self._eval_grouped(
                data, obs, pred_ref, weights, groupby, pred_alt
            )

        ref_score = self._eval_single(data, obs, pred_ref, weights)
        alt_score = self._eval_single(data, obs, pred_alt, weights)

        if ref_score == 0:
            raise ZeroDivisionError(
                "Reference score is zero, cannot calculate skill score"
            )
        if alt_score == 0:
            raise ValueError(
                "Alternative score is zero, skill score calculation may be unreliable"
            )

        return 1.0 - (alt_score / ref_score)

    def _eval_single(
        self, data: pd.DataFrame, obs: str, pred: str, weights: str
    ) -> float:
        """
        Calculate metric for single DataFrame or group.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the data
        obs : str
            Column name for observed values
        pred : str
            Column name for predicted values
        weights : str
            Column name for sample weights

        Returns
        -------
        float
            Calculated metric value
        """
        obs_values = data[obs].to_numpy()
        pred_values = data[pred].to_numpy()
        weight_values = data[weights].to_numpy()
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
            Predicted values column name (pred_ref for skill calculations)
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
            # Error calculation
            result_column_name = f"{pred_ref}_{self.value}"
            grouped_results = (
                df.groupby(groupby)
                .apply(
                    self._eval_single,
                    obs,
                    pred_ref,
                    weights,
                    include_groups=False,
                )
                .reset_index()
            )
        else:
            # Skill calculation
            result_column_name = f"{pred_alt}_{self.value}_skill"

            ref_scores = (
                df.groupby(groupby)
                .apply(
                    self._eval_single,
                    obs,
                    pred_ref,
                    weights,
                    include_groups=False,
                )
                .reset_index()
            )
            alt_scores = (
                df.groupby(groupby)
                .apply(
                    self._eval_single,
                    obs,
                    pred_alt,
                    weights,
                    include_groups=False,
                )
                .reset_index()
            )

            if (ref_scores.iloc[:, -1] == 0).any():
                zero_ref_groups = ref_scores[ref_scores.iloc[:, -1] == 0][groupby].to_dict('records')
                raise ZeroDivisionError(
                    f"Reference score is zero for groups {zero_ref_groups}, cannot calculate skill score"
                )
            if (alt_scores.iloc[:, -1] == 0).any():
                zero_alt_groups = alt_scores[alt_scores.iloc[:, -1] == 0][groupby].to_dict('records')
                raise ValueError(
                    f"Alternative score is zero for groups {zero_alt_groups}, skill score calculation may be unreliable"
                )

            grouped_results = ref_scores.copy()
            grouped_results.iloc[:, -1] = 1.0 - (
                alt_scores.iloc[:, -1] / ref_scores.iloc[:, -1]
            )
        grouped_results = grouped_results.rename(
            columns={grouped_results.columns[-1]: result_column_name}
        )

        return grouped_results
