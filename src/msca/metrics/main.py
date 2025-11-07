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

        return self._eval_single(data, obs, pred, weights).iloc[0]

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
            ref_scores = self._eval_grouped(
                data=data,
                obs=obs,
                pred=pred_ref,
                weights=weights,
                groupby=groupby,
            )
            alt_scores = self._eval_grouped(
                data=data,
                obs=obs,
                pred=pred_alt,
                weights=weights,
                groupby=groupby,
            )

            ref_score_col = self._get_metric_column_name(pred_ref)
            alt_score_col = self._get_metric_column_name(pred_alt)
            result_column_name = f"{alt_score_col}_skill"

            if (ref_scores[ref_score_col] == 0).any():
                zero_ref_groups = ref_scores[ref_scores[ref_score_col] == 0][
                    groupby
                ].to_dict("records")
                raise ZeroDivisionError(
                    f"Reference score is zero for groups {zero_ref_groups}, cannot calculate skill score"
                )

            grouped_results = ref_scores.copy()
            grouped_results[result_column_name] = 1.0 - (
                alt_scores[alt_score_col] / ref_scores[ref_score_col]
            )

            grouped_results = grouped_results.drop(columns=[ref_score_col])

            return grouped_results

        ref_score = self._eval_single(data, obs, pred_ref, weights).iloc[0]
        alt_score = self._eval_single(data, obs, pred_alt, weights).iloc[0]

        if ref_score == 0:
            raise ZeroDivisionError(
                "Reference score is zero, cannot calculate skill score"
            )

        return 1.0 - (alt_score / ref_score)

    def _eval_single(
        self, data: pd.DataFrame, obs: str, pred: str, weights: str
    ) -> pd.Series:
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
        pd.Series
            Series with named metric value: f"{pred}_{self.value}"
        """
        if data.empty:
            raise ValueError(
                "Input dataframe is empty, at least one row is required "
                "to calculate single metrics."
            )

        obs_values = data[obs].to_numpy()
        pred_values = data[pred].to_numpy()
        weight_values = data[weights].to_numpy()

        column_name = self._get_metric_column_name(pred)

        match self:
            case Metric.ROOT_MEAN_SQUARED_ERROR:
                mse_value = metrics.mean_squared_error(
                    y_true=obs_values,
                    y_pred=pred_values,
                    sample_weight=weight_values,
                )
                result = np.sqrt(mse_value)
            case Metric.MEAN_ABSOLUTE_ERROR:
                result = metrics.mean_absolute_error(
                    y_true=obs_values,
                    y_pred=pred_values,
                    sample_weight=weight_values,
                )
            case Metric.MEAN_SQUARED_ERROR:
                result = metrics.mean_squared_error(
                    y_true=obs_values,
                    y_pred=pred_values,
                    sample_weight=weight_values,
                )
            case Metric.MEAN_ABSOLUTE_PERCENTAGE_ERROR:
                result = metrics.mean_absolute_percentage_error(
                    y_true=obs_values,
                    y_pred=pred_values,
                    sample_weight=weight_values,
                )
            case Metric.MEDIAN_ABSOLUTE_ERROR:
                result = metrics.median_absolute_error(
                    y_true=obs_values,
                    y_pred=pred_values,
                    sample_weight=weight_values,
                )
            case _:
                raise ValueError(f"Unsupported metric type: {self}")

        return pd.Series({column_name: result})

    def _eval_grouped(
        self,
        data: pd.DataFrame,
        obs: str,
        pred: str,
        weights: str,
        groupby: list[str],
    ) -> pd.DataFrame:
        """
        Calculate error metrics or skill scores for each group in the DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame
        obs : str
            Observed values column name
        pred : str
            Predicted values column name
        weights : str
            Weights column name
        groupby : list[str]
            Grouping column names

        Returns
        -------
        pd.DataFrame
            DataFrame with groupby columns and calculated metric/skill column
        """
        if data.empty:
            raise ValueError(
                "Input dataframe is empty, at least one row is required "
                "to calculate metrics by group."
            )

        df = data.copy()
        grouped_results = (
            df.groupby(groupby)
            .apply(
                self._eval_single,
                obs,
                pred,
                weights,
            )
            .reset_index()
        )

        return grouped_results

    def _get_metric_column_name(self, pred: str) -> str:
        """
        Template the metric column name from the predicted values
        column name and the Metric's string representation.

        Parameters
        ----------
        pred : str
            Predicted values column name

        Returns
        -------
        str
            Name of the metric value column, formatted as "{pred}_{self.value}"
        """
        return f"{pred}_{self.value}"
