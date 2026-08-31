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
        winsorize: tuple[float, float] | None = None,
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
        winsorize : tuple[float, float], optional
            Lower/upper quantiles in [0, 1] for clipping per-observation error
            contributions before aggregating, computed per group; None uses
            the standard metric

        Returns
        -------
        Union[float, pd.DataFrame]
            Single metric value if no groupby, DataFrame with grouped results if groupby specified
        """
        if winsorize is not None:
            self._validate_winsorize(winsorize)

        if groupby is not None:
            return self._eval_grouped(
                data, obs, pred, weights, groupby=groupby, winsorize=winsorize
            )

        return self._eval_single(
            data, obs, pred, weights, winsorize=winsorize
        ).iloc[0]

    def eval_skill(
        self,
        data: pd.DataFrame,
        obs: str,
        pred_alt: str,
        pred_ref: str,
        weights: str,
        groupby: list[str] | None = None,
        winsorize: tuple[float, float] | None = None,
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
        winsorize : tuple[float, float], optional
            Lower/upper quantiles in [0, 1] for clipping the per-group skill
            scores; the underlying error scores stay unwinsorized. Requires
            ``groupby``. None uses the standard skill score

        Returns
        -------
        Union[float, pd.DataFrame]
            Single skill score if no groupby, DataFrame with grouped skill scores if groupby specified
        """
        if winsorize is not None:
            self._validate_winsorize(winsorize)
            if groupby is None:
                raise ValueError(
                    "winsorize on skill scores requires groupby; a single "
                    "skill value has no distribution to winsorize."
                )

        if groupby is not None:
            # Clip the per-group skill scores below, not the ref/alt error
            # scores: winsorizing those independently drives skill negative.
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
            skill_scores = 1.0 - (
                alt_scores[alt_score_col] / ref_scores[ref_score_col]
            )

            if winsorize is not None:
                lower_q, upper_q = winsorize
                skill_scores = skill_scores.clip(
                    lower=skill_scores.quantile(lower_q),
                    upper=skill_scores.quantile(upper_q),
                )

            grouped_results[result_column_name] = skill_scores
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
        self,
        data: pd.DataFrame,
        obs: str,
        pred: str,
        weights: str,
        winsorize: tuple[float, float] | None = None,
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
        winsorize : tuple[float, float], optional
            Quantiles for clipping per-observation error contributions; see
            :meth:`eval`. None uses the standard metric

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

        if winsorize is not None:
            result = self._eval_winsorized(
                obs_values, pred_values, weight_values, winsorize=winsorize
            )
            return pd.Series({column_name: result})

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
                result = self._weighted_quantile(
                    np.abs(obs_values - pred_values), weight_values, q=0.5
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
        winsorize: tuple[float, float] | None = None,
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
        winsorize : tuple[float, float], optional
            Quantiles for clipping per-observation error contributions; see
            :meth:`eval`. None uses the standard metric

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
                winsorize=winsorize,
            )
            .reset_index()
        )

        return grouped_results

    def _eval_winsorized(
        self,
        obs_values: np.ndarray,
        pred_values: np.ndarray,
        weight_values: np.ndarray,
        winsorize: tuple[float, float],
    ) -> float:
        """
        Compute the metric after clipping per-observation error contributions
        to the winsorize quantiles. See :meth:`eval` for details.
        """
        lower_q, upper_q = winsorize
        residuals = obs_values - pred_values

        # Per-observation error contribution that the metric aggregates.
        match self:
            case Metric.ROOT_MEAN_SQUARED_ERROR | Metric.MEAN_SQUARED_ERROR:
                contributions = residuals**2
            case Metric.MEAN_ABSOLUTE_ERROR | Metric.MEDIAN_ABSOLUTE_ERROR:
                contributions = np.abs(residuals)
            case Metric.MEAN_ABSOLUTE_PERCENTAGE_ERROR:
                # Mirror scikit-learn's guard against division by zero.
                epsilon = np.finfo(np.float64).eps
                contributions = np.abs(residuals) / np.maximum(
                    np.abs(obs_values), epsilon
                )
            case _:
                raise ValueError(f"Unsupported metric type: {self}")

        lower_bound = self._weighted_quantile(
            contributions, weight_values, q=lower_q
        )
        upper_bound = self._weighted_quantile(
            contributions, weight_values, q=upper_q
        )
        contributions = np.clip(contributions, lower_bound, upper_bound)

        # Aggregate the clipped contributions the same way the metric does.
        match self:
            case Metric.ROOT_MEAN_SQUARED_ERROR:
                return float(
                    np.sqrt(np.average(contributions, weights=weight_values))
                )
            case Metric.MEDIAN_ABSOLUTE_ERROR:
                return self._weighted_quantile(
                    contributions, weight_values, q=0.5
                )
            case _:
                return float(np.average(contributions, weights=weight_values))

    @staticmethod
    def _weighted_quantile(
        values: np.ndarray, weights: np.ndarray, q: float
    ) -> float:
        """
        Weighted ``q``-quantile of ``values``.

        Sorts the values and reads off the point where the cumulative weight
        crosses ``q`` of the total, interpolating between neighbours. With
        equal weights this reduces to :func:`numpy.quantile`.
        """
        order = np.argsort(values)
        values = values[order]
        weights = weights[order]
        cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights)
        return float(np.interp(q, cdf, values))

    @staticmethod
    def _validate_winsorize(winsorize: tuple[float, float]) -> None:
        """
        Validate the winsorize quantile bounds, failing fast with a clear message.

        Parameters
        ----------
        winsorize : tuple[float, float]
            Lower and upper quantiles; must satisfy 0 <= lower <= upper <= 1
        """
        lower_q, upper_q = winsorize
        if not 0.0 <= lower_q <= upper_q <= 1.0:
            raise ValueError(
                "winsorize quantiles must satisfy 0 <= lower <= upper <= 1, "
                f"got {winsorize}."
            )

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
