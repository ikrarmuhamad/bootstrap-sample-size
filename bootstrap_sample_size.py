import numpy as np
from typing import Callable, Optional
import matplotlib.pyplot as plt

class BootstrapSampleSize :
    """
    Purpose
    --------

    Find the minimum sample size using the bootstrap method.

    The sample size is increased step by step (multiples of ``n_steps``). At
    each size, the data is resampled ``n_bootstrap`` times and the half-width
    (margin error) of the bootstrap confidence interval is computed. The search
    stops at the first sample size whose margin error is smaller than
    ``margin_error``.

    Parameters
    ----------
    data : np.ndarray
        One-dimensional numeric data to resample from.
    n_bootstrap : int, default=1000
        Number of bootstrap resamples at each sample size.
    n_steps : int, default=10
        Starting sample size and the increment at each iteration
        (10, 20, 30, ...).
    metrics : Callable, optional
        Function that takes an array and returns a single number, e.g.
        ``np.mean``, ``np.median`` or ``np.std``. Defaults to ``np.mean``.
    margin_error : float, default=0.03
        Target margin error (interval half-width), in the same scale as the
        data.
    confidence : float, default=0.95
        Confidence level of the interval, between 0 and 1.
    max_steps : int, default=100_000
        Upper limit for the sample size. If it is exceeded before the target
        is reached, ``find_n_sample`` raises ``ValueError``.
    seed : int, optional
        Seed of the random number generator, for reproducible results.

    Attributes
    ----------
    rng : np.random.Generator
        Random number generator created from ``seed``.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.default_rng(42).beta(0.5, 0.5, size=3000)
    >>> model = BootstrapSampleSize(data, margin_error=0.03, seed=42)
    >>> n = model.find_n_sample()
    >>> n = model.find_n_sample(plot=True)  # plot every iteration
    """

    def __init__(
        self, 
        data : np.ndarray, 
        n_bootstrap : int=1000, 
        n_steps : int=10, 
        metrics : Optional[Callable] = None,
        margin_error : float=0.03,
        confidence: float = 0.95,
        max_steps: int = 100_000,
        seed: Optional[int] = None,
        ):

        self.data = data
        self.n_bootstrap = n_bootstrap
        self.n_steps = n_steps
        self.metrics = metrics if metrics is not None else np.mean
        self.margin_error = margin_error
        self.confidence = confidence
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

    def _bootstraping(self, steps : int) -> np.ndarray:
        """
        Build the bootstrap distribution of ``metrics``.

        Parameters
        ----------
        steps : int
            Size of each resample (drawn with replacement).

        Returns
        -------
        np.ndarray
            ``n_bootstrap`` metric values, sorted in ascending order.
        """
        dist_metrics = [
            self.metrics(self.rng.choice(self.data, steps, replace=True))
            for _ in range(self.n_bootstrap) 
            ]
        
        return np.sort(dist_metrics)

    def _find_half_width(self, num_result : np.ndarray) -> tuple[float,float,float,float]:
        """
        Compute the half-width (margin error) of the confidence interval.

        Parameters
        ----------
        num_result : np.ndarray
            Bootstrap distribution from ``_bootstraping``.

        Returns
        -------
        tuple
            ``(half_width, alpha, q_low, q_high)``: half of the interval width,
            the probability in each tail, and the lower and upper bounds.
        """
        alpha =(1 - self.confidence) / 2
        q_low, q_high = np.quantile(num_result, [alpha, 1 - alpha])
        half_width = (q_high - q_low) / 2
        return float(half_width), alpha, q_low, q_high

    def find_n_sample(self, plot : Optional[bool] = False):
        """
        Find the minimum sample size that meets the ``margin_error`` target.

        Parameters
        ----------
        plot : bool, default=False
            If True, show a histogram of the bootstrap distribution with the
            interval and margin error at every iteration.

        Returns
        -------
        int
            First sample size whose margin error is smaller than
            ``margin_error``.

        Raises
        ------
        ValueError
            If the target is not reached up to ``max_steps``.
        """
        steps = self.n_steps
        while steps <= self.max_steps:
            dist_bootstrap = self._bootstraping(steps=steps)
            width, alpha, q_low, q_high = self._find_half_width(num_result=dist_bootstrap)
            # print(f"========= n-sample : {steps} =========== \n margin error : {width} \n expected margin error : {self.margin_error}")
            if plot:
                fig, ax = plt.subplots(figsize=(8, 4))
                counts, _, _ = ax.hist(dist_bootstrap, bins=40, color="#9ecae1", edgecolor="white")
                ymax = counts.max()
                mid = (q_high + q_low)/2
                E = width
                E_target = self.margin_error

                ax.axvspan(q_low, q_high, color="red" if width > self.margin_error else "green", alpha=0.08)
                ax.axvline(q_low, color="red", linestyle="--")
                ax.axvline(q_high, color="red", linestyle="--")
                ax.axvline(mid, color="black")

                ax.axvline(mid - E_target, color="#BA7517", ls=":", lw=2, label=f"Target mid ± {E_target:g}")
                ax.axvline(mid + E_target, color="#BA7517", ls=":", lw=2)

                y_e = ymax * 1.08
                ax.annotate("", xy=(q_low, y_e), xytext=(mid, y_e),
                            arrowprops=dict(arrowstyle="<->", color="green", lw=1.8))
                ax.annotate("", xy=(q_high, y_e), xytext=(mid, y_e),
                            arrowprops=dict(arrowstyle="<->", color="green", lw=1.8))
                ax.text((q_low + mid) / 2, y_e * 1.02, f"E = {E:.4f}", ha="center", color="green")
                ax.text((mid + q_high) / 2, y_e * 1.02, f"E = {E:.4f}", ha="center", color="green")

                y_w = ymax * 1.28
                ax.annotate("", xy=(q_low, y_w), xytext=(q_high, y_w),
                            arrowprops=dict(arrowstyle="<->", color="purple", lw=1.8))
                ax.text(mid, y_w * 1.02, f"CI Width = 2E = {2*E:.4f}", ha="center", color="purple")


                ax.text(q_low, -ymax * 0.25, f"Lower Bound\n{q_low:.4f}", ha="center", va="top", color="red")
                ax.text(q_high, -ymax * 0.25, f"Upper Bound\n{q_high:.4f}", ha="center", va="top", color="red")
                ax.text(mid, -ymax * 0.25, f"Mid Point\n{mid:.4f}", ha="center", va="top")

                ax.set_ylim(0, ymax * 1.5)
                ax.set_xlabel("Mean (bootstrap)", labelpad=40)
                ax.set_ylabel("Freq")
                ax.set_title(f"CI 95% and Margin Error (n={steps}, B={self.n_bootstrap})")
                plt.tight_layout()
                plt.show()
            if width < self.margin_error:
                print(f"n-sample minimum found | n-sample : {steps}")
                return steps
            steps += self.n_steps
        raise ValueError(f"Margin error not found")
            

    
    