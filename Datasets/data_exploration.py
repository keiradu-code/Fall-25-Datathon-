import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class ConsistencyCheck:
    """
    Bins and aggregates data for a single predictor variable to check consistency/exposure across bins.
    Supports numeric, categorical, and low-cardinality numeric variables.

    Parameters:
        data (pd.DataFrame): Input data.
        pred_var (str): Predictor variable to bin.
        exp_var (str): Exposure variable (e.g., vehicle count).
        nbins (int): Number of bins for numeric variables (default 5).

    Methods:
        binning(): Performs binning based on pred_var type and stores result in self.binned_data.
        aggregate(year_var): Aggregates exposure and bin statistics by bin and year.
        plot(): Plots exposure percentage by bin and year.
    """

    def __init__(self, data, pred_var, exp_var, nbins=5):
        self.data = data.copy()
        self.pred_var = pred_var
        self.exp_var = exp_var
        self.nbins = nbins
        self.binned_data = None
        self.agg_df = None
        self.year_var = None

    def binning(self):
        pred_var = self.pred_var
        exp_var = self.exp_var
        year_var = self.year_var
        nbins = self.nbins
        df = self.data.copy()

        is_numeric = pd.api.types.is_numeric_dtype(df[pred_var])
        n_unique = df[pred_var].nunique(dropna=True)

        treat_as_cat = is_numeric and n_unique < 20

        if is_numeric and not treat_as_cat:
            # Numeric variable with enough unique values: bin by equal exposure
            missing_mask = df[pred_var].isna()
            df_non_missing = df[~missing_mask].copy()
            df_missing = df[missing_mask].copy()

            df_sorted = df_non_missing.sort_values(pred_var).reset_index(drop=True)
            df_sorted["cum_exp"] = df_sorted[exp_var].cumsum()
            total_exp = df_sorted[exp_var].sum()
            bin_edges_cumexp = [i * total_exp / nbins for i in range(nbins + 1)]

            # Find pred_var values just below each cum_exp bin edge
            pred_var_edges = []
            for edge in bin_edges_cumexp:
                idx = df_sorted[df_sorted["cum_exp"] <= edge].last_valid_index()
                if idx is not None:
                    pred_val = df_sorted.loc[idx, pred_var]
                else:
                    pred_val = df_sorted[pred_var].min()
                pred_var_edges.append(pred_val)
            # Ensure edges are sorted and unique, and cover the full range
            pred_var_edges = sorted(set(pred_var_edges))
            if pred_var_edges[0] > df_sorted[pred_var].min():
                pred_var_edges = [df_sorted[pred_var].min()] + pred_var_edges
            if pred_var_edges[-1] < df_sorted[pred_var].max():
                # pred_var_edges.append(df_sorted[pred_var].max())
                pred_var_edges[-1] = df_sorted[pred_var].max()
            # Add a small epsilon to the last edge to include the max value
            # pred_var_edges[-1] = pred_var_edges[-1] + 1e-8

            df_sorted["bin"] = pd.cut(
                df_sorted[pred_var],
                bins=pred_var_edges,
                labels=False,
                include_lowest=True,
            )
            df_sorted = df_sorted.drop(columns=["cum_exp"])

            if not df_missing.empty:
                # Assign missing values to a special bin
                df_missing = df_missing.copy()
                df_missing["bin"] = -1
                df_sorted = pd.concat(
                    [df_sorted, df_missing], axis=0, ignore_index=True
                )

            self.binned_data = df_sorted
        elif treat_as_cat:
            # Numeric variable with low cardinality: treat as categorical
            ordered_cats = sorted(df[pred_var].dropna().unique())
            cat_to_bin = {cat: i for i, cat in enumerate(ordered_cats)}
            df["bin"] = df[pred_var].map(cat_to_bin)
            df.loc[df[pred_var].isna(), "bin"] = -1
            self.binned_data = df
        else:
            # Categorical variable: use category codes as bins
            ordered_cats = sorted(df[pred_var].dropna().unique())
            df["bin"] = pd.Categorical(
                df[pred_var], categories=ordered_cats, ordered=True
            )
            df["bin"] = df["bin"].astype(object)
            df.loc[df[pred_var].isna(), "bin"] = -1
            self.binned_data = df

    def aggregate(self, year_var):
        pred_var = self.pred_var
        exp_var = self.exp_var
        self.year_var = year_var
        df = self.binned_data.copy()

        is_numeric = pd.api.types.is_numeric_dtype(self.data[pred_var])
        n_unique = self.data[pred_var].nunique(dropna=True)
        treat_as_cat = is_numeric and n_unique < 20

        if is_numeric and not treat_as_cat:
            # Numeric variable with enough unique values: aggregate by bin and year
            agg_df = df.groupby(["bin", year_var], as_index=False).agg({exp_var: "sum"})
            total_exp_per_year = agg_df.groupby(year_var)[exp_var].transform("sum")
            agg_df[f"{exp_var}_pct"] = agg_df[exp_var] / total_exp_per_year * 100

            bin_stats = (
                df[df["bin"] != -1]
                .groupby(["bin", year_var])[pred_var]
                .agg(["min", "max"])
                .reset_index()
            )
            bin_stats = bin_stats.rename(
                columns={"min": f"{pred_var}_min", "max": f"{pred_var}_max"}
            )
            agg_df = agg_df.merge(bin_stats, on=["bin", year_var], how="left")
        elif treat_as_cat:
            # Numeric variable with low cardinality: aggregate by bin and year, map bin to value
            agg_df = df.groupby(["bin", year_var], as_index=False).agg({exp_var: "sum"})
            total_exp_per_year = agg_df.groupby(year_var)[exp_var].transform("sum")
            agg_df[f"{exp_var}_pct"] = agg_df[exp_var] / total_exp_per_year * 100
            unique_vals = np.sort(self.data[pred_var].dropna().unique())
            bin_to_val = {i: v for i, v in enumerate(unique_vals)}
            agg_df[f"{pred_var}_cat"] = agg_df["bin"].map(bin_to_val)
        else:
            # Categorical variable: aggregate by bin and year
            agg_df = df.groupby(["bin", year_var], as_index=False).agg({exp_var: "sum"})
            total_exp_per_year = agg_df.groupby(year_var)[exp_var].transform("sum")
            agg_df[f"{exp_var}_pct"] = agg_df[exp_var] / total_exp_per_year * 100
            agg_df[f"{pred_var}_cat"] = agg_df["bin"]
        self.agg_df = agg_df

    def plot(self):
        import matplotlib.pyplot as plt

        pred_var = self.pred_var
        exp_var = self.exp_var
        year_var = self.year_var
        agg_df = self.agg_df

        plt.figure(figsize=(10, 6))

        if f"{pred_var}_max" in agg_df.columns:
            # Numeric variable with enough unique values: plot by bin max
            x_col = f"{pred_var}_max"
            y_col = f"{exp_var}_pct"
            bin_labels = []
            bin_positions = []
            for b in sorted(agg_df["bin"].unique()):
                if b == -1 or b == "-1":
                    bin_labels.append("missing")
                    bin_positions.append(b)
                else:
                    max_val = agg_df.loc[agg_df["bin"] == b, x_col].max()
                    try:
                        label_val = round(float(max_val), 3)
                    except Exception:
                        label_val = max_val
                    bin_labels.append(f"< {label_val}")
                    bin_positions.append(b)
            for year, group in agg_df.groupby(year_var):
                plt.plot(
                    group["bin"], group[y_col], marker="o", label=f"{year_var}: {year}"
                )
            plt.xticks(bin_positions, bin_labels)
            plt.xlabel(f"{pred_var} bin")
            plt.ylabel(y_col + " (%)")
            plt.title(f"Line Chart of {y_col} by {pred_var} bin and {year_var}")
        else:
            # Categorical or low-cardinality: plot by bin
            x_col = "bin"
            y_col = f"{exp_var}_pct"
            categories = agg_df[x_col].unique()
            import numpy as np

            cat_labels = np.array([str(c) for c in categories])
            positions = np.arange(len(categories))
            for year, group in agg_df.groupby(year_var):
                group_sorted = group.set_index(x_col).reindex(categories).reset_index()
                plt.plot(
                    positions,
                    group_sorted[y_col],
                    marker="o",
                    label=f"{year_var}: {year}",
                )
            plt.xticks(positions, cat_labels)
            plt.xlabel(pred_var)
            plt.ylabel(y_col + " (%)")
            plt.title(f"Line Chart of {y_col} by {pred_var} and {year_var}")

        plt.legend(title=year_var)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()


class ConsistencyCheckList:
    """
    Runs ConsistencyCheck for a list of predictor variables and provides batch plotting.

    Parameters:
        data (pd.DataFrame): Input data.
        pred_var_lst (list of str): List of predictor variables to check.
        expo_var (str): Exposure variable (e.g., vehicle count).
        year_var (str): Year variable for grouping.
        nbins (int): Number of bins for numeric variables (default 5).

    Methods:
        run_all(): Runs ConsistencyCheck for each predictor in the list.
        plot_all(): Plots all consistency plots for the predictors.
    """

    def __init__(self, data, pred_var_lst, expo_var, year_var, nbins=5):
        self.data = data
        self.pred_var_lst = pred_var_lst
        self.expo_var = expo_var
        self.year_var = year_var
        self.nbins = nbins
        self.checks = {}  # Store ConsistencyCheck objects by pred_var

    def run_all(self):
        # Run ConsistencyCheck for each predictor variable in the list
        for pred_var in self.pred_var_lst:
            cc = ConsistencyCheck(
                data=self.data,
                pred_var=pred_var,
                exp_var=self.expo_var,
                nbins=self.nbins,
            )
            cc.binning()
            cc.aggregate(year_var=self.year_var)
            self.checks[pred_var] = cc

    def plot_all(self):
        # Plot all consistency plots for the predictor variables
        for pred_var, cc in self.checks.items():
            print(f"\n--- Consistency Plot for {pred_var} ---")
            cc.plot()


class PredictivenessCheck:
    """
    Bins, aggregates, and plots predictiveness statistics for a given predictor and target.
    Usage:
        pc = PredictivenessCheck(df, pred_var, exp_var, var_1, var_2=None)
        pc.aggregate()
        pc.plot()
    """

    def __init__(self, df, pred_var, exp_var, var_1, var_2=None):
        self.df = df.copy()
        self.pred_var = pred_var
        self.exp_var = exp_var
        self.var_1 = var_1
        self.var_2 = var_2
        self.agg_df = None

    def binning(self, nbins=5):
        """
        Use ConsistencyCheck.binning() to create binned_data for PredictivenessCheck.
        The number of bins can be specified (default 5).
        """
        cc = ConsistencyCheck(
            data=self.df, pred_var=self.pred_var, exp_var=self.exp_var, nbins=nbins
        )
        cc.binning()
        self.binned_data = cc.binned_data.copy() if hasattr(cc, "binned_data") else None

    def aggregate(self):
        """
        Aggregate binned data by 'bin', summing specified variables and computing ratios.
        Handles three cases:
        1. Numeric variable with enough unique values (is_numeric and not treat_as_cat)
        2. Numeric variable with low cardinality (is_numeric and treat_as_cat)
        3. Categorical variable (is_numeric == False)
        """
        df = self.binned_data
        var_1 = self.var_1
        exp_var = self.exp_var
        var_2 = self.var_2
        pred_var = self.pred_var

        is_numeric = np.issubdtype(df["bin"].dtype, np.number)
        n_unique = df[pred_var].nunique(dropna=True)
        treat_as_cat = is_numeric and n_unique < 20

        group_cols = ["bin"]
        agg_dict = {var_1: "sum", exp_var: "sum"}
        if var_2 is not None and var_2 in df.columns:
            agg_dict[var_2] = "sum"

        if is_numeric and not treat_as_cat:
            # Numeric variable with enough unique values: aggregate by bin
            agg_df = df.groupby(group_cols, as_index=False).agg(agg_dict)
            agg_df[f"{var_1}_over_{exp_var}"] = agg_df[var_1] / agg_df[exp_var]
            if var_2 is not None and var_2 in agg_df.columns:
                agg_df[f"{var_2}_over_{exp_var}"] = agg_df[var_2] / agg_df[exp_var]
            # Add min/max of pred_var for each bin if available
            if pred_var is not None and pred_var in df.columns:
                bin_stats = (
                    df[df["bin"] != -1]
                    .groupby("bin")[pred_var]
                    .agg(["min", "max"])
                    .reset_index()
                )
                bin_stats = bin_stats.rename(
                    columns={"min": f"{pred_var}_min", "max": f"{pred_var}_max"}
                )
                agg_df = agg_df.merge(bin_stats, on="bin", how="left")
            # Set bin_label: missing bin (-1) as 'missing', others as just the value (3 decimals)
            if pred_var is not None and f"{pred_var}_max" in agg_df.columns:

                def make_label(row):
                    if row["bin"] == -1:
                        return "missing"
                    else:
                        return f"{row[f'{pred_var}_max']:.2f}"

                agg_df["bin_label"] = agg_df.apply(make_label, axis=1)
            else:
                agg_df["bin_label"] = agg_df["bin"].astype(str)
        elif treat_as_cat:
            # Numeric variable with low cardinality: treat as categorical
            agg_df = df.groupby(group_cols, as_index=False).agg(agg_dict)
            agg_df[f"{var_1}_over_{exp_var}"] = agg_df[var_1] / agg_df[exp_var]
            if var_2 is not None and var_2 in agg_df.columns:
                agg_df[f"{var_2}_over_{exp_var}"] = agg_df[var_2] / agg_df[exp_var]
            # Map bin to value using pred_var
            if pred_var is not None and pred_var in df.columns:
                unique_vals = np.sort(df[pred_var].dropna().unique())
                bin_to_val = {i: v for i, v in enumerate(unique_vals)}
                agg_df["bin_label"] = agg_df["bin"].map(bin_to_val).astype(str)
            else:
                agg_df["bin_label"] = agg_df["bin"].astype(str)
        else:
            # Categorical variable: use bin as label
            agg_df = df.groupby(group_cols, as_index=False).agg(agg_dict)
            agg_df[f"{var_1}_over_{exp_var}"] = agg_df[var_1] / agg_df[exp_var]
            if var_2 is not None and var_2 in agg_df.columns:
                agg_df[f"{var_2}_over_{exp_var}"] = agg_df[var_2] / agg_df[exp_var]
            agg_df["bin_label"] = agg_df["bin"].astype(str)
        self.agg_df = agg_df
        self.overall_mean = (
            self.agg_df[self.var_1].sum() / self.agg_df[self.exp_var].sum()
            if self.agg_df[self.exp_var].sum() != 0
            else np.nan
        )

    def plot(
        self,
        var_1_color="blue",
        var_2_color="red",
        book_avg_color="green",
        figsize=(10, 6),
    ):
        """
        Plot aggregated bin statistics from agg_df.
        - x-axis: 'bin' (with missing bin first if present), x-ticks as bin_label (for continuous: '<' + value, for others: label)
        - Left y-axis: line chart of var_1/exp_var (var_1_color), and var_2/exp_var (var_2_color, if provided)
        - Right y-axis: bar chart of exp_var (grey, alpha=0.5)
        - Adds a dashed horizontal line at the overall mean of var_1/exp_var (book_avg_color).
        - Computes Kendall's tau correlation between pred_var and var_1 and displays it in the plot title.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.stats import kendalltau

        agg_df = self.agg_df
        var_1 = self.var_1
        var_2 = self.var_2
        exp_var = self.exp_var
        pred_var = self.pred_var
        df = self.binned_data

        # Sort bins, put missing bin (-1 or 'missing') first if present
        bins = agg_df["bin"].tolist()
        if -1 in bins:
            agg_df = pd.concat(
                [agg_df[agg_df["bin"] == -1], agg_df[agg_df["bin"] != -1]],
                ignore_index=True,
            )
        elif "missing" in bins:
            agg_df = pd.concat(
                [
                    agg_df[agg_df["bin"] == "missing"],
                    agg_df[agg_df["bin"] != "missing"],
                ],
                ignore_index=True,
            )

        x = np.arange(len(agg_df))

        def is_float_str(s):
            try:
                float(s)
                return True
            except:
                return False

        # Determine if pred_var is numeric and has < 20 unique values (treat_as_cat logic)
        is_numeric = pd.api.types.is_numeric_dtype(self.df[self.pred_var])
        n_unique = self.df[self.pred_var].nunique(dropna=True)
        treat_as_cat = is_numeric and n_unique < 20

        bin_labels = []
        for i, row in agg_df.iterrows():
            label = (
                str(row["bin_label"])
                if "bin_label" in agg_df.columns
                else str(row["bin"])
            )
            if row["bin"] == -1 or label.lower() == "missing":
                bin_labels.append("missing")
            elif is_float_str(label):
                # Only add '<' if not treat_as_cat (i.e., truly binned numeric)
                if not treat_as_cat:
                    bin_labels.append(f"< {label}")
                else:
                    bin_labels.append(label)
            else:
                bin_labels.append(label)

        # Compute Kendall's tau correlation between pred_var and var_1 using input data
        tau_str = ""
        try:
            # Use only rows where both pred_var and var_1 are not null
            valid = self.df[[pred_var, var_1]].dropna()
            if not valid.empty:
                tau, pval = kendalltau(valid[pred_var], valid[var_1])
                tau_str = f" [Kendall's tau: {tau:.2f}]"
        except Exception as e:
            tau_str = ""

        fig, ax1 = plt.subplots(figsize=figsize)

        # Plot var_1/exp_var line
        y1 = agg_df[f"{var_1}_over_{exp_var}"]
        (line1,) = ax1.plot(
            x, y1, marker="o", color=var_1_color, label=f"{var_1}/{exp_var}"
        )
        lines = [line1]
        labels = [f"{var_1}/{exp_var}"]
        if var_2 is not None and f"{var_2}_over_{exp_var}" in agg_df.columns:
            y2 = agg_df[f"{var_2}_over_{exp_var}"]
            (line2,) = ax1.plot(
                x, y2, marker="o", color=var_2_color, label=f"{var_2}/{exp_var}"
            )
            lines.append(line2)
            labels.append(f"{var_2}/{exp_var}")
        ax1.set_xlabel(f"{pred_var} bin")
        ax1.set_ylabel(f"Ratio to {exp_var}")
        ax1.set_xticks(x)
        ax1.set_xticklabels(bin_labels, rotation=0, ha="center")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Add dashed horizontal line for overall mean ratio
        ax1.axhline(
            self.overall_mean,
            color=book_avg_color,
            linestyle="--",
            linewidth=2,
            label=f"Overall {var_1}/{exp_var}",
        )
        lines.append(
            plt.Line2D([], [], color=book_avg_color, linestyle="--", linewidth=2)
        )
        labels.append(f"Overall {var_1}/{exp_var}")

        # Plot exp_var as bars on secondary y-axis
        ax2 = ax1.twinx()
        bars = ax2.bar(
            x,
            agg_df[exp_var],
            color="grey",
            alpha=0.15,
            width=0.7,
            label=exp_var,
            align="center",
        )
        ax2.set_ylabel(exp_var)

        # Combine legends from both axes and place inside the plot (upper left, not covered by bars)
        all_handles = lines + [bars]
        all_labels = labels + [exp_var]
        ax1.legend(
            all_handles,
            all_labels,
            loc="upper left",
            bbox_to_anchor=(0.01, 0.99),
            frameon=True,
        )

        plt.title(f"Binned {pred_var}: Ratios and {exp_var} by Bin {tau_str}")
        plt.tight_layout()
        plt.show()

    def top_lift(self):
        self.top_lift = (
            list(self.agg_df[f"{self.var_1}_over_{self.exp_var}"])[-1]
            / self.overall_mean
            if self.overall_mean != 0
            else np.nan
        )
        return self.top_lift


class PredictivenessCheckList:
    """
    Runs PredictivenessCheck for a list of predictor variables and provides batch plotting.

    Parameters:
        data (pd.DataFrame): Input data.
        pred_var_lst (list of str): List of predictor variables to check.
        exp_var (str): Exposure variable (e.g., vehicle count).
        var_1 (str): Numerator variable for predictiveness ratio.
        var_2 (str, optional): Second numerator variable for predictiveness ratio.
        nbins (int): Number of bins for numeric variables (default 5).

    Methods:
        run_all(): Runs PredictivenessCheck for each predictor in the list.
        plot_all(): Plots all predictiveness plots for the predictors.
    """

    def __init__(self, data, pred_var_lst, exp_var, var_1, var_2=None, nbins=5):
        self.data = data
        self.pred_var_lst = pred_var_lst
        self.exp_var = exp_var
        self.var_1 = var_1
        self.var_2 = var_2
        self.nbins = nbins
        self.checks = {}  # Store PredictivenessCheck objects by pred_var

    def run_all(self):
        # Run PredictivenessCheck for each predictor variable in the list
        for pred_var in self.pred_var_lst:
            pc = PredictivenessCheck(
                df=self.data,
                pred_var=pred_var,
                exp_var=self.exp_var,
                var_1=self.var_1,
                var_2=self.var_2,
            )
            pc.binning(nbins=self.nbins)
            pc.aggregate()
            self.checks[pred_var] = pc

    def plot_all(self, var_1_color="blue", var_2_color="red", book_avg_color="green"):
        # Plot all predictiveness plots for the predictor variables
        for pred_var, pc in self.checks.items():
            print(f"\n--- Predictiveness Plot for {pred_var} ---")
            pc.plot(
                var_1_color=var_1_color,
                var_2_color=var_2_color,
                book_avg_color=book_avg_color,
            )
