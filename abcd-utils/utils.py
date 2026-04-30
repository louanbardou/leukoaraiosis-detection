"""Collection of utility tools"""
import itertools
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path

root_dir = Path("/wynton/group/rsl/ABCD_data")
sessionID_sort = ["Baseline"] + [f"Year {i}" for i in range(1, 10)]


SESSION_MAPPING = {
    "baseline_year_1_arm_1": "Baseline",
    "1_year_follow_up_y_arm_1": "Year 1",
    "2_year_follow_up_y_arm_1": "Year 2",
    "3_year_follow_up_y_arm_1": "Year 3",
    "4_year_follow_up_y_arm_1": "Year 4",
    "baselineYear1Arm1": "Baseline",
    "2YearFollowUpYArm1": "Year 2",
    "4YearFollowUpYArm1": "Year 4",
}


def qc_filter(df: pd.DataFrame, fsqc=True, dmri=False) -> pd.DataFrame:
    dfqc = pd.read_csv(
        root_dir / "4.0/imaging/concatenated/imaging_qc.csv", index_col=0
    )[["src_subject_id", "event_name", "fsqc_qc", "iqc_dmri_ok_ser"]].rename(
        columns={"event_name": "eventname"}
    )
    dfqc = unify_naming(dfqc)
    if fsqc and dmri:
        filter = (dfqc.fsqc_qc == "accept") & (dfqc.iqc_dmri_ok_ser > 0)
    elif fsqc and not dmri:
        filter = dfqc.fsqc_qc == "accept"
    elif not fsqc and dmri:
        filter = dfqc.iqc_dmri_ok_ser > 0
    else:
        return df
    # dfqc = ut.get_events(dfqc, ['baseline']).reset_index(drop=True)
    subjsess_filter = dfqc[filter][["subjectID", "sessionID"]]
    return df.merge(subjsess_filter, on=["subjectID", "sessionID"], how="inner")


def rename_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename base columns"""
    return df.rename(columns={"src_subject_id": "subjectID", "eventname": "sessionID"})


def split_subject_name(subjectID: str) -> str:
    """Helper"""
    try:
        return subjectID.split("INV")[1]
    except:
        return subjectID


def get_sorted_events(eventlist: list) -> list:
    """Sorts events"""
    intersected_events = set(eventlist) & set(sessionID_sort)
    ordered_events = sorted(
        list(intersected_events), key=lambda x: sessionID_sort.index(x)
    )
    return ordered_events


def optimize_memory_use(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize memory use"""
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category")
        elif df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        elif df[col].dtype == "int64":
            df[col] = df[col].astype("int32")
    if "sessionID" in df.columns:
        df["sessionID"] = pd.Categorical(
            df["sessionID"],
            categories=get_sorted_events(df["sessionID"].unique()),
            ordered=True,
        )
    return df


def unify_naming(df: pd.DataFrame, optimize=True):
    """Unifies subject & session mapping - e.g. remove `NDAR_INV`"""
    df = rename_base_columns(df)
    df = df.assign(
        subjectID=df.subjectID.apply(split_subject_name)
    ).replace(  # TODO: faster with replace(r".*INV","")?
        {"sessionID": SESSION_MAPPING}
    )
    if optimize:
        df = optimize_memory_use(df)
    return df


def abcd_merge(dfs: list[pd.DataFrame], merge_type: str = "outer") -> pd.DataFrame:
    """Merges list of abcd dataframes"""
    merge_columns = ["subjectID", "sessionID"]
    dfs = list(map(unify_naming, dfs))
    for df in dfs:
        # df = rename_base_columns(df)
        assert all(col in df.columns for col in merge_columns)
    df = reduce(
        lambda df1, df2: pd.merge(df1, df2, on=merge_columns, how=merge_type), dfs
    )
    return df


def get_events(df: pd.DataFrame, events: list):
    """Tool to easily get baseline and year2"""
    if "eventname" in df.columns:
        df = unify_naming(df)
    elif "sessionID" not in df.columns:
        raise ValueError("Not the right columns")

    event_dict = {"baseline": "Baseline", "year2": "Year 2"}
    event_names = [event_dict[ev] for ev in events]
    return df[df.sessionID.isin(event_names)]


def get_event(df: pd.DataFrame, event: str):
    """Tool to easily get baseline or year2"""
    return get_events(df, [event])


def plot_vyshyvanka(
    df: pd.DataFrame,
    ax: plt.Axes,
    fillna=None,
    coef_width=50,
    line_opacity=0.6,
    marker_opacity=0.1,
) -> tuple[plt.Axes, int, np.ndarray]:
    """
    It will build the flow plot "Vyshyvanka": the transitions between possible states of
    observations on the Y-axis over some time periods on the X-axis.

    Line width and marker opacity correlate with the number of observations at a current
    point.


    :param df: dataframe where columns are observations and rows are time points
    :param ax: an instance of ``matplotlib.axes.Axes`` to plot on
    :param coef_width: the bigger it, the thicker will be the lines
    :param line_opacity: lines opacity
    :param marker opacity: markers opacity

    :return: (``matplotlib.axes.Axes``, number_of_observations_without_na,
        possible_states)

    **Example:**

    .. code-block:: python

        import random
        import pandas as pd
        import matplotlib.pyplot as plt


        obs_num = 1000
        time_points = 8
        states = [1, 2, 3, 4, 5]

        tmp = []
        for _ in range(time_points):
            tmp.append(random.choices(states, k=obs_num,
                                      weights=[0.05, 0.2 ,0.5, 0.2 ,0.05]))

        df = pd.DataFrame(tmp)


        fig, ax = plt.subplots(figsize=(15, 7))
        fig.patch.set_facecolor('white')

        ax, num, relevant_states = plot_vyshyvanka(df, ax)
        _ = ax.set_yticks(relevant_states)
        _ = ax.set_xlabel('Time points')
        _ = ax.set_ylabel('States')
        _ = ax.set_title(f'State transitions over time (n={num})')


    """
    # Dealing with NA:
    if fillna is not None:
        df_plot = df.fillna(fillna)
    else:
        df_plot = df.dropna(axis=1)
    obs_num = len(df_plot.columns)

    # iterate over time points using sliding window with length=2, stride=1
    for row1, row2 in zip(
        itertools.islice(df_plot.iterrows(), 0, None, 1),
        itertools.islice(df_plot.iterrows(), 1, None, 1),
    ):
        # get row idxs (from df.iterrows() output)
        x1 = row1[0]
        x2 = row2[0]
        # get row values
        tmp = pd.concat([row1[1], row2[1]], axis=1).to_numpy()
        # count unique pairs was-now states
        pairs, nums = np.unique(tmp, axis=0, return_counts=True)
        # get all possible states
        states = np.unique(pairs.flatten())
        # calculate line widths for each pair (state transition)
        # line width correlates with observations amount,
        # where ``obs_num * coef_width`` is the maximum width
        widths = nums / obs_num * coef_width

        # add axis to widths in order to merge it with pairs for comfy iteration
        for row in np.hstack([pairs, np.expand_dims(widths, axis=1)]):
            y1, y2, w = row
            # plot line
            ax.plot(
                [x1, x2],
                [y1, y2],
                linewidth=w,
                color="red",
                alpha=line_opacity,
                solid_capstyle="round",
            )
            # plot markers
            # marker opacity correlates with observations amount
            ax.plot(x1, y1, "o", markersize=2, color="black", alpha=marker_opacity)
            ax.plot(x2, y2, "o", markersize=2, color="black", alpha=marker_opacity)

    return ax, obs_num, states
