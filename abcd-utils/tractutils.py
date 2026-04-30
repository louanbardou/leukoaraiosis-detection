import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import unify_naming

map_session_id = {
    "old": "new",
    "baseline_year_1_arm_1": "baselineYear1Arm1",
    "2_year_follow_up_y_arm_1": "2YearFollowUpYArm1",
    "4_year_follow_up_y_arm_1": "4YearFollowUpYArm1",
}
map_session_id_inv = {val: key for key, val in map_session_id.items()}
map_bundles = {
    "CST_R": "RightCorticospinal",
    "CST_L": "LeftCorticospinal",
    "UNC_R": "RightUncinate",
    "UNC_L": "LeftUncinate",
    "IFO_R": "RightIFOF",
    "IFO_L": "LeftIFOF",
    "ARC_R": "RightArcuate",
    "ARC_L": "LeftArcuate",
    "ATR_R": "RightThalamicRadiation",
    "ATR_L": "LeftThalamicRadiation",
    "CGC_R": "RightCingulumCingulate",
    "CGC_L": "LeftCingulumCingulate",
    "HCC_R": "RightCingulumHippocampus",
    "HCC_L": "LeftCingulumHippocampus",
    "FP": "CallosumForcepsMajor",
    "FA": "CallosumForcepsMinor",
    "ILF_R": "RightILF",
    "ILF_L": "LeftILF",
    "SLF_R": "RightSLF",
    "SLF_L": "LeftSLF",
}

BUNDLE_DICT = {
    "CST_R": "Right Corticospinal",
    "CST_L": "Left Corticospinal",
    "UNC_R": "Right Uncinate",
    "UNC_L": "Left Uncinate",
    "IFO_R": "Right IFOF",
    "IFO_L": "Left IFOF",
    "ARC_R": "Right Arcuate",
    "ARC_L": "Left Arcuate",
    "ATR_R": "Right Thalamic Radiation",
    "ATR_L": "Left Thalamic Radiation",
    "CGC_R": "Right Cingulum Cingulate",
    "CGC_L": "Left Cingulum Cingulate",
    "HCC_R": "Right Cingulum Hippocampus",
    "HCC_L": "Left Cingulum Hippocampus",
    "FP": "Callosum Forceps Major",
    "FA": "Callosum Forceps Minor",
    "ILF_R": "Right ILF",
    "ILF_L": "Left ILF",
    "SLF_R": "Right SLF",
    "SLF_L": "Left SLF",
    "VOF_R": "Right Vertical Occipital",
    "VOF_L": "Left Vertical Occipital",
    "pARC_R": "Right Posterior Arcuate",
    "pARC_L": "Left Posterior Arcuate",
    "AntFrontal": "Callosum: AntFrontal",
    "Motor": "Callosum: Motor",
    "Occipital": "Callosum: Occipital",
    "Orbital": "Callosum: Orbital",
    "PostParietal": "Callosum: PostParietal",
    "SupFrontal": "Callosum: SupFrontal",
    "SupParietal": "Callosum: SupParietal",
    "Temporal": "Callosum: Temporal",
}

root_dir = Path("/wynton/group/rsl/ABCD_data")
labeled_data = root_dir / "labeled_data"
confounding_file = labeled_data / "confounding_factors.csv"


def optimize_memory_use(df: pd.DataFrame, intformat=np.int8) -> pd.DataFrame:
    cols = list(set(["subjectID", "sessionID", "tractID"]) & set(df.columns))
    for col in cols:
        df[col] = df[col].astype("category").cat.remove_unused_categories()
    if "nodeID" in df.columns:
        df["nodeID"] = df["nodeID"].astype(intformat)
    return df


def tracto_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Melts along the DTI measures"""
    id_vars = ["tractID", "nodeID", "subjectID", "sessionID"]
    df = df.melt(
        id_vars=id_vars,
        value_vars=["FA", "MD", "AD", "GA", "RD"],
        var_name="dti_measure",
        value_name="dti_value",
    )
    return df


def load_tractometry(
    tracto_fp: Union[Path, str], long_type: bool = False
) -> pd.DataFrame:
    """Loads tractometry data into pd.DataFrame

    Args:
        tracto_fp (Union[Path, str]): Path to tractometry file, or name if using
            standard ones.
        long_type (bool, optional): whether to return the dataframe in long format.
            Defaults to False.

    Returns:
        pd.DataFrame: tractometry data
    """
    if isinstance(tracto_fp, str):
        tracto_fp = root_dir / f"4.0/imaging/concatenated/{tracto_fp}.pkl"

    tractometry = pd.read_pickle(tracto_fp)
    tractometry = (
        tractometry.assign(
            subjectID=tractometry.subjectID.apply(lambda x: x.split("INV")[1])
        )
        .replace({"sessionID": map_session_id_inv})
        .rename(
            columns={
                oc: oc.split("_")[1].upper()
                for oc in tractometry.columns
                if re.match(r"^d[tk]{1}i_", oc)
            }
        )
        # .assign(tractID=tractometry.tractID.apply(lambda x: map_bundles[x]))
        # # 3x faster than df.replace
    )
    if long_type:
        tractometry = tracto_to_long(tractometry)
    tractometry = optimize_memory_use(tractometry)
    return tractometry


def load_labels(
    labels_path: Path, label_ids: Union[None, list] = None, to_categorical: bool = True
) -> tuple:
    """Loads labels' file, cleaning it up so it matches the format for tractometry
        analysis. Assumes it's coming from NDA R file exported to csv.

    Args:
        labels_path (Path): The file location of the csv containing labels (should be
            under the format ['src_subject_id', 'eventname', 'label_name'])
        label_ids (list, optional): The name of the labels to pick if there are several.
            Defaults to None, will pick all available labels.

    Returns:
        tuple: dataframe ready for merging with tractometry data, new_label_ids
    """

    labels_df = pd.read_csv(labels_path)
    labels_df = labels_df.rename(
        columns={"src_subject_id": "subjectID", "eventname": "sessionID"}
    )
    labels_df = labels_df.assign(
        subjectID=labels_df.subjectID.apply(lambda x: x.split("INV")[1])
    )
    # labels_df = labels_df.replace({'sessionID': map_session_id})
    label_names = labels_df.columns.drop(["subjectID", "sessionID"])
    if label_ids is None:
        if len(label_names) >= 1:
            label_ids = label_names
        else:
            sys.exit("Label's file doesn't have a label.")
    elif any(lid not in label_names for lid in label_ids):
        sys.exit(
            f"Label {label_ids} provided isn't in columns headers of labels' file."
        )

    # only keep subject ID, session ID, and labels
    keep_labels = ["subjectID", "sessionID"]
    keep_labels.extend(label_ids)
    labels_df = labels_df[keep_labels]
    # Clean up dataframe
    new_label_ids = []
    for label_id in label_ids:
        (
            categorical,
            new_label_id,
            new_grouping,
            label_order,
            custom_function,
        ) = load_json(labels_path.with_suffix(".json"), label_id)
        if custom_function is not None:
            labels_df.loc[:, label_id] = labels_df[label_id].apply(
                eval(custom_function)
            )
        labels_df = labels_df.rename(columns={label_id: new_label_id})
        if (
            to_categorical
            and categorical
            and (label_order is not None or new_grouping is not None)
        ) or (to_categorical and not categorical and new_grouping is not None):
            labels_df = categorize(
                labels_df,
                new_label_id,
                new_grouping,
                categorical=categorical,
                label_order=label_order,
            )
        new_label_ids.append(new_label_id)
    return labels_df, new_label_ids


def load_confounding(
    labels_df: Union[pd.DataFrame, None] = None,
    file_path: Path = confounding_file,
    genetics_fp: Path = None,
    PC_number: int = 10,
):
    """Creates a dataframe with classic confounding factors

    Args:
        labels_df (pd.DataFrame): main dataframe to add covariates to.
        file_path (Path, optional): csv containing covariates. Defaults to
            `confounding_file`.
        genetics_fp (Path, optional): whether to include genetic Principal Components
            instead of race and ethnicity. Enter a valid file path to work.
        PC_number (int, optional): number of genetical PCs to include. Defaults to 10.

    Returns:
        tuple: (new dataframe with confounding factors, confounding labels)
    """

    confounding, confounding_labels = load_labels(file_path)
    if genetics_fp is not None:
        to_drop = ["Race"]
        if all([col in confounding.columns for col in to_drop]):
            confounding = confounding.drop(columns=to_drop)
        else:
            RuntimeWarning(f"Could not drop {to_drop} from confounding columns")
        confounding_labels = [c for c in confounding_labels if c not in to_drop]
        PCs_df = pd.read_csv(genetics_fp, sep="\t")
        PCs_df = PCs_df.drop(columns=["FID"]).rename(columns={"IID": "subjectID"})
        PCs_df = PCs_df.assign(
            subjectID=PCs_df.subjectID.apply(lambda x: x.split("INV")[1])
        ).drop(
            columns=[
                pc
                for pc in PCs_df.columns.drop("subjectID")
                if isinstance(pc, str) and int(pc.split("C")[1]) > PC_number
            ]
        )
        confounding = pd.merge(
            left=confounding,
            right=PCs_df,
            left_on=["subjectID"],
            right_on=["subjectID"],
        )
        confounding_labels.extend(PCs_df.columns.drop("subjectID"))

    if labels_df is None:
        return confounding, confounding_labels
    else:
        with_confounding = pd.merge(
            left=labels_df,
            right=confounding,
            left_on=["subjectID", "sessionID"],
            right_on=["subjectID", "sessionID"],
        )
        return with_confounding, confounding_labels


def merge_labels_features(
    tractometry: Union[None, pd.DataFrame] = None,
    labels_df: Union[None, pd.DataFrame] = None,
    long_type: bool = True,
) -> pd.DataFrame:
    """Merges tractometry data (features) with labels.

    Args:
        tractometry (pd.DataFrame, optional): Tractometry dataframe. Defaults to None.
        labels_df (pd.DataFrame, optional): Labels dataframe. Defaults to None.
        long_type (bool, optional): Whether to melt the dataframe. Defaults to True.

    Returns:
        pd.DataFrame: dataframe with features and labels in a long format
    """

    if tractometry is None:
        tractometry = load_tractometry()
    if labels_df is None:
        print("Loading random labels")
        labels_dict = {
            subj: np.random.choice(["blue", "brown"])
            for subj in np.unique(tractometry.subjectID)
        }
        tractometry = tractometry.assign(
            eye_color=tractometry.subjectID.apply(lambda x: labels_dict[x])
        )
        return tractometry
    else:
        label_ids = labels_df.columns.drop(["subjectID", "sessionID"])
        data = pd.merge(
            left=tractometry,
            right=labels_df,
            left_on=["subjectID", "sessionID"],
            right_on=["subjectID", "sessionID"],
        )
        data = data.rename(
            columns={
                oc: oc.split("_")[1].upper()
                for oc in data.columns
                if isinstance(oc, str) and re.match(r"^d[tk]{1}i_", oc)
            }
        )
        # data = data.assign(tractID=data.tractID.apply(lambda x: map_bundles[x]))
        if long_type:
            id_vars = ["tractID", "nodeID", "subjectID", "sessionID"]
            id_vars.extend(label_ids)
            print(data.columns)
            data = data.melt(
                id_vars=id_vars,
                value_vars=["FA", "MD", "AD", "GA", "RD"],
                var_name="dti_measure",
                value_name="dti_value",
            )
        return data


def categorize(
    df: pd.DataFrame,
    col_name: str,
    new_grouping: dict,
    categorical: bool,
    label_order: Union[list, None] = None,
) -> pd.DataFrame:
    """Categorizes a given column of a dataframe based on provided new grouping
        information.

    Args:
        df (pd.DataFrame): dataframe containing the column to be (re)categorized
        col_name (str): name of column to be (re)categorized
        new_grouping (dict): if data is categorical, use a dictionary to map old values
            to new values; if data is continuous, use dict, which keys indicate where to
            cut (will add the max automatically), and values are the new name to use
        categorical (boolean)
        label_order (list): order for the categorical data (defaults to None)
    Returns:
        pd.DataFrame: dataframe with (re)categorized column
    """

    if categorical:
        df = df.replace({col_name: new_grouping})
        if label_order is None:
            label_order = np.unique(new_grouping.values())
        df[col_name] = pd.Categorical(
            df[col_name], categories=label_order, ordered=True
        )
    else:
        # new_grouping.append(np.ceil(df[col_name].max()))
        cuts = list(new_grouping.keys())
        series_max = np.ceil(df[col_name].max())
        if series_max > cuts[-1]:
            cuts.append(series_max)
        df = df.assign(
            **{
                col_name: pd.cut(
                    df[col_name], cuts, labels=list(new_grouping.values()), right=False
                )
            }
        )
    return df


def load_json(experiment_path: Path, label_id: str) -> tuple:
    """Loads json file associated with experiment and returns new mapping associated
        with a given variable.

    Args:
        experiment_path (Path): path of experiment json file
        label_id (str): variable name

    Returns:
        tuple: (type of data for label_id, new_label_id, dictionary mapping how to
            categorize it, group order if exists)
    """

    with open(experiment_path, "r") as fp:
        full_dic = json.load(fp)

    my_dict = full_dic.get(label_id)
    if my_dict:
        categorical = my_dict.get("categorical")
        key_dict = my_dict.get("label_dict")
        if not categorical:
            force_categorical = my_dict.get("force_categorical")
            if force_categorical:
                key_dict = {float(key): value for key, value in key_dict.items()}
        if not (new_label_id := my_dict.get("pretty_name")):
            new_label_id = label_id
        label_order = my_dict.get("label_order")
        custom_function = my_dict.get("function")
        return (categorical, new_label_id, key_dict, label_order, custom_function)
    else:
        return (None, label_id, None, None, None)


# Manipulate dataframe


def center_cut(df: pd.DataFrame, cut: tuple = (25, 75)) -> pd.DataFrame:
    """Returns dataframe where the nodeID is cut between two indicated values.

    Args:
        df (pd.DataFrame): dataframe with column `nodeID` to be filtered on.
        cut (tuple, optional): extreme values to filter nodeID on. Defaults to (25,75).

    Returns:
        pd.DataFrame: filtered dataframe
    """

    df = df[(df.nodeID >= cut[0]) & (df.nodeID <= cut[1])]
    return df.reset_index(drop=True)


def get_dti(df: pd.DataFrame, dti_measure: str) -> pd.DataFrame:
    """Simplifies dataframe by taking only the dti_measure indicated

    Args:
        df (pd.DataFrame): dataframe to act on
        dti_measure (str): dti measure to keep ('FA', ...)

    Returns:
        pd.DataFrame: dataframe with only dti_measure
    """

    df = df[df.dti_measure == dti_measure]
    df = df.drop(columns=["dti_measure"])
    df = df.rename(columns={"dti_value": dti_measure})
    return df.reset_index(drop=True)


def filter_sessions(
    data: pd.DataFrame, sessions=["baselineYear1Arm1", "2YearFollowUpYArm1"]
) -> pd.DataFrame:
    """Filters dataframe to keep only subjects with data for the sessions

    Args:
        data (pd.DataFrame): tractometry dataframe
        sessions (list, optional): list of session names. If "all", takes all sessions
            available and find subjects with data for all of them.
            Defaults to ['baselineYear1Arm1','2YearFollowUpYArm1'].

    Returns:
        pd.DataFrame: filtered dataframe
    """

    if sessions == "all":
        sessions = data.sessionID.unique()
    subjects = (
        data.groupby(["subjectID", "sessionID"])[["tractID"]]
        .count()
        .reset_index()
        .pivot(index="subjectID", columns="sessionID")["tractID"][sessions]
        .dropna()
        .index.tolist()
    )
    # consider this:
    # gp = df.groupby('subjectID').agg({
    #     'sessionID': 'unique',
    # })
    # all(s in gp.iloc[5,0] for s in sessions)
    print(f"{len(subjects)} subjects with {sessions=}")
    return data[data.subjectID.isin(subjects)]


def load_PCs(genetics_fp: Path, num_PC: int = 10) -> pd.DataFrame:
    """Loads principal components of genetic information

    Args:
        genetics_fp (Path): path to genetic Principal Genetic Components file
        num_PC (int, optional): the number of principal components to load.
            Defaults to 10.

    Returns:
        pd.DataFrame: dataframe of genetic principal components by subject
    """

    PCs_df = (
        pd.read_csv(genetics_fp, sep="\t")
        .rename(columns={"IID": "subjectID"})
        .drop(columns=["FID"])
    )
    PCs_df = PCs_df.assign(
        subjectID=PCs_df.subjectID.apply(lambda x: x.split("INV")[1])
    )
    PCs_df = PCs_df.drop(
        columns=[
            pc
            for pc in PCs_df.columns.drop("subjectID")
            if type(pc) is str and int(pc.split("C")[1]) > num_PC
        ]
    )
    return PCs_df


def beautify(tracto: pd.DataFrame) -> pd.DataFrame:
    """Makes tracto_df ready for plotting

    Args:
        tracto (pd.DataFrame): tractometry dataframe.

    Returns:
        pd.DataFrame: tractometry dataframe, with embelishments for
    """
    if "tractID_b" not in tracto.columns:
        tracto = tracto.assign(tractID_b=tracto.tractID.map(BUNDLE_DICT))
    return tracto
