from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import json
from abcd_utils import utils
import warnings

DATA_DIR = Path("/wynton/group/abcd/")

sessionID_sort = ["Baseline"] + [f"Year {i}" for i in range(1, 10)]


def search_nda(search_term: str):
    response = requests.get(
        f"https://nda.nih.gov/api/datadictionary/dataelement/{search_term}"
    )
    return response.json()


def create_label_map(new_categories):
    """Generates a mapping of levels to their respective categories.

    Args:
        new_categories (dict): A dictionary with categories as keys and lists of levels
            as values.

    Returns:
        dict: A mapping where each level is a key and its corresponding category is the
            value.

    Example:
        Given a dictionary {"A": ["1", "2"], "B": ["3"]}, it returns
            {"1": "A", "2": "A", "3": "B"}.
    """
    label_map = {}
    for category, levels in new_categories.items():
        for level in levels:
            label_map[level] = category
    return label_map


class DataLoader:
    def __init__(self, version: str, data_dir: Path = DATA_DIR):
        self.version = version
        self.data_dir = Path(data_dir)
        assert self.data_dir.exists()
        self.utils_dir = self.data_dir / version / "utils"
        assert self.utils_dir.exists()
        self.var_list = self.load_var_list()

    def load_var_list(self):
        if self.version == "5.0":
            return pd.read_csv(self.utils_dir / f"data_dictionary_{self.version}.csv")
        elif self.version == "5.1":
            return pd.read_csv(self.utils_dir / f"data_dictionary_{self.version}.csv")
        elif (
            file_path := self.utils_dir / f"variable_list_v{self.version}.csv"
        ).exists():
            return pd.read_csv(file_path)
        else:
            raise Exception(
                f"No data dictionary for version {self.version} at\n{file_path}"
            )

    def find_variable(
        self, search_term: str, search_in=None, alias_search=True
    ) -> pd.DataFrame:
        """Find variable based on search string (can be regex). Option to choose
            dataframe to search in.

        Args:
            search_term (str): search string, can be regex
            search_in ([pd.DataFrame], optional): Dataframe to search in, e.g. to do
                chained search. Defaults to dd.
            alias_search (bool, optional): whether to perform search using alias as
                well. Defaults to True.

        Returns:
            pd.DataFrame: filtered dataframe

        Examples:
            Chained search example:
                dm = find_variable('desikan_mean')
                find_variable('dti', dm)
            Regex search equivalent:
                find_variable(r'dti.*desikan_mean|desikan_mean.*dti')
        """

        if search_in is None:
            search_in = self.var_list
        column_sets = [
            ["ElementName", "Aliases", "ElementDescription"],
            ["ElementName", "description", "instrument"],
            [
                "table_name",
                "var_name",
                "var_label",
                "notes",
                "condition",
                "table_name_nda",
            ],
        ]

        for cols in column_sets:
            if all(c in search_in.columns for c in cols):
                break
        else:
            raise Exception("Wrong input searchable dataframe")

        if alias_search:
            try:
                alias = search_nda(search_term)["name"]
            except:
                print("No alias found")
                alias = None
            if alias and alias != search_term:
                search_terms = [search_term, alias]
            else:
                search_terms = [search_term]
        else:
            search_terms = [search_term]

        res = search_in[
            np.logical_or.reduce(
                [
                    search_in[col].str.contains(search_term, na=False, case=False)
                    for col in cols
                    for search_term in search_terms
                ]
            )
        ]
        pd.set_option("display.max_colwidth", None)
        return res

    def get_variable_data(
        self, variable_name: str, mdf=None, try_substitution: bool = False
    ) -> pd.DataFrame:
        """Loads data corresponding to the specified variable from tabulated datafiles.

        Args:
            variable_name (str): NDA variable name
            mdf (pd.DataFrame, optional): Contains NDA variable name and path to
                corresponding file. Defaults to var_list.
            try_substitution (bool, optional): if set to True, will search equivalence
                to NDA name. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame with `subjectID`, `sessionID`, `variable_name`
        """
        if mdf is None:
            mdf = self.var_list
        var_name_list = ["var_name", "ElementName"]
        for var_name in var_name_list:
            if var_name in mdf.columns:
                break
        else:
            raise Exception(
                f"Wrong input dataframe: no column from {var_name_list} found"
            )
        if var_name in mdf.columns:
            mdf = mdf.set_index(var_name)
        if not isinstance(variable_name, str):
            raise ValueError(
                "Wrong `variable_name` parameter: needs to be a single string."
            )

        def variable_not_found(error_text: str = ""):
            return FileNotFoundError(f"This variable has not been found.\n{error_text}")

        if try_substitution and variable_name not in mdf.index:
            var_search = self.find_variable(variable_name)
            if len(var_search) == 1:
                variable_name = var_search[var_name].iloc[0]
            else:
                raise variable_not_found(
                    f"Searched for alternatives, {len(var_search)} found: "
                    f"{list(var_search[var_name])}"
                )
        try:
            if "file_path" in mdf.columns:
                file_path = self.data_dir / mdf.loc[variable_name].file_path
            elif "table_name" in mdf.columns:
                # print('variable_name', variable_name)
                gen = (self.data_dir / self.version / "tabulated").glob(
                    f"**/{mdf.loc[variable_name]['table_name']}*"
                )
                file_path = next(gen)
                # print(file_path)
                assert file_path is not None
        except:
            raise variable_not_found()
        sep = "\t" if file_path.suffix == ".txt" else ","
        skiprows = [1] if file_path.suffix == ".txt" else None
        vars_to_keep = ["src_subject_id", "eventname", variable_name]
        df = pd.read_csv(file_path, sep=sep, skiprows=skiprows, header=0)[vars_to_keep]
        df = df.drop_duplicates()
        df = utils.unify_naming(df)
        return df

    def get_variables_data(
        self, variable_list: list, try_substitution: bool = False
    ) -> pd.DataFrame:
        """Loads data corresponding to the specified variable list from tabulated
            datafiles. Wrapper of get_variable_data.

        Args:
            variable_list (list): NDA variable name list
            try_substitution (bool, optional): if set to True, will search equivalence
                to NDA name. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame with `subjectID`, `sessionID`,
                `variable_name_1`, ...
        """
        dfs = []
        err = []
        for var in variable_list:
            try:
                dfs.append(
                    self.get_variable_data(var, try_substitution=try_substitution)
                )
            except:
                err[var] = self.find_variable(var)
        df = reduce(
            lambda df1, df: pd.merge(
                df1, df, on=["subjectID", "sessionID"], how="inner"
            ),
            dfs,
        )
        return utils.unify_naming(df)

    def get_levels(self, variable: str):
        try:
            nda_dict = search_nda(variable)
        except:
            print("No nda entry found")
            return None
        return {k.split(" = ")[0]: k.split(" = ")[1] for k in nda_dict.split(" ; ")}

    def get_var_list(self, rootstr: str) -> list:
        search = self.find_variable(rootstr)
        search = search[search.var_name.str.contains(rootstr)]
        return search.var_name.to_list()

    def consolidate_session_data(
        self, var_names: list, new_name: str, merge_func=np.mean
    ) -> pd.DataFrame:
        """Consolidates data across variables for each subjectID and sessionID, for
           variables that are session-specific.

        Args:
            var_names (list): list of variables names to consolidate.
                For example: ``self.get_var_list(var_rootstr)``
            new_name (str): name of the new variable to create

        Returns:
            pd.DataFrame: consolidated dataframe
        """
        df = self.get_variables_data(var_names)

        # Consolidate values across columns for each subjectID and sessionID
        def consolidate_row(row):
            non_nan_values = row[var_names].dropna().unique()
            if len(non_nan_values) > 1:
                warnings.warn(
                    f"Multiple distinct non-NaN values for subjectID {row['subjectID']}"
                    f" and sessionID {row['sessionID']}.\n"
                    f"Merging {non_nan_values} -> {merge_func(non_nan_values)}"
                )
            return merge_func(non_nan_values) if len(non_nan_values) > 0 else np.nan

        # Apply the consolidation function to each row
        df[new_name] = df.apply(consolidate_row, axis=1)
        # Drop the individual variable columns after consolidation
        df = df[["subjectID", "sessionID", new_name]]

        return df

    def get_demographics(
        self,
        demos: list = None,
        with_labels: bool = True,
        add_n_genetics: int = 10,
        use_new_cat: bool = True,
    ) -> pd.DataFrame:
        """Generate ABCD demographic dataframe.

        Args:
            demos (list, optional): list of demographics variable to load. Defaults to
                the ones included in the json file.
            with_labels (bool, optional): convert scales to labels. Defaults to True.
            add_n_genetics (int, optional): number of genetic principal components
                to add. Defaults to 10.
            use_new_cat (bool, optional): whether to use the new categories defined in
                the json file. Defaults to True.

        Returns:
            pd.DataFrame: ABCD demographic dataframe
        """
        demos_json_fp = Path(__file__).parent / "variable_mapping.json"
        assert demos_json_fp.exists()
        with open(demos_json_fp, "r") as f:
            demos_dict = json.load(f)
        if demos is None:
            demos = list(demos_dict.keys())
        vars_to_load = [
            newvar
            for var in demos
            if var in demos_dict
            for rootstr in demos_dict[var]["roots"][self.version]
            for newvar in self.get_var_list(rootstr)
        ] + [var for var in demos if var not in demos_dict]
        df = self.get_variables_data(vars_to_load)
        # Merge variables that need to be merged
        for new_var_name in demos:
            if new_var_name not in demos_dict:
                continue
            roots = demos_dict[new_var_name]["roots"][self.version]
            var_names = [vn for rootstr in roots for vn in self.get_var_list(rootstr)]
            dunno_map = {777: 0, 999: 0}
            # Concatenate values since they were populated by session
            for rootstr in roots:
                df[rootstr] = (
                    df.filter(like=rootstr).bfill(axis=1).iloc[:, 0].replace(dunno_map)
                )
            # Get maximum value between 2 parents
            df[new_var_name] = (
                df[roots].max(axis=1).replace({v: k for k, v in dunno_map.items()})
            )
            if new_var_name not in roots:
                df = df.drop(columns=roots + var_names)
            if demos_dict[new_var_name].get("fill"):
                # fill data from baseline event to all other events, using subjectID
                # as key
                df[new_var_name] = df.groupby("subjectID")[new_var_name].ffill()
        if with_labels:
            cat_vars = [
                var
                for var in demos
                if var in demos_dict
                and ("levels" in demos_dict[var] or "new_categories" in demos_dict[var])
            ]
            for var in cat_vars:
                # Determine the appropriate mapping for each variable
                if use_new_cat and "new_categories" in demos_dict[var]:
                    label_map = create_label_map(demos_dict[var]["new_categories"])
                elif "levels" in demos_dict[var]:
                    label_map = demos_dict[var]["levels"]
                else:
                    continue  # Skip if no appropriate mapping is found
                # Apply the mapping
                df[var] = df[var].fillna(-1).astype(int).astype(str).replace(label_map)
                # Revert -1s to NaNs and set the column as a categorical type
                df[var] = (
                    df[var]
                    .replace({"-1": pd.NA})
                    .astype("category")
                    .cat.remove_unused_categories()
                )
        if add_n_genetics > 0:
            to_add = [f"genetic_pc_{i+1}" for i in range(add_n_genetics)]
            gen_df = self.get_variables_data(to_add)
            df = utils.abcd_merge([df, gen_df])
            for col in to_add:  # fill genetic data from baseline event to other events
                df[col] = df.groupby("subjectID")[col].ffill()

        df["sessionID"] = (
            df["sessionID"].astype("category").cat.remove_unused_categories()
        )
        return df


def create_data_file_linker(version="4.0", data_dir=DATA_DIR):
    """Creates data file linker to help loading variables faster from abcd-sync data.

    Args:
        version (str, optional): Data version. Defaults to '4.0'.
    """
    tab_dir = data_dir / f"{version}/tabulated"
    utils_dir = data_dir / f"{version}/utils"
    utils_dir.mkdir(exist_ok=True)
    csv_dirs = [tab_dir / "img", tab_dir / "non_img", tab_dir / "released"]
    ff = [f for d in csv_dirs for f in d.iterdir() if f.is_file()]

    def load_file(file_path: Path) -> pd.DataFrame:
        with open(file_path, "r") as fp:
            if file_path.suffix == ".txt":
                sep = "\t"
                columns = fp.readline().strip().replace('"', "").split(sep)
                expl = fp.readline().strip().replace('"', "").split(sep)
            elif file_path.suffix == ".csv":
                sep = ","
                columns = fp.readline().strip().replace('"', "").split(sep)
                expl = [] * len(columns)
            else:
                raise ValueError(f"Unknown file extension for {file_path}.")
        # assert len(columns) == len(expl)
        if not all(
            col in columns
            for col in [
                "src_subject_id",
            ]
        ):
            raise ValueError("Missing columns")
        myd = {
            col: [ex, file_path.relative_to(data_dir)] for col, ex in zip(columns, expl)
        }
        return pd.DataFrame.from_dict(myd, orient="index").rename(
            columns={0: "description", 1: "file_path"}
        )

    list_df = []
    errors = []
    for myf in ff:
        try:
            df = load_file(myf)
            list_df.append(df)
        except Exception as e:
            print(myf, e)
            errors.append(myf)
    print(f"{len(errors)=}, {len(list_df)=}")
    df = pd.concat(list_df).reset_index()
    df = df[~df.duplicated(subset=["index", "description"], keep="first")].rename(
        columns={"index": "variable"}
    )
    df.to_csv(utils_dir / f"variable_list_v{version}.csv", index=False)
    df.to_pickle(utils_dir / f"variable_list_v{version}.pkl")
    return


def create_var_list_nda(version="4.0", data_dir=DATA_DIR, overwrite=False):
    """Creates data file linker to help loading variables faster from nda released data.

    Args:
        version (str, optional): Data version. Defaults to '4.0'.
    """
    tab_dir = data_dir / f"{version}/tabulated"
    if not tab_dir.exists():
        raise FileNotFoundError(f"No tabulated data available for {version=}.")
    utils_dir = data_dir / f"{version}/utils"
    utils_dir.mkdir(exist_ok=True)
    out_fname = utils_dir / f"variable_list_v{version}.csv"
    if out_fname.exists() and not overwrite:
        raise FileExistsError(
            f"Variable linker already exists! use `overwrite=True` to bypass.\n"
            f"{out_fname}"
        )

    ff = [f for f in tab_dir.iterdir() if f.is_file() and f.suffix in [".txt", ".csv"]]
    print(f"Gathering {len(ff)} files...")

    def load_file(file_path: Path) -> pd.DataFrame:
        required_columns = [
            "src_subject_id",
        ]
        with open(file_path, "r") as fp:
            if file_path.suffix == ".txt":
                sep = "\t"
                columns = fp.readline().strip().replace('"', "").split(sep)
                expl = fp.readline().strip().replace('"', "").split(sep)
            elif file_path.suffix == ".csv":
                sep = ","
                columns = fp.readline().strip().replace('"', "").split(sep)
                expl = [] * len(columns)
            else:
                raise ValueError(f"Unknown file extension for {file_path}.")
        # assert len(columns) == len(expl)
        if not all(col in columns for col in required_columns):
            raise ValueError("Missing columns")
        myd = {
            col: [ex, file_path.relative_to(data_dir)] for col, ex in zip(columns, expl)
        }
        return pd.DataFrame.from_dict(myd, orient="index").rename(
            columns={0: "description", 1: "file_path"}
        )

    list_df = []
    errors = []
    for myf in ff:
        try:
            df = load_file(myf)
            list_df.append(df)
        except Exception as e:
            print(myf, e)
            errors.append(myf)
    print(f"{len(errors)=}, {len(list_df)=}")
    df = pd.concat(list_df).reset_index()
    df = df[~df.duplicated(subset=["index", "description"], keep="first")].rename(
        columns={"index": "ElementName"}
    )
    df.to_csv(out_fname, index=False)
    df.to_pickle(out_fname.with_suffix(".pkl"))
    return df
