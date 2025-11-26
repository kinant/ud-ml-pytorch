import numpy as np
import pandas as pd
import gc

from sklearn.preprocessing import (OneHotEncoder, StandardScaler, FunctionTransformer, OrdinalEncoder)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import set_config

# So that our sklearn estimators/transformers output a pandas dataframe
set_config(transform_output="pandas")

from pprint import pprint
import json

BINARY = "binary"
BIN_NO_NUM = "non_numeric_binary"
NUMERIC = "numeric"
ORDINAL = "ordinal"
INTERVAL = "interval"

CATEGORICAL = "categorical"
MULTI = "multi_categorical"
MIXED = "mixed"
MIXED_ENG = "mixed_engineered"
OTHER = "other"

ENGINEERED_FEATURES = "engineered_features"
ENGINEERED_CAT= "engineered_categorical"
ENCODED_FEATURES = "encoded_features"

# For feature actions
TO_ENGINEER = "features_to_engineer"
TO_DROP = "features_to_drop"
TO_KEEP = "features_to_keep"
TO_SCALE = "features_to_scale"

# Re-encoded feature
OST_WEST = 'OST_WEST_KZ'

# Features to engineer
PRA_JUG = 'PRAEGENDE_JUGENDJAHRE'
CAM_INT = 'CAMEO_INTL_2015'
WOHN = 'WOHNLAGE'
LP_LEB = 'LP_LEBENSPHASE_FEIN'

# Engineered (new) features
P_DECADE = 'PRAEGENDE_DECADE'
P_MOVEMENT = 'PRAEGENDE_MOVEMENT'

C_WEALTH = 'CAMEO_WEALTH'
C_LIFE = 'CAMEO_LIFESTAGE'

WOHN_QUAL = 'WOHN_NEIGHBORHOOD_QUALITY'
WOHN_RURAL = 'WOHN_IS_RURAL'
WOHN_BUILD = 'WOHN_NEW_BUILDING'

LP_STAGE = 'LP_LIFE_STAGE'
LP_INCOME = 'LP_INCOME_LEVEL'
LP_AGE = 'LP_AGE_CLASS'
LP_HOMEOWN = 'LP_IS_HOMEOWNER'
LP_INDEP = 'LP_IS_INDEPENDENT'

# Data filenames
DATA_AZDIAS = "Udacity_AZDIAS_Subset.csv"

# FEATURE SUMMARY COLUMN LABLES
FS_TYPE = "type"
FS_ATTR = "attribute"
FS_OBJ = "object"
FS_MISS_UNKWN = "missing_or_unknown"

# FEATURES REMOVED BY COLUMN THRESHOLD
NA_FEATURES = "na_features"

MAPPINGS = {
    PRA_JUG: {
        P_DECADE: {
            1: 1940, 2: 1940, 3: 1950, 4: 1950, 5: 1960, 6: 1960, 7: 1960, 8: 1970, 9: 1970, 10: 1980,
            11: 1980, 12: 1980, 13: 1980, 14: 1990, 15: 1990
        },
        P_MOVEMENT: {
            1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 0, 8: 1, 9: 0, 10: 1, 11: 0, 12: 1, 13: 0, 14: 1,
            15: 0
        }
    },
    CAM_INT: {
        C_WEALTH: lambda x: int(x) // 10 if not pd.isna(x) else np.nan,
        C_LIFE: lambda x: int(x) % 10 if not pd.isna(x) else np.nan
    },
    LP_LEB: {
        LP_STAGE: {
            # Single
            1: 1,  2: 1,  3: 1,  4: 1,  5: 1, 6: 1,  7: 1,  8: 1,  9: 1, 10: 1, 11: 1, 12: 1, 13: 1,
            # Couple
            14: 2, 15: 2, 16: 2, 17: 2, 18: 2, 19: 2, 20: 2,
            # Single Parent
            21: 3, 22: 3, 23: 3,
            # Family
            24: 4, 25: 4, 26: 4, 27: 4, 28: 4,
            # Multiperson
            29: 5, 30: 5, 31: 5, 32: 5, 33: 5, 34: 5, 35: 5, 36: 5, 37: 5, 38: 5, 39: 5, 40: 5,
        },
        LP_INCOME: {
            # Low Income
            1: 1,  2: 1,  5: 1,  6: 1, 14: 1, 15: 1, 21: 1, 24: 1, 29: 1, 31: 1,
            # Average Income
            3: 2,  4: 2,  7: 2,  8: 2, 16: 2, 22: 2, 25: 2, 30: 2, 32: 2,
            # High Income
            23: 3,
            # Wealthy
            10: 4, 18: 4,
            # Top Earner
            13: 5, 20: 5, 28: 5, 35: 5, 39: 5, 40: 5,
        },
        LP_INDEP: {
            9: 1, 17: 1, 26: 1, 33: 1, 36: 1,
        },
        LP_AGE: {
            # Younger age
            1: 1,  3: 1, 14: 1, 18: 1, 29: 1,
            30: 1, 33: 1, 34: 1, 35: 1,
            # Middle age
            2: 2,  4: 2, 39: 2,
            # Higher age
            13: 3, 15: 3, 16: 3, 19: 3,
            20: 3, 31: 3, 32: 3, 36: 3,
            # Advanced age
            5: 4,  7: 4, 11: 4, 37: 4,
            # Retirement age
            6: 5,  8: 5, 12: 5, 38: 5, 40: 5,
        },
        LP_HOMEOWN: {
            10: 1, 11: 1, 12: 1, 18: 1, 19: 1,
            27: 1, 34: 1, 37: 1, 38: 1,
        }
    },
    WOHN: {
        WOHN_QUAL: {
            1: 1,   # very good
            2: 2,   # good
            3: 3,   # average
            4: 4,   # poor
            5: 5,   # very poor
        },
        WOHN_RURAL: {7: 1, 8: 1},
        WOHN_BUILD: {8: 1}
    }
}

# CHOSEN FEATURES TO DROP

# ============================================================================
# Custom Transformers
# ============================================================================
class ReplaceMissingTransformer(BaseEstimator, TransformerMixin):
    """Replace missing value codes with NaN"""

    def __init__(self, feature_summary):
        self.feature_summary = feature_summary
        self.missing_summary = self._get_missing_summary()
        self.converted_na_total = 0

    def _get_missing_summary(self):
        """
        Function that returns a dictionary mapping each attribute in a feature summary
        To a list of the missing or unknown codes for that attribute
        :param fs: Feature summary
        :return: Dictionary mapping attribute to a list of missing or unknown codes
        """

        def get_numeric_value(s):
            """
            Basic function that attempts to cast a string into an int.
            :param s: the string to cast to int
            :return: the integer represented by the string or the original string if invalid string
            """
            try:
                return int(s)
            except ValueError:
                return s

        # Init dictionary
        missing_summary = {}

        # iterate over attributes (to use as key) and missing_or_unknown values (as strings now)
        for attribute, missing_code_str in zip(self.feature_summary[FS_ATTR], self.feature_summary[FS_MISS_UNKWN]):

            # clean up the string and create a list
            temp_list = list(missing_code_str.replace(" ", "").replace("[", "").replace("]", "").split(","))

            # check that the list is not empty
            if len(temp_list) > 0 and temp_list[0] != "":
                # convert strings into numerical values
                missing_summary[attribute] = [get_numeric_value(s) for s in temp_list]

        return missing_summary

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()

        # we iterate over each attribute and list pair in the dictionary
        # we don't iterate over each column, as not all of them have missing_or_unknown codes
        for attribute, lst in self.missing_summary.items():

            # Skip over any attribute that is not in the DataFrame
            if attribute not in X:
                continue

            # Get the sum of missing or unknown values for that attribute/feature in the general pop. data
            # We use isin() function to match with the list of missing_or_unknown values and then sum them up
            n_miss = X[attribute].isin(lst).sum()

            # Replace all the values with NaN
            X_out[attribute] = X_out[attribute].replace(lst, np.nan)

            # increment total
            self.converted_na_total += n_miss

        return X_out

class FeatureDropper(BaseEstimator, TransformerMixin):
    """
    Custom transformer that drops features from a dataset.
    This class is designed so that a single feature dropper can be used
    to drop features at several places, keeping track of relevant information.
    More than one can also be used for different purposes.
    """

    def __init__(self):
        self.all_dropped_features = [] # keep track of all the features dropped by an instance
        self.last_dropped_features = [] # keep track of the last features that were dropped
        self.features_to_drop = [] # init list of the features to drop
        self.n_dropped_last = 0 # count of the last features dropped
        self.n_total_dropped = 0 # count of the total features drop

    def set_features_to_drop(self, features_to_drop):
        """
        Sets the features that will be dropped
        :param features_to_drop: features that will be dropped
        """
        self.features_to_drop = features_to_drop.copy()

    def check_features_set(self):
        """
        Checks that the features to drop list is valid
        :return:
        """
        if self.features_to_drop is None or len(self.features_to_drop) == 0:
            print(f"Features to drop not set. Please set by calling set_features_to_drop first.")
            print(f"No changes will be made to dataframe")
            return False
        return True

    def fit(self, X, y=None):

        # Keep track of the list of features coming in
        self.features_in_ = X.columns.to_list()
        return self

    def transform(self, X):

        # Check that we have set the features to drop
        self.check_features_set()
        X = X.copy()

        # Reset attributes
        self.last_dropped_features = []
        self.n_dropped_last = 0

        print(f"Dropping Features: {self.features_to_drop}")

        # Iterate over each feature set to be dropped
        for feature in self.features_to_drop:
            try:
                X.drop(feature, axis=1, inplace=True)
            except KeyError:
                print(f"{feature} not found in dataframe")
            else:
                # Update attributes
                self.all_dropped_features.append(feature)
                self.last_dropped_features.append(feature)
                self.n_dropped_last += 1
                self.n_total_dropped += 1

        # Clear the features to drop, for the next use
        self.features_to_drop.clear()
        print(f"Number of features dropped: {self.n_dropped_last}")
        return X

    def get_feature_names_out(self):
        """
        We need to set the get_feature_names_out when we use custom transformers
        So that we can get the correct feature/columns out after transformation
        :return:
        """

        # Return those features that came in that are not any that were dropped
        # In other words, the remaining features
        return [feature
                for feature in self.features_in_
                if feature not in self.last_dropped_features]

    def get_n_total_dropped(self):
        return self.n_total_dropped

    def get_n_last_dropped_features(self):
        return self.n_dropped_last

    def get_last_dropped_features(self):
        return self.last_dropped_features

    def get_all_dropped_features(self):
        return self.all_dropped_features

"""
PREPROCESSING PIPELINE SUMMARY
===============================================================================
PIPELINE STEPS:
---------------

1. IMPUTE MISSING VALUES (ReplaceMissingTransformer)

2. DROP HIGH-MISSING COLUMNS (HighMissingColDropper)

3. SPLIT DATASET BY ROW MISSINGNESS (DatasetSplitter)

4. ENGINEER MIXED-TYPE FEATURES (MixedTypeEngineer)

5. ENCODE CATEGORICAL FEATURES (CategoricalEncoder)

6. DROP ORIGINAL ENCODED FEATURES (FeatureDropper)

7. STANDARDIZE FEATURES (StandardScaler)
"""

class AzdiasPreprocessor():

    def __init__(self, data_source, summary_source, na_feature_threshold=20, na_sample_threshold=25):
        self.data_source_ = data_source
        self.summary_source_ = summary_source
        self.na_feature_threshold_ = na_feature_threshold
        self.na_sample_threshold_ = na_sample_threshold

        self.feature_groups_ = {
            BINARY: {
                P_MOVEMENT, LP_HOMEOWN, LP_INDEP, WOHN_RURAL, WOHN_BUILD
            },
            BIN_NO_NUM: set(),
            ORDINAL: {C_WEALTH, C_LIFE, LP_INCOME, LP_AGE, WOHN_QUAL},
            NUMERIC: set(),
            INTERVAL: {P_DECADE},
            MULTI: set(),
            ENGINEERED_FEATURES: {
                P_DECADE, P_MOVEMENT, C_WEALTH, C_LIFE, LP_INCOME,
                LP_INDEP, LP_AGE, LP_HOMEOWN, WOHN_QUAL, WOHN_RURAL,
                WOHN_BUILD
            },
            ENGINEERED_CAT: {LP_STAGE},
            ENCODED_FEATURES: set(),
            TO_ENGINEER: {
                PRA_JUG, CAM_INT, WOHN, LP_LEB
            },
            TO_SCALE: set(),
            TO_DROP: {
                NA_FEATURES: set(),
                OTHER: {
                    "LP_FAMILIE_GROB", "LP_STATUS_GROB", "NATIONALITAET_KZ", "SOHO_KZ",
                    "CAMEO_DEUG_2015", "LP_LEBENSPHASE_GROB", "PLZ8_BAUMAX"
                }
            }
        }

        self.feature_summary_ = {}
        self.feature_dictionary_ = {}

        self._load_data()

    def _set_na_features_to_drop(self, df):

        replace_missing_transformer = ReplaceMissingTransformer(self.feature_summary_)

        df = replace_missing_transformer.fit_transform(df)

        df_missing_pct = (df.isna().sum().sort_values() / df.shape[0]) * 100

        self.feature_groups_[TO_DROP][NA_FEATURES].update(
            df_missing_pct[df_missing_pct > 20].index.tolist()
        )

        del df_missing_pct
        del df

    def _load_data(self):

        df_population_raw = pd.read_csv(self.data_source_, delimiter=";")
        self.feature_summary_ = pd.read_csv(self.summary_source_, delimiter=";")

        self._set_na_features_to_drop(df_population_raw)
        self._set_feature_groups(df_population_raw)

        del df_population_raw
        del self.feature_summary_

    def _set_feature_groups(self, df):

        self._set_na_features_to_drop(df)

        feature_list =  [
            f for f in df.columns.to_list()
            if f not in self.feature_groups_[TO_DROP][NA_FEATURES]
        ]

        ordinal_features = list(
            self.feature_summary_[
                (self.feature_summary_[FS_ATTR].isin(feature_list)) &
                (self.feature_summary_[FS_TYPE] == ORDINAL)][FS_ATTR]
        )

        numerical_features = list(
            self.feature_summary_[
                (self.feature_summary_[FS_ATTR].isin(feature_list)) &
                (self.feature_summary_[FS_TYPE] == NUMERIC)][FS_ATTR]
        )

        self.feature_groups_[ORDINAL].update(ordinal_features)
        self.feature_groups_[NUMERIC].update(numerical_features)

        categorical_features = list(
            self.feature_summary_[
                (self.feature_summary_[FS_ATTR].isin(feature_list)) &
                (self.feature_summary_[FS_TYPE] == CATEGORICAL)][FS_ATTR]
        )

        # iterate over each categorical
        for feature in categorical_features:

            # skip over the dropped categorical features
            if feature in self.feature_groups_[TO_DROP][OTHER]:
                # continue and do not append to any list
                continue

            # check for binary ones
            if df[feature].nunique() == 2:

                # check if numeric or not

                if df[feature].dtype == FS_OBJ:

                    if feature not in self.feature_groups_[BIN_NO_NUM]:
                        self.feature_groups_[BIN_NO_NUM].add(feature)
                else:
                    if feature not in self.feature_groups_[BINARY]:
                        self.feature_groups_[BINARY].add(feature)

            # else, it is a multi-level categorical
            else:
                if feature not in self.feature_groups_[MULTI]:
                    self.feature_groups_[MULTI].add(feature)