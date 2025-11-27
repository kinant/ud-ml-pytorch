import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import set_config
from sklearn.utils.validation import check_is_fitted

# So that our sklearn estimators/transformers output a pandas dataframe
set_config(transform_output="pandas")

BINARY = 'binary'
BIN_NO_NUM = 'non_numeric_binary'
NUMERIC = 'numeric'
ORDINAL = 'ordinal'
INTERVAL = 'interval'

CATEGORICAL = 'categorical'
MULTI = 'multi_categorical'
MIXED = 'mixed'
MIXED_ENG = 'mixed_engineered'
OTHER = 'other'

ENGINEERED_FEATURES = 'engineered_features'
ENGINEERED_CAT= 'engineered_categorical'
ENCODED_FEATURES = 'encoded_features'

# For feature actions
TO_ENGINEER = 'features_to_engineer'
TO_DROP = 'features_to_drop'
TO_KEEP = 'features_to_keep'
TO_SCALE = 'features_to_scale'

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
DATA_AZDIAS = 'Udacity_AZDIAS_Subset.csv'

# CATEGORICALS TO DROP:
CATEGORICALS_TO_DROP = [
    "LP_FAMILIE_GROB",
    "LP_STATUS_GROB",
    "NATIONALITAET_KZ",
    "SOHO_KZ",
    "CAMEO_DEUG_2015"
]

# FEATURE SUMMARY COLUMN LABELS
FS_TYPE = 'type'
FS_ATTR = 'attribute'
FS_OBJ = 'object'
FS_MISS_UNKWN = 'missing_or_unknown'

# FEATURES REMOVED BY COLUMN THRESHOLD
NA_FEATURES = 'na_features'

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

# PIPELINE CLEANING STEPS
REPLACE_MISSINGS = 'replace_missings'
DROP_FEATURES = 'drop_features'
SPLIT_DATA = 'split_data'
DROP_CATEGORICAL = 'drop_categorical'
ENGINEER = 'engineer'
ENCODE = 'encode'

# PIPELINE IMPUTING AND SCALING STEPS
IMPUTE = 'impute'
SCALE = 'scale'

# ============================================================================
# Custom Transformers
# ============================================================================
class ReplaceMissingTransformer(BaseEstimator, TransformerMixin):
    """Replace missing value codes with NaN"""

    def __init__(self, feature_summary):
        self.feature_summary = feature_summary
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

    def fit(self, X):

        self.missing_summary_ = self._get_missing_summary()

        return self

    def transform(self, X):

        # Check that instance has been fitted
        check_is_fitted(self)

        X_out = X.copy()

        counter = 0
        # we iterate over each attribute and list pair in the dictionary
        # we don't iterate over each column, as not all of them have missing_or_unknown codes
        for attribute, lst in self.missing_summary_.items():

            # Skip over any attribute that is not in the DataFrame
            if attribute not in X_out:
                continue

            # Get the sum of missing or unknown values for that attribute/feature in the general pop. data
            # We use isin() function to match with the list of missing_or_unknown values and then sum them up
            n_miss = X_out[attribute].isin(lst).sum()

            # Replace all the values with NaN
            X_out[attribute] = X_out[attribute].replace(lst, np.nan)

            # increment total
            self.converted_na_total += n_miss
            counter += 1

        return X_out

    def set_output(self, *, transform=None):
        return self

class FeatureDropper(BaseEstimator, TransformerMixin):
    """
    Custom transformer that drops features from a dataset.
    This class is designed so that a single feature dropper can be used
    to drop features at several places, keeping track of relevant information.
    More than one can also be used for different purposes.
    """

    def __init__(self, features_to_drop):
        self.dropped_features = [] # keep track of all the features dropped by an instance
        self.features_to_drop = features_to_drop

    def fit(self, X):
        # Keep track of the list of features coming in
        self.features_in_ = X.columns.to_list()

        return self

    def transform(self, X):

        # Check that instance has been fitted
        check_is_fitted(self)

        X_out = X.copy()
        counter = 0

        print(f"Dropping Features: {self.features_to_drop}")
        # Iterate over each feature set to be dropped
        for feature in self.features_to_drop:
            try:
                X_out = X_out.drop(feature, axis=1)
                counter += 1
                self.dropped_features.append(feature)

            except KeyError:
                print(f"{feature} not found in dataframe")

        print(f"Dropped {counter} features")
        # Clear the features to drop
        self.features_to_drop.clear()

        return X_out

    def get_feature_names_out(self):
        """
        We need to set the get_feature_names_out when we use custom transformers
        So that we can get the correct feature/columns out after transformation
        :return:
        """

        # Return those features that came in that are not any that were dropped
        # In other words, the remaining features

        output_features = [feature
                for feature in self.features_in_
                if feature not in self.dropped_features]

        return output_features

class DatasetSplitter(BaseEstimator, TransformerMixin):
    """
    Custom Transformer that splits a dataset by threshold of the count of NaN values per row.
    This might not be the most correct way to do it, following sklearn conventions, because
    eventually we are returning 2 different subsets of the data but it works for this project.
    """

    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.missing_per_row_ = X.isna().sum(axis=1)
        return self

    def transform(self, X):

        # Check that instance has been fitted
        check_is_fitted(self)

        # Split the dataset into two, based on the threshold
        # Technically learned attributes should be set in fit()
        # But I do it here since I consider it transforming the data
        X_low = X[self.missing_per_row_ <= self.threshold]
        n_dropped = len(X) - len(X_low)

        # Reset index
        X_low = X_low.reset_index(drop=True)

        print(f"Kept {len(X_low)} rows, dropped {n_dropped} rows")

        return X_low

    def get_high_missing_subset(self, X):
        # Since a transformer can only return one dataset,
        # we use a separate function to return the high missings
        # subset

        # Check that the instance has been fitted
        check_is_fitted(self)

        X_high = X[self.missing_per_row_ > self.threshold]
        X_high = X_high.reset_index(drop=True)

        return X_high

    def set_output(self, *, transform=None):
        return self

class MixedFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Class that engineers mixed features
    """

    def __init__(self, mappings):
        self.mappings = mappings

    def fit(self, X, y=None):
        # Get the input features
        self.feature_names_in_ = X.columns.tolist()

        return self

    def transform(self, X):
        check_is_fitted(self)
        print(f"MixedFeatureEngineer Encoding...")
        X_out = X.copy()

        # Iterate over each nested dictionary
        for original_feature, new_feature_maps in self.mappings.items():
            # Iterate over each mapping dictionary for that new feature
            for new_feature, new_feature_map in new_feature_maps.items():
                # Perform the map
                X_out[new_feature] = X_out[original_feature].map(new_feature_map)
                print(f"MixedFeatureEngineer Encoded {new_feature} from {original_feature}")
            # Drop the original feature, ignore any errors if the feature is not found
            try:
                X_out = X_out.drop(original_feature, axis=1)
            except KeyError:
                print(f"MixedFeatureEngineer could not drop {original_feature}")
            else:
                print(f"MixedFeatureEngineer Dropped {original_feature}")

        return X_out

    def inverse_transform(self, X):
        # In this case, for analysis, we do not want to inverse transform
        # So we return the original data
        return X

    def get_feature_names_out(self, input_features=None):

        # Return output feature names by adding the new ones
        # Filter out the original features and extend the new ones
        output_features = [feature for feature in self.feature_names_in_
                          if feature not in [PRA_JUG, CAM_INT, LP_LEB, WOHN]]

        output_features.extend([P_DECADE, P_MOVEMENT, C_WEALTH, C_LIFE, LP_INCOME,
                       LP_INDEP, LP_AGE, LP_STAGE, LP_HOMEOWN, WOHN_QUAL, WOHN_RURAL, WOHN_BUILD])

        return output_features

class Encoder(BaseEstimator, TransformerMixin):

    def __init__(self, ordinal_features, onehot_features):
        self.ordinal_features = ordinal_features
        self.onehot_features = onehot_features

    def fit(self, X):

        print(f"Encoder Fitting Ordinal Encoder...")
        print(f"onehot_features: {self.onehot_features}")
        print(f"ordinal_features: {self.ordinal_features}")

        ordinal_encoder_out = []
        onehot_encoder_out = []

        self.output_features_ = [
            feature for feature in X.columns.tolist()
            if feature not in self.ordinal_features + self.onehot_features
        ]

        if self.ordinal_features:
            print(f"Encoder Creating and Fitting Ordinal Encoder...")
            self.ordinal_encoder_ = OrdinalEncoder(
                categories=[['O', 'W']],
                handle_unknown='use_encoded_value',
                unknown_value=np.nan,
                encoded_missing_value=np.nan
            )

            self.ordinal_encoder_.fit(X[self.ordinal_features])

            ordinal_encoder_out = self.ordinal_encoder_.get_feature_names_out()
            self.output_features_.extend(ordinal_encoder_out)

        if self.onehot_features:
            print(f"Encoder Creating and Fitting Onehot Encoder...")
            self.onehot_encoder_ = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.onehot_encoder_.fit(X[self.onehot_features])

            onehot_encoder_out = self.onehot_encoder_.get_feature_names_out()
            self.output_features_.extend(onehot_encoder_out)

        self.encoded_features_ = ordinal_encoder_out + onehot_encoder_out

        return self

    def transform(self, X):
        print(f"Encoder Engineering Features...")
        # Check that the instance is fitted
        check_is_fitted(self)

        X_out = X.copy()

        encoder = ColumnTransformer(
            transformers=[
                ('ordinal', self.ordinal_encoder_ if self.ordinal_features else "passthrough", self.ordinal_features),
                ('onehot', self.onehot_encoder_ if self.onehot_features else "passthrough", self.onehot_features)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        ).set_output(transform="pandas")

        X_out = encoder.fit_transform(X_out)

        onehot_features_out = encoder.named_transformers_["onehot"].get_feature_names_out() \
            if self.onehot_features else []

        print(f"Ordinal encoded {len(self.ordinal_features)} feature(s)")

        print(f"One-hot encoded {len(self.onehot_features)} feature(s) "
                f"into {len(onehot_features_out)} features")

        return X_out

    def get_feature_names_out(self, input_features=None):
        return self.output_features_

    def get_encoded_features_out(self, input_features=None):
        return self.encoded_features_

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

    def __init__(
            self,
            data_source,
            summary_source,
            na_feature_threshold=20,
            na_sample_threshold=25,
            cleaning_pipeline=None,
            impute_pipeline=None,
            scale_pipeline=None,
            split_data=False
    ):
        """
        Initialize the preprocessor.
        :param data_source: path to the general population data file
        :param summary_source: path to the feature summary file
        :param na_feature_threshold: percentage threshold for dropping high-missing features
        :param na_sample_threshold: threshold for splitting population data by row/sample missingness
        :param cleaning_pipeline: custom processing pipeline to apply to the data
        :param impute_pipeline: custom impute pipeline to apply to the data
        :param scaling_pipeline: custom scaling pipeline to apply to the data
        """
        self._data_source = data_source
        self._summary_source = summary_source
        self._na_feature_threshold = na_feature_threshold
        self._na_sample_threshold = na_sample_threshold

        self._feature_groups = {
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
                CATEGORICAL: {
                    "LP_FAMILIE_GROB", "LP_STATUS_GROB", "NATIONALITAET_KZ", "SOHO_KZ",
                    "CAMEO_DEUG_2015", "LP_LEBENSPHASE_GROB", "PLZ8_BAUMAX"
                }
            }
        }

        self._feature_summary = {}

        self._load_data()

        if cleaning_pipeline:
            self._cleaning_pipeline = cleaning_pipeline
        else:
            print(f"No pipeline parameter passed in. Building default pipeline...")
            self._cleaning_pipeline = self._build_default_cleaning_pipeline()

        # self._impute_scale_pipeline = None
        #
        # if impute_pipeline and scale_pipeline:
        #     self._impute_pipeline = impute_pipeline
        #     self._scale_pipeline = scale_pipeline
        # else:
        #     print(f"No impute and scale pipeline parameter passed in.")
        #     self._impute_pipeline = None
        #     self._scale_pipeline = None
        #
        # self._data_cleaned = False
        # self._data_imputed_scaled = False

        del self._feature_summary

    @property
    def feature_groups(self):
        return self._feature_groups

    @property
    def cleaning_pipeline(self):
        if not self._cleaning_pipeline:
            print(f"Pipeline not fitted yet. Returning unfitted pipeline...")
        return self._cleaning_pipeline

    # @property
    # def scale_pipeline(self):
    #     if not(self._data_imputed_scaled and self._data_imputed_scaled):
    #         print(f"Pipeline not fitted yet. Returning unfitted pipeline...")
    #     else:
    #         return self._scale_pipeline

    # @property
    # def impute_pipeline(self):
    #     if not(self._data_imputed_scaled and self._data_imputed_scaled):
    #         print(f"Pipeline not fitted yet. Returning unfitted pipeline...")
    #
    #     return self._impute_pipeline

    def _set_na_features_to_drop(self, df):

        replace_missing_transformer = ReplaceMissingTransformer(self._feature_summary)
        replace_missing_transformer.fit_transform(df)
        df = replace_missing_transformer.transform(df)

        df_missing_pct = (df.isna().sum().sort_values() / df.shape[0]) * 100

        self._feature_groups[TO_DROP][NA_FEATURES].update(
            df_missing_pct[df_missing_pct > 20].index.tolist()
        )

        del df_missing_pct
        return df

    def _load_data(self):
        print(f"Loading data from...{self._data_source}")
        df_population_raw = pd.read_csv(self._data_source, delimiter=";")
        self._feature_summary = pd.read_csv(self._summary_source, delimiter=";")

        self._set_feature_groups(df_population_raw)

        del df_population_raw

    def _set_feature_groups(self, df):
        df = self._set_na_features_to_drop(df)

        feature_list =  [
            f for f in df.columns.to_list()
            if f not in self._feature_groups[TO_DROP][NA_FEATURES]
        ]

        ordinal_features = list(
            self._feature_summary[
                (self._feature_summary[FS_ATTR].isin(feature_list)) &
                (self._feature_summary[FS_TYPE] == ORDINAL)][FS_ATTR]
        )

        numerical_features = list(
            self._feature_summary[
                (self._feature_summary[FS_ATTR].isin(feature_list)) &
                (self._feature_summary[FS_TYPE] == NUMERIC)][FS_ATTR]
        )

        self._feature_groups[ORDINAL].update(ordinal_features)
        self._feature_groups[NUMERIC].update(numerical_features)

        categorical_features = list(
            self._feature_summary[
                (self._feature_summary[FS_ATTR].isin(feature_list)) &
                (self._feature_summary[FS_TYPE] == CATEGORICAL)][FS_ATTR]
        )

        # iterate over each categorical
        for feature in categorical_features:

            # skip over the dropped categorical features
            if feature in self._feature_groups[TO_DROP][CATEGORICAL]:
                # continue and do not append to any list
                continue

            # check for binary ones
            if df[feature].nunique() == 2:
                # check if numeric or not
                if df[feature].dtype == FS_OBJ:

                    if feature not in self._feature_groups[BIN_NO_NUM]:
                        self._feature_groups[BIN_NO_NUM].add(feature)
                else:
                    if feature not in self._feature_groups[BINARY]:
                        self._feature_groups[BINARY].add(feature)

            # else, it is a multi-level categorical
            else:
                if feature not in self._feature_groups[MULTI]:
                    self._feature_groups[MULTI].add(feature)

    def _build_default_cleaning_pipeline(self):
        print(f"Building cleaning pipeline...")

        # 1. Replace Missings
        replace_missing_transformer = ReplaceMissingTransformer(self._feature_summary)

        # 2. Drop High NaN Features
        drop_na_features_transformer = FeatureDropper(features_to_drop=list(self._feature_groups[TO_DROP][NA_FEATURES]))

        # 3. Split data into two subsets
        splitter = DatasetSplitter(threshold=25)

        # 4. After splitting the data, we can proceed to drop the drop multi-level
        # categoricals chosen by examination. We drop these AFTER splitting the data
        # as to not affect the results of splitting
        drop_cat_features_transformer = FeatureDropper(features_to_drop=list(self._feature_groups[TO_DROP][CATEGORICAL]))

        # 4. Engineer mixed-type features
        engineer_transformer = MixedFeatureEngineer(mappings=MAPPINGS)

        # 5. Encode Features
        categorical_features = list(self._feature_groups[MULTI].union(self._feature_groups[ENGINEERED_CAT]))

        encoder = Encoder(
            ordinal_features=list(self._feature_groups[BIN_NO_NUM]),
            onehot_features=categorical_features
        )

        pipeline = Pipeline(
            steps=[
                (REPLACE_MISSINGS, replace_missing_transformer),
                (DROP_FEATURES, drop_na_features_transformer),
                (SPLIT_DATA, splitter),
                (DROP_CATEGORICAL, drop_cat_features_transformer),
                (ENGINEER, engineer_transformer),
                (ENCODE, encoder)
            ],
            verbose=True
        ).set_output(transform="pandas")

        return pipeline

    # def _set_impute_scale_pipeline(self, X):
    #     print(f"Building imputation and scaling pipeline...")
    #
    #     imputer = ColumnTransformer(
    #         transformers=[
    #             (BINARY, SimpleImputer(strategy='most_frequent'), list(self._feature_groups[BINARY])),
    #             (BIN_NO_NUM, SimpleImputer(strategy='most_frequent'), list(self._feature_groups[BIN_NO_NUM])),
    #             (ORDINAL, SimpleImputer(strategy='median'), list(self._feature_groups[ORDINAL])),
    #             (NUMERIC, SimpleImputer(strategy='median'), list(self._feature_groups[NUMERIC])),
    #             (INTERVAL, SimpleImputer(strategy='median'), list(self._feature_groups[INTERVAL]))
    #         ],
    #         remainder='passthrough',
    #         verbose_feature_names_out=False
    #     ).set_output(transform="pandas")
    #
    #     imputer.fit(X)
    #
    #     scaler = StandardScaler()
    #
    #     scaler.fit(X)
    #
    #     self._impute_pipeline = imputer
    #     self._scaler_pipeline = scaler

    def clean_data(self, X):
        X_out = X.copy()

        print(f"Cleaning data...")

        if self._cleaning_pipeline:
            X_out = self._cleaning_pipeline.fit_transform(X)
            self._data_cleaned = True
            self.feature_groups[ENCODED_FEATURES] = self._cleaning_pipeline.named_steps[ENCODE].get_encoded_features_out(X_out)
        else:
            print(f"No cleaning pipeline provided...returning original data")
            self._data_cleaned = False

        return X_out

    def impute_and_scale_data(self, X):

        X_out = X.copy()

        imputer = ColumnTransformer(
            transformers=[
                (BINARY, SimpleImputer(strategy='most_frequent'), list(self._feature_groups[BINARY])),
                (BIN_NO_NUM, SimpleImputer(strategy='most_frequent'), list(self._feature_groups[BIN_NO_NUM])),
                (ORDINAL, SimpleImputer(strategy='median'), list(self._feature_groups[ORDINAL])),
                (NUMERIC, SimpleImputer(strategy='median'), list(self._feature_groups[NUMERIC])),
                (INTERVAL, SimpleImputer(strategy='median'), list(self._feature_groups[INTERVAL]))
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        ).set_output(transform="pandas")

        scaler = StandardScaler()

        X_out = imputer.fit_transform(X_out)
        X_out = scaler.fit_transform(X_out)

        return X_out









