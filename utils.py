import pandas as pd
import numpy as np

from functools import reduce
from sklearn.model_selection import train_test_split


def load_data(all_feature_dict, set_index=None, index_dtype=None, how_join='left', verbose=False):
    """

    :param all_feature_dict: List of Dictionary of features.
                In the format of [{'filename': {'features': [], 'target': []}]
    :param set_index:
    :param index_dtype: String. Data type of the index. If set_index is none, this field does not matter.
    :param how_join: String. Joining method. Supports 'left', 'right', 'outer', 'inner' only.
    :param verbose:
    :return: merged dataframe
    """

    # Test the feature dict
    if verbose:
        print("Length of feature dictionary: %s" % len(all_feature_dict))

    features_list = []
    target = None

    for i, feature_dict in enumerate(all_feature_dict):

        if verbose:
            print("The filename of the first file: %s" % feature_dict['file'])
            print("The features to be extracted of the first file: %s" % feature_dict['features'])

        transformed_features, file_target = load_individual_file(feature_dict)

        if file_target is not None:
            # Set file target as target if it is not none
            target = file_target

        if set_index is not None:
            transformed_features = transformed_features.set_index(set_index)
            transformed_features.index = transformed_features.index.astype(index_dtype)

        # Drop unnecessary columns
        try:
            drop_columns = feature_dict['drop_columns']
            print(drop_columns)
            transformed_features = transformed_features.drop(columns=drop_columns, axis=1)
        except KeyError:
            print("No columns to be dropped specified in feature dictionary. No columns will be dropped.")

        features_list.append(transformed_features)

    # Join the dataframes
    df_merged = reduce(lambda left, right: pd.merge(left, right, how='left', left_index=True, right_index=True),
                       features_list)

    # Final check if target is present in the dataset, if not, raise error
    if target is None:
        raise ValueError("Target cannot be None. There should be at least one target to continue. ")
    else:
        assert type(target) == pd.Series
        print("Target column: %s" % target.name)

    return df_merged, target


def transform(dataframe, feature_param, verbose=False):
    """

    :param dataframe:
    :param feature_param: Dictionary. Contains, feature dtypes, transformation and aggregation.
    :param verbose:
    :return:
    """

    # 1. Coerce the dataset into the correct type
    try:
        for _, x in enumerate(feature_param['features']):

            target_type = x['dtype']

            if target_type is not None:

                if verbose:
                    print("Coercing dtype of %s to %s" % (x['column_name'], target_type))

                dataframe[x['column_name']] = dataframe[x['column_name']].astype(target_type)

                if target_type == "category":
                    dataframe[x['column_name']] = dataframe[x['column_name']].cat.codes

            else:
                if verbose:
                    print("No target type specified for column %s" % x['column_name'])
    except KeyError:
        print("No coercion required.")

    # # 2. ADHOC TRANSFORMATION
    try:
        for _, transformation in enumerate(feature_param['transformation']):

            dataframe = make_transform(dataframe, transformation)

    except KeyError:
        print("No transformation required. ")

    # 3. Aggregation

    try:
        agg_feature = feature_param['aggregation']
        groupby = agg_feature['groupby']
        agg_params = agg_feature['agg_params']
        rename_after_agg = agg_feature['rename']
        post_transformation = agg_feature['post_transformation']

        dataframe_agg = dataframe.groupby(groupby).agg(agg_params)
        dataframe_agg = dataframe_agg.rename(index=str, columns=rename_after_agg)

        for _, transformation in enumerate(post_transformation):

            dataframe = make_transform(dataframe_agg, transformation)

        dataframe = dataframe.reset_index()

    except KeyError as err:
        print("No aggregation required. ")
        print(err)

    # 4. Convert dummy data
    try:
        dummy_features = feature_param['dummy_columns']
        dataframe = pd.get_dummies(dataframe, columns=dummy_features)
    except KeyError as err:
        print("No Dummy conversion specified.")

    # 5. Remove outliers
    try:
        outliers = feature_param['outliers']
        print(outliers['columns'])
        dataframe = remove_outliers(dataframe, outliers['columns'], num_std=int(outliers['num_std']))
    except KeyError as err:
        print("No outliers removal specified.")

    return dataframe


def make_transform(dataframe, transformation, verbose=False):
    """

    :param dataframe:
    :param transformation: Dictionary.
    :param verbose: Boolean.
    :return:
    """

    transform_type = transformation['type']
    transform_column = transformation['column_name']
    action = transformation['action']
    params = transformation['parameters']
    assign = transformation['assign']

    if transform_type == 'series':

        if verbose:
            print("Transforming %s with function %s and parameters %s. Assign to %s" %
                  (transform_column, action, params, assign))

        param_action = list(params.keys())[0]
        param_value = list(params.values())[0]

        if param_action == 'func':
            # Eval the function if a function is passed in
            params = {param_action: eval(param_value)}

        series = getattr(dataframe[transform_column], action)(**params)
        dataframe = dataframe.assign(**{assign: series})

    elif transform_type == 'cross-series':
        other = dataframe[transformation['other']]
        assert type(other) == pd.Series
        print("Transforming %s with function %s and parameters %s, cross column %s. Assign to %s" %
              (transform_column, action, params, other.name, assign))
        params = {'other': other, **params}

        series = getattr(dataframe[transform_column], action)(**params)
        dataframe = dataframe.assign(**{assign: series})

    return dataframe


def load_individual_file(feature_dict):
    """

    :param feature_dict:
    :return:
    """
    features_column_names = [feature_column['column_name'] for feature_column in feature_dict['features']]

    dataframe = pd.read_csv(feature_dict['file'])
    features = dataframe.loc[:, features_column_names]

    # Transform the dataset
    transformed_features = transform(features, feature_dict, verbose=True)

    try:
        target = transformed_features[feature_dict['target']]
        # Drop target
        transformed_features = transformed_features.drop(feature_dict['target'], axis=1)
        return transformed_features, target
    except KeyError:
        print("No target to be extracted")
        return transformed_features, None


def remove_outliers(dataframe, columns, num_std=4):
    """

    :param dataframe:
    :param columns:
    :param num_std:
    :return:
    """
    index_to_remove = pd.Index([])

    for column in columns:

        print('Column: %s' % column)

        std = dataframe[column].std()
        mean = dataframe[column].mean()

        outliers = dataframe[abs(dataframe[column] - mean) > (num_std * std)]

        print(outliers.index)

        index_to_remove = index_to_remove.append(outliers.index)

    index_to_remove = index_to_remove.drop_duplicates()

    dataframe = dataframe.drop(index_to_remove, axis=0)

    return dataframe


def split_data(features, target=None, test_size=0.1, random_state=42, stratify=None, verbose=False):

    if verbose:
        print("Splitting data...")

    # Split the dataframe to features and target
    X_train, X_val, y_train, y_val = train_test_split(features,
                                                      target,
                                                      test_size=test_size,
                                                      random_state=random_state,
                                                      stratify=stratify)

    return X_train, X_val, y_train, y_val
