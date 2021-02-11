import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np


def fillna_groupby(dataframe, group_by_list, curve_list):
    """
    Remove FORMATION nulls to groupby without error
    """
    dataframe = dataframe[dataframe['FORMATION'].notnull()]
    print(f"Filling missing values from curves:\n{curve_list}")
    print("-" * 40)
    for group in group_by_list:
        print(f"Group by: {group}")
        df_gby = dataframe.groupby(group)
        dataframe[curve_list] = df_gby[curve_list].transform(impute_median)
        print(f"Total number of missing values: {sum(dataframe[curve_list].isnull().sum())}")
        print("=" * 40)
    return dataframe


def impute_median(series):
    return series.fillna(series.median())


def median_filter(dataframe, columns, kernel=51):
    """
    Run median filter over series with default kernel of 51
    """
    frame = pd.DataFrame()
    well_names = dataframe.WELL.unique()
    for i in range(len(well_names)):
        data = dataframe[columns][dataframe.WELL == well_names[i]].rolling(kernel).median()
        # print(f"{data.index[:5]}\n{df[df.WELL==wells[i]].index[:5]}")
        data = data.combine_first(dataframe[dataframe.WELL == well_names[i]])
        frame = frame.append(data)
    return frame


def proportional_sample(df, proportion=0.5):
    """
    Create random sampled dataframe maintaining label proportionality
    :param df: input dataframe
    :param proportion: proportion of dataframe
    :return: frame: a dataframe
    """
    counts = df['FORCE_2020_LITHOFACIES_LITHOLOGY'].value_counts()
    total = df.shape[0]
    frame = pd.DataFrame()
    names_percent = {}
    N = df['FORCE_2020_LITHOFACIES_LITHOLOGY'].shape[0]
    for item in counts.iteritems():
        names_percent[item[0]] = [lithology_keys[item[0]], float(item[1]) / N]
    for k, v in names_percent.items():
        x = shuffle(df)
        sample = (int(total * v[1] * proportion))
        frame = frame.append(x[x.FORCE_2020_LITHOFACIES_LITHOLOGY == k].sample(sample))
    return frame


def formations(df):
    all_formations = df['FORMATION'].unique()
    all_formations_dict = dict(zip(all_formations, range(len(all_formations))))
    df = df.replace(all_formations_dict)
    return df


group_by_list = [['WELL', 'FORMATION'],
                 ['WELL', 'GROUP'],
                 ['WELL'],
                 ['FORMATION'],
                 ['GROUP'],
                 ]

curves = ['RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'NPHI', 'PEF', 'DTC']


lithology_keys = {30000: 'Sandstone',
                  65030: 'Sandstone/Shale',
                  65000: 'Shale',
                  80000: 'Marl',
                  74000: 'Dolomite',
                  70000: 'Limestone',
                  70032: 'Chalk',
                  88000: 'Halite',
                  86000: 'Anhydrite',
                  99000: 'Tuff',
                  90000: 'Coal',
                  93000: 'Basement'}

lithology_numbers = {v: k for k, v in enumerate(lithology_keys.keys())}


def train_test_val_split(dataframe):
    train, test = train_test_split(dataframe, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.2, random_state=42)
    print(len(train), 'train examples')
    print(len(val), 'validation examples')
    print(len(test), 'test examples')

    train_labels = train['FORCE_2020_LITHOFACIES_LITHOLOGY']
    train = train[curves]
    test_labels = test['FORCE_2020_LITHOFACIES_LITHOLOGY']
    test = test[curves]
    val_labels = val['FORCE_2020_LITHOFACIES_LITHOLOGY']
    val = val[curves]

    train = tf.keras.utils.normalize(train, axis=1)
    test = tf.keras.utils.normalize(test, axis=1)
    val = tf.keras.utils.normalize(val, axis=1)

    train = np.array(train)
    test = np.array(test)
    val = np.array(val)

    train_labels = train_labels.values
    test_labels = test_labels.values
    val_labels = val_labels.values

    print(f"Training data shape: {train.shape}")
    print(f"Training labels shape: {train_labels.shape}")
    print(f"Testing data shape: {test.shape}")
    print(f"Testing labels shape: {test_labels.shape}")
    print(f"Validation data shape: {val.shape}")
    print(f"Validation labels shape: {val_labels.shape}")

    return train, train_labels, test, test_labels, val, val_labels


def main(path, kernel=51, proportion=0.1):
    data = pd.read_csv(path, sep=';')
    data = data.drop(['X_LOC', 'Y_LOC', 'Z_LOC', 'CALI', 'SGR', 'SP', 'BS',
                      'ROP', 'DTS', 'DCAL', 'DRHO', 'MUDWEIGHT', 'RMIC', 'ROPA', 'RXO'], axis=1)
    FORMATION_mask = data['FORMATION'].notnull().values
    data = data[FORMATION_mask]
    # -------------- FILL, FILTER, ETC ------------- #
    data_fillna = fillna_groupby(data, group_by_list, curves)
    data_fillna_medfilt = median_filter(data_fillna, curves, kernel=kernel)
    # Create random sample dataframe maintaining proportional labels
    data_fillna_medfilt_p10 = proportional_sample(data_fillna_medfilt, proportion)
    df = data_fillna_medfilt_p10.copy()
    df['FORCE_2020_LITHOFACIES_LITHOLOGY'] = df['FORCE_2020_LITHOFACIES_LITHOLOGY'].map(lithology_numbers)

    return df
