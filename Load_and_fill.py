import numpy as np
import pandas as pd

group_by_list = [['WELL', 'FORMATION'],
                 ['WELL', 'GROUP'],
                 ['WELL'],
                 ['FORMATION'],
                 ['GROUP'],
                 ]

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

lithology_numbers = {30000: 0,
                     65030: 1,
                     65000: 2,
                     80000: 3,
                     74000: 4,
                     70000: 5,
                     70032: 6,
                     88000: 7,
                     86000: 8,
                     99000: 9,
                     90000: 10,
                     93000: 11}

curve_columns = ['RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'NPHI', 'DTC', 'PEF', 'RXO']


def fillna_groupby(dataframe, group_by_list, curve_list):
    """
    Perform fillna() by grouping WELL, GROUP, FORMATION and using median values
    :param dataframe: Input is the raw csv file loaded to initial dataframe
    :param group_by_list: Input list (of list) of columns to groupby()
    :param curve_list: List of columns (curves) to fillna(median)
    :return: dataframe
    """
    # remove FORMATION nulls to groupby without error
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
    frame = pd.DataFrame()

    well_names = dataframe.WELL.unique()

    for i in range(len(well_names)):
        data = dataframe[columns][dataframe.WELL == well_names[i]].rolling(kernel).median()
        # print(f"{data.index[:5]}\n{df[df.WELL==wells[i]].index[:5]}")
        data = data.combine_first(dataframe[dataframe.WELL == well_names[i]])

        frame = frame.append(data)

    return frame


def load(csv, sep=';'):
    """
    Load data into dataframe
    :param csv:
    :param sep:
    :return:
    """
    data = pd.read_csv(csv, sep=sep)
    return data


def load_and_fill_median(csv, sep=";"):
    """
    :param csv:
    :param sep:
    Calls 3 functions and uses a median value as a fill
    Function 1: load()
    Function 2: fillna_groupby()
    Function 3: median_filter()
    :return: Pandas DataFrame
    """

    data = load(csv, sep=sep)

    data_fillna = fillna_groupby(data, group_by_list, curve_columns)

    data_mf = median_filter(data_fillna, curve_columns, 101)

    # data_mf['FORCE_2020_LITHOFACIES_LITHOLOGY'] = data_mf['FORCE_2020_LITHOFACIES_LITHOLOGY'].map(lithology_numbers)

    return data_mf


def load_and_fill_0(csv, sep=';'):
    """
    :param csv:
    :param sep:

    Calls 2 functions and uses a median value as a fill
    Function 1: load()
    Function 2: fillna_groupby()
    :return: Pandas DataFrame
    """

    data = load(csv, sep=sep)

    data_fillna = data.fillna(0)

    data_mf = median_filter(data_fillna, curve_columns, 101)

    # data_mf['FORCE_2020_LITHOFACIES_LITHOLOGY'] = data_mf['FORCE_2020_LITHOFACIES_LITHOLOGY'].map(lithology_numbers)

    return data_mf
