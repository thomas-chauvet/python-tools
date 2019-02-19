# Source: https://www.kaggle.com/nickycan/compress-70-of-dataset

"""
In pandas default is to put numerical value in float or int 64.
The goal of this script is to reduce the sice of the pandas DataFrame
by changing the type of the float / int 64 in float / int (8), 16 or 32.
In order to do this, we check the min and max value in the column
and if it stays in the bound of a lower float type, we change the type.

We do the same for int type.
"""

import numpy as np
import pandas as pd

UINT8_MAX = np.iinfo(np.uint8).max
UINT16_MAX = np.iinfo(np.uint16).max
UINT32_MAX = np.iinfo(np.uint32).max

INT8_MIN    = np.iinfo(np.int8).min
INT8_MAX    = np.iinfo(np.int8).max
INT16_MIN   = np.iinfo(np.int16).min
INT16_MAX   = np.iinfo(np.int16).max
INT32_MIN   = np.iinfo(np.int32).min
INT32_MAX   = np.iinfo(np.int32).max

FLOAT16_MIN = np.finfo(np.float16).min
FLOAT16_MAX = np.finfo(np.float16).max
FLOAT32_MIN = np.finfo(np.float32).min
FLOAT32_MAX = np.finfo(np.float32).max


def memory_usage(data):
    memory_usage = data.memory_usage()
    return memory_usage.sum() / (1024*1024)


def compress_rate(memory_before_compress, memory_after_compress):
    return (memory_before_compress - memory_after_compress) / memory_before_compress


def compress_pandas_dataframe(data, verbose=True):
    """
        Compress datatype as small as it can
        Parameters
        ----------
        path: pandas Dataframe

        Returns
        -------
            Compressed pandas Dataframe
    """
    memory_original_dataframe = memory_usage(data)
    memory_before_compress = memory_original_dataframe

    length_interval      = 50
    length_float_decimal = 4

    for col in data.columns:
        col_dtype = data[col].dtype
        if col_dtype != 'object':
            col_min = data[col].min()
            col_max = data[col].max()
            if col_dtype == 'float64':
                if (col_min > FLOAT16_MIN) and (col_max < FLOAT16_MAX):
                    data[col] = data[col].astype(np.float16)
                    new_col_dtype = "float16"
                elif (col_min > FLOAT32_MIN) and (col_max < FLOAT32_MAX):
                    data[col] = data[col].astype(np.float32)
                    new_col_dtype = "float32"
                else:
                    new_col_dtype = None
                    pass
            if col_dtype == 'int64':
                if col_min >= 0:
                    if  col_max < UINT8_MAX:
                        data[col] = data[col].astype(np.uint8)
                        new_col_dtype = "uint8"
                    elif col_max < UINT16_MAX:
                        data[col] = data[col].astype(np.uint16)
                        new_col_dtype = "uint16"
                    elif col_max < UINT32_MAX:
                        data[col] = data[col].astype(np.uint16)
                        new_col_dtype = "uint32"
                    else:
                        new_col_dtype = None
                        pass
                else:
                    if (col_min > INT8_MIN) and (col_max < INT8_MAX):
                        data[col] = data[col].astype(np.int8)
                        new_col_dtype = "int8"
                    elif (col_min > INT16_MIN) and (col_max < INT16_MAX):
                        data[col] = data[col].astype(np.int16)
                        new_col_dtype = "int16"
                    elif (col_min > INT32_MIN) and (col_max < INT32_MAX):
                        data[col] = data[col].astype(np.int32)
                        new_col_dtype = "int32"
                    else:
                        new_col_dtype = None
                        pass

            if verbose:
                if new_col_dtype:
                    memory_after_compress = memory_usage(data)
                    print("Column '{0}' converted from type '{1}' to '{2}'.".format(
                        col, col_dtype, new_col_dtype))
                    print("Compression rate is {0:.2%} with a memory usage of {1:.2f}MB".format(
                        compress_rate(memory_before_compress, memory_after_compress), 
                        memory_after_compress))
                    memory_before_compress = memory_after_compress
                    print('='*length_interval)
            
    if verbose:
        print()
        memory_after_compress = memory_usage(data)
        print("Memory before compress was {0:.2f}MB and now it is {1:.2f}MB.".format(
            memory_original_dataframe, memory_after_compress))
        print("Total compress rate: {0:.2%}".format(compress_rate(memory_original_dataframe, memory_after_compress)))

    return data