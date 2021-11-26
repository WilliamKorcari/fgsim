import numpy as np


def idx_cluster(records_array):
    # creates an array of indices, sorted by unique element
    idx_sort = np.argsort(records_array)

    # sorts records array so all unique elements are together 
    sorted_records_array = records_array[idx_sort]

    # returns the unique values, the index of the first occurrence of a value, and the count for each element
    vals, idx_start, count = np.unique(sorted_records_array, return_counts=True, return_index=True)

    # splits the indices into separate arrays
    res = np.split(idx_sort, idx_start[1:])
    return res


def new_coord(arr, idx):
    new_arr = [arr[idx[i]] for i in range(len(idx))]

    for i in range(len(new_arr)):
        new_arr[i] = new_arr[i][0]
        new_arr[i] = np.array(new_arr[i])
    return np.array(new_arr)

def summed_e(arr, idx):
    new_arr = [arr[idx[i]] for i in range(len(idx))]

    for i in range(len(new_arr)):
        new_arr[i] = new_arr[i].sum()
    return new_arr

def bool_mask_matched_indeces(arr, size):
    boolArr = []
    for i in range(size):
        boolArr.append([])
        boolArr[i] = arr[i]>0
    return boolArr

def side_indeces_bool_mask(arr, size):
    posArr = []
    negArr = []
    for i in range(size):
        posArr.append([])
        negArr.append([])
        posArr[i] = arr[i]>0
        negArr[i] = arr[i]<0
    return posArr, negArr

def filter_array(arr, boolArr, size):
    newArr = []
    for i in range(size):
        newArr.append([])
        newArr[i] = arr[i][boolArr[i]]
    return newArr

