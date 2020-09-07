import pandas as pd
import numpy as np


def preprocessing_data(number,data):
    print(number)
    new_data_dict = dict()
    x = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    for (columnName, columnData) in data.iteritems():
        new_data_dict[columnName] = list(columnData.values)
    for key, value in new_data_dict.items():
        if key == 'title':
            continue
        empty_list = list(value)
        empty_list.sort()
        for i in  range(len(empty_list)):
            for j in range(len(value)):
                if value[j] == empty_list[i]:
                    value[j] = float(i/number)
                else:
                    continue
    for key, value in new_data_dict.items():
        for i in range(len(value)):
            if key == 'title':
                continue
            counter = 0
            for j in range(len(x) - 1) :
                if value[i] > x[j]:
                    if value[i] <= x[j+1]:
                        value[i] = (x[j+1]*10)
    new_df = creating_data(new_data_dict,number)
    return new_df
def creating_data(new_data_dict,number):
    ####################################### creating columns ##########################
    new_df = pd.DataFrame()
    list1 = list([1,2,3,4,5,6,7,8,9,10])
    for key in new_data_dict.keys() :
        if key == 'title':
            temp = pd.DataFrame(columns=[key])
            new_df = pd.concat([temp, new_df], axis=1)
            continue
        else:
            for i in list1:
                temp = pd.DataFrame(columns=[key + str(i)])
                new_df = pd.concat([temp, new_df], axis=1)
    ####################################### creating empty rows ##########################
    for i in range(number):
        new_df = new_df.append(pd.Series(0, index=new_df.columns), ignore_index=True)
    for key,value in new_data_dict.items():
        if key == 'title':
            for i in range(len(value)):
                new_df[key][i] = value[i]
            continue
        else:
            for j in list1:
                for i in range(len(value)):
                    if int(value[i]) != j:
                        continue
                    else:
                        new_df[key+str(j)][i] = 1
    return new_df
def creating_user_dataframe(columns):
    df = pd.DataFrame()
    for column in columns:
        temp = pd.DataFrame(columns=[column])
        df = pd.concat([temp, df], axis=1)
    return df
def appending_series(df,songs_dict,index):
    df = df.append(pd.Series(0, index=df.columns), ignore_index=True)
    for key,value in songs_dict.items():
        df[key][index] = value
    return df