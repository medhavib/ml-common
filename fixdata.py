def fixdata(data, train_data):
    #first step is to delete any columns in data that are not present in train_data
    columns_to_delete = data.columns.difference(train_data.columns)
    data = data.drop(columns_to_delete, 1)
    
    #second step is to add columns which are in training data and set their values to zero
    columns_to_add = train_data.columns.difference(data.columns)
    df_add = pd.DataFrame(columns=columns_to_add, index =data.index)
    df_add.fillna(0, inplace=True)
    
    return pd.concat([data, df_add], axis=1)
    