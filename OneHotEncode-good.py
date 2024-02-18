def oneHotEncode2(df, le_dict = {}):
    if not le_dict:
        columnsToEncode = list(df.select_dtypes(include=['category','object']))
        train = True;
    else:
        columnsToEncode = le_dict.keys()   
        train = False;
        
    for feature in columnsToEncode:
        if train:
            le_dict[feature] = LabelEncoder()
        try:
            if train:
                df[feature] = le_dict[feature].fit_transform(df[feature])
            else:
                df[feature] = le_dict[feature].transform(df[feature])
                
            df = pd.concat([df, 
                              pd.get_dummies(df[feature]).rename(columns=lambda x: feature + '_' + str(x))], axis=1)
            df = df.drop(feature, axis=1)
        except:
            print('Error encoding '+feature)
            #df[feature]  = df[feature].convert_objects(convert_numeric='force')
            df[feature]  = df[feature].apply(pd.to_numeric, errors='coerce')
    return (df, le_dict)