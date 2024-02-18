def oneHotEncode(df, addColumnsToEncode = []):
    columnsToEncode = list(df.select_dtypes(include=['category','object']))
#    columnsToEncode = columnsToEncode + addColumnsToEncode

    for elem in addColumnsToEncode: 
        if elem not in columnsToEncode:
            columnsToEncode.append(elem)
    
    for feature in columnsToEncode:
        #print ('Encoding ' + feature)
        try:
            df = pd.concat([df, 
                              pd.get_dummies(df[feature]).rename(columns=lambda x: feature + '_' + str(x))], axis=1)
            df = df.drop(feature, axis=1)
        except:
            print('Error encoding '+feature)
            raise
    return df