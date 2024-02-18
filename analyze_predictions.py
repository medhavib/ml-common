def pltpredictions(real, predictions):
    plt.plot(real, predictions, '.')
    plt.title("Analyzing our predictions")
    plt.xlabel("Real Prices")
    plt.ylabel("Predicted Prices")
    plt.show()

#sample code snip to plot predictions against real values
pred_train, pred_test, tar_train, tar_test  =   train_test_split(predictors, targets, test_size=.1)
y_pred = clf.predict(pred_test)
pltpredictions(tar_test, y_pred)

#put the predicted values and real values in a spreadsheet and add an extra column which compares 
#the difference between real and predicted. Then sort by this difference. Review the records in this file to identify hidden *bugs*
#in your model.

train_predictions_df = pd.DataFrame({'Prediction': y_pred, 'Real': tar_test, 'Diff':y_pred - tar_test,
                                    'Percentage': 100*abs(y_pred - tar_test)/tar_test}, index = pred_test.index)
train_predictions_df.sort_values('Percentage', inplace=True, ascending=False)
train_predictions_df.to_csv('../data/HousePrices_train_predictions.csv')