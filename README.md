lin_model=LinearRegression()

lin_model.fit(X_train,Y_train)

y_train_predict = lin_model.predict(X_train)

rmse=(np.sqrt(mean_squared_error(Y_train, y_train_predict)))

print("The model performance for testing set")
print('RMSE is {}'.format(rmse))
print("\n")

#on testing set
y_test_predict  = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))

print("The model performance for testing set")
print('RMSE is {}'.format(rmse))
