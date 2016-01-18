"""
Loading the boston dataset and examining its target (label) distribution.
"""

# Load libraries
import numpy as np
import pylab as pl
from sklearn import datasets, grid_search
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import median_absolute_error, make_scorer
import tabulate as tb

def load_data():
    '''Load the Boston dataset.'''

    boston = datasets.load_boston()
    return boston


def explore_city_data(city_data):
    '''Calculate the Boston housing statistics.'''

    # Get the labels and features from the housing data
    housing_prices = city_data.target
    housing_features = city_data.data
    housing_feature_headers = city_data.feature_names

    # Size of data
    data_size = housing_features.shape[0]

    # Number of features
    number_of_features = housing_features.shape[1]
    
    # Compile feature headers including target into an array
    feature_headers = np.empty(number_of_features+1, dtype='S7')
    feature_headers[:-1] = housing_feature_headers
    feature_headers[-1] = "TARGET"

    # Compile attribute headers (stuff to be explored) in an array
    attribute_headers = np.array(["0:MIN", "1:MAX", "2:MEAN", "3:MEDIAN", "4:STD. DEV."])
    
    # Compile all the attributes-per-feature in an array
    feature_wise_attributes_data = np.zeros([5, number_of_features+1])
    
    # Calculate min of each feature
    feature_wise_attributes_data[0,:-1] = np.min(housing_features, axis=0)
    feature_wise_attributes_data[0, -1] = np.min(housing_prices)

    # Calculate max of each feature
    feature_wise_attributes_data[1,:-1] = np.max(housing_features, axis=0)
    feature_wise_attributes_data[1, -1] = np.max(housing_prices)

    # Calculate mean of each feature
    feature_wise_attributes_data[2,:-1] = np.mean(housing_features, axis=0)
    feature_wise_attributes_data[2, -1] = np.mean(housing_prices)

    # Calculate median of each feature
    feature_wise_attributes_data[3,:-1] = np.median(housing_features, axis=0)
    feature_wise_attributes_data[3, -1] = np.median(housing_prices)

    # Calculate standard deviation of each feature
    feature_wise_attributes_data[4,:-1] = np.std(housing_features, axis=0)
    feature_wise_attributes_data[4, -1] = np.std(housing_prices)

    explored_data = datasets.base.Bunch(
                                        data_size=data_size, 
                                        number_of_features=number_of_features,
                                        feature_headers=feature_headers,
                                        attribute_headers=attribute_headers,
                                        feature_wise_attributes_data=feature_wise_attributes_data )
    return explored_data

def tabulate_explored_data(explored_data):
    data = np.concatenate((explored_data.attribute_headers.reshape(1,-1),
                           explored_data.feature_wise_attributes_data.transpose()))
    data = data.transpose()
    print "Data statistics:"
    print(tb.tabulate(data, headers=explored_data.feature_headers.tolist(),tablefmt="grid"))


def performance_metric(label, prediction):
    '''Calculate and return the appropriate performance metric.'''

    # http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
    
    # median_absolute_error vs mean_absolute_error vs mean_squared_error
    # median_absolute_error appears fairly robust against 
    # outliers of [target - prediction] as compared to mean_absolute_error
    # mean_squared_error behaves similar to mean_absolute_error but it seems
    # amplify the error, which could hurt the scoring in grid search
    error_loss = median_absolute_error(label, prediction)
    return error_loss


def split_data(city_data):
    '''Randomly shuffle the sample set. Divide it into training and testing set.'''

    # Get the features and labels from the Boston housing data
    X, y = city_data.data, city_data.target
    size_of_data = city_data.data.shape[0]
    
    # Create an array containing random indices
    # np.random should have some kind of time based seed by default
    index_array = np.arange(size_of_data)
    np.random.shuffle(index_array)

    # Specify ratio of train:test division
    ratio_of_train_to_test = 0.8

    # Split the data
    X_train = city_data.data[index_array[:size_of_data*ratio_of_train_to_test]]
    y_train = city_data.target[index_array[:size_of_data*ratio_of_train_to_test]]
    X_test = city_data.data[index_array[size_of_data*ratio_of_train_to_test:]]
    y_test = city_data.target[index_array[size_of_data*ratio_of_train_to_test:]]

    return X_train, y_train, X_test, y_test


def learning_curve(depth, X_train, y_train, X_test, y_test):
    '''Calculate the performance of the model after a set of training data.'''

    # We will vary the training set size so that we have 50 different sizes
    sizes = np.linspace(1, len(X_train), 50)
    train_err = np.zeros(len(sizes))
    test_err = np.zeros(len(sizes))

    print "Decision Tree with Max Depth: "
    print depth

    for i, s in enumerate(sizes):

        # Create and fit the decision tree regressor model
        regressor = DecisionTreeRegressor(max_depth=depth)
        regressor.fit(X_train[:s], y_train[:s])

        # Find the performance on the training and testing set
        train_err[i] = performance_metric(y_train[:s], regressor.predict(X_train[:s]))
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))


    # Plot learning curve graph
    learning_curve_graph(sizes, train_err, test_err)


def learning_curve_graph(sizes, train_err, test_err):
    '''Plot training and test error as a function of the training size.'''

    pl.figure()
    pl.title('Decision Trees: Performance vs Training Size')
    pl.plot(sizes, test_err, lw=2, label = 'test error')
    pl.plot(sizes, train_err, lw=2, label = 'training error')
    pl.legend()
    pl.xlabel('Training Size')
    pl.ylabel('Error')
    pl.show()


def model_complexity(X_train, y_train, X_test, y_test):
    '''Calculate the performance of the model as model complexity increases.'''

    print "Model Complexity: "

    # We will vary the depth of decision trees from 2 to 25
    max_depth = np.arange(1, 25)
    train_err = np.zeros(len(max_depth))
    test_err = np.zeros(len(max_depth))

    for i, d in enumerate(max_depth):
        # Setup a Decision Tree Regressor so that it learns a tree with depth d
        regressor = DecisionTreeRegressor(max_depth=d)

        # Fit the learner to the training data
        regressor.fit(X_train, y_train)

        # Find the performance on the training set
        train_err[i] = performance_metric(y_train, regressor.predict(X_train))

        # Find the performance on the testing set
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))

    # Plot the model complexity graph
    model_complexity_graph(max_depth, train_err, test_err)


def model_complexity_graph(max_depth, train_err, test_err):
    '''Plot training and test error as a function of the depth of the decision tree learn.'''

    pl.figure()
    pl.title('Decision Trees: Performance vs Max Depth')
    pl.plot(max_depth, test_err, lw=2, label = 'test error')
    pl.plot(max_depth, train_err, lw=2, label = 'training error')
    pl.legend()
    pl.xlabel('Max Depth')
    pl.ylabel('Error')
    pl.show()


def fit_predict_model(city_data):
    '''Find and tune the optimal model. Make a prediction on housing data.'''

    # Get the features and labels from the Boston housing data
    X, y = city_data.data, city_data.target

    # Setup a Decision Tree Regressor
    regressor = DecisionTreeRegressor()

    parameters = {'max_depth':(1,2,3,4,5,6,7,8,9,10)}

    performance_metric_scorer = make_scorer(performance_metric, greater_is_better=False)

    reg = grid_search.GridSearchCV(regressor, parameters, scoring=performance_metric_scorer)

    # Fit the learner to the training data
    print "Final Model: "
    print reg.fit(X, y)
    print "Optimum Max Depth with GridSearch =", reg.best_params_['max_depth']
    
    # Use the model to predict the output of a particular sample
    x = [11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]
    x_arr = np.array(x).reshape(1, -1)
    y = reg.predict(x_arr)
    print "House: " + str(x)
    print "Prediction: " + str(y)
    

def main():
    '''Analyze the Boston housing data. Evaluate and validate the
    performanance of a Decision Tree regressor on the Boston data.
    Fine tune the model to make prediction on unseen data.'''

    # Load data
    city_data = load_data()

    # Explore the data
    data_statistics = explore_city_data(city_data)
    tabulate_explored_data(data_statistics)

    # Training/Test dataset split
    X_train, y_train, X_test, y_test = split_data(city_data)

    # Close all previous session's plots
    pl.close('all')

    # Learning Curve Graphs
    max_depths = [1,2,3,4,5,6,7,8,9,10]
    for max_depth in max_depths:
        learning_curve(max_depth, X_train, y_train, X_test, y_test)

    # Model Complexity Graph
    model_complexity(X_train, y_train, X_test, y_test)

    # Tune and predict Model
    fit_predict_model(city_data)


main()