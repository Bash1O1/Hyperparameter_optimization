Introduction
Hyperparameter optimization is a crucial step in the development of machine learning models. Hyperparameters are the configuration settings used to tune how a model
learns from the data. Unlike model parameters, which are learned from the training data, hyperparameters are set before the training process begins. Effective 
hyperparameter tuning can significantly improve the performance of a model. This report explores Bayesian Optimization technique for hyperparameter optimization.

Hyperparameters
Hyperparameters are parameters whose values control the learning process and determine the values of model parameters that a learning algorithm ends up learning. The
prefix ‘hyper_’ suggests that they are ‘top-level’ parameters that control the learning process and the model parameters that result from it.

RandomForestClassifier
The Random Forest or Random Decision Forest is a supervised Machine learning algorithm used for classification, regression, and other tasks using decision trees.
Random Forests are particularly well-suited for handling large and complex datasets, dealing with high-dimensional feature spaces, and providing insights into 
feature importance. This algorithm’s ability to maintain high predictive accuracy while minimizing overfitting makes it a popular choice across various domains,
including finance, healthcare, and image analysis, among others.

Which Hyperparameters are optimized
There are many hyperparameters of RandomForestClassifier but for simplicity we have used only four hyperparameters for tuning which are:

•n_estimators: The n_estimator parameter controls the number of decision trees inside the classifier. The appropriate no. of trees isvery important for high
accuracy of the model. Larger no. of trees can increase time complexity of the model.
• max_depth: It governs the maximum height up to which the trees inside the forest can grow. It is one of the most important hyperparameters when it comes to 
increasing the accuracy of the model. If we increase the depth of the tree the model accuracy increases up to a certain limit but then it will start to decrease
gradually because of overfitting in the model. It is important to set its value appropriately to avoid overfitting.
• max_features: Random Forest takes random subsets of features and tries to find the best split. max_features help to find the number of features to take into
account in order to make the best split.
• criterion: It measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.

Bayesian Optimization
Bayesian Optimization is an iterative algorithm that attempts to minimize (or maximize) an objective function by building a surrogate model to approximate the 
function and using this model to select the next set of hyperparameters to evaluate.
The steps involved in Bayesian Optimization are:
• Initialization: Start with a small set of random hyperparameter values and evaluate the objective function at these points.
• Surrogate Model: Fit a probabilistic model (usually a Gaussian Process) to the evaluated points.
• Acquisition Function: Use the surrogate model to select the next set of hyperparameters to evaluate by optimizing an acquisition function.
• Evaluation: Evaluate the objective function at the selected hyperparameters.
• Update: Update the surrogate model with the new observation.
• Iteration: Repeat steps until a stopping criterion is met (e.g., a maximum number of iterations).

Implementation
• Step 1: Define the Objective function
• Step 2: Define the Hyperparameter spaceGiven 
• Step 3: Run The Optimization Algorithm
• Step 4: Evaluate the results

Results
It can be observed that after hyperparameter tuning the model accuracy 
is increased by a significant amount. Both the cross validation score as 
well as ROC-AUC is increased.

Conclusion
Bayesian Optimization is a powerful method for hyperparameter optimization that efficiently explores the hyperparameter space by balancing exploration and 
exploitation. This approach often finds better hyperparameters in fewer iterations compared to traditional methods like grid search and random search. In this
report, we demonstrated how to implement Bayesian Optimization using the scikit-optimize library to optimize the hyperparameters of a RandomForestClassifier. By
defining an appropriate objective function and hyperparameter space, we efficiently found a set of hyperparameters that maximized the model's accuracy. After that
we compared our model accuracy with the one optimized using hyperopt using Learning distribution curve. Bayesian Optimization can be extended to other machine
learning models and performance metrics, making it a versatile tool for hyperparameter tuning in various applications.


