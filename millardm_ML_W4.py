# -------------------------------------------- #

# CS7CS4 Machine Learning
# Week 4 Assignment
#
# Name:         Michael Millard
# Student ID:   24364218
# Due date:     19/10/2024
# Dataset 1 ID: # id:20-40-20-0
# Dataset 2 ID: # id:11-22-11-0

# -------------------------------------------- #
# Imports
# -------------------------------------------- #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures     
from sklearn.neighbors import KNeighborsClassifier      
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve

# -------------------------------------------- #
# Read in data and set labels
# -------------------------------------------- #

# Set which database (true = 1, false = 2)
dataset1 = False

# NB: dataset csv file must be in same directory as this solution
labels = ["X1", "X2", "y"]
if (dataset1):
    df = pd.read_csv("millardm_W4_dataset_1.csv", names=labels)
else:
    df = pd.read_csv("millardm_W4_dataset_2.csv", names=labels)
print("Dataframe head:")
print(df.head())

# Split data frame up into X (input features) and y (target values) 
X1 = df["X1"].to_numpy()
X2 = df["X2"].to_numpy()
X = np.column_stack((X1, X2))
y = df["y"].to_numpy()
print("Number of elements: ", len(y))
print("Number of +1 labels: ", len(y[y == 1]))
print("Number of -1 labels: ", len(y[y == -1]))
print("Labels ratio (+1/-1): ", len(y[y == 1]) / len(y[y == -1]))

# Split dataset up into training dataset and test dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X1_train, X1_test, X2_train, X2_test = X_train[:, 0], X_test[:, 0], X_train[:, 1], X_test[:, 1]

# -------------------------------------------- #
# Question (i)(a)
# -------------------------------------------- #
# Visualize dataset
# ----------------------- #

# Configure plotting params
plt.rc('font', size=16) 
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['legend.framealpha'] = 0.9
y_plus_color, y_minus_color, y_pred_plus_color, y_pred_minus_color = 'lime', 'darkturquoise', 'red', 'blue'

# Create scatter plotof entire dataset ('+' for +1 labels, 'o' for -1 labels)
plt.figure(figsize=(8, 6))
plt.scatter(X1[y == 1], X2[y == 1], marker='+', color=y_plus_color, label='y = +1')
plt.scatter(X1[y == -1], X2[y == -1], marker='o', color=y_minus_color, label='y = -1')

# Label axes and add legend
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc=4) # BR corner

# Save and show the plot
if (dataset1):
    plt.savefig("dataset_scatter_i_a.png")
else:
    plt.savefig("dataset_2_scatter_i_a.png")
plt.show()

# ----------------------- #
# Finding optimal values of q and C for dataset individually (q first, then C)
# ----------------------- #
# q cross validation
# ----------------------- #

# Arrays for mean and std dev of f1 and ROC AUC scores for each q value
mean_f1_score = [] 
std_f1_score = []
mean_auc_score = [] 
std_auc_score = []

# Range of q values to sweep through
q_range = [1, 2, 3, 4, 5, 6, 7, 8]

# Sweep through q values (same logistic regression model each time)
log_model = LogisticRegression(penalty='l2', max_iter=1000)
for q in q_range:
    # Create polynomial features dataset
    X_poly = PolynomialFeatures(q).fit_transform(X)
    # Give cross_val_score entire dataset
    f1_cv_scores = cross_val_score(log_model, X_poly, y, cv=5, scoring='f1')
    auc_cv_scores = cross_val_score(log_model, X_poly, y, cv=5, scoring='roc_auc')
    mean_f1_score.append(np.array(f1_cv_scores).mean())
    std_f1_score.append(np.array(f1_cv_scores).std())
    mean_auc_score.append(np.array(auc_cv_scores).mean())
    std_auc_score.append(np.array(auc_cv_scores).std())

# f1 Score error bar
plt.errorbar(q_range, mean_f1_score, yerr=std_f1_score, linewidth=3)
plt.xlabel('q')
plt.ylabel('f1 Score')
if (dataset1):
    plt.savefig("ind_q_f1_errorbar_i_a.png")
else:
    plt.savefig("ind_q_f1_errorbar_i_a_2.png")
plt.show()

# ROC AUC error bar
plt.errorbar(q_range, mean_auc_score, yerr=std_auc_score, linewidth=3)
plt.xlabel('q')
plt.ylabel('ROC AUC')
if (dataset1):
    plt.savefig("ind_q_auc_errorbar_i_a.png")
else:
    plt.savefig("ind_q_auc_errorbar_i_a_2.png")
plt.show()

# Take the best q value as the one with the highest worst case score score (f1 - std dev) and create best X_poly
worst_cases = np.array(mean_f1_score) - np.array(std_f1_score)
q_best = q_range[np.argmax(worst_cases)]
X_poly = PolynomialFeatures(q_best).fit_transform(X)
X_train_poly = PolynomialFeatures(q_best).fit_transform(X_train)
X_test_poly = PolynomialFeatures(q_best).fit_transform(X_test)

# Print out results for each q value
print("\nq cross-validaion:")
for i in range(len(q_range)):
    print("q, mean f1, std dev f1, mean auc, std dev auc, worst case = %i & %.4f & %.4f & %.4f & %.4f & %.4f"%(q_range[i], mean_f1_score[i], std_f1_score[i], mean_auc_score[i], std_auc_score[i], worst_cases[i]))  
print("Best q value is: ", q_best)

# ----------------------- #
# C cross validation
# ----------------------- #

# Empty mean and std dev arrays to reuse for C value sweep
mean_f1_score = [] 
std_f1_score = []
mean_auc_score = [] 
std_auc_score = []

# Range of C values to sweep through
C_range = [0.1, 1, 2, 4, 6, 8, 10]

# Sweep through C values
for C_ in C_range:
    # Create new logistic regression model with L2 penalty and new C value
    log_model = LogisticRegression(penalty='l2', C=C_, max_iter=2500)
    # Give cross_val_score entire dataset
    f1_cv_scores = cross_val_score(log_model, X_poly, y, cv=5, scoring='f1')
    auc_cv_scores = cross_val_score(log_model, X_poly, y, cv=5, scoring='roc_auc')
    mean_f1_score.append(np.array(f1_cv_scores).mean())
    std_f1_score.append(np.array(f1_cv_scores).std())
    mean_auc_score.append(np.array(auc_cv_scores).mean())
    std_auc_score.append(np.array(auc_cv_scores).std())

# f1 Score errorbar
plt.errorbar(C_range, mean_f1_score, yerr=std_f1_score, linewidth=3)
plt.xlabel('C')
plt.ylabel('f1 Score')
if (dataset1):
    plt.savefig("ind_C_f1_errorbar_i_a.png")
else:
    plt.savefig("ind_C_f1_errorbar_i_a_2.png")
plt.show()

# ROC AUC errorbar
plt.errorbar(C_range, mean_auc_score, yerr=std_auc_score, linewidth=3)
plt.xlabel('C')
plt.ylabel('ROC AUC')
if (dataset1):
    plt.savefig("ind_C_auc_errorbar_i_a.png")
else:
    plt.savefig("ind_C_auc_errorbar_i_a_2.png")
plt.show()

# Take the best C value as the one with the highest worst case score score (f1 - std dev) and create best logistic regression model
worst_cases = np.array(mean_auc_score) - np.array(std_auc_score)
C_best = C_range[np.argmax(worst_cases)]
best_log_model = LogisticRegression(penalty='l2', C=C_best, max_iter=2500).fit(X_train_poly, y_train)

# Print out results for each C value
print("\nC cross-validaion:")
for i in range(len(C_range)):
    print("C, mean f1, std dev f1, mean auc, std dev auc, worst case = %.1f & %.4f & %.4f & %.4f & %.4f & %.4f"%(C_range[i], mean_f1_score[i], std_f1_score[i], mean_auc_score[i], std_auc_score[i], worst_cases[i]))  
print("Best C value is: ", C_best)   

# ----------------------- #
# Combining q and C cross validations into one nested loop search
# ----------------------- #

best_C_vals = []
best_f1_means = []
best_f1_std_devs = []
best_auc_means = []
best_auc_std_devs = []
best_worst_cases = []

# Sweep through q values
for q in q_range:
    # Construct new polynomial features dataset based on q value
    X_poly = PolynomialFeatures(q).fit_transform(X)
    
    # Mean and std dev arrays
    mean_f1_score = []
    std_f1_score = []
    mean_auc_score = [] 
    std_auc_score = []

    # Sweep through C values
    for C_ in C_range:
        # Create new logistic regression model with L2 penalty and new C value
        log_model = LogisticRegression(penalty='l2', C=C_, max_iter=2500)
        # Give cross_val_score entire dataset (poly)
        f1_cv_scores = cross_val_score(log_model, X_poly, y, cv=5, scoring='f1')
        auc_cv_scores = cross_val_score(log_model, X_poly, y, cv=5, scoring='roc_auc')
        mean_f1_score.append(np.array(f1_cv_scores).mean())
        std_f1_score.append(np.array(f1_cv_scores).std())
        mean_auc_score.append(np.array(auc_cv_scores).mean())
        std_auc_score.append(np.array(auc_cv_scores).std())
    
    ## f1 Score errorbar
    #plt.errorbar(C_range, mean_f1_score, yerr=std_f1_score, linewidth=3)
    #plt.title("q = {}".format(q))
    #plt.xlabel('C')
    #plt.ylabel('f1 Score')
    #if (dataset1):
    #    plt.savefig("q={i}_C_f1_errorbar_i_a.png".format(i=q))
    #else:
    #    plt.savefig("q={i}_C_f1_errorbar_i_a_2.png".format(i=q))
    #plt.show()
    #
    ## ROC AUC errorbar
    #plt.errorbar(C_range, mean_auc_score, yerr=std_auc_score, linewidth=3)
    #plt.title("q = {}".format(q))
    #plt.xlabel('C')
    #plt.ylabel('ROC AUC')
    #if (dataset1):
    #    plt.savefig("q={i}_C_auc_errorbar_i_a.png".format(i=q))
    #else:
    #    plt.savefig("q={i}_C_auc_errorbar_i_a_2.png".format(i=q))
    #plt.show()
    
    # Take the best C value as the one with the highest worst case score score (f1 - std dev)
    worst_cases = np.array(mean_auc_score) - np.array(std_auc_score)
    best_C_vals.append(C_range[np.argmax(worst_cases)])
    best_f1_means.append(mean_f1_score[np.argmax(worst_cases)])
    best_f1_std_devs.append(std_f1_score[np.argmax(worst_cases)])
    best_auc_means.append(mean_auc_score[np.argmax(worst_cases)])
    best_auc_std_devs.append(std_auc_score[np.argmax(worst_cases)])
    best_worst_cases.append(np.max(worst_cases))
    
# Print combined results
print("\nq and C combined cross-validaion:")
q_best = q_range[np.argmax(best_worst_cases)]
C_best = best_C_vals[np.argmax(best_worst_cases)]
for i in range(len(q_range)):
    print("q, best C, best f1 mean, best f1 std dev, best auc mean, best auc std dev, worst case = %i & %.1f & %.4f & %.4f & %.4f & %.4f & %.4f"%(q_range[i], best_C_vals[i], best_f1_means[i], best_f1_std_devs[i], best_auc_means[i], best_auc_std_devs[i], best_worst_cases[i]))  
print("Best combination was: q, C = %i, %.1f"%(q_best, C_best))

# Set best polynomial features dataset (best q value) and train best model (best C value)
X_poly = PolynomialFeatures(q_best).fit_transform(X)
X_train_poly = PolynomialFeatures(q_best).fit_transform(X_train)
X_test_poly = PolynomialFeatures(q_best).fit_transform(X_test)
best_log_model = LogisticRegression(penalty='l2', C=C_best, max_iter=2500).fit(X_train_poly, y_train)

# ----------------------- #
# Plotting best model based on optimal q and C values
# ----------------------- #

plt.figure(figsize=(8, 6))

# Scatter of test data
plt.scatter(X1_test[y_test == 1], X2_test[y_test == 1], marker='+', color=y_plus_color, label='y_test = +1')
plt.scatter(X1_test[y_test == -1], X2_test[y_test == -1], marker='o', color=y_minus_color, label='y_test = -1')

# Scatter of best logistic regression model predictions on training data
y_pred = best_log_model.predict(X_test_poly) 
plt.scatter(X1_test[y_pred == 1], X2_test[y_pred == 1], marker='+', color=y_pred_plus_color, label='y_pred = +1')
plt.scatter(X1_test[y_pred == -1], X2_test[y_pred == -1], marker='o', color=y_pred_minus_color, label='y_pred = -1')

# Plot decision boundary
num_pts = 500
X1_grid = np.linspace(np.min(X1), np.max(X1), num_pts).reshape(-1, 1) 
X2_grid = np.linspace(np.min(X2), np.max(X2), num_pts).reshape(-1, 1)

# Make grid of input features
X_grid = []
for i in X1_grid:
    for j in X2_grid:
        X_grid.append([i, j])
X_grid = np.array(X_grid).reshape(-1, 2)
X1_grid, X2_grid = X_grid[:, 0], X_grid[:, 1]

# Create polynomial features for X_grid
X_grid_poly = PolynomialFeatures(q_best).fit_transform(X_grid)
y_grid = best_log_model.predict(X_grid_poly)
# Decision boundary line
plt.contour(X1_grid.reshape(num_pts, num_pts), X2_grid.reshape(num_pts, num_pts), y_grid.reshape(num_pts, num_pts), colors='green')
# Areas
plt.contourf(X1_grid.reshape(num_pts, num_pts), X2_grid.reshape(num_pts, num_pts), y_grid.reshape(num_pts, num_pts), alpha=0.3, cmap=plt.cm.coolwarm)

# Label axes and add legend
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc=4) # BR corner

# Save and show the plot
if (dataset1):
    plt.savefig("best_log_reg_model_i_a.png")
else:
    plt.savefig("best_log_reg_model_i_a_2.png")
plt.show()  

# Print classification report
print(classification_report(y_test, y_pred))

# -------------------------------------------- #
# Question (i)(b)
# -------------------------------------------- #
# Uniform weights
# ----------------------- #

# Range of k values to sweep through
k_range = [1, 3, 5, 7, 9, 15, 25, 35, 45, 55]

# Mean and std dev arrays
mean_f1_score = []
std_f1_score = []
mean_auc_score = [] 
std_auc_score = []

# Sweep through k values
for k in k_range:
    # Uniform weights case - train kNN model for each k on original dataset
    knn_model = KNeighborsClassifier(n_neighbors=k, weights='uniform')
    # Give cross_val_score entire dataset
    f1_cv_scores = cross_val_score(knn_model, X, y, cv=5, scoring='f1')
    auc_cv_scores = cross_val_score(knn_model, X, y, cv=5, scoring='roc_auc')
    mean_f1_score.append(np.array(f1_cv_scores).mean())
    std_f1_score.append(np.array(f1_cv_scores).std())
    mean_auc_score.append(np.array(auc_cv_scores).mean())
    std_auc_score.append(np.array(auc_cv_scores).std())

# f1 Score error bar
plt.errorbar(k_range, mean_f1_score, yerr=std_f1_score, linewidth=3)
plt.xlabel('k')
plt.ylabel('f1 Score')
plt.title('kNN Model with Uniform Weights')
if (dataset1):
    plt.savefig("k_f1_errorbar_uniform_i_b.png")
else:
    plt.savefig("k_f1_errorbar_uniform_i_b_2.png")
plt.show()

# ROC AUC errorbar
plt.errorbar(k_range, mean_auc_score, yerr=std_auc_score, linewidth=3)
plt.xlabel('k')
plt.ylabel('ROC AUC')
plt.title('kNN Model with Uniform Weights')
if (dataset1):
    plt.savefig("k_auc_errorbar_uniform_i_b.png")
else:
    plt.savefig("k_auc_errorbar_uniform_i_b_2.png")
plt.show()

# Take the best k value as the one with the highest worst case score score (f1 - std dev) and create best kNN model
worst_cases = np.array(mean_auc_score) - np.array(std_auc_score)
best_uni_worst_case = np.max(worst_cases) # Keep best result to compare to distance kNN
k_best = k_range[np.argmax(worst_cases)]
best_knn_model = KNeighborsClassifier(n_neighbors=k_best, weights='uniform').fit(X_train, y_train)

# Print out results for k cross-validation
print("\nk cross-validaion (uniform weights):")
for i in range(len(k_range)):
    print("k, mean f1, std dev f1, mean auc, std dev auc, worst case = %i & %.4f & %.4f & %.4f & %.4f & %.4f"%(k_range[i], mean_f1_score[i], std_f1_score[i], mean_auc_score[i], std_auc_score[i], worst_cases[i]))  
print("Best k value is: ", k_best)   

# ----------------------- #
# Plotting best model based on optimal k value
# ----------------------- #

plt.figure(figsize=(8, 6))

# Scatter of test data
plt.scatter(X1_test[y_test == 1], X2_test[y_test == 1], marker='+', color=y_plus_color, label='y_test = +1')
plt.scatter(X1_test[y_test == -1], X2_test[y_test == -1], marker='o', color=y_minus_color, label='y_test = -1')

# Scatter of best kNN model predictions
y_pred = best_knn_model.predict(X_test)
plt.scatter(X1_test[y_pred == 1], X2_test[y_pred == 1], marker='+', color=y_pred_plus_color, label='y_pred = +1')
plt.scatter(X1_test[y_pred == -1], X2_test[y_pred == -1], marker='o', color=y_pred_minus_color, label='y_pred = -1')

# Plot decision boundary
y_grid = best_knn_model.predict(X_grid)
# Decision boundary line
plt.contour(X1_grid.reshape(num_pts, num_pts), X2_grid.reshape(num_pts, num_pts), y_grid.reshape(num_pts, num_pts), colors='green')
# Areas
plt.contourf(X1_grid.reshape(num_pts, num_pts), X2_grid.reshape(num_pts, num_pts), y_grid.reshape(num_pts, num_pts), alpha=0.3, cmap=plt.cm.coolwarm)

# Label axes and add legend
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc=4) # BR corner

# Save and show the plot
if (dataset1):
    plt.savefig("best_knn_model_uniform_i_b.png")
else:
    plt.savefig("best_knn_model_uniform_i_b_2.png")
plt.show() 

# Print classification report
print(classification_report(y_test, y_pred))

# ----------------------- #
# Distance weights
# ----------------------- #

# Mean and std dev arrays
mean_f1_score = []
std_f1_score = []
mean_auc_score = [] 
std_auc_score = []

# Sweep through k values
for k in k_range:
    # Distance weights case - train kNN model for each k on original dataset
    knn_model = KNeighborsClassifier(n_neighbors=k, weights='distance')
    # Give cross_val_score entire dataset
    f1_cv_scores = cross_val_score(knn_model, X, y, cv=5, scoring='f1')
    auc_cv_scores = cross_val_score(knn_model, X, y, cv=5, scoring='roc_auc')
    mean_f1_score.append(np.array(f1_cv_scores).mean())
    std_f1_score.append(np.array(f1_cv_scores).std())
    mean_auc_score.append(np.array(auc_cv_scores).mean())
    std_auc_score.append(np.array(auc_cv_scores).std())

# f1 Score error bar
plt.errorbar(k_range, mean_f1_score, yerr=std_f1_score, linewidth=3)
plt.xlabel('k')
plt.ylabel('f1 Score')
plt.title('kNN Model with Distance Weights')
if (dataset1):
    plt.savefig("k_f1_errorbar_distance_i_b.png")
else:
    plt.savefig("k_f1_errorbar_distance_i_b_2.png")
plt.show()

# ROC AUC errorbar
plt.errorbar(k_range, mean_auc_score, yerr=std_auc_score, linewidth=3)
plt.xlabel('k')
plt.ylabel('ROC AUC')
plt.title('kNN Model with Distance Weights')
if (dataset1):
    plt.savefig("k_auc_errorbar_distance_i_b.png")
else:
    plt.savefig("k_auc_errorbar_distance_i_b_2.png")
plt.show()

# Take the best k value as the one with the highest worst case score score (f1 - std dev) and create best kNN model
worst_cases = np.array(mean_auc_score) - np.array(std_auc_score)
best_dist_worst_case = np.max(worst_cases) # Keep best result to compare to distance kNN
k_dist_best = k_range[np.argmax(worst_cases)]
best_knn_dist_model = KNeighborsClassifier(n_neighbors=k_dist_best, weights='distance').fit(X_train, y_train)

# Print out results for k cross-validation
print("\nk cross-validaion (distance weights):")
for i in range(len(k_range)):
    print("k, mean f1, std dev f1, mean auc, std dev auc, worst case = %i & %.4f & %.4f & %.4f & %.4f & %.4f"%(k_range[i], mean_f1_score[i], std_f1_score[i], mean_auc_score[i], std_auc_score[i], worst_cases[i]))  
print("Best k value is: ", k_best)   

# ----------------------- #
# Plotting best model based on optimal k value
# ----------------------- #

plt.figure(figsize=(8, 6))

# Scatter of test data
plt.scatter(X1_test[y_test == 1], X2_test[y_test == 1], marker='+', color=y_plus_color, label='y_test = +1')
plt.scatter(X1_test[y_test == -1], X2_test[y_test == -1], marker='o', color=y_minus_color, label='y_test = -1')

# Scatter of best kNN model predictions
y_pred = best_knn_dist_model.predict(X_test)
plt.scatter(X1_test[y_pred == 1], X2_test[y_pred == 1], marker='+', color=y_pred_plus_color, label='y_pred = +1')
plt.scatter(X1_test[y_pred == -1], X2_test[y_pred == -1], marker='o', color=y_pred_minus_color, label='y_pred = -1')

# Plot decision boundary
y_grid = best_knn_dist_model.predict(X_grid)
# Decision boundary line
plt.contour(X1_grid.reshape(num_pts, num_pts), X2_grid.reshape(num_pts, num_pts), y_grid.reshape(num_pts, num_pts), colors='green')
# Areas
plt.contourf(X1_grid.reshape(num_pts, num_pts), X2_grid.reshape(num_pts, num_pts), y_grid.reshape(num_pts, num_pts), alpha=0.3, cmap=plt.cm.coolwarm)

# Label axes and add legend
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc=4) # BR corner

# Save and show the plot
if (dataset1):
    plt.savefig("best_knn_model_distance_i_b.png")
else:
    plt.savefig("best_knn_model_distance_i_b_2.png")
plt.show() 

# Print classification report
print(classification_report(y_test, y_pred))

# -------------------------------------------- #
# Question (i)(c)
# -------------------------------------------- #

# Confusion matrix and classification report for logistic regression model
print("\nConfusion matrix and classification report for logistic regression model")
y_pred = best_log_model.predict(X_test_poly) 
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion matrix and classification report for kNN model 
print("\nConfusion matrix and classification report for kNN model")
# Take whichever kNN performed best
if (best_uni_worst_case > best_dist_worst_case): # Uniform weights better
    y_pred = best_knn_model.predict(X_test)  
else: # Distance weights better
    y_pred = best_knn_dist_model.predict(X_test)  
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion matrix and classification report for a most frequent dummy classifier
print("\nConfusion matrix and classification report for most frequent dummy classifier")
dummy_most_model = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
y_pred = dummy_most_model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion matrix and classification report for a random dummy classifier
print("\nConfusion matrix and classification report for random dummy classifier")
dummy_rand_model = DummyClassifier(strategy="uniform").fit(X_train, y_train)
y_pred = dummy_rand_model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -------------------------------------------- #
# Question (i)(d)
# -------------------------------------------- #

# ROC for best logistic regression model
fpr, tpr, _ = roc_curve(y_test, best_log_model.decision_function(X_test_poly))
plt.plot(fpr, tpr, label="Logistic Regression")
#plt.plot([0, 1], [0, 1], color='green', linestyle='--')
#plt.xlabel('False positive rate')
#plt.ylabel('True positive rate')
#plt.title("ROC for Logistic Regression Model")
#if (dataset1):
#   plt.savefig("roc_log_i_d.png")
#else:
#   plt.savefig("roc_log_i_d_2.png")
#plt.show()

# ROC for best kNN model - take whichever kNN performed best
if (best_uni_worst_case > best_dist_worst_case): # Uniform weights better
    fpr, tpr, _ = roc_curve(y_test, best_knn_model.predict_proba(X_test)[:, 1])
else: # Distance weights better
    fpr, tpr, _ = roc_curve(y_test, best_knn_dist_model.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label="kNN")
#plt.xlabel('False positive rate')
#plt.ylabel('True positive rate')
#plt.plot([0, 1], [0, 1], color='green', linestyle='--')
#plt.title("ROC for kNN Model")
#if (dataset1):
#   plt.savefig("roc_knn_i_d.png")
#else:
#   plt.savefig("roc_knn_i_d_2.png")
#plt.show()

# ROC for most frequent dummy classifier
fpr, tpr, _ = roc_curve(y_test, dummy_most_model.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label="Most Frequent Classifier")
#plt.xlabel('False positive rate')
#plt.ylabel('True positive rate')
#plt.plot([0, 1], [0, 1], color='green', linestyle='--')
#plt.title("ROC for Most Frequent Dummy Classifier")
#if (dataset1):
#   plt.savefig("roc_dummy_most_i_d.png")
#else:
#   plt.savefig("roc_dummy_most_i_d_2.png")
#plt.show()

# ROC for most frequent dummy classifier
fpr, tpr, _ = roc_curve(y_test, dummy_rand_model.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label="Random Classifier")
plt.plot([0, 1], [0, 1], color='green', linestyle='--')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
#plt.title("ROC for Random Dummy Classifier")
#if (dataset1):
#   plt.savefig("roc_dummy_rand_i_d.png")
#else:
#   plt.savefig("roc_dummy_rand_i_d_2.png")
plt.legend(loc=4)
if (dataset1):
    plt.savefig("rocs_first_dataset.png")
else:
    plt.savefig("rocs_second_dataset.png")
plt.show()

# -------------------------------------------- #
# Question (i)(e)
# -------------------------------------------- #

# Written answer 

# -------------------------------------------- #
# END OF ASSIGNMENT
# -------------------------------------------- #