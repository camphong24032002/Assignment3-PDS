# Practical Compound Data Analyst
## Abstract and Sumary
- The Practical Data Science guidance strategy influences local revenue via shopping online intention record campaigns. In order to improve and address several credit-related problems throughout defined features from Google Analyst.
- By researching and learning a supervised model with the given data, the issues of classification would determine whether customers get out revenue or not. In addition, some problems of clustering data would figure out the groups of customer segments that this page should focus on.
- The review illustrates that the model training, using multi-types of model architectures, results in an approximate accuracy score, simply because the data of this campaign still is not a significant size that could lead to a perfect prediction for the future. Which means, the comparison of these models just be considered via its training time and efficiency. Along with that, the clustering model would define the customer’s groups less particularly due to the size of data, however, these models still could be applied as a solution for commercial problems.
- Finally, this research recommends changing the model strategy to raise efficiency or influence prediction in the development and implementation of modeling architectures.

## Introduction
The main steps of the data science process are covered in this assignment. In Ipython (Jupyter Notebook), we will create and put into practice the necessary steps to finish the associated tasks. The goal of this assignment is to provide you with a realistic understanding of how the data science approach typically solves problems.
With considerable decision, we prefer to hand on the problem of Modeling data For particular. We will focus on 2 principal problems for analysts which include Classification and Clustering problems with the chosen dataset: Online Shoppers Purchasing Intention Data Set.


## Frist let install the data
```
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv', sep=',')
```
# Preprocessing
In this type of dataset, we move along to the problem of defining or classifying customers as well as clustering customer segments as our goal, with the label class being Revenue.
Along our defined target, we design the preprocess pipeline using several below methods of preprocessing data:
- **Data cleaning:** in this process, missing, inconsistent, or irrelevant data are found and removed. This can involve eliminating redundant data, adding values when they are missing, and dealing with outliers.
- **Data transform:** entails changing the data into a format that is better suited for data mining. This may involve encoding categorical data, producing dummy variables, and normalizing numerical data.
- **Data discretization:** is the process of converting continuous numerical data into categorical data, which can then be used in categorical data mining techniques like decision trees.

![Untitled](https://user-images.githubusercontent.com/81562297/221422399-d34bf34d-4d58-46ac-a73a-cb82f6c61587.png)

# Features Engineering
With statistical analysis, each input variable's relationship to the target variable is assessed, and the input variables with the strongest relationships are chosen using statistical-based feature selection techniques. These techniques can be quick and efficient, but the choice of statistical measures depends on the data type of the input and output variables.

![Untitled](https://user-images.githubusercontent.com/81562297/221422579-414466f0-2a76-4568-b44b-cab80ce675c4.png)

# Pipline Architecture
Data from the input has been minimized before being divided into two main groups, numerical data and categorical data. The Missing Value is subject to several variables depending on the type of data, such as the ‘mean’ value for numerical data and the "most frequent" value for categorical data. Additionally, before the entire data set was standard scaled and the major features were chosen, categorical data would be OneHotEncoded.

![Untitled](https://user-images.githubusercontent.com/81562297/221423207-0b892e18-ddea-483b-a79d-7d8ae44329ef.png)

# Modeling
## MLP Classifier
- Choose the right features: The very first model would train using the Feature Engineering technique described above. Then, using the SelectKbest() method in Grid Search, the new feature would be chosen.
- Select appropriate model: In this type of problem, we would focus on the Neural Network Model first, which means we will apply MLPClassifier() from Sklearn.
- Train and evaluate the model appropriately:
Due to the defined parameters of Neural Network, in which include:
```
full_pipeline = Pipeline(steps=[
    ('preprocess', preprocess_pipeline),
    ('selectkbest', SelectKBest(chi2, k=7)),
    ('mlpclassifier', MLPClassifier(hidden_layer_sizes=(50), activation='relu', solver='adam', random_state=0, learning_rate_init=0.05, alpha=100, max_iter=1000))
])
full_pipeline
```

## Decision Tree
- Choose appropriate features: The only first model would train utilizing the Feature Engineering method described above. The new feature would then be identified in Grid Search using the SelectKbest() method.
- Select appropriate model: In the second type , we would focus on the Decision Tree, which means we will apply DecisionTreeClassifier() from Sklearn.
- Train and evaluate the model appropriately:
From the defined parameters of decision model, which is:
```
tree_pipeline = Pipeline(steps=[
    ('preprocess', preprocess_pipeline),
    ('selectkbest', SelectKBest(chi2, k=10)),
    ('decisiontree', DecisionTreeClassifier(random_state=0, max_depth=10))
])
tree_pipeline
```

## K-Means
- The K-means algorithm clusters data by attempting to divide it into n groups of equal variance.
- The K-means algorithm seeks centroids with the lowest inertia, or within-cluster sum-of-squares criterion:

```
for n_clusters in clusters_size:
    best_score, best_init = 0 , 0
    for n_init in init_size:
        kmeans = KMeans(n_clusters = n_clusters, n_init=n_init)
        kmeans.fit(matrix)
        silhouette_avg = silhouette_score(matrix_test, kmeans.predict(matrix_test))*100
        train_avg =  silhouette_score(matrix, kmeans.predict(matrix))*100
        test_scores.append(silhouette_avg)
        train_scores.append(train_avg)
        if (silhouette_avg>best_score):
            best_score, best_init = silhouette_avg, n_init
    print(f'For n_clusters = {n_clusters}, and n_init = {best_init}, the average silhouette_score is: {round(silhouette_avg, 5)}')
```


## Hierarchical clustering
- Hierarchical clustering is a broad class of clustering algorithms that create nested clusters by successively merging or splitting them. A tree is used to represent the cluster hierarchy.
- The AgglomerativeClustering object uses a bottom-up approach to perform hierarchical clustering: each observation begins in its own cluster, and clusters are successively merged together.

```
matrix = train_X_df[['Informational_Duration', 'PageValues']].copy()
for n_clusters in range(2,8):
    model = AgglomerativeClustering(n_clusters = n_clusters)
    clusters = model.fit_predict(matrix)
    silhouette_avg = silhouette_score(matrix, clusters)*100
    print(f'For n_clusters = {n_clusters}, the average silhouette_score is : {silhouette_avg.round(3)}')
```


## Innovative Model
- Following some investigation, we discovered that the Grid Search CV is a practical way to enhance a model for greater accuracy and better parameters.
- GridSearchCV is a method for adjusting hyperparameters to find the best values for a particular model. The value of a model's hyperparameters has a substantial impact on its performance.
- Mentioning that there is no way to determine the best values for hyperparameters in advance, it is ideal to explore every conceivable value before deciding what the best ones are. We utilize GridSearchCV to automate the tweaking of hyperparameters because doing it manually could take a lot of time and resources.

![Untitled](https://user-images.githubusercontent.com/81562297/221423377-d6aa2ff5-28f9-4d6e-be69-76c80d4625ca.png)

# Referenece
- Statistic Correlation: https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/ 
- Transform Target: https://scikit-learn.org/stable/modules/compose.html#transforming-target-in-regression
- Scaling Method: https://ndquy.github.io/posts/cac-phuong-phap-scaling/ 
- Wiki_Chi2: https://en.wikipedia.org/wiki/Chi-squared_test 
- Wiki_PersonalChi2: https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test 
- Overfitting Confronting: https://towardsdatascience.com/3-techniques-to-avoid-overfitting-of-decision-trees-1e7d3d985a09
- Unsuppervised Learning Research: https://towardsdatascience.com/unsupervised-learning-and-data-clustering-eeecb78b422a 
- Clustering Coefficient: https://towardsdatascience.com/silhouette-coefficient-validating-clustering-techniques-e976bb81d10c 
- KMean Clustering: https://towardsdatascience.com/kmeans-clustering-for-classification-74b992405d0a 
