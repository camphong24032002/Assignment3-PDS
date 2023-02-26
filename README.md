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

# Features Engineering
With statistical analysis, each input variable's relationship to the target variable is assessed, and the input variables with the strongest relationships are chosen using statistical-based feature selection techniques. These techniques can be quick and efficient, but the choice of statistical measures depends on the data type of the input and output variables.


# Modeling
