[image1]: assets/decision_tree.png "image1"
[image2]: assets/linear_regression.png "image2"
[image3]: assets/hyperplane.png "image3"
[image4]: assets/SVR1.png "image4"
[image5]: assets/knn.png "image5"
[image6]: assets/dnn.png "image6"
[image7]: assets/rand_forest1.png "image7"

# Machine Learning
In this repo a short overview of important Machine Learning algorithms is provided.

## Content

- [How machine learning works](#ml_works)
    - [Step 1: Select and prepare a training data set](#step_1)
    - [Step 2: Choose an algorithm to run on the training data set ](#step_2)
    - [Supervised Learning](#sl)
        - [Linear regression](#linear_reg)
        - [Logistic regression](#log_reg)
        - [Support vector machine](#svm)
        - [Decision trees](#dec_trees)
        - [Random Forest](#random_forest)
        - [Instance-based algorithms -KNN](#instance_based)
        - [Naive Bayes](#naive_bayes)
        - [Neural networks](#neural_net)
    - [Unsupervised Learning](#usl)
        - [Clustering algorithms](#clustering)
        - [K-means](#k_means)
        - [Association algorithms](#asso_algo)
    - [Step 3: Training the algorithm to create the model](#training)
    - [Step 4: Using and improving the model ](#using)
- [Setup Instructions](#Setup_Instructions)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)

Machine learning is a method of data analysis that automates ***analytical model building***. Machine learning focuses on applications that learn from experience and improve their decision-making or predictive accuracy over time. 
It is a branch of artificial intelligence based on the idea that systems can ***learn from data***, ***identify patterns*** and ***make decisions*** with minimal human intervention.

## How machine learning works <a name="ml_works"></a> 

There are four basic steps for building a machine learning application (or model). These are typically performed by data scientists working closely with the business professionals for whom the model is being developed.

### Step 1: Select and prepare a training data set <a name="step_1"></a> 

**Training data** is a data set used to  solve the underlying problem. The training data can be **labeled** to call out classifications the model will need to identify. In addition, training data can be **unlabeled**, and the model will need to extract features and assign classifications on its own.

In either case, the training data needs to be properly prepared: **cleaned**, **randomized**, and **checked for imbalances or biases** that could impact the training. It should also be divided into two subsets: the **training subset**, which will be used to train the application, and the **validation subset**, used to test and refine it. 

### Step 2: Choose an algorithm to run on the training data set <a name="step_2"></a> 

An algorithm is a set of **statistical processing steps**. The type of algorithm depends on the **type (labeled or unlabeled) and amount of data** in the training data set and on the **type of problem** to be solved.

Common types of machine learning algorithms for use with labeled data include the following:

- **Regression algorithms**: is used to understand the relationship between dependent and independent variables. It is commonly used to make projections, such as for sales revenue for a given business. Linear regression, logistical regression, and polynomial regression are popular regression algorithms. Output values are continuous.

- **Classification** uses an algorithm to accurately assign test data into specific categories. It recognizes specific entities within the dataset and attempts to draw some conclusions on how those entities should be labeled or defined. Common classification algorithms are linear classifiers, support vector machines (SVM), decision trees, k-nearest neighbor, and random forest, which are described in more detail below.

### Supervised Learning <a name="sl"></a>
### Linear regression <a name="linear_reg"></a>
Linear regression is used to predict the value of a dependent variable based on the value of an independent variable. For example, a linear regression algorithm could be trained to predict a salesperson’s annual sales (the dependent variable) based on its relationship to the salesperson’s education or years of experience (the independent variables.)
A linear regression line has an equation of the form 

<img src="https://render.githubusercontent.com/render/math?math=Y = b0 %2B b1 \cdot X  %2B e" width="150px">

where ***X*** is the explanatory (independent) variable and ***Y*** is the dependent variable. The slope of the line is ***b1***, and ***b0*** is the intercept (Y value when X = 0). ***e*** is the error term (also known as the residual errors), the part of Y that can be explained by the regression model.

The figure below illustrates the linear regression model, where:

- the best-fit regression line is in blue
- the intercept (b0) and the slope (b1) are shown in green
- the error terms (e) are represented by vertical red lines

![image2]

From the scatter plot above, it can be seen that not all the data points fall exactly on the fitted regression line. Some of the points are above the blue curve and some are below it; overall, the residual errors (**e**) have approximately mean zero.

The sum of the squares of the residual errors are called the **Residual Sum of Squares or RSS**.

The average variation of points around the fitted regression line is called the **Residual Standard Error or RSE**. This is one of the metrics used to evaluate the overall quality of the fitted regression model. The lower the **RSE**, the better it is.

Since the mean error term is zero, the outcome variable y can be approximately estimated as follow:

<img src="https://render.githubusercontent.com/render/math?math=Y = b0 %2B b1 \cdot X" width="150px">

Mathematically, the beta coefficients (**b0** and **b1**) are determined so that the **RSS** is as minimal as possible. This method of determining the beta coefficients is technically called least squares regression or ordinary least squares (**OLS**) regression.

Once, the beta coefficients are calculated, a **t-test** is performed to check whether or not these coefficients are significantly different from zero. A non-zero beta coefficients means that there is a **significant relationship** between the predictors (**X**) and the outcome variable (**Y**).

### Logistic regression <a name="log_reg"></a>
Logistic regression can be used when the dependent variable is binary in nature: A or B, 0 or 1, yes or no, diseased or non-diseased.

There can be one or multiple independent predictor variables (X). 

Other synonyms are **binary logistic regression**, **binomial logistic regression** and **logit model**.

Logistic regression does not return directly the class of observations. It allows us to estimate the **probability (p) of class membership**. The probability will range between **0** and **1**. You need to decide the threshold probability at which the category flips from one to the other. By default, this is set to p = 0.5, but in reality it should be settled based on the analysis purpose.


**The Logistic function**: The standard logistic regression function, for predicting the outcome of an observation given a predictor variable (**X**), is an **s-shaped curve** defined as 

<img src="https://render.githubusercontent.com/render/math?math=p = \frac{1} {1 %2B exp(-Y)}" width="130px">

where:

<img src="https://render.githubusercontent.com/render/math?math=Y = b0 %2B b1 \cdot X" width="130px">


**p** is the probability of event to occur (1) given **X**. Mathematically, this is written as **p(event=1|X)** and abbreviated as 

<img src="https://render.githubusercontent.com/render/math?math=p = \frac{1} {1 %2B exp(-(b0 %2B b1 \cdot X)}" width="200px">

It can be demonstrated that the formula becomes a linear combination of predictors:


<img src="https://render.githubusercontent.com/render/math?math=log[p/(1-p)] = b0 %2B b1 \cdot X" width="250px">


When you have multiple predictor variables, the logistic function looks like: 

<img src="https://render.githubusercontent.com/render/math?math=log[p/(1-p)] = b0 %2B b1 \cdot x1 %2B b2 \cdot x2 %2B ... %2B bn \cdot xn" width="450px">


**b0** and **b1** are the regression **beta coefficients**. A positive b1 indicates that increasing x will be associated with increasing p. Conversely, a negative b1 indicates that increasing x will be associated with decreasing p.

The quantity **log[p/(1-p)]** is called the logarithm of the **odd**, also known as **log-odd** or **logit**.

**The odds reflect the likelihood that the event will occur**. It can be seen as the ratio of “successes” to “non-successes”. Technically, odds are the probability of an event divided by the probability that the event will not take place. For example, if the probability of being diabetes-positive is 0.5, the probability of “won’t be” is 1-0.5 = 0.5, and the odds are 1.0.

Note that, the probability can be calculated from the odds as 

<img src="https://render.githubusercontent.com/render/math?math=p = Odds/(1 %2B Odds)" width="220px">

### Support vector machine <a name="svm"></a>
A support vector machine is a popular supervised learning model used for both classification and regression. That said, it is typically leveraged for classification problems, constructing a hyperplane where the distance between two classes of data points is at its maximum.


There are a few important parameters of SVM:

**Kernel**: A kernel helps us find a hyperplane in the higher dimensional space without increasing the computational cost. Usually, the computational cost will increase if the dimension of the data increases. This increase in dimension is required when we are unable to find a separating hyperplane in a given dimension and are required to move in a higher dimension:

![image3]

**Hyperplane**: This is basically a separating line between two data classes in SVM. But in Support Vector Regression, this is the line that will be used to predict the continuous output.

**Decision Boundary**: A decision boundary can be thought of as a demarcation line (for simplification) on one side of which lie positive examples and on the other side lie the negative examples. On this very line, the examples may be classified as either positive or negative. This same concept of SVM will be applied both in Support Vector Regression and Classification.

**How does it look like for regression?**
The problem of regression is to find a function that approximates mapping **from an input domain to real numbers** on the basis of a training sample. 

![image4]

Consider these two red lines as the ***decision boundary*** and the green line as the hyperplane. Our objective, when we are moving on with SVR, is to basically consider the points that are within the decision boundary line. Our ***best fit line is the hyperplane*** that has a maximum number of points.

The first thing that we’ll understand is what is the decision boundary (the danger red line above!). Consider these lines as being at any distance, say ‘a’, from the hyperplane. So, these are the lines that we draw at distance ‘+a’ and ‘-a’ from the hyperplane. This ‘a’ in the text is basically referred to as epsilon.

The **equation of the hyperplane**:

<img src="https://render.githubusercontent.com/render/math?math=Y = wx %2B b " width="130px">

The **equation of decision boundary**:

<img src="https://render.githubusercontent.com/render/math?math=wx %2B b = %2B a" width="130px">

and

<img src="https://render.githubusercontent.com/render/math?math=wx %2B b = -a" width="130px">

Thus, any hyperplane that satisfies our SVR should satisfy:

<img src="https://render.githubusercontent.com/render/math?math=-a < Y- wx %2B b < %2B a" width="220px">

Hence, we are going to take only those points that are within the decision boundary and have the least error rate.


### Decision trees <a name="dec_trees"></a>
The decision tree method is a powerful and popular predictive machine learning technique that is used for both **Classification** and **Regression**. So, it is also known as Classification and Regression Trees (CART). We use Classification Trees when need to classify the targets (e.g. passenger survived or died) and Regression Trees when we need to predict continuous values like price of a house. In general, Decision Tree algorithms are referred to as **CART** or **Classification and Regression Trees**.

In case of classification: Decision Trees use classified data to make recommendations based on a set of decision rules.

A decision tree is drawn upside down with its **root at the top**. In the image at the bottom, the text in black (gender, age, sibsp) represents a **condition/internal node**, based on which the tree splits into **branches/edges**. The end of the branch that doesn’t split anymore is the **decision/leaf**, in this case, whether the passenger died or survived, represented as red and green text, respectively.

![image1]

**Recursive Binary Splitting**:
In this procedure all the features are considered and different split points are tried and tested using a cost function. The split with the best cost (or lowest cost) is selected.

Titanic dataset: There are 3 features, and hence 3 candidate splits:

- Calculate how much accuracy each split will cost
- The split that costs least is chosen (here: sex of the passenger) 
- Greedy algorithm: recursive algorithm, all input variables and all possible split points are evaluated and chosen in a greedy manner
- Root node as best predictor/classifier.

**Cost of a split**: For regression predictive modeling problems the cost function that is minimized to choose split points is the **sum squared error** or SSE across all training samples that fall within the rectangle

    Regression : sum(Y — prediction)²

where **Y** is the output for the training sample and **prediction** is the predicted output for the rectangle.    

For classification the Gini index function is used which provides an indication of how “pure” the leaf nodes are (how mixed the training data assigned to each node is).

    Classification : G = sum(pk * (1 — pk))

where **G** is the **Gini index** over all classes, **pk** are the proportion of training instances with class k in the rectangle of interest. A node that has all classes of the same type (perfect class purity) will have G=0, whereas a G that has a 50-50 split of classes for a binary classification problem (worst purity) will have a G=0.5.

**Stopping Criterion**: As a problem usually has a large set of features, it results in large number of splits, which in turn gives a huge tree. Such trees are complex and can lead to overfitting. So, we need to know when to stop. One way of doing this is to set a **minimum number of training inputs** to use on each leaf. For example, we can use a minimum of 10 passengers to reach a decision(died or survived), and ignore any leaf that takes less than 10 passengers. Another way is to set maximum depth of your model. Maximum depth refers to the the length of the longest path from a root to a leaf.

**Pruning** The performance of a tree can be further increased by pruning. It involves removing the branches that make use of features having low importance. This way, we reduce the complexity of tree, and thus increasing its predictive power by reducing overfitting.

The fastest and simplest pruning method is to work through each leaf node in the tree and evaluate the effect of removing it using a hold-out test set. Leaf nodes are removed only if it results in a drop in the overall cost function on the entire test set. You stop removing nodes when no further improvements can be made.

More sophisticated pruning methods can be used such as cost complexity pruning (also called weakest link pruning) where a learning parameter (alpha) is used to weigh whether nodes can be removed based on the size of the sub-tree.

### Random Forest <a name="random_forest"></a>
Decision Trees (classification or regression) are the building blocks of the random forest model.

Random forest, like its name implies, consists of a large number of individual Decision Trees that operate as an ensemble. Each individual tree in the random forest spits out a class prediction and the class with the most votes becomes our model’s prediction (see figure below).

![image7]

The fundamental concept behind random forest is a simple but powerful one — the wisdom of crowds. In data science speak, the reason that the random forest model works so well is:

A large number of relatively uncorrelated models (trees) operating as a committee will outperform any of the individual constituent models.

The reason for this effect is that the trees protect each other from their individual errors. While some trees may be wrong, many other trees will be right, so as a group the trees are able to move in the correct direction. So the prerequisites for random forest to perform well are:

1. We need **features that have at least some predictive power** so that models built using those features do better than random guessing.
2. The predictions (and therefore the errors) made by the individual trees need to have **low correlations** with each other.

### Instance-based algorithms <a name="instance_based"></a>
A good example of an instance-based algorithm is **K-Nearest Neighbor** or **KNN**. KNN algorithm is a simple algorithm that can be used to solve both classification and regression problems. The KNN algorithm assumes that similar things exist in close proximity. 

![image5]

Notice in the image above that most of the time, similar data points are close to each other. KNN captures the idea of similarity (sometimes called distance, proximity, or closeness).

The straight-line distance - **Euclidean distance** - is a popular and familiar choice.

**The KNN Algorithm**

1. **Load** the data
2. **Initialize K** to your chosen number of neighbors
3. **For each sample** in the data
    - **Calculate the distance** between the query example and the current example from the data.
    - **Add** the distance and the index of the example to an **ordered collection**
4. **Sort** the ordered collection of distances and indices from smallest to largest (in ascending order) by the distances
5. **Pick the first K entries** from the sorted collection
6. **Get the labels** of the selected K entries
7. If **regression**, return the **mean of the K labels**
8. If **classification**, return the **mode of the K labels**

### Naive Bayes <a name="naive_bayes"></a>
A Naive Bayes classifier is a probabilistic machine learning model that’s used for classification task. It adopts the principle of class conditional independence from the Bayes Theorem. 

Bayes Theorem:

<img src="https://render.githubusercontent.com/render/math?math=P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}" width="220px">

Using Bayes theorem, we can find the ***probability of A*** happening, given that ***B has occurred***. Here, **B** is the **evidence** and **A** is the **hypothesis**. The assumption made here is that the predictors/features are independent. That is presence of one particular feature does not affect the other. Hence, it is called naive.

Example- playing golf: It is not suitable for playing golf if the outlook is rainy, temperature is hot, humidity is high and it is windy. 

- Y: Dependent variable - playing golf
- X = (x1, x2, ..., xn): Independent variables (features) like temperature, humidity etc.

<img src="https://render.githubusercontent.com/render/math?math=P(Y|X) = \frac{P(X|Y) \cdot P(Y)}{P(X)}" width="220px">

By substituting X and expanding using the chain rule we get:

<img src="https://render.githubusercontent.com/render/math?math=P(Y|x1,x2,...xn) = \frac{P(x1|Y) \cdot P(x2|Y) \cdot ... \cdot P(xn|Y) \cdot P(Y)}{P(x1) \cdot P(x2) \cdot ... \cdot P(xn)}" width="500px">

Now, you can obtain the values for each by looking at the dataset and substitute them into the equation. For all entries in the dataset, the denominator does not change, it remain static. Therefore, the denominator can be removed and a proportionality can be introduced.

<img src="https://render.githubusercontent.com/render/math?math=P(Y|x1,x2,...xn) = P(x1|Y) \cdot P(x2|Y) \cdot ... \cdot P(xn|Y) \cdot P(Y)" width="600px">

In the actualcase of ***playing golf***, the class variable **Y** has only two outcomes, **yes** or **no**. There could be cases where the classification could be multivariate. Therefore, we need to find the class y with maximum probability.

<img src="https://render.githubusercontent.com/render/math?math=Y = argmax(P(Y|x1,x2,...xn))" width="350px">


### Neural networks <a name="neural_net"></a>
Primarily leveraged for **deep learning** algorithms, neural networks process training data by mimicking the interconnectivity of the human brain through layers of nodes. Each node is made up of inputs, weights, a bias (or threshold), and an output. If that output value exceeds a given threshold, it “fires” or activates the node, passing data to the next layer in the network. Neural networks learn this mapping function through supervised learning, adjusting based on the loss function through the process of gradient descent. When the cost function is at or near zero, we can be confident in the model’s accuracy to yield the correct answer.

![image6]

### Unsupervised Learning <a name="usl"></a>

Algorithms for use with **unlabeled data** include the following:

### Clustering algorithms <a name="clustering"></a>
Think of clusters as groups. Clustering focuses on **identifying groups of similar records** and labeling the records according to the group to which they belong. This is done **without prior knowledge** about the groups and their characteristics. In other words, we try to find **homogeneous subgroups** within the data such that data points in each cluster are as similar as possible according to a similarity measure such as **euclidean-based distance** or **correlation-based distance**. The decision of which similarity measure to use is application-specific. Types of clustering algorithms include the K-means, TwoStep, and Kohonen clustering.

### K-means <a name="k_means"></a>

The way K-means algorithm works is as follows:

1. Specify **number of clusters K**.
2. **Initialize centroids** by first shuffling the dataset and then randomly selecting K data points for the centroids without replacement.
3. Compute the **sum of the squared distance** between data points and all centroids.
5. **Assign** each data point to the **closest cluster** (centroid).
6. **Compute the centroids** for the clusters by taking the average of the all data points that belong to each cluster.
7. **Keep iterating** 3-6 until there is no change to the centroids. i.e assignment of data points to clusters isn’t changing.

The approach K-means follows to solve the problem is called Expectation-Maximization. The E-step is assigning the data points to the closest cluster. The M-step is computing the centroid of each cluster. 


### Association algorithms <a name="asso_algo"></a>
Association algorithms find patterns and relationships in data and identify frequent ‘if-then’ relationships called association rules. These are similar to the rules used in data mining.

## Step 3: Training the algorithm to create the model <a name="training"></a>

Training the algorithm is an iterative process–it involves running variables through the algorithm, comparing the output with the results it should have produced, adjusting weights and biases within the algorithm that might yield a more accurate result, and running the variables again until the algorithm returns the correct result most of the time. The resulting trained, accurate algorithm is the machine learning model—an important distinction to note, because 'algorithm' and 'model' are incorrectly used interchangeably, even by machine learning mavens.

## Step 4: Using and improving the model <a name="using"></a>

The final step is to use the model with new data and, in the best case, for it to improve in accuracy and effectiveness over time. Where the new data comes from will depend on the problem being solved. For example, a machine learning model designed to identify spam will ingest email messages, whereas a machine learning model that drives a robot vacuum cleaner will ingest data resulting from real-world interaction with moved furniture or new objects in the room.


## Setup Instructions <a name="Setup_Instructions"></a>
The following is a brief set of instructions on setting up a cloned repository.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.



### Prerequisites: Installation of Python via Anaconda and Command Line Interaface <a name="Prerequisites"></a>
- Install [Anaconda](https://www.anaconda.com/distribution/). Install Python 3.7 - 64 Bit

- Upgrade Anaconda via
```
$ conda upgrade conda
$ conda upgrade --all
```

- Optional: In case of trouble add Anaconda to your system path. Write in your CLI
```
$ export PATH="/path/to/anaconda/bin:$PATH"
```

### Clone the project <a name="Clone_the_project"></a>
- Open your Command Line Interface
- Change Directory to your project older, e.g. `cd my_github_projects`
- Clone the Github Project inside this folder with Git Bash (Terminal) via:
```
$ git clone https://github.com/ddhartma/Spark-Big-Data-Analytics.git
```

- Change Directory
```
$ cd Spark-Big-Data-Analytics
```

- Create a new Python environment, e.g. spark_env. Inside Git Bash (Terminal) write:
```
$ conda create --name spark_env
```

- Activate the installed environment via
```
$ conda activate spark_env
```

- Install the following packages (via pip or conda)
```
numpy = 1.17.4
pandas = 0.24.2
pyspark
```

- Check the environment installation via
```
$ conda env list
```

## Acknowledgments <a name="Acknowledgments"></a>
* This project is part of the Udacity Nanodegree program 'Data Science'. Please check this [link](https://www.udacity.com) for more information.

## Further Links <a name="Further_Links"></a>
Machine Learning
* [Machine Learning - IBM](https://www.ibm.com/cloud/learn/machine-learning)
* [Supervised Learning - IBM](https://www.ibm.com/cloud/learn/supervised-learning)
* [Decision Trees in Machine Learning](https://towardsdatascience.com/decision-trees-in-machine-learning-641b9c4e8052)

* [Classification And Regression Trees for Machine Learning](https://machinelearningmastery.com/classification-and-regression-trees-for-machine-learning/)

* [Naive Bayes](https://towardsdatascience.com/naive-bayes-classifier-81d512f50a7c)

* [Understanding Random Forest](https://towardsdatascience.com/understanding-random-forest-58381e0602d2)

Git/Github
* [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html)
* [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)
* [5 types of Git workflows](https://buddy.works/blog/5-types-of-git-workflows)

Docstrings, DRY, PEP8
* [Python Docstrings](https://www.geeksforgeeks.org/python-docstrings/)
* [DRY](https://www.youtube.com/watch?v=IGH4-ZhfVDk)
* [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)