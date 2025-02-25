[image1]: assets/decision_tree.png 
[image2]: assets/linear_regression.png 
[image3]: assets/hyperplane.png 
[image4]: assets/SVR1.png 
[image5]: assets/knn.png 
[image6]: assets/dnn.png
[image7]: assets/rand_forest1.png 
[image8]: assets/8.png 
[image9]: assets/9.png 
[image10]: assets/10.png 
[image11]: assets/11.png 
[image12]: assets/12.png 
[image13]: assets/13.png 
[image14]: assets/14.png 
[image15]: assets/15.png 
[image16]: assets/16.png 
[image17]: assets/17.png 
[image18]: assets/18.png 
[image19]: assets/19.png 
[image20]: assets/20.png 
[image21]: assets/21.png 
[image22]: assets/22.png 
[image23]: assets/23.png 
[image24]: assets/24.png 
[image25]: assets/25.png 
[image26]: assets/26.png 
[image27]: assets/27.png 
[image28]: assets/28.png 
[image29]: assets/29.png 
[image30]: assets/30.png 
[image31]: assets/31.png 
[image32]: assets/32.png 
[image33]: assets/33.png 
[image34]: assets/34.png 
[image35]: assets/35.png 
[image36]: assets/36.png 
[image37]: assets/37.png 
[image38]: assets/38.png 
[image39]: assets/39.png 
[image40]: assets/40.png 
[image41]: assets/41.png 
[image42]: assets/42.png 
[image43]: assets/43.png 
[image44]: assets/44.png 
[image45]: assets/45.png 
[image46]: assets/46.png 
[image47]: assets/47.png 
[image48]: assets/48.png 
[image49]: assets/49.png 


# Machine Learning Concepts
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
        - [Ensemle Learning](#ensemble)
        - [Computational Learning Theory](#comp_learn_theo)
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

# How machine learning works <a id="ml_works"></a> 

There are four basic steps for building a machine learning application (or model). These are typically performed by data scientists working closely with the business professionals for whom the model is being developed.

## Step 1: Select and prepare a training data set <a id="step_1"></a> 

**Training data** is a data set used to  solve the underlying problem. The training data can be **labeled** to call out classifications the model will need to identify. In addition, training data can be **unlabeled**, and the model will need to extract features and assign classifications on its own.

In either case, the training data needs to be properly prepared: **cleaned**, **randomized**, and **checked for imbalances or biases** that could impact the training. It should also be divided into two subsets: the **training subset**, which will be used to train the application, and the **validation subset**, used to test and refine it. 

## Step 2: Choose an algorithm to run on the training data set <a id="step_2"></a> 

An algorithm is a set of **statistical processing steps**. The type of algorithm depends on the **type (labeled or unlabeled) and amount of data** in the training data set and on the **type of problem** to be solved.

Common types of machine learning algorithms for use with labeled data include the following:

- **Regression**: is used to understand the relationship between dependent and independent variables. It is commonly used to make projections, such as for sales revenue for a given business. Linear regression, logistical regression, and polynomial regression are popular regression algorithms. Output values are continuous.

- **Classification** uses an algorithm to accurately assign test data into specific categories. It recognizes specific entities within the dataset and attempts to draw some conclusions on how those entities should be labeled or defined. Common classification algorithms are linear classifiers, support vector machines (SVM), decision trees, k-nearest neighbor, and random forest, which are described in more detail below.

## Supervised Learning <a id="sl"></a>
## Linear regression <a id="linear_reg"></a>
- Linear regression is used to predict the value of a **dependent variable** based on the value of an **independent variable**. 
- For example, a linear regression algorithm could be trained to predict a salesperson’s annual sales (the dependent variable) based on its relationship to the salesperson’s education or years of experience (the independent variables).
- A linear regression line has an equation of the form 

    <img src="https://render.githubusercontent.com/render/math?math=Y = b0 %2B b1 \cdot X  %2B e" width="170px">

    where ***X*** is the explanatory (independent) variable and ***Y*** is the dependent variable. The slope of the line is ***b1***, and ***b0*** is the intercept (Y value when X = 0). ***e*** is the error term (also known as the residual errors), the part of Y that can be explained by the regression model. Overall, the residual errors (**e**) have approximately mean zero.

The figure below illustrates the linear regression model, where:

- the best-fit regression line is in blue
- the intercept (b0) and the slope (b1) are shown in green
- the error terms (e) are represented by vertical red lines

![image2]

- The sum of the squares of the residual errors are called the **RSS (Residual Sum of Squares)**.

    <img src="https://render.githubusercontent.com/render/math?math=RSS=\sum_{i=1}^n (y_i - \hat{y}_i)^2" width="210px">
    
    - **RSS** = Residual Sum of Squares
    - **y<sub>i</sub>** = label (true value)
    - **ŷ<sub>i</sub>** = predicted value

- The average variation of points around the fitted regression line is called the **RSE (Residual Standard Error)**. This is one of the metrics used to evaluate the overall quality of the fitted regression model. The lower the **RSE**, the better it is.

    <img src="https://render.githubusercontent.com/render/math?math=RSE = \sqrt{\frac{1}{n-2}RSS}" width="190px">


Since the mean error term is zero, the outcome variable y can be approximately estimated like this:

<img src="https://render.githubusercontent.com/render/math?math=Y = b0 %2B b1 \cdot X" width="150px">

Mathematically, the beta coefficients (**b0** and **b1**) are determined so that the **RSS** is as minimal as possible. This method of determining the beta coefficients is technically called least squares regression or ordinary least squares (**OLS**) regression.

Once, the beta coefficients are calculated, a **t-test** is performed to check whether or not these coefficients are significantly different from zero. A non-zero beta coefficients means that there is a **significant relationship** between the predictors (**X**) and the outcome variable (**Y**).

### Code Example - Single independent variable
- Open Jupyter Notebook ```Linear Regression (Python).ipynb```
    ### Load Data
    ```python 
    import pandas as pd
    df = pd.read_csv("./wohnungspreise.csv")
    df.head()
    ```
    ![image45]

    ### Visualize Data
    ```python
    %matplotlib inline
    import matplotlib.pyplot as plt

    plt.scatter(df["Quadratmeter"], df["Verkaufspreis"])
    plt.show()
    ```
    ![image46]

    ### Scikit Learn: Linear Regression
    ```python
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(df[["Quadratmeter"]], df[["Verkaufspreis"]])

    print("Intercept: " + str(model.intercept_))
    print("Coef: " + str(model.coef_))

    RESULTS:
    ------------
    Intercept: [ 3143.28481869]
    Coef: [[ 5071.35242619]]
    ```

    ### Visualize Prediction
    ```python
    min_x = min(df["Quadratmeter"])
    max_x = max(df["Quadratmeter"])
    predicted = model.predict([[min_x], [max_x]])

    plt.scatter(df["Quadratmeter"], df["Verkaufspreis"])
    plt.plot([min_x, max_x], predicted, color = "red")
    plt.show()
    ```
    ![image47]

### Code Example - Multiple independent variables
 - Open Jupyter Notebook ```Linear Regression multi-var.ipynb``` 
    ### Load Data
    ```python 
    import pandas as pd
    df = pd.read_csv("./hotels.csv")
    df.head()
    ```
    ![image48]

    ### Train Test Split
    ```python
    X = df[["Gewinn", "Quadratmeter"]].values
    Y = df[["Preis in Mio"]].values

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 0, test_size = 0.25)
    ```
    ### Scikit Learn: Linear Regression
    ```python
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(X_train, y_train)

    RESULTS:
    ------------
    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

    print(model.intercept_)
    [ 6.48370247]

    print(model.coef_)
    [[  6.39855984e-06   3.89642288e-03]]
    ```
    ### Test
    ```python
    y_test_pred = model.predict(X_test) 

    for i in range(0, len(y_test_pred)):
        print(str(y_test_pred[i][0]) + " - " + str(y_test[i][0]))

    RESULTS:
    ------------
    15.4842877705 - 14.56
    19.258489367 - 16.32
    13.0593775565 - 13.09
    25.4906851541 - 22.87
    6.81600626706 - 8.94
    21.6278221281 - 19.13
    13.8979465655 - 13.77
    etc.
    ```

    ### R<sup>2</sup>
    ```python
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_test_pred)
    print(r2)

    RESULTS:
    ------------
    0.878324952758
    ```
    ### R<sup>2</sup> (2nd approach)
    ```python
    print(model.score(X_test, y_test))

    RESULTS:
    ------------
    0.878324952758
    ```

### Coefficient of Determination (R<sup>2</sup>)
**R<sup>2</sup>** is the proportion of the variation in the dependent variable that is predictable from the independent variable(s). 

- **Mean** of observed Data

    <img src="https://render.githubusercontent.com/render/math?math=\bar{y}= \frac{1}{n} \sum_{i=1}^{n} y_i" width="130px">

- The **total sum of squares (SST)** (proportional to the variance of the data):

    <img src="https://render.githubusercontent.com/render/math?math=SST= \sum_{i=1}^{n} (y_i - \bar{y})^2 " width="200px">

- The sum of squares of residuals, also called the **residual sum of squares (SSR)**:

    <img src="https://render.githubusercontent.com/render/math?math=SSR= \sum_{i=1}^{n} (y_i - \hat{y_i})^2 = \sum_{i=1}^{n} e_i" width="300px">

- The regression sum of squares, also called the **explained sum of squares (SSE)**:

    <img src="https://render.githubusercontent.com/render/math?math=SSE= \sum_{i=1}^{n} (\hat{y_i} - \bar{y})^2 = \sum_{i=1}^{n} e_i" width="300px">

- In a simple linear regression, the **total sum of squares** equals the sum of the two other sums of squares defined above:

    <img src="https://render.githubusercontent.com/render/math?math=SST = SSR %2B SSE" width="200px">

   
- The most general definition of is **R<sup>2</sup>**:

    <img src="https://render.githubusercontent.com/render/math?math=R^2= 1- \frac{SSR}{SST}" width="170px">

- In the **best case** R<sup>2</sup>=1), the **modeled values** exactly match the **observed values**, which results in **SSR = 0** and **R<sup>2</sup> = 1**. 
- A **baseline model** (R<sup>2</sup>=0), which always predicts **ȳ**, will have **R<sup>2</sup> = 0**. 
- Models that have worse predictions than this baseline will have a negative R<sup>2</sup>, i.e. **R<sup>2</sup> < 0**.

### Code - R<sup>2</sup>
- Open Juypter Notebook ```R2 - Score.ipynb```

### Adjusted R<sup>2</sup>
**Adjusted R<sup>2</sup>** accounts for the phenomenon of the R<sup>2</sup> automatically and spuriously increasing when extra explanatory variables are added to the model:

<img src="https://render.githubusercontent.com/render/math?math=\bar{R}^2 = 1- (1-R^2)\frac{n-1}{n-p-1}" width="200px">

with **n** is the sample size and **p** is the total number of explanatory variables in the model.

- **Adjusted R<sup>2</sup>** can be negative.
- **Adjusted R<sup>2</sup>** will always be smaller than **R<sup>2</sup>**.
- **Adjusted R<sup>2</sup>** increases only when the increase in **R<sup>2</sup>** is more than one would expect by chance.
- If a set of explanatory variables with a predetermined hierarchy of importance are introduced into a regression one at a time, with the **Adjusted R<sup>2</sup>** computed each time, the level at which **Adjusted R<sup>2</sup>** reaches a maximum, and decreases afterward, would be the regression with the ideal combination of having the best fit without excess/unnecessary terms. 

## Logistic regression <a id="log_reg"></a>
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

## Support vector machine <a id="svm"></a>
A support vector machine is a popular supervised learning model used for both classification and regression. That said, it is typically leveraged for classification problems, constructing a hyperplane where the **distance between two classes of data points is at its maximum**.

There are a few important parameters of SVM:

**Kernel**: A kernel helps us find a hyperplane in the higher dimensional space without increasing the computational cost. Usually, the computational cost will increase if the dimension of the data increases. This increase in dimension is required when we are unable to find a separating hyperplane in a given dimension and are required to move in a higher dimension:

![image3] ![image4]

**Hyperplane**: This is basically a separating line between two data classes in SVM. But in Support Vector Regression, this is the line that will be used to predict the continuous output.

**Decision Boundary**: A decision boundary can be thought of as a demarcation line (for simplification) on one side of which lie positive examples and on the other side lie the negative examples. On this very line, the examples may be classified as either positive or negative. This same concept of SVM will be applied both in Support Vector Regression and Classification.

**How does it look like for classification?**
- Example binary problem: 2 classes for each of the data points **x<sub>i</sub>** as being­ **-1** or **1**, i.e. **y<sub>i</sub> ∈ {­-1, 1}**.
- Hyperplane function has the equation **w<sup>T</sup> x + b** and is defined such that for all points that have a class 
    - **y<sub>i</sub> = ­-1**:

        <img src="https://render.githubusercontent.com/render/math?math=w^Tx %2B b \leq -1" width="130px">

    - **y<sub>i</sub> = ­1**:

        <img src="https://render.githubusercontent.com/render/math?math=w^Tx %2B b \geq 1" width="130px">

- In the training data there should be no points between the boundaries **w<sup>T</sup> x + b = -1** and **w<sup>T</sup> x + b = 1**. A region called **margin**.  
- The dividing plane is the hyperplane function **w<sup>T</sup> x + b = 0**.
- We classify new points by their function 

    <img src="https://render.githubusercontent.com/render/math?math=\hat{y}_i = sign(w^Tx %2B b)" width="200px">

    ![image34]

- **Goal**: Maximize size of the margin!
    - Find points which lie on opposite margins and are as close as possible to one another, i.e. **x<sub>+</sub>** and **x<sub>-</sub>** 
    - These points define 
        - **w<sup>T</sup> x<sub>+</sub> + b = 1** and
        - **w<sup>T</sup> x<sub>-</sub> + b = -1**
    - Subtract the two equations 
        - **w<sup>T</sup>(x<sub>+</sub> - x<sub>-</sub>) = 2**
    - Size of the margin: As vector **w** and vector **x<sub>+</sub> - x<sub>-</sub>** are parallel:
        - **||x<sub>+</sub> - x<sub>-</sub>|| = 2/ ||w||**
    - Maximizing the size of the margin **2/ ||w||** is equivalent to finding the minimum of **||w||** resulting in:
        - **y<sub>i</sub>(w<sup>T</sup> x<sub>+</sub> + b) ≥ 1**
    - Technically, minimizing **||w||** is the same as minimizing **1/2||w||<sup>2</sup>** an we can use [Lagrange multiplier](https://en.wikipedia.org/wiki/Lagrange_multiplier) to solve this task.
    - All points satisfying the equation **y<sub>i</sub>(w<sup>T</sup> x<sub>+</sub> + b) ≥ 1** are **support vectors** for the model.

    ![image36]

- The process above for linearly separable data (**hard-margin SVMs**) 
    - In case of mostly linearly separable data, we can consider using **soft­-margin SVMs**.
    - In case of really non-lineary separable data, we can use **kernel functions**, to be able to capture a nonlinear dividing curve between classes.

### Soft­-margin SVMs
- Data does not need to be completely linearly separable. It allows to classify some points incorrectly. 
- We provide for each point a non-negative slack variable **ξ<sub>i</sub>**, that illustrates to what degree each point is misclassified    
    - **y<sub>i</sub>(w<sup>T</sup> x<sub>+</sub> + b) ≥ 1 - ξ<sub>i</sub>**
- If a point is classified correctly than
    - **ξ<sub>i</sub> = 0**
- If a point is misclassified than 
    - **ξ<sub>i</sub> > 0**
- **Goal**: Minimize
    - **1/2||w||<sup>2</sup> + C∑ξ<sub>i</sub>**
    - C is a regularization parameter (with smaller C we punish errors less --> thus increase the size of the margin)
    - For C --> infinity we implement a hard punishment resulting in hard-margin SVM again.

### Kernel functions
- Used for data which is only non-linearly separable.
- The kernel function should capture some aspect of similarity in the data.
- It also represents domain knowledge regarding the structure of the data.
- In general we can write the Lagrange multiplier function we want to maximize as

    ![image35]

    - **k(x<sub>i</sub>, x<sub>j</sub>) = x<sub>i</sub><sup>T</sup>x<sub>j</sub>** --> dividing hyperplane
    - **k(x<sub>i</sub>, x<sub>j</sub>) = (x<sub>i</sub><sup>T</sup>x<sub>j</sub>)<sup>2</sup>** --> dividing hypersphere
    - **k(x<sub>i</sub>, x<sub>j</sub>) = (x<sub>i</sub><sup>T</sup>x<sub>j</sub> + C)<sup>d</sup>** --> general form for **polynomial kernels**

    ![image37]

- It is important to note that natural task of SVMs lies in binary classification.
- For classification tasks involving more than two groups, use multiple binary classifiers
    - we may create one classifier for each class in a one­ versus ­all fashion or
    - set up classifiers in a paiwise comparison and select the class ,wich wins the most pairwise matchups

**How does it look like for regression?**
- As with **SVMs** we project data in an **SVR** task using a kernel function so that they can be fit by a hyperplane. Instead of dividing the data into classes, however, the hyperplane now provides an estimate for the data’s output value.
- In addition the margin and error are treated differently. A parameter ε is specified  such that small deviations from the regression hyperplane do not contribute to error costs, i.e. when we attempt to minimize
    - **1/2||w||<sup>2</sup> + C∑ξ<sub>i</sub>**

- Essentially, SVR operates in much the same way as SVM does.
- For each point in the training data, we instead have two slack variables, **ξ<sub>i</sub>** and **ξ<sub>i</sub><sup>'</sup>**, one for positive deviations and one for negative deviations fromt he regression hyperplane. This results in two Lagrangian multipliers associated with each point
    - **0 ≤ α<sub>i</sub>, α<sub>i</sub><sup>'</sup> ≤ C** 
    
    and a respecified constraint on weight values 
    - **∑(α<sub>i</sub> − α<sub>i</sub><sup>'</sup>) = 0** 

    When solved, the regression function takes the form

    - **∑(α<sub>i</sub> − α<sub>i</sub><sup>'</sup>)k(x<sub>i</sub><sup>T</sup>x) + b**

## Decision trees <a id="dec_trees"></a>
Nice overview: [Decision Tree Algorithm, Explained](https://www.kdnuggets.com/2020/01/decision-tree-algorithm-explained.html)

The decision tree method is a powerful and popular predictive machine learning technique that is used for both **Classification** and **Regression**. 

**Classification**:
We use Classification Trees when need to classify the targets (e.g. passenger survived or died) 
Decision Tree which has a categorical target variable then it called a **Categorical variable decision tree**.

**Regression**:
We use  Regression Trees when we need to predict continuous values like price of a house. 
Decision Tree which has a continuous target variable then it is called **Continuous Variable Decision Tree**. 

In general, Decision Tree algorithms are referred to as **CART** or **Classification and Regression Trees**.

### Important Terminology related to Decision Trees
1. **Root Node**: It represents the entire population or sample and this further gets divided into two or more homogeneous sets.
2. **Splitting**: It is a process of dividing a node into two or more sub-nodes.
3. **Decision Node**: When a sub-node splits into further sub-nodes, then it is called the decision node.
4. **Leaf / Terminal Node**: Nodes do not split is called Leaf or Terminal node.
5. **Pruning**: When we remove sub-nodes of a decision node, this process is called pruning. You can say the opposite process of splitting.
6. **Branch / Sub-Tree**: A subsection of the entire tree is called branch or sub-tree.
7. **Parent and Child Node**: A node, which is divided into sub-nodes is called a parent node of sub-nodes whereas sub-nodes are the child of a parent node. 

    ![image8]

Decision trees **classify the examples by sorting them down the tree** from the root to some leaf/terminal node, with the leaf/terminal node providing the classification of the example.

**Each node in the tree acts as a test case for some attribute**, and each edge descending from the node corresponds to the possible answers to the test case. This process is recursive in nature and is repeated for every subtree rooted at the new node.

### How do Decision Trees work?

The decision of **making strategic splits** heavily affects a tree’s accuracy. The decision criteria are different for classification and regression trees.

Decision trees use multiple algorithms to decide to split a node into two or more sub-nodes. The creation of sub-nodes increases the homogeneity of resultant sub-nodes. In other words, we can say that the purity of the node increases with respect to the target variable. The decision tree splits the nodes on all available variables and then selects the split which results in most homogeneous sub-nodes.

### Important algorithm: ID3
1. It begins with the original set **S** as the **root node**.
2. On each iteration of the algorithm, it iterates through the very unused attribute of the set S and **calculates Entropy(H)** and **Information gain(IG)** of this attribute.
3. It then selects the attribute which has the **smallest Entropy** or **largest Information gain**.
4. The set **S** is then **split by the selected attribute** to produce a subset of the data.
5. The algorithm continues to **recur** on each subset, considering only attributes never selected before. 

### Attribute Selection Measures 
If the dataset consists of **N attributes** then deciding **which attribute to place at the root** or at different levels of the tree as internal nodes is a complicated step. By just randomly selecting any node to be the root can’t solve the issue. If we follow a **random approach**, it may give us **bad results** with low accuracy.

For solving this attribute selection problem use some criteria like:
- Entropy,
- Information gain,
- Gini index,
- Gain Ratio, see [Decision Tree Algorithm, Explained](https://www.kdnuggets.com/2020/01/decision-tree-algorithm-explained.html)
- Reduction in Variance, see [Decision Tree Algorithm, Explained](https://www.kdnuggets.com/2020/01/decision-tree-algorithm-explained.html)
- Chi-Square, see [Decision Tree Algorithm, Explained](https://www.kdnuggets.com/2020/01/decision-tree-algorithm-explained.html)

These criteria will **calculate values for every attribut**e**. The values are **sorted**, and attributes are placed in the tree by following the order i.e, the **attribute with a high value(in case of information gain) is placed at the root**.
While using **Information Gain**as a criterion, we assume attributes to be **categorical**, and for the **Gini index**, attributes are assumed to be **continuous**.

### Entropy
Entropy is a **measure of the randomness** in the information being processed. **The higher the entropy the harder it is to draw any conclusions from that information**. Flipping a coin is an example of an action that provides information that is random.

![image9]

From the above graph, it is quite evident that the 
- entropy **H(X) = 0** when the probability is either 0 or 1. 
- entropy **H(X) = 1** when the probability is 0.5 because it projects perfect randomness in the data 

**ID3 follows the rule — A branch with an entropy of zero is a leaf node and A brach with entropy more than zero needs further splitting**. 

Mathematically Entropy for 1 attribute is represented as:

![image10]

- S → Current state
- p<sub>i</sub> → Probability/percentage of an event/class i in a node of state S

Mathematically Entropy for multiple attributes is represented as:

![image12]
- T → Current state
- X → Selected attribute

### Information Gain
Information gain or IG is a statistical property that **measures how well a given attribute separates the training examples** according to their target classification. Constructing a **decision tree is all about finding an attribute that returns the highest information gain and the smallest entropy**.

![image11]

Information gain is a decrease in entropy. It computes the difference between **entropy before split and average entropy after split** of the dataset based **on given attribute values**. ID3 (Iterative Dichotomiser) decision tree algorithm uses information gain.

Mathematically, IG is represented as:

![image13]

- before → the dataset before the split
- K → number of subsets generated by the split
- (j, after) → subset j after the split

### Gini Index
You can understand the Gini index as a **cost function used to evaluate splits in the dataset**. It is calculated by **subtracting the sum of the squared probabilities of each class from one**. It favors larger partitions and easy to implement whereas information gain favors smaller partitions with distinct values.

![image14]

Gini Index works with the categorical target variable “Success” or “Failure”. It performs only Binary splits.

**Higher value of Gini index implies higher inequality, higher heterogeneity**. 

Steps to Calculate Gini index for a split:
- Calculate Gini for sub-nodes, using the above formula for success(p) and failure(q) (p²+q²).
- Calculate the Gini index for split using the weighted Gini score of each node of that split. 

CART (Classification and Regression Tree) uses the Gini index method to create split points.

### How to avoid/counter Overfitting in Decision Trees?
The common problem with Decision trees, especially having a table full of columns, they fit a lot. Sometimes it looks like the tree memorized the training data set. If there is no limit set on a decision tree, it will give you 100% accuracy on the training data set because in the worse case it will end up making 1 leaf for each observation. Thus this affects the accuracy when predicting samples that are not part of the training set.

Here are two ways to remove overfitting:
1. Pruning Decision Trees.
2. Random Forest 


### Inductive Bias
The [inductive bias](https://en.wikipedia.org/wiki/Inductive_bias) (also known as learning bias) of a learning algorithm is the set of assumptions that the learner uses to predict outputs of given inputs that it has not encountered. 

Inductive Bias in ID3:
    - Good Splits at thee top
    - Correct over incorrect (ID3 prefers splits that model the data better than splits that model the data worse)
    - Shorter Trees are preferred

### Continuous Attributes
For example age --> **create ranges**

![image19]

### When should we stop training (ID3)?
Answers: 
    - When everything is classified correctly.
    - No more attributes!
    - No overfitting!

### Pruning Decision Trees
In pruning, you trim off the branches of the tree, i.e., remove the decision nodes starting from the leaf node such that the overall accuracy is not disturbed. This is done by segregating the actual training set into two sets: training data set, D and validation data set, V. Prepare the decision tree using the segregated training data set, D. Then continue trimming the tree accordingly to optimize the accuracy of the validation data set, V.

![image15]

### Example code
- Libraries
    ```python
    import numpy as np
    import matplotlib.pyplot as plt 
    import pandas as pd

    # train test split
    from sklearn.model_selection import train_test_split
    
    # feature scaling
    from sklearn.preprocessing import StandardScaler

    # DecisionTreeClassifier
    from sklearn.tree import DecisionTreeClassifier

    # For evaluation of accuracy
    from sklearn import metrics

    # confusion_matrix
    from sklearn.metrics import confusion_matrix

    # ListedColormap
    from matplotlib.colors import ListedColormap

    # For Tree visualization
    from sklearn.tree import export_graphviz
    from sklearn.externals.six import StringIO  
    from IPython.display import Image  
    import pydotplus
    ```
- Load the dataset. It consists of 5 features, UserID, Gender, Age, EstimatedSalary and Purchased.
    ```python
    data = pd.read_csv('/Users/ML/DecisionTree/Social.csv')
    data.head()
    ```
- Create feature columns, dependent and independent variables
    ```python
    feature_cols = ['Age','EstimatedSalary']
    X = data.iloc[:,[2,3]].values
    y = data.iloc[:,4].values
    ```
- Train-Test-Split
    ```python
    X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size = 0.25, random_state= 0)
    ```
- Perform feature scaling
    ```python
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    ```
- Fit the model in the Decision Tree classifier
    ```python
    classifier = DecisionTreeClassifier()
    classifier = classifier.fit(X_train,y_train)
    ```
- Make predictions
    ```python
    y_pred = classifier.predict(X_test)
    ```
- Check accuracy
    ```python
    print('Accuracy Score:', metrics.accuracy_score(y_test,y_pred))

    RESULTS:
    ------------
    The decision tree classifier gave an accuracy of 91%.
    ```
- Confusion Matrix
    ```python
    cm = confusion_matrix(y_test, y_pred)
    
    RESULTS:
    ------------

    array([[64,  4],
       [ 2, 30]])

    # It means 6 observations have been classified as false.
    ```
- Let us first visualize the model prediction results.
    ```python
    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop= X_set[:,0].max()+1, step = 0.01),np.arange(start = X_set[:,1].min()-1, stop= X_set[:,1].max()+1, step = 0.01))
    plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)alpha=0.75, cmap = ListedColormap(("red","green")))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())for i,j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1], c = ListedColormap(("red","green"))(i),label = j)
    plt.title("Decision Tree(Test set)")
    plt.xlabel("Age")
    plt.ylabel("Estimated Salary")
    plt.legend()
    plt.show()
    ```
    ![image16]
- Let us also visualize the tree:
    ```python
    dot_data = StringIO()
    export_graphviz(classifier, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    Image(graph.create_png())
    ```
    ![image17]
- Optimizing the Decision Tree Classifier
    - **criterion**: optional (default=”gini”) or Choose attribute selection measure: This parameter allows us to use the different-different attribute selection measure. Supported criteria are “gini” for the Gini index and “entropy” for the information gain.
    - **splitter**: string, optional (default=”best”) or Split Strategy: This parameter allows us to choose the split strategy. Supported strategies are “best” to choose the best split and “random” to choose the best random split.
    - **max_depth**: int or None, optional (default=None) or Maximum Depth of a Tree: The maximum depth of the tree. If None, then nodes are expanded until all the leaves contain less than min_samples_split samples. The higher value of maximum depth causes overfitting, and a lower value causes underfitting (Source).
    ```python
    # Create Decision Tree classifer object
    classifier = DecisionTreeClassifier(criterion="entropy", max_depth=3) 

    # Train Decision Tree Classifer
    classifier = classifier.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = classifier.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    RESULTS:
    ------------
    The classification rate increased to 94%.
    ```
- Visualize the **pruned** Decision Tree
    ```python
    dot_data = StringIO()
    export_graphviz(classifier, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True, feature_names = feature_cols,class_names=['0','1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    Image(graph.create_png())
    ```
    ![image18]

    This pruned model is less complex, explainable, and easy to understand than the previous decision tree model plot.


## Random Forest <a id="random_forest"></a>
Decision Trees (classification or regression) are the building blocks of the random forest model.

Two key concepts that give it the name random:
1. A random sampling of training data set when building trees.
2. Random subsets of features considered when splitting nodes. 

Random forest, like its name implies, consists of a large number of individual Decision Trees that operate as an ensemble. Each individual tree in the random forest spits out a class prediction and the class with the most votes becomes our model’s prediction (see figure below).

![image7]

The fundamental concept behind random forest is a simple but powerful one — the wisdom of crowds. In data science speak, the reason that the random forest model works so well is:

A large number of relatively uncorrelated models (trees) operating as a committee will outperform any of the individual constituent models.

The reason for this effect is that the trees protect each other from their individual errors. While some trees may be wrong, many other trees will be right, so as a group the trees are able to move in the correct direction. So the prerequisites for random forest to perform well are:

1. We need **features that have at least some predictive power** so that models built using those features do better than random guessing.
2. The predictions (and therefore the errors) made by the individual trees need to have **low correlations** with each other.

## Instance-based algorithms <a id="instance_based"></a>
- Decision trees, regression, neural networks, SVMs, Bayes nets: all of these can be described as **eager learners**. We fit a function that best fits our training data.
- Here, we look at an example of a **​lazy learner** in the ​k​­nearest neighbor salgorithm. In contrast to eager learners, lazy learners do not compute a function to fit the training data before new data is received. Instead, new instances are compared to the training data itself to make a classification or regression judgment. Essentially, the data itself is the function to which new instances are fit.

![image20]

Instead of creating a function (function approximation) we create a lookup table for data points (like a database). However, how can we implement **generalization** and insert interpolation between database data points?

--> **Nearest Neighbour** 

A good example of an instance-based algorithm is **K-Nearest Neighbor** or **KNN**. KNN algorithm is a simple algorithm that can be used to solve both classification and regression problems. The KNN algorithm assumes that similar things exist in proximity. 

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

### Bias in KNN
- **Locality** --> Near points are similar
- **Smoothness** --> Averaging
- **All features matters equaly**

### Curse of Dimensionality
As the number of Features or Dimensions grow, the amount of data we need to generalize accurately grows exponentially.

![image21]

Here: Number of data points increases with **5<sup>d</sup>**

## Ensemle Learning <a id="ensemble"></a> 
**Famous example: AdaBoost** - The AdaBoost algorithm trains multiple weak classifiers on training data, and then combines those weak classifiers into a  single boosted classifier. The combination is done through a weighted sum of the weak classifiers with weights dependent on the weak classifier accuracy.

Interseting overview: [Ensemle Learning](http://www.scholarpedia.org/article/Ensemble_learning)

### Weak Learner
- Does better than random guessing,
- Error **P<sub>D</sub>[h(x) ≠ c(x)] ≤ 1/2** (The probability for a distribution where the hypothesis is incrorrect for a given data point and class label)

### Bagging:

![image22]

An Example:

![image23]

### Boosting:

![image24]

- Training set **{(x<sub>i</sub>, y<sub>i</sub>)}**
    - **x<sub>i</sub>** = a vector that in general may contain multiple features
    - **y<sub>i</sub>** = **{-1, +1}** --> classification label
- **w<sub>i</sub>** = importance weight that (how important is the example for the current learning task). It is often domain knowledge. One advantage of using importance weights is that we can generalize the notion of error to take into account those weights.Let’s say we have a classifier **G** that takes input features **x<sub>i</sub>** and produces a prediction **G(x<sub>i</sub>)** ∈ **{-1, +1}**.
    We can measure its training error by simply counting all the misclassified training examples:

    ![image25]

    - The function I will return **1** whenever the true label **y<sub>i</sub>** does not agree with the classification prediction **G(x<sub>i</sub>)** and **0** when they do agree. A better way to compute the error is to take advantage of the importance weight:

    ![image26]

    - Here we multiply each misclassification by the importance weight **w<sub>i</sub>**. In this way, our error metric is more sensitive to misclassified examples that have a greater importance weight. Even if we get many examples wrong, we may still get a low error rate, because in a sense some examples are more important than others.

    - The factor in the denominator **∑wi** is simply a normalization factor, ensure that the error is normalized (between 0 and N) in case some of the weights become very large. In the boosting algorithm, the importance weights **w<sub>i</sub>** are sequentially updated by the algorithm itself. Here is an outline of the algorithm­ we are going to go through it step by step:

### Boosting in Pseudo-code:

1. Initialize the importance weights **w<sub>i</sub> = 1/N** for all training examples **i**.
2. For m = 1 to M:
    - Fit a classifier **G<sub>m</sub>(x)** to the training data using the weights **w<sub>i</sub>**. 
    - Compute the error

        ![image27]
    
    - Compute the alpha parameter:

        ![image28]

        for **i = 1,2, .. N**

        The parameter **&alpha;<sub>m</sub>** is therefore telling us how well the classifier **G<sub>m</sub>** performs on training data -  ­large **&alpha;<sub>m</sub>** implies low error and therefore accurate performance.

    - Update weights:

        ![image29]

        The last step inside the loop is to update the importance weights **w<sub>i</sub>**. The old weight is multiplied by an exponential  term that depends on both **&alpha;<sub>m</sub>** and whether the classifier was correct at predicting the training example **i**  corresponding to the importance weight **w<sub>i</sub>**. Suppose that a training example **j** is difficult to classify - ­i.e. a classifier **G<sub>m</sub>** fails to classify **j** correctly meaning 

        ![image31]


        As a result, the importance weight of **j** will increase:
        
        ![image32]

        Then the next classifier **G<sub>m+1</sub>** will “pay more attention” to example **j** during classification Gm+1 training, since **j** now has a greater weight. The opposite holds for examples that were correctly classified in the previous iteration - ­future classifiers will have a lower priority of correctly classifying such examples.


3. Return

    ![image30]

    Finally, we combine all classifiers **G<sub>m</sub>** for m = 1...M into a single boosted classifier **G** by doing a weighted sum on the weights. In this way, classifiers that have a poor accuracy (high error rate, low **&alpha;<sub>m</sub>**) are penalized in the final sum.

![image33]

- Adding more weak learners error stays the same but confidence (margin see SVM) increases.

![image38]

## Computational Learning Theory <a id="comp_learn_theo"></a> 
A formal way of 
- defining learning problems
- showing specific algorithms work
- show these problems are fundamentally hard (cannot be solved by the chosen algorithm)

### Resources in Machine Learning
Theory of computing analyzes how algorithms use resources
- time: O(nlogn)
- space: O(n<sup>2</sup>)
- samples (data)

### Inductive Learning
Learning from examples. Some terminologies
1. Probability of successful training, **1- &delta;** (maybe failure due to noisy data)
2. Number of examples to train on, **m**
3. Complexity of hypothesis class, **complexity of H** (maybe more data needed to nail things down)
4. Accuracy to which target concept is approximated, **&epsilon;**
5. Manner in which training examples are presented, **batch** or **online**
6. Manner in which training examples are selected

### Selecting training examples
Learner / Teacher
1. Learner asks questions of teacher (like: **learner** provides input X, teacher answers with C(X)?)
2. Teacher gives examples to help learner (like: **teacher** choses X and tells C(X))
3. Fixed distribution (X chosen from D by **nature**).
4. Evil-worst distribution

### How many questions to ask before nailing down the right answer in case of
- teacher: **only 1** (teacher knows the answer, he can ask the 'nailing' quaestion)
- learner: **log<sub>2</sub>|H|**, tries to split with each question the training set into half (because the learner does not no the answer), (H: set of possibilities)

### Some definitions
**Computational complexity**:
How much computational effort is needed for a learner to converge?

**Sample complexity - batch**: 
How many training examples are needed for a learner to create a successful hypothesis?

**Mistake bounds - online**:
How many missclassifications can a learner make over an inifinite run?

**Version Spaces**
- **Training set**: S ⊆ X --> class c(x) ∀ x∈S
- **True hypothesis**: c ∈ H
- **candidate hypothesis**: h ∈ H
- **consistent learner**: produces c(x) = h(x) for x∈S
- **version space**: VS(S) = {h s.t. h∈H consistent wrt S}

    ![image40]

### PAC Learning (Probably Approximately Correct)
What is the error of a hypothesis?
There are two errors:

- **Training error**: fraction of training examples misclassified by h.
- **True error**: fraction of examples that would be misclassified on sample drawn from D.

    - error<sub>D</sub>(h) = P<sub>x from D</sub>[c(x) ≠ h(x)]

### Definitions
- c: concept class
- L: Learner
- H: Hypothesis space
- n: |H|, size of hypothesis space
- D: distribution over inputs
- O ≤ &epsilon; ≤ 1/2 error goal
- O ≤ &delta; ≤ 1/2 certainty goal

### Rule
C is PAC-learnable by learning algorithm L using H if (only if) learner L will, with probability 1-&delta;, output a hypothesis h∈H such that error<sub>D</sub>(h)≤&epsilon; in time and samples polynomial in 1/&epsilon;, 1/&delta; & n.

### &epsilon;-exhausted version space
- VS(S) &epsilon;-exhausted iff (if and only if) ∀ h∈VS(S) error<sub>D</sub>(h)≤&epsilon;

![image41]

### Haussler Theorem - Bound True Error

![image42]

![image43]

![image44]

## Naive Bayes <a id="naive_bayes"></a>
A Naive Bayes classifier is a probabilistic machine learning model that’s used for classification task. 

The assumption made here is that the predictors/features are independent. That is presence of one particular feature does not affect the other. Hence, it is called **naive**.

Bayes Theorem allows us to integrate prior information with our data to come up with new information that we can use to confirm our possible suspicions.

### Bayes Theorem:

<img src="https://render.githubusercontent.com/render/math?math=P(h|D) = \frac{P(D|h) \cdot P(h)}{P(D)}" width="220px">

where 
- **P(h|D)** is the **posterior propability** ( --> the ***probability of h*** happening, given that ***D has occurred***)
- **h** represents a **hypothesis** and 
- **D** our **data**

**Simple Example**: Fishing in a river
- **P(C) = 60%** are catfish
- **P(S) = 40%** are stripers
- keep only fish if fish weight >  5lbs --> Law of total probability: **P(K) = P(K|S)*P(S) + P(K|C)*P(C) = .5*.4 + .6*1 = 0.8 (80%)**
- 50% of stripers are > 5lbs (**P(K|S) = 50%**) and all catfish are > 5lbs (**P(K|C) = 100%**) 
- **Question**: If we catch a fish and we get to keep it what is the probability the fish is a striper?

- **Answer**: Use Bayes Theorem
   
    <img src="https://render.githubusercontent.com/render/math?math=P(S|K) = \frac{P(K|S) \cdot P(S)}{P(K)} = \frac{.5*.4}{.8} = 0.25" width="400px">

    There is a 25% chance that we keep a fish that is a stripper.

### Using Bayes Theorem for Bayesian Learning
- We can now use Bayes rule to make a decision on which hypothesis is optimal in relation to our training data. We can do this by finding the maximum­ probability hypothesis given the data across all hypothesis. In math notation this looks like:

    <img src="https://render.githubusercontent.com/render/math?math=h_{MAP} = argmax_{h \in H} P(h|D)" width="350px">

    MAP = Maximum a Posteriori

    For all entries in the dataset, the denominator does not change, it remain static. Therefore, the denominator can be removed and a proportionality can be introduced.

- If all hypothesis P(h)'s are equivalent we instead can compute **P(h|D)** using **maximum likelihood**

    <img src="https://render.githubusercontent.com/render/math?math=h_{ML} = argmax_{h \in H} P(D|h)" width="350px">    

    ML = Maximum Likelihood

### Bayesian Classification 
- So far we found that the most probable hypothesis is **h<sub>MAP</sub> for a given dataset **D**. 
- Question: Given a new **x**, what is the best classification? 
- Often **h<sub>MAP</sub>(x)** is not the most propable classification.
- **Bayes optimal classifier**:
    
    <img src="https://render.githubusercontent.com/render/math?math=argmax_{v_j \in V} \sum_{h_j\in H} P(v_j|h_i)P(h_i|D)" width="350px">

    **v<sub>j</sub>** = set of possible classifications assigned by H

- Example:

    ![image49]

## Neural networks <a id="neural_net"></a>
Primarily leveraged for **deep learning** algorithms, neural networks process training data by mimicking the interconnectivity of the human brain through layers of nodes. Each node is made up of inputs, weights, a bias (or threshold), and an output. If that output value exceeds a given threshold, it “fires” or activates the node, passing data to the next layer in the network. Neural networks learn this mapping function through supervised learning, adjusting based on the loss function through the process of gradient descent. When the cost function is at or near zero, we can be confident in the model’s accuracy to yield the correct answer.

![image6]

## Unsupervised Learning <a id="usl"></a>

Algorithms for use with **unlabeled data** include the following:

## Clustering algorithms <a id="clustering"></a>
Think of clusters as groups. Clustering focuses on **identifying groups of similar records** and labeling the records according to the group to which they belong. This is done **without prior knowledge** about the groups and their characteristics. In other words, we try to find **homogeneous subgroups** within the data such that data points in each cluster are as similar as possible according to a similarity measure such as **euclidean-based distance** or **correlation-based distance**. The decision of which similarity measure to use is application-specific. Types of clustering algorithms include the K-means, TwoStep, and Kohonen clustering.

## K-means <a id="k_means"></a>

The way K-means algorithm works is as follows:

1. Specify **number of clusters K**.
2. **Initialize centroids** by first shuffling the dataset and then randomly selecting K data points for the centroids without replacement.
3. Compute the **sum of the squared distance** between data points and all centroids.
5. **Assign** each data point to the **closest cluster** (centroid).
6. **Compute the centroids** for the clusters by taking the average of the all data points that belong to each cluster.
7. **Keep iterating** 3-6 until there is no change to the centroids. i.e. assignment of data points to clusters isn’t changing.

The approach K-means follows to solve the problem is called Expectation-Maximization. The E-step is assigning the data points to the closest cluster. The M-step is computing the centroid of each cluster. 


## Association algorithms <a id="asso_algo"></a>
Association algorithms find patterns and relationships in data and identify frequent ‘if-then’ relationships called association rules. These are similar to the rules used in data mining.

## Step 3: Training the algorithm to create the model <a id="training"></a>

Training the algorithm is an iterative process–it involves running variables through the algorithm, comparing the output with the results it should have produced, adjusting weights and biases within the algorithm that might yield a more accurate result, and running the variables again until the algorithm returns the correct result most of the time. The resulting trained, accurate algorithm is the machine learning model—an important distinction to note, because 'algorithm' and 'model' are incorrectly used interchangeably, even by machine learning mavens.

## Step 4: Using and improving the model <a id="using"></a>

The final step is to use the model with new data and, in the best case, for it to improve in accuracy and effectiveness over time. From where the new data comes from will depend on the problem being solved. For example, a machine learning model designed to identify spam will ingest email messages, whereas a machine learning model that drives a robot vacuum cleaner will ingest data resulting from real-world interaction with moved furniture or new objects in the room.


# Setup Instructions <a id="Setup_Instructions"></a>
The following is a brief set of instructions on setting up a cloned repository.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.



## Prerequisites: Installation of Python via Anaconda and Command Line Interface <a id="Prerequisites"></a>
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

## Clone the project <a id="Clone_the_project"></a>
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

# Acknowledgments <a id="Acknowledgments"></a>
* This project is part of the Udacity Nanodegree program 'Data Science'. Please check this [link](https://www.udacity.com) for more information.

# Further Links <a id="Further_Links"></a>
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