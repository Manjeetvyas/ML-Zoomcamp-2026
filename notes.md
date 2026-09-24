# Module 1 -- Introduction

## 1.1  Introduction to machine learning

<b>Model Training</b>

Features of Data  + Target of data -> feed to machine learning -> Result will be a machine learning model which will predict target.


<b>Using a Model</b>

Feeding features from data to machine learning model -> Predict a Target from the data

<b>Features and Target</b>

Features are useful/key information about the objective.

Target is the end value we want to predict.

## 1.2 Rule Based System vs Machine Learning

### Key Comparison 

| Feature | Rule-Based System | Machine Learning System |
| :--- | :--- | :--- |
| **Logic Source** | Explicit rules written by software engineers | Automatically learned patterns from data |
| **Input to System** | Data + Rules | Data + Historical Labels |
| **Output** | Prediction / Decision | Model / Rules $\rightarrow$ then Predictions |
| **Adaptability** | Low (requires manual code updates) | High (retrain on new data) |
| **Maintenance** | Complex & fragile as rules grow | Structured pipeline & automated training |


## 1.3 Supervised Machine Learning

Supervised ML is first we feed/train ml algo on some features and target and tell exactly what we want to predict from our model. Then on test data we provide features on which it predicts target.

```text
[ 1, 0, 1, 1, 1 ]                                    [1]
[ 1, 1, 1, 0, 1 ]                                    [1]     
[ 1, 1, 1, 1, 1 ]   = X (features)           [1]   =  y(target)
[ 1, 0, 1, 0, 0 ]                                    [1]
[ 1, 0, 1, 0, 1 ]                                    [1]
```

So, our primary goal is to train the ml model in such a way that while using the model, ``` g(X)≈y ``` where, ``` g = ML model, 
X = Features matrix, 
y= Target vector ```

<b>Types of Supervised Machine Learning</b>

1.Regression - where your ml model predicts a numerical value. Ex- Car/House price prediction.

2.Classification - where your ml model predicts a categorical value.
  
2.1 Multiclass Classification- where your ml model have to predict a value from more than 3 category. Ex- Object detection.

2.2 Binary Classification-  where your ml model have to predict a value from 2 category. Ex- Spam detection.

3.Ranking - where your ml model predicts the probability of you liking something/a product (aka Ranking). Ex- Recommendation system.

## 1.4 CRISP-DM

<b>What is CRISP-DM?</b>

CRISP-DM stands for **Cross Industry Standard Process for Data Mining**. It is a standard methodology for organizing and structuring end-to-end Machine Learning projects.

It is an **iterative process** (not a one-way street), meaning insights gained in later steps often require looping back to refine earlier steps.

---

### The 6 Steps of CRISP-DM

```mermaid
flowchart TD
    BU[1. Business Understanding] <--> DU[2. Data Understanding]
    DU --> DP[3. Data Preparation]
    DP <--> M[4. Modeling]
    M --> E[5. Evaluation]
    E --> BU
    E --> D[6. Deployment]
    D --> BU
```



## 1.5 Model Selection Process

<b>Goal</b>
Train multiple candidate models (e.g., Logistic Regression, Decision Trees, Neural Networks) and select the one that generalizes best to unseen data.

---

### The 3-Way Data Split

To avoid the **Multiple Comparison Problem** (a model getting lucky on a single validation set by pure chance), split data into three separate sets (e.g., 60% / 20% / 20%):

* **Train Set ($X_{train}, y_{train}$):** Used strictly to fit/train candidate models.
* **Validation Set ($X_{val}, y_{val}$):** Used to evaluate, compare, and select the best model.
* **Test Set ($X_{test}, y_{test}$):** Untouched set used only at the very end to verify real-world generalization.

```mermaid
flowchart LR
    Full[Full Dataset] --> Train["Train Set (60%)"]
    Full --> Val["Validation Set (20%)"]
    Full --> Test["Test Set (20%)"]
```

---

### Workflow

```mermaid
flowchart TD
    A[Train Multiple Models on Train Set] --> B[Evaluate on Validation Set]
    B --> C[Select Best Model]
    C --> D[Combine Train + Val Data]
    D --> E[Retrain Best Model]
    E --> F[Final Verification on Test Set]
```

<b>Steps:</b>
1. **Train & Validate:** Train candidate models on `Train`, compare accuracy on `Validation`, and pick the top performer.
2. **Combine & Retrain:** Merge `Train + Validation` data and retrain the winning model to maximize training data.
3. **Test:** Evaluate once on `Test` to ensure validation performance was not due to luck.

## 1.6 Github Codespaces 

In this module we majorly learned how to submit homework of this zoomcamp using github.

## 1.7 Introduction to Numpy

 <b>refer introduction_to_numpy.ipynb in Folder 1.7</b>

## 1.8 Linear Algebra Refresher

 <b>refer linear_algebra_refresher.ipynb in Folder 1.8</b>


<b>1. Vector Operations</b>

![2 * u = result (White)](https://latex.codecogs.com/svg.latex?%5Ccolor%7Bwhite%7D2%20%5Ctimes%20%5Cbegin%7Bbmatrix%7D2%5C%5C4%5C%5C5%5C%5C6%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D4%5C%5C8%5C%5C10%5C%5C12%5Cend%7Bbmatrix%7D)


<b>2. Multiplication
- Vector-Vector Multiplication</b>





![u + v equation (white)](https://latex.codecogs.com/svg.latex?%5Ccolor%7Bwhite%7Du%2Bv%3D%5Cbegin%7Bbmatrix%7D2%5C%5C4%5C%5C5%5C%5C6%5Cend%7Bbmatrix%7D%2B%5Cbegin%7Bbmatrix%7D1%5C%5C0%5C%5C0%5C%5C2%5Cend%7Bbmatrix%7D%3D%5Cbegin%7Bbmatrix%7D3%5C%5C4%5C%5C5%5C%5C8%5Cend%7Bbmatrix%7D)

<b>        

- Matrix-Vector Multiplication</b>


![Matrix Vector Vertical](https://latex.codecogs.com/svg.latex?%5Ccolor%7Bwhite%7D%5Cbegin%7Baligned%7D%20U%20%26%3D%20%5Cbegin%7Bbmatrix%7D2%264%265%266%5C%5C1%262%261%262%5C%5C3%261%262%261%5Cend%7Bbmatrix%7D%2C%20%5Cquad%20v%20%3D%20%5Cbegin%7Bbmatrix%7D1%5C%5C0%5C%5C0%5C%5C2%5Cend%7Bbmatrix%7D%20%5C%5C%5C%5C%20Uv%20%26%3D%20%5Cbegin%7Bbmatrix%7Du_0%5ET%20v%5C%5Cu_1%5ET%20v%5C%5Cu_2%5ET%20v%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%202%5Ccdot%201%20%2B%204%5Ccdot%200%20%2B%205%5Ccdot%200%20%2B%206%5Ccdot%202%20%5C%5C%201%5Ccdot%201%20%2B%202%5Ccdot%200%20%2B%201%5Ccdot%200%20%2B%202%5Ccdot%202%20%5C%5C%203%5Ccdot%201%20%2B%201%5Ccdot%200%20%2B%202%5Ccdot%200%20%2B%201%5Ccdot%202%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D14%5C%5C5%5C%5C5%5Cend%7Bbmatrix%7D%20%5Cend%7Baligned%7D)


<b>        

- Matrix-Matrix Multiplication</b>

![Matrix Matrix Vertical Expanded to k](https://latex.codecogs.com/svg.latex?%5Ccolor%7Bwhite%7D%5Cbegin%7Baligned%7D%20U%20%26%3D%20%5Cbegin%7Bbmatrix%7D2%264%265%266%5C%5C1%262%261%262%5C%5C3%261%262%261%5Cend%7Bbmatrix%7D%2C%20%5Cquad%20V%20%3D%20%5Cbegin%7Bbmatrix%7D1%261%262%5C%5C0%260.5%261%5C%5C0%262%261%5C%5C2%261%260%5Cend%7Bbmatrix%7D%20%5C%5C%5C%5C%20UV%20%26%3D%20%5Cbegin%7Bbmatrix%7Du_0%5ET%20v_0%20%26%20u_0%5ET%20v_1%20%26%20u_0%5ET%20v_2%20%5C%5C%20u_1%5ET%20v_0%20%26%20u_1%5ET%20v_1%20%26%20u_1%5ET%20v_2%20%5C%5C%20%5Cvdots%20%26%20%5Cvdots%20%26%20%5Cvdots%20%5C%5C%20u_k%5ET%20v_0%20%26%20u_k%5ET%20v_1%20%26%20u_k%5ET%20v_2%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D14%2620%2613%5C%5C5%266%265%5C%5C5%268.5%269%5Cend%7Bbmatrix%7D%20%5Cend%7Baligned%7D)


<b>3. Identity Matrix</b>


![Identity Matrix Definition](https://latex.codecogs.com/svg.latex?%5Ccolor%7Bwhite%7DI%20%3D%20%5Cbegin%7Bbmatrix%7D1%260%260%5C%5C0%261%260%5C%5C0%260%261%5Cend%7Bbmatrix%7D%2C%20%5Cquad%20UI%20%3D%20IU%20%3D%20U)


<b>4. Inverse of a Matrix</b>

***NOTE :  To calculate inverse of a matrix , the matrix should be a square matrix***

![Matrix Inverse Calculation](https://latex.codecogs.com/svg.latex?%5Ccolor%7Bwhite%7D%5Cbegin%7Baligned%7DV_s%20%26%3D%20%5Cbegin%7Bbmatrix%7D1%261%262%5C%5C0%260.5%261%5C%5C0%262%261%5Cend%7Bbmatrix%7D%20%5C%5C%5C%5C%20V_s%5E%7B-1%7D%20%26%3D%20%5Cbegin%7Bbmatrix%7D1%261%262%5C%5C0%260.5%261%5C%5C0%262%261%5Cend%7Bbmatrix%7D%5E%7B-1%7D%20%3D%20%5Cbegin%7Bbmatrix%7D1%262%260%5C%5C0%26-%5Cfrac%7B2%7D%7B3%7D%26%5Cfrac%7B4%7D%7B3%7D%5C%5C0%26%5Cfrac%7B4%7D%7B3%7D%26-%5Cfrac%7B1%7D%7B3%7D%5Cend%7Bbmatrix%7D%20%5Cend%7Baligned%7D)



<b>5. A x A⁻¹ = Identity Matrix</b>



![Matrix Inverse Rule 3x3](https://latex.codecogs.com/svg.latex?%5Ccolor%7Bwhite%7DA%20%5Ctimes%20A%5E%7B-1%7D%20%3D%20%5Cbegin%7Bbmatrix%7D1%260%260%5C%5C0%261%260%5C%5C0%260%261%5Cend%7Bbmatrix%7D)


![Matrix Inverse Example](https://latex.codecogs.com/svg.latex?%5Ccolor%7Bwhite%7D%5Cbegin%7Bbmatrix%7D4%267%5C%5C2%266%5Cend%7Bbmatrix%7D%20%5Ctimes%20%5Cbegin%7Bbmatrix%7D0.6%26-0.7%5C%5C-0.2%260.4%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D1%260%5C%5C0%261%5Cend%7Bbmatrix%7D)



## 1.9 Introduction to Pandas

 <b>refer introduction_to_pandas.ipynb in Folder 1.9</b>

## 1.10 Summary of Module-1

Module 1 completed.


# Module 2 -- Regression

## 2.1 Project Overview - Car Price prediction Project

### Project Objective
Predict the price of a car based on a dataset from kaggle using linear regression.

---

### Project Plan
1. Data preparation and EDA (Exploratory Data Analysis).
2. Use Linear Regression for predicting price.
3. Understand the internals of linear regression.
4. Evaluating the model with RMSE.
5. Feature Engineering.
6. Regularization.

---

## 2.2 Data Preparation

 <b>car_price_prediction_project.ipynb in Folder Module 2</b>


## 2.3 Exploratory Data Analysis

 <b>refer car_price_prediction_project.ipynb in Folder Module 2</b>

## 2.4 Setting up Validation Framework

***To implement Linear Regression we have to split the data into 3 parts-> TRAIN , VALIDATION, TEST***

```mermaid
flowchart LR
    subgraph Test
        Xtest["X_test"]
        ytest["y_test"]
    end

    subgraph Validation
        Xv["X_v"]
        yv["y_v"]
    end

    subgraph Train
        Xt["X_t"]
        yt["y_t"]
    end

```


 
 
  Before splitting the dataframe, we will shuffle it, because sequential order of dataset can be a risk -- creating subsets that represent completely different distributions or biases.

 After splitting the data, 
 1. Extract target vectors: Isolate the target variable into y_train, y_val, and y_test to keep features and labels distinct.
   
 2. Drop target from feature sets: Remove the price column from X_train, X_val, and X_test to prevent target leakage.
 
 3. Apply log transformation: Transform the target vectors using $\log(1 + y)$ to reduce right-skewness, stabilize variance, and mitigate the impact of price outliers on error metrics.

 refer car_price_prediction_project.ipynb in Folder Module 2.


## 2.5 Linear Regression

<b>Goal:</b>

![Linear Regression Matrix Formulation](https://latex.codecogs.com/svg.latex?%5Cdpi%7B150%7D%20%5Ccolor%7Bwhite%7D%20%5Cbegin%7Baligned%7D%20g%28X%29%20%26%5Capprox%20y%20%5C%5C%20%5Ctext%7Bwhere%3A%7D%20%5Cquad%20g%20%26%3A%20%5Ctext%7Bmodel%20%28linear%20regression%29%7D%20%5C%5C%20X%20%26%3A%20%5Ctext%7Bfeature%20matrix%20(Train)%7D%20%5C%5C%20y%20%26%3A%20%5Ctext%7Btarget%20%28Price%29%7D%20%5Cend%7Baligned%7D)

---
<b>Lets perform Linear regression on a single row of data i.e df.iloc[10]</b> :


$$g(x_i) = w_0 + \sum_{j=1}^3 w_j \cdot x_{ij}$$

Where:

* $x_i$ is the feature vector of row $i$ (the data point or observation).
* $w_0$ is the base weight (bias / intercept).
* $w_1$ to $w_3$ are the feature weights corresponding to each feature $j$.
* $x_{ij}$ is the value of feature $j$ for row $i$.


 refer car_price_prediction_project.ipynb in Folder Module 2.

---

## 2.6 Linear Regression: Vector Form

***1. Implementing LR on a single vector of feature matrix using this approach:***

$$g(x_i) = w_0 + \sum_{j=1}^3 w_j \cdot x_{ij}$$
 

***2. Implementing LR on a single vector of feature matrix using new approach:***

$$g(x_i) = \sum_{j=0}^{n} w_j \cdot x_{ij}$$

This approach implements the linear regression using the bias-trick representation. Rather than separating the weight $w_0$ from the feature weights, it absorbs the bias term directly into the weight vector as its first element, creating $\mathbf{w} = [w_0, w_1, \dots, w_n]^T$. Correspondingly, the input feature vector has its first feature as 1, yielding $\mathbf{x}_i = [1, x_{i1}, \dots, x_{in}]^T$. 

This approach makes dot product b/w weights and features vector possible without a additional add operation, because now the shape of both is same.

***3. Implementing LR on a custom feature matrix using new approach:***

$$\mathbf{g}(\mathbf{X}) = \mathbf{X}\mathbf{w}$$

$$
\begin{bmatrix} g(\mathbf{x}_1) \\ \vdots \\ g(\mathbf{x}_m) \end{bmatrix}
=
\begin{bmatrix} 1 & x_{11} & \cdots & x_{1n} \\ \vdots & \vdots & \ddots & \vdots \\ 1 & x_{m1} & \cdots & x_{mn} \end{bmatrix}
\begin{bmatrix} w_0 \\ \vdots \\ w_n \end{bmatrix},
$$

where X is a square matrix with first column elements as 1.

***refer car_price_prediction_project.ipynb in Folder Module 2.***


## 2.7 Training Linear Regression Model: Normal Equation

***Deriving the Normal Equation for Linear Regression***

#### 1. Our Goal
Our goal is to find a weight vector $w$ such that the predicted values closely approximate our target values $y$:
$$g(X) = Xw \approx y$$

---

#### 2. The Ideal Scenario (If $X$ were Square and Invertible)
If $X$ is a square and invertible matrix, we can solve for $w$ by multiplying both sides by the inverse of $X$ ($X^{-1}$):

$$X^{-1} \cdot Xw = X^{-1} \cdot y$$

Since $X^{-1} \cdot X = I$ (the Identity matrix):
$$I \cdot w = X^{-1} \cdot y$$

And since any matrix multiplied by the Identity matrix remains unchanged ($I \cdot w = w$):
$$w = X^{-1} \cdot y$$

---

#### 3. The Catch with Rectangular Matrices
There is a major catch with this approach: **to find a standard inverse, the matrix must be square** (equal rows and columns). 

In real-world machine learning, our feature matrix $X$ is almost always a **rectangular matrix** (more samples/rows than features/columns). Because of this, a standard inverse $X^{-1}$ does not exist.

---

#### 4. Using the Gram Matrix to Fix Shape
To turn our rectangular feature matrix into a square matrix, we multiply it by its transpose ($X^T$). 

The result, $X^T X$, is a **Gram matrix**, which is **always a square matrix**. 

While a Gram matrix's inverse doesn't *always* exist (it depends on conditions like linear independence and having enough samples), for our current intuition, we assume the inverse exists.

---

#### 5. Deriving the Normal Equation (Pseudoinverse)
To solve for $w$ using the Gram matrix, we multiply both sides of our original equation by the pseudoinverse components:

$$(X^T X)^{-1} \cdot X^T \cdot Xw = (X^T X)^{-1} \cdot X^T \cdot y$$

*(Where $X^T X$ is our Gram matrix)*

Since $(X^T X)^{-1} \cdot (X^T \cdot X) = I$ (the Identity matrix):
$$I \cdot w = (X^T X)^{-1} \cdot X^T \cdot y$$

Finally, since $I \cdot w = w$, we arrive at the final **Normal Equation** for Linear Regression:
$$w = (X^T X)^{-1} X^T y$$

---

<u>Now that we know how to implement normal equation, lets implement this in code</u>, here- ***car_price_prediction_project.ipynb in Folder Module 2.***

This normal equation will help us to get ideal weights for our model's test and validation.

## 2.8 Baseline Model for Car Price Prediction Project

In this section,
1. Extract Numerical columns from our X-train dataframe to make them features.
2. Then we fill missing values of those features as 0.
3. We finally make feature matrix  'X' which consist values of features but without bias.
4. Now we will add bias to our feature matrix.
5. We will print our y-train vector to see its okay or not.
6. Now we will feed the X-train and y-train to the previously made train_linear_regression() model, this model uses normal equation to calculate weights for validation and test, by using X-train and y-train data.
7. Now, we will do dot product of X-train and weights, which will be our Prediction (y_pred).
8. Now, we will make a seaborn histplot to see residual (difference b/w predicted target {y_pred} vs target value {y_train}).
   
---
***We can say that:*** The model successfully captures the central tendency and overall spread of the target variable, though it exhibits a slight under-prediction shift and fails to capture the low-end cluster.

***Refer car_price_prediction_project.ipynb in Folder Module 2.***

## 2.9 RMSE (Root Mean Squared Error)

RMSE indirectly tells you the quality of your model.

![RMSE Formula](https://latex.codecogs.com/svg.image?\color{white}RMSE=\sqrt{\frac{1}{m}\sum_{i=1}^{m}(g(x_i)-y_i)^2})

Where:
* $g(x_i)$ is the prediction for $x_i$
* $y_i$ is the target value.
* $m$ is the number of observations.

Now lets perform RMSE on y-train and y-pred.

***Refer car_price_prediction_project.ipynb in Folder Module 2.***


## 2.10 Computing RMSE on Validation Data

Now lets perform RMSE on y-validation and y-pred (came from X-validation and y-validation).

***Refer car_price_prediction_project.ipynb in Folder Module 2.***


## 2.11 Feature Engineering
In this section we will add a new feature to our feature matrix X-train, which will help us in RMSE.

We used new feature 'age' which is calculated as (2017 - year).

and as a result we can see now that the RMSE is much improved.

***Refer car_price_prediction_project.ipynb in Folder Module 2.***

## 2.12 Categorical Variable

In this section we will add our several categorical variables from our df_train as new feature to our feature matrix X-train, which will help us in reducing RMSE.

we will make list of all categorical columns we want to add as features.

Then we will make a dictionary which will store 'column name' as key and 'Top 5 values of those columns' as value.

then we will add these these columns as features to our X-train, one by one.

Now we will run our model and check weather our RMSE improved or not.

***Refer car_price_prediction_project.ipynb in Folder Module 2.***

## 2.13 Regularization

***<u>Even though our RMSE is significantly improved, we will still apply Ridge (L_2) regularization as soon as we will add
new features to the the feature matrix X-train, because sometimes in the feature matrix X-train. Two or more than features 
become same, and for these cases the  $(X^T X)^{-1}$ i.e inverse of GRAM MATRIX doesnt exist.</u>***

---

>More on Regularization

<i>The Pitfalls of Unchecked Feature Expansion

While adding new features (such as interaction terms, polynomial variables, or external data) is a common strategy to improve model performance, expanding the feature matrix without caution can severely backfire. Often, this leads to an unexpected increase in the validation Root Mean Squared Error (RMSE) due to a few critical issues:

1. Overfitting to Noise: High-dimensional models tend to memorize random noise and anomalies in the training set rather than learning true underlying patterns.
2. Multicollinearity and Variance: Newly added features are frequently correlated with existing ones, causing regression coefficients to swing wildly and destabilize predictions.
3. The Curse of Dimensionality: As dimensions multiply, data points become sparse, making it harder for the algorithm to form reliable generalizations.

***Why Regularization is Essential***

Because expanding your feature space naturally increases model complexity and variance, regularization must always be performed after adding new features to the feature matrix. Regularization acts as a mathematical speed bump by penalizing excessively large coefficients:

Ridge ($L_2$) Regularization: Smoothly shrinks coefficient values toward zero, dampening the influence of noisy or redundant features.

Lasso ($L_1$) Regularization: Can drive coefficients down to zero entirely, effectively performing automated feature selection.

By pairing feature matrix enrichment with proper regularization, you keep coefficient complexity in check, prevent your RMSE from spiraling, and ensure robust real-world performance. </i>

---

So, We perform ridge regularization, using $(X^T X + \lambda I)$ , where Lambda is Alpha/Regularixation Parameter multiplied by Identity matrix.

We took the synthetic alpha for now, in the next module we will see how to calculate alpha for your model.

## 2.14 Tuning the model

In this section, we will take several samples of Regularization Alpha b/w 0 to 10, and iterate over them to find minimum RMSE.

after iterating over all samples, we found out that our model already providing us minimum RMSE without adding regularization alpha, but none the less we have to add a regularization alpha for a safer side, so will choose the minimum one, i.e $\lambda$ = 0.00001

## 2.15 Using the Final Model

Now that our Price Prediction Linear Regression Model is completely Validated.

Lets make a Final model, where data is:

`df_full_train = df_train + df_validation`

and it will be tested on `X-test and y-test`

1. First we will make our Feature Matrix `X-full` and target vector `y-full` out of `df_full_train`.
2. Now we will prepare weight using those.
3. Now we will calculate RMSE score of `X-full` , is it as same as previous models or not.
4. Because RMSE is similar to the previous models, we can say our Final Model is completed.
5. Now its time to test our model:  We will select a single car data out of `X-test`.
6. As we predicted the price of this car `y-pred` is very close to the real price of the car `y-test`, we can say our model is working Fantastic.
---




































