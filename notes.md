# Module 1
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

CRISP-DM stands for **CRoss Industry Standard Process for Data Mining**. It is a standard methodology for organizing and structuring end-to-end Machine Learning projects.

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

 <b>introduction_to_numpy.py in Folder 1.7</b>

## 1.8 Linear Algebra Refresher

 <b>linear_algebra_refresher.py in Folder 1.8</b>


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