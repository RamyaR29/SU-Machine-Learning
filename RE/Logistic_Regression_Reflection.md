# Reflection: Logistic Regression

**Resources reviewed:**  
- Video: https://www.youtube.com/watch?v=C5268D9t9Ak  
- Article: https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/

---

## What I Learned

### Key concepts from the logistic regression section

- **Logistic regression is for classification, not regression.** Despite the name, it’s used to predict a binary (or categorical) outcome (e.g., yes/no, spam/not spam), not a continuous number. The “regression” part refers to modeling a relationship between inputs and an outcome, but the outcome is a probability that we then turn into a class label.

- **The sigmoid (logistic) function** is what makes the model output a probability between 0 and 1. It takes any real number and squashes it into (0, 1) with an S-shaped curve. So we get something like “probability of class 1” instead of unbounded values like in linear regression.

- **Decision boundary:** The model learns a boundary (a line in 2D, or a plane/hyperplane in higher dimensions) that separates the two classes. On one side we predict one class, on the other side the other class. The boundary comes from setting the linear combination of features (plus bias) to zero; the sigmoid then turns that into a 0.5 probability threshold.

- **Odds and log-odds (logit):** Logistic regression actually models the *log-odds* of the positive class. Odds = p/(1−p), and log-odds can be any real number, so we can use a linear model in that space. The sigmoid is the inverse of the logit: it converts the linear score back into a probability.

- **Maximum Likelihood Estimation (MLE)** is used to train the model, not least squares like in linear regression. We choose the coefficients that make the observed labels most likely under the model. That fits the idea of predicting probabilities and then classifying.

- **Interpretation of coefficients:** A positive coefficient means that as that feature increases, the log-odds of the positive class increase, so the predicted probability of the positive class goes up. The size of the coefficient tells you how strong that effect is (often in “log-odds units” or, when exponentiated, as an odds ratio).

### Most important ideas I now understand clearly

1. **Why we need the sigmoid:** We need outputs between 0 and 1 for probability. Linear regression can go below 0 or above 1, so it’s not suitable for probability. The sigmoid fixes that.

2. **Classification vs. regression:** Logistic regression gives a *probability*, and we turn it into a class by using a threshold (usually 0.5). So we get both a score and a decision.

3. **Linear in log-odds, nonlinear in probability:** The model is linear in the log-odds (the equation is a linear combination of features). When we pass that through the sigmoid, the relationship between features and *probability* becomes nonlinear. That’s why we can get curved decision boundaries in the original feature space.

---

## What I’m Still Struggling With

### 1. Maximum Likelihood and the loss function

I get that we use MLE instead of minimizing squared error, but I still find it hard to see *exactly* how the log-loss (cross-entropy) comes from “maximizing the probability of the data.” I’d like to see the step-by-step link: how writing down “probability of observed labels given the model” leads to the formula we actually minimize (the negative log-likelihood). I also don’t yet have an intuitive feel for why that particular loss shape is a good fit for classification.

### 2. Gradient descent for logistic regression

I know the coefficients are learned by iteratively updating them (e.g., gradient descent), but the details are still fuzzy. How is the gradient of the log-loss with respect to each coefficient actually computed, and why does that update rule improve the model? I’d like to work through one small example (e.g., one feature, a few points) with pen and paper so it feels less like a black box.

### 3. Multiclass and regularization

The video and article focused on binary logistic regression. I’m not fully clear on how this extends to more than two classes (e.g., softmax and one-vs-rest). I also want to understand how regularization (like L1/L2) is added to logistic regression and how we choose the strength of regularization in practice.

---

*Reflection completed as part of the logistic regression assignment.*
