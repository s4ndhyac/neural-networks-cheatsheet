# Machine Learning\Neural-networks-cheatsheet

- Plot and visualize statistical data pattern
  - check min, max, mean
  - use histograms (distribution of data in intervals) | limitation - only look at one feature at a time
  - use scatter plots - useful for looking at 2 features (2-D scatter plot) at the same time -> use all pairwise scatter plots to plot scatter plots for all features 2 at a time
  - use different colors for different classes
- supervised, unsupervised and semi-supervised, reinforcement learning
- two types - classification and regression
- data matrix
- data labels -> numerical values (categorical values)
- X -> features
- y -> target values
- y -> discrete -> classification | y -> continuous -> regression
- ML = meta-programming:
  1. Apply rules to examples
  2. get feedback on performance
  3. change predictor to do better (learn from feedback step 2 and improve step 1)
- y = {0, 1} -> binary classification
- y = {0, 1, 2, ...., k} -> multi-class classification
- y = continuous function -> regression
- Fundamental question -> How to find the optimal threshold parameter theta
- Loss function/ Cost function -> Loss = f(x, y) belongs to R tells us how good our model is, quality, performance
- How to improve loss? We train model on training data set
- 3 important things:
  - what is the model? f(x, theta)
  - what is the loss function? how do you quaantify the performance of your model? L(y`, y)
  - Learning (changes theta)
- Linear regression - 
  - Overfitting and complexity
    - simpler model performs better on test dataset
    - There is a inflection point where minima of the curve where predictive error is minimum for model complexity
- 3 splits -> train, validation, test
  - Learn parameters through train dataset
  - Evaluate models on validation and pick best performer
  - Reserve test dataset to benchmark performance
- what if sample size small?
  - solution: k-fold cross-validation
  - Divide into k folds
  - train on k-1 folds and evaluate on the remaining fold
  - pick model with best average performance in k trials
- Nearest neighbour regression - 1 D (Input vector x) - piecewise linear
- Nearest neighbour classifier - 2 D (Input vector x) - ambiguity along boundaries - boundary is piecewise linear - orthogonal to line connecting two data points
- In practice k-nearest neighbour used
  - regression: average the y values of k - nearest training examples
  - classification: class of max of k-nearest
    - increasing k simplifies decision boundary
    - majority voting -> less emphasis on individual points

- Bayes Classifier:
- categories (X is discrete) 
- Conditional probability given class label
- for continuous x we use density distributions like gaussian or histogram
- Naive Bayes -> Assumes conditional independence
- Naive Bayes -> linear | Bayesian - Exponential

#### Bayes Error
- Given: P(x,y) Prob. distribution of (x,y), x \in R   
- Suppose we predict y\` as follows:
y\` = 1 when x > x\`  because P(y=1|x) > P(y=0|x)  and 0 when x < x\` because P(y=0|x) > P(y=1|x)  but this also results in some wrong predictions for a gaussian prob. distribution
- Error function I = [y\`!=y] = { 1 if y\`!=y (error) and 0 if y\`==y}
- So Error = Integral over x and y of Probablity times Error function
- Some types of errors are more acceptable than others depending on context 
- for example: Wrt. medical data false positives are more acceptable than false negatives
- confusion matrix -> in the case of a binary classifier 2X2 matrice with 4 cells -> false +ve, true +ve, false -ve, true -ve
- liklihood ratio -> 
- ROC curve -> x-axis -> false positive rate (1 - specificity) vs. y-axis -> True positive rate (sensitivity)
  - This curve always goes from one corner 0 to opposite corner 1
  - Area under curve -> measure of confidence, higher the better, ideal curve is x=0 vertical line coinciding with y-axis, 0.5 < AUC < 1
- For a gaussian distribution use logs







  

