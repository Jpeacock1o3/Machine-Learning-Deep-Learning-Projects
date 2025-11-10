
## Project 1 — *Classification and Regression Methods*
**Goal:** Build linear and logistic models from scratch using **NumPy**, focusing on fundamental mathematical derivations, optimization, and numerical stability.

### Parts of the assignment
1. **Data Preprocessing Pipeline**
   - Designed a reusable preprocessing class handling missing values, outliers, string normalization, encoding, and feature scaling.
   - Applied to multiple real-world datasets (Penguins, Wine, Buffalo Open Data, etc.).

2. **Logistic Regression (from scratch)**
   - Implemented **gradient descent**-based binary classifier using the **sigmoid** function and **cross-entropy loss**.
   - Explored learning rates, initialization, and convergence behavior.
   - Evaluated using **Accuracy, Precision, Recall, F1-score**, and **Confusion Matrix** visualizations.

3. **Linear Regression (OLS)**
   - Derived and implemented the **closed-form Ordinary Least Squares** solution.
   - Computed **MSE** and **R²** scores and visualized residuals.

4. **Ridge Regression**
   - Extended OLS with **L2 regularization**.
   - Investigated how different regularization strengths (λ values) affect overfitting and generalization.

5. **Elastic Net Regression**
   - Implemented hybrid **L1 + L2** regularization using **gradient descent**.
   - Experimented with weight initialization strategies (Zero, Random, Xavier) and stopping criteria (fixed iterations vs gradient threshold).

### Key Skills Demonstrated
- Pure NumPy-based model construction  
- Derivation and implementation of optimization algorithms  
- Bias–variance analysis  
- Visual diagnostics (loss curves, prediction scatterplots, correlation heatmaps)  
 

### Results
- Logistic Regression accuracy on test data: **> 70%**
- Ridge and Elastic Net models achieved improved generalization over OLS
- Developed fully reusable data-cleaning and normalization pipeline


## Project 2 — *Neural Networks and CNNs*
**Goal:** Implement, train, and optimize neural networks and convolutional models using **PyTorch** without pre-trained architectures.

### Parts of the project
1. **Fully Connected Neural Network (FCNN)**
   - Built a custom NN with multiple hidden layers and ReLU activations.
   - Trained on a tabular dataset to predict binary targets.
   - Achieved test accuracy > **75%** with tuned hyperparameters.

2. **Hyperparameter Optimization**
   - Systematically tuned dropout, batch size, learning rate, and optimizer type.
   - Applied techniques such as **learning rate scheduling**, **early stopping**, and **batch normalization** to boost performance.

3. **Convolutional Neural Network (CNN)**
   - Implemented a CNN from scratch for **multi-class image classification (36 categories × 28×28 px)**.
   - Integrated advanced techniques from Part II to achieve **>85% test accuracy**.
   - Visualized training vs validation accuracy/loss, confusion matrices, and ROC curves.

4. **VGG-13 Architecture**
   - Re-implemented **VGG-13 (Version B)** for adapted 28×28 dataset input.
   - Modified layer depth, pooling strategy, and output dimensions.
   - Trained from scratch to high accuracy while maintaining model efficiency.


### Tools & Techniques
- **Frameworks:** PyTorch, Torchinfo, Torchmetrics, Matplotlib, Seaborn  
- **Optimization:** Adam, RMSProp, SGD with momentum  
- **Regularization:** Dropout, Weight Decay, Data Augmentation  
- **Visualization:** Confusion Matrix, ROC Curve, Accuracy/Loss plots  


## Technologies Used
| Category | Tools |
|-----------|-------|
| Programming | Python 3.11 |
| Core Libraries | NumPy, Pandas, Matplotlib, Seaborn |
| ML Framework | PyTorch |
| Metrics | Torchmetrics, scikit-learn (for metrics only) |
| Environment | UB CCR GPU cluster, Local JupyterLab |


