# 1. Load the necessary library
# If you haven't installed it yet, run: install.packages("wsrf")
library(wsrf)

# 2. Load the iris dataset
data(iris)
ds <- iris

# 3. Prepare Train/Test Split
# Set a seed for reproducibility
seed <- 42
set.seed(seed)

# Create indices for training (70%) and testing (30%)
train_indices <- sample(nrow(ds), 0.7 * nrow(ds))
train_data <- ds[train_indices, ]
test_data <- ds[-train_indices, ]

# 4. Define the target variable and formula
target <- "Species"
form <- as.formula(paste(target, "~ ."))

# 5. Train the wsrf model
# We use parallel=FALSE for this simple example as per the guide
print("Training the model...")
model.wsrf <- wsrf(form, data = train_data, parallel = FALSE)

# Display model summary
print(model.wsrf)

# 6. Test the model
# Predict classes for the test data
predictions <- predict(model.wsrf, newdata = test_data, type = "class")$class
actual <- test_data[[target]]

# 7. Calculate Accuracy
accuracy <- mean(predictions == actual)
print(paste("Model Accuracy on Test Set:", round(accuracy * 100, 2), "%"))