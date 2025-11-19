import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Set Matplotlib style for better visualization
plt.style.use('seaborn-v0_8-darkgrid')

# --- 1. Data Generation Function ---

def create_xor_data(n_samples=500):
    X = np.random.uniform(-1, 1, size=(n_samples, 2))
    
    # Labels (y): 1st and 3rd quadrants are 1 (blue), 2nd and 4th quadrants are 0 (orange)
    # This checks for (X1 > 0 AND X2 > 0) OR (X1 < 0 AND X2 < 0)
    y = np.logical_or(
        np.logical_and(X[:, 0] > 0, X[:, 1] > 0),
        np.logical_and(X[:, 0] < 0, X[:, 1] < 0)
    ).astype(int)
    
    # Add a small amount of Gaussian noise to make the problem more realistic
    X += np.random.normal(0, 0.05, size=X.shape)
    return X, y

# --- 2. Model Building and Training Function ---

def build_and_train_model(X, y, model_name, config, epochs=100, learning_rate=0.03):

    # 1. Network Structure
    model = Sequential(name=model_name)
    
    # Hidden Layers
    for i, (units, activation) in enumerate(config):
        input_shape = (X.shape[1],) if i == 0 else None
        model.add(Dense(units, activation=activation, input_shape=input_shape, 
                        name=f'Hidden_{i+1}_{units}_{activation}'))

    # Output Layer: Use Sigmoid for binary classification (output 0 or 1)
    model.add(Dense(1, activation='sigmoid', name='Output_Sigmoid'))

    # Print model summary (network structure)
    model.summary()
    
    # 2. Key Parameters and Activation Function
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Compile the model
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print(f"-> Learning Rate: {learning_rate}")
    print(f"-> Loss Function: Binary Crossentropy")

    # Train the model
    # validation_split=0.2 is used to track validation loss, simulating the Test Loss curve
    history = model.fit(X, y, epochs=epochs, verbose=0, validation_split=0.2)
    
    # Final performance evaluation
    loss, acc = model.evaluate(X, y, verbose=0)
    print(f"-> Final Accuracy after Training: {acc:.4f}")
    
    return model, history

# --- 3. Visualization Function ---

def plot_results(X, y, model, history, title):
    """Plots data points, decision boundary, and learning curves."""
    
    # --- Decision Boundary Plot ---
    plt.subplot(1, 2, 1)
    
    # Generate grid data
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Predict grid points
    grid_data = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_data, verbose=0)
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary (using 0.5 as the threshold)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu, levels=[-0.1, 0.5, 1.1])
    
    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k', marker='o')
    plt.title(f'Classification Problem {title} - Decision Boundary\nStructure: {model.name}', fontsize=10)
    plt.xlabel('$X_1$')
    plt.ylabel('$X_2$')

    # --- Learning Curve Plot ---
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title(f'Classification Problem {title} - Learning Curve (Loss)', fontsize=10)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(fontsize=8)


# --- 4. Main Execution Block ---

if __name__ == '__main__':

    tf.random.set_seed(114514)
    np.random.seed(114514)

    # Prepare plotting canvas
    plt.figure(figsize=(14, 6))
    plt.suptitle("Solution for Quadrant Classification (XOR) Problem", fontsize=16, y=1.05)

    X_xor, y_xor = create_xor_data()

    xor_config = [(4, 'relu')]
    
    model_xor, history_xor = build_and_train_model(
        X_xor, y_xor, 
        model_name="XOR_Solution", 
        config=xor_config, 
        epochs=100,
        learning_rate=0.03
    )

    # Visualize results (a)
    plot_results(X_xor, y_xor, model_xor, history_xor, '(a) Quadrant/XOR')

    # Adjust layout and display the chart
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()