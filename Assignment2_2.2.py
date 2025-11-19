import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import make_circles 
import matplotlib.pyplot as plt

# Set Matplotlib style for better visualization
plt.style.use('seaborn-v0_8-darkgrid')

# --- 1. Data Generation Function ---

def create_circle_data(n_samples=500, factor=0.5, noise=0.05):
    X, y = make_circles(n_samples=n_samples, factor=factor, noise=noise)
    # X are the 2D coordinates; y labels are 0 or 1
    return X, y

# --- 2. Model Building and Training Function ---

def build_and_train_model(X, y, model_name, config, epochs=50, learning_rate=0.01):

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
                  loss='binary_crossentropy', # Appropriate loss for binary classification
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

    # Prepare plotting canvas (figsize adjusted for one plot set)
    plt.figure(figsize=(14, 6))
    plt.suptitle("Solution for Concentric Circles Classification Problem", fontsize=16, y=1.05)
    
    
    # --- Problem (b): Concentric Circles Classification ---
    X_circle, y_circle = create_circle_data()

    # This non-linear structure is required to separate the inner and outer rings.
    circle_config = [(5, 'tanh'), (3, 'tanh')]
    
    model_circle, history_circle = build_and_train_model(
        X_circle, y_circle, 
        model_name="Circle_Solution", 
        config=circle_config, 
        epochs=50,
        learning_rate=0.01
    )

    # Visualize results (b)
    plot_results(X_circle, y_circle, model_circle, history_circle, '(b) Concentric Circles')

    # Adjust layout and display the chart
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()