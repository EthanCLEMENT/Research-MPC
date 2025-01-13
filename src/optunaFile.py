import optuna

def objective(trial):
    # Define the search space
    lstm_hidden_size = trial.suggest_int('lstm_hidden_size', 16, 128, step=16)
    nn_hidden_size = trial.suggest_int('nn_hidden_size', 16, 128, step=16)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    batch_size = trial.suggest_int('batch_size', 32, 128, step=32)
    num_epochs = 50  # Fixed for simplicity; can be parameterized as well.

    # Update DataLoader with new batch_size
    train_seq_loader = DataLoader(train_seq_dataset, batch_size=batch_size, shuffle=True)
    test_seq_loader = DataLoader(test_seq_dataset, batch_size=batch_size, shuffle=False)
    
    # Create the model
    input_size = X_train_seq.shape[2]
    output_size = y_train_seq.shape[1]
    model = LSTMSNN(input_size, lstm_hidden_size, nn_hidden_size, output_size, num_layers)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in train_seq_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    # Evaluate on the validation set
    mse, _, _ = evaluate_model(model, test_seq_loader)
    return mse

# Create and run the Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

# Display the best hyperparameters
print("Best hyperparameters:")
print(study.best_params)

# Retrain the model using the best hyperparameters
best_params = study.best_params
best_model = LSTMSNN(
    input_size=X_train_seq.shape[2], 
    lstm_hidden_size=best_params['lstm_hidden_size'],
    nn_hidden_size=best_params['nn_hidden_size'], 
    output_size=y_train_seq.shape[1], 
    num_layers=best_params['num_layers']
)

# Update DataLoader with the best batch size
train_seq_loader = DataLoader(train_seq_dataset, batch_size=best_params['batch_size'], shuffle=True)
test_seq_loader = DataLoader(test_seq_dataset, batch_size=best_params['batch_size'], shuffle=False)

optimizer = optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])

# Train the best model
train_modelLSTMNN(best_model, optimizer, train_seq_loader, num_epochs=100)

# Evaluate the best model
mse_best, pred_best, actual_best = evaluate_model(best_model, test_seq_loader)
print(f"Best Model MSE: {mse_best:.4f}")
