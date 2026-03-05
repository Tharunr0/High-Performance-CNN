import torch
import torch.nn as nn
import torch.optim as optim
from src import config, dataset, model, train

def main():
    print(f"Project: High-Performance CNN Vision System")
    print(f"Device: {config.DEVICE}")

    train_loader, val_loader, class_names = dataset.get_data_loaders()
    print(f"Classes: {class_names}")
  
    cnn_model = model.build_model()
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Optimization
    optimizer = optim.AdamW(cnn_model.parameters(), lr=config.LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler() # For Mixed Precision
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=3, verbose=True
    )

    best_acc = 0.0
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        
        train_loss, train_acc = train.train_one_epoch(
            cnn_model, train_loader, optimizer, criterion, scaler
        )
        val_loss, val_acc = train.validate(cnn_model, val_loader, criterion)
        
        scheduler.step(val_acc)
        
        print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
 
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(cnn_model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"--> Saved Best Model (Acc: {best_acc:.2f}%)")

if __name__ == "__main__":
    main()