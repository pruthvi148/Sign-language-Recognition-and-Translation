import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import argparse
import os
import json
from .dataset import SignLanguageDataset
from .bilstm import SignLanguageModel

def train_model(data_dir, epochs=50, batch_size=32, lr=0.001, save_dir='models', min_samples=5):
    os.makedirs(save_dir, exist_ok=True)
    
    full_dataset = SignLanguageDataset(data_dir, min_samples=min_samples)
    num_classes = len(full_dataset.classes)
    print(f"Loaded {len(full_dataset)} sequences across {num_classes} classes (min {min_samples} samples each).")
    print(f"Classes: {full_dataset.classes}")
    
    if len(full_dataset) == 0:
        print("No data found!")
        return
        
    with open(os.path.join(save_dir, 'classes.json'), 'w') as f:
        json.dump(full_dataset.classes, f)
        
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_size > 0 else None
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SignLanguageModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
        train_acc = 100. * train_correct / total if total > 0 else 0
        
        val_loss = 0.0
        val_acc = 0.0
        
        if val_loader:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            val_loss /= len(val_loader)
            val_acc = 100. * val_correct / val_total if val_total > 0 else 0
            scheduler.step(val_loss)
        else:
            val_loss = train_loss / len(train_loader)
            
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} - LR: {current_lr:.6f} | Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.1f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.1f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f"  -> Saved best model!")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='processed_data')
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--min_samples', type=int, default=5)
    args = parser.parse_args()
    train_model(args.data_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, save_dir=args.save_dir, min_samples=args.min_samples)
