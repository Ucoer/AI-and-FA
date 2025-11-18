import pandas as pd
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from pathlib import Path
import time

# -------- CONFIG --------
DATA_ROOT   = Path(r"/home/uctzyro/Scratch/data/nii_zipped_cases")
CSV_INFO    = Path(r"/home/uctzyro/Scratch/data/case_list_complete.xlsx")
NUM_EPOCHS  = 3 #for separated age groups, using epochs = 10
BATCH_SIZE  = 16
LR          = 1e-4
N_FOLDS     = 5
TARGET_COL  = 'age'
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')
NUM_WORKERS = 4           
#shape
TARGET_SHAPE = (64, 64, 64)

class CT3DDataset(Dataset):
    def __init__(self, df, data_root):
        self.df = df.reset_index(drop=True)
        self.root = data_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        case = row['case_folder']
        paths = list((self.root / case).glob('*.nii*'))
        assert len(paths) == 1, f"Expected 1 NIfTI in {case}, found {len(paths)}"
        
        try:
            img = nib.load(str(paths[0]))
            vol = img.get_fdata(dtype = np.float32)
        except Exception as e:
            print(f'fail to load {case} ({paths[0]}): {e}')
            
            # return a zero-volume + dummy label so DataLoader continues
            vol = np.zeros((1, *TARGET_SHAPE), dtype=np.float32)
            label = torch.tensor(0., dtype=torch.float32)
            return torch.from_numpy(vol), label
        
        #resample to TARGET_SHAPE
        factors = [t / float(o) for t, o in zip(TARGET_SHAPE, vol.shape)]
        vol = zoom(vol, factors, order=1)
        #normalize
        vol = (vol - vol.mean()) / (vol.std() + 1e-6)
        #add channel
        vol = np.expand_dims(vol, axis=0)

        tensor = torch.from_numpy(vol)
        label = torch.tensor(float(row[TARGET_COL]), dtype=torch.float32)
        return tensor, label

#CNN conv+ReLU+Pool
class Simple3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool3d(1)
        )
        self.regressor = nn.Linear(64, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x).squeeze(1)


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            print(f'predictions: {out}')
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def main():
    df = pd.read_excel(CSV_INFO, dtype=str)
    
    #ingore corrupted files
    valid_cases = {d.name for d in DATA_ROOT.iterdir() if d.is_dir()}
    df = df[df['case_folder'].isin(valid_cases)].reset_index(drop=True)
    
    #detect age column
    cols = df.columns.tolist()
    age_cols = [c for c in cols if 'age' in c.lower()]
    if not age_cols:
        raise KeyError(f"No 'age' column in {CSV_INFO}: {cols}")
    if age_cols[0] != 'age':
        df.rename(columns={age_cols[0]: 'age'}, inplace=True)
    df['age'] = df['age'].astype(float)

    #splitting data
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(train_df), 1):
        print(f"*** Fold {fold}/{N_FOLDS} ***")
        train_ds = CT3DDataset(train_df.iloc[tr_idx], DATA_ROOT)
        val_ds   = CT3DDataset(train_df.iloc[val_idx], DATA_ROOT)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                   shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                                   shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        model = Simple3DCNN().to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        best_val = float('inf')
        
        for epoch in range(1, NUM_EPOCHS+1):
            start_time = time.time()
            
            tr_loss = train_epoch(model, train_loader, criterion, optimizer)
            val_loss = eval_epoch(model, val_loader, criterion)
            
            epoch_time = time.time() - start_time
            #print time for each epoch
            print(f"Epoch {epoch}: train_loss={tr_loss:.4f}, val_loss={val_loss:.4f}, time={epoch_time/60:.2f} min", flush=True)
            
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), f"best_model_fold{fold}.pt")

    #final test
    test_ds = CT3DDataset(test_df, DATA_ROOT)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    model = Simple3DCNN().to(DEVICE)
    model.load_state_dict(torch.load(f"best_model_fold{N_FOLDS}.pt"))
    test_loss = eval_epoch(model, test_loader, criterion)
    print(f"\nTest MSE: {test_loss:.4f}")
    
    # After model evaluation
    predictions = []
    actuals = []

    # Iterate through the test set to gather predictions and actual values
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            out = model(x)
        predictions.extend(out.cpu().numpy())  # Store predicted values
        actuals.extend(y.cpu().numpy())  # Store actual values

    # Save predictions and actual values into a DataFrame
    results_df = pd.DataFrame({'Predictions': predictions, 'Actuals': actuals})

    # Save the DataFrame to an Excel file
    results_df.to_excel('cnn_predictions.xlsx', index=False)  # Saves predictions to an Excel file
    print("Predictions saved to 'cnn_predictions.xlsx'")


if __name__ == "__main__":
    main()


