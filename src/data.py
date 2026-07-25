import numpy as np

def load_dataset(name="nsl"):
    name = name.lower()
    if name == "nsl":
        Xtr = np.load('data/processed/nsl_X_train.npy')
        ytr = np.load('data/processed/nsl_y_train.npy')
        Xte = np.load('data/processed/nsl_X_test.npy')
        yte = np.load('data/processed/nsl_y_test.npy')

        # --- Debug prints for NSL ---
        print("NSL y_train unique:", np.unique(ytr))
        print("NSL y_test unique:", np.unique(yte))
        print("NSL X_train shape:", Xtr.shape)
        print("NSL X_test shape:", Xte.shape)
        print("Sample y_train:", ytr[:20])

    elif name == "cic":
        Xtr = np.load('data/processed/cic_X_train.npy')
        ytr = np.load('data/processed/cic_y_train.npy')
        Xte = np.load('data/processed/cic_X_test.npy')
        yte = np.load('data/processed/cic_y_test.npy')

        # --- Debug prints for CIC ---
        print("CIC y_train unique:", np.unique(ytr))
        print("CIC y_test unique:", np.unique(yte))
        print("CIC X_train shape:", Xtr.shape)
        print("CIC X_test shape:", Xte.shape)
        print("Sample y_train:", ytr[:20])

    else:
        raise ValueError("Use dataset='nsl' or 'cic'")

    return Xtr, ytr, Xte, yte

if __name__ == "__main__":
    # Test loading both datasets
    print("Loading NSL dataset...")
    load_dataset("nsl")
    print("\nLoading CIC dataset...")
    load_dataset("cic") 