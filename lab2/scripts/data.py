import numpy as np

# load
data = np.load("asset/demo.npz", allow_pickle=True)

# print keys
print("Keys in file:", data.files)


for k in data.files:
    v = data[k]
    print(f"Key: {k}")
    print(f"  Type: {type(v)}")
    print(v)
    
    if isinstance(v, np.ndarray):
        print(f"  Shape: {v.shape}")
        print(f"  Dtype: {v.dtype}")
        
        # object arrays (episode lists)
        if v.dtype == object:
            print(f"  Number of episodes: {len(v)}")
            for i, item in enumerate(v):  # show first few
                print(f"    Episode {i}: shape {np.array(item).shape}")
        else:
            print(f"  Value:\n{v}")
    else:
        print(f"  Value: {v}")
    
    print("-" * 50)
