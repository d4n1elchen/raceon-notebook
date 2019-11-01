#%%
import matplotlib.pyplot as plt
import numpy as np

#%%
scan_hist = np.load('scan_hist.npy')
plt.imshow(np.flip(scan_hist, axis=0), extent=[0,300,0,600])
plt.show()

# %%
