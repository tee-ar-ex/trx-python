# tractography_file_format
Official repository to present specifications, showcase examples, discuss issues and keep track of everything.
Currently host only two file format propositions, the TRX file format (as memmap vs zarr). Anyone is free to contribute.

Try it out with:
```bash
pip install -e .
```

```python
import numpy as np  
from trx_file_memmap import load, save, TrxFile

trx = load('complete_big_v4.trx')

# Access the header (dict) / streamlines (ArraySequences)
trx.header
trx.streamlines

# Access the dpp (dict) / dps (dict)
trx.data_per_point
trx.data_per_streamlines

# Access the groups (dict) / dpg (dict)
trx.groups
trx.data_per_group

# Get a random subset of 10000 streamlines
indices = np.arange(len(trx.streamlines._lengths))
np.random.shuffle(indices)
sub_trx = trx.select(indices[0:10000])
save(sub_trx, 'random_1000.trx')

# Get sub-groups only, from the random subset
for key in sub_trx.groups.keys():
	group_trx = sub_trx.get_group(key) 
    save(group_trx, '{}.trx'.format(key)) 

# Pre-allocate memmaps and append 100x the random subset
alloc_trx = TrxFile(nb_streamlines=1500000, nb_points=500000000, init_as=trx)
for i in range(100):
    alloc_trx.append(sub_trx)

# Resize to remove the unused portion of the memmap
alloc_trx.resize()
```
