# Try-On Preprocessing

Before you start, make a .env file in your project's main folder. Put these environment variables inside it.
```
U2NET_CLOTH_SEG_CHECKPOINT_PATH=cloth_segm.pth
```

#### Remember to load environment variables before you start running scripts.

```
from dotenv import load_dotenv

load_dotenv()
```

### segment garment

```
from tryon.preprocessing import segment_garment

segment_garment(inputs_dir=<inputs_dir>,
               outputs_dir=<outputs_dir>, cls=<cls>)
```

possible values for cls: lower, upper, all

### extract garment

```
from tryon.preprocessing import extract_garment

extract_garment(inputs_dir=<inputs_dir>,
               outputs_dir=<outputs_dir>, cls=<cls>)
```