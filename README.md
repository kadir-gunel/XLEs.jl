# XLEs.jl


Algorithms for bilingual lingual embedding alignments. This is a direct re-implementation of [vecmap](https://github.com/artetxem/vecmap), originally written in Python. Currently, the unsupervised bilingual alignment is implemented. The supervised version is relatively simple once you have a training dataset. Just use the ```maporthogonal``` function. This will rotate the source space towards the target space, but it's advisable to use a test set for verification. You can download datasets from the vecmap repository. You will also need word embeddings; you can either train your own embeddings or download them from various sources such as the GloVe repository, MUSE, or vecmap (again).

Please note that these algorithms do not align different representations trained with different models. What does this mean? Simply put, unsupervised alignment algorithms do not guarantee alignment between, for instance, GloVe and FastText embeddings. It can happen, but it's not guaranteed.

I dedicated a whole chapter in my PhD thesis to this phenomenon of cross-model alignment, and it appears that there is no one-size-fits-all solution; there's no silver bullet that can address all scenarios.


## Main scripts
Please go to [XLEScripts](https://github.com/kadir-gunel/XLEscripts), there are various scripts for word alignment. Additionally, apart from the orthogonal approach there is one interesting approach for alignment which is based on optimal transport (BLOT.jl). You can find more information on the XLEScript repo. 


## Why reimplementation ? 
The original code is pretty hard to read, hindering the understanding of the rotation matrix concept and how it is applied to word embeddings.

## Does it work on GPU ?
Yes, it actually requires to use a gpu. **No CPU implementation**

