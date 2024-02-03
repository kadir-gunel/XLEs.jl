# XLEs.jl


Algorithms for bilingual lingual embedding alignments. This is the direct re-implementation of [vecmap](https://github.com/artetxem/vecmap) which was written in python. Right now, the unsupervised bilingual alignment is implemented. The supervsied version is pretty simple once you have a training data set. Just use the ```maporthogonal``` function. This will rotate the source space towards the target space; but also use a test set just to be sure.  

### Note to myself
1. share a main script.
2. share also optimal transport approach.



## Why reimplementation ? 
The original code is pretty hard to read this prevents the understanding of the rotation matrix concept and how it is applied to word embeddings.

## Does it work on GPU ?
Yes, it actually requires to use a gpu. **No CPU implementation**

