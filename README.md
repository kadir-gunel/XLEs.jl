# XLEs.jl


Algorithms for bilingual lingual embedding alignments. This is the direct re-implementation of [vecmap](https://github.com/artetxem/vecmap) which was written in python. Right now, the unsupervised bilingual alignment is implemented. The supervsied version is pretty simple once you have a training data set. Just use the ```maporthogonal``` function. This will rotate the source space towards the target space; but also use a test set just to be sure. You can download datasets from vecmap repository. You will also need word embeddings, either train your own embeddings or just download from different sources like GloVe repo, MUSE or vecmap (again).

Be aware that these algorithms do not align different representations that are trained with different models. What does it mean? It simply means that unsupervised alignment algorithms do not guarantee you to align, for instance, GloVe and FastText embeddings. It can but it may not. 

I wrote a complete chapter in my phd thesis about this phenomenon. And there seems to be no silver bullet which can rule them all. Sorry. 



### Note to myself
1. share a main script.
2. share also optimal transport approach.



## Why reimplementation ? 
The original code is pretty hard to read this prevents the understanding of the rotation matrix concept and how it is applied to word embeddings.

## Does it work on GPU ?
Yes, it actually requires to use a gpu. **No CPU implementation**

