# Semantic spaces module

This is a python module that allows to compute semantic metrics based on
distributional semantics models.

For example, to find words that are semantically similar to the word 'brain':

```python
from semspaces.space import SemanticSpace

space = SemanticSpace.from_csv('space.w2v.gz')

space.most_similar(['brain'])

{'brain': [(u'brain', 0.0),
  (u'brains', 0.34469844325620635),
  (u'cerebrum', 0.4426992023455152),
  (u'cerebellum', 0.4483798859566903),
  (u'cortical', 0.469348588934828),
  (u'brainstem', 0.4791188497952641),
  (u'cortex', 0.479544888313173),
  (u'ganglion', 0.49717579235842546),
  (u'thalamus', 0.5030885466349713),
  (u'thalamic', 0.5059524199702277)]}
```

The module wraps dense and sparse matrix implementations to provide convenience
methods for computing semantic statistics as well as easy input and output of
the data.

# Installation

```bash
pip install -r requirements.txt
python setup.py install
```

# Semantic spaces

You can download a set of validated semantic spaces for English and Dutch
[here](http://zipf.ugent.be/snaut/spaces/) (see Mandera, Keuleers, & Brysbaert,
in press). 

# Contribute 

- Issue Tracker: https://github.com/pmandera/semspaces/issues
- Source Code: https://github.com/pmandera/semspaces

# Authors

The tool was developed at Center for Reading Research, Ghent University by
[Paweł Mandera](http://crr.ugent.be/pawel-mandera).

# License

The project is licensed under the Apache License 2.0.

# References

Mandera, P., Keuleers, E., & Brysbaert, M. (in press). Explaining human
performance in psycholinguistic tasks with models of semantic similarity based
on prediction and counting: A review and empirical validation. *Journal of
Memory and Language*.
