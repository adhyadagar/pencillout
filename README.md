# Python environment

You will need to install RDKit, pandas, numpy, and pytorch using conda.

# Data Processing

`natural_products.txt` currently contains a comma separated list of activities for each bioactivity. This will need to be converted out into a one-hot encoding for each bioactivity. For example:

```
drug	bioactivity
drug_1	antibacterial,antifungal
```

would have to be separated out into:

```
drug	antibacterial	antifungal	antiviral
drug_1	1	1	0
```

The activities we can consider for now are antibacterial, antifungal, and antiviral.
