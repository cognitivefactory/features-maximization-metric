# Usage

## Basic usecase: "_What are the physical characteristics that most distinguish men from women?_"

1. Load Python dependencies:
```python
###
### Python dependencies.
###

from cognitivefactory.feature_maximization_metric.fmc import FeaturesMaximizationMetric
from scipy.sparse import csr_matrix
from typing import List
```

1. Define problem data:
```python
###
### Data.
###

# Define people characteristics that will be studied.
characteristics_studied: List[str] = [
	"Shoes size",
	"Hair size",
	"Nose size",
]

# Get people characteristics.
people_characteristics: csr_matrix = csr_matrix(
	[
		[9, 5, 5],
		[9, 10, 5],
		[9, 20, 6],
		[5, 15, 5],
		[6, 25, 6],
		[5, 25, 5],
	]
)

# Get people genders.
people_genders: List[str] = [
	"Man",
	"Man",
	"Man",
	"Woman",
	"Woman",
	"Woman",
]
```

1. Modelize the problem:
```python
###
### Feature Maximization Metrics.
###

# Main computation.
fmc_computer: FeaturesMaximizationMetric = FeaturesMaximizationMetric(
	data_vectors=people_characteristics,
	data_classes=people_genders,
	list_of_possible_features=characteristics_studied,
	amplification_factor=1,
)
```

1. Determine relevant characteristics:
```python
###
### Analysis 1: Delete characteristics that aren't relevant.
###

print(
	"\n",
	"1. Which characteristic seems not relevant to distinguish men from women ?",
)
for characteristic in characteristics_studied:
	if not fmc_computer.features_selection[characteristic]:
		print(
			"    - '{0}' seems not relevant.".format(characteristic)
		)
```
```output
1. Which characteristic seems not relevant to distinguish men from women ?
    - 'Nose size' seems not relevant.
```

1. Describe gender by relevant characteristics.:
```python
###
### Analysis 2: Describe gender by relevant characteristics.
###

print(
	"\n",
	"2. According to remaining characteristics:",
)
for gender in sorted(set(people_genders)):
	print(
		"    - Which characteristic seems important to recognize a '{0}' ?".format(gender)
	)

	for characteristic in fmc_computer.get_most_active_features_by_a_classe(
		classe=gender,
	):
		print(
			"        - '{0}' seems important (fmeasure of '{1:.2f}', contrast of '{2:.2f}').".format(
				characteristic,
				fmc_computer.features_fmeasure[characteristic][gender],
				fmc_computer.features_contrast[characteristic][gender],
			)
		)
```
```output
2. According to remaining characteristics:
    - Which characteristic seems important to recognize a 'Man' ?
        - 'Shoes size' seems important (fmeasure of '0.45', contrast of '1.32').
    - Which characteristic seems important to recognize a 'Woman' ?
        - 'Hair size' seems important (fmeasure of '0.66', contrast of '1.25').
```
