# Data

Raw datasets are not included in this repository because of applicable
source-platform and third-party data-use restrictions.

## Datasets used in the study

1. **Booking.com source dataset**
   - 2,011 hotel reviews retained after preprocessing.
   - Used for weak-label generation, model development, and internal evaluation.
   - The collected review records are not redistributed here.

2. **Ott et al. benchmark**
   - 1,600 reviews with independently defined deceptive/truthful labels.
   - Used as an independent benchmark.
   - Obtain the dataset from its original publication/research resources.

3. **External hotel-review collection**
   - 5,823 original reviews.
   - 4,842 retained after preprocessing.
   - Used only for cross-domain behaviour analysis.
   - Verified suspicious/normal labels were unavailable, so it is not treated as supervised external validation.

Prepared input files should follow the column requirements documented in the corresponding scripts.
