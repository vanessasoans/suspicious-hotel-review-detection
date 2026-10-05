# Data

The datasets used in this study are described below. Raw review records
are not included in this repository because of applicable source-platform
and third-party data-use restrictions.

## 1. Booking.com source dataset

- **File used in the study:** `booking_reviews_traindataset.csv`
- **Reviews retained:** 2,011
- **Source:** Booking.com
- **Use:** Primary model development, weak-label generation, and internal
  five-fold evaluation.
- **Availability:** The collected Booking.com review records are not
  redistributed in this repository because of applicable source-platform
  and data-use restrictions.

The dataset contains hotel information and review-level fields including
hotel name, hotel URL, reviewer information, review score, review title,
positive and negative review text, review date, stay date, room type,
and traveller type.

## 2. Ott et al. deceptive-opinion benchmark

- **File used in the study:** `deceptive-opinion(1).csv`
- **Reviews:** 1,600
- **Labels:** Independently defined deceptive and truthful labels
- **Source:** Ott et al.
- **Use:** Independent benchmark evaluation.

The benchmark was introduced in:

Ott, M., Choi, Y., Cardie, C., & Hancock, J. T. (2011).
Finding deceptive opinion spam by any stretch of the imagination.
Proceedings of the 49th Annual Meeting of the Association for
Computational Linguistics, 309--319.

Official publication and dataset resources:

https://aclanthology.org/P11-1032/

The Ott benchmark was not used to generate the weak labels for the
Booking.com dataset.

## 3. External hotel-review collection

- **File used in the study:** `reviews_cleaned(4).csv`
- **Original reviews:** 5,823
- **Reviews retained after preprocessing:** 4,842
- **Source:** External hotel-review collection
- **Use:** Cross-domain behaviour analysis only
- **Verified labels:** Not available

The external collection contains hotel reviews with TripAdvisor-based
hotel URLs and was not used for model training, feature selection,
parameter tuning, or model selection.

Because verified suspicious/normal labels were unavailable, predictions
on this collection are treated as model outputs for cross-domain
behaviour analysis rather than supervised external validation.

The raw external review records are not redistributed in this repository
because of applicable third-party data-use restrictions.

## Reproducibility

The repository provides the source code, aggregate experimental results,
and selected figures required to document the experimental methodology.
Reproduction of the complete experiments requires access to the
underlying datasets and the required pretrained Sentence Transformer
model.
