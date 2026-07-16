# Customer Segmentation Report

- **Total customers analyzed:** 5881
- **Number of segments (k):** 4

## Cluster Profiles

Values below are per-cluster feature means.

| Cluster | Segment Name | Recency | Frequency | Monetary | Average Order Value | Customer Lifetime Days | Purchase Frequency | Basket Size | Size | % of Base |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Loyal Customers | 92.11 | 8.40 | 3,399.55 | 29.46 | 424.31 | 0.02 | 226.60 | 3623 | 61.61% |
| 1 | At Risk / Lost | 380.25 | 1.51 | 538.74 | 52.18 | 23.67 | 0.83 | 203.56 | 2234 | 37.99% |
| 2 | At Risk / Lost (2) | 357.00 | 2.67 | 71,482.87 | 18,987.57 | 98.67 | 0.35 | 57,261.83 | 3 | 0.05% |
| 3 | Champions | 24.14 | 149.86 | 173,345.06 | 166.60 | 685.24 | 0.22 | 961.47 | 21 | 0.36% |

## Segment Descriptions

**Cluster 0 — Loyal Customers:** 
3623 customers (61.61% of base). Avg recency 92 days, avg frequency 8.4 orders, avg monetary value 3,399.55.

**Cluster 1 — At Risk / Lost:** 
2234 customers (37.99% of base). Avg recency 380 days, avg frequency 1.5 orders, avg monetary value 538.74.

**Cluster 2 — At Risk / Lost (2):** 
3 customers (0.05% of base). Avg recency 357 days, avg frequency 2.7 orders, avg monetary value 71,482.87.

**Cluster 3 — Champions:** 
21 customers (0.36% of base). Avg recency 24 days, avg frequency 149.9 orders, avg monetary value 173,345.06.

## Evaluation Metrics (Best Algorithm)

- **Algorithm:** KMeans
- **Silhouette Score:** 0.4620
- **Davies-Bouldin Index:** 0.9273
- **Calinski-Harabasz Index:** 2282.10
