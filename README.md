# O-value:Outperformance-Standardization-for-Classification-Metrics
OPS (Outperformance Standardization) is a universal standardization method for **confusion-matrix-based classification performance (CMBCP)** metrics. It maps any supported metric to a common **[0, 1]** scale with a consistent interpretation across **different class imbalance rates**.

The resulting **OPS value (“o-value”)** represents the **percentile rank** of the observed classification performance within a **reference distribution of possible performances** under the same test-set class imbalance structure. This makes it easier to **evaluate, compare, and monitor** classification performance when the imbalance rate varies across test sets.
