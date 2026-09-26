# Dataset Profile: Amazon ML Challenge — Business Entity Resolution

This document contains profiling findings for the training and test datasets.

---

## 1. Names of All Train/Test TSV Files

### Training Files (`dataset/train/`)
- `dataset/train/train_ground_truth.tsv`
- `dataset/train/train_source1.tsv`
- `dataset/train/train_source2.tsv`
- `dataset/train/train_source3.tsv`

### Test Files (`dataset/test/`)
- `dataset/test/test_source1.tsv`
- `dataset/test/test_source2.tsv`
- `dataset/test/test_source3.tsv`

---

## 2. Number of Rows in Each File

| Split | File Name | Header Lines | Data Rows | Total Lines |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | `train_ground_truth.tsv` | 1 | 2,206,821 | 2,206,822 |
| **Train** | `train_source1.tsv` | 1 | 2,206,821 | 2,206,822 |
| **Train** | `train_source2.tsv` | 1 | 5,034,616 | 5,034,617 |
| **Train** | `train_source3.tsv` | 1 | 5,285,603 | 5,285,604 |
| **Test** | `test_source1.tsv` | 1 | 1,732,544 | 1,732,545 |
| **Test** | `test_source2.tsv` | 1 | 4,887,273 | 4,887,274 |
| **Test** | `test_source3.tsv` | 1 | 5,082,316 | 5,082,317 |
| **Total** | *All 7 TSV files* | **7** | **26,435,994** | **26,436,001** |

---

## 3. Column Names

### Source Files (`*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`)
Each source file contains the following **4 columns**:
1. `entity_id`
2. `business_name`
3. `business_address`
4. `country`

### Ground Truth File (`train_ground_truth.tsv`)
Contains the following **2 columns**:
1. `source1_entity_id`
2. `matched_entity_ids`

---

## 4. Missing-Value Counts for `business_name`, `business_address`, `country`

| Split | File Name | Total Rows | Missing `business_name` | Missing `business_address` | Missing `country` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | `train_source1.tsv` | 2,206,821 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| **Train** | `train_source2.tsv` | 5,034,616 | 0 (0.00%) | 168,967 (3.36%) | 0 (0.00%) |
| **Train** | `train_source3.tsv` | 5,285,603 | 0 (0.00%) | 175,916 (3.33%) | 0 (0.00%) |
| **Test** | `test_source1.tsv` | 1,732,544 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| **Test** | `test_source2.tsv` | 4,887,273 | 0 (0.00%) | 129,408 (2.65%) | 0 (0.00%) |
| **Test** | `test_source3.tsv` | 5,082,316 | 0 (0.00%) | 136,098 (2.68%) | 0 (0.00%) |
| **Total** | *All Source Files* | **24,229,173** | **0 (0.00%)** | **610,389 (2.52%)** | **0 (0.00%)** |

*Note: All missing address entries are empty tab-separated fields (`""`). There are no whitespace-only or NaN token entries.*

---

## 5. Number of Unique Entity IDs

### Per-File Unique Entity IDs
- `train_source1.tsv`: **2,206,821** unique IDs (0 duplicates)
- `train_source2.tsv`: **5,034,616** unique IDs (0 duplicates)
- `train_source3.tsv`: **5,285,603** unique IDs (0 duplicates)
- `test_source1.tsv`: **1,732,544** unique IDs (0 duplicates)
- `test_source2.tsv`: **4,887,273** unique IDs (0 duplicates)
- `test_source3.tsv`: **5,082,316** unique IDs (0 duplicates)

### Aggregate Unique Entity IDs
- **Training Source Files Total:** **12,527,040** unique IDs
- **Test Source Files Total:** **11,702,133** unique IDs
- **Overall Total Unique Entity IDs:** **24,229,173** unique IDs across all source files

*Note: All entity IDs are mutually exclusive across splits and sources (zero overlap between train and test, zero duplicate IDs within any file).*

---

## 6. Country Distribution

### Training Set Distribution

| Country | `train_source1.tsv` | `train_source2.tsv` | `train_source3.tsv` | Total Train Records | Percentage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **US** | 1,323,633 (59.98%) | 3,016,817 (59.92%) | 3,170,056 (59.97%) | 7,510,506 | 59.95% |
| **India** | 883,188 (40.02%) | 2,017,799 (40.08%) | 2,115,547 (40.03%) | 5,016,534 | 40.05% |
| **Total** | **2,206,821** | **5,034,616** | **5,285,603** | **12,527,040** | **100.00%** |

### Test Set Distribution

| Country | `test_source1.tsv` | `test_source2.tsv` | `test_source3.tsv` | Total Test Records | Percentage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **India** | 809,986 (46.75%) | 2,312,565 (47.32%) | 2,405,000 (47.32%) | 5,527,551 | 47.24% |
| **US** | 663,106 (38.27%) | 1,871,330 (38.29%) | 1,945,701 (38.28%) | 4,480,137 | 38.28% |
| **France** | 259,452 (14.98%) | 703,378 (14.39%) | 731,615 (14.39%) | 1,694,445 | 14.48% |
| **Total** | **1,732,544** | **4,887,273** | **5,082,316** | **11,702,133** | **100.00%** |

*Note: The test set introduces `France` (14.48% of test records), which does not appear in the training set.*

---

## 7. Analysis of `train_ground_truth.tsv`

- **Number of S1 entities:** **2,206,821** (matches 100% of entities in `train_source1.tsv` with 0 duplicate S1 rows)
- **Number with zero matches (singletons):** **123,247** (5.59%)
- **Number with one match:** **119,157** (5.40%)
- **Number with multiple matches:** **1,964,417** (89.01%)
- **Average number of matches per S1 entity:** **3.4613** (7,638,365 total matches / 2,206,821 S1 entities)

### Breakdown of Match Counts per S1 Entity

| Number of Matches | Number of S1 Entities | Percentage |
| :--- | :--- | :--- |
| **0 matches** | 123,247 | 5.59% |
| **1 match** | 119,157 | 5.40% |
| **2 matches** | 375,212 | 17.00% |
| **3 matches** | 530,841 | 24.05% |
| **4 matches** | 484,115 | 21.94% |
| **5 matches** | 321,957 | 14.59% |
| **6 matches** | 164,868 | 7.47% |
| **7 matches** | 63,968 | 2.90% |
| **8 matches** | 18,680 | 0.85% |
| **9 matches** | 4,205 | 0.19% |
| **10 matches** | 534 | 0.02% |
| **11 matches** | 37 | 0.00% |
| **Total S1 Entities** | **2,206,821** | **100.00%** |

### Matched Entities Source Breakdown
- **Source 2 matched references:** 3,693,619 (48.36%)
- **Source 3 matched references:** 3,944,746 (51.64%)
- **Total matched entity pairs:** 7,638,365
