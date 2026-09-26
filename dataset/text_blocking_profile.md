# Text and Blocking Profile: Amazon ML Challenge — Business Entity Resolution

This analysis investigates the textual properties of business names and addresses across the training datasets (`train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`) and the ground truth match structure (`train_ground_truth.tsv`). It provides empirical findings to guide text normalization, candidate retrieval (blocking), and downstream feature engineering under the locked architecture.

---

## 1. Business Name Analysis

### Character Length and Token Count Distributions
*Evaluated across all 12,527,040 training records across Sources 1, 2, and 3.*

| Source | Min Chars | Median Chars | Mean Chars | Max Chars | Min Tokens | Median Tokens | Mean Tokens | Max Tokens |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Source 1** | 3 | 24 | 24.03 | 105 | 1 | 4 | 3.55 | 16 |
| **Source 2** | 2 | 25 | 25.10 | 104 | 1 | 4 | 3.50 | 15 |
| **Source 3** | 2 | 25 | 25.20 | 123 | 1 | 4 | 3.53 | 18 |
| **Overall** | **2** | **25** | **24.96** | **123** | **1** | **4** | **3.52** | **18** |

### Duplicate Raw Business Names
- **Source 1:** 1,539,229 unique names among 2,206,821 records (30.25% duplicate reduction ratio). **845,385 records (38.31%)** share their raw name with at least one other record in Source 1.
- **Source 2:** 4,402,009 unique names among 5,034,616 records (12.57% duplicate reduction ratio). **872,386 records (17.33%)** share their raw name.
- **Source 3:** 4,651,609 unique names among 5,285,603 records (11.99% duplicate reduction ratio). **892,270 records (16.88%)** share their raw name.

### Top Common Business-Name Patterns
1. **English Legal & Entity Suffixes:**
   - `limited` (1,683,726 occurrences)
   - `private` (1,561,261 occurrences)
   - `llc` / `l.l.c` (1,538,653 occurrences)
   - `ltd` (955,399 occurrences)
   - `inc` (1,027,077 occurrences)
   - `pvt` (517,973 occurrences)
   - `corp` / `corporation` (428,222 occurrences)
   - `co` (306,623 occurrences)
   - `llp` (179,030 occurrences)
2. **Devanagari Transliterated Corporate Terms (Sources 2 & 3):**
   - `लिमिटेड` (*Limited*): 296,635 occurrences
   - `प्राइवेट` (*Private*): 246,472 occurrences
3. **Generic Domain and Industry Terms:**
   - `center` (443,309), `services` (338,112), `partners` (346,702), `group` (321,697), `care` (231,814), `associates` (186,950), `health` (127,783), `solutions` (105,420), `trading` (95,200).

### Concrete Noise Examples from Matched Pairs
- **Punctuation Differences:**
  - `Obsidian, LLC` vs. `Obsidian, [[LLC]]` *(brackets inserted around corporate suffix)*
  - `Summit Health LLC` vs. `Summit Health L.L.C.` *(acronym period variation)*
  - `Zander Blue Co` vs. `ZANDER BLUE [CO]` *(bracket enclosure of entity type)*
  - `Renee G. Garcia, O.D.` vs. `Renee G Garcia, OD` *(credential punctuation differences)*
  - `Vadodara Industries Pvt Ltd` vs. `VADODARA INDUSTRIES PVT (LTD)` *(parentheses around legal form)*
  - `Brookens, Barrie & Smith` vs. `Brookens, Barrie &  Smith` *(spacing around ampersands)*
- **Abbreviation / Legal Suffix Variations:**
  - `Diksha Technologies Private Limited` vs. `Diksha Technologies Private Ltd` (`Limited` vs. `Ltd`)
  - `Varela's Pizza` vs. `Varela's Pizza Inc.` *(suffix presence vs. absence)*
  - `Perfect Finance Pvt Ltd` vs. `Perfect Finance Pvt` *(omitted `Ltd`)*
  - `Shir Velasco Trusted Guggenheim` vs. `Shir Velasco Trusted Guggenheim Corporation` *(omitted vs. full form)*
  - `Caldeon Nova` vs. `>> caldeon nova` *(leading noise tokens `>>`)*
- **Word-Order Variations:**
  - `Hendricks and Flowers Inc` vs. `Hendricks and Inc Flowers` *(legal suffix inserted internally)*
  - `LRD Homecare Pvt Ltd` vs. `LRD Ltd Pvt Homecare` *(legal suffix transposed ahead of business category)*
  - `Mumbai Producer Clinic` vs. `CLINIC PRODUCER MUMBAI` *(full inverted token permutation)*
  - `Crystal Staffing Solutions LLC` vs. `Llc Crystal Staffing Solutions` *(suffix fronted)*
  - `AP Hospitality Inc` vs. `AP Inc Hospitality` *(transposed suffix token)*
- **Obvious Typos / Morphological Variations:**
  - `Velasquez, Ragland & Simmons Inc` vs. `VELASQUEZ, RAGLAND & SIMMON INC` *(dropped final letter 's')*
  - `Urban Bakery LLC` vs. `Urban Bakery LLC Center` *(spurious trailing descriptor)*
  - `33466 Warwick Hills Road` vs. `33466 Warwik Hills Road` *(internal character typo `ck` -> `k`)*
  - `Orellana Investments LLC` vs. `Orellana Investments Investments Llc` *(token stuttering/duplication)*

---

## 2. Business Address Analysis

### Missing-Address Percentage by Source
- **Source 1:** 0 missing (**0.00%**)
- **Source 2:** 168,967 missing (**3.36%**)
- **Source 3:** 175,916 missing (**3.33%**)

### Character Length and Token Count Distributions
*Evaluated across all non-empty address records.*

| Source | Min Chars | Median Chars | Mean Chars | Max Chars | Min Tokens | Median Tokens | Mean Tokens | Max Tokens |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Source 1** | 11 | 41 | 52.07 | 256 | 2 | 7 | 8.03 | 43 |
| **Source 2** | 8 | 37 | 47.83 | 249 | 2 | 6 | 7.54 | 46 |
| **Source 3** | 2 | 42 | 48.32 | 240 | 1 | 6 | 7.42 | 43 |
| **Overall** | **2** | **40** | **48.80** | **256** | **1** | **6** | **7.58** | **46** |

### Duplicate Raw Addresses
- **Source 1:** 2,130,606 unique addresses (3.45% duplicate reduction ratio; 5.27% of records share an address).
- **Source 2:** 4,337,261 unique addresses (10.86% duplicate reduction ratio; 19.52% of records share an address).
- **Source 3:** 4,632,764 unique addresses (9.33% duplicate reduction ratio; 16.83% of records share an address).

### Concrete Noise Examples from Matched Pairs
- **Abbreviation Variation:**
  - `3315 Fremont Street, Peoria, IL` vs. `3315 FREMONT ST, PEORIA, IL` (`Street` vs. `ST`)
  - `8706 Kentucky Derby Drive, Waxhaw, NC` vs. `8706 KENTUCKY DERBY DR, WAXHAW, NC` (`Drive` vs. `DR`)
  - `H No 847 S No 113/12 Opp Patil Enclave, Pune, Maharashtra` vs. `... Pune, MH` (`Maharashtra` vs. `MH`)
  - `5 Krishna Shopping Centre, Opp Dahisar Stn, P J Rd Dahisar E, Mumbai...` vs. `... P J RD ...` (`Rd` vs. `Road`, `Stn` vs. `Station`)
- **Missing Address Components:**
  - `Rajasthan, Jaipur, Banipark, Gokul Apartment, E-3A Kanti Chandra Road, G-1` vs. `G-1, BANIPARK, JAIPUR, Rajasthan` *(missing road name and apartment block)*
  - `D-4, Chandana Apartments82, Infantry Road, Bangalore, Karnataka` vs. `D-4, CHANDANA APARTMENTS82, INFANTRY ROAD, Karnataka` *(missing city `Bangalore`)*
  - `Wz-187C Shop No.13, 14 Kh. No.47 S/F. Vikaspuri Budhela Village Behind Oxford School, Delhi, West Delhi, Delhi` vs. `WZ-187C SHOP NO.13, DELHI, WEST DELHI, Delhi` *(missing village, landmark, and khasra number)*
- **Landmark-Based References:**
  - `Af-684, Nandgram Near Mother India Public School. Ph. 989, 9487203, Ghaziabad, Uttar Pradesh` *(landmark: `Near Mother India Public School`)*
  - `H No 847 S No 113/12 Opp Patil Enclave, Pune, Maharashtra` *(landmark: `Opp Patil Enclave`)*
  - `Lucknow, 3/77, Lucknow, Vipul Khand, Opp. Study Hall School Gomtinagar, Uttar Pradesh` *(landmark: `Opp. Study Hall School Gomtinagar`)*
  - `... Vikaspuri Budhela Village Behind Oxford School ...` *(landmark: `Behind Oxford School`)*
- **Reordered Address Components:**
  - `1 Greenbrier Drive, Kimberling City, MO` vs. `MO, KIMBERLING CITY, 1 GREENBRIER DRIVE` *(State, City, Street vs. Street, City, State)*
  - `Greensboro, 1604 Washington Street, NC` vs. `1604 WASHINGTON STREET, GREENSBORO, NC` *(City, Street, State vs. Street, City, State)*
  - `Lucknow, 3/77, Lucknow, Vipul Khand...` vs. `3/77, VIPUL KHAND, ..., LUCKNOW...` *(City fronted vs. City appended)*
- **Numeric Tokens (House Numbers, PIN/ZIP-like codes):**
  - US Street & Suite Numbers: `3315 Fremont Street`, `8706 Kentucky Derby Drive`, `20085 US 23`, `Unit APT 1049`
  - Indian Municipal / Plot / Survey / Khasra Numbers: `H No 847 S No 113/12`, `Wz-187C Shop No.13, 14 Kh. No.47 S/F`, `E-3A Kanti Chandra Road, G-1`, `No. 109, Ferns Meadows`
  - PIN codes and phone fragments: `9487203`, `560001`, `110018`

---

## 3. Normalized Text Preview

*The following deterministic samples illustrate proposed text normalization behavior. Normalization rules are not finalized.*

### Business Name Preview

| Sample # | Raw Business Name | Proposed Normalized Name | Normalization Operation |
| :--- | :--- | :--- | :--- |
| **1** | `Obsidian, [[LLC]]` | `obsidian llc` | Remove noise brackets, lowercase, normalize entity suffix |
| **2** | `Summit Health L.L.C.` | `summit health llc` | Remove period separators in acronyms, lowercase |
| **3** | `Diksha Technologies Private Limited` | `diksha technologies private limited` | Lowercase, whitespace standardization |
| **4** | `VADODARA INDUSTRIES PVT (LTD)` | `vadodara industries private limited` | Expand `pvt` -> `private`, `ltd` -> `limited`, strip parentheses |
| **5** | `>> caldeon nova` | `caldeon nova` | Strip leading symbol noise `>>`, lowercase |
| **6** | `[Corp] Dick Regional Armada` | `dick regional armada corporation` | Expand `corp` -> `corporation`, remove brackets |
| **7** | `Hendricks and  Flowers Inc` | `hendricks and flowers incorporated` | Collapse excess whitespace, expand `inc` -> `incorporated` |
| **8** | `Velasquez, Ragland & Simmons Inc` | `velasquez ragland and simmons incorporated` | Standardize `&` -> `and`, remove comma, expand suffix |

### Business Address Preview

| Sample # | Raw Business Address | Proposed Normalized Address | Normalization Operation |
| :--- | :--- | :--- | :--- |
| **1** | `3315 FREMONT ST, PEORIA, IL` | `3315 fremont street peoria il` | Lowercase, expand `st` -> `street`, remove commas |
| **2** | `8706 KENTUCKY DERBY DR, WAXHAW, NC` | `8706 kentucky derby drive waxhaw nc` | Lowercase, expand `dr` -> `drive`, remove commas |
| **3** | `MO, KIMBERLING CITY, 1 GREENBRIER DRIVE` | `1 greenbrier drive kimberling city mo` | Component reordering to canonical street-city-state format |
| **4** | `33466 WARWICK HILLS ROAD, <NULL>, YUCAIPA, CA` | `33466 warwick hills road yucaipa ca` | Strip `<NULL>` token artifact, lowercase |
| **5** | `H No 847 S No 113/12 Opp Patil Enclave, Pune, MH` | `h no 847 s no 113 12 opp patil enclave pune maharashtra` | Expand state `mh` -> `maharashtra`, clean punctuation |
| **6** | `108 Norle Street, College Twp, PA` | `108 norle street college township pennsylvania` | Expand `twp` -> `township`, expand state `pa` -> `pennsylvania` |

---

## 4. Ground-Truth Match Structure

*Measured empirically over 345,997 matched pairs across 94,406 Source 1 entities from `train_ground_truth.tsv`.*

| Metric | Measured Value | Analysis / Implication |
| :--- | :--- | :--- |
| **Exact Raw-Name Match %** | **4.60%** | Only 4.6% of true matching pairs share an identical raw name string. Exact name matching alone misses >95% of true matches. |
| **Exact Raw-Address Match %** | **2.24%** | Only 2.24% of true matching pairs share an identical raw address string. |
| **Exact Normalized-Name Match %** | **24.41%** | Baseline normalization improves name agreement by **5.3×** over raw matching. |
| **Exact Normalized-Address Match %** | **11.37%** | Baseline normalization improves address agreement by **5.1×** over raw matching. |
| **Same-Country %** | **100.00%** | **0.00%** cross-country matches observed among true pairs. Country compatibility is an absolute hard constraint. |
| **Missing-Address Rate Among True Matches** | **4.43%** | 4.43% of true matches have an empty address field in the matched Source 2 or Source 3 entity. |

---

## 5. False-Match Risk Analysis

Empirical identification of ambiguity patterns that risk creating false merges:

### 1. Repeated / Same Business Names Across Different Addresses (Multi-location Businesses)
- In Source 1 alone, **177,793 unique business names (11.55% of all unique names)** appear across multiple distinct addresses.
- Top occurrences of identical business names across distinct addresses in Source 1:
  - `Primary Care Group`: appears at **253 distinct addresses** across multiple states.
  - `Ear Nose & Throat Group`: appears at **251 distinct addresses**.
  - `Pediatric Group`: appears at **222 distinct addresses**.
  - `Womens Health Group`: appears at **220 distinct addresses**.
  - `Physical Therapy Group`: appears at **218 distinct addresses**.
  - `Pediatric Dental Group`: appears at **216 distinct addresses**.
  - `Behavioral Health Group`: appears at **215 distinct addresses**.
  - `Chiropractic Group`: appears at **209 distinct addresses**.
  - `Orthopedic Group`: appears at **208 distinct addresses**.
  - `Eye Group`: appears at **207 distinct addresses**.
- **Risk:** High name-similarity without strong address verification will trigger false merges across distinct branches or generic medical/corporate entities.

### 2. Repeated Addresses Hosting Different Business Names (Multi-Tenant Commercial Hubs)
- In Source 1 alone, **40,089 distinct addresses (1.88%)** host multiple distinct businesses.
- Examples of multi-tenant addresses hosting 12 to 14 distinct entities:
  - `108 Norle Street, College Twp, PA`: **14 distinct businesses** (e.g., `Chiropractic Express Care`, `GDP Prairie Companies`, `Harbor Guild Inc`).
  - `104 Roadrunner Circle, Elephant Butte, NM`: **13 distinct businesses** (e.g., `Precision Financial Industries`, `Kelly, Hall and Mendez, Inc.`, `Underhill Institute of Technology`).
  - `43 Franklin Street, Delaware, OH`: **13 distinct businesses**.
  - `55, Lavkush Vihar, Singhpur, ... Kanpur Nagar, Uttar Pradesh`: **13 distinct businesses**.
  - `No. 109, Ferns Meadows, Avalahalli Main Road, Bangalore, Karnataka`: **12 distinct businesses**.
- **Risk:** Relying on address similarity alone will produce false merges among co-located but completely independent businesses.

### 3. Common Names Likely to Create Hard Negatives
- Near-identical entity names differing by subtle modifiers:
  - `Sai Enterprises` vs. `Sri Sai Enterprises` vs. `Shree Sai Enterprises`
  - `Precision Payment Holdings` vs. `Precision Financial Industries`
  - `Reliance Digital` vs. `Reliance Retail` vs. `Reliance Trends`
- Shared generic tokens like `Center`, `Associates`, `Services`, `Holdings`, `India`, `Group`.

### 4. Cases Where Name Alone Is Insufficient
- Generic corporate titles (`Family Practice`, `Vision Center`, `Orthopedic Associates`) where identity is entirely defined by address.
- Missing address records (4.43% of true matches): when the address is missing in Source 2 or 3, models cannot evaluate address agreement and must rely on exact core-token name matches, country consistency, and lack of negative signals.

---

## 6. Memory and Scale Recommendations

*These findings are operational recommendations based on measured dataset scale (~12.5M train records, ~11.7M test records, ~1.73M test Source 1 queries).*

### Retrieval Branch Top-K (Recommendation)
- **Character TF-IDF Top-K:** Recommend **20 to 30**. Character n-grams (3-to-5-grams) effectively bridge spelling variations, typos, and Devanagari transliterations.
- **Word TF-IDF Top-K:** Recommend **15 to 20**. Word n-grams (unigrams and bigrams) retrieve exact token and multi-word sequence matches.
- **Embedding Retrieval Top-K:** Recommend **15 to 20**. Dense vector representations capture semantic similarities and word reordering.

### Maximum Candidate Count After Union (Recommendation)
- With 3 retrieval branches each nominating 15–30 candidates, the unified candidate set per Source 1 entity will typically comprise **25 to 45 unique candidates** (due to inter-branch candidate overlap).
- Setting a strict upper bound of **40 to 50 candidates per Source 1 entity** preserves high recall ceiling while capping total test inference pairs at ~60–85 million pairs.

### Chunk Size for Feature Generation (Recommendation)
- Recommend batching feature generation in chunks of **50,000 to 100,000 candidate pairs** (equivalent to ~1,500 to 2,500 Source 1 query entities).
- This maintains resident process RAM under **1.5 to 2.0 GB**, ensuring safe execution within available system memory.

### Separate vs. Combined Fitting of Character and Word TF-IDF (Recommendation)
- **Strongly recommend fitting separately.**
  1. **Vocabulary Specialization:** Character n-grams (sub-word units) and word n-grams (lexical tokens) have fundamentally different frequency dynamics and optimal pruning thresholds (`min_df`, `max_features`).
  2. **Memory Footprint:** Fitting character n-grams independently allows constraining character vocabulary size (e.g., 50,000 to 100,000 features) without multiplying word token combinations.
  3. **Distinct Feature Signals:** Keeping the retrieval branches separate preserves distinct character vs. word cosine similarity scores to feed into downstream LightGBM features.
