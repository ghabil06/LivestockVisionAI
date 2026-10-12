# Batch B3: Indian Cattle Dataset Suitability & Audit Report

- **Audited Archive:** `Annotated Multiview Image Dataset of Prominent Ind.zip`
- **Total Scanned Images:** 7226 images
- **Unique Biological Images:** 4975 images
- **Exact Binary Duplicates:** 2251 images (31.2% duplication rate)

---

## 1. Binary Deduplication Audit by Breed

| Breed Category | Total Files | Unique Images | Duplicate Files | Duplication Rate (%) | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Bargur** | 470 | 470 | 0 | 0.0% | Clean |
| **Gir** | 3648 | 2443 | 1205 | 33.0% | Moderate Duplication ⚠️ |
| **Hallikar** | 1951 | 937 | 1014 | 52.0% | Severe Duplication ⚠️ |
| **Ongole** | 343 | 311 | 32 | 9.3% | Moderate Duplication ⚠️ |
| **Tharparkar** | 814 | 814 | 0 | 0.0% | Clean |
| **TOTAL** | **7226** | **4975** | **2251** | **31.2%** | **Mandatory Deduplication** |

### Audit Analysis:
1. **Hallikar:** Over **52.0%** of all Hallikar images (1,014 out of 1,951) are exact byte-for-byte duplicates.
2. **Gir:** Over **33.0%** of all Gir images (1,205 out of 3,648) are exact byte-for-byte duplicates.
3. **Contamination Hazard:** Partitioning this dataset at random without hashing causes massive train-test contamination.

---

## 2. Modality & Annotation Verification

| Modality / Attribute | Status in Dataset | Verdict |
| :--- | :--- | :--- |
| **Scale Weight Measurements** | COMPLETELY_ABSENT (0%) | **Strictly Unusable for Weight Estimation** |
| **Dentition / Age Records** | COMPLETELY_ABSENT (0%) | **Strictly Unusable for Age Estimation** |
| **Individual Animal IDs** | UNTRACKED (Sequential filenames) | **Unusable for Multi-View Tracking** |
| **Metadata & EXIF** | STRIPPED (All uniform 800x600 pixels) | Stripped by preprocessing |

---

## 3. Representation of Native Tamil Nadu Breeds

| Tamil Nadu Native Breed | Presence in Archive | Unique Image Count | Description | Operational Status |
| :--- | :---: | :---: | :--- | :--- |
| **KANGAYAM** | ❌ Absent | 0 | Tiruppur / Karur / Erode draught cattle (Most prominent TN draught breed) | **MISSING_CRITICAL_GAP** |
| **BARGUR** | ✅ Present | 470 | Anthiyur / Bhavani hills (Erode district hill draught cattle) | **AVAILABLE** |
| **UMBLACHERY** | ❌ Absent | 0 | Nagapattinam / Thiruvarur (Cauvery delta dwarf draught breed) | **MISSING_CRITICAL_GAP** |
| **PULIKULAM** | ❌ Absent | 0 | Madurai / Sivagangai (Indigenous Jallikattu cattle breed) | **MISSING_CRITICAL_GAP** |
| **ALAMBADI** | ❌ Absent | 0 | Dharmapuri / Salem (Hilly draught breed) | **MISSING_CRITICAL_GAP** |

> [!CAUTION]
> **Tamil Nadu Breed Support Policy:**
> 4 out of the 5 prominent indigenous cattle breeds of Tamil Nadu (**Kangayam, Umblachery, Pulikulam, Alambadi**) are **entirely missing** from this dataset. The platform **must never claim** Tamil Nadu indigenous breed support based on this archive.

---

## 4. Operational Verdict & Governance Boundary

- **Permitted Scope:** Deduplicated 5-breed visual classifier (`4975` clean images).
- **Prohibited Scope:** Weight estimation, age regression, or Tamil Nadu regional compliance claims.
- **Policy Guard:** Strictly prohibit claiming Tamil Nadu breed support until Kangayam, Umblachery, Pulikulam, and Alambadi are collected.
