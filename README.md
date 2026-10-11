# Leveraging Unpaired Data to Mitigate Paired Data Scarcity and Foster Generalizable Multimodal Models

> **Undergraduate Thesis | B.Sc. in Computer Science and Engineering, Brac University**  
> **Authors:** Md. Dhokhan Kamal, Ummay Saiba, Md. Zobayer, Tahsin Binta Towhid  
> **Supervisor:** Dr. Muhammad Iqbal Hossain, Associate Professor, Dept. of CSE, Brac University  
> **Defense Date:** October 2025 | Group: T2520826  

---

## 📄 Overview & Documentation
- **[Download Final Thesis Report (PDF)](BracU_CSE_Thesis_Report.pdf)**
- **[View Defense Presentation Slides (PDF)](BracU_CSE_Thesis_Defense_Slides.pdf)**

---

## 📌 Abstract
The escalating threat of antimicrobial resistance (AMR) necessitates accelerated discovery of novel antimicrobial peptides (AMPs). Existing computational models frequently focus on binary activity classification or ignore species-conditioned pathogen sensitivity and practical chemical synthesis constraints. 

In this work, we present a **Biophysical Fusion Ensemble** framework for species-conditioned Minimum Inhibitory Concentration (MIC) prediction. Our architecture integrates:
1. **ESM-2 Protein Language Model Embeddings (320-D):** Capturing deep contextual, structural, and evolutionary sequence relationships.
2. **12 Handcrafted Biophysical Descriptors:** Explicitly characterizing net charge, molecular weight, hydrophobic moments, amphipathicity, and amino acid composition statistics.
3. **Learnable Target-Species Embeddings:** Conditioning predictions on specific bacterial pathogens across 286 species.
4. **5-Model Ensemble Strategy:** Mitigating variance and providing calibrated prediction uncertainty estimates.

---

## 🔬 Dataset & Evaluation
- **Curated Bacteria-Only Benchmark:** 32,607 activity records, 6,148 unique peptides, across 286 bacterial targets (stratified 70/15/15 split without sequence leakage).
- **Internal Test Performance:**
  - Mean Absolute Error (MAE): **0.3625** (14.7% reduction over baseline)
  - Root Mean Squared Error (RMSE): **0.4774**
  - Fold Error: **2.3039×**
- **Independent External Validation (DRAMP Benchmark):**
  - Evaluated on 3,497 previously unseen peptide-species pairs: **MAE = 0.6152**, **RMSE = 0.8514**, **Pearson $r$ = 0.4064**, and **Spearman $\rho$ = 0.3988**.

---

## 🛠️ Tech Stack & Methods
- **Deep Learning:** PyTorch, Transformer (Self-Attention), ESM-2 (`esm2_t6_8M_UR50D`)
- **Scientific Computing & ML:** NumPy, Pandas, Scikit-Learn, SciPy
- **Visualization:** Matplotlib, Seaborn
