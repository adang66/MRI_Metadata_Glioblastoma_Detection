# 🧠 Machine Learning Approaches to MRI-Based Glioblastoma Prognosis  

This project explores machine learning methods to predict glioblastoma (GBM) prognosis by leveraging both **MRI slices** and **clinical metadata**. Using the **UCSF Preoperative Diffuse Glioma MRI (UCSF-PDGM)** dataset, we implemented models that combine imaging data and patient-level clinical/molecular data to classify tumor grades under the **WHO CNS grading system**.  

---

## 📌 Overview  
Glioblastoma is the most common malignant brain tumor in adults and remains one of the most fatal cancers. Despite MRI being the standard for neuro-oncologic practice, its utility for prognosis remains unclear.  

**Goal:** Build models that integrate MRI imaging and metadata (age, sex, IDH mutation, MGMT status, etc.) to predict WHO tumor grade.  

---

## 📊 Dataset  
- **Source:** UCSF-PDGM (TCIA)  
- **Patients:** 360 with MRI + metadata  
- **MRI:** Central tumor slices (T1c), normalized & resized to 224×224  
- **Metadata:** Age, sex, IDH, MGMT, survival  

---

## ⚙️ Methods  
**Models:**  
- **AlexNet (MRI-only)**  
- **AlexNet + MLP (MRI + metadata fusion)**  

**Training:**  
- Optimizer: Adam  
- Loss: Cross-Entropy  
- Early stopping based on validation metrics  

**Metrics:**  
- Accuracy  
- Macro-averaged F1  

---

## 📈 Results  

| Model               | Accuracy | F1 (macro) |
|----------------------|----------|------------|
| **AlexNet**          | **89.0%** | **0.7811** |
| Multimodal AlexNet   | 88.67%   | 0.5538     |  

**Key Insight:** MRI features alone outperformed multimodal fusion due to noisy/missing metadata.  

---

## 🚀 Next Steps  
- Improve metadata encoding  
- Add more imaging modalities (FLAIR, T2)  
- Explore LLMs for metadata processing  

---

## 👥 Contributors  
- Maxine Baghdadi  
- Arpit Dang  
- Xiaoke Song  
- Yitong Wang  

**Advisor:** Dr. Jay Hou  
