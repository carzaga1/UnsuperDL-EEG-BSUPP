# Project Thesis Title: *An Unsupervised Deep Learning Algorithm for Burst Suppression Detection in Pediatric EEG Data: Assessing Generalizability.* 
   
## Description
 
This repository contains the implementation and analysis of a deep learning framework designed to enhance the detection of burst suppression (BS) patterns and compute bursts per minute (BPM) on pediatric EEG data from children in the pediatric intensive care unit (PICU) at Universitäts-Kinderspital Zürich (KISPI).

The project pairs a pre-validated unsupervised surrogate algorithm with a deep learning architecture to improve performance and accuracy in classifying BS patterns. Initially, the unsupervised algorithm is validated using pediatric EEG data alone. Subsequently, the algorithm is integrated with an unsupervised deep learning architecture and compared to its standalone counterpart. Both the standalone and paired versions will be benchmarked against manual annotations from experienced EEG specialists.

This work aims to advance the automation of burst suppression detection, providing a fully interpretable and patient-specific solution that can be integrated seamlessly into ICU workflows. It also seeks to demonstrate the algorithm's generalizability across different patient groups, ensuring its potential application in both adult and pediatric care.

## Authors

**HSLU MSc Student**:  Carlos Arzaga  

**Supervisors**: [Oliver Staubli](https://www.oliverstaubli.ch/de/about/work/), [Prof. Dr. Mirko Birbaumer](https://www.hslu.ch/de-ch/hochschule-luzern/ueber-uns/personensuche/profile/?pid=1537) 

**Co-Supervisors**: [Prof. Dr. med Georgia Ramantani](https://www.kispi.uzh.ch/kinderspital/person/ramantani-phd-georgia), [Prof. Dr. Med Emanuela Keller](https://www.usz.ch/team/emanuela-keller/)

**Internal Support**: [Dr. med Andrea Rüegger](https://www.kispi.uzh.ch/kinderspital/person/rueegger-andrea), [Alex Lo Biundo Santo Pietro](https://www.kispi.uzh.ch/kinderspital/person/lo-biundo-santo-pietro)

**External Support**: [Jenny Schmid](https://www.ifi.uzh.ch/en/ivda/team/schmid0.html), [Marko Seric](https://www.usz.ch/team/marko-seric/) 

## Research Objectives: 

1. Assess the generalizability (geographical/domain) via an external validation of a pre-validated unsupervised ML algorithm for BS detection in pediatric patients. 

2. Improve BS detection by pairing the ML algorithm with a DL architecture.  
    - Identify the most effective unsupervised DL architecture for EEG data analysis. 
        - Self Organizing Maps (SOM) 
        - Autoencoder (AE) 
        - Artificial Neural Networks (ANN) 
    - Evaluate new architecture pairing(s).

3. Investigate the agreement between human EEG annotators and the DL algorithm. 

## Methodology: 

- **Data**: Retrospective collection of EEG data from pediatric patients in neurocritical care at the Universitäts-Kinderspital Zürich (KISPI). 
- **Data Preparation**: Anonymization and preprocessing of EEG and clinical data. 
- **Data Labelling**: Manual labeling of BS patterns by experienced EEG annotators. 
    - Dr. Med. Rüegger Andrea | Kinderspital Zürich 
    - Alex Lo Biundo Santo Pietro | Kinderspital Zürich 

- **Model Development**: Iterative process in training and validation of unsupervised ML and DL models. 
- **Evaluation**: Assessment of model performance using metrics like sensitivity, specificity, precision, AUROC, F1-score, NPV, MAE, and Cohen's kappa.

<!-- ## Version History

* 0.0001
-->
## License

This project is licensed under the Apache 2.0 License - see the [Licence.txt](LICENSE.txt) file for details
-->
## Acknowledgments

