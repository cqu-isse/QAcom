# QAcom
Quality Assurance for Automated Commit Message Generation

Accepted by SANER2021

# Note
Since we use Differential Evolution(DE) to tune the threshold in QAcom, the reusult will be a little different from the paper.

## Structure
- data: Cleaned dataset, Top1000 dataset, Top 2000 dataset
- archive: the results of preserved generated messages
- utils: tool classes
	- VocabularyAndMapping.py: build vacabulary and mapping
	- IgnoreRate.py: calculate the ignore-rate
	- EvaluationMetrics.py: evaluation the BLEU[1,2,3,4], METEOR, ROUGE-L
	- DifferentialEvolution.py: a tuning algorithms DE
- QAcom.py: implementation of QAcom


## Run
- python>=3.6
- QAcom arguments: dataset (Cleaned, Top1000, Top2000), 
    commit messages generation approach (NMT, NNGen, PtrGNCMsg)

$ pip install -r requirements.txt

$ python QAcom.py Cleaned NNGen
