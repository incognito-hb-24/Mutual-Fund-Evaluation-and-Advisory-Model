.PHONY: reproduce phase1 phase2 phase3

phase1:
	python -m src.phase1_calendar_merge

phase2:
	python -m src.phase2_features

phase3:
	python -m src.phase3_classification

reproduce: phase1 phase2 phase3
