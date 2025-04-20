train:
	python train.py --data data/crowd.yaml --batch 16 --imgsz 640 --epochs 100

run:
	uvicorn main:app --host 0.0.0.0 --port 8000 --reload

#python train.py --data data/crowd.yaml --batch 8 --imgsz 320 --epochs 10
