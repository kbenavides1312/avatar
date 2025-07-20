.PHONY: serve

install:
	python3 -m venv venv || echo "venv already exists"
	venv/bin/pip install -r requirements.txt

run-test:
	python3 test.py

serve:
	venv/bin/python src/server.py

build:
	docker build -t api .

run:
	docker run -p 3001:3001 --env-file .env api

run-local:
	docker compose up | grep api

publish:
	docker tag api:latest $(ECR_REPO)/smet_backend:latest
	docker push $(ECR_REPO)/smet_backend:latest

build-and-publish:
	make build
	make publish
