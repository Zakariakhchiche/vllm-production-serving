.PHONY: test lint plan up bench
test:            ## tests unitaires et d'intégration (sans GPU)
	pytest -q
lint:
	ruff check .
plan:            ## dimensionnement et commande vllm pour chaque profil
	@for p in profiles/*.yaml; do echo "== $$p"; python -m serving.render_args $$p; done
up:              ## pile locale GPU : 2 répliques + routeur + Prometheus + Grafana
	docker compose -f deploy/docker-compose.yml up -d
bench:           ## banc de charge via le routeur
	python -m bench.loadtest --url http://localhost:8080 --model assistant-7b --concurrency 32 --requests 256 --api-key $$VLLM_API_KEY
