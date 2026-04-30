# День 14 — Demo Version

Демо показывает полный RAG-сценарий:

1. загрузка PDF-документов;
2. извлечение текста, chunking и индексация в Qdrant;
3. чат с AI по документам;
4. ссылки на исходные PDF и страницы/чанки.

## Dataset

Используем технический demo dataset на 300 arXiv PDF:

```bash
python3 scripts/prepare_demo_dataset.py
```

Результат:

- `data/demo_documents/technical_300/` — 300 PDF;
- `data/demo_documents/technical_300/manifest.json` — состав датасета;
- `data/demo_documents/technical_300/demo_questions.json` — вопросы для показа.

По умолчанию PDF создаются hardlink-ами, поэтому папка выглядит как отдельный dataset, но не дублирует гигабайты на диске.

## Run Demo Stack

Перед запуском убедись, что Ollama доступна и модели установлены:

```bash
ollama pull nomic-embed-text
ollama pull llama3
```

Запуск backend/frontend в demo-режиме:

```bash
docker compose -f docker-compose.yml -f docker-compose.demo.yml up --build
```

Demo override использует отдельные пути:

- `data/demo_uploads`
- `data/demo_qdrant`
- collection `demo_documents`

Так демо не смешивается с рабочими экспериментами.

## Bulk Upload And Index

В интерфейсе можно нажать `Файлы` для выбора нескольких документов или `Папка` для загрузки всей demo-папки. Frontend отфильтрует неподдерживаемые файлы и отправит документы батчами.

CLI-вариант после старта backend:

```bash
python3 scripts/bulk_ingest.py \
  --input-dir data/demo_documents/technical_300 \
  --upload-url http://127.0.0.1:8000/upload \
  --limit 300
```

Скрипт сохранит отчёт в `data/reports/bulk_ingest_report_*.json`.

Проверка индекса:

```bash
curl http://127.0.0.1:8000/documents
```

Ожидаемо: `total_documents` около 300 и `total_chunks` больше нуля.

## Chat Demo

Открыть:

```text
http://127.0.0.1:5173
```

В интерфейсе есть быстрые demo-вопросы. Ответ AI должен вернуться вместе с источниками: файл, страница, chunk и ссылка на PDF.

Примеры вопросов также лежат здесь:

```bash
cat data/demo_documents/technical_300/demo_questions.json
```
