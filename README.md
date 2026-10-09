# Dice Detector

Локальный CPU-сервис для ответа на вопрос «есть ли на фотографии стандартные RPG-кубики?». В ответе также доступны типы (`d4`, `d6`, `d8`, `d10`, `d12`, `d20`), confidence и bounding boxes.

Проект использует ONNX-модель `G-G-Games/diecamera-models/dice-shape`. Веса не входят в репозиторий: загрузка выполняется отдельно и требует явного принятия AGPL-3.0.

## Быстрый старт

Требования: Python 3.11 или 3.12 и [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run dice-detector download-model --accept-license
uv run dice-detector detect ./photo.jpg
```

Пример сокращённого результата:

```json
{
  "has_dice": true,
  "count": 1,
  "detections": [
    {
      "label": "d20",
      "confidence": 0.91,
      "box": {"x_min": 42.0, "y_min": 51.0, "x_max": 184.0, "y_max": 196.0}
    }
  ]
}
```

## HTTP API

```bash
uv run dice-detector serve --host 0.0.0.0
curl -F 'file=@photo.jpg' http://127.0.0.1:18080/v1/detect
```

Маршруты:

- `GET /health` — процесс API работает;
- `GET /ready` — модель загружена и сервис готов к inference;
- `POST /v1/detect` — multipart-поле `file` с изображением;
- `GET /docs` — OpenAPI UI.

При отсутствии модели процесс стартует, `/health` возвращает `200`, а `/ready` и `/v1/detect` — `503`. Это позволяет отличать работоспособность контейнера от готовности модели.

## Docker

Сначала загрузите модель на хосте, затем соберите и запустите контейнер:

```bash
uv run dice-detector download-model --accept-license
docker compose up --build
```

Модель подключается read-only из `./models`, API публикуется на `127.0.0.1:18080`. Для внешнего доступа используйте существующий reverse proxy и аутентификацию.

## Настройки

CLI принимает `--confidence`, `--iou`, `--threads` и `--model-dir`. Для запуска через `uvicorn dice_detector.api:app` доступны переменные:

- `DICE_DETECTOR_MODEL_DIR` — каталог модели, по умолчанию `models`;
- `DICE_DETECTOR_CONFIDENCE` — минимальный итоговый score, по умолчанию `0.85`;
- `DICE_DETECTOR_HIGH_CONFIDENCE_FALLBACK` — повторная проверка сложных кадров с отражением, поворотом, усилением контраста и фрагментами изображения; по умолчанию `true`;
- `DICE_DETECTOR_IOU` — порог NMS, по умолчанию `0.45`;
- `DICE_DETECTOR_ONNX_THREADS` — число CPU-потоков, по умолчанию не больше 6;
- `DICE_DETECTOR_MAX_UPLOAD_BYTES` — предел файла, по умолчанию 10 MiB;
- `DICE_DETECTOR_MAX_SOURCE_PIXELS` — жёсткий защитный предел исходного изображения, по умолчанию 150 млн пикселей;
- `DICE_DETECTOR_MAX_IMAGE_PIXELS` — предел in-memory копии для inference, по умолчанию 4 млн пикселей; более крупные фото уменьшаются с сохранением пропорций.

Итоговый confidence вычисляется как `objectness × probability класса`. `has_dice=true`, если после threshold и class-agnostic NMS осталась хотя бы одна детекция. Fallback не снижает порог: детекция из преобразованного кадра принимается только при совпадении двух вариантов, а фрагменты проверяются только для кадров с близким к порогу исходным score. Поэтому все возвращаемые детекции по-прежнему имеют confidence не ниже настроенного порога.

## Ограничения

- Модель обучена на ограниченном наборе камер, фонов и физических кубиков. Перед эксплуатацией проверьте её на собственном наборе положительных и отрицательных фотографий.
- Для сложного отрицательного кадра fallback может выполнить до 13 inference-проходов, увеличивая задержку ответа.
- Обычный `d6` визуально нельзя надёжно отличить от «RPG d6» только по назначению.
- Модель определяет наличие и тип кубика, но не значение верхней грани.
- При большом числе параллельных запросов ограничьте concurrency на уровне ASGI/reverse proxy: inference использует CPU-потоки.

## Проверки

```bash
uv run pytest
uv run ruff check .
uv run mypy
```

## Лицензия и происхождение модели

Код проекта предназначен для использования на условиях AGPL-3.0-only. Веса и метаданные модели принадлежат `G-G-Games` и распространяются отдельно по AGPL-3.0. Зафиксированы revision и SHA-256 обоих файлов; downloader проверяет checksum до атомарной установки. Подробности приведены в `THIRD_PARTY_NOTICES.md`.
