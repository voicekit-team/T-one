[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/t-tech/T-one)
[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-Image-blue)](https://hub.docker.com/r/tinkoffcreditsystems/t-one)
[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/voicekit-team/T-one/blob/master/LICENSE)
[![Python version](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)

<div align="center">
  <img src=".github/logo.png" height="124px"/><br/>
    <h1>T-one</h1>
  <p>Потоковый ASR-пайплайн на основе CTC-модели для русского языка</p>
  <p><a href="./README.md">English</a> | <b>Русский</b></p>
</div>

## 📝 О проекте

Этот репозиторий предоставляет полноценный потоковый пайплайн автоматического распознавания речи (ASR) **на русском языке, специализированный для домена телефонии**. Пайплайн включает в себя предобученную потоковую акустическую [CTC](https://huggingface.co/learn/audio-course/chapter3/ctc)-модель, модуль разделения на фразы (сплиттер), а также декодер, что позволяет использовать его как готовое решение для задач распознавания речи в реальном времени. Входной аудиопоток обрабатывается сегментами по 300 мс, границы фраз вычисляются с помощью сплиттера, а декодирование выполняется либо жадным методом, либо с использованием KenLM и beam search декодера.
Проект разработан ООО "ТЦР" и является практичной, высокопроизводительной реализацией ASR с модульными компонентами.

## ⚡️ Быстрый старт / Демо

Для быстрого запуска можно воспользоваться готовым Docker-образом и запустить сервис с веб-интерфейсом. Вы можете **распознавать аудиофайлы** или **использовать микрофон** для потокового распознавания — прямо в браузере.
*Рекомендуется использовать компьютер с минимум 4 ядрами CPU и 8 ГБ ОЗУ для стабильной работы.*

Запустите контейнер:
```bash
docker run -it --rm -p 8080:8080 tinkoffcreditsystems/t-one:0.1.0
```

Откройте в браузере `http://localhost:8080`.

![](.github/demo_website.png)

Также вы можете собрать образ из [Dockerfile](Dockerfile) самостоятельно и запустить контейнер:
```bash
docker build -t t-one .
docker run -it --rm -p 8080:8080 t-one
```

## 🛠️ Установка и использование

Убедитесь, что в вашей системе (Linux или macOS) установлены [Python](https://www.python.org/downloads/) (требуется версия `3.9` или выше) и [Poetry](https://python-poetry.org/docs/#installation) (рекомендуется версия `2.1` или новее).

Для Windows рекомендуется использовать контейнеризированную среду, например, Docker или [Windows Subsystem for Linux (WSL)](https://www.google.com/url?sa=E&q=https%3A%2F%2Fdocs.microsoft.com%2Fen-us%2Fwindows%2Fwsl%2Finstall), поскольку одна из зависимостей проекта, KenLM, официально не поддерживается в Windows. Хотя её установка в Windows и возможна (вы можете удалить `kenlm` из зависимостей Poetry и попытаться собрать её вручную с помощью [Vcpkg](https://github.com/kpu/kenlm?tab=readme-ov-file#building-kenlm---using-vcpkg)), этот процесс, как правило, сложнее и чреват проблемами с зависимостями.

### Установка и запуск веб-сервиса

1. Склонируйте репозиторий:
   ```bash
   git clone https://github.com/voicekit-team/T-one.git
   ```

2. Перейдите в каталог репозитория:
   ```bash
   cd T-one
   ```

3. Создайте и активуйте виртуальное окружение Python:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Для Windows используйте .venv\Scripts\activate
   ```

4. Для локальной установки пакета с зависимостями и запуска демонстрационного веб-сервиса вы можете использовать [Makefile](Makefile):
   ```bash
   make install
   make up_dev
   ```

5. Откройте в браузере `http://localhost:8080`

Как альтернативный вариант, вы можете установить пакет вручную с помощью Poetry после активации виртуального окружения:

1. Установите пакет и его зависимости для демонстрации:
   ```bash
   poetry install -E demo
   ```

2. Запустите демонстрационный веб-сервис:
   ```bash
   uvicorn --host 0.0.0.0 --port 8080 tone.demo.website:app --reload
   ```

3. Откройте в браузере `http://localhost:8080`

Подготовьте аудиофайл с **речью на русском языке** в одном из популярных форматов (.wav, .mp3, .flac, .ogg) или используйте [аудиофайл](tone/demo/audio_examples/audio_short.flac) из примеров репозитория.

### Запуск модели в Python

1. Вы также можете запустить офлайн-распознавание напрямую в Python:
   ```python
   from tone import StreamingCTCPipeline, read_audio, read_example_audio


   audio = read_example_audio() # либо read_audio("your_audio.flac")

   pipeline = StreamingCTCPipeline.from_hugging_face()
   print(pipeline.forward_offline(audio))  # офлайн-распознавание

   # запуск с передачей параметров сессии onnxruntime: 
   from tone.onnx_wrapper import ort

   # доступные опции и их описание: https://onnxruntime.ai/docs/api/python/api_summary.html#onnxruntime.SessionOptions
   so = ort.SessionOptions()
   so.inter_op_num_threads = 1
   so.intra_op_num_threads = 3

   pipeline = StreamingCTCPipeline.from_hugging_face(session_options=so)
   print(pipeline.forward_offline(audio))  # офлайн-распознавание
   ```

2. Пример потоковой обработки смотрите в разделе ["Расширенный пример использования"](#-расширенный-пример-использования).

### Triton Inference Server

Для экспорта акустической модели T-one в `TensorRT` и эффективному запуску в `Triton Inference Server` смотрите [подробную инструкцию](docs/triton_inference_server.ru.md).

### Дообучение модели

1. Установите зависимости для дообучения:
   ```bash
   poetry install -E finetune
   ```
2. Ознакомьтесь с [примером дообучения](examples/finetune_example.ipynb).

## 🎙️ Обзор ASR-пайплайна

Входящий аудиопоток делится на лету на сегменты по 300 мс (см. изображение ниже). Каждый сегмент подаётся в акустическую модель на основе архитектуры Conformer. Модель принимает на вход аудиосегмент и скрытое состояние от предыдущего окна, что позволяет сохранять акустический контекст между сегментами, а возвращает новое скрытое состояние вместе с лог-вероятностями (logprobs) символов алфавита для каждого фрейма. Алфавит состоит из букв русского языка, пробела и специального blank токена, который служит разделителем между группами символов.

Сплиттер принимает на вход лог-вероятности текущего аудиосегмента и своё собственное состояние, а возвращает своё обновлённое состояние и набор найденных фраз. Для каждого входного фрейма сплиттер определяет, содержит ли он речь. Когда сплиттер обнаруживает речевой фрейм, фраза считается начатой (первый фрейм с речью задаёт начало фразы). Завершается фраза в момент, когда сплиттер находит N последовательных фреймов без речи (последний фрейм с речью до начала тишины задаёт конец фразы). Каждая фраза содержит соответствующие лог-вероятности и временные метки начала и конца фразы.

Декодер получает на вход лог-вероятности фраз после сплиттера и преобразует их в текст. Используются два метода: Greedy decoder либо Beam Search decoder с языковой моделью KenLM. Итоговый текст с таймингами фраз возвращается клиенту.
![streaming\_acoustic\_scheme](.github/asr_pipeline.png)
(Здесь blank токен и пробел обозначены как '-' и '|')

Для более подробного изучения архитектуры, деталей реализации и принятых решений — читайте нашу [статью](https://habr.com/ru/companies/tbank/articles/929850).

## 💪 Расширенный пример использования

```python
from tone import StreamingCTCPipeline, read_stream_example_audio


pipeline = StreamingCTCPipeline.from_hugging_face()

state = None  # Текущее состояние ASR-пайплайна (None — начальное)
for audio_chunk in read_stream_example_audio():  # Используйте любой источник чанков аудио
    new_phrases, state = pipeline.forward(audio_chunk, state)
    print(new_phrases)

# Завершение обработки и получение оставшихся фраз
new_phrases, _ = pipeline.finalize(state)
print(new_phrases)
```

## 📊 Метрики качества

Для оценки качества распознавания используется метрика Word Error Rate ([WER](https://huggingface.co/spaces/evaluate-metric/wer)), которую можно интерпретировать как процент некорректно распознанных слов по сравнению с эталонной текстовой разметкой.

### Сравнение WER для различных моделей
| Категория | T-one (71M) | GigaAM-RNNT v2 (243M) | GigaAM-CTC v2 (242M) | Vosk-model-ru 0.54 (65M) | Vosk-model-small-streaming-ru 0.54 (20M) | Whisper large-v3 (1540M) |
|:--|:--|:--|:--|--:|:--|:--|
| Колл-центр | **8.63** | 10.22 | 10.57 | 11.28 | 15.53 | 19.39 |
| Прочая телефония | **6.20** | 7.88 | 8.15 | 8.69 | 13.49 | 17.29 |
| Именованные сущности | **5.83** | 9.55 | 9.81 | 12.12 | 17.65 | 17.87 |
| CommonVoice 19 (test split) | 5.32 | **2.68** | 3.14 | 6.22 | 11.3 | 5.78 |
| OpenSTT asr_calls_2_val исходная разметка | 20.27 | **20.07** | 21.24 | 22.64 | 29.45 | 29.02 |
| OpenSTT asr_calls_2_val исправленная разметка | **7.94** | 11.14 | 12.43 | 13.22 | 21.03 | 20.82 |

## 📈 Метрики производительности

Ниже приведены метрики производительности, полученные с помощью `trtexec`, после экспорта модели в TensorRT engine для различных GPU:

| Device               | Configuration   | Max  & Optimal Batch Size  |   RPS (Requests Per Second) |   SPS (Seconds Per Second) |
|:---------------------|:----------------|:---------------------------|:----------------------------|:---------------------------|
| T4                   | TensorRT        |                       32   |                        5952 |                       1786 |
| A30                  | TensorRT        |                      128   |                       17408 |                       5222 |
| A100                 | TensorRT        |                      256   |                       26112 |                       7833 |
| H100                 | TensorRT        |                     1024   |                       57344 |                      17203 |

Подробнее о методе оценки производительности можно прочитать [здесь](docs/performance_testing.ru.md).

## 📜 Лицензия

Код и модели в этом репозитории распространяются под лицензией [Apache 2.0](./LICENSE).

## 📚 Дополнительные материалы
- [T-one — открытая русскоязычная потоковая модель для телефонии](https://habr.com/ru/companies/tbank/articles/929850).
- [Как улучшить качество и скорость обучения системы распознавания речи](https://www.youtube.com/watch?v=OQD9o1MdFRE).
