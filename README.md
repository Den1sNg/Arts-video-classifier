# Arts Video Classifier

Лабораторная работа по классификации видео из категории **Arts and Entertainment** набора данных HowTo100M. В проекте видео сначала преобразуются в предобученные CLIP-эмбеддинги изображений, а затем последовательности этих эмбеддингов классифицируются Transformer-моделью, обученной с нуля.

## Кратко о задаче

Цель работы: построить классификатор видео по тематическим подкатегориям раздела Arts из HowTo100M.

Вариант лабораторной предполагает:

| Требование | Реализация в проекте |
|---|---|
| Данные | HowTo100M, категория `Arts and Entertainment` |
| Входные признаки | Предобученные эмбеддинги изображений |
| Модель эмбеддингов | `openai/clip-vit-base-patch32` |
| Классификатор | Transformer Encoder |
| Обучение классификатора | С нуля, только на собранных видеоэмбеддингах |
| Платформа | Google Colab |
| Итоговый датасет | 599 успешно обработанных видео |

## Идея решения

Исходный HowTo100M содержит метаданные и YouTube `video_id`, но не хранит сами видео в удобном готовом виде. Поэтому пайплайн был построен так, чтобы не занимать много места на диске:

1. Из CSV-файла HowTo100M выбираются ролики из категории `Arts and Entertainment`.
2. Для каждого ролика формируется YouTube-ссылка.
3. Ссылки проверяются через `yt-dlp`.
4. Рабочие ролики скачиваются по одному во временный файл.
5. Из каждого видео извлекаются 16 кадров.
6. Каждый кадр пропускается через предобученный CLIP.
7. Для одного видео сохраняется тензор эмбеддингов размера `16 x 512`.
8. Временный видеофайл удаляется.
9. На готовых эмбеддингах обучается Transformer-классификатор.

Такой подход позволяет не хранить исходные видео постоянно: в проекте сохраняются только маленькие `.pt` файлы с эмбеддингами и итоговая модель.

## Структура проекта

```text
laba2_project/
  README.md
  requirements.txt
  .gitignore

  src/
    collect_working_links.py
    make_embeddings_from_links.py
    train_transformer_on_embeddings.py

  data/
    arts_video_links.csv
    arts_working_links.csv
    arts_checked_links.csv
    embeddings_manifest.csv

  embeddings/
    books/
    carnivals_circuses_and_theme_parks/
    cosplay_and_role_playing/
    costumes/
    exhibited_arts/
    movies/
    music/
    performing_arts/

  models/
    best_model.pt

  results/
    model_results/
      results.json
      metrics.json
      classification_report.txt
      confusion_matrix.csv
      history.csv
      training_history.csv
      train_split.csv
      val_split.csv
      test_split.csv
      labels.json
```

Назначение основных папок:

| Папка | Назначение |
|---|---|
| `src/` | Python-скрипты для проверки ссылок, извлечения эмбеддингов и обучения модели |
| `data/` | CSV-файлы со ссылками, статусами проверки и манифестом эмбеддингов |
| `embeddings/` | Готовые CLIP-эмбеддинги видео в формате `.pt` |
| `models/` | Сохраненная лучшая Transformer-модель |
| `results/model_results/` | Метрики, история обучения, confusion matrix и train/val/test split |

## Описание скриптов

### `src/collect_working_links.py`

Скрипт проверяет YouTube-ссылки из `arts_video_links.csv` и сохраняет:

| Файл | Что хранит |
|---|---|
| `arts_working_links.csv` | Ссылки, которые удалось проверить как доступные |
| `arts_checked_links.csv` | Все проверенные ссылки: успешные и неуспешные |

Проверка выполняется через `yt-dlp --simulate`, то есть видео при этом не скачивается полностью.

### `src/make_embeddings_from_links.py`

Скрипт создает CLIP-эмбеддинги:

1. Читает `arts_working_links.csv`.
2. Пропускает уже обработанные `video_id` из `embeddings_manifest.csv`.
3. Скачивает одно видео во временный файл.
4. Извлекает 16 кадров.
5. Получает CLIP-эмбеддинги размерности 512 для каждого кадра.
6. Сохраняет `.pt` файл в `embeddings/<label>/<video_id>.pt`.
7. Записывает результат в `embeddings_manifest.csv`.
8. Удаляет временное видео.

Один объект классификации представлен матрицей:

```text
16 кадров x 512 признаков
```

### `src/train_transformer_on_embeddings.py`

Скрипт обучает Transformer-классификатор:

1. Читает `embeddings_manifest.csv`.
2. Берет только строки со статусом `ok`.
3. Находит соответствующие `.pt` файлы.
4. Делит данные на `train`, `val` и `test`.
5. Обучает Transformer Encoder с нуля.
6. Выбирает лучшую модель по `val_f1_macro`.
7. Сохраняет модель и итоговые метрики.

## Использованные технологии

| Компонент | Назначение |
|---|---|
| Python | Основной язык реализации |
| PyTorch | Обучение Transformer-модели |
| Transformers | Загрузка CLIP |
| CLIP | Получение предобученных image embeddings |
| OpenCV | Извлечение кадров из видео |
| yt-dlp | Проверка и скачивание YouTube-видео |
| scikit-learn | Train/val/test split и метрики |
| pandas | Работа с CSV |
| Google Colab | Запуск пайплайна и обучение |

## Подготовка окружения

Проект рассчитан на запуск в Google Colab. Для установки зависимостей:

```bash
pip install -r requirements.txt
```

В Colab удобнее запускать так:

```python
!pip install -q numpy pandas torch scikit-learn opencv-python pillow transformers tqdm yt-dlp safetensors
```

Если используется GPU, PyTorch автоматически выберет `cuda`. Если GPU недоступен, скрипты могут работать на CPU, но извлечение CLIP-эмбеддингов будет заметно медленнее.

## Запуск пайплайна

### 1. Проверка рабочих ссылок

```python
!python /content/collect_working_links.py
```

Скрипт поддерживает продолжение работы: уже проверенные ссылки повторно не проверяются.

### 2. Извлечение эмбеддингов

```python
!python /content/make_embeddings_from_links.py
```

Скрипт также поддерживает продолжение работы. Если в `embeddings_manifest.csv` уже есть успешная запись для видео, оно не обрабатывается повторно.

### 3. Обучение модели

```python
!python /content/train_transformer_on_embeddings.py
```

После обучения появляются:

| Файл | Содержимое |
|---|---|
| `models/best_model.pt` | Лучший checkpoint модели |
| `results/model_results/results.json` | Итоговые метрики и отчет |
| `results/model_results/classification_report.txt` | Classification report |
| `results/model_results/confusion_matrix.csv` | Матрица ошибок |
| `results/model_results/history.csv` | История обучения по эпохам |
| `results/model_results/train_split.csv` | Обучающая выборка |
| `results/model_results/val_split.csv` | Валидационная выборка |
| `results/model_results/test_split.csv` | Тестовая выборка |

## Данные

После фильтрации и обработки получилось 599 видеоэмбеддингов.

Распределение по классам:

| Класс | Количество видео |
|---|---:|
| `music` | 349 |
| `costumes` | 136 |
| `books` | 51 |
| `movies` | 33 |
| `cosplay_and_role_playing` | 10 |
| `performing_arts` | 10 |
| `carnivals_circuses_and_theme_parks` | 6 |
| `exhibited_arts` | 4 |
| **Итого** | **599** |

Разбиение на выборки:

| Выборка | Количество |
|---|---:|
| Train | 419 |
| Validation | 90 |
| Test | 90 |
| **Total used** | **599** |

## Архитектура модели

Классификатор принимает последовательность из 16 CLIP-эмбеддингов:

```text
video -> 16 frames -> CLIP -> tensor [16, 512] -> Transformer -> class
```

Основные параметры обучения:

| Параметр | Значение |
|---|---:|
| Input shape | `16 x 512` |
| Number of classes | 8 |
| Batch size | 32 |
| Epochs | 25 |
| Learning rate | `1e-4` |
| Weight decay | `1e-4` |
| Optimizer | AdamW |
| Loss | CrossEntropyLoss with class weights |
| Best model criterion | `val_f1_macro` |

## Результаты

Итоговые метрики на тестовой выборке:

| Метрика | Значение |
|---|---:|
| Test accuracy | 0.8667 |
| Test F1 macro | 0.5532 |
| Best validation F1 macro | 0.5873 |
| Test size | 90 |

Classification report:

| Класс | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| `books` | 1.00 | 1.00 | 1.00 | 9 |
| `carnivals_circuses_and_theme_parks` | 1.00 | 0.50 | 0.67 | 2 |
| `cosplay_and_role_playing` | 0.25 | 1.00 | 0.40 | 1 |
| `costumes` | 0.70 | 0.82 | 0.76 | 17 |
| `exhibited_arts` | 0.00 | 0.00 | 0.00 | 0 |
| `movies` | 0.67 | 0.67 | 0.67 | 3 |
| `music` | 1.00 | 0.88 | 0.94 | 58 |
| `performing_arts` | 0.00 | 0.00 | 0.00 | 0 |
| **Accuracy** |  |  | **0.87** | **90** |
| **Macro avg** | **0.58** | **0.61** | **0.55** | **90** |
| **Weighted avg** | **0.92** | **0.87** | **0.89** | **90** |

Confusion matrix:

| True \ Pred | books | carnivals | cosplay | costumes | exhibited | movies | music | performing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| books | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| carnivals | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
| cosplay | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| costumes | 0 | 0 | 2 | 14 | 0 | 0 | 0 | 1 |
| exhibited | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| movies | 0 | 0 | 0 | 0 | 1 | 2 | 0 | 0 |
| music | 0 | 0 | 1 | 5 | 0 | 1 | 51 | 0 |
| performing | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Анализ результатов

Модель достигла высокой общей точности `86.7%`, но `macro F1 = 55.3%` заметно ниже. Это связано с сильным дисбалансом классов: большая часть датасета относится к `music` и `costumes`, а некоторые классы представлены единичными примерами.

Лучше всего модель распознает:

| Класс | F1-score |
|---|---:|
| `books` | 1.00 |
| `music` | 0.94 |
| `costumes` | 0.76 |

Слабее всего оцениваются классы с малым числом объектов. Например, `cosplay_and_role_playing` имеет только 1 пример в тестовой выборке, а `exhibited_arts` и `performing_arts` не попали в test split, поэтому их итоговый F1 равен 0.

История обучения показывает, что `train_loss` снижался с `2.1977` до `0.0750`, то есть модель действительно обучалась. Лучший `val_f1_macro = 0.5873` был достигнут на 9-й эпохе. После этого loss продолжал уменьшаться, но качество на validation не росло стабильно, что указывает на частичное переобучение.

## Ограничения

1. HowTo100M содержит старые YouTube-ссылки, часть роликов уже удалена, закрыта или недоступна.
2. Google Colab часто получает ошибку YouTube `Sign in to confirm you're not a bot`.
3. Из-за ограничений Colab и YouTube удалось обработать 599 видео, а не весь исходный набор.
4. Классы сильно несбалансированы.
5. Для малых классов метрики нестабильны, потому что в test split попало слишком мало объектов.

## Вывод

В рамках лабораторной работы был реализован полный пайплайн классификации видео:

1. Извлечение ссылок из HowTo100M.
2. Проверка доступности YouTube-видео.
3. Последовательное скачивание видео без постоянного хранения.
4. Извлечение 16 кадров из каждого ролика.
5. Получение предобученных CLIP-эмбеддингов.
6. Обучение Transformer-классификатора с нуля.
7. Оценка модели на тестовой выборке.

Итоговая модель обучена на 599 видео и достигла `86.7%` accuracy на тестовой выборке. Основное ограничение результата связано не с архитектурой модели, а с доступностью видео и дисбалансом классов.

## Загрузка проекта на GitHub

Рекомендуемое название репозитория:

```text
arts-video-classifier
```

Команды для первого коммита:

```bash
cd "D:\Teterin's labs\Laba2\laba2_project"
git init
git add .
git commit -m "Add arts video classifier lab project"
```

После создания пустого репозитория на GitHub:

```bash
git branch -M main
git remote add origin https://github.com/<your-username>/arts-video-classifier.git
git push -u origin main
```

Если GitHub отклонит push из-за больших файлов, нужно убрать тяжелые файлы из Git или использовать Git LFS для `.pt` файлов.
