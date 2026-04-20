# Car Classification: Bad Buy Prediction

Предсказание того, является ли автомобиль «лимоном» (плохой покупкой), на основе данных об аукционах и характеристиках автомобиля.

## Задача

Бинарная классификация: определить, будет ли автомобиль проблемным (`IsBadBuy = 1`) или нет (`IsBadBuy = 0`).

## Данные

Источник: [Kaggle - Don't Get Kicked!](https://www.kaggle.com/c/DontGetKicked) 

**Целевая переменная:** `IsBadBuy` - 1 если автомобиль проблемный, 0 иначе.

**Основные признаки:**
- Информация об аукционе (`Auction`, `AUCGUART`, `PRIMEUNIT`)
- Характеристики автомобиля (`Make`, `Model`, `VehicleAge`, `VehOdo`)
- Ценовые показатели (MMR цены на покупку и продажу)
- География (`VNST`, `VNZIP1`)

## Что реализовано

### Обработка данных
| Этап | Описание |
| :--- | :--- |
| Разделение по дате | Train/Val/Test на основе `PurchDate` (без утечек) |
| One-Hot Encoding | Для категорий с ≤16 уникальных значений |
| Count Encoding | Для категорий с >16 уникальных значений |
| Бинарные признаки | `PRIMEUNIT`, `AUCGUART` -> 0/1 |
| Заполнение пропусков | `most_frequent` для категорий, `median` для чисел |
| Нормализация | StandardScaler для числовых признаков |

### Feature Engineering
| Признак | Формула |
| :--- | :--- |
| `current_to_acquisition` | `MMRCurrentAuctionAveragePrice / MMRAcquisitionAuctionAveragePrice` |
| `clean_to_avg` | `MMRCurrentRetailCleanPrice / MMRCurrentRetailAveragePrice` |
| `wear_index` | `VehicleAge * VehOdo` |
| `odo_per_year` | `VehOdo / (VehicleAge + 1)` |
| `price_gap` | `MMRCurrentRetailAveragePrice - MMRAcquisitionAuctionAveragePrice` |
| `avg_price_by_make` | Средняя цена по марке |
| `make_popularity` | Частота встречаемости марки |

### Модели (собственные реализации)
| Модель | Метод |
| :--- | :--- |
| **LogisticReg** | SGD с L2-регуляризацией, сигмоида |
| **KNN** | Евклидово расстояние, k ближайших соседей |
| **NaiveBayes** | Гауссовское распределение, логарифмирование для устойчивости |

### Метрики (собственные реализации)
| Метрика | Описание |
| :--- | :--- |
| `my_gini` | Коэффициент Джини (2 × ROC-AUC − 1) через ранги |
| `recall` | TP / (TP + FN) |
| `precision` | TP / (TP + FP) |
| `f1_score` | 2 × (precision × recall) / (precision + recall) |
| `auc_pr` | Площадь под PR-кривой (метод трапеций) |

### Отбор признаков
- **Ручной отбор** - удаление слабых коэффициентов (порог 1% от max)
- **L1-регуляризация** - признаки с нулевыми весами

### Оптимизация гиперпараметров (Optuna)
- LogisticRegression: `C`, `penalty`, `solver`, `class_weight`
- Метрика оптимизации: ROC-AUC

## 📈 Результаты

| Модель | Gini (val) |
| :--- | :--- |
| LogisticRegression (базовая) | 0.378 |
| GaussianNB (базовая) | **0.420** |
| KNN (базовая) | 0.312 |
| LogisticRegression + Feature Engineering | 0.403 |
| LogisticRegression + Feature Engineering + Optuna | **0.442** |
| LogisticRegression (Test) | **0.473** |

### Сравнение собственных vs sklearn

| Модель | Gini (my) | Gini (sklearn) |
| :--- | :--- | :--- |
| LogisticRegression | **0.378** | 0.378 |
| GaussianNB | 0.420 | **0.421** |
| KNN | **0.327** | 0.327 |

> Результаты идентичны - реализации корректны.

### Отбор признаков

| Метод | Gini |
| :--- | :--- |
| Без отбора | 0.403 |
| Ручной отбор (коэффициенты) | 0.364 |
| L1-отбор | **0.400** |

### AUC-PR (собственная vs sklearn)

| Метод | AUC-PR |
| :--- | :--- |
| sklearn (`average_precision_score`) | 0.425 |
| Собственная реализация | **0.425** |
