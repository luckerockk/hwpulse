# HWPULSE

Терминальный монитор системы в реальном времени (CPU/GPU/RAM)

## Возможности

- Цветовые пороги для нагрузки, температуры и мощности.
- История Min/Max для каждой метрики с момента запуска.
- Графики по основным метрикам (в режиме по умолчанию).
- Режимы JSON для интеграции с внешними программами.

## Требования

- Linux
- Python **3.9+** (рекомендуется 3.10+)
- `free` (пакет `procps`)
- Для GPU-метрик: `nvidia-smi`
- Для температуры CPU: `sensors` (пакет `lm-sensors`)
- Для CPU power: `/sys/class/powercap/intel-rapl:0/energy_uj` (если нет, метрика будет недоступна)

## Установка зависимостей (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y python3 procps util-linux lm-sensors
```

Опционально инициализировать датчики:

```bash
sudo sensors-detect --auto
```

Для GPU-метрик нужен установленный драйвер NVIDIA (`nvidia-smi` должен работать).

## Файлы

- `hwpulse.py` - основной скрипт.
- `install.sh` - установка `hwpulse` (копирование в `/opt/hwpulse` + launcher в `/usr/local/bin`).
- `uninstall.sh` - удаление `hwpulse` из `/usr/local/bin` и скрипта из `/opt/hwpulse`.

## Режимы работы

1. UI с графиками (по умолчанию):

```bash
python3 hwpulse.py
```

2. UI без графиков:

```bash
python3 hwpulse.py --nograph
```

3. JSON-режим по stdin/stdout (для IPC):

```bash
python3 hwpulse.py --json-stdio
```

- Процесс работает постоянно.
- Команда запроса: `get` (строка `get\n` в stdin).
- На каждый `get` возвращается один JSON в stdout.

4. Одноразовый JSON-снимок:

```bash
python3 hwpulse.py --json-once
```

- Скрипт один раз собирает метрики, печатает JSON и завершает работу.

Справка:

```bash
python3 hwpulse.py --help
```

## Формат JSON

- Корневые поля:
  - `timestamp` - Unix time в миллисекундах.
  - `systemName` - имя системы (из `PRETTY_NAME`).
  - `kernel` - версия ядра (из `uname -r`).
  - `cpu` - объект CPU (всегда присутствует).
  - `gpu` - объект GPU (присутствует только если GPU определился).

- Внутри `cpu`/`gpu`:
  - `modelName`
  - доступные метрики для соответствующего устройства.

Принципы заполнения:

- Если метрика не поддерживается на системе, ключ этой метрики не добавляется в JSON.
- Если метрика поддерживается, но в конкретный момент не считалась, значение будет `null`.

Пример:

```json
{
  "timestamp": 1770978423351,
  "systemName": "Debian GNU/Linux 12 (bookworm)",
  "kernel": "6.12.69+deb13-amd64",
  "cpu": {
    "modelName": "AMD Ryzen 9 9950X 16-Core Processor",
    "loadPercent": 0.0,
    "tempC": 49.8,
    "powerW": 6.8,
    "freqAvgMhz": 3131.0,
    "freqMaxMhz": 5062.0,
    "ramUsedMb": 3310.0,
    "ramTotalMb": 61835.0
  },
  "gpu": {
    "modelName": "NVIDIA GeForce RTX 5090",
    "loadPercent": 0.0,
    "tempC": 43.0,
    "powerW": 33.02,
    "fanPercent": 0.0,
    "coreMhz": 240.0,
    "memMhz": 405.0,
    "vramUsedMb": 147.0,
    "vramTotalMb": 32607.0
  }
}
```

## Глобальная команда `hwpulse`

Чтобы получить команду `hwpulse` из любой директории:

```bash
chmod +x install.sh
./install.sh
```

После этого доступны:

```bash
hwpulse
hwpulse --nograph
hwpulse --json-stdio
hwpulse --json-once
```

## Удаление

```bash
chmod +x uninstall.sh
./uninstall.sh
```
