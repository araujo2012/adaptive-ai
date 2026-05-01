# Adaptive AI

Adaptive AI e um pacote Python local-first para criar, treinar, avaliar e persistir modelos baseados em matrizes neurais adaptativas. O projeto foi pensado como um MVP simples de incorporar em outros projetos: a API publica principal e a classe `AdaptiveAI`, os dados ficam em disco no proprio workspace e a unica dependencia de runtime e `numpy`.

## Principais funcionalidades

- Cadastro e persistencia local de datasets de entrada/saida.
- Treinamento assincrono com controle por tempo maximo.
- Estrategias de treino `fixed` e `sample_square`.
- Avaliacao por tolerancia por coluna de saida.
- Predicao usando matrizes treinadas ou matrizes fornecidas manualmente.
- Registro de modelos, metricas e historico de jobs em SQLite.
- Pausa e cancelamento de jobs de treinamento.
- Armazenamento local em `.adaptive_ai/`, sem depender de servicos externos.

## Requisitos

- Python 3.11 ou superior.
- `numpy>=2.0`.

Para desenvolvimento e testes, use tambem `pytest>=8.0`.

## Instalacao

### Instalando direto do GitHub

```bash
python -m pip install "adaptive-ai @ git+https://github.com/araujo2012/adaptive-ai.git"
```

### Usando em outro projeto

Em um `requirements.txt`:

```txt
adaptive-ai @ git+https://github.com/araujo2012/adaptive-ai.git
```

Em um `pyproject.toml`:

```toml
dependencies = [
    "adaptive-ai @ git+https://github.com/araujo2012/adaptive-ai.git",
]
```

### Instalacao local para desenvolvimento

```bash
git clone https://github.com/araujo2012/adaptive-ai.git
cd adaptive-ai
python -m venv .venv
```

No Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

No macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install -e ".[test]"
```

## Uso basico

```python
import time

from adaptive_ai import AdaptiveAI


ai = AdaptiveAI(path="./workspace")

ai.set_input_output(
    inputs=[[0], [1], [2], [3]],
    outputs=[[0], [0], [1], [1]],
)

job = ai.start_training(
    max_seconds=2.0,
    tolerances=[0.25],
    amount_strategy="fixed",
    fixed_steps=50,
    learning_rate=0.1,
    seed=42,
)

while True:
    current_job = ai.get_training_job(job["job_id"])
    if current_job["status"] in {"completed", "failed", "canceled", "paused"}:
        break
    time.sleep(0.05)

models = ai.get_models()
best_model = ai.get_model(models[0]["model_id"])

prediction = ai.predict_with_matrices([[2.5]], best_model["matrices"])
print(prediction)
```

## Persistencia

Cada instancia de `AdaptiveAI` usa um diretorio de workspace:

```python
ai = AdaptiveAI(path="./workspace")
```

Dentro desse diretorio, o pacote cria `.adaptive_ai/` com:

- `adaptive_ai.sqlite3`: metadados de modelos, jobs e logs.
- `arrays/dataset.npz`: dataset atual.
- `models/*.npz`: matrizes dos modelos salvos.

O diretorio `.adaptive_ai/` e gerado em runtime e ja esta no `.gitignore` deste repositorio.

## API principal

| Metodo | Descricao |
| --- | --- |
| `set_input_output(inputs, outputs)` | Define o dataset atual e remove modelos antigos. |
| `put_input_output(inputs, outputs)` | Adiciona novas amostras ao dataset existente. |
| `get_dataset()` | Retorna os arrays `inputs` e `outputs` como `numpy.ndarray`. |
| `start_training(...)` | Inicia um job de treinamento em segundo plano. |
| `get_training_job(job_id)` | Consulta status, metricas basicas e erros de um job. |
| `get_training_logs(job_id, limit=100)` | Retorna logs recentes de treinamento. |
| `pause_training(job_id)` | Solicita pausa de um job em execucao. |
| `cancel_training(job_id)` | Solicita cancelamento de um job em execucao. |
| `get_models()` | Lista modelos salvos sem carregar as matrizes. |
| `get_model(model_id)` | Retorna metadados e matrizes de um modelo especifico. |
| `predict_with_matrices(inputs, matrices)` | Executa predicao usando uma lista de matrizes. |
| `evaluate_predictions(predicted, expected, tolerances)` | Calcula `accepted_count`, `accepted_rate` e `mse`. |
| `evaluate_matrices(inputs, outputs, matrices, tolerances)` | Prediz e avalia matrizes em uma unica chamada. |
| `train_matrices(inputs, outputs, matrices, steps, learning_rate)` | Treina matrizes diretamente sem iniciar um job persistido. |

## Formato dos dados

As entradas e saidas podem ser listas Python ou arrays NumPy. Internamente, o pacote converte os dados para `numpy.float64`.

```python
ai.set_input_output(
    inputs=[[0, 0], [1, 1]],
    outputs=[[0], [1]],
)
```

Como o treinamento usa ativacao sigmoid, os valores de `outputs` devem estar entre `0` e `1`.

As tolerancias devem ter o mesmo tamanho da dimensao de saida:

```python
metrics = ai.evaluate_predictions(
    predicted=[[0.52], [0.88]],
    expected=[[0.5], [1.0]],
    tolerances=[0.15],
)
```

## Treinamento

`start_training` cria um job assincrono e retorna imediatamente os metadados iniciais do job.

Parametros principais:

- `max_seconds`: tempo maximo de treinamento.
- `tolerances`: tolerancia aceita para cada coluna de saida.
- `amount_strategy`: `fixed` ou `sample_square`.
- `fixed_steps`: numero de passos por rodada quando `amount_strategy="fixed"`.
- `learning_rate`: taxa de aprendizado.
- `seed`: semente opcional para reprodutibilidade.

Status possiveis de um job:

- `running`
- `completed`
- `paused`
- `canceled`
- `failed`

## Desenvolvimento

Instale o pacote em modo editavel com dependencias de teste:

```bash
python -m pip install -e ".[test]"
```

Execute a suite de testes:

```bash
python -m pytest
```

## Observacoes e limitacoes

- Este e um MVP local-first, sem servidor, API HTTP ou CLI.
- Apenas um job de treinamento pode ficar em execucao por workspace.
- Os modelos sao salvos localmente e nao devem ser commitados no repositorio.
- A saida esperada precisa estar no intervalo `0` a `1`.
- O treinamento atual usa NumPy e CPU.

## Licenca

Este repositorio ainda nao inclui um arquivo `LICENSE`. Se o pacote for consumido por outros projetos, defina a licenca do repositorio para deixar as permissoes de uso explicitas.
