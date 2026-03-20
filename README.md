# TODO
Подготовить данные для обработки через ноутбук data_processing. Там обработка для датасета маестро, цель добавить еще 1200-1300 песен из другого датасета

- LMD
- Pop-909

Слить в один датасет, нормализовать как в препроцессинге здесь, выложить в опенсорс

На нормализованном датасете переобучить SING и SINGLS, получить устойчивый результат

# SING
Similarity Incentivized Neural Generator, or SING, is a music generation system which uses self-similarity as attention.

## Required packages

- numpy
- pandas
- matplotlib
- pretty_midi
- mido
- PyTorch 
- tqdm 
- sparsemax

## Running the code

Each notebook has its own purpose.

**data_processing.ipynb**, as the name suggests, processes the MAESTRO dataset into a form usable by SING. This repository already contains preprocessed data, so it is unnecessary to rerun this code.

**att_lstm.py** is the file used to train SING and the LSTM ablation. Running it will train the model again; by default, it will train for 30 epochs. The "best_models" folder contains the best models with and without the attention mechanism.

**model-selection.py** can be used to find the model with lowest loss on the validation dataset.

**test-sim.ipynb** contains the code used to find MSE over the standardized SSM.

**gen-example.ipynb** uses the best pre-trained model to generate new pieces of music. From the notebook, you can both view the self-similarity matrix of the new generated piece and save the audio of the piece to "blank.midi", a file that is overwritten each time the relevant code in the notebook is run.

These notebooks were initially designed to be run on Google Colab, and adapting to run locally will require edits to the code.

# К защите
0. Формализация
	•	Представление: symbolic (REMI / MIDI).
	•	Структура задаётся в тактах:
(label, start_bar, end_bar).
	•	SSM считается по bar-level embeddings (cosine similarity).
1. Подготовка данных
	1.	Привести LMD / POP909 / MAESTRO к единому symbolic формату.
	2.	Для каждого трека:
	•	bar-level признаки
	•	построить SSM
	•	boundary detection
	•	сегментация (A/B/C)
	3.	Сохранить:
MIDI + SSM + segment plan.
2. Structure → SSM Generator
Задача:
segment plan → SSM
Реализация:
	•	генерировать bar embeddings z₁…z_T
	•	SSM = cosine(z_i, z_j)
Loss:
	•	L1 между SSM_pred и SSM_gt
	•	regularization диагонали
	•	block-consistency (одинаковые сегменты ближе)

3. Text → Prefix Generator
Задача:
text prompt → 8–16 тактов MIDI
	•	Небольшой Transformer
	•	conditioning на text embedding
	•	Prefix отвечает за стиль, не за структуру
(Можно упростить и использовать сэмплирование по жанру.)
4. Интеграция с SING
Вход:
	•	Prefix (symbolic)
	•	Predicted SSM
Нужно:
•	стабилизировать attention scaling
•	добавить коэффициент силы структуры
Обязательные абляции:
	1.	No SSM
	2.	GT SSM
	3.	Predicted SSM
	4.	Random SSM
5. End-to-End Pipeline
	1.	Text → prefix
	2.	Structure → SSM
	3.	Prefix + SSM → SING
	4.	Генерация полного трека