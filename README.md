ÜBERSICHT

Dieses Projekt implementiert einen Deep Q-Network (DQN) Agenten zur Lösung der OpenAI-Gym-Umgebung „LunarLander-v2“.

Der Agent lernt durch Interaktion mit der Umgebung, eine Raumsonde sicher auf einer Landefläche zu landen. Ziel ist es, die kumulative Belohnung (Reward) durch Reinforcement Learning zu maximieren.

UMGEBUNG

Umgebung: LunarLander-v2 (OpenAI Gym)

Zustandsraum: 8-dimensionale kontinuierliche Beobachtungsvektoren

Aktionsraum: 4 diskrete Aktionen

Gelöst-Bedingung: Durchschnittlicher Reward ≥ 200

Das Ziel besteht darin, durch Optimierung der Reward-Funktion stabile und sichere Landungen zu erreichen.

VERWENDETE METHODEN

Im Projekt werden folgende Reinforcement-Learning-Techniken eingesetzt:

Deep Q-Network (DQN)

Experience Replay (Replay Buffer)

Target Network

Soft Target Updates

ε-greedy Explorationsstrategie

Mini-Batch Gradient Descent

Diese Methoden erhöhen die Stabilität und Konvergenz des Trainingsprozesses.

MODELLARCHITEKTUR

Das Q-Netzwerk besteht aus:

Eingabeschicht (Zustandsdimension = 8)

Dense-Schicht (64 Neuronen, ReLU-Aktivierung)

Dense-Schicht (64 Neuronen, ReLU-Aktivierung)

Ausgabeschicht (lineare Aktivierung, Q-Werte für jede Aktion)

Das Target-Netzwerk besitzt die gleiche Architektur und wird regelmäßig mittels Soft Update aktualisiert.

TRAININGSPARAMETER

Diskontfaktor (Gamma): 0,995

Lernrate: 0,001

Größe des Replay Buffers: 100.000

Periodische Aktualisierung des Target-Netzwerks

Epsilon-greedy Strategie mit schrittweiser Reduktion

Der trainierte Agent erreicht eine stabile durchschnittliche Belohnung von über +200 und gilt damit als erfolgreich trainiert.

VERWENDETE TECHNOLOGIEN

Python

TensorFlow / Keras

NumPy

OpenAI Gym

AUSFÜHRUNG

Abhängigkeiten installieren:

pip install -r requirements.txt

Training starten:

python lunar_lander_dqn.py

Nach dem Training wird das Modell gespeichert als:
lunar_lander_model.h5

PROJEKTSTRUKTUR

lunar-lander-dqn/
lunar_lander_dqn.py
utils.py
images/
videos/
requirements.txt
README.txt

AUTOR

Ahmad Taher Mohamad
