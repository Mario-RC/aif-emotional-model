---------
OVERVIEW
---------
The emotional dialogues between a USER and a CHATBOT consist of a PROMPT (P) and a RESPONSE (R). The RESPONSE is composed of 3 sentences (R1, R2 and R3). Each dialogue is about a different topic. The USER and the CHATBOT may have the same or different emotions.
P: USER utterance containing a specific emotion.
R1: First sentence that must be empathic with respect to the PROMPT.
R2: Second sentence expressing the chatbot's internal emotion.
R3: A question to encourage the USER to continue the conversation, trying to avoid yes/no answers.

--------
EMOTIONS
--------
There are 7 possible emotions:
[A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise and [G] Neutral.

-----
TASKS
-----
We are going to evaluate all these features of the dialogue through different taks.

Task1: Evaluate the empathiness of the response provided by the chatbot given a previous turn from the user (binary: Yes/No).
Task2: Select the emotion provided by the user. Also, annotate the emotion provided by the chatbot considering the previous sentence from the user (7 possible emotions).
Task3: Evaluate the adequacy of the follow-up question formulated by the chatbot considering the previous turn from the user (binary: Yes/No).
Task4: Give an overall rate of the response provided by the chatbot considering the previous user turn (1-5).

----------
EVALUATION
----------
Task 1, 3, and 4: Rate all rows of each task from 1 to 5, with 1 being the worst and 5 being the best.
Task 2: Choose the emotion expressed in P and R2 among the possible emotions.

-----------------
DIALOGUE EXAMPLES
-----------------
USER (P)
P - (FEAR) Yes, thank you so much for the encouragement. But I'm still nervous about sharing my work with others.
CHATBOT (R)
R1 - (FEAR) It's natural to be nervous, but think of how fulfilling it will be to share your words with the world.
R2 - (DISGUST) And don't listen to the critics who don't understand the work you put in. Their opinion is not your truth.
R3 - (NEUTRAL) Do you have any trusted friends or family members who can give you feedback on your work?

USER (P)
P - (ANGER) The Assyrians had a reputation for extreme cruelty towards those they conquered.
CHATBOT (R)
R1 - (ANGER) It's true, their conquests often involved the destruction of entire cities.
R2 - (FEAR) It's terrifying to think about the scale of their violence and how it impacted the lives of so many.
R3 - (NEUTRAL) Have you ever visited the ruins of an ancient Assyrian city?





30% iguales para todos los anotadores - IAA
70% diferentes para cada anotador - variabilidad
Tarea 4, 30% iguales que tareas 1, 2 y 3. 70% diferentes 

Combinacciones de emociones diferentes para tener muchos datos anotados diferentes, mejora la variabilidad

Desplegable con emociones o números para las anotaciones







----------------
POR CADA USUARIO
----------------
           IIA (30%)     Variabilidad (70%)
          -----------   --------------------
Modelo 1:      3                 7

Modelo 2:      3                 7

Modelo 3:      3                 7

Modelo 4:      3                 7

Modelo 5:      3                 7
              ----              ----
               15                35

------------------------------------------------
FRASES DE TEST TOTALES PARA TODOS LOS ANOTADORES
------------------------------------------------
(35 VARIABILIDAD * 5 USUARIOS) + 15 IAA = 190