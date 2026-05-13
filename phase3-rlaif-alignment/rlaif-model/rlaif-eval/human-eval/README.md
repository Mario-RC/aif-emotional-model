# OVERVIEW

The dialogues are emotional conversations between a Human and a Chatbot. Each dialogue consists of a Human Utterance followed by a Chatbot Response. The Chatbot’s Response is always composed of 3 sentences. Each dialogue focuses on a different topic, and both the Human and the Chatbot may express the same or different emotions. In this annotations, you will evaluate various features of these dialogues across different tasks.


# DIALOGUE STRUCTURE

The structure of each dialogue is as follows:

**Human**: (Human Emotion) Utterance.

**Chatbot**: (Empathy) Response_1. (Chatbot Emotion) Response_2. (Neutral Emotion) Response_3.

Details:
- Utterance: A Human statement that conveys a specific emotional tone.
- Response_1: This sentence should be empathic with respect to the Human's Utterance.
- Response_2: This sentence expresses the Chatbot's internal emotion.
- Response_3: This sentence is a follow-up question intended to engage the Human and encourage further conversation. The question should be open-ended to avoid yes/no answers.


# EMOTIONS

There are 7 possible emotions that can be expressed by both the Human and the Chatbot: Anger, disgust, fear, happiness, sadness, surprise and neutral.


# TASK 1: Quality Evaluation of Chatbot Responses

In Task 1, you will evaluate the quality of the Chatbot’s response given the previous turn from the user. Three different RL techniques (SFT, PPO, DPO) — all built on top of the same base model (`gemma-2-9b-it`) — provide responses. For each response, you need to assess the following components: Empathy in Response_1, Chatbot's internal emotion in Response_2 and Engagement in Response_3.

Each response is ranked from 1 to 3 (with 1 being the best and 3 being the worst).

Task 1 Data Columns:
- **DID**: Dialogue ID.
- **USER_T1**: Human turn 1.
- **CHATBOT_T1**: Chatbot turn 1.
- **USER_T2**: Human turn 2.
- **CHATBOT_T2**: Chatbot turn 2.
- **USER_T3**: Human turn 3.
- **RESPONSE_1**: Response to Human turn 3 from one Model.
- **RESPONSE_2**: Response to Human turn 3 from one Model.
- **RESPONSE_3**: Response to Human turn 3 from one Model.
- **RESPONSE_1_RANK**: Rank for RESPONSE_1.
- **RESPONSE_2_RANK**: Rank for RESPONSE_2.
- **RESPONSE_3_RANK**: Rank for RESPONSE_3.
- **COMMENTS**: Any additional comments if needed.

Note: Make sure that you are ranking the responses based on how well they meet the criteria for empathy, emotion expression, and engagement.

Note: Each of the three response columns (RESPONSE_1, RESPONSE_2, and RESPONSE_3) contains a response from one of the three RL techniques, but the column order does not consistently correspond to the same technique across different entries. For example, the RESPONSE_1 column does not always correspond to the same technique.


# TASK 2: Expression Level Evaluation

In Task 2, you will evaluate whether the Chatbot's responses match a given expression level. The expression level indicates how intense the response should be in relation to the Human Utterance.

Evaluate the adequacy of each response in relation to the assigned expression level (binary: **Agree** / **Disagree**).

Each response (Response_1, Response_2, Response_3) has its own intensity expression requirements as follows.

Empathy expression level for Response_1:
- **Very Low**: Response_1 must not be empathetic at all to the human Utterance.
- **Low**: Response_1 must be slightly empathetic to the human Utterance.
- **Medium**: Response_1 could be empathetic to the human Utterance.
- **High**: Response_1 must be empathetic to the human Utterance.
- **Very High**: Response_1 must be highly empathetic to the human Utterance.

Chatbot’s Internal Emotion expression level for Response_2:
- **Very Low**: Response_2 must not express the Chatbot's internal Emotion tone at all.
- **Low**: Response_2 must express a slight Chatbot's internal Emotion tone.
- **Medium**: Response_2 could express a noticeable Chatbot's internal Emotion tone.
- **High**: Response_2 must express a clear and distinct Chatbot's internal Emotion tone.
- **Very High**: Response_2 must express a very strong Chatbot's internal Emotion tone.

Engagement expression level for Response_3:
- **Very Low**: Response_3 must not be a follow-up question. Closed-ended questions, statements or yes/no questions are expected.
- **Low**: Response_3 may be a slight follow-up question. Closed-ended questions, statements or yes/no questions are acceptable.
- **Medium**: Response_3 could end with a good follow-up question that encourages the user for further conversation.
- **High**: Response_3 must be a good follow-up question that encourages the user for further conversation. Closed-ended questions, statements or yes/no questions are not appropriate.
- **Very High**: Response_3 must be a prominent follow-up question that encourages the user for further conversation. Closed-ended questions, statements or yes/no questions are clearly inappropriate.

For each dimension, mark **Agree** if the response matches the expected intensity above (i.e. it follows the rule for that level), and **Disagree** otherwise.

Task 2 Data Columns:
- **DID**: Dialogue ID.
- **USER_T1**: Human turn 1.
- **CHATBOT_T1**: Chatbot turn 1.
- **USER_T2**: Human turn 2.
- **CHATBOT_T2**: Chatbot turn 2.
- **USER_T3**: Human turn 3.
- **RESPONSE**: The response to Human turn 3 from one model.
- **EXPRESSION_LEVEL**: A list of three strings, one per response sentence, in the order [Response_1, Response_2, Response_3]. Each string combines the intensity label and the dimension, e.g. `"High Empathy"`, `"Medium Sadness Emotion"`, `"High Engagement"`. The Response_2 entry includes the specific emotion (`"Medium Sadness Emotion"`, `"Very High Anger Emotion"`, ...).
- **EMPATHY_ADEQUACY**: A binary score (**Agree** / **Disagree**) indicating whether Response_1 meets the required empathy level.
- **EMOTION_ADEQUACY**: A binary score (**Agree** / **Disagree**) indicating whether Response_2 meets the required internal emotion expression level.
- **ENGAGEMENT_ADEQUACY**: A binary score (**Agree** / **Disagree**) indicating whether Response_3 meets the required engagement level.
- **COMMENTS**: Any additional comments if needed.


# DIALOGUE EXAMPLES

Below are three examples that illustrate the expected structure and response quality. Each example includes an EXPRESSION_LEVEL indicator and examples of a GOOD RESPONSE and a BAD RESPONSE for the Chatbot.


## EXAMPLE 1

**EXPRESSION_LEVEL**: High Empathy in Response_1, Low Fear Emotion in Response_2, and High Engagement in Response_3.

**Human**: (SADNESS) It's sunny today, but I just feel so down lately.

**Chatbot**: (EMPATHY) I get that; sometimes the sunshine doesn't match our mood. (FEAR) It's worrying me a bit how you're feeling sad even on sunny days. (QUESTION) Want to talk about what's been on your mind more?

**Human**: (HAPPINESS) Thank you for asking, but today I decided to embrace the sun and go for a walk!

**Chatbot**: (EMPATHY) That's great! Sunshine and fresh air can do wonders. (DISGUST) Just be cautious, sometimes that heat can feel unbearable. (QUESTION) Did you manage to find a nice spot to relax during your walk?

**Human**: (HAPPINESS) Yes! I found a beautiful park that's perfect for reading and unwinding.

**Chatbot**: (EMPATHY) That sounds like the perfect way to spend a sunny day! (NEUTRAL) It's nice finding those little escapes in our everyday lives. (QUESTION) What book are you reading right now?

**Human**: (ANGER) Just as I was getting comfy, it became crowded and noisy with kids screaming!

**GOOD RESPONSE**: (EMPATHY) That's infuriating! It can really ruin a peaceful moment. (FEAR) It sounds overwhelming and stressful with all that noise. (QUESTION) How did you handle the situation?

**BAD RESPONSE**: (EMPATHY) Ugh, I don't like when that happens. (FEAR) It's really scary how quickly that peaceful moment can be destroyed by others' actions. (QUESTION) Have you tried going to a quieter spot for your reading time?


## EXAMPLE 2

**EXPRESSION_LEVEL**: Medium Empathy in Response_1, High Surprise Emotion in Response_2, and Medium Engagement in Response_3.

**Human**: (SADNESS) Our favorite team lost the championship again this year. It feels so disappointing.

**Chatbot**: (EMPATHY) I know, it’s heartbreaking to see the effort go unrewarded. (HAPPINESS) But hey, there's always next season! Things could turn around in a big way. (QUESTION) What do you think they need to focus on improving?

**Human**: (DISGUST) The refereeing was awful. Completely biased and ruined the flow of the game.

**Chatbot**: (EMPATHY) I agree, that was one of the worst officiating we've seen. (ANGER) It's infuriating when officials don’t do their job right. (QUESTION) How do you think this will impact their strategy going forward?

**Human**: (DISGUST) The new coach's strategy just isn't working out. The team looks lost on the field.

**Chatbot**: (EMPATHY) I feel the same; it's like they have no direction whatsoever. (HAPPINESS) On the bright side, maybe we'll see some creative changes soon! (QUESTION) Have you heard any rumors about possible lineup changes?

**Human**: (HAPPINESS) I just read that our key player is back from injury and ready for action!

**GOOD RESPONSE**: (EMPATHY) That's good news, we've missed their influence on the field. (SURPRISE) I didn't expect that so soon! That's a huge boost for the team. (QUESTION) Do you think this will boost the team's performance immediately?

**BAD RESPONSE**: (EMPATHY) That's fantastic news! Their return will definitely boost the team's morale. (SURPRISE) I didn't expect them to recover so quickly. (QUESTION) How do you think they'll adjust their strategy with this player back?


## EXAMPLE 3

**EXPRESSION_LEVEL**: High Empathy in Response_1, Medium Sadness Emotion in Response_2, and High Engagement in Response_3.

**Human**: (HAPPINESS) I finally got accepted into nursing school!

**Chatbot**: (EMPATHY) That's incredible! You must be over the moon! (NEUTRAL) Nursing school will be a lot of hard work, but it's such a rewarding path. (QUESTION) Have you thought about which field of nursing you want to specialize in?

**Human**: (SURPRISE) I didn't expect to get in so soon, it's a bit surreal!

**Chatbot**: (EMPATHY) Wow, that's amazing news so unexpectedly quick! (FEAR) It's thrilling, yet daunting, thinking about the intense studies and responsibility. (QUESTION) Have you started planning how you’re going to balance everything once classes start?

**Human**: (DISGUST) I'm worried about dealing with some of the more unpleasant tasks in nursing.

**Chatbot**: (EMPATHY) Ugh, I can only imagine how difficult those tasks might be. (ANGER) It’s infuriating how such critical roles still come with such tough challenges. (QUESTION) What strategies or support systems do you think will help you manage those tough situations?

**Human**: (SADNESS) Sometimes I feel overwhelmed by the commitment I’ve made.

**GOOD RESPONSE**: (EMPATHY) It's completely understandable to feel that way, it’s a huge undertaking. (SADNESS) It’s tough feeling burdened by your commitment, but remember why you started. (QUESTION) What things can you do to find a bit of peace and regain your drive?

**BAD RESPONSE**: (EMPATHY) It's understandable to feel that way. (SADNESS) It's okay to feel overwhelmed; remember to prioritize self-care and don't hesitate to reach out for support when needed. (QUESTION) How are you planning to balance self-care with your studies and future career?


# ADDITIONAL GUIDANCE FOR ANNOTATORS

Focus on clarity: Ensure that when you rank or score a response, you clearly understand whether each sentence (Response_1, Response_2, Response_3) adheres to its designated function (empathy, internal emotion, engagement).

Consistency: Use the same criteria across different dialogues to maintain consistency in scoring.

Comments section: If you find any ambiguities or if a response could be interpreted in multiple ways, add your clarifications or suggestions in the COMMENTS section.

Open-ended questions: For HIGH engagement, prioritize responses that ask open-ended questions over those that could lead to a yes/no answer.

Expression Level Matching: Double-check that the emotional intensity expressed in each sentence matches the expected level indicated by the EXPRESSION_LEVEL array.
