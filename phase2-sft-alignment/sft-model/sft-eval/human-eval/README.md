---------
OVERVIEW
---------
The emotional dialogues between a USER and a CHATBOT consist of a PROMPT and a RESPONSE. The RESPONSE is always composed of 3 sentences. Each dialogue is about a different topic. The USER and the CHATBOT may have the same or different emotions.

USER PROMPT:
Utterance containing a specific emotion.
MODEL RESPONSES:
SENTENCE 1: First sentence that must be empathic with respect to the PROMPT.
SENTENCE 2: Second sentence expressing the chatbot's internal emotion.
SENTENCE 3: A question to encourage the USER to continue the conversation, trying to avoid yes/no answers.

--------
EMOTIONS
--------
There are 7 possible emotions:
[A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise and [G] Neutral.

ANGER: a strong feeling of annoyance, displeasure, or hostility. Example: "she could barely restrain her anger at this comment".
DISGUST: a feeling of revulsion, aversion or strong disapproval aroused by something unpleasant or offensive. Example: "the sight filled her with disgust".
FEAR: an unpleasant emotion caused by the belief that someone or something is dangerous, likely to cause pain, or a threat. Example: "farmers fear that they will lose business"
HAPPINESS: the state of being happy. Example: "she struggled to find happiness in her life"
SADNESS: the condition or quality of being sad. Example: "a source of great sadness"
SURPRISE: an unexpected or astonishing event, fact, or thing. Example: "the announcement was a complete surprise"
NEUTRAL: having no strongly marked or positive characteristics or features. Example: "the tone was neutral, devoid of sentiment"

--------------------
TASKS AND EVALUATION
--------------------
We are going to evaluate all these features of the dialogue through different taks. In all files, each column corresponds to a different dialogue. In MODEL<X>_RESPONSE_EMOTION, X corresponds to a different model.

Task1:
Empathiness evaluation of the response provided by the different chatbots given the previous turn from the user.
Rank the different responses returned by each model (MODEL<X>_EMPATHY_QUALITY) to the user's statement (USER_PROMPT).
The best answer will be ranked as 1 and the worst answer as 5.

Task2:
Select the emotion expressed by the user (USER_PROMPT column). In addition, note the emotion provided by each model (MODEL<X>_RESPONSE column) taking into account the user's previous sentence.
There are 7 possible emotions available in the USER_EMOTION and MODEL<X>_RESPONSE_EMOTION columns.

Task3:
Evaluate the adequacy of the follow-up question formulated by the models considering the previous turn from the user.
As in Task 1, rank the different responses returned by each model (MODEL<X>_QUESTION_QUALITY) to the user's statement (USER_PROMPT).
The best answer will be ranked as 1 and the worst answer as 5.

Task4:
Give an overall rate of each response provided by the models, considering the previous user turn.
Rate all model responses (MODEL<X>_OVERALL_QUALITY) of each model from 1 to 5, with 1 being a bad response and 5 being a good response.

In case you encounter any problem, you can write it in the COMMENTS column.

-----------------
DIALOGUE EXAMPLES
-----------------
USER (PROMPT)
FEAR - Yes, thank you so much for the encouragement. But I'm still nervous about sharing my work with others.
CHATBOT (RESPONSE)
EMPATHY - It's natural to be nervous, but think of how fulfilling it will be to share your words with the world.
DISGUST - And don't listen to the critics who don't understand the work you put in. Their opinion is not your truth.
NEUTRAL - Do you have any trusted friends or family members who can give you feedback on your work?

USER (PROMPT)
ANGER - The Assyrians had a reputation for extreme cruelty towards those they conquered.
CHATBOT (RESPONSE)
EMPATHY - It's true, their conquests often involved the destruction of entire cities.
FEAR - It's terrifying to think about the scale of their violence and how it impacted the lives of so many.
NEUTRAL - Have you ever visited the ruins of an ancient Assyrian city?

USER (PROMPT)
HAPPINESS - Guess what, I just got engaged!
CHATBOT (RESPONSE)
EMPATHY - Congratulations! That's wonderful news!
FEAR - I'm so happy for you, but I must admit, wedding planning sounds terrifying.
NEUTRAL - Have you set a date yet?

USER (PROMPT)
HAPPINESS - We're thinking about next summer, and it's so exciting, even though we haven't set a date yet.
CHATBOT (RESPONSE)
EMPATHY - Summer weddings are beautiful!
NEUTRAL - Although, wedding planning can be challenging, try to maintain a calm approach.
NEUTRAL - Have you picked a venue yet?

USER (PROMPT)
SADNESS - It's a bit overwhelming. We've started browsing, but there are so many options, we have thought about a wedding planner.
CHATBOT (RESPONSE)
EMPATHY - I completely understand the overwhelm.
SURPRISE - That's a fantastic idea! I hadn't even thought about hiring a wedding planner.
NEUTRAL - Do you have any ideas for the theme or venue?

USER (PROMPT)
DISGUST - We're leaning towards a beach wedding, but they're too expensive.
CHATBOT (RESPONSE)
EMPATHY - It's true, beach weddings can get pricey.
DISGUST - It's normal for the cost to put you off.
NEUTRAL - Have you considered other scenic outdoor options that might be more budget-friendly?

USER (PROMPT)
DISGUST - I'm really disgusted with all these pop-up ads online!
CHATBOT (RESPONSE)
EMPATHY - They're so intrusive and annoying.
HAPPINESS - I have just the thing! There is this great ad-blocker I have been using that works wonders.
(NEUTRAL) Have you tried an ad-blocker before?

USER (PROMPT)
FEAR - I've always dreamed of visiting an island in the South Pacific, but I'm afraid of flying over the ocean.
CHATBOT (RESPONSE)
EMPATHY - I can understand why that would be scary.
ANGER - It's frustrating when fear holds us back from doing things we really want to do.
NEUTRAL - Have you thought about taking a cruise instead?

USER (PROMPT)
ANGER - I had a terrible experience with this one cleaner though. She broke a glass and didn't tell me about it.
CHATBOT (RESPONSE)
EMPATHY - That's unacceptable, did you confront her about it?
SADNESS - I'm so sorry you had to go through that, I hope you can find a better cleaner.
NEUTRAL - Have you considered switching services or reaching out to the company about the issue?
