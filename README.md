# Hello! Here are a few Kaggle competitions (the ones still publicly available) that I've participated in during my ~~dreadful~~ time at Cornell University.


1. Spoken Digit Pair Recognition

Link: https://www.kaggle.com/competitions/spoken-digit-pair-recognition

Leaderboard name: Jayadev

- This one was fun, but I spent way too long on it. 😅 I trained an 11-layer Convolutional Neural Net (CNN), which is surprisingly good for predicting audio 🎧. Some context: I was just an undergraduate junior at the time, taking some cancerous master's course with a bunch of PhD students 🤯. They wanted us to use PyTorch apparently, but since I don't go to class, I ended up using TensorFlow 🤫. Anyway, I gaslighted everyone into thinking that the professor was in the competition by using his name and picture as my profile pic on the leaderboard 📉.  I ended up using ensemble methods to learn the missing label to achieve 100% test accuracy.  

2. Named Entity Recognition

Link: https://www.kaggle.com/competitions/cs-4740-fa22-hw1-named-entity-recognition/leaderboard

Leaderboard name: Coin Flip Predictor

- I used a transformer architecture for this one 🤖. Why would you use anything other than the best NLP model? 🤔 They wanted us to leverage Hidden Markov Models (HMM), but I told them that school is a scam 🎓, and we're not living in 1966 anymore 👴.

3. Force Plate Gait Analysis (Binary Classification)

Link: https://www.kaggle.com/competitions/cs4780-spring-2023-kaggle-competition/leaderboard

Leaderboard name: cg528

- This one was a joint collaboration between me and my friend Christy. One day Christy texted me about her class Kaggle competition and I was like "SGTM, I'm in" 🤙. The funny thing is that I was never in the class, but the rules of the competition state that: 
*You may only work in groups of up to 2 students.*
After (poorly) interpreting the legal documnetation, I concluded that I was a student at Cornell, so I decided to come out of Kaggle retirement and explore this headache inducing problem. We weren't sure the exact format of the data, as the TA **refused** to tell Christy - this is a crime because it is important to know what kind of data you're working with (whether it's time series, images, etc). 😅 Anyway, we used set reduction to determine best features, made a training mask (no imputation of missing values), and stacked/aggregated predictions (ensemble methods) for **every individual label** - the dataset was small enough that we were computationally able to do this. I guess we did well enough to the point where the vet school asked us for the model - which we forgot to provide, oops! 😳 Since I'm all for open source and helping out dogs, here it is (in case you happen to be apart of the vet school and are reading this) 🙌 !






## Lessons learned from Kaggling

- Spend the majority of time during competition on messing around with data.  It's a  worthy investment of your time.  Open interactive shell and play with the numpy/pandas raw data until it makes sense.

- It's not enough to "work hard" - remove the mentality that you need to reach the maximum submissions per day.  These people hardly win in my experience.  There needs  to be some finesse in what you are doing to solve the problem.

- Don't do the most obvious thing.  Everyone is going to load the data and plug it into a model, and repeat with a different model.  That's the naïve approach.  

- Don't use the public leaderboard to validate your model - it will overfit!