# ATS-AI-Driver
This was an attempt to create a self-driving model to play American Truck Simulator

Video of it progressing...ish : https://youtu.be/_vVOMTsVJwI
Video of the most recent version : https://youtu.be/RmVj31C7lfU

Overview:
I learned a lot about training AI models and how many variables there are to consider when trying to improve it. I now understand why companies are buying so many GPUs :)

How it works:
This is imitation learning. The collection related scripts take a screenshot of the game and preprocess that image, this includes cropping, greyscaling and scaling. The camera angle is the interior angle and fixed in place allowing for a consistent view of the windscreen. All of this data is collected into a repository in WSL(The data totaled 25gb). The training is done in WSL. The neural net takes a stack of 4 frames and the current steering and throttle and gives a predicted steering and throttle.
The play related scripts run the model and give it the game state and tracks an internal steering and throttle state and keypresses based on the predicted values.

Issues with tensorflow:
I intially tried to use GPU tensorflow in windows to avoid having to use WSL, due to using the game in windows. Eventually it was too annoying so I figured out how to use WSL and did the training there.

Progression:
A constant issue throughout was whether the bad driving was due to lack of data, bad logic, bad interfacing with the game or a bad neural net architecture. Eventually training was taking too long for rapid iteration, increasaing the model size made training take even longer and collecting data took a while. Input logic was changed often and ended up being more logical with some oversights e.g. if steering goes from 0.8 to 0.4, but 0.4 is still above steering right threshold cauing the model to steer right but the truck is clearly steering left.

I wanted to use the same data but extract more information. Gemini suggested flipping each image to prevent constantly steering in a single direction off the road, I then also added blurred and sharpened images. Im unsure whether that improved it but it seems like a good idea for generality.

The model didn't always take the current_steering and current_throttle, this was added later, when i realised. This meant needing to track the state. I hadn't figured out how to get the exact data values from the game itself meaning I needed to guess how the steering values should be measured based on keyboard inputs.

I realised that instead of collecting data anywhere, it would make better sense to let the AI drive and whenever I intervene switch to collecting data. Allows for the model to better recover itself.

I eventually switched to training on a single road in the hopes of seeing meaningful progress and to keep better track of where the failures were (on that road it was a few sharp bends).  

Reflection, future ideas:
I at some point uncapped the frame rate of the model. This was an attempt to give it quicker reaction times to avoid weaving in the lane. I think this may have removed a lot of temporal understanding as the frames may happen in too quick succession causing not much different between them. Maybe this would have been good in a *much* larger much with larger frame stack but 4 frame stack maybe not.

This project made me really realise why companies are increasing compute so much. So many issues can be solved with more compute.

Future ideas include inputting the navigation in the game to the model and including wing mirrors. Better tracking using controller inputs and a python script inbetween the game and controller may allow for better tracking of steering_angle (I made the script just didn't try training any models, but the idea seems good). 
