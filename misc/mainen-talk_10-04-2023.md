# Zach Mainen

Serotonin and latent state regulation

- 5HT projects widely throughout the brain
- 14 receptor subtypes

### slow behavioral state tracking of serotonin neurons
- tracking sleep/wake cycle
- Ranade and Mainen, 2008: serotonin neurons not just slow modulatory thing. 
- Serotonin was marketed as a reward signal because of SSRIs but that seems to be made up

### reversal learning task
- Matias et al 2017
- associated a bunch of cues with outcomes. then change the outcome:cue map. 
- big serotonin responses after the reversal
- kinda looks like surprise / unsigned prediction error

### visual association task
- pair two images in sequence with reward / no reward
- 4 different images total
- how do DR neurons resond to novel stimuli, familiarization, late training?
- Question: how do you disentangle effects of serotonin on movement vs reward? Hard to do this in general since there's a bunch of feedback loops. E.g., mice tend to anticipate reward by slowing down.

### serotonin resposnes to images
- responds strongly to novel stim
- decreases with fmailiarity
- increases with learning as stim become "meaningful" (outcome predictive)

### hypothesis: serotonin as a prediction error in unsupervised learning
- novelty unaccounted by RL: responses to unpredicted events in absence of reward

### Consider a latent variable model / autoencoder
- e.g. Sequential VAE. Gives you a latent space prediction error. Is that like the update step in a Kalman filter?  
- Could potentially account for responses to novel stimuli.
- Also adaptation as learning updates world model and makes more predictive
- Selectivity for predictive stimuli? 

### what would you do with broadcast latent state prediction erors?
- is the unsupervised learning signal too low dimensional?
- learning/control systems arbitration

### Serotonin stimulation with large scale recordings
- stimulate optogenetically
- neuropixel recording through many brain regions
- using HMMs to cluster population brain states based on firing rates 
- over trials, you can see effects of serotonin stim on state transition probabilities

### questions
- does serotonin synchronize brain states?
- learning not only the rules but how to compartmentalize the rules???
- don't want to create a new context every time you see a slightly new world?
- depression / schizophrenia perhaps arise from malfunctions of state inference and switching?