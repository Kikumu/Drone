# Reinforcement-Learning-repo
A repo where I **look into the details** and **practice** on how to properly utilize reinforcement learning. I try out different AI models and evaluate their performance
to test out which does best. Please note this is a project i do in my free time and will update periodically when i get the time to complete it!

*Please note - running current model on cpu will be mind numbingly slow. I highly recommend either reducing network size (i.e parameters/layers/number of hidden layers) and reducing batch size(if running on cpu). Running on cuda cores is highly recommended*

## Phases(base line alg and setup).
1) Classic control(1/3): pendulum(solved), mountain car, mountain car continuous.
2) Mujoco (0/10)
3) Robotics (0/10)

## Model(s) used/Evaluation metrics.
### Classic control.

#### Pendulum(TD(0)) - Doesnt work.

Episodes trained | #- | #- 
--- | --- | --- 
Average/Overall reward(per 200 eps) | - | - 

#### Pendulum(Policy grad monte carlo) - Doesnt work.

Episodes trained | #- | #- 
--- | --- | --- 
Average/Overall reward(per 200 eps) | - | - 


#### Pendulum(DDPG) - Works like a charm.

Episodes trained | #300 | #- 
--- | --- | --- 
Average/Overall reward(last 200 eps) | - | - 


#### Mountain car(LSTM + 2 FC LAYER).

Episodes trained | #- | #- 
--- | --- | --- 
Average/Overall reward(per 200 eps) | - | - 

#### Mountain car(Convnet + 2 FC LAYER).

Episodes trained | #- | #- 
--- | --- | --- 
Average/Overall reward(per 200 eps) | - | - 


#### Mountain car cont.(LSTM + 2 FC LAYER).

Episodes trained | #- | #- 
--- | --- | --- 
Average/Overall reward(per 200 eps) | - | - 

#### Mountain car cont.(Convnet + 2 FC LAYER).

Episodes trained | #- | #- 
--- | --- | --- 
Average/Overall reward(per 200 eps) | - | - 




# Credits.

All credits goes to deeplizard. Visit their yt channel here: https://www.youtube.com/channel/UC4UJ26WkceqONNF5S26OiVw

# Other resources.
1) https://arxiv.org/abs/2005.14419
2) https://d3c33hcgiwev3.cloudfront.net/Ph9QFZnEEemRfw7JJ0OZYA_808e8e7d9a544e1eb31ad11069d45dc4_RLbook2018.pdf?Expires=1624492800&Signature=KJBvWJSHe5fr1OvTQWuLDGKdaTUFcmQe9EwvOYhIhfmPfQaQzPJZr4NWGDKgGEjKrHSrBMbmlabY2-AcuORcoUSBB4522vSZipuZb3LXlg552LU4NUI50-G55oKxgX2mGe3Z5X2pQA1QDV1ijBdz1bCZNvnlpPgTz6Oz8iOU4WQ_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A
3) https://arxiv.org/abs/1509.02971
