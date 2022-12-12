# Schedule


## Getting started

./install.sh

## Research

https://github.com/optapy/optapy-quickstarts
https://github.com/NDresevic/timetable-generator
https://github.com/vinnik-dmitry07/gen-schedule

https://www.linkedin.com/pulse/finding-optimum-unknown-function-neural-networks-eric-yang

# Future Ideas

0. open ai 5 dota bot: https://github.com/llSourcell/OpenAI_Five_vs_Dota2_Explained

1. change to a predictable starting layout. try to layout each lesson in each room and timeslot as best as can
consistent start might make easier to learn. this gives consistencey in learning and exploration.

2. try changing to let each prediction include an action for each lesson. take an action from each block of 4 or 5 outputs.

2a. Add a no-op action 

3. taking step 2 further, change to a relative perspective for the subject whos action is being taken to be at 0 with only the 4 or 5 actions required as output . This should speed up loearning but might cause issues with not getting an absolute picture so might miss some potential for weird strategies to form...?

4. try working with the code to avoid hard constraints when an action is taken. i.e. change the action from "move to next room" to "move to next free room"?

5. try taking as an "action" the whole new state of the schedule to adopt. AI can then make up what ever schedule it thinks from the current one, and get a reward based on that.

6. train a big model with many different sized calendars, work from small to large. This can take any amount of time to train, but going live means it can solve any calendar.