# Schedule


## Getting started

./install.sh

## Research

https://github.com/optapy/optapy-quickstarts
https://github.com/NDresevic/timetable-generator
https://github.com/vinnik-dmitry07/gen-schedule

https://www.linkedin.com/pulse/finding-optimum-unknown-function-neural-networks-eric-yang

# Future Ideas

-1. Instead of a single score, why not a score per cell? let the ai try to maximize the per cell, then provide an action per cell. I might need to break out of the gym structure for this

0. open ai 5 dota bot: https://github.com/llSourcell/OpenAI_Five_vs_Dota2_Explained
ppo: https://openai.com/blog/openai-baselines-ppo/

1. change to a predictable starting layout. try to layout each lesson in each room and timeslot as best as can
consistent start might make easier to learn. this gives consistencey in learning and exploration.

1a. if we can teach a model to any schedule up to N lessons in X rooms in Y timeslots ... i.e. the single model can handle
a schedule with 5 lessons in 2 rooms and 3 timeslots as well as 10 lessons in 10 rooms annd 20 timeslots, then
step #1 is somewhat pointless as the model then becomes: "just train this sucker real good, and use just generate schedules as required". Train a big model with many different sized calendars, work from small to large datasets. This can take any amount of time to train, but going live means it can solve any calendar.

1aa. allow consecutive training by loading models at the start of a training session, save it out after. then do an evaluation to ensure its improved.

2. try changing to let each prediction include an action for each lesson. take an action from each block of 4 or 5 outputs.

2a. Add a no-op action 

3. taking step 2 further, change to a relative perspective for the subject whos action is being taken to be at 0 with only the 4 or 5 actions required as output . This should speed up loearning but might cause issues with not getting an absolute picture so might miss some potential for weird strategies to form...?

4. try working with the code to avoid hard constraints when an action is taken. i.e. change the action from "move to next room" to "move to next free room"?

5. try taking as an "action" the whole new state of the schedule to adopt. AI can then make up what ever schedule it thinks from the current one, and get a reward based on that.

# Learning

1. Simple deep q learning is faster than double soft q learning ~50% but is more unstable. For the schedule problem that might not be a problem as we just want to first solution that is valid.

2. Simply supplying the state and expecting a good action doesn't help the model know where the probblem is that it is trying to solve. It also means the model is prone to repeat its actions in a loop until randomnness breaks it out of it.

# A typical school problem

Our timetable is constructed into a 35 lesson week. An additional timeslot is provided for assemblies and ‘community’ lessons on Friday and Monday, with 7 lessons Tuesday-Thursday. It is typically broken up as 2 lessons – recess – 3 (or 4) lessons – lunch – 2 lessons. Typically, a subject line is 5 lessons composed of 2 double lessons and a single lesson each week.

There appear to be 33 rooms, including specialist rooms. That is a can of worms, as some of those rooms may only be used for half a week as there are no other classes able to use those rooms.

There would be about 15 separate subjects, or groupings of subjects. Each student does a range of subjects, according to the year level they are in. Each subject has between 2-5 lessons per week.

There are about 40 teachers, some of these are part time, others are full time.

The common constraints are…

    Having classrooms available for specific classes
    Having classes grouped together at the same time (ie all year 9 maths classes taught at the same time by 4 different teachers)
    Having the ability to vertically align classes so that students can study a year level above what they are in (ie a year 8 students studying year 9 maths)
    Part time teachers having allocated days off (ie someone who is 0.6FTE shouldn’t be teaching 5 days a week)

We use a program called ‘General Access’. It is also referred to as timetabler. The graphical interface looks like it has come from Windows XP.
