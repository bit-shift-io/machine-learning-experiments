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
https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8


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

3. Still hitting a limit the algorithm is not learning past. 
Is it that the problem is not solvable? OR is it that the random exploration is not good?
So why not try a array of what action is tried at each step. then when we want t new random action,
we increment to try the next action, so exploration is no longer random, but planned out


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




You say there are 15 subjects per which I have assumed are just 5 lessons per week, which would give me a need for 75 time slots per week, which doesn't fit into the 35 per week (7 per day).

I have assumed you have some kind of elective system to deal with this?

Yes, you are correct, it does run off an elective scheme, it also runs such that there is non-equal allocation of lessons per subject that varies from year to year.

 

Year 7: 5 lessons each for English, Maths, Science, Humanities, 3 lessons each for German, PE, 2 for Christian living. That is 28 lessons so far. 2 for an arts elective line (music or drama, the class is split in to and will do one semester of each, but essentially requires 2 class rooms with 2 teachers for the year), then there is another elective line which has 5 lessons. This is split between design tech, home ec, art and digital tech. They do digital tech / art for a term each, then home ec and tech for term each. The digital tech/art and home ec/design tech semesters swap over with year 8 (as year 7 digital tech/art happens, year 8s do home ec/design tech, then they switch over). So you would need to be running this elective at the same time as a year 8 class. (there are 3 classes per year level).

 

Year 8: Essentially mirror what happens with year 7, as far as I can see. (there are 3 classes per year level).

 

Year 9: 5 lessons each for Maths, English, Science, Humanities, elective A, elective B, then 3 for PE, 2 for Christian living. Elective A and B basically, students choose from a pool of available options, there are typically 6 choices that all year 9 students have access to at the same time. The rest of their core subjects are done with the same class (there are 3 classes per year level).

 

Year 10: 5 lessons each for Maths, English, Science, Elective A and Elective B, 3 for PE and PLP (the most useless subject in existence, but a SACE requirement), then 4 for history or geography (a semester of each). Operates off a similar premise as other years. (there are 3 classes per year level).

 

Year 11: 7 lines with 5 lessons. One of these lines is split, 3 lessons of research project and 2 lessons of Christian living. One line is a study line (they share this with year 12s in the same room with the same teacher, class size can be as high as the mid 40s dependent on subject combinations).

 

Year 12: 7 lines with 5 lessons. One of these lines is split, 3 lessons of Christian living and 2 lessons as early departure. Two lines are a study line (they share this with year 11s in the same room with the same teacher, class size can be as high as the mid 40s dependent on subject combinations). 