# Explaining Voice

This application was initially created for the University of Leeds Be Curious 2022 event. The application has five different tabs:

1. Record Audio
This tab allows for participants to record their voice, view the waveform and some features. By clicking the add button the recording is added to the database and the age of the participant is predicted using a decision tree. Every five participants a new model is trained.
2. Pitch
This tab allows participants to see the changes made to a waveform when the pitch and volume is changed. Using two sliders the pitch and volume of a sine wave can be changed, this can be heard and seen.
3. Accuracy
This tab shows how the accuracy of the model changes as more recordings are taken. It also displays the number of people in each age group for the initial model and the number of new people in each age group. 
4. Beat the computer
This tab is a game to see if participants can beat the computer in guessing whether recordings are boys or girls. By pressing play file, a person saying /a/ will be played, the participant can then guess whether that person is a boy or a girl and the computer's guess is also shown alongside the correct answer. The score is tallied on the right side of the screen. 
5. Beat the computer - hard
This tab is a game to see if participants can beat the computer in guessing how old people are from recordings. By pressing play file, a person saying /a/ will be played, the participant can then guess which age category that person fits into, the computer's guess is also shown alongside the correct answer. The score is tallied on the right side of the screen. 

The initial model and the guessing games use recordings from the Saarbruecken voice dataset which can be found here: http://www.stimmdatenbank.coli.uni-saarland.de/help_en.php4

There are two versions of this application, the first was used at the Be Curious 2022 event, the second is an improved version created based on experience at the event.
