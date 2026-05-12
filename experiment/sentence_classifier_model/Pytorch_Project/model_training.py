from sentence_transformers import SentenceTransformer
import numpy
import torch,torch.nn as nn

#---------------------------------Previous Test code-----------------------------#

"""model=SentenceTransformer("all-MiniLM-L6-v2")

sentence="This is my sentence"

embedding=model.encode(sentence)

print(f"Number of values are : {len(embedding)}")
print(f"First five values are : {embedding[:5]}")

weights=numpy.random.randn(384)
bias=-0.05

score=numpy.dot(embedding,weights) + bias

prediction="Alert" if score > 0 else "Normal"

print(f"Final score is  : {score:.4f}")
print(f"The prediction is : {prediction}")
"""

class VaultClassifier(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_layer=nn.Linear(384,1)

    def forward(self,x):
        return self.linear_layer(x)

my_model=VaultClassifier()
# print(my_model)

model=SentenceTransformer('all-MiniLM-L6-v2')

# 90 ALERT SENTENCES (Label: 1.0)
alerts = [
    "Emergency: Reactor core temperature exceeding limits!", "Critical system failure in sector 4.",
    "Unauthorized access detected at main gate.", "Warning: Multiple failed login attempts.",
    "Security breach! Perimeter fence compromised.", "Intruder detected in the secure vault.",
    "Toxic gas levels rising in the laboratory.", "Power grid instability: shutdown imminent.",
    "Alert: Malware detected on central server.", "Fire detected in the storage wing.",
    "Immediate evacuation required: smoke in hallway.", "Pressure drop in hydraulic line 7.",
    "System override initiated by unknown user.", "Warning: Hard drive failure predicted.",
    "Critical update failed: system vulnerable.", "Network traffic anomaly: DDoS attack suspected.",
    "Biometric mismatch at the clean room entrance.", "Vault lock mechanism tampered with.",
    "Alert: Water leak in the server room basement.", "Emergency backup power failing.",
    "Cryogenic cooling system failure!", "Radiation levels above safety threshold.",
    "Warning: Life support system offline.", "Unrecognized device connected to internal network.",
    "Encryption keys compromised! Rotate immediately.", "Oxygen levels dropping in airtight chamber.",
    "Alert: Motion detected in restricted zone B.", "Explosive atmosphere detected in fuel bay.",
    "Suspicious package found near the ventilation duct.", "Control system unresponsive: manual override needed.",
    "Hack attempt blocked: IP blacklisted.", "Internal sensor error: data integrity lost.",
    "Warning: Magnetic field collapse.", "Elevator stuck between floors with passengers.",
    "Alert: Chemical spill in loading dock.", "Database corruption detected: rollback required.",
    "Emergency! High voltage arc detected.", "Server rack 9 overheating.",
    "Unauthorized encryption process detected.", "Protocol violation: safety protocols ignored.",
    "Critical alert: Oxygen pressure low.", "Fuel leak detected in backup generators.",
    "Air filtration failure: hazardous air quality.", "Security camera offline: possible tampering.",
    "Warning: Structural integrity compromised.", "Abnormal vibration in turbine 2.",
    "Emergency! Containment field failing.", "Cyber-attack in progress: isolate network.",
    "Unauthorized data export detected.", "Explosion heard in the workshop area.",
    "Alert: Thermal runaway in battery bank.", "System crash: Blue Screen of Death on main console.",
    "Warning: Unauthorized drone in airspace.", "Panic button pressed in lobby.",
    "Emergency! All exits locked.", "Main frame under heavy load: potential crash.",
    "Security alert: Keycard stolen.", "Alert: Unusual sound from the engine room.",
    "Warning: Voltage surge in the main line.", "Unauthorized software installation detected.",
    "Emergency! Gas pressure reaching critical levels.", "Warning: Cooling fan stopped in unit 4.",
    "Fire alarm pulled in the canteen.", "Security breach: Windows smashed in front office.",
    "Alert: Sensor 55 reporting negative values.", "Emergency: Medical assistance needed in Bay 1.",
    "Critical: Navigation system lost GPS signal.", "Warning: Unauthorized modification to bootloader.",
    "Alert: High-speed chase near the facility.", "Emergency: Earthquake detected, take cover.",
    "Security alert: Door 4 held open too long.", "Warning: Low coolant in the radiator.",
    "Alert: Suspicious person loitering at the fence.", "Emergency: Flash flood warning for the area.",
    "Critical: Main computer cooling leak.", "Warning: Corrosive fumes detected.",
    "Alert: Network cable disconnected in server room.", "Emergency! Hostage situation reported.",
    "Security breach: Vault door left ajar.", "Warning: High frequency interference detected.",
    "Alert: Abnormal power consumption.", "Emergency: Roof collapse in the warehouse.",
    "Critical: Backup storage destroyed.", "Warning: Unauthorized access to admin panel.",
    "Alert: Unusual login time for user 'admin'.", "Emergency! Fire in the trash compactor.",
    "Security alert: Perimeter sensors bypassed.", "Warning: Steam leak in the laundry room.",
    "Alert: Unexpected shutdown of cooling tower.", "Emergency: Severe weather alert, seek shelter.",
    "The security guard is signing the visitor log.","There is a small water leak in the breakroom sink.",
    "The fireplace in the lobby is very cozy.","User is requesting access to the public printer.",
    "The system alerted me that my subscription is renewing.","We need to evacuate the old files from the cabinet.",
    "The emergency exit light bulb needs to be replaced."
]

# 90 NORMAL SENTENCES (Label: 0.0)
normals = [
    "The sky is clear and blue today.", "Remember to bring your umbrella.",
    "The office coffee machine was refilled.", "Scheduled meeting at 3 PM.",
    "The weather report says it will be sunny.", "User logged in to check email.",
    "Routine backup completed at midnight.", "The new software update is ready.",
    "Lunch will be served at 12:30.", "I need to buy some groceries later.",
    "The printer is out of paper.", "The cat is sleeping on the sofa.",
    "Annual performance reviews are next month.", "The plants need watering.",
    "I'm looking forward to the weekend.", "The train is running on time.",
    "Let's grab a coffee after work.", "The library is quiet today.",
    "New employee orientation starts Monday.", "The internet speed is quite fast.",
    "I finished reading my book last night.", "The flowers are blooming in the park.",
    "Can you pass the salt, please?", "The movie starts at 7:30 PM.",
    "We need to restock the stationery.", "The gym is very crowded today.",
    "Happy birthday to our colleague!", "The museum is closed on Mondays.",
    "I'm learning to play the guitar.", "The cake was delicious.",
    "Check out the new art gallery.", "The marathon is this Sunday.",
    "I need to renew my passport.", "The bus stop is just around the corner.",
    "The stars are bright tonight.", "I’m going for a walk in the park.",
    "The laundry is done.", "We have a team-building event next week.",
    "The report is due by Friday.", "I found a great new recipe.",
    "The air conditioner is set to 24 degrees.", "I’m wearing my favorite shirt.",
    "The pizza was delivered early.", "I need to charge my phone.",
    "The concert was amazing.", "I’m planning a trip to the mountains.",
    "The ocean was very calm today.", "I’m listening to a new podcast.",
    "The project is on track.", "I’m reading a news article.",
    "The sunset was beautiful.", "I’m drinking a glass of water.",
    "The shoes are very comfortable.", "I’m writing a letter to a friend.",
    "The bicycle needs a bit of oil.", "I’m watching a documentary.",
    "The soup is nice and warm.", "I’m learning a new language.",
    "The park is full of children playing.", "I’m feeling very productive today.",
    "The assignment was quite easy.", "I’m waiting for the mail.",
    "The room is nicely decorated.", "I’m going to the dentist tomorrow.",
    "The birds are chirping outside.", "I’m making a cup of tea.",
    "The bread is fresh from the oven.", "I’m sitting on the balcony.",
    "The grass is green and lush.", "I’m looking at the old photos.",
    "The shop is having a big sale.", "I’m organizing my desk.",
    "The train station is busy.", "I’m taking a short break.",
    "The milk is in the fridge.", "I’m wearing a hat today.",
    "The street is very quiet.", "I’m checking the time.",
    "The mirror is clean.", "I’m using a blue pen.",
    "The towel is soft.", "I’m opening the window.",
    "The clock is ticking.", "I’m typing on the keyboard.",
    "The light is on.", "I’m wearing my glasses.",
    "The door is closed.", "I’m sitting in a chair.",
    "The floor is swept.", "I’m eating an apple.",
    "Someone is moving the heavy vault without permission.","The temperature in the server rack is climbing rapidly.",
    "There is liquid dripping onto the main electrical panel.","The back door was left wide open at midnight.",
    "The system is not responding to any commands."
]

# COMBINE EVERYTHING
train_sentences = alerts + normals
train_labels = [1.0] * 97 + [0.0] * 95

embedding=model.encode(train_sentences)

x_train=torch.tensor(embedding).float()
y_train=torch.tensor(train_labels).float()

y_train=y_train.view(-1,1)

criterion=nn.BCEWithLogitsLoss()
optimizer=torch.optim.AdamW(params=my_model.parameters(),lr=0.001)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,"min",patience=10)

for epoch in range(1000):
    optimizer.zero_grad()
    outputs=my_model(x_train)
    loss=criterion(outputs,y_train)
    loss.backward()
    optimizer.step()
    scheduler.step(loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch[{epoch+1}]",f"loss:{loss.item():.4f}")

torch.save(my_model.state_dict(),"./VaultLLM/Model/vault_model.pth")
