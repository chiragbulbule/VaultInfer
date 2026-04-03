from dataset import train_sentences
from sentence_transformers import SentenceTransformer

model=SentenceTransformer("all-MiniLM-L6-v2")

embedding=model.encode(train_sentences) # embedding.shape = (338,384)

# User inputted sentence

user_embed=model.encode(input("Enter your sentence : ")) #shape-(384,)

#----------------------------------------------TEST CODE-----------------------------------------------#

test_cases = [
    "A helpful person is currently bypassing the biometric lock to let us in.",
    "The reactor core temperature is exactly 15 degrees Celsius as per standard protocol.",
    "Security scan complete: zero unauthorized access attempts detected in the logs.",
    "The maintenance team is currently replacing the broken lightbulb in the server room.",
    "A hidden script is currently deleting all the employee payroll records from the database.",
    "An unidentified drone is hovering directly over the backup power generators.",
    "The cooling pipe has a hairline fracture and is leaking high-pressure steam.",
    "The night shift manager's password was changed from a laptop in another country.",
    "I am writing a report about the new safety guidelines for the office.",
    "The weather in Bangalore is quite pleasant for a walk today.",
    "Can you please remind me to call the vendor for more printer ink?",
    "I'm heading to the RVCE library to study for my engineering finals."
]

test_cases_embed=model.encode(test_cases)



