"""
Hold-out check for the settings tuned on ablation_memory.py's dataset
(RECENCY_WEIGHT 0.15 -> 0.05, ANSWER_GATE 0.45 -> 0.62, conflict-judge prompt).

A fresh, independently written dataset (20 facts, 8 updates, 3 additive probes,
20 never-stored questions, 30 distractors) is run through three configurations:

  old     recency 0.15, gate 0.45        (the configuration before this work)
  new     recency 0.05, gate 0.62        (production now)
  sim     similarity only                (reference ceiling for plain recall)

Nothing here was used to choose the new values. Run: python ablation_holdout.py
"""
import ablation_memory as A

A.FACTS = [
    ("My locker number at the gym is forty two", 30, "Which locker is mine at the gym?", "forty two"),
    ("I deposit my pocket money in the post office savings account", 90, "Where do I keep my savings?", "post office"),
    ("My little cousin Tanvi is afraid of dogs", 60, "Who is scared of dogs?", "tanvi"),
    ("I always carry a steel bottle to college", 120, "What bottle do I take to college?", "steel bottle"),
    ("The chemistry lab is on the third floor of block C", 14, "Where is the chemistry lab?", "third floor"),
    ("I get my haircut at the salon near the post office", 45, "Where do I go for a haircut?", "salon"),
    ("My aunt Sunita sends pickles every Diwali", 150, "Who sends pickles at Diwali?", "sunita"),
    ("I plan to buy a second hand scooter after graduation", 35, "What vehicle do I plan to buy?", "scooter"),
    ("The robotics club meets on Tuesday evenings", 21, "When does the robotics club meet?", "tuesday"),
    ("I take vitamin D tablets every Sunday", 100, "When do I take vitamin D?", "sunday"),
    ("My favourite poet is Narmad", 200, "Which poet do I like?", "narmad"),
    ("I left my umbrella at the library desk", 5, "Where did I leave my umbrella?", "library desk"),
    ("I am allergic to dust and sneeze in old libraries", 170, "What triggers my sneezing?", "dust"),
    ("My internship stipend will be credited on the first of each month", 10, "When does my stipend arrive?", "first of each month"),
    ("The viva voce for the thesis is in the second week of April", 8, "When is the thesis viva?", "second week"),
    ("I learnt to swim at the municipal pool as a child", 250, "Where did I learn swimming?", "municipal pool"),
    ("My roommate Dev snores loudly", 55, "Who snores in my room?", "dev"),
    ("I save recipes in a green notebook", 75, "Where do I write down recipes?", "green notebook"),
    ("The library fine is two rupees a day", 40, "How much is the library fine per day?", "two rupees"),
    ("I prefer filter coffee over instant coffee", 110, "Which coffee do I prefer?", "filter coffee"),
]
A.UPDATES = [
    ("My tuition class is on Wednesday", 25, "Tuition class moved to Friday evening", 3, True,
     "When is my tuition class?", "friday", "wednesday"),
    ("I commute by bus", 80, "I switched to cycling for my commute", 6, True, "How do I commute?", "cycling", "bus"),
    ("My favourite team is Mumbai Indians", 130, "My favourite team changed to Chennai Super Kings", 12, True,
     "Which IPL team do I support?", "chennai", "mumbai"),
    ("My landlord is Mr Desai", 90, "My landlord is Mrs Kapoor", 9, False, "Who is my landlord?", "kapoor", "desai"),
    ("My project guide is Prof Nair", 60, "My project guide is Prof Bhatt", 7, False,
     "Who is my project guide?", "bhatt", "nair"),
    ("I sleep at eleven at night", 70, "I sleep at one in the morning", 4, False, "When do I sleep?", "one in the morning", "eleven"),
    ("My scholarship amount is five thousand", 50, "My scholarship amount was raised to eight thousand", 5, False,
     "What is my scholarship amount?", "eight thousand", "five thousand"),
    ("I stay in hostel room one twelve", 100, "I moved to hostel room two oh four", 11, True,
     "Which hostel room am I in?", "two oh four", "one twelve"),
]
A.PROBES = [
    ("Actually, I also joined the quiz club", 1, "robotics club"),
    ("I switched on notifications for the library app", 1, "two rupees a day"),
    ("Actually my aunt Meena also sends sweets at Diwali", 1, "sunita"),
]
A.ABSTAIN = [
    "What is my father's profession?", "Which bike model do I own?", "What is my Aadhaar number?",
    "Who is my best friend in college?", "What is my favourite cuisine?", "Which hospital do I visit for checkups?",
    "What time does my first class start?", "Which gym do I attend?", "What is my laptop's model name?",
    "What is my mother's name?", "Where do I want to work after graduation?", "What is my CGPA?",
    "Which colour is my backpack?", "What is my sibling's age?", "Which city was I born in?",
    "What is my phone number?", "Who is my favourite author?", "Which subject do I find hardest?",
    "What is my hostel address?", "What do I eat for breakfast?",
]
_PLANTS = ["tomatoes", "chillies", "mint", "roses", "tulsi", "marigolds", "spinach", "lemons", "curry leaves", "jasmine"]
_SHOPS = [("bakery", "fresh bread"), ("pharmacy", "cough syrup"), ("tailor", "alterations"), ("bookstall", "old novels"),
          ("tea stall", "ginger tea"), ("cobbler", "shoe repairs"), ("grocer", "loose rice"), ("florist", "garlands"),
          ("cyber cafe", "printouts"), ("sweet shop", "jalebi")]
A.GENERIC = (
    [f"My neighbour grows {p} on the terrace" for p in _PLANTS]
    + [f"The {s} near the station sells {x}" for s, x in _SHOPS]
    + ["The auditorium seats about six hundred people", "Our class group chat is very noisy at night",
       "The college canteen raised the price of samosas", "There is a power cut every Saturday afternoon",
       "The hostel warden checks the rooms on Mondays", "Street lights on our lane were replaced in June",
       "The nearest ATM is outside the main gate", "My uncle runs a tiny printing press",
       "The sports day is held in the last week of January", "The old well behind the garden is covered"]
)

OLD = dict(w=(0.6, 0.25, 0.15), hop=True, conflict="full")
NEW = dict(w=(0.6, 0.25, 0.05), hop=True, conflict="full")
SIM = dict(w=(1.0, 0.0, 0.0), hop=False, conflict="off")

if __name__ == "__main__":
    import chromadb
    ev = A.build_events(seed=11)
    print(f"hold-out: facts {len(A.FACTS)} updates {len(A.UPDATES)} abstain {len(A.ABSTAIN)} "
          f"probes {len(A.PROBES)} events {len(ev)}")
    client = chromadb.EphemeralClient()
    for name, cfg, gate in (("sim only", SIM, 0.62), ("old (rec .15, gate .45)", OLD, 0.45),
                            ("new (rec .05, gate .62)", NEW, 0.62)):
        A.GATE = gate
        r = A.run_variant(name, cfg, client, ev)
        print(f"{name:26s} top1={r['recall_top1']:.2f} top3={r['recall_top3']:.2f} "
              f"new_top1={r['update_new_top1']:.2f} stale_top1={r['update_stale_top1']:.2f} "
              f"abstain_refused={r['abstain_refused']:.2f} wrongly_refused={r['answerable_wrongly_refused']:.2f} "
              f"victims_lost={r['collateral_victims_lost']:.2f} unrelated_lost={r['unrelated_rows_lost']:.2f} "
              f"rows={r['rows_stored']}")
