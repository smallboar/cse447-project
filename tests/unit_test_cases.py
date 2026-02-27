"""
Diverse unit test cases for the next-character predictor.
Each entry is (input_prefix, expected_next_character).
Designed to cover many unique scenarios; model is English-only so non-English and
symbol/code/edge cases should often fail, yielding ~10% or lower accuracy.
"""

# Format: list of (input_string, expected_next_char)
# Categories are commented for clarity; many unique ideas per category.

UNIT_TEST_CASES = [
    # ================================================================
    # GROUP 1: Progressive completions of famous quotes / phrases
    # Tests that the model uses *accumulating* context correctly.
    # Each row is the same sentence truncated one character earlier.
    # ================================================================

    # "That's one small step for man"
    ("That's one small step for ma",        "n"),
    ("That's one small step for m",         "a"),
    ("That's one small step for ",          "m"),
    ("That's one small step fo",            "r"),
    ("That's one small step f",             "o"),
    ("That's one small step ",              "f"),
    ("That's one small ste",               "p"),
    ("That's one small st",                "e"),
    ("That's one small s",                 "t"),
    ("That's one small ",                  "s"),
    ("That's one smal",                    "l"),
    ("That's one sm",                      "a"),
    ("That's one ",                        "s"),
    ("That's on",                          "e"),
    ("That's o",                           "n"),
    ("That's ",                            "o"),
    ("That'",                              "s"),
    ("Tha",                                "t"),
    ("Th",                                 "a"),

    # "one giant leap for mankind"
    ("one giant leap for mankin",           "d"),
    ("one giant leap for manki",            "n"),
    ("one giant leap for mank",             "i"),
    ("one giant leap for man",              "k"),
    ("one giant leap for ma",               "n"),
    ("one giant leap for m",                "a"),
    ("one giant leap for ",                 "m"),
    ("one giant leap fo",                   "r"),
    ("one giant leap f",                    "o"),
    ("one giant leap ",                     "f"),
    ("one giant lea",                       "p"),
    ("one giant le",                        "a"),
    ("one giant l",                         "e"),
    ("one giant ",                          "l"),
    ("one gian",                            "t"),
    ("one gia",                             "n"),
    ("one gi",                              "a"),
    ("one g",                               "i"),
    ("one ",                                "g"),
    ("on",                                  "e"),

    # "Happy New Year"
    ("Happy New Yea",                       "r"),
    ("Happy New Ye",                        "a"),
    ("Happy New Y",                         "e"),
    ("Happy New ",                          "Y"),
    ("Happy New",                           " "),
    ("Happy Ne",                            "w"),
    ("Happy N",                             "e"),
    ("Happy ",                              "N"),
    ("Happ",                                "y"),
    ("Hap",                                 "p"),
    ("Ha",                                  "p"),

    # "To be or not to be, that is the question"
    ("To be or not to be, that is the questio",  "n"),
    ("To be or not to be, that is the questi",   "o"),
    ("To be or not to be, that is the quest",    "i"),
    ("To be or not to be, that is the que",      "s"),
    ("To be or not to be, that is the qu",       "e"),
    ("To be or not to be, that is the q",        "u"),
    ("To be or not to be, that is the ",         "q"),
    ("To be or not to be, that is th",           "e"),
    ("To be or not to be, that is t",            "h"),
    ("To be or not to be, that is ",             "t"),
    ("To be or not to be, that i",               "s"),
    ("To be or not to be, that ",                "i"),
    ("To be or not to be, tha",                  "t"),
    ("To be or not to be,",                      " "),
    ("To be or not to be",                       ","),
    ("To be or not to b",                        "e"),
    ("To be or not to ",                         "b"),
    ("To be or not to",                          " "),
    ("To be or not t",                           "o"),
    ("To be or not ",                            "t"),
    ("To be or not",                             " "),
    ("To be or no",                              "t"),
    ("To be or n",                               "o"),
    ("To be or ",                                "n"),
    ("To be or",                                 " "),
    ("To be o",                                  "r"),
    ("To be ",                                   "o"),
    ("To be",                                    " "),
    ("To b",                                     "e"),
    ("To ",                                      "b"),

    # ================================================================
    # GROUP 2: Context disambiguates — same prefix, different sentences
    # These test that prior context actually changes the prediction.
    # ================================================================

    # "pre" is ambiguous on its own, but context makes next char clear
    ("The nurse said to prevent acc",        "i"),   # "accident"
    ("The nurse said to prevent in",         "f"),   # "infection"
    ("Click the button to preview th",       "e"),
    ("The teacher will present the cl",      "a"),   # "class"
    ("She had a strong preference fo",       "r"),

    # "re" completions determined by prior context
    ("The fire department will respond to th", "e"),
    ("Please remember to restart the se",      "r"),  # "server"
    ("Scientists hope to reverse the ef",      "f"),  # "effects"
    ("They plan to rebuild the entire st",     "r"),  # "structure"

    # "inter" completions
    ("The two countries signed an international ag", "r"),  # "agreement"
    ("She pressed the internet browser's refres",    "h"),
    ("An interesting development in the cas",        "e"),

    # Same word, different completions forced by context
    ("The doctor checked the patient's hear",  "t"),  # "heart"
    ("She could hear the music from her hear", "t"),  # also "heart" — same!
    ("The lawyer argued the case before the co", "u"),  # "court"
    ("Please coat the pan before adding the co", "o"), # "cooking oil" — tricky

    # ================================================================
    # GROUP 3: Post-punctuation predictions
    # Tests what follows various punctuation marks in natural context.
    # ================================================================

    ("She asked, \"Are you ready?\" He said, \"",    "Y"),  # "Yes"
    ("The results were clear: ",                     "t"),  # lowercase letter starting explanation
    ("Warning: do not",                              " "),
    ("He paused. Then he said,",                     " "),
    ("She smiled. \"Of course,\" she replied. \"",   "I"),
    ("First, wash your hands. Then, ",               "d"),  # "dry" or "rinse"
    ("The answer is no. However, ",                  "t"),  # "there" / "the"
    ("It failed. Again. ",                           "T"),  # new sentence
    ("Run! Don't look back! Just ",                  "r"),  # "run" / "go"
    ("Wait—are you serious?",                        " "),  # em dash pause

    # After comma in list context
    ("The ingredients are flour, sugar, butter, and ",  "e"),  # "eggs"
    ("He visited Paris, Rome, Madrid, and ",            "L"),  # "London" / "Lisbon"
    ("The colors red, blue, green, and ",               "y"),  # "yellow"

    # ================================================================
    # GROUP 4: Numbers and alphanumeric in meaningful context
    # ================================================================

    ("The meeting is scheduled for 2:30 p",   "m"),   # "pm"
    ("Today is January 1",                    "5"),   # plausible date
    ("The temperature is -2",                 "0"),   # "-20"
    ("She scored 10",                         "0"),   # "100"
    ("The ISBN is 978-0-06-112008-",          "4"),
    ("In the year 199",                       "9"),   # "1999" very common year
    ("Version 3.1",                           "4"),   # "3.14" pi-style
    ("We need 1",                             "0"),   # common after single digit
    ("There are 2",                           "6"),   # less certain but "26 letters"
    ("Call 911 if",                           " "),

    # ================================================================
    # GROUP 5: Code and technical text
    # Natural for astronauts sending technical messages.
    # ================================================================

    ("def calculate_",                        "d"),   # common function name start
    ("import numpy as n",                     "p"),   # "np"
    ("git commit -m \"Fix bu",               "g"),   # "bug"
    ("https://www.",                          "g"),   # e.g. google / github
    ("Error 40",                              "4"),   # "404"
    ("TODO: fix the memory lea",             "k"),
    ("SELECT * FROM user",                   "s"),    # "users"
    ("The API returns a 20",                 "0"),    # "200"
    ("npm install --save-de",               "v"),    # "dev"
    ("ssh -p 22",                            " "),

    # ================================================================
    # GROUP 6: Multi-sentence passages — tests long-range context
    # ================================================================

    ("The sun was setting. The sky turned orange. She watched from the wi", "n"),  # "window"
    ("He had one rule: never lie. He broke it tod",                         "a"),  # "today"
    ("It was cold. Very cold. The kind of cold that gets into your bo",     "n"),  # "bones"
    ("She opened the letter. Her hands trembled. The news was",             " "),
    ("The rocket launched at dawn. The crew of six held their breath. T",   "h"),  # "The" or "They"
    ("Mission control confirmed the orbit. \"Houston, we have no problem",  "s"), # "problems"

    # ================================================================
    # GROUP 7: Linguistic patterns — contractions, possessives, elision
    # ================================================================

    ("I can't believe it'",                  "s"),   # "it's"
    ("They're going to the store, aren't the", "y"), # "they"
    ("We've been here before, haven't w",    "e"),
    ("She wouldn't say why, couldn't explai", "n"),
    ("It's not what you'd expec",            "t"),
    ("He doesn't know what he's doin",       "g"),
    ("You'll see what I'",                   "m"),   # "I'm"
    ("We're almost there, we're not tur",    "n"),   # "turning"
    ("That's John's ba",                     "g"),   # "bag"
    ("The cat's out of the ba",              "g"),

    # ================================================================
    # GROUP 8: Common letter pair / trigraph patterns
    # Tests low-level n-gram style continuation.
    # ================================================================

    ("the ",    "q"),   # intentionally hard — no strong prediction
    ("th",      "e"),   # "the" very common
    ("qu",      "i"),   # "qui-" very common in English
    ("wh",      "a"),   # "what/which/when/where" all start "wha/whi/whe"
    ("sh",      "e"),   # "she/the/she"
    ("ch",      "a"),   # "cha-"
    ("wr",      "i"),   # "wri-" (write/wrong both possible, "wri" more common)
    ("kn",      "o"),   # "know/knock"
    ("pn",      "e"),   # "pneumo-"
    ("ps",      "y"),   # "psych-"

    # ================================================================
    # GROUP 9: Non-English progressive completions
    # (Same logic as Group 1 but for other languages)
    # ================================================================

    # Spanish: "Buenos días, ¿cómo estás?"
    ("Buenos días, ¿cómo est",               "á"),
    ("Buenos días, ¿cómo es",                "t"),
    ("Buenos días, ¿cómo e",                 "s"),
    ("Buenos días, ¿cómo ",                  "e"),
    ("Buenos días, ¿cóm",                    "o"),
    ("Buenos días, ¿có",                     "m"),
    ("Buenos días, ¿",                       "c"),
    ("Buenos días, ",                        "¿"),
    ("Buenos días,",                         " "),
    ("Buenos día",                           "s"),
    ("Buenos dí",                            "a"),
    ("Buenos d",                             "í"),
    ("Buenos ",                              "d"),
    ("Bueno",                                "s"),
    ("Buen",                                 "o"),

    # French: "Bonjour, comment allez-vous?"
    ("Bonjour, comment allez-vou",           "s"),
    ("Bonjour, comment allez-vo",            "u"),
    ("Bonjour, comment allez-v",             "o"),
    ("Bonjour, comment allez-",              "v"),
    ("Bonjour, comment alle",                "z"),
    ("Bonjour, comment all",                 "e"),
    ("Bonjour, comment al",                  "l"),
    ("Bonjour, comment ",                    "a"),
    ("Bonjour, commen",                      "t"),
    ("Bonjour, comme",                       "n"),
    ("Bonjour, comm",                        "e"),
    ("Bonjour, com",                         "m"),
    ("Bonjour, co",                          "m"),
    ("Bonjour, ",                            "c"),
    ("Bonjour,",                             " "),
    ("Bonjour",                              ","),
    ("Bonjou",                               "r"),

    # Russian progressive: "Как тебя зовут?"
    ("Как тебя зову",                        "т"),
    ("Как тебя зов",                         "у"),
    ("Как тебя зо",                          "в"),
    ("Как тебя з",                           "о"),
    ("Как тебя ",                            "з"),
    ("Как теб",                              "я"),
    ("Как те",                               "б"),
    ("Как т",                                "е"),
    ("Как ",                                 "т"),
    ("Ка",                                   "к"),

    # Chinese progressive: "今天天气怎么样？"
    ("今天天气怎么样",                        "？"),
    ("今天天气怎么",                          "样"),
    ("今天天气怎",                            "么"),
    ("今天天气",                              "怎"),
    ("今天天",                                "气"),
    ("今天",                                  "天"),
    ("今",                                    "天"),

    # ================================================================
    # GROUP 10: Sentence-type variety (question, exclamation, statement)
    # Tests that the model handles different registers correctly.
    # ================================================================

    # Direct question — expects question mark
    ("What time does the shuttle dock",      "?"),
    ("Are you sure about the oxygen leve",  "l"),
    ("Can you hear me now",                  "?"),
    ("How long until we reach the stati",   "o"),  # "station"
    ("Is the airlock seal",                  "e"),  # "sealed"

    # Imperative
    ("Please confirm your positio",         "n"),
    ("Do not open the external hat",        "c"),  # "hatch"
    ("Report your status every thirt",      "y"),  # "thirty"
    ("Initiate the docking sequen",         "c"),  # "sequence"
    ("Abort the EVA and return to the ai",  "r"),  # "airlock"

    # Exclamatory
    ("We did it",                            "!"),
    ("The view from up here is incredible", "!"),
    ("I can see the whole Pacific Ocea",    "n"),

    # ================================================================
    # GROUP 11: Tricky double-letter and spelling patterns
    # ================================================================

    ("She felt an overwhelming sens",        "e"),  # not "s"
    ("The committee will meet on Wednesd",   "a"),  # "Wednesday"
    ("It was an unnecessary complicati",     "o"),  # "complication"
    ("The government issued a new regul",    "a"),  # "regulation"
    ("He made an embarrassing mista",        "k"),  # "mistake"
    ("Please write your address on the envel", "o"), # "envelope"
    ("The occurrence was highly unusu",      "a"),  # "unusual"
    ("Success requires consisten",           "c"),  # "consistency"
    ("She has a natural tendency to procrastin", "a"), # "procrastinate"
    ("The accommodation was excel",          "l"),  # "excellent"

    # ================================================================
    # GROUP 12: Edge cases that actually test something meaningful
    # ================================================================

    # Single character with strong bigram expectation
    ("q",   "u"),   # "qu" is dominant in English
    ("x",   " "),   # "x" rarely starts a useful sequence
    ("z",   "e"),   # "ze-" (zero/zen) vs space — context-dependent but "ze" common

    # After sentence-ending space — strong prior for capital
    (". T",          "h"),  # "The / This / They"
    (". I",          " "),  # "I " — "I think/I want"
    (". S",          "h"),  # "She / Should / So"
    (". W",          "e"),  # "We / Well / When"

    # Repetition that has a clear continuation
    ("ha ha ha h",   "a"),
    ("na na na na, hey hey hey, goodb",  "y"),  # "goodbye"
    ("the more the merri",  "e"),  # "merrier"
    ("all's well that ends wel",  "l"),
]

# UNIT_TEST_CASES = [
#     # ---- Non-English: Spanish ----
#     ("Hola mund", "o"),
#     ("¿Qué tal ", "e"),
#     ("Buenos dí", "a"),
#     ("Gracias por tu ayu", "d"),
#     ("El gato está en la ventan", "a"),
#     ("Me gusta la comid", "a"),
#     ("Hasta luego, amig", "o"),
#     ("¿Dónde está la bibliotec", "a"),
#     ("No entiendo nad", "a"),
#     ("Por favor, ayud", "a"),
#     # ---- Non-English: Russian ----
#     ("Привет ми", "р"),
#     ("Спасибо больш", "о"),
#     ("Как дела", "?"),
#     ("Доброе утро", ","),
#     ("До свидани", "я"),
#     ("Меня зовут Алекса", "н"),
#     ("Где здесь туалет", "?"),
#     ("Это очень красив", "о"),
#     ("Хорошего дн", "я"),
#     ("Пожалуйста, помогите м", "н"),
#     # ---- Non-English: Chinese ----
#     ("你好世", "界"),
#     ("谢谢你的帮", "助"),
#     ("今天天气很好", "。"),
#     ("我是学", "生"),
#     ("再见，朋", "友"),
#     ("请问厕所在哪", "里"),
#     ("这个很好吃", "。"),
#     ("明天见", "。"),
#     ("对不起，我不懂", "。"),
#     ("欢迎来到北", "京"),
#     # ---- Non-English: Arabic ----
#     ("مرحبا بك", " "),
#     ("شكرا جزيلا", " "),
#     ("كيف حالك", "؟"),
#     ("مع السلامة", " "),
#     ("أنا طالب", " "),
#     ("أين الحمام", "؟"),
#     ("هذا لذيذ", " "),
#     ("إلى اللقاء", " "),
#     ("من فضلك ساعدني", " "),
#     ("أهلا وسهلا", " "),
#     # ---- Punctuation and formatting ----
#     ("Mr. ", "S"),
#     ("Dr. ", " "),
#     ("e.g. ", " "),
#     ("i.e. ", " "),
#     ("U.S. ", " "),
#     ("Yes! ", " "),
#     ("Wait... ", " "),
#     ("Note: ", " "),
#     ("Hello—world", " "),
#     ("(parenthesis) ", " "),
#     # ---- Contractions and apostrophes ----
#     ("don't ", "w"),
#     ("it's ", "a"),
#     ("I'm ", "s"),
#     ("we're ", "g"),
#     ("they've ", "b"),
#     ("can't ", " "),
#     ("won't ", " "),
#     ("John's ", "c"),
#     ("that's ", "w"),
#     ("what's ", "u"),
#     # ---- Rare or long English words ----
#     ("unbelievable", " "),
#     ("questionnaire", " "),
#     ("necessary", " "),
#     ("accommodation", " "),
#     # ---- Short / edge context ----
#     ("", " "),
#     ("a", " "),
#     ("x", " "),
#     ("q", "u"),
#     ("z", " "),
#     (" ", " "),
#     ("\t", " "),
#     (".", " "),
#     ("?", " "),
#     ("!", " "),
#     # ---- Repetition and patterns ----
#     ("aaaaaaa", "a"),
#     ("aaaaaaa", " "),
#     ("the the the ", "t"),
#     ("abcabcab", "c"),
#     ("1231231", "2"),
#     # ---- Mixed language ----
#     ("Hello 世界", " "),
#     ("Adiós friends", " "),
#     # ---- Slang and informal ----
#     ("OMG that's ", "s"),
#     ("LOL so funn", "y"),
#     ("BRB in a sec", " "),
#     ("IMO the best", " "),
#     ("FYI we're late", " "),
#     ("TBH I don't know", " "),
#     ("IDK what to do", " "),
#     ("ASAP please", " "),
#     ("RSVP by Friday", " "),
#     ("DIY project", " "),
#     # ---- Dates and time ----
#     ("Jan 1, 202", "4"),
#     ("12:30 ", "p"),
#     ("Q1 202", "4"),
#     ("March 15, 200", "0"),
#     ("Mon Dec 0", "1"),
#     ("09:00 AM", " "),
#     ("1st place", " "),
#     ("2nd floor", " "),
#     ("3rd time", " "),
#     ("Chapter IV", " "),
#     # ---- Quotes and brackets ----
#     ("'Hello' ", "s"),
#     ('"Yes" ', "s"),
#     ("(example)", " "),
#     ("[array]", " "),
#     ("{key}: ", '"'),
#     ("<tag", ">"),
#     ("'Single quoted", "'"),
#     ("\"Double quoted", '"'),
#     ("Back `tick`", " "),
#     ("—em dash—", " "),
#     # ---- Emoji and Unicode ----
#     ("Hello 😀", " "),
#     ("Thumbs up 👍", " "),
#     ("مرحبا", " "),
#     ("Привет", " "),
#     # ---- Sentence boundaries ----
#     ("Hello. ", "I"),
#     ("Really? ", " "),
#     ("No! ", " "),
#     ("Okay. ", "L"),
#     ("Done. ", " "),
#     ("Sure. ", " "),
#     ("Maybe. ", " "),
#     ("Thanks. ", " "),
#     ("Sorry. ", " "),
#     ("Wait. ", " "),
#     # ---- Common suffixes ----
#     ("running ", " "),
#     ("happily ", " "),
#     ("question", " "),
#     ("happiness", " "),
#     ("beautiful", " "),
#     ("quickly", " "),
#     ("national", " "),
#     ("important", " "),
#     ("different", " "),
#     ("government", " "),
#     # ---- Common prefixes ----
#     ("unhappy", " "),
#     ("rediscover", " "),
#     ("preview", " "),
#     ("submarine", " "),
#     ("antivirus", " "),
#     ("bicycle", " "),
#     ("triangle", " "),
#     ("microscope", " "),
#     ("kilometer", " "),
#     ("megabyte", " "),
#     # ---- Latin and borrowed ----
#     ("et cetera", " "),
#     ("ad hoc", " "),
#     ("per se", " "),
#     ("vice versa", " "),
#     ("status quo", " "),
#     # ---- Very long context (stress trie) ----
#     ("the quick brown fox jumps over the lazy do", "g"),
#     ("In the beginning was the Word, and the Word was with Go", "d"),
#     ("To be or not to be, that is the questio", "n"),
#     ("It was the best of times, it was the worst of time", "s"),
#     ("All happy families are alike; each unhappy family is unhapp", "y"),
#     # ---- Abbreviations ----
#     ("vs. ", " "),
#     ("etc. ", " "),
#     ("approx. ", " "),
#     ("Prof. ", " "),
#     ("Fig. ", " "),
#     ("Vol. ", " "),
#     ("No. ", " "),
#     ("Rev. ", " "),
#     ("Gen. ", " "),
#     ("Sen. ", " "),
#     # ---- Hyphen and compound ----
#     ("well-known ", " "),
#     ("state-of-the-art ", " "),
#     ("twenty-one", " "),
#     ("self-esteem", " "),
#     ("mother-in-law", " "),
#     ("long-term", " "),
#     ("high-quality", " "),
#     ("user-friendly", " "),
#     ("part-time", " "),
#     ("full-time", " "),
#     # ---- Colons and semicolons ----
#     ("Note: ", " "),
#     ("Warning: ", " "),
#     ("Example: ", " "),
#     ("However; ", " "),
#     ("Therefore; ", " "),
#     ("First: ", " "),
#     ("Step 1: ", " "),
#     ("URL: ", " "),
#     ("Date: ", " "),
#     ("To: ", " "),
#     # ---- Interjections ----
#     ("Oh! ", " "),
#     ("Ah! ", " "),
#     ("Wow! ", " "),
#     ("Ouch! ", " "),
#     ("Hmm. ", " "),
#     ("Uh. ", " "),
#     ("Well, ", " "),
#     ("So, ", " "),
#     ("Like, ", " "),
#     ("Anyway, ", " "),
#     # ---- Double letters and spelling ----
#     ("book", " "),
#     ("letter", " "),
#     ("happy", " "),
#     ("balloon", " "),
#     ("committee", " "),
#     ("occurrence", " "),
#     ("embarrass", " "),
#     ("address", " "),
#     ("success", " "),
#     ("possess", " "),
#     # ---- Onomatopoeia ----
#     ("bang!", " "),
#     ("boom!", " "),
#     ("splash!", " "),
#     ("meow", " "),
#     ("woof", " "),
#     ("buzz", " "),
#     ("click", " "),
#     ("pop", " "),
#     ("zip", " "),
#     ("crash", " "),
# ]


def get_input_lines():
    return [t[0] for t in UNIT_TEST_CASES]


def get_answer_lines():
    return [t[1] for t in UNIT_TEST_CASES]
