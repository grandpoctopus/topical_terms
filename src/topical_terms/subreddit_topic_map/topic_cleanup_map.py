TOPIC_CLEANUP_MAP = {
    '"': "",
    # for some reason the scraping put wallpapers on a lot of things
    # that it doesn't belong on
    ":Wallpapers": "",
    # Netflix related got put on a lot of things that are not netflix related
    "TV:Netflix Related": "TV:NR",
    ":Netflix Related": "",
    "TV:NR": "TV:Netflix Related",
    # automotive classified as writing automotive
    "Automotive:Writing": "Automotive",
    # outdoors classified as outdoors car companies
    "Outdoors:Car companies": "Outdoors",
    # Design/carcompanies
    "Design:Car companies": "Design",
    # nostaliga/time has politics in it
    "Nostalgia/Time:Politics": "Nostalgia/Time",
    # photography film all have car companies
    "Photography/Film:Car companies": "Photography/Film",
    # cryptocurrency in a lot of nonrelated things
    ":CryptoCurrency": "",
    # self improvement in a lot of thing it shouln't be in
    "Self-Improvement:Sex": "Self-Improvement",
    "Self-Improvement:Programming": "Programming:Self-Improvement",
    "Self-Improvement:": "",
    # politics in things it shouldn't be in Unexpected:Politics
    "Shitty:Politics": "Shitty",
    "Unexpected:Politics": "Unexpected",
    "Visually Appealing:Politics": "Visually Appealing",
    "Categorize Later:Politics": "Categorize Later",
    # instruments in with sports
    "Sports:Instruments": "Sports",
    # scary weird mixed in with support
    "Support:Scary/Weird": "Support",
    # scary weird mixed in with support
    "Tech Related:Car companies": "Tech Related",
    # Tech Support
    "Tools:Tech Support": "Tools",
    "Travel:Tech Support": "Travel",
    # TV:Soccer
    "TV:Soccer": "TV",
    # Neckbeard mixed with cute
    # Cute:Neckbeard
    "Cute:Neckbeard": "Cute",
    "Dogs:Neckbeard": "Dogs",
    # Conspiracy:Dogs
    "Conspiracy:Dogs": "Conspiracy",
    "Cringe:Dogs": "Cringe",
    "Facebook:Dogs": "Cringe",
    "Meta:Dogs": "Cringe",
    # CringScifi
    "Cringe:Sci-fi": "Cringe",
    # General:Physics
    "General:Physics": "General",
}
