include_cols = [
    "symptom:Fever",
    "symptom:Low-grade fever",
    "symptom:Cough",
    "symptom:Sore throat",
    "symptom:Headache",
    "symptom:Fatigue",
    "symptom:Vomiting",
    "symptom:Diarrhea",
    "symptom:Shortness of breath",
    "symptom:Chest pain",
    "symptom:Dizziness",
    "symptom:Confusion",
    "symptom:Generalized tonic–clonic seizure",
    "symptom:Weakness",
]

hhs_regions = {
    1: ["CT", "ME", "MA", "NH", "RI", "VT"],
    2: ["NJ", "NY", "PR", "VI"],
    3: ["DE", "DC", "MD", "VA", "PA", "WV"],
    4: ["AL", "FL", "GA", "KY", "MS", "NC", "SC", "TN"],
    5: ["IL", "IN", "MI", "MN", "OH", "WI"],
    6: ["AR", "LA", "NM", "OK", "TX"],
    7: ["IA", "KS", "MO", "NE"],
    8: ["CO", "MT", "ND", "SD", "UT", "WY"],
    9: ["AZ", "CA", "HI", "NV", "AS", "GU"],
    10: ["AK", "ID", "OR", "WA"],
}

hhs_neighbors = {
    1: [2],
    2: [1, 3],
    3: [2, 4, 5],
    4: [3, 5, 6, 7],
    5: [3, 4, 7, 8],
    6: [4, 7, 8, 9],
    7: [4, 5, 6, 8],
    8: [5, 6, 7, 9, 10],
    9: [6, 8, 10],
    10: [8, 9],
}
