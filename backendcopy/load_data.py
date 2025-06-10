import pandas as pd
import ast
import numpy as np
import re
import spacy

# Load spaCy model for optional NLP use
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    """
    Lemmatize and remove stopwords from a string.
    (Not strictly required for gender extraction, but included for optional use.)
    """
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])

def load_and_clean_data(filepath: str = "backendcopy/data/fra_perfumes.csv") -> pd.DataFrame:
    """
    Load and clean fragrance dataset. Extract gender from 'name', clean ratings and accords,
    and return structured DataFrame.
    """
    # 1. Load CSV
    df = pd.read_csv(filepath)

    # 2. Normalize column names
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # 3. Extract gender from end of name using regex
    gender_pattern = re.compile(
        r"(for men and women|for women and men|for women|for men)$",
        re.IGNORECASE
    )

    extracted_genders = []
    cleaned_names = []

    for raw_name in df["name"].astype(str):
        stripped = raw_name.strip()
        match = gender_pattern.search(stripped)

        if match:
            gen = match.group(0).lower().strip()
            extracted_genders.append(gen)
            new_name = stripped[: match.start()].strip()
            cleaned_names.append(new_name)
        else:
            extracted_genders.append("")
            cleaned_names.append(stripped)

    df["name"] = cleaned_names
    df["gender"] = extracted_genders

    # 4. Remove 'for ' prefix and normalize values
    df["gender"] = df["gender"].str.replace("for ", "", regex=False).str.strip()
    df["gender"] = df["gender"].replace({
        "women and men": "unisex",
        "men and women": "unisex"
    })

    # 5. Convert rating fields to numeric
    df["rating_value"] = pd.to_numeric(df["rating_value"], errors="coerce")
    df["rating_count"] = pd.to_numeric(df["rating_count"].astype(str).str.replace(",", ""), errors="coerce")

    # 6. Convert main_accords to list
    df["main_accords"] = df["main_accords"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else []
    )

    # 7. Compute log-weighted rating score
    df["rating_score"] = np.where(
        df["rating_count"].notnull() & (df["rating_count"] > 0),
        df["rating_value"] * np.log1p(df["rating_count"]),
        0
    )

    # 8. Final column selection
    df = df[[
        "name",
        "gender",
        "rating_value",
        "rating_count",
        "rating_score",
        "main_accords",
        "description",
        "url"
    ]]

    return df

if __name__ == "__main__":
    # Example usage
    filepath = "backendcopy/data/fra_perfumes.csv"
    cleaned_df = load_and_clean_data(filepath)
    print(cleaned_df.head(10))