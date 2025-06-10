# backendcopy/test_search.py

from search import search_perfumes

def print_section(title, items):
    print(f"\n--- {title} ({len(items)} items) ---")
    if not items:
        print("No matches found.")
        return

    for i, r in enumerate(items, 1):
        print(f"{i}. {r['name']}")
        print(f"   Category        : {r['category']}")
        print(f"   Main Accords    : {r['main_accords']}")
        print(f"   Keyword Matches : {r['keyword_matches']}")

        # Show similarity as a percentage if present
        if "score_norm" in r and isinstance(r["score_norm"], (float, int)):
            pct = r["score_norm"] * 100
            print(f"   Similarity      : {pct:.2f}%")
        else:
            print("   Similarity      : N/A")

        print(f"   Gender String   : {r['gender_str']}")
        print(f"   URL             : {r['url']}\n")


if __name__ == "__main__":
    # Example query (add all relevant keywords)
    query = (
"I want to smell like i am walking in a clean park outdoors with the sun glowing"    )

    # Use a smaller fetch_factor to limit memory usage (e.g., 5 instead of 10)
    results = search_perfumes(query, per_category=5, fetch_factor=5)

    print_section("Top Men-Only Matches",      results["top_men"])
    print_section("Top Women-Only Matches",    results["top_women"])
    print_section("Top Unisex Matches",          results["top_unisex"])
