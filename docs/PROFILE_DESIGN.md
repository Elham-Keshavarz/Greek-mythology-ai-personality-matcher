# Archetype profile design (manual baseline)

These vectors are **human-authored interpretations of narrative roles**. There is no historical or psychometric dataset of measured Big Five traits for mythological figures. They should not be referred to as research-derived labels.

Each profile's 5 traits are specified on `[0,1]` in `data/archetype_profiles.json` as a **transparent, editable design hypothesis**. Names/descriptions draw on the [Greek-gods dataset](https://github.com/katkaypettitt/greek-gods) and basic mythological roles. Before offering stronger explanations, add multiple specific primary/secondary citations for each deity.

The current five axes are Openness (IPIP Factor V *Intellect/Imagination*), Conscientiousness, Extraversion, Agreeableness, and Neuroticism (the inverse of IPIP Factor IV *Emotional Stability*).

## Suggested evaluation experiments

1. Change each archetype dimension by ±0.1 and quantify match-rank stability.
2. Vary one user trait at a time and plot nearest-archetype changes.
3. Compare raw Euclidean, normalized Euclidean, and cosine-distance rankings. Do not pick a metric just because it produces appealing results.
4. Ask independent mythology readers to annotate character traits and report inter-rater agreement; avoid implying ground truth without this step.
5. Separately evaluate RAG QA (answer correctness, evidence sufficiency, abstention, and citation accuracy).

## Design limitations

Choosing a Greek god is **recreational storytelling**, not a scientific diagnostic or a valid classification of a person. The score is not a model confidence and should not be shown as such.

## Extended profiles (45 deities)

Profiles in `data/archetype_profiles.json` now include: `gender`, `category`, and `source_url`. Gender represents the deity's traditional portrayal, not a validated psychological property. The 37 new Big Five vectors are **author-curated illustrative hypotheses**, not measured ground truth. There is no training set, psychometric validation, or scientific evidence of how a mythical figure would respond to a modern personality inventory. With more profiles, small distance changes can produce different rankings, so document sensitivity analyses before interpreting outcomes.

For a gender-selected matching view, filter the eligible deity pool first, then calculate the *same* five-dimensional Euclidean distance as in the original MVP; do not alter user traits or give gender-based scoring bonuses.
