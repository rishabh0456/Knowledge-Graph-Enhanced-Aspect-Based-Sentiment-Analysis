# kg.py - build, load and query the pruned ConceptNet graph
import csv, gzip, json, collections
 
KEEP = {"/r/RelatedTo", "/r/IsA", "/r/HasProperty", "/r/UsedFor",
        "/r/CapableOf", "/r/PartOf", "/r/AtLocation", "/r/Synonym"}
 
def build_kg(assertions_gz, out="kg.json", max_per_node=50):
    """Keep English-English edges of selected relations, top-N by weight."""
    kg = collections.defaultdict(list)
    with gzip.open(assertions_gz, "rt", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            rel, src, dst, meta = row[1], row[2], row[3], row[4]
            if rel not in KEEP:
                continue
            if not (src.startswith("/c/en/") and dst.startswith("/c/en/")):
                continue
            w = json.loads(meta).get("weight", 1.0)
            h, t = src.split("/")[3], dst.split("/")[3]
            r = rel.split("/")[-1]
            kg[h].append((t, r, w))
            kg[t].append((h, r, w))
    kg = {k: sorted(v, key=lambda e: -e[2])[:max_per_node]
          for k, v in kg.items()}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(kg, f)
    return kg
 
def load_kg(path="kg.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
 
def get_knowledge(aspect, kg, k=5):
    """Return up to k neighbour concepts (as plain text) of an aspect."""
    key = aspect.lower().strip().replace(" ", "_")
    edges = kg.get(key)
    if not edges:                      # fall back to the single words
        edges = sorted((e for t in key.split("_") for e in kg.get(t, [])),
                       key=lambda e: -e[2])
    seen, out = set(), []
    for concept, _rel, _w in edges:
        text = concept.replace("_", " ")
        if text not in seen and text != key.replace("_", " "):
            seen.add(text)
            out.append(text)
        if len(out) == k:
            break
    return out

