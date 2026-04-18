import re
import streamlit as st
from neo4j import GraphDatabase
from openai import OpenAI

# --- 1. CONFIGURATION ---
# This tells the app: "Go get the values I saved in the Streamlit vault"
OPENAI_KEY = st.secrets["OPENAI_KEY"]
NEO4J_URI = st.secrets["NEO4J_URI"]
NEO4J_USER = st.secrets["NEO4J_USER"]
NEO4J_PWD = st.secrets["NEO4J_PWD"]

client = OpenAI(api_key=OPENAI_KEY)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))


# --- 2. HELPERS ---
def safe_index_name(rel_type: str) -> str:
    return "rel_index_" + re.sub(r"[^A-Za-z0-9_]", "_", rel_type.lower())


def get_embedding(text: str):
    return client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    ).data[0].embedding


def parse_numeric(val):
    try:
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        cleaned = str(val).replace(",", "").strip()
        return float(cleaned)
    except Exception:
        return None


def format_value_unit(val, unit):
    if val is None:
        return "N/A"
    if unit:
        return f"{val} {unit}"
    return str(val)


def deduplicate_facts(facts):
    """
    Deduplicate near-identical facts.
    Keeps the first good version.
    """
    seen = set()
    clean = []

    for f in facts:
        parts = [p.strip() for p in f.split("|")]
        key_parts = []

        for p in parts:
            if p.startswith("Score:"):
                continue
            key_parts.append(p)

        key = " | ".join(key_parts)

        if key not in seen:
            seen.add(key)
            clean.append(f)

    return clean


# --- 3. QUERY TYPE DETECTION ---
def is_reasoning_query(user_query: str) -> bool:
    q = user_query.lower()
    reasoning_terms = [
        "greater than", "less than", "higher than", "lower than",
        "compare", "comparison", "year over year", "yoy",
        "quarter over quarter", "qoq",
        "percentage", "percent", "largest", "smallest",
        "increase", "decrease", "more than", "fewer than"
    ]
    return any(term in q for term in reasoning_terms)


def is_driver_query(user_query: str) -> bool:
    q = user_query.lower()
    driver_terms = [
        "driver", "drivers", "cause", "causes", "factor", "factors",
        "determinant", "determinants", "contributor", "contributors",
        "main reason", "primary reason", "influencing"
    ]
    return any(term in q for term in driver_terms)


def detect_metric_reference(user_query: str):
    q = user_query.lower()

    metric_map = {
        "net income": "net income",
        "operating income": "operating income",
        "net sales": "net sales",
        "revenue": "net sales",
        "liabilities": "liabilities",
        "total assets": "total assets",
        "earnings per share": "earnings per share",
        "eps": "earnings per share",
        "gross margin": "gross margin"
    }

    for k, v in metric_map.items():
        if k in q:
            return v
    return None


def detect_year(user_query: str):
    years = re.findall(r"\b(20\d{2})\b", user_query)
    return years[0] if years else None


# --- 4. ADVANCED RAG: QUERY REWRITE ---
def rewrite_query(user_query: str) -> str:
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You rewrite user questions into clear, precise financial-analysis search queries. "
                        "Keep the meaning the same. Be short and specific. "
                        "If the query is already clear, return it unchanged."
                    )
                },
                {
                    "role": "user",
                    "content": f"Rewrite this query for graph retrieval:\n{user_query}"
                }
            ]
        )
        rewritten = completion.choices[0].message.content.strip()
        return rewritten if rewritten else user_query
    except Exception as e:
        print(f"⚠️ Query rewrite failed: {e}")
        return user_query


# --- 5. ADVANCED RAG: QUERY EXPANSION ---
def expand_query(user_query: str):
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate up to 3 short alternate search queries for a financial knowledge graph. "
                        "Use finance synonyms where useful. "
                        "Examples: revenue -> net sales, debt -> liabilities, profit -> net income. "
                        "Return each query on a new line only. No numbering."
                    )
                },
                {
                    "role": "user",
                    "content": f"Expand this search query:\n{user_query}"
                }
            ]
        )
        raw = completion.choices[0].message.content.strip()
        variations = [q.strip("-• ").strip() for q in raw.splitlines() if q.strip()]

        seen = set()
        clean = []
        for q in [user_query] + variations:
            if q.lower() not in seen:
                seen.add(q.lower())
                clean.append(q)
        return clean[:4]
    except Exception as e:
        print(f"⚠️ Query expansion failed: {e}")
        return [user_query]


# --- 6. STANDARD GRAPH SEARCH ---
def search_graph_once(user_query, node_threshold=0.25, rel_threshold=0.20, top_k_nodes=8, top_k_rels=8):
    query_emb = get_embedding(user_query)
    fact_pool = []

    with driver.session() as session:
        node_results = session.run("""
            CALL db.index.vector.queryNodes('node_index', $top_k, $emb)
            YIELD node, score
            WHERE score >= $threshold
            RETURN node.name AS name, score
            ORDER BY score DESC
        """, emb=query_emb, threshold=node_threshold, top_k=top_k_nodes)

        anchors = [res["name"] for res in node_results]

        print(f"\n🔍 QUERY: {user_query}")
        print(f"📍 ANCHORS: {anchors}")

        if not anchors:
            return []

        rel_types_res = session.run("CALL db.relationshipTypes()")
        all_types = [r["relationshipType"] for r in rel_types_res]

        for r_type in all_types:
            idx_name = safe_index_name(r_type)

            rel_query = f"""
                CALL db.index.vector.queryRelationships('{idx_name}', $top_k, $emb)
                YIELD relationship, score
                WHERE score >= $rel_threshold
                MATCH (n)-[relationship]->(target)
                WHERE n.name IN $anchors
                RETURN
                    n.name AS sub,
                    type(relationship) AS rel,
                    target.name AS obj,
                    relationship.value AS val,
                    relationship.unit AS unit,
                    relationship.date AS date,
                    relationship.period AS period,
                    score AS score
            """

            try:
                rels = session.run(
                    rel_query,
                    emb=query_emb,
                    anchors=anchors,
                    rel_threshold=rel_threshold,
                    top_k=top_k_rels
                )

                for f in rels:
                    val_str = format_value_unit(f["val"], f["unit"])
                    time_parts = [p for p in [f["period"], f["date"]] if p]
                    time_info = " | ".join(time_parts) if time_parts else "N/A"

                    fact_line = (
                        f"FACT: {f['sub']} {f['rel']} {f['obj']} "
                        f"| Value: {val_str} "
                        f"| Time: {time_info} "
                        f"| Score: {round(f['score'], 4)}"
                    )
                    fact_pool.append(fact_line)

            except Exception as e:
                print(f"⚠️ Relationship search failed for {idx_name}: {e}")
                continue

    return deduplicate_facts(fact_pool)


# --- 7. FACT RERANKING ---
def rerank_facts(user_query: str, facts: list[str], keep_top_n: int = 8):
    try:
        numbered_facts = "\n".join([f"{i+1}. {fact}" for i, fact in enumerate(facts)])

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are ranking grounded financial facts for relevance. "
                        "Return only the numbers of the most relevant facts for answering the user's question. "
                        f"Return at most {keep_top_n} numbers, comma-separated. No explanation."
                    )
                },
                {
                    "role": "user",
                    "content": f"QUESTION:\n{user_query}\n\nFACTS:\n{numbered_facts}"
                }
            ]
        )

        raw = completion.choices[0].message.content.strip()
        selected = []

        for part in raw.split(","):
            part = part.strip()
            if part.isdigit():
                idx = int(part) - 1
                if 0 <= idx < len(facts):
                    selected.append(facts[idx])

        selected = selected[:keep_top_n] if selected else facts[:keep_top_n]
        return deduplicate_facts(selected)

    except Exception as e:
        print(f"⚠️ Reranking failed: {e}")
        return deduplicate_facts(facts[:keep_top_n])


# --- 8. NORMAL GRAPH RAG FLOW ---
def get_grounded_context(user_query):
    rewritten = rewrite_query(user_query)
    expanded_queries = expand_query(rewritten)

    print(f"📝 REWRITTEN QUERY: {rewritten}")
    print(f"🧠 EXPANDED QUERIES: {expanded_queries}")

    all_facts = []

    for q in expanded_queries:
        facts = search_graph_once(q)
        all_facts.extend(facts)

    all_facts = deduplicate_facts(all_facts)

    if not all_facts:
        return None, rewritten, expanded_queries

    reranked_facts = rerank_facts(user_query, all_facts)
    return "\n".join(reranked_facts), rewritten, expanded_queries


# --- 9. GRAPH REASONING MODE ---
def run_reasoning_query(user_query: str):
    metric_ref = detect_metric_reference(user_query)
    year = detect_year(user_query)

    if not metric_ref or not year:
        return None

    q = user_query.lower()

    if any(term in q for term in ["greater than", "higher than", "more than"]):
        return reasoning_greater_than(metric_ref, year)

    if any(term in q for term in ["less than", "lower than", "fewer than"]):
        return reasoning_less_than(metric_ref, year)

    return None


def reasoning_greater_than(reference_metric: str, year: str):
    with driver.session() as session:
        ref_query = """
        MATCH (s:Entity)-[r]->(o:Entity)
        WHERE toLower(o.name) CONTAINS toLower($metric)
          AND (
                toString(r.date) CONTAINS $year OR
                toString(r.period) CONTAINS $year
              )
          AND r.value IS NOT NULL
        RETURN o.name AS metric_name, r.value AS value, r.unit AS unit, r.period AS period, r.date AS date
        LIMIT 1
        """
        ref_result = session.run(ref_query, metric=reference_metric, year=year).single()

        if not ref_result:
            return {
                "mode": "reasoning",
                "success": False,
                "message": f"Could not find reference metric '{reference_metric}' for {year}.",
                "evidence": []
            }

        ref_value = parse_numeric(ref_result["value"])
        ref_unit = ref_result["unit"]
        ref_name = ref_result["metric_name"]

        if ref_value is None:
            return {
                "mode": "reasoning",
                "success": False,
                "message": f"Found '{ref_name}' for {year}, but its value could not be parsed numerically.",
                "evidence": []
            }

        cmp_query = """
        MATCH (s:Entity)-[r]->(o:Entity)
        WHERE (
                toString(r.date) CONTAINS $year OR
                toString(r.period) CONTAINS $year
              )
          AND r.value IS NOT NULL
        RETURN o.name AS metric_name, r.value AS value, r.unit AS unit, r.period AS period, r.date AS date, type(r) AS rel
        """
        rows = session.run(cmp_query, year=year)

        bigger = []
        evidence = []

        for row in rows:
            metric_name = row["metric_name"]
            metric_value = parse_numeric(row["value"])
            metric_unit = row["unit"]

            if metric_value is None:
                continue

            if ref_unit and metric_unit and str(ref_unit).lower() != str(metric_unit).lower():
                continue

            evidence_line = (
                f"FACT: {metric_name} | Value: {format_value_unit(row['value'], metric_unit)} "
                f"| Time: {row['period'] or ''} {row['date'] or ''} | Rel: {row['rel']}"
            )
            evidence.append(evidence_line)

            if metric_name.lower() != ref_name.lower() and metric_value > ref_value:
                bigger.append({
                    "metric_name": metric_name,
                    "value": metric_value,
                    "unit": metric_unit,
                    "period": row["period"],
                    "date": row["date"],
                    "rel": row["rel"]
                })

        bigger = sorted(bigger, key=lambda x: x["value"], reverse=True)

        return {
            "mode": "reasoning",
            "success": True,
            "reference_metric": ref_name,
            "reference_value": ref_value,
            "reference_unit": ref_unit,
            "year": year,
            "results": bigger,
            "evidence": deduplicate_facts(evidence)[:20]
        }


def reasoning_less_than(reference_metric: str, year: str):
    with driver.session() as session:
        ref_query = """
        MATCH (s:Entity)-[r]->(o:Entity)
        WHERE toLower(o.name) CONTAINS toLower($metric)
          AND (
                toString(r.date) CONTAINS $year OR
                toString(r.period) CONTAINS $year
              )
          AND r.value IS NOT NULL
        RETURN o.name AS metric_name, r.value AS value, r.unit AS unit, r.period AS period, r.date AS date
        LIMIT 1
        """
        ref_result = session.run(ref_query, metric=reference_metric, year=year).single()

        if not ref_result:
            return {
                "mode": "reasoning",
                "success": False,
                "message": f"Could not find reference metric '{reference_metric}' for {year}.",
                "evidence": []
            }

        ref_value = parse_numeric(ref_result["value"])
        ref_unit = ref_result["unit"]
        ref_name = ref_result["metric_name"]

        if ref_value is None:
            return {
                "mode": "reasoning",
                "success": False,
                "message": f"Found '{ref_name}' for {year}, but its value could not be parsed numerically.",
                "evidence": []
            }

        cmp_query = """
        MATCH (s:Entity)-[r]->(o:Entity)
        WHERE (
                toString(r.date) CONTAINS $year OR
                toString(r.period) CONTAINS $year
              )
          AND r.value IS NOT NULL
        RETURN o.name AS metric_name, r.value AS value, r.unit AS unit, r.period AS period, r.date AS date, type(r) AS rel
        """
        rows = session.run(cmp_query, year=year)

        smaller = []
        evidence = []

        for row in rows:
            metric_name = row["metric_name"]
            metric_value = parse_numeric(row["value"])
            metric_unit = row["unit"]

            if metric_value is None:
                continue

            if ref_unit and metric_unit and str(ref_unit).lower() != str(metric_unit).lower():
                continue

            evidence_line = (
                f"FACT: {metric_name} | Value: {format_value_unit(row['value'], metric_unit)} "
                f"| Time: {row['period'] or ''} {row['date'] or ''} | Rel: {row['rel']}"
            )
            evidence.append(evidence_line)

            if metric_name.lower() != ref_name.lower() and metric_value < ref_value:
                smaller.append({
                    "metric_name": metric_name,
                    "value": metric_value,
                    "unit": metric_unit,
                    "period": row["period"],
                    "date": row["date"],
                    "rel": row["rel"]
                })

        smaller = sorted(smaller, key=lambda x: x["value"])

        return {
            "mode": "reasoning",
            "success": True,
            "reference_metric": ref_name,
            "reference_value": ref_value,
            "reference_unit": ref_unit,
            "year": year,
            "results": smaller,
            "evidence": deduplicate_facts(evidence)[:20]
        }


# --- 10. DRIVER ANALYSIS MODE ---
def run_driver_analysis(user_query: str):
    metric_ref = detect_metric_reference(user_query)
    year = detect_year(user_query)

    if not metric_ref:
        return None

    with driver.session() as session:
        query = """
        MATCH (s:Entity)-[r]->(o:Entity)
        WHERE r.value IS NOT NULL
          AND (
                $year IS NULL OR
                toString(r.date) CONTAINS $year OR
                toString(r.period) CONTAINS $year
              )
        RETURN o.name AS metric_name, r.value AS value, r.unit AS unit, r.period AS period, r.date AS date, type(r) AS rel
        """
        rows = session.run(query, year=year)

        candidates = []
        evidence = []

        for row in rows:
            metric_name = row["metric_name"]
            metric_value = parse_numeric(row["value"])
            if metric_value is None:
                continue

            # exclude the target metric itself
            if metric_ref.lower() in metric_name.lower():
                continue

            rel_name = str(row["rel"]).lower()
            name_lower = str(metric_name).lower()

            # keep finance-style likely drivers
            good_terms = [
                "income", "sales", "margin", "earnings", "expense",
                "tax", "comprehensive", "operating"
            ]
            if any(t in name_lower for t in good_terms) or any(t in rel_name for t in good_terms):
                candidates.append({
                    "metric_name": metric_name,
                    "value": metric_value,
                    "unit": row["unit"],
                    "period": row["period"],
                    "date": row["date"],
                    "rel": row["rel"]
                })

                evidence.append(
                    f"FACT: {metric_name} | Value: {format_value_unit(row['value'], row['unit'])} "
                    f"| Time: {row['period'] or ''} {row['date'] or ''} | Rel: {row['rel']}"
                )

        candidates = sorted(candidates, key=lambda x: x["value"], reverse=True)[:10]

        return {
            "mode": "driver_analysis",
            "success": True,
            "target_metric": metric_ref,
            "year": year,
            "results": candidates,
            "evidence": deduplicate_facts(evidence)[:20]
        }


def write_driver_answer(user_query: str, driver_result: dict):
    results = driver_result.get("results", [])
    target_metric = driver_result.get("target_metric")
    year = driver_result.get("year")

    if not results:
        return f"I could not identify clear grounded financial drivers for {target_metric} from the available graph facts."

    lines = []
    for r in results[:5]:
        lines.append(
            f"- {r['metric_name']}: {r['value']} {r['unit'] or ''} "
            f"({r['period'] or ''} {r['date'] or ''})"
        )

    prompt = (
        f"User question: {user_query}\n\n"
        f"Target metric: {target_metric}\n"
        f"Year: {year or 'not specified'}\n\n"
        f"Related financial metrics:\n" + "\n".join(lines) + "\n\n"
        "Write a concise answer. Do not claim strict causality unless the evidence clearly states it. "
        "Use wording like 'likely related metrics' or 'main associated financial factors' if needed. "
        "Do not invent facts."
    )

    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a cautious financial analyst. Use only the structured evidence provided."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Driver answer generation failed: {e}"


# --- 11. LLM WRITER FOR REASONING RESULTS ---
def write_reasoning_answer(user_query: str, reasoning_result: dict):
    if not reasoning_result["success"]:
        return reasoning_result["message"]

    results = reasoning_result["results"]
    ref_metric = reasoning_result["reference_metric"]
    ref_value = reasoning_result["reference_value"]
    ref_unit = reasoning_result["reference_unit"] or ""
    year = reasoning_result["year"]

    if not results:
        return (
            f"I found {ref_metric} for {year} with a value of {ref_value} {ref_unit}, "
            f"but I did not find other metrics in the same period with values matching the requested comparison."
        )

    lines = []
    for r in results[:10]:
        lines.append(
            f"- {r['metric_name']}: {r['value']} {r['unit'] or ''} "
            f"({r['period'] or ''} {r['date'] or ''})"
        )

    prompt = (
        f"User question: {user_query}\n\n"
        f"Reference metric: {ref_metric} = {ref_value} {ref_unit} in {year}\n\n"
        f"Matching results:\n" + "\n".join(lines) + "\n\n"
        "Write a concise financial analyst answer using only these results. "
        "Do not invent facts. Preserve units exactly as given."
    )

    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a concise financial analyst. Use only the provided structured results."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Reasoning answer generation failed: {e}"


# --- 12. UI LOGIC ---
st.set_page_config(page_title="GraphRAG Analyst", layout="wide")
st.title("🍏 Apple 10-Q Graph Explorer")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("evidence"):
            with st.expander("🔍 View Grounded Evidence"):
                st.code(msg["evidence"])
        if msg.get("debug"):
            with st.expander("🛠 Query Optimization Details"):
                st.json(msg["debug"])

if prompt := st.chat_input("Ask about Net Sales, Assets, Liabilities, drivers, or comparisons..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Searching Graph..."):
        debug_info = {"original_query": prompt}

        if is_driver_query(prompt):
            driver_result = run_driver_analysis(prompt)
            if driver_result:
                response_text = write_driver_answer(prompt, driver_result)
                evidence_text = "\n".join(driver_result.get("evidence", []))
                debug_info["mode"] = "driver_analysis"
                debug_info["target_metric"] = driver_result.get("target_metric")
                debug_info["year"] = driver_result.get("year")
                debug_info["result_count"] = len(driver_result.get("results", []))
            else:
                response_text = "I could not determine the target metric for the driver analysis."
                evidence_text = None
                debug_info["mode"] = "driver_analysis_failed"

        elif is_reasoning_query(prompt):
            reasoning_result = run_reasoning_query(prompt)

            if reasoning_result:
                response_text = write_reasoning_answer(prompt, reasoning_result)
                evidence_text = "\n".join(reasoning_result.get("evidence", []))
                debug_info["mode"] = "graph_reasoning"
                debug_info["reasoning_result_summary"] = {
                    "success": reasoning_result.get("success"),
                    "reference_metric": reasoning_result.get("reference_metric"),
                    "reference_value": reasoning_result.get("reference_value"),
                    "year": reasoning_result.get("year"),
                    "result_count": len(reasoning_result.get("results", [])) if reasoning_result.get("results") else 0
                }
            else:
                graph_data, rewritten, expanded_queries = get_grounded_context(prompt)
                debug_info["mode"] = "standard_graph_rag_fallback"
                debug_info["rewritten_query"] = rewritten
                debug_info["expanded_queries"] = expanded_queries

                if not graph_data:
                    response_text = (
                        "I couldn't find enough grounded graph evidence for that question. "
                        "Try being more specific, such as naming the metric and year."
                    )
                    evidence_text = None
                else:
                    completion = client.chat.completions.create(
                        model="gpt-4o",
                        temperature=0,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a Professional Financial Analyst.\n"
                                    "- Use only the FACT CONTEXT provided.\n"
                                    "- If the facts contain multiple dates, compare them only if clearly supported.\n"
                                    "- If the answer is not supported by the facts, say so clearly.\n"
                                    "- Be concise."
                                )
                            },
                            {
                                "role": "user",
                                "content": f"FACT CONTEXT:\n{graph_data}\n\nQUESTION: {prompt}"
                            }
                        ]
                    )
                    response_text = completion.choices[0].message.content
                    evidence_text = graph_data

        else:
            graph_data, rewritten, expanded_queries = get_grounded_context(prompt)
            debug_info["mode"] = "standard_graph_rag"
            debug_info["rewritten_query"] = rewritten
            debug_info["expanded_queries"] = expanded_queries

            if not graph_data:
                response_text = (
                    "I couldn't find enough grounded graph evidence for that question. "
                    "Try being more specific, such as naming the metric and year."
                )
                evidence_text = None
            else:
                try:
                    completion = client.chat.completions.create(
                        model="gpt-4o",
                        temperature=0,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a Professional Financial Analyst.\n"
                                    "- Use only the FACT CONTEXT provided.\n"
                                    "- Preserve units exactly as given.\n"
                                    "- If the answer is not supported by the facts, say so clearly.\n"
                                    "- If multiple periods appear, mention them clearly rather than merging them incorrectly.\n"
                                    "- Be concise."
                                )
                            },
                            {
                                "role": "user",
                                "content": f"FACT CONTEXT:\n{graph_data}\n\nQUESTION: {prompt}"
                            }
                        ]
                    )
                    response_text = completion.choices[0].message.content
                    evidence_text = graph_data
                except Exception as e:
                    response_text = f"Generation failed: {e}"
                    evidence_text = graph_data

    assistant_payload = {
        "role": "assistant",
        "content": response_text,
        "evidence": evidence_text,
        "debug": debug_info
    }

    st.session_state.messages.append(assistant_payload)

    with st.chat_message("assistant"):
        st.markdown(response_text)

        if evidence_text:
            with st.expander("🔍 View Grounded Evidence"):
                st.code(evidence_text)

        with st.expander("🛠 Query Optimization Details"):
            st.json(debug_info)
