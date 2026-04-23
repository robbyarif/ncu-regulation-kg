"""Minimal KG query template for Assignment 4.

Keep these APIs unchanged for auto-test:
- generate_text(messages, max_new_tokens=220)
- get_relevant_articles(question)
- generate_answer(question, rule_results)

Keep Rule fields aligned with build_kg output:
rule_id, type, action, result, art_ref, reg_name
"""

import os
import json
import sqlite3
from typing import Any

from neo4j import GraphDatabase
from dotenv import load_dotenv

from llm_loader import load_local_llm, get_tokenizer, get_raw_pipeline


# ========== 0) Initialization ==========
load_dotenv()

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = (
	os.getenv("NEO4J_USER", "neo4j"),
	os.getenv("NEO4J_PASSWORD", "password"),
)

# Avoid local proxy settings interfering with model/Neo4j access.
for key in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
	if key in os.environ:
		del os.environ[key]


try:
	driver = GraphDatabase.driver(URI, auth=AUTH)
	driver.verify_connectivity()
except Exception as e:
	print(f"⚠️ Neo4j connection warning: {e}")
	driver = None


# ========== 1) Public API (query flow order) ==========
# Order: extract_entities -> build_typed_cypher -> get_relevant_articles -> generate_answer

def generate_text(messages: list[dict[str, str]], max_new_tokens: int = 220) -> str:
	"""
	Call local HF model via chat template + raw pipeline.

	Interface:
	- Input:
	  - messages: list[dict[str, str]] (chat messages with role/content)
	  - max_new_tokens: int
	- Output:
	  - str (model generated text, no JSON guarantee)
	"""
	tok = get_tokenizer()
	pipe = get_raw_pipeline()
	if tok is None or pipe is None:
		load_local_llm()
		tok = get_tokenizer()
		pipe = get_raw_pipeline()
	prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
	return pipe(prompt, max_new_tokens=max_new_tokens)[0]["generated_text"].strip()


def extract_entities(question: str) -> dict[str, Any]:
	"""Parse question to {question_type, subject_terms, aspect} using LLM."""
	system_prompt = (
		"You are a linguistic analyzer. Extract the search intent from the user's regulation-related question.\n"
		"Return a JSON object with:\n"
		"- question_type: (e.g., penalty, requirement, procedure, fee, credits)\n"
		"- subject_terms: list of key nouns (e.g., ['student ID', 'exam', 'graduation'])\n"
		"- aspect: what specifically is asked (e.g., 'minutes late', 'cost', 'minimum score')\n\n"
		"Question: " + question + "\n"
		"Response format: {\"question_type\": ..., \"subject_terms\": [...], \"aspect\": ...}"
	)
	
	messages = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": question}
	]
	
	response = generate_text(messages, max_new_tokens=150)
	try:
		start = response.find("{")
		end = response.rfind("}") + 1
		if start != -1 and end != -1:
			return json.loads(response[start:end])
	except Exception:
		pass
		
	return {
		"question_type": "general",
		"subject_terms": [question],
		"aspect": "general",
	}


def build_typed_cypher(entities: dict[str, Any]) -> tuple[str, str]:
	"""Return (typed_query, broad_query) with score and required fields."""
	q_type = entities.get("question_type", "general")
	subjects = entities.get("subject_terms", [])
	aspect = entities.get("aspect", "general")
	
	search_text = " ".join(subjects) + " " + aspect
	
	# Typed: Focus on Rule properties matching the extracted type/subject
	cypher_typed = f"""
	CALL db.index.fulltext.queryNodes('rule_idx', $search_text) YIELD node, score
	WHERE node.type CONTAINS $q_type OR any(s IN $subjects WHERE node.subject CONTAINS s OR node.action CONTAINS s)
	RETURN node.rule_id as id, node.type as type, node.action as action, 
	       node.result as result, node.art_ref as art_ref, node.reg_name as reg_name, 
	       'rule' as source_type, score
	ORDER BY score DESC LIMIT 10
	"""

	# Broad: Full-text search on Articles then Rule traversal
	cypher_broad = f"""
	CALL db.index.fulltext.queryNodes('article_content_idx', $search_text) YIELD node as a, score
	MATCH (a)-[:CONTAINS_RULE]->(r:Rule)
	RETURN r.rule_id as id, r.type as type, r.action as action, 
	       r.result as result, r.art_ref as art_ref, r.reg_name as reg_name, 
	       'rule' as source_type, score
	ORDER BY score DESC LIMIT 10
	"""

	return cypher_typed, cypher_broad


def get_relevant_articles(question: str) -> list[dict[str, Any]]:
	"""Run typed+broad KG retrieval + secondary SQLite fallback."""
	if driver is None:
		return []
		
	entities = extract_entities(question)
	typed_q, broad_q = build_typed_cypher(entities)
	
	subjects = entities.get("subject_terms", [])
	aspect = entities.get("aspect", "")
	# Clean search text for Lucene (escape or remove special chars like /)
	search_text = " ".join(subjects) + " " + aspect
	search_text = search_text.replace("/", " ").replace("(", " ").replace(")", " ").replace("-", " ").strip()
	
	params = {
		"search_text": search_text,
		"q_type": entities.get("question_type", ""),
		"subjects": subjects
	}
	
	results = []
	seen_keys = set()
	
	with driver.session() as session:
		# 1) KG Retrieval
		for q in [typed_q, broad_q]:
			res = session.run(q, **params)
			for record in res:
				key = f"{record['reg_name']}-{record['id']}"
				if key not in seen_keys:
					results.append(dict(record))
					seen_keys.add(key)
					
	# 2) Secondary DB fallback (Articles table via keyword search)
	# This improves recall for questions where rules might be too atomic/sparse.
	if len(results) < 5:
		try:
			conn = sqlite3.connect("ncu_regulations.db")
			cursor = conn.cursor()
			# Simple keyword search on content
			kw_query = " OR ".join([f"content LIKE '%{s}%'" for s in subjects if len(s) > 1])
			if kw_query:
				cursor.execute(f"SELECT article_number, content, reg_id FROM articles WHERE {kw_query} LIMIT 5")
				rows = cursor.fetchall()
				for row in rows:
					key = f"db-{row[0]}-{row[2]}"
					if key not in seen_keys:
						results.append({
							"id": row[0],
							"art_ref": row[0],
							"action": "Article Content",
							"result": row[1],
							"reg_name": f"Regulation {row[2]}",
							"source_type": "article",
							"score": 1.0
						})
						seen_keys.add(key)
			conn.close()
		except Exception:
			pass
				
	return results


def generate_answer(question: str, rule_results: list[dict[str, Any]]) -> str:
	"""Generate grounded answer from retrieved rules only."""
	if not rule_results:
		return "I'm sorry, I couldn't find any specific regulations related to your question in the current knowledge graph."

	context_parts = []
	for r in rule_results:
		if r.get("source_type") == "rule":
			part = (
				f"- Rule [{r['id']}] from Article {r['art_ref']} of {r['reg_name']}:\n"
				f"  Condition: {r['action']}\n"
				f"  Result: {r['result']}"
			)
		else:
			part = f"- Article {r['art_ref']} (Direct DB snippet):\n{r['result']}"
		context_parts.append(part)
	
	context_str = "\n\n".join(context_parts[:12]) # Keep context manageable
	
	system_prompt = (
		"You are a helpful university regulation assistant at NCU.\n"
		"Answer the user's question using the provided context below. "
		"The context contains several Rules and Article snippets. Search THROUGH ALL of them carefully to find the answer.\n"
		"If multiple rules are relevant, synthesize them into a complete answer.\n"
		"If the answer is not in any of the provided rules, say you don't know.\n"
		"Be direct and specific (e.g., mention exact times, points, or fees).\n"
		"Always cite the Rule ID (e.g., [R-0001]) or Article number and Regulation name for your evidence.\n\n"
		"Context Evidence:\n"
		+ context_str + "\n\n"
		"Question: " + question
	)
	
	messages = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": question}
	]
	
	return generate_text(messages, max_new_tokens=250)


def main() -> None:
	"""Interactive CLI (provided scaffold)."""
	if driver is None:
		return

	load_local_llm()

	print("=" * 50)
	print("🎓 NCU Regulation Assistant (Template)")
	print("=" * 50)
	print("💡 Try: 'What is the penalty for forgetting student ID?'")
	print("👉 Type 'exit' to quit.\n")

	while True:
		try:
			user_q = input("\nUser: ").strip()
			if not user_q:
				continue
			if user_q.lower() in {"exit", "quit"}:
				print("👋 Bye!")
				break

			results = get_relevant_articles(user_q)
			answer = generate_answer(user_q, results)
			print(f"Bot: {answer}")

		except KeyboardInterrupt:
			print("\n👋 Bye!")
			break
		except NotImplementedError as e:
			print(f"⚠️ {e}")
			break
		except Exception as e:
			print(f"❌ Error: {e}")

	driver.close()


if __name__ == "__main__":
	main()

