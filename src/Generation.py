import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datasets import load_dataset
from evaluate import load
from huggingface_hub import InferenceClient
import huggingface_hub
from tqdm import tqdm
import time
import ast
import logging
from Corpus import CodeNormalizer, ProgrammingSolutionsCorpus
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

# Generate a random number (for example, between 1000 and 9999)
random_number = random.randint(1000, 9999)
print("Random number: ", random_number)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/rag_generation_evaluation_{random_number}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Enable code evaluation safely
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

class CodeGenerator:
    """Class to generate code using a large language model with RAG."""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-70B-Instruct"):
        """Initialize the code generator with a specific LLM model."""
        hf_api_key = os.getenv("HF_API_KEY")
        if hf_api_key is None:
            raise ValueError("Must provide HuggingFace API key as HF_API_KEY environment variable")
        
        self.model_name = model_name
        self.client = huggingface_hub.InferenceClient(api_key=hf_api_key)
        if "llama" not in self.model_name.lower():
            self.pipe = pipeline("text-generation", model="Qwen/CodeQwen1.5-7B-Chat")
    
    def _create_prompt(self, query: str, retrieved_codes: List[str], k: int, reverse_order: bool = False) -> str:
        """Create a prompt for code generation using the query and retrieved examples.
        
        Args:
            query: The coding problem to solve
            retrieved_codes: List of code examples to include in the prompt
            k: Maximum number of examples to include
            reverse_order: If True, reverses the order of examples (to test recency bias)
        """
        examples = retrieved_codes[:k]
        
        # Reverse order if specified (to test recency bias)
        if reverse_order:
            examples = examples[::-1]
            
        examples_text = "\n\n".join([f"Example {i+1}:\n```python\n{code}\n````" for i, code in enumerate(examples)])
        
        if k == 0:
            prompt = f"""Write this Python function:
{query}
Please write a correct, efficient Python function, of the same name that solves the given problem. Return only the python function after "```python", without the docstring. Have import statements if necessary. 
"""
            return prompt
        else:
            prompt = f"""Write this Python function:
    {query}
    Here are {min(k, len(examples))} similar solutions that might help:
    {examples_text}
    Please write a correct, efficient Python function, of the same name that solves the given problem. Return only the python function after "```python", without the docstring. Have import statements if necessary. 
    """
            return prompt
    
    def generate_code(self, query: str, retrieved_codes: List[str], k: int, reverse_order: bool = False, max_retries: int = 3) -> str:
        """Generate code using the LLM with RAG context.
        
        Args:
            query: The coding problem to solve
            retrieved_codes: List of code examples to use as context
            k: Maximum number of examples to include
            reverse_order: If True, reverses the order of examples (to test recency bias)
            max_retries: Maximum number of retry attempts on API failure
        """
        prompt = self._create_prompt(query, retrieved_codes, k, reverse_order)
        if k == 1:
            print("Retrieved Documents: ", retrieved_codes)

        # Check if using a Llama model
        if "llama" in self.model_name.lower():
            # Use InferenceClient for Llama models
            for attempt in range(max_retries):
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=512,
                        temperature=0.2,
                        stop=["````"]
                    )
                    #print("Prompt: ", prompt)
                    generated_code = completion.choices[0].message.content.strip()
                    
                    # Apply the same code extraction logic
                    return self._extract_code(generated_code)
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to generate code: {e}")
                        return ""
                    wait_time = 2 ** attempt
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        else:
            # For other models, use the pipeline
            for attempt in range(max_retries):
                try:
                    messages = [{"role": "user", "content": prompt}]
                    # Generate response without stop parameter since it's not supported
                    response = self.pipe(messages, max_new_tokens=512, temperature=0.2)
                    generated_code =  response[0]["generated_text"][1]['content']
                    # print("Generated code: ", generated_code)
                    return self._extract_code(generated_code)
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to generate code with pipeline model: {e}")
                        return ""
                    wait_time = 2 ** attempt
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        
        return ""


    def _extract_code(self, generated_text: str) -> str:
        """Extract code from generated text by handling various markdown formats.
        
        Args:
            generated_text: Raw text generated by the model
            
        Returns:
            Extracted code with markdown and unnecessary markers removed
        """
        if "```python" in generated_text:
            code_parts = generated_text.split("```python", 1)[1]
            if "```" in code_parts:
                extracted_code = code_parts.split("```", 1)[0].strip()
            else:
                extracted_code = code_parts.strip()
        # If no python marker but still has backticks
        elif generated_text.startswith("```") and "```" in generated_text[3:]:
            extracted_code = generated_text.split("```", 2)[1].strip()
        else:
            extracted_code = generated_text.strip()
        
        # Final cleanup - ensure no trailing backticks
        if extracted_code.endswith("```"):
            extracted_code = extracted_code[:-3].strip()
        
        return extracted_code

class CodeRepository:
    """Class to load and manage the code corpus."""
    
    def __init__(self, corpus: str = "code-rag-bench/programming-solutions", normalization_type: str = "none"):
        """Initialize the code repository with HumanEval and potentially other code sources."""
        self.corpus = load_dataset(corpus)
        self.code_cache = {}  # Cache for code content by ID
        self.normalization_type = normalization_type
        self.code_normalizer = ProgrammingSolutionsCorpus()  # Initialize the code normalizer
        logger.info(f"Loaded corpus with {sum(len(self.corpus[split]) for split in self.corpus)} entries")
        logger.info(f"Using normalization type: {self.normalization_type}")
        
    def get_solution_by_id(self, doc_id: str) -> str:
        """Get the code content for a document ID with the current normalization type."""
        # Create a cache key that includes normalization type
        cache_key = f"{doc_id}_{self.normalization_type}"
        
        # Check cache first
        if cache_key in self.code_cache:
            return self.code_cache[cache_key]
        
        # Search through all splits in the corpus
        for split in self.corpus:
            for item in self.corpus[split]:
                # Check meta column for direct task_id match
                if "meta" in item and isinstance(item["meta"], dict) and "task_id" in item["meta"]:
                    if item["meta"]["task_id"] == doc_id:
                        # Extract the raw code content
                        if "text" in item:
                            raw_code = item["text"]
                            
                            # Determine the task type for proper normalization
                            task_type = "humaneval" if "HumanEval" in doc_id else "mbpp"
                            
                            # Apply normalization based on the specified type
                            normalized_code = self.code_normalizer.normalize_code(
                                raw_code, 
                                normalize_type=self.normalization_type,
                                task=task_type
                            )
                            
                            # Cache the normalized code
                            self.code_cache[cache_key] = normalized_code
                            return normalized_code
        
        # Log the miss but don't throw an error
        logger.info(f"No solution found for ID {doc_id} with normalization {self.normalization_type}")
        self.code_cache[cache_key] = ""  # Cache the miss
        return ""

class HumanEvalEvaluator:
    """Class to evaluate generated code on HumanEval benchmarks."""
    
    def __init__(self):
        """Initialize the evaluator with the HumanEval dataset and code_eval metric."""
        self.dataset = load_dataset("openai_humaneval")
        self.code_eval = load("code_eval")
    
    def get_problem_by_task_id(self, task_id: str) -> Dict[str, Any]:
        """Get the problem details by task_id."""
        # Normalize task_id format (HumanEval/X or just X)
        if not task_id.startswith("HumanEval/"):
            try:  
                # If it's just a number, add the prefix
                task_id_int = int(task_id)
                task_id = f"HumanEval/{task_id_int}"
            except ValueError:
                # If it's not a number, assume it's already in the right format
                pass
        
        for item in self.dataset["test"]:
            if item["task_id"] == task_id:
                return item
        
        # If not found, try searching more flexibly
        if task_id.startswith("HumanEval/"):
            # Try without the prefix
            task_id_stripped = task_id[len("HumanEval/"):]
            for item in self.dataset["test"]:
                if item["task_id"].endswith(task_id_stripped):
                    return item
        
        raise ValueError(f"Task ID {task_id} not found in HumanEval dataset")
    
    def evaluate_solution(self, task_id: str, generated_code: str) -> Dict[str, Any]:
        """Evaluate a generated solution against the test cases."""
        problem = self.get_problem_by_task_id(task_id)
        
        # Prepare the test cases
        test_cases = [problem["test"]]
        
        # Check if entry point is in the generated code
        entry_point = problem["entry_point"]
        if not any(line.strip().startswith(f"def {entry_point}") for line in generated_code.split('\n')):
            logger.warning(f"Entry point '{entry_point}' not found in solution for {task_id}")
            return {
                "task_id": task_id,
                "pass@1": 0.0,
                "results": "entry point not found"
            }
        
        # Compute pass@k metrics
        candidates = [[generated_code]]
        try:
            pass_at_k, results = self.code_eval.compute(
                references=test_cases,
                predictions=candidates,
                k=[1]
            )
            
            return {
                "task_id": task_id,
                "pass@1": pass_at_k["pass@1"],
                "results": results
            }
        except Exception as e:
            logger.error(f"Error evaluating solution for {task_id}: {e}")
            return {
                "task_id": task_id,
                "pass@1": 0.0,
                "results": str(e)
            }

class JSONInputProcessor:
    """Process JSON input files with reranking information."""
    
    def __init__(self, json_path: str):
        """Initialize the processor with the path to the JSON file."""
        self.data = self._load_json(json_path)
        logger.info(f"Loaded JSON input with {len(self.data)} queries")
        
    def _load_json(self, json_path: str) -> List[Dict]:
        """Load and parse the JSON input file."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Error loading JSON file {json_path}: {e}")
            raise
    
    def get_retrieved_docs(self, query_id: int, k: int) -> Tuple[str, List[str]]:
        """Get the query and top-k retrieved documents for a query ID.
        
        Args:
            query_id: The ID of the query
            k: Number of top documents to retrieve
            
        Returns:
            Tuple of (query_text, list_of_doc_ids)
        """
        for item in self.data:
            if item["query_id"] == query_id:
                query = item["query"]
                docs = item["reranked_docs"][:k]
                return query, docs
        
        raise ValueError(f"Query ID {query_id} not found in the JSON data")
    
    def get_all_query_ids(self) -> List[int]:
        """Get all query IDs from the JSON data."""
        return [item["query_id"] for item in self.data]
    
    def get_true_id(self, query_id: int) -> str:
        """Get the true ID for a query ID."""
        for item in self.data:
            if item["query_id"] == query_id:
                return item.get("true_id", "")
        return ""

class RAGEvaluator:
    """Class to orchestrate the RAG process and evaluation using JSON input."""
    
    def __init__(
        self, 
        json_input_path: str,
        model_name: str = "meta-llama/Llama-3.1-70B-Instruct",
        output_dir: str = "results/generation/outputs",
        metric_dir: str = "results/generation/metrics",
        compare_dir: str = "results/generation/recency_bias",
    ):
        """Initialize the RAG evaluator with JSON input."""
        self.json_input_path = json_input_path
        self.model_name = model_name
        self.json_processor = JSONInputProcessor(json_input_path)
        self.generator = CodeGenerator(model_name=model_name)
        logger.info(f"Generation model: {model_name}")
        self.evaluator = HumanEvalEvaluator()
        
        # Extract normalization type from the JSON filename
        normalization_type = self._extract_normalization_type(json_input_path)
        logger.info(f"Extracted normalization type from filename: {normalization_type}")
        
        self.code_repo = CodeRepository(normalization_type=normalization_type)
        self.output_dir = output_dir
        self.metric_dir = metric_dir
        self.compare_dir = compare_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(compare_dir, exist_ok=True)
        os.makedirs(compare_dir + "/pivot", exist_ok=True)
        os.makedirs(metric_dir, exist_ok=True)
        
        logger.info(f"Initialized RAG evaluator with JSON input from {json_input_path}")
    
    def _extract_normalization_type(self, json_path: str) -> str:
        """Extract normalization type from the JSON filename."""
        try:
            # Parse from pattern: docs_dataset_model_embeddings_normtype_k{k}.json
            filename = os.path.basename(json_path)
            parts = filename.split('_')
            
            # The normalization type should be the second-to-last part before the k{number}
            if len(parts) >= 5:
                for i, part in enumerate(parts):
                    if part.startswith('k') and i > 0:
                        norm_type = parts[i-1]
                        # Validate normalization type
                        valid_types = ["none", "docstring", "variables", "functions", "both"]
                        if norm_type in valid_types:
                            return norm_type
            
            logger.warning(f"Could not extract normalization type from filename: {filename}, using 'none'")
            return "none"
        except Exception as e:
            logger.error(f"Error extracting normalization type: {e}")
            return "none"
    
    def _get_output_path(self) -> str:
        """Generate a CSV output path with the same base name as the JSON input."""
        json_basename = os.path.basename(self.json_input_path)
        csv_basename = json_basename.replace('.json', '')
        return os.path.join(self.output_dir, csv_basename)
        
    def run_evaluation(
        self, 
        k_values: List[int] = [1, 5, 10], 
        reverse_order: bool = False,
        query_ids: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Run the RAG evaluation for different k values using JSON input."""
        results = []
        
        # Get all query IDs if not specified
        if query_ids is None:
            query_ids = self.json_processor.get_all_query_ids()
        
        for query_id in tqdm(query_ids, desc="Processing queries"):
            try:
                # Get the true ID for evaluation
                true_id = self.json_processor.get_true_id(query_id)
                
                for k in k_values:
                    # Create a descriptive suffix for the ordering type
                    order_suffix = "reversed" if reverse_order else "normal"
                    logger.info(f"Processing query {query_id} with k={k} (order: {order_suffix})")
                    
                    # Get query and retrieved document IDs
                    query, doc_ids = self.json_processor.get_retrieved_docs(query_id, k)
                    
                    if not doc_ids and k > 0:
                        logger.warning(f"No documents retrieved for query {query_id}, k={k}")
                        continue
                    
                    retrieved_codes = []
                    if k != 0:
                        # Get actual code content for each document ID
                        for doc_id in doc_ids:
                            code = self.code_repo.get_solution_by_id(doc_id)
                            if code:
                                retrieved_codes.append(code)
                        
                        if not retrieved_codes:
                            logger.warning(f"No code content found for documents of query {query_id}, k={k}")
                            continue
                    
                    # Generate code
                    generated_code = self.generator.generate_code(query, retrieved_codes, k, reverse_order)
                    print("Generated code: ", generated_code)
                    
                    if not generated_code:
                        logger.warning(f"No code generated for query {query_id}, k={k}")
                        continue
                    
                    # Evaluate solution
                    eval_result = self.evaluator.evaluate_solution(true_id, generated_code)
                    
                    # Store results
                    result_entry = {
                        "query_id": query_id,
                        "true_id": true_id,
                        "k": k,
                        "order_type": order_suffix,
                        "retrieved_docs": doc_ids,
                        "generated_code": generated_code,
                        "pass@1": eval_result["pass@1"]
                    }
                    
                    # Store detailed results if they're not too large
                    if isinstance(eval_result["results"], str) or len(str(eval_result["results"])) < 1000:
                        result_entry["result_details"] = str(eval_result["results"])
                    else:
                        result_entry["result_details"] = "Results too large to store"
                    
                    results.append(result_entry)
                    
                    global random_number
                    # Save intermediate results frequently
                    if len(results) % 10 == 0:
                        interim_df = pd.DataFrame(results)
                        interim_path = f"results/generation/interim_results_{random_number}.csv"
                        interim_df.to_csv(interim_path, index=False)
                        
            except Exception as e:
                logger.error(f"Error processing query {query_id}: {e}")
                results.append({
                    "query_id": query_id,
                    "true_id": self.json_processor.get_true_id(query_id),
                    "k": "error",
                    "order_type": "error",
                    "retrieved_docs": [],
                    "generated_code": "",
                    "pass@1": 0.0,
                    "result_details": str(e)
                })
        
        # Convert results to dataframe
        results_df = pd.DataFrame(results)
        
        # Save results with the same name as the input JSON file
        order_type = "reversed" if reverse_order else "normal"
        csv_output_path = self._get_output_path() + "_" + order_type + ".csv"
        results_df.to_csv(csv_output_path, index=False)
        logger.info(f"Results saved to {csv_output_path}")
        
        # Calculate aggregated metrics
        metrics = self._calculate_metrics(results_df)
        metrics_basename = csv_output_path.split('/')[-1]
        metrics_path = os.path.join(self.metric_dir, f"metrics_{metrics_basename}")
        metrics.to_csv(metrics_path, index=False)
        logger.info(f"Evaluation complete. Results saved to {metrics_path} and {csv_output_path}")
        return results_df
    
    def _calculate_metrics(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate aggregated metrics from the results."""
        metrics = []
        
        # Convert k to numeric, filtering out error entries
        valid_results = results_df[results_df["k"] != "error"].copy()
        valid_results["k"] = pd.to_numeric(valid_results["k"])
        
        # Group by both k and order_type if present
        if "order_type" in valid_results.columns:
            for (k, order_type), group in valid_results.groupby(["k", "order_type"]):
                metrics.append({
                    "k": int(k),
                    "order_type": order_type,
                    "num_problems": len(group),
                    "pass@1": group["pass@1"].mean(),
                    "passed_count": (group["pass@1"] > 0).sum(),
                    "passed_percentage": (group["pass@1"] > 0).mean() * 100
                })
        else:
            # Backward compatibility for older results
            for k, group in valid_results.groupby("k"):
                metrics.append({
                    "k": int(k),
                    "num_problems": len(group),
                    "pass@1": group["pass@1"].mean(),
                    "passed_count": (group["pass@1"] > 0).sum(),
                    "passed_percentage": (group["pass@1"] > 0).mean() * 100
                })
        
        metrics_df = pd.DataFrame(metrics)
        logger.info(f"Metrics:\n{metrics_df.to_string()}")
        return metrics_df

    def compare_orderings(self, k_values: List[int] = [1, 5, 10], query_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Run evaluation with both normal and reversed ordering to compare recency bias effects.
        
        Args:
            k_values: List of k values to evaluate
            query_ids: Optional list of specific query IDs to evaluate (default: all)
            
        Returns:
            DataFrame with comparison results
        """
        logger.info("Running evaluation with normal document ordering...")
        normal_results = self.run_evaluation(k_values, reverse_order=False, query_ids=query_ids)
        
        logger.info("Running evaluation with reversed document ordering...")
        reversed_results = self.run_evaluation(k_values, reverse_order=True, query_ids=query_ids)
        
        # Combine results and calculate comparison metrics
        normal_metrics = self._calculate_metrics(normal_results)
        normal_metrics["ordering"] = "normal"
        
        reversed_metrics = self._calculate_metrics(reversed_results)
        reversed_metrics["ordering"] = "reversed"
        
        combined_metrics = pd.concat([normal_metrics, reversed_metrics])
        
        # Save comparison results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        comparison_path = os.path.join(self.compare_dir, f"recency_bias_comparison_{timestamp}.csv")
        combined_metrics.to_csv(comparison_path, index=False)
        
        # Create a pivot table for easier comparison
        if "order_type" in combined_metrics.columns:
            pivot = pd.pivot_table(
                combined_metrics, 
                values=["pass@1", "passed_percentage"],
                index=["k"], 
                columns=["order_type"]
            )
        else:
            pivot = pd.pivot_table(
                combined_metrics, 
                values=["pass@1", "passed_percentage"],
                index=["k"], 
                columns=["ordering"]
            )
            
        pivot_path = os.path.join(self.compare_dir+"/pivot", f"recency_bias_pivot_{timestamp}.csv")
        pivot.to_csv(pivot_path)
        
        logger.info(f"Comparison complete. Results saved to {comparison_path}")
        logger.info(f"Pivot table saved to {pivot_path}")
        
        return combined_metrics



def main():
    """Main function to run the RAG evaluation."""
    # Check for API key
    if "HF_API_KEY" not in os.environ:
        raise ValueError("Please set HF_API_KEY environment variable")
    
    import argparse
    parser = argparse.ArgumentParser(description="Run RAG evaluation on HumanEval")
    parser.add_argument(
        "--json_input", 
        type=str, 
        required=True,
        help="Path to the JSON input file with reranking information"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="meta-llama/Llama-3.1-70B-Instruct",
        help="LLM model to use for code generation"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results/generation/outputs",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--k_values", 
        type=str, 
        default="1,5,10",
        help="Comma-separated list of k values to evaluate"
    )
    parser.add_argument(
        "--reverse_order", 
        action="store_true",
        help="Reverse the order of examples (to test recency bias)"
    )
    parser.add_argument(
        "--compare_orderings", 
        action="store_true",
        help="Run evaluation with both normal and reversed ordering to compare recency bias"
    )
    parser.add_argument(
        "--query_ids", 
        type=str, 
        default=None,
        help="Comma-separated list of query IDs to evaluate (default: all)"
    )
    
    args = parser.parse_args()
    k_values = [int(k) for k in args.k_values.split(',')]
    
    # Parse query IDs if provided
    query_ids = None
    if args.query_ids:
        query_ids = [int(qid) for qid in args.query_ids.split(',')]
    
    # Initialize evaluator (it will extract the normalization type from the JSON filename)
    evaluator = RAGEvaluator(
        json_input_path=args.json_input,
        model_name=args.model_name,
        output_dir=args.output_dir
    )
    
    # Run evaluation
    if args.compare_orderings:
        evaluator.compare_orderings(k_values=k_values, query_ids=query_ids)
    else:
        evaluator.run_evaluation(
            k_values=k_values, 
            reverse_order=args.reverse_order, 
            query_ids=query_ids
        )

if __name__ == "__main__":
    main()

## Llama 3.1 70B - Instruct
# python src/Generation.py --json_input /home/gganeshl/Code-aware-reranking/results/humaneval_best_saved/retrieved_docs/docs_openai_humaneval_Llama-3.1-8B-Instruct_GIST-large-Embedding-v0_none_k5.json --k_values 1,5,10
# python src/Generation.py --json_input /home/gganeshl/Code-aware-reranking/results/humaneval_best_saved/retrieved_docs/docs_openai_humaneval_Llama-3.1-8B-Instruct_GIST-large-Embedding-v0_docstring_k5.json --k_values 1,5,10
# python src/Generation.py --json_input /home/gganeshl/Code-aware-reranking/results/humaneval_best_saved/retrieved_docs/docs_openai_humaneval_Llama-3.1-8B-Instruct_GIST-large-Embedding-v0_variables_k5.json --k_values 1,5,10
# python src/Generation.py --json_input /home/gganeshl/Code-aware-reranking/results/humaneval_best_saved/retrieved_docs/docs_openai_humaneval_Llama-3.1-8B-Instruct_GIST-large-Embedding-v0_functions_k5.json --k_values 1,5,10
# python src/Generation.py --json_input /home/gganeshl/Code-aware-reranking/results/humaneval_best_saved/retrieved_docs/docs_openai_humaneval_Llama-3.1-8B-Instruct_GIST-large-Embedding-v0_both_k5.json --k_values 0,1,5,10

## Llama 3.1 8B Instruct
# python src/Generation.py --json_input /home/gganeshl/Code-aware-reranking/results/humaneval_best_saved/retrieved_docs/docs_openai_humaneval_Llama-3.1-8B-Instruct_GIST-large-Embedding-v0_none_k5.json --k_values 1,5,10 --model_name meta-llama/Llama-3.1-8B-Instruct
# python src/Generation.py --json_input /home/gganeshl/Code-aware-reranking/results/humaneval_best_saved/retrieved_docs/docs_openai_humaneval_Llama-3.1-8B-Instruct_GIST-large-Embedding-v0_docstring_k5.json --k_values 1,5,10 --model_name meta-llama/Llama-3.1-8B-Instruct
# python src/Generation.py --json_input /home/gganeshl/Code-aware-reranking/results/humaneval_best_saved/retrieved_docs/docs_openai_humaneval_Llama-3.1-8B-Instruct_GIST-large-Embedding-v0_variables_k5.json --k_values 1,5,10 --model_name meta-llama/Llama-3.1-8B-Instruct
# python src/Generation.py --json_input /home/gganeshl/Code-aware-reranking/results/humaneval_best_saved/retrieved_docs/docs_openai_humaneval_Llama-3.1-8B-Instruct_GIST-large-Embedding-v0_functions_k5.json --k_values 1,5,10 --model_name meta-llama/Llama-3.1-8B-Instruct
# python src/Generation.py --json_input /home/gganeshl/Code-aware-reranking/results/humaneval_best_saved/retrieved_docs/docs_openai_humaneval_Llama-3.1-8B-Instruct_GIST-large-Embedding-v0_both_k5.json --k_values 0,1,5,10 --model_name meta-llama/Llama-3.1-8B-Instruct

