import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from evaluate import load
from huggingface_hub import InferenceClient
from tqdm import tqdm
import time
import ast
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_evaluation.log"),
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
        self.client = InferenceClient(api_key=hf_api_key)
    
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
            
        examples_text = "\n\n".join([f"Example {i+1}:\n```python\n{code}\n```" for i, code in enumerate(examples)])
        
        prompt = f"""I need you to write a Python function to solve the following problem:

{query}

Here are {min(k, len(examples))} example solutions for similar problems that might help:

{examples_text}

Based on the problem description and these examples, please write a Python function that solves the given problem. 
Your solution should be correct, efficient, and follow good coding practices.

```python
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
        
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=512,
                    temperature=0.2,
                    stop=["```"]
                )
                
                generated_code = completion.choices[0].message.content.strip()
                # Remove markdown code blocks if present
                if generated_code.startswith("```python"):
                    generated_code = generated_code[10:].strip()
                if generated_code.endswith("```"):
                    generated_code = generated_code[:-3].strip()
                
                return generated_code
                
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to generate code: {e}")
                    return ""
                wait_time = 2 ** attempt
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        
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

class RAGEvaluator:
    """Class to orchestrate the RAG process and evaluation."""
    
    def __init__(
        self, 
        reranking_results_path: str,
        model_name: str = "meta-llama/Llama-3.1-70B-Instruct",
        output_dir: str = "rag_results"
    ):
        """Initialize the RAG evaluator with reranking results."""
        self.reranking_df = pd.read_csv(reranking_results_path)
        self.generator = CodeGenerator(model_name=model_name)
        self.evaluator = HumanEvalEvaluator()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Verify required columns
        required_cols = ["query_id"]
        missing_cols = [col for col in required_cols if col not in self.reranking_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in reranking results: {missing_cols}")
        
        logger.info(f"Loaded reranking results with {len(self.reranking_df)} rows")
        logger.info(f"Columns: {self.reranking_df.columns.tolist()}")
    
    def _parse_retrieved_codes(self, row: pd.Series, k: int) -> List[str]:
        """Parse retrieved codes from the dataframe row for a given k."""
        column_name = f"k={k}"
        
        if column_name not in row:
            logger.warning(f"Column '{column_name}' not found in row")
            return []
        
        retrieved_codes = row[column_name]
        
        # Handle different formats of retrieved codes in the dataframe
        if pd.isna(retrieved_codes):
            return []
        
        # If it's already a list, return it
        if isinstance(retrieved_codes, list):
            return retrieved_codes
        
        # If it's a string, try to parse it as a list
        if isinstance(retrieved_codes, str):
            try:
                # Try parsing as JSON
                parsed = json.loads(retrieved_codes)
                if isinstance(parsed, list):
                    return parsed
                return [parsed]
            except json.JSONDecodeError:
                pass
                
            try:
                # Try parsing as Python literal
                parsed = ast.literal_eval(retrieved_codes)
                if isinstance(parsed, list):
                    return parsed
                return [parsed]
            except (SyntaxError, ValueError):
                # If all else fails, return as single item list
                return [retrieved_codes]
        
        # If it's some other type, return as single item list
        return [str(retrieved_codes)]
    
    def run_evaluation(
        self, 
        k_values: List[int] = [1, 5, 10], 
        reverse_order: bool = False,
        num_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """Run the RAG evaluation for different k values.
        
        Args:
            k_values: List of k values to evaluate (number of examples to use)
            reverse_order: If True, reverses the order of examples to test recency bias
            num_samples: Number of samples to evaluate (default: all)
        """
        results = []
        
        # Limit the number of samples if specified
        df_to_process = self.reranking_df.sample(n=num_samples) if num_samples else self.reranking_df
        
        for idx, row in tqdm(df_to_process.iterrows(), total=len(df_to_process)):
            task_id = row.get("query_id", f"unknown_{idx}")
            
            try:
                # Get problem details
                problem = self.evaluator.get_problem_by_task_id(task_id)
                query = problem["prompt"]
                
                for k in k_values:
                    # Create a descriptive suffix for the ordering type
                    order_suffix = "reversed" if reverse_order else "normal"
                    logger.info(f"Processing task {task_id} with k={k} (order: {order_suffix})")
                    
                    # Get retrieved code examples for this k
                    retrieved_codes = self._parse_retrieved_codes(row, k)
                    logger.info(f"Retrieved {len(retrieved_codes)} examples for k={k}")
                    
                    # Skip if no examples were retrieved
                    if not retrieved_codes:
                        logger.warning(f"No examples retrieved for k={k}, skipping")
                        continue
                    
                    # Generate code
                    generated_code = self.generator.generate_code(query, retrieved_codes, k, reverse_order)
                    
                    # Skip if no code was generated
                    if not generated_code:
                        logger.warning(f"No code generated for task {task_id} with k={k}, skipping")
                        continue
                    
                    # Evaluate solution
                    eval_result = self.evaluator.evaluate_solution(task_id, generated_code)
                    
                    # Store results
                    result_entry = {
                        "task_id": task_id,
                        "k": k,
                        "order_type": order_suffix,
                        "generated_code": generated_code,
                        "pass@1": eval_result["pass@1"]
                    }
                    
                    # Store detailed results if they're not too large
                    if isinstance(eval_result["results"], str) or len(str(eval_result["results"])) < 1000:
                        result_entry["result_details"] = str(eval_result["results"])
                    else:
                        result_entry["result_details"] = "Results too large to store"
                    
                    results.append(result_entry)
                    
                    # Save intermediate results frequently
                    if len(results) % 10 == 0:
                        interim_df = pd.DataFrame(results)
                        interim_path = os.path.join(self.output_dir, "interim_results.csv")
                        interim_df.to_csv(interim_path, index=False)
            
            except Exception as e:
                logger.error(f"Error processing task {task_id}: {e}")
                results.append({
                    "task_id": task_id,
                    "k": "error",
                    "order_type": "error",
                    "generated_code": "",
                    "pass@1": 0.0,
                    "result_details": str(e)
                })
        
        # Convert results to dataframe
        results_df = pd.DataFrame(results)
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        order_type = "reversed" if reverse_order else "normal"
        results_path = os.path.join(self.output_dir, f"rag_evaluation_results_{order_type}_{timestamp}.csv")
        results_df.to_csv(results_path, index=False)
        
        # Calculate aggregated metrics
        metrics = self._calculate_metrics(results_df)
        metrics_path = os.path.join(self.output_dir, f"rag_metrics_{order_type}_{timestamp}.csv")
        metrics.to_csv(metrics_path, index=False)
        
        logger.info(f"Evaluation complete. Results saved to {results_path}")
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

    def compare_orderings(self, k_values: List[int] = [1, 5, 10], num_samples: Optional[int] = None) -> pd.DataFrame:
        """Run evaluation with both normal and reversed ordering to compare recency bias effects.
        
        Args:
            k_values: List of k values to evaluate
            num_samples: Number of samples to evaluate (default: all)
            
        Returns:
            DataFrame with comparison results
        """
        logger.info("Running evaluation with normal document ordering...")
        normal_results = self.run_evaluation(k_values, reverse_order=False, num_samples=num_samples)
        
        logger.info("Running evaluation with reversed document ordering...")
        reversed_results = self.run_evaluation(k_values, reverse_order=True, num_samples=num_samples)
        
        # Combine results and calculate comparison metrics
        normal_metrics = self._calculate_metrics(normal_results)
        normal_metrics["ordering"] = "normal"
        
        reversed_metrics = self._calculate_metrics(reversed_results)
        reversed_metrics["ordering"] = "reversed"
        
        combined_metrics = pd.concat([normal_metrics, reversed_metrics])
        
        # Save comparison results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        comparison_path = os.path.join(self.output_dir, f"recency_bias_comparison_{timestamp}.csv")
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
            
        pivot_path = os.path.join(self.output_dir, f"recency_bias_pivot_{timestamp}.csv")
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
        "--reranking_results", 
        type=str, 
        required=True,
        help="Path to the reranking results CSV file"
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
        default="results/rag_evaluation",
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
        "--num_samples", 
        type=int, 
        default=None,
        help="Number of samples to evaluate (default: all)"
    )
    
    args = parser.parse_args()
    k_values = [int(k) for k in args.k_values.split(',')]
    
    # Initialize evaluator
    evaluator = RAGEvaluator(
        reranking_results_path=args.reranking_results,
        model_name=args.model_name,
        output_dir=args.output_dir
    )
    
    # Run evaluation
    if args.compare_orderings:
        evaluator.compare_orderings(k_values=k_values, num_samples=args.num_samples)
    else:
        evaluator.run_evaluation(
            k_values=k_values, 
            reverse_order=args.reverse_order, 
            num_samples=args.num_samples
        )

if __name__ == "__main__":
    main()