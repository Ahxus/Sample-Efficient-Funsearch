# implementation/efficiency_utils.py

import ast
import hashlib
from typing import Any, Sequence, Tuple, Dict, Optional, Set, Type, TYPE_CHECKING
import traceback
import time
import os
import numpy as np


# Note: Adjust paths if your structure differs slightly
from . import code_manipulation

# Use TYPE_CHECKING to import only for type hints, breaking the runtime cycle
if TYPE_CHECKING:
    from . import evaluator  # Import Sandbox type hint

# Type Aliases for clarity
AstHash = str
FingerprintHash = str
Score = float
FingerprintResult = Tuple[Any, ...] | str  # Tuple of outputs or error string

class NormalizationASTVisitor(ast.NodeTransformer):
    """
    Simple visitor to remove docstrings and potentially other basic normalizations for AST hashing.
    """
    def visit_Expr(self, node: ast.Expr) -> Any:
        # Remove docstrings which appear as Expr -> Constant(string) at the module/class/function level
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            # Check if it's likely a standalone expression docstring (often at start of body)
            # This is a heuristic; might remove intended multiline strings in rare cases.
            # More robust check would involve checking parent node types.
            return None
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        # Remove function docstring specifically
        if (node.body and isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Constant) and
                isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:]
        # Process the rest of the function body
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
         # Handle async functions similarly
        if (node.body and isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Constant) and
                isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:]
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
         # Handle class docstrings
        if (node.body and isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Constant) and
                isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:]
        self.generic_visit(node)
        return node

    # Optional: Add visits for Pass, Break, Continue if you want to normalize their presence/absence
    # def visit_Pass(self, node: ast.Pass) -> Any:
    #     return None # Example: Remove all Pass statements

def calculate_ast_hash(function_body: str) -> Optional[AstHash]:
    """
    Calculates a hash of the normalized AST of the function body.
    Normalization removes docstrings. Uses ast.dump for canonical representation.

    Args:
        function_body: The string containing the function's body (indented).

    Returns:
        A SHA256 hash string if successful, None otherwise.
    """
    if not function_body.strip():
        return None # Handle empty body

    # Wrap body in a dummy function for valid parsing of indented code
    code_to_parse = f"def dummy_func():\n{function_body}"
    try:
        tree = ast.parse(code_to_parse)
        # Apply normalization (remove docstrings)
        normalizer = NormalizationASTVisitor()
        normalized_tree = normalizer.visit(tree)
        ast.fix_missing_locations(normalized_tree) # Important after transformations

        # Use ast.dump for a canonical string representation (more stable than unparse)
        # sort_keys=True helps ensure consistency across Python versions/runs
        ast_dump_str = ast.dump(normalized_tree, sort_keys=True)

        # Calculate SHA256 hash
        hasher = hashlib.sha256()
        hasher.update(ast_dump_str.encode('utf-8'))
        return hasher.hexdigest()
    except SyntaxError:
        # Don't warn on basic syntax errors, LLM might generate invalid code
        return None
    except Exception as e:
        # print(f"Warning: AST parsing/hashing failed for body:\n{function_body}\nError: {e}")
        # traceback.print_exc() # Uncomment for detailed debugging
        return None

# Use string forward reference for Sandbox type hint
def calculate_fingerprint(
    program_str: str,
    sandbox: 'evaluator.Sandbox', # String forward reference
    function_to_run: str,
    function_to_evolve: str,
    full_inputs: Any,
    fingerprint_inputs: Sequence[Any], # Assumed non-empty if called
    timeout_seconds: int
) -> Tuple[Optional[FingerprintHash], FingerprintResult, float]:
    """
    Calculates a functional fingerprint by running the program on a subset of inputs.

    Args:
        program_str: The complete program code as a string.
        sandbox: An instance of the Sandbox to run the code.
        function_to_run: Name of the evaluation function (e.g., 'evaluate').
        function_to_evolve: Name of the function being evolved (e.g., 'priority').
        full_inputs: The original full input data structure passed to the main evaluator.
        fingerprint_inputs: A sequence of specific inputs (e.g., keys or indices from full_inputs)
                           to use for fingerprinting. Must not be empty.
        timeout_seconds: Timeout for *each* individual sandbox run for fingerprinting.

    Returns:
        Tuple: (Fingerprint hash, Tuple of results/errors, Total fingerprint calc time)
               Hash is None if hashing fails. Results tuple always returned.
    """
    start_time = time.time()
    results = []

    for fp_input in fingerprint_inputs:
        try:
            # Execute program via sandbox for one fingerprint input
            output, runs_ok = sandbox.run(
                program=program_str,
                function_to_run=function_to_run,
                function_to_evolve=function_to_evolve,
                inputs=full_inputs,
                test_input=fp_input, # Specific input for this run
                timeout_seconds=timeout_seconds,
                is_fingerprint_run=True # Flag for sandbox optimization/logging
            )

            if runs_ok and output is not None:
                # Ensure output is hashable (convert complex types if necessary)
                if isinstance(output, (list, dict, set)):
                    # Simple, order-dependent string serialization; might need refinement
                    # for sets or complex dicts if order matters for function equivalence.
                    results.append(str(output))
                elif isinstance(output, float) and (output != output or output == float('inf') or output == float('-inf')):
                     # Handle NaN/Infinity as strings for consistent hashing
                     results.append(str(output))
                else:
                    # Assume other basic types (int, float, bool, str) are hashable
                    results.append(output)
            else:
                # Record failure reason (timeout or program error indicated by runs_ok=False)
                results.append(f"RunFail_or_Timeout_{fp_input}")

        except Exception as e:
            # Record exception during sandbox execution for this input
            # print(f"Debug: Exception during fingerprint sandbox run for {fp_input}: {e}") # Optional debug
            results.append(f"Exception_{fp_input}_{type(e).__name__}")

    fingerprint_time = time.time() - start_time
    results_tuple = tuple(results) # Ensure order is preserved

    # Generate hash based on collected results (including error strings)
    try:
        # Convert results tuple to a string representation for hashing
        # Using repr() might be more robust than str() for complex types
        results_repr = repr(results_tuple)

        hasher = hashlib.sha256()
        hasher.update(results_repr.encode('utf-8'))
        fingerprint_hash = hasher.hexdigest()
        return fingerprint_hash, results_tuple, fingerprint_time

    except Exception as e:
        # print(f"Warning: Fingerprint hashing failed. Results: {results_tuple}\nError: {e}")
        # traceback.print_exc()
        return None, results_tuple, fingerprint_time


class DuplicateChecker:
    """Manages caches for AST and Fingerprint hashes to detect duplicates."""

    def __init__(self, log_dir: Optional[str] = None):
        self._ast_cache: Set[AstHash] = set()
        # Cache stores fingerprint hash -> score mapping
        self._fingerprint_cache: Dict[FingerprintHash, Score] = {}
        self.ast_hits = 0
        self.fingerprint_hits = 0
        self.checks_performed = 0
        self._log_file = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True) # Ensure directory exists
            log_path = os.path.join(log_dir, "cache_stats.log")
            try:
                self._log_file = open(log_path, "a") # Append mode
                # Write header if file is new/empty
                if os.path.getsize(log_path) == 0:
                     self._log_file.write("Timestamp,Type,ASTHash,FPHash,TimeSeconds,Score\n")
                     self._log_file.flush()
            except IOError as e:
                print(f"Warning: Could not open cache log file '{log_path}'. Error: {e}")
                self._log_file = None

    def __del__(self):
        # Ensure file is closed on object deletion
        if self._log_file and not self._log_file.closed:
            self._log_file.close()

    # Use string forward reference for Sandbox type hint
    def check_program(
        self,
        function_body: str,
        program_str: str,
        sandbox: 'evaluator.Sandbox', # String forward reference
        function_to_run: str,
        function_to_evolve: str,
        full_inputs: Any,
        fingerprint_inputs: Optional[Sequence[Any]], # Can be None
        fingerprint_timeout_seconds: int,
        # Flags to control checks
        use_ast_check: bool,
        use_fingerprint_check: bool,
    ) -> Tuple[bool, Optional[Score], Optional[AstHash], Optional[FingerprintHash], float]:
        """
        Performs duplicate checks conditionally based on flags.

        Args:
            ... (standard args) ...
            use_ast_check: If True, perform AST hash check.
            use_fingerprint_check: If True, perform Fingerprint check (if AST passes/disabled).

        Returns:
            Tuple: (is_duplicate, cached_score, ast_hash, fingerprint_hash, check_time)
        """
        start_time = time.time()
        self.checks_performed += 1 # Count check attempt
        ast_hash = None
        fingerprint_hash = None
        check_time = 0.0
        cached_score = None
        is_duplicate = False

        # 1. AST Check (Conditional)
        if use_ast_check:
            ast_hash = calculate_ast_hash(function_body)
            if ast_hash and ast_hash in self._ast_cache:
                self.ast_hits += 1
                is_duplicate = True
                cached_score = None # AST hits don't provide a score
                check_time = time.time() - start_time
                self._log_stat(f"AST_HIT,{ast_hash},None,{check_time:.4f},None")
                return is_duplicate, cached_score, ast_hash, fingerprint_hash, check_time

        # 2. Fingerprint Check (Conditional)
        # Only proceed if FP check is enabled, inputs available, AND no AST hit occurred
        if use_fingerprint_check and fingerprint_inputs and not is_duplicate:
            try:
                fp_hash_calc, _, fp_time = calculate_fingerprint(
                    program_str=program_str, sandbox=sandbox, function_to_run=function_to_run,
                    function_to_evolve=function_to_evolve, full_inputs=full_inputs,
                    fingerprint_inputs=fingerprint_inputs, timeout_seconds=fingerprint_timeout_seconds
                )
                fingerprint_hash = fp_hash_calc # Assign the calculated hash
            except Exception as e:
                 # print(f"Error during fingerprint calculation: {e}") # Optional debug
                 fingerprint_hash = None # Ensure hash is None if calculation fails

            if fingerprint_hash and fingerprint_hash in self._fingerprint_cache:
                self.fingerprint_hits += 1
                is_duplicate = True
                cached_score = self._fingerprint_cache[fingerprint_hash]
                check_time = time.time() - start_time # Total time includes AST check if done
                self._log_stat(f"FP_HIT,{ast_hash},{fingerprint_hash},{check_time:.4f},{cached_score}")
                return is_duplicate, cached_score, ast_hash, fingerprint_hash, check_time

        # 3. No duplicate found by enabled checks
        check_time = time.time() - start_time
        if not is_duplicate: # Only log miss if no hit occurred
            self._log_stat(f"MISS,{ast_hash},{fingerprint_hash},{check_time:.4f},None")
        # Return current state (is_duplicate=False if reached here)
        return is_duplicate, cached_score, ast_hash, fingerprint_hash, check_time


    def register_program(self, ast_hash: Optional[AstHash], fingerprint_hash: Optional[FingerprintHash], score: Score) -> None:
        """Adds the hashes of a successfully evaluated program to the caches if not None."""
        registered_ast = False
        registered_fp = False
        # Only add to cache if the hash was actually calculated (i.e., check was likely enabled and successful)
        if ast_hash:
            self._ast_cache.add(ast_hash)
            registered_ast = True
        if fingerprint_hash:
            self._fingerprint_cache[fingerprint_hash] = score
            registered_fp = True

        # Log registration if at least one hash was added
        if registered_ast or registered_fp:
            self._log_stat(f"REGISTER,{ast_hash},{fingerprint_hash},0.0,{score}") # Log 0.0 time for registration event


    def get_stats(self) -> dict:
        """Returns current cache statistics."""
        ast_cache_size = len(self._ast_cache)
        fp_cache_size = len(self._fingerprint_cache)
        # Avoid division by zero
        ast_hit_rate = np.divide(self.ast_hits, self.checks_performed, where=self.checks_performed > 0, out=np.array(0.0)).item()
        # FP hit rate is relative to checks that were *not* AST hits
        non_ast_hit_checks = self.checks_performed - self.ast_hits
        fp_hit_rate = np.divide(self.fingerprint_hits, non_ast_hit_checks, where=non_ast_hit_checks > 0, out=np.array(0.0)).item()
        total_hits = self.ast_hits + self.fingerprint_hits
        total_hit_rate = np.divide(total_hits, self.checks_performed, where=self.checks_performed > 0, out=np.array(0.0)).item()

        return {
            "checks_performed": self.checks_performed,
            "ast_hits": self.ast_hits,
            "fingerprint_hits": self.fingerprint_hits,
            "total_hits": total_hits,
            "ast_cache_size": ast_cache_size,
            "fingerprint_cache_size": fp_cache_size,
            "ast_hit_rate": f"{ast_hit_rate:.2%}",
            "fingerprint_hit_rate (of non-AST hits)": f"{fp_hit_rate:.2%}",
            "total_hit_rate": f"{total_hit_rate:.2%}",
        }

    def _log_stat(self, message: str):
        """Logs a message to the statistics file if configured."""
        if self._log_file and not self._log_file.closed:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            log_entry = f"{timestamp},{message}\n"
            try:
                self._log_file.write(log_entry)
                self._log_file.flush() # Ensure it's written immediately
            except Exception as e:
                print(f"Error writing to cache log: {e}")
                # Optionally try to close and reopen? Or just stop logging.
                try: self._log_file.close()
                except: pass
                self._log_file = None # Stop trying to log