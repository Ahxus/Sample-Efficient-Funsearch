# implementation/profile.py

import os.path
from typing import Dict, Optional
import logging
import json
import time
from . import code_manipulation # Use relative import
# Guard against missing tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None # Define as None if import fails
import numpy as np

class Profiler:
    """Profiles FunSearch runs, logging statistics and efficiency metrics."""
    def __init__(
            self,
            log_dir: str | None = None,
            max_log_nums: int | None = None,
            efficiency_enabled: bool = False,
    ):
        """
        Args:
            log_dir: Folder path for tensorboard log files and JSON samples.
            max_log_nums: Stop logging if exceeding max_log_nums samples processed.
            efficiency_enabled: Whether any cache checking mechanism (AST or FP) is active.
        """
        logging.getLogger().setLevel(logging.INFO) # Configure root logger
        self._log_dir = log_dir
        self._json_dir = None
        if log_dir:
             self._json_dir = os.path.join(log_dir, 'samples')
             os.makedirs(self._json_dir, exist_ok=True)

        self._max_log_nums = max_log_nums
        self._num_samples_processed = 0
        self._cur_best_program_sample_order: Optional[int] = None
        self._cur_best_program_score: float = -np.inf
        self._all_sampled_functions: Dict[int, code_manipulation.Function] = {}

        # --- Counters for Efficiency Reporting ---
        self._num_full_evals_success = 0
        self._total_full_eval_time = 0.0
        self._num_failed_evals = 0
        self._total_failed_eval_time = 0.0

        self._num_cache_hits_processed = 0
        self._total_hit_processing_time = 0.0
        self._num_cache_hits_valid_score = 0
        # -----------------------------------------

        self._efficiency_enabled = efficiency_enabled # Store overall status

        self._tot_sample_time = 0.0
        self._writer: Optional[SummaryWriter] = None
        if log_dir and SummaryWriter:
            try:
                self._writer = SummaryWriter(log_dir=log_dir)
            except Exception as e:
                print(f"Warning: Failed to initialize TensorBoard SummaryWriter: {e}")
                self._writer = None
        elif log_dir and not SummaryWriter:
            print("Warning: tensorboard package not found. TensorBoard logging disabled.")


    def _write_tensorboard(self):
        """Writes current statistics to TensorBoard."""
        if not self._writer: return

        step = self._num_samples_processed
        safe_best_score = self._cur_best_program_score if np.isfinite(self._cur_best_program_score) else 0

        avg_full_eval_time = np.divide(self._total_full_eval_time, self._num_full_evals_success, where=self._num_full_evals_success > 0, out=np.array(0.0)).item()
        avg_hit_processing_time = np.divide(self._total_hit_processing_time, self._num_cache_hits_processed, where=self._num_cache_hits_processed > 0, out=np.array(0.0)).item()
        total_processing_time = self._total_full_eval_time + self._total_failed_eval_time + self._total_hit_processing_time
        avg_overall_time = np.divide(total_processing_time, step, where=step > 0, out=np.array(0.0)).item()

        try:
            self._writer.add_scalar('Score/Best_Program_Score', safe_best_score, global_step=step)
            self._writer.add_scalars('Counts/Program_Outcomes', {
                'Successful_Full_Eval': self._num_full_evals_success,
                'Failed_Full_Eval': self._num_failed_evals,
                'Processed_Cache_Hits': self._num_cache_hits_processed,
            }, global_step=step)
            self._writer.add_scalars('Time/Average_Processing_Seconds', {
                'Avg_Successful_Full_Eval': avg_full_eval_time,
                'Avg_Cache_Hit_Processing': avg_hit_processing_time,
                'Avg_Overall_Eval_Check': avg_overall_time,
            }, global_step=step)
        except Exception as e:
             print(f"Warning: Error writing to TensorBoard: {e}")


    def _write_json(self, program: code_manipulation.Function, is_cached_duplicate: bool):
        """Writes program details to a JSON file."""
        if not self._json_dir: return

        sample_order = program.global_sample_nums if program.global_sample_nums is not None else 0 # Handle initial program
        filename = f'sample_{sample_order}.json'
        path = os.path.join(self._json_dir, filename)

        content = {
            'sample_order': sample_order,
            'score': program.score,
            'evaluate_time_sec': f"{program.evaluate_time:.4f}" if program.evaluate_time is not None else None,
            'sample_time_sec': f"{program.sample_time:.4f}" if program.sample_time is not None else None,
            'is_cached_duplicate': is_cached_duplicate,
            'function_signature': f"def {program.name}({program.args}){f' -> {program.return_type}' if program.return_type else ''}:",
            'function_docstring': program.docstring,
            'function_body': program.body,
        }

        try:
            with open(path, 'w') as json_file:
                json.dump(content, json_file, indent=2)
        except IOError as e:
            print(f"Warning: Could not write JSON log file '{path}'. Error: {e}")


    def register_function(self, program: code_manipulation.Function, is_cached_duplicate: bool = False, **kwargs):
        """Registers a program (function object) and updates statistics."""
        # Check logging limit
        if self._max_log_nums is not None and self._num_samples_processed >= self._max_log_nums:
            if self._num_samples_processed == self._max_log_nums:
                 print(f"Info: Reached maximum log number ({self._max_log_nums}). Further samples not logged.")
                 self._num_samples_processed += 1
            return

        sample_orders: int = program.global_sample_nums if program.global_sample_nums is not None else 0

        self._num_samples_processed += 1

        # Update detailed counters
        eval_time = program.evaluate_time if program.evaluate_time is not None else 0.0
        score = program.score

        if is_cached_duplicate:
            self._num_cache_hits_processed += 1
            self._total_hit_processing_time += eval_time
            if score is not None: self._num_cache_hits_valid_score += 1
        else:
            if score is not None:
                self._num_full_evals_success += 1
                self._total_full_eval_time += eval_time
            else:
                self._num_failed_evals += 1
                self._total_failed_eval_time += eval_time

        if program.sample_time: self._tot_sample_time += program.sample_time
        self._all_sampled_functions[sample_orders] = program # Store/overwrite

        is_new_best = False
        if score is not None and score > self._cur_best_program_score:
             if not np.isclose(score, self._cur_best_program_score):
                  is_new_best = True
                  self._cur_best_program_score = score
                  self._cur_best_program_sample_order = sample_orders

        # Logging and Output
        self._record_and_verbose(sample_orders, is_cached_duplicate, program, is_new_best) # Pass is_new_best
        self._write_tensorboard()
        self._write_json(program, is_cached_duplicate)


    # --- Modified _record_and_verbose ---
    def _record_and_verbose(self, sample_orders: int, is_cached_duplicate: bool, program: code_manipulation.Function, is_new_best: bool):
        """Prints a formatted log message for the processed program, matching the desired style."""

        header_status = "Cached Program" if is_cached_duplicate else "Evaluated Program"

        # Construct the header
        header = f"================= {header_status} #{sample_orders} ================="
        print(header)

        # Print the full function string
        print(str(program).strip())

        # Print the separator
        print("------------------------------------------------------")

        # Print the stats block
        score_str = f"{program.score:.1f}" if program.score is not None else "N/A"
        sample_time_str = f"{program.sample_time:.4f}s" if program.sample_time is not None else "N/A"
        eval_time_str = f"{program.evaluate_time:.4f}s" if program.evaluate_time is not None else "N/A"

        print(f"Score        : {score_str}")
        print(f"Sample time  : {sample_time_str}")
        print(f"Eval/Check T : {eval_time_str}")
        print(f"Sample order : {sample_orders}")

        # --- Add Status line specifically for cache hits ---
        if is_cached_duplicate:
            # Determine specific cache hit status
            if program.score is not None:
                cache_status_detail = "Cache Hit (Registered)" # FP hit or validated hit with score
            else:
                cache_status_detail = "Cache Hit (AST/Invalid)" # AST hit or failed validation
            print(f"Status       : {cache_status_detail}")
        # ----------------------------------------------------

        # Print the footer
        print("==================================================") # 50 '=' chars

        # Optionally print NEW BEST indicator after the block
        if is_new_best:
             print("*** NEW BEST SCORE ***")

        print() # Add a blank line for spacing


    def log_efficiency_report(self):
        """Calculates and prints the final efficiency report, matching the desired style."""
        header_len = 75

        # Header
        print("\n" + "=" * 29 + " Efficiency Report " + "=" * 29)

        # Timestamp and Mode
        print(f"Timestamp                 : {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}")
        print(f"Sample Efficiency Mode    : {'ENABLED (AST or FP)' if self._efficiency_enabled else 'DISABLED'}")
        print(f"Total Samples Processed   : {self._num_samples_processed}")

        # Separator
        print("-" * header_len)

        # Overall Performance
        total_processing_time = self._total_full_eval_time + self._total_failed_eval_time + self._total_hit_processing_time
        avg_overall_time = np.divide(total_processing_time, self._num_samples_processed, where=self._num_samples_processed > 0, out=np.array(0.0)).item()
        print(f"[Overall Performance]")
        print(f"  Avg. Eval/Check Time per Sample: {avg_overall_time:.4f} seconds")

        # Separator
        print("-" * header_len)

        # Full Evaluation Stats
        avg_full_eval_time = np.divide(self._total_full_eval_time, self._num_full_evals_success, where=self._num_full_evals_success > 0, out=np.array(0.0)).item()
        avg_failed_eval_time = np.divide(self._total_failed_eval_time, self._num_failed_evals, where=self._num_failed_evals > 0, out=np.array(0.0)).item()
        print(f"[Full Evaluations (Non-Cache Hits)]")
        print(f"  Successful              : {self._num_full_evals_success}")
        print(f"  Failed/Invalid          : {self._num_failed_evals}")
        print(f"  Avg. Time (Successful)  : {avg_full_eval_time:.4f} seconds")
        print(f"  Avg. Time (Failed)      : {avg_failed_eval_time:.4f} seconds")

        # Separator
        print("-" * header_len)

        # Cache Hit Stats & Gain (Conditional)
        if self._efficiency_enabled:
            avg_hit_processing_time = np.divide(self._total_hit_processing_time, self._num_cache_hits_processed, where=self._num_cache_hits_processed > 0, out=np.array(0.0)).item()
            print(f"[Cache Hits Processed]")
            print(f"  Total Hit Count         : {self._num_cache_hits_processed}")
            print(f"  Avg. Processing Time    : {avg_hit_processing_time:.4f} seconds")

            print("-" * header_len)

            print(f"[Estimated Efficiency Gain (vs No Cache)]")
            if self._num_cache_hits_processed > 0 and self._num_full_evals_success > 0:
                 baseline_cost = avg_full_eval_time
                 avg_time_saved_per_hit = max(0, baseline_cost - avg_hit_processing_time)
                 total_time_saved = avg_time_saved_per_hit * self._num_cache_hits_processed
                 estimated_total_eval_time_no_cache = self._total_full_eval_time + self._total_failed_eval_time + (baseline_cost * self._num_cache_hits_processed)

                 print(f"  Baseline Avg Full Eval  : {baseline_cost:.4f} seconds")
                 print(f"  Avg Time Saved per Hit  : {avg_time_saved_per_hit:.4f} seconds (Estimated)")
                 print(f"  Total Est. Time Saved   : {total_time_saved:.2f} seconds")

                 if estimated_total_eval_time_no_cache > 0:
                     percentage_saved = (total_time_saved / estimated_total_eval_time_no_cache) * 100
                     print(f"  Reduction in Eval Time  : {percentage_saved:.1f}% (Estimated)")
                 else:
                      print(f"  Reduction in Eval Time  : N/A (No baseline time)")

            elif self._num_cache_hits_processed == 0:
                 print("  No cache hits occurred. Efficiency gain is 0%.")
            else:
                 print("  Cannot estimate gain vs baseline: No successful full evaluations for comparison.")
        else:
            print("[Cache Hits Processed]")
            print("  N/A (Efficiency Disabled)")
            print("-" * header_len)
            print("[Estimated Efficiency Gain (vs No Cache)]")
            print("  N/A (Efficiency Disabled)")

        # Footer
        print("=" * header_len + "\n")