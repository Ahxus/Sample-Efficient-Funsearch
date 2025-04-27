# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A single-threaded implementation of the FunSearch pipeline."""
from __future__ import annotations

from typing import Any, Tuple, Sequence, Optional # Ensure Optional is imported

# Standard Library Imports
import random
import json
import time # Although not directly used here, often useful

# Local Implementation Imports
from . import code_manipulation
from . import config as config_lib
from . import evaluator
from . import programs_database
from . import sampler
from . import profile

from . import efficiency_utils


def _extract_function_names(specification: str) -> Tuple[str, str]:
    """Returns the name of the function to evolve and of the function to run.

    Ensures that the specification contains exactly one function decorated with
    `@funsearch.run` and exactly one function decorated with `@funsearch.evolve`.
    """
    run_functions = list(code_manipulation.yield_decorated(specification, 'funsearch', 'run'))
    if len(run_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@funsearch.run`.')
    evolve_functions = list(code_manipulation.yield_decorated(specification, 'funsearch', 'evolve'))
    if len(evolve_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@funsearch.evolve`.')
    return evolve_functions[0], run_functions[0]


def main(
        specification: str,
        inputs: Sequence[Any], # e.g., dict for bin packing, list for others
        config: config_lib.Config,
        max_sample_nums: int | None,
        class_config: config_lib.ClassConfig,
        # --- Sample Efficiency Flags ---
        use_ast_hash_check: bool = True,    # Control AST check
        use_fingerprint_check: bool = True, # Control Fingerprint check
        # --- Fingerprint Configuration (used only if use_fingerprint_check is True) ---
        num_fingerprint_inputs: int = 10,
        fingerprint_timeout_seconds: int = 5,
        # --- Validation Flag ---
        validate_cache_hits: bool = False,  # Control cache hit validation
        # --- Other Kwargs ---
        **kwargs
):
    """
    Launches a FunSearch experiment with optional sample efficiency mechanisms.

    Args:
        specification: The string containing the initial code template.
        inputs: A sequence or mapping of inputs for the problem domain.
        config: Experiment configuration object for database, sampler counts etc.
        max_sample_nums: Maximum number of LLM samples to process. None for unlimited.
        class_config: Configuration specifying the LLM and Sandbox classes.
        use_ast_hash_check: Whether to enable the AST-based duplicate check.
        use_fingerprint_check: Whether to enable the fingerprint-based duplicate check.
        num_fingerprint_inputs: How many inputs to use for fingerprinting (if enabled).
        fingerprint_timeout_seconds: Timeout for each sandbox run during fingerprinting.
        validate_cache_hits: If True, fully evaluate cache hits for verification.
        **kwargs: Additional arguments, typically includes 'log_dir'.
    """
    # 1. Initial Setup
    function_to_evolve, function_to_run = _extract_function_names(specification)
    template = code_manipulation.text_to_program(specification)
    database = programs_database.ProgramsDatabase(config.programs_database, template, function_to_evolve)

    # 2. Determine Overall Efficiency Status and Setup Profiler
    efficiency_enabled = use_ast_hash_check or use_fingerprint_check
    log_dir = kwargs.get('log_dir', None)
    profiler_instance = None
    if log_dir:
        profiler_instance = profile.Profiler(
            log_dir=log_dir,
            efficiency_enabled=efficiency_enabled # Pass overall status
        )

    # 3. Setup Duplicate Checker and Fingerprint Inputs (Conditional)
    duplicate_checker_instance: Optional[efficiency_utils.DuplicateChecker] = None
    selected_fingerprint_inputs: Optional[Sequence[Any]] = None
    active_fingerprint_check = use_fingerprint_check # Start assuming it's active if requested

    if efficiency_enabled:
        # Create checker only if at least one check method is requested
        duplicate_checker_instance = efficiency_utils.DuplicateChecker(log_dir=log_dir)

    if use_fingerprint_check:
        print("=> Fingerprint Check Requested.")
        # Attempt to select fingerprint inputs
        if isinstance(inputs, dict):
            all_input_keys = list(inputs.keys())
            if len(all_input_keys) == 0:
                print("=> Warning: Input dictionary is empty. Disabling Fingerprint Check.")
                selected_fingerprint_inputs = None
                active_fingerprint_check = False
            elif len(all_input_keys) <= num_fingerprint_inputs:
                selected_fingerprint_inputs = all_input_keys
                print(f"=> Using all {len(selected_fingerprint_inputs)} inputs for fingerprinting.")
            else:
                selected_fingerprint_inputs = random.sample(all_input_keys, num_fingerprint_inputs)
                print(f"=> Sampled {num_fingerprint_inputs} inputs for fingerprinting: {selected_fingerprint_inputs}")
        elif isinstance(inputs, (list, tuple)):
            num_total_inputs = len(inputs)
            if num_total_inputs == 0:
                print("=> Warning: Input sequence is empty. Disabling Fingerprint Check.")
                selected_fingerprint_inputs = None
                active_fingerprint_check = False
            elif num_total_inputs <= num_fingerprint_inputs:
                selected_fingerprint_inputs = list(range(num_total_inputs)) # Use indices
                print(f"=> Using all {len(selected_fingerprint_inputs)} input indices for fingerprinting.")
            else:
                selected_fingerprint_inputs = random.sample(range(num_total_inputs), num_fingerprint_inputs) # Sample indices
                print(f"=> Sampled {num_fingerprint_inputs} input indices for fingerprinting: {selected_fingerprint_inputs}")
        else:
            print("=> Warning: Unknown input type for fingerprint selection. Disabling Fingerprint Check.")
            selected_fingerprint_inputs = None
            active_fingerprint_check = False # Turn off if cannot select

        # Final check if selection failed
        if not selected_fingerprint_inputs and use_fingerprint_check:
            print("=> Warning: Could not select fingerprint inputs. Disabling Fingerprint Check.")
            active_fingerprint_check = False
    else:
         print("=> Fingerprint Check Disabled.")

    # Print final status of checks
    print(f"=> Effective AST Check Status: {'ENABLED' if use_ast_hash_check else 'DISABLED'}")
    print(f"=> Effective Fingerprint Check Status: {'ENABLED' if active_fingerprint_check else 'DISABLED'}")
    if efficiency_enabled:
        print(f"=> Cache Hit Validation: {'ENABLED' if validate_cache_hits else 'DISABLED'}")

    # 4. Create Evaluators
    evaluators_list = []
    for i in range(config.num_evaluators):
        evaluators_list.append(evaluator.Evaluator(
            database=database,
            template=template,
            function_to_evolve=function_to_evolve,
            function_to_run=function_to_run,
            inputs=inputs, # Full input set
            timeout_seconds=config.evaluate_timeout_seconds,
            sandbox_class=class_config.sandbox_class,
            # Pass the checker instance (might be None)
            duplicate_checker=duplicate_checker_instance,
            # Pass the actual status of checks decided above
            use_ast_hash_check=use_ast_hash_check,
            use_fingerprint_check=active_fingerprint_check, # Use the effective status
            # Pass fingerprint details (might be None)
            fingerprint_inputs=selected_fingerprint_inputs,
            fingerprint_timeout_seconds=fingerprint_timeout_seconds,
            # Pass validation flag
            validate_cache_hits=validate_cache_hits
        ))

    # 5. Analyse Initial Program
    # We send the initial implementation to be analysed by one of the evaluators.
    initial_function_body = template.get_function(function_to_evolve).body
    if evaluators_list:
        print("=> Analyzing initial program...")
        # Initial analysis typically doesn't involve LLM sampling time or island ID.
        evaluators_list[0].analyse(
            sample=initial_function_body,
            island_id=None,
            version_generated=None,
            profiler=profiler_instance,
            global_sample_nums=0 # Assign sample number 0 to initial program
            )
    else:
        print("=> Warning: No evaluators configured.")


    # 6. Create Samplers
    # Set global max sample nums.
    # Sampler needs the list of evaluators to distribute work.
    samplers_list = [sampler.Sampler(database, evaluators_list, config.samples_per_prompt, max_sample_nums=max_sample_nums, llm_class=class_config.llm_class)
                for _ in range(config.num_samplers)]

    # 7. Start Sampling Loop
    # This loop can be executed in parallel on remote sampler machines. As each
    # sampler enters an infinite loop, without parallelization only the first
    # sampler will do any work.
    print(f"\n=> Starting sampling process (Max Samples: {max_sample_nums or 'Unlimited'})...")
    try:
        if samplers_list:
            # For simplicity in single-threaded, just run the first sampler
            # In a distributed setup, you'd launch these in parallel.
            samplers_list[0].sample(profiler=profiler_instance) # Pass profiler
        else:
            print("=> Warning: No samplers configured.")

    except KeyboardInterrupt:
        print("\n=> KeyboardInterrupt received. Shutting down.")
    finally:
        # 8. Final Reporting
        print("\n=> FunSearch run finished or interrupted.")
        # Log final cache statistics (only if checker was created)
        if duplicate_checker_instance:
             final_stats = duplicate_checker_instance.get_stats()
             print("\n" + "="*30 + " Cache Statistics " + "="*30)
             print(json.dumps(final_stats, indent=2))
             print("="*80 + "\n")
             # Log stats to checker's log file too
             if hasattr(duplicate_checker_instance, '_log_stat'):
                 try:
                    # Attempt to close the file handle cleanly first
                    if duplicate_checker_instance._log_file and not duplicate_checker_instance._log_file.closed:
                         duplicate_checker_instance._log_stat(f"FINAL_STATS,{json.dumps(final_stats)}")
                         duplicate_checker_instance._log_file.close()
                 except Exception as e:
                     print(f"Warning: Could not write final cache stats to log file: {e}")

        # Call the efficiency report from the profiler
        if profiler_instance:
            profiler_instance.log_efficiency_report()

        print("=> Exiting.")


