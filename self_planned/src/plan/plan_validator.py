"""
Plan Validator - Validates and auto-corrects plan reads against available context.

This module implements Layer 2 of the multi-layer defense system:
- Validates that all stage reads reference keys that will exist at runtime
- Auto-corrects invalid dot-paths (e.g., 'input.CI_information' → 'input')
- Prevents placeholder resolution failures during execution
"""

from dataclasses import dataclass
from plan.models import Plan, Stage


@dataclass
class ValidationResult:
    """Result of plan validation."""
    valid: bool
    warnings: list[str]
    errors: list[str]
    corrections_made: int


class PlanValidator:
    """
    Validates and auto-corrects plan reads against available context.

    This prevents errors where the planner generates reads like 'input.CI_information'
    but the actual context only has 'input' as a flat string.
    """

    def validate_and_fix_reads(
        self,
        plan: Plan,
        initial_context_keys: set[str]
    ) -> tuple[Plan, ValidationResult]:
        """
        Validate all stage reads and auto-correct invalid dot-paths.

        The validation process:
        1. Start with initial context keys (e.g., {'input'})
        2. For each stage in sequence:
           - Check that all reads exist in available_keys
           - Auto-correct dot-paths if parent key exists
           - Add stage's writes to available_keys for subsequent stages

        Args:
            plan: The generated plan to validate
            initial_context_keys: Keys available at start (e.g., {'input'})

        Returns:
            Tuple of (corrected_plan, validation_result)
        """
        available_keys = set(initial_context_keys)
        warnings = []
        errors = []
        corrections_made = 0

        for stage in plan.stages:
            fixed_reads, stage_warnings, stage_errors, stage_corrections = \
                self._validate_stage_reads(stage, available_keys)

            warnings.extend(stage_warnings)
            errors.extend(stage_errors)
            corrections_made += stage_corrections

            # Update stage reads with fixed version
            stage.reads = fixed_reads

            # Add this stage's writes to available keys for subsequent stages
            available_keys.update(stage.writes)

        result = ValidationResult(
            valid=len(errors) == 0,
            warnings=warnings,
            errors=errors,
            corrections_made=corrections_made
        )

        return plan, result

    def _validate_stage_reads(
        self,
        stage: Stage,
        available_keys: set[str]
    ) -> tuple[list[str], list[str], list[str], int]:
        """
        Validate and fix reads for a single stage.

        Returns:
            Tuple of (fixed_reads, warnings, errors, corrections_count)
        """
        fixed_reads = []
        warnings = []
        errors = []
        corrections = 0

        for read_key in stage.reads:
            # Case 1: Exact match - valid, keep as-is
            if read_key in available_keys:
                fixed_reads.append(read_key)
                continue

            # Case 2: Dot-path (e.g., 'input.CI_information', 'input.variables')
            if '.' in read_key:
                parent = read_key.split('.')[0]

                if parent in available_keys:
                    # Auto-correct to parent key
                    if parent not in fixed_reads:
                        fixed_reads.append(parent)

                    warnings.append(
                        f"Stage '{stage.id}': Auto-corrected '{read_key}' → '{parent}' "
                        f"('{read_key}' doesn't exist, using parent key)"
                    )
                    corrections += 1
                    continue
                else:
                    # Parent doesn't exist either
                    errors.append(
                        f"Stage '{stage.id}': Invalid read '{read_key}' - "
                        f"neither '{read_key}' nor '{parent}' exist. "
                        f"Available: {sorted(available_keys)}"
                    )
                    continue

            # Case 3: Unknown simple key - error
            errors.append(
                f"Stage '{stage.id}': Unknown read key '{read_key}'. "
                f"Available: {sorted(available_keys)}"
            )

        return fixed_reads, warnings, errors, corrections

    def validate_writes(self, plan: Plan) -> list[str]:
        """
        Validate that stage writes don't conflict.

        Returns list of warnings for any issues found.
        """
        warnings = []
        all_writes = set()

        for stage in plan.stages:
            for write_key in stage.writes:
                if write_key in all_writes:
                    warnings.append(
                        f"Stage '{stage.id}': Write key '{write_key}' "
                        f"was already written by a previous stage"
                    )
                all_writes.add(write_key)

        return warnings

    def get_context_flow(self, plan: Plan, initial_keys: set[str]) -> dict[str, set[str]]:
        """
        Compute the available context keys at each stage.

        Useful for debugging and understanding data flow.

        Returns:
            Dict mapping stage_id to set of available keys at that stage
        """
        flow = {}
        available = set(initial_keys)

        for stage in plan.stages:
            flow[stage.id] = set(available)  # Copy current state
            available.update(stage.writes)

        return flow
