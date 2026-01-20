"""
Generic Validator Generator

Generates validation functions from structured constraints extracted by the formalizer agent.
This is fully algorithm-agnostic - validators are built dynamically from constraint specifications.
"""

from typing import Any, Callable
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of a single constraint validation."""
    passed: bool
    constraint_id: str
    constraint_type: str
    message: str
    details: dict | None = None


@dataclass
class StageValidationResult:
    """Result of validating all constraints for a stage."""
    stage_id: str
    all_passed: bool
    results: list[ValidationResult]

    @property
    def failed_constraints(self) -> list[ValidationResult]:
        return [r for r in self.results if not r.passed]


class GenericValidatorGenerator:
    """
    Generates validation functions from structured constraint specifications.

    Supports 7 generic constraint types:
    - COUNT: Check collection sizes
    - COMPLETENESS: Verify all items processed
    - MAPPING: Verify relationships between collections
    - PROPERTY: Check specific properties
    - RANGE: Check value bounds
    - ORDER: Check processing sequence
    - PROHIBITION: Verify forbidden actions didn't occur
    """

    def __init__(self, structured_constraints: dict):
        """
        Initialize with structured constraints from formalizer agent.

        Args:
            structured_constraints: Dict with 'stage_constraints' key containing
                                   per-stage constraint lists
        """
        self.constraints = structured_constraints
        self.stage_constraints = structured_constraints.get("stage_constraints", {})
        self._validators: dict[str, list[Callable]] = {}
        self._build_validators()

    def _build_validators(self):
        """Build validator functions for each stage from constraints."""
        for stage_id, constraints in self.stage_constraints.items():
            self._validators[stage_id] = []

            # Handle case where constraints might be a list of dicts or a single dict
            if isinstance(constraints, dict):
                constraints = [constraints]
            elif not isinstance(constraints, list):
                # Skip invalid constraint formats
                continue

            for constraint in constraints:
                # Skip non-dict constraints (strings, etc.)
                if not isinstance(constraint, dict):
                    continue

                validator = self._create_validator(constraint)
                if validator:
                    self._validators[stage_id].append((constraint, validator))

    def _create_validator(self, constraint: dict) -> Callable | None:
        """Create a validator function from a constraint specification."""
        constraint_type = constraint.get("type", "").lower()

        validators_map = {
            "count": self._create_count_validator,
            "completeness": self._create_completeness_validator,
            "mapping": self._create_mapping_validator,
            "property": self._create_property_validator,
            "range": self._create_range_validator,
            "order": self._create_order_validator,
            "prohibition": self._create_prohibition_validator,
        }

        creator = validators_map.get(constraint_type)
        if creator:
            return creator(constraint)
        return None

    def _resolve_path(self, obj: dict, path: str) -> Any:
        """
        Resolve a dot-notation path to a value in a nested object.

        Examples:
            _resolve_path({"output": {"items": [1,2,3]}}, "output.items") -> [1,2,3]
            _resolve_path({"graph": {"edges": 5}}, "graph.edges") -> 5
        """
        parts = path.split(".")
        current = obj
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list) and part.isdigit():
                current = current[int(part)]
            else:
                return None
            if current is None:
                return None
        return current

    def _evaluate_expression(self, expr: str, variables: dict, context: dict) -> Any:
        """
        Evaluate a simple expression with variable substitution.

        Supports:
        - len(path): Length of collection at path
        - n * factor: Basic multiplication
        - n + offset: Basic addition
        - Numeric literals

        Args:
            expr: Expression string (e.g., "n * 2", "len(input.items)")
            variables: Variable definitions (e.g., {"n": "len(input.data)"})
            context: Full context dict for path resolution
        """
        # First, resolve all variables
        resolved_vars = {}
        for var_name, var_expr in variables.items():
            resolved_vars[var_name] = self._evaluate_simple_expr(var_expr, context)

        # Then evaluate the main expression with resolved variables
        return self._evaluate_simple_expr(expr, context, resolved_vars)

    def _evaluate_simple_expr(self, expr: str, context: dict, vars_dict: dict | None = None) -> Any:
        """Evaluate a simple expression."""
        vars_dict = vars_dict or {}
        expr = expr.strip()

        # Handle len() function
        if expr.startswith("len(") and expr.endswith(")"):
            path = expr[4:-1]
            value = self._resolve_path(context, path)
            return len(value) if value is not None else 0

        # Handle numeric literal
        try:
            return float(expr) if "." in expr else int(expr)
        except ValueError:
            pass

        # Handle variable reference
        if expr in vars_dict:
            return vars_dict[expr]

        # Handle basic arithmetic (n * factor, n + offset)
        for op in ["*", "+", "-", "/"]:
            if op in expr:
                left, right = expr.split(op, 1)
                left_val = self._evaluate_simple_expr(left.strip(), context, vars_dict)
                right_val = self._evaluate_simple_expr(right.strip(), context, vars_dict)
                if left_val is not None and right_val is not None:
                    if op == "*":
                        return left_val * right_val
                    elif op == "+":
                        return left_val + right_val
                    elif op == "-":
                        return left_val - right_val
                    elif op == "/":
                        return left_val / right_val if right_val != 0 else None

        return None

    def _compare(self, actual: Any, operator: str, expected: Any) -> bool:
        """Compare actual value against expected using operator."""
        if actual is None or expected is None:
            return False

        operators = {
            "==": lambda a, e: a == e,
            "!=": lambda a, e: a != e,
            ">": lambda a, e: a > e,
            ">=": lambda a, e: a >= e,
            "<": lambda a, e: a < e,
            "<=": lambda a, e: a <= e,
        }

        compare_fn = operators.get(operator)
        if compare_fn:
            try:
                return compare_fn(actual, expected)
            except (TypeError, ValueError):
                return False
        return False

    # === Validator Creators ===

    def _create_count_validator(self, constraint: dict) -> Callable:
        """Create a COUNT constraint validator."""
        check = constraint.get("check", {})
        object_path = check.get("object_path", "")
        operator = check.get("operator", "==")
        value_expr = check.get("value_expression", "0")
        variables = check.get("variables", {})

        def validator(context: dict) -> ValidationResult:
            actual_value = self._resolve_path(context, object_path)
            actual_count = len(actual_value) if hasattr(actual_value, "__len__") else actual_value
            expected = self._evaluate_expression(value_expr, variables, context)

            passed = self._compare(actual_count, operator, expected)

            return ValidationResult(
                passed=passed,
                constraint_id=constraint.get("id", "unknown"),
                constraint_type="count",
                message=f"COUNT check: {object_path} {operator} {expected}",
                details={
                    "actual": actual_count,
                    "expected": expected,
                    "operator": operator,
                    "path": object_path,
                }
            )

        return validator

    def _create_completeness_validator(self, constraint: dict) -> Callable:
        """Create a COMPLETENESS constraint validator."""
        check = constraint.get("check", {})
        action = check.get("action", "process")
        collection_path = check.get("collection_path", "")
        result_path = check.get("result_path", "")
        requirement = check.get("requirement", "all")

        def validator(context: dict) -> ValidationResult:
            source = self._resolve_path(context, collection_path)
            result = self._resolve_path(context, result_path) if result_path else None

            if source is None:
                return ValidationResult(
                    passed=False,
                    constraint_id=constraint.get("id", "unknown"),
                    constraint_type="completeness",
                    message=f"COMPLETENESS: Source collection '{collection_path}' not found",
                    details={"collection_path": collection_path}
                )

            source_count = len(source) if hasattr(source, "__len__") else 0
            result_count = len(result) if result and hasattr(result, "__len__") else 0

            if requirement == "all":
                passed = result_count >= source_count if result else source_count > 0
            else:
                passed = result_count > 0

            return ValidationResult(
                passed=passed,
                constraint_id=constraint.get("id", "unknown"),
                constraint_type="completeness",
                message=f"COMPLETENESS: {action} {requirement} items from {collection_path}",
                details={
                    "source_count": source_count,
                    "result_count": result_count,
                    "requirement": requirement,
                }
            )

        return validator

    def _create_mapping_validator(self, constraint: dict) -> Callable:
        """Create a MAPPING constraint validator."""
        check = constraint.get("check", {})
        source_path = check.get("source_path", "")
        target_path = check.get("target_path", "")
        relation = check.get("relation", "one_to_one")

        def validator(context: dict) -> ValidationResult:
            source = self._resolve_path(context, source_path)
            target = self._resolve_path(context, target_path)

            if source is None or target is None:
                return ValidationResult(
                    passed=False,
                    constraint_id=constraint.get("id", "unknown"),
                    constraint_type="mapping",
                    message=f"MAPPING: Could not resolve paths",
                    details={"source_path": source_path, "target_path": target_path}
                )

            source_count = len(source) if hasattr(source, "__len__") else 0
            target_count = len(target) if hasattr(target, "__len__") else 0

            if relation == "one_to_one":
                passed = source_count == target_count
            elif relation == "one_to_many":
                passed = target_count >= source_count
            elif relation == "many_to_one":
                passed = source_count >= target_count
            else:
                passed = True

            return ValidationResult(
                passed=passed,
                constraint_id=constraint.get("id", "unknown"),
                constraint_type="mapping",
                message=f"MAPPING: {source_path} -> {target_path} ({relation})",
                details={
                    "source_count": source_count,
                    "target_count": target_count,
                    "relation": relation,
                }
            )

        return validator

    def _create_property_validator(self, constraint: dict) -> Callable:
        """Create a PROPERTY constraint validator."""
        check = constraint.get("check", {})
        object_path = check.get("object_path", "")
        property_name = check.get("property", "")
        expected_value = check.get("expected_value")

        def validator(context: dict) -> ValidationResult:
            obj = self._resolve_path(context, object_path)

            if obj is None:
                return ValidationResult(
                    passed=False,
                    constraint_id=constraint.get("id", "unknown"),
                    constraint_type="property",
                    message=f"PROPERTY: Object at '{object_path}' not found",
                    details={"object_path": object_path}
                )

            # Check if property exists
            if isinstance(obj, dict):
                actual = obj.get(property_name)
                exists = property_name in obj
            else:
                actual = getattr(obj, property_name, None)
                exists = hasattr(obj, property_name)

            if expected_value is not None:
                passed = actual == expected_value
            else:
                passed = exists and actual is not None

            return ValidationResult(
                passed=passed,
                constraint_id=constraint.get("id", "unknown"),
                constraint_type="property",
                message=f"PROPERTY: {object_path}.{property_name}",
                details={
                    "exists": exists,
                    "actual": actual,
                    "expected": expected_value,
                }
            )

        return validator

    def _create_range_validator(self, constraint: dict) -> Callable:
        """Create a RANGE constraint validator."""
        check = constraint.get("check", {})
        value_path = check.get("value_path", "")
        min_val = check.get("min")
        max_val = check.get("max")

        def validator(context: dict) -> ValidationResult:
            actual = self._resolve_path(context, value_path)

            if actual is None:
                return ValidationResult(
                    passed=False,
                    constraint_id=constraint.get("id", "unknown"),
                    constraint_type="range",
                    message=f"RANGE: Value at '{value_path}' not found",
                    details={"value_path": value_path}
                )

            passed = True
            if min_val is not None:
                passed = passed and actual >= min_val
            if max_val is not None:
                passed = passed and actual <= max_val

            return ValidationResult(
                passed=passed,
                constraint_id=constraint.get("id", "unknown"),
                constraint_type="range",
                message=f"RANGE: {min_val} <= {value_path} <= {max_val}",
                details={
                    "actual": actual,
                    "min": min_val,
                    "max": max_val,
                }
            )

        return validator

    def _create_order_validator(self, constraint: dict) -> Callable:
        """Create an ORDER constraint validator."""
        check = constraint.get("check", {})
        sequence_path = check.get("sequence_path", "")
        expected_order = check.get("expected_order", [])

        def validator(context: dict) -> ValidationResult:
            sequence = self._resolve_path(context, sequence_path)

            if sequence is None:
                return ValidationResult(
                    passed=False,
                    constraint_id=constraint.get("id", "unknown"),
                    constraint_type="order",
                    message=f"ORDER: Sequence at '{sequence_path}' not found",
                    details={"sequence_path": sequence_path}
                )

            # Check if sequence follows expected order
            passed = True
            if expected_order:
                # Check that items appear in correct relative order
                last_idx = -1
                for item in expected_order:
                    if item in sequence:
                        idx = list(sequence).index(item) if not isinstance(sequence, list) else sequence.index(item)
                        if idx < last_idx:
                            passed = False
                            break
                        last_idx = idx

            return ValidationResult(
                passed=passed,
                constraint_id=constraint.get("id", "unknown"),
                constraint_type="order",
                message=f"ORDER: Sequence follows expected order",
                details={
                    "sequence": list(sequence) if hasattr(sequence, "__iter__") else sequence,
                    "expected_order": expected_order,
                }
            )

        return validator

    def _create_prohibition_validator(self, constraint: dict) -> Callable:
        """Create a PROHIBITION constraint validator."""
        check = constraint.get("check", {})
        forbidden_path = check.get("forbidden_path", "")
        forbidden_value = check.get("forbidden_value")
        forbidden_condition = check.get("forbidden_condition", "exists")

        def validator(context: dict) -> ValidationResult:
            actual = self._resolve_path(context, forbidden_path)

            if forbidden_condition == "exists":
                # Fail if path exists and has value
                passed = actual is None
            elif forbidden_condition == "equals":
                # Fail if value equals forbidden value
                passed = actual != forbidden_value
            elif forbidden_condition == "contains":
                # Fail if collection contains forbidden value
                passed = forbidden_value not in actual if actual else True
            else:
                passed = True

            return ValidationResult(
                passed=passed,
                constraint_id=constraint.get("id", "unknown"),
                constraint_type="prohibition",
                message=f"PROHIBITION: {forbidden_path} must not {forbidden_condition}",
                details={
                    "actual": actual,
                    "forbidden_value": forbidden_value,
                    "forbidden_condition": forbidden_condition,
                }
            )

        return validator

    # === Public API ===

    def validate_stage(self, stage_id: str, context: dict) -> StageValidationResult:
        """
        Validate a stage's output against its constraints.

        Args:
            stage_id: ID of the stage to validate
            context: Full execution context including stage outputs

        Returns:
            StageValidationResult with all constraint check results
        """
        validators = self._validators.get(stage_id, [])
        results = []

        for constraint, validator in validators:
            try:
                result = validator(context)
                results.append(result)
            except Exception as e:
                results.append(ValidationResult(
                    passed=False,
                    constraint_id=constraint.get("id", "unknown"),
                    constraint_type=constraint.get("type", "unknown"),
                    message=f"Validation error: {str(e)}",
                    details={"error": str(e)}
                ))

        all_passed = all(r.passed for r in results) if results else True

        return StageValidationResult(
            stage_id=stage_id,
            all_passed=all_passed,
            results=results
        )

    def get_stage_ids(self) -> list[str]:
        """Get list of stage IDs that have constraints."""
        return list(self._validators.keys())

    def has_constraints_for_stage(self, stage_id: str) -> bool:
        """Check if a stage has any constraints defined."""
        return stage_id in self._validators and len(self._validators[stage_id]) > 0
