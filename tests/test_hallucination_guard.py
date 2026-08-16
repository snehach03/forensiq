"""
test_hallucination_guard.py

Phase 9: Unit tests for the LLM hallucination guard (Phase 7).

Only testing the pure functions (_extract_numbers, _allowed_base_values,
_is_close_to_any, _flag_unrecognized_numbers) - NOT generate_narrative()
itself, since that makes a real Groq API call. Unit-testing something
that hits a live external API is the wrong tool for the job (slow,
costs money, network-flaky); that function is better covered by manual/
integration checks instead.

Special focus: _is_close_to_any()'s percentage-conversion logic (x100 /
÷100) - this is the exact piece that was fragile before (per Phase 7
design notes: string-matching was replaced with numeric-closeness for
this reason), so it gets the most thorough coverage here.

TODO Sneha: fix this import path to match your actual file location.
"""

from forensiq.llm.generate_narrative import (  # TODO: fix path
    _allowed_base_values,
    _extract_numbers,
    _flag_unrecognized_numbers,
    _is_close_to_any,
)


class TestExtractNumbers:
    def test_extracts_decimal_and_integer(self):
        text = "Revenue grew by 12.5% to reach 45 million dollars."
        result = _extract_numbers(text)
        assert result == [12.5, 45.0]

    def test_extracts_negative_numbers(self):
        text = "SHAP contribution of -0.23 pushed the score down."
        result = _extract_numbers(text)
        assert -0.23 in result

    def test_no_numbers_returns_empty_list(self):
        text = "This company shows no significant risk indicators."
        assert _extract_numbers(text) == []

    def test_ignores_bare_dash_and_dot(self):
        # Regex edge case - a lone "-" or "." shouldn't become a phantom
        # number (the function explicitly filters these out).
        text = "Range: - to . (placeholder text, no real numbers)"
        assert _extract_numbers(text) == []


class TestAllowedBaseValues:
    def test_includes_fiscal_year(self, sample_llm_snapshot):
        allowed = _allowed_base_values(sample_llm_snapshot)
        assert 2021.0 in allowed

    def test_includes_risk_score_and_red_flag_count(self, sample_llm_snapshot):
        allowed = _allowed_base_values(sample_llm_snapshot)
        assert 0.65 in allowed
        assert 5.0 in allowed

    def test_includes_decision_threshold(self, sample_llm_snapshot):
        allowed = _allowed_base_values(sample_llm_snapshot)
        assert 0.40 in allowed

    def test_includes_shap_contributions_and_their_abs_values(self, sample_llm_snapshot):
        allowed = _allowed_base_values(sample_llm_snapshot)
        # -0.10 is a reducer's raw contribution; 0.10 is its abs() value -
        # both should be present since the LLM might describe either
        # "reduced risk by -0.10" or "reduced risk by 0.10".
        assert -0.10 in allowed
        assert 0.10 in allowed

    def test_extracts_numbers_embedded_in_rule_text(self, sample_llm_snapshot):
        # "Leverage spiked more than 25% YoY" -> 25.0 should be pulled
        # out and allowed, since the LLM is allowed to quote rule text.
        allowed = _allowed_base_values(sample_llm_snapshot)
        assert 25.0 in allowed
        assert 1.23 in allowed


class TestIsCloseToAny:
    def test_exact_match(self):
        assert _is_close_to_any(0.65, {0.65}) is True

    def test_within_tolerance(self):
        # Default tolerance is 0.05 - 0.66 vs 0.65 should still pass
        assert _is_close_to_any(0.66, {0.65}) is True

    def test_outside_tolerance(self):
        assert _is_close_to_any(0.80, {0.65}) is False

    def test_percentage_conversion_x100(self):
        # This is the core fragile case: LLM writes "65%" instead of
        # "0.65" - the guard should recognize 65 as legitimate because
        # 0.65 * 100 = 65.
        assert _is_close_to_any(65.0, {0.65}) is True

    def test_percentage_conversion_div100(self):
        # Reverse direction: allowed set has 25 (from "25% YoY" rule
        # text), LLM writes "0.25" - should also be recognized.
        assert _is_close_to_any(0.25, {25.0}) is True

    def test_zero_base_does_not_crash(self):
        # base=0 hits the "base / 100 if base else 0" branch - this
        # test guards against a possible ZeroDivisionError if that
        # fallback were ever removed.
        assert _is_close_to_any(0.0, {0.0}) is True

    def test_unrelated_value_is_false(self):
        assert _is_close_to_any(999.9, {0.65, 5.0, 25.0, 0.40}) is False


class TestFlagUnrecognizedNumbers:
    def test_small_counting_numbers_ignored(self, sample_llm_snapshot):
        # "top 3 features" - 3 is a counting number (<=9), should never
        # be flagged even though it's not in the snapshot's allowed set.
        text = "The model's top 3 risk drivers all point toward distress."
        suspicious = _flag_unrecognized_numbers(text, sample_llm_snapshot)
        assert 3.0 not in suspicious

    def test_legitimate_snapshot_number_not_flagged(self, sample_llm_snapshot):
        text = "The company's risk score stands at 0.65 for fiscal year 2021."
        suspicious = _flag_unrecognized_numbers(text, sample_llm_snapshot)
        assert suspicious == []

    def test_legitimate_number_in_percentage_form_not_flagged(self, sample_llm_snapshot):
        # 0.65 written as "65%" - should still pass, not be flagged.
        text = "The risk probability is approximately 65% for this year."
        suspicious = _flag_unrecognized_numbers(text, sample_llm_snapshot)
        assert suspicious == []

    def test_fabricated_number_is_flagged(self, sample_llm_snapshot):
        # 847.2 has no relationship to anything in the snapshot -
        # a classic hallucinated statistic the guard should catch.
        text = "Revenue increased by an unprecedented 847.2 percent."
        suspicious = _flag_unrecognized_numbers(text, sample_llm_snapshot)
        assert 847.2 in suspicious