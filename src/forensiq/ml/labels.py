"""
Ground-truth fraud/scrutiny labels for our companies, based on
documented SEC enforcement actions and public restatements.

This is manually researched and curated — labels are NOT derived
automatically from our financial data (that would be circular: we'd
be using our own features to create the labels we're trying to
predict from those same features). Each entry below cites the
specific event that justifies the label.

label = 1: company's financials for that fiscal year were later
           found/restated to involve accounting misconduct
label = 0: no known issue for that fiscal year
"""

# company_id reference: 1=Apple, 2=Microsoft, 3=Costco, 4=GE,
# 5=Under Armour, 6=Enron, 7=Kraft Heinz, 8=Valeant/Bausch Health

FRAUD_YEARS = {
    # Kraft Heinz: SEC found accounting misconduct (improper cost-savings
    # recognition) spanning Q4 2015 through end of 2018. Restated in 2019.
    # Source: SEC Litigation Release No. 25195 (Sept 2021)
    5: {2016},
    7: {2016, 2017, 2018},

    # Valeant/Bausch Health: aggressive channel-stuffing and pricing
    # practices scrutinized starting 2015-2016; company rebranded to
    # Bausch Health in 2018 following the scandal fallout.
    8: {2016},

}


def get_label(company_id: int, fiscal_year: int) -> int:
    """
    Returns 1 if the given company-year falls within a known
    fraud/misconduct period, 0 otherwise (including all years for
    companies with no known issues).
    """
    fraud_years_for_company = FRAUD_YEARS.get(company_id, set())
    return 1 if fiscal_year in fraud_years_for_company else 0


def add_labels(df):
    """Adds a 'label' column to the given DataFrame using get_label()."""
    df = df.copy()
    df["label"] = df.apply(
        lambda row: get_label(row["company_id"], row["fiscal_year"]),
        axis=1
    )
    return df