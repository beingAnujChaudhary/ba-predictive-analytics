# Task 1 — Modeling Lounge Eligibility at Heathrow Terminal 3

## Business Problem

British Airways requires a scalable approach to estimate future lounge demand at Heathrow Terminal 3.

Since future flight schedules are uncertain, the Airport Planning team requires a reusable lookup model instead of aircraft-specific calculations.

---

# Objective

Develop a lookup table capable of estimating the percentage of passengers eligible for each lounge tier based on flight characteristics.

---

# Data Understanding

The provided schedule contains information including:

- Flight Number
- Flight Time
- Route
- Destination
- Aircraft Type
- Cabin Capacity
- Eligible Lounge Passengers

---

# Methodology

The following workflow was used:

1. Data Inspection
2. Data Cleaning
3. Exploratory Data Analysis
4. Passenger Capacity Analysis
5. Flight Categorization
6. Lounge Eligibility Percentage Calculation
7. Lookup Table Generation

---

# Flight Grouping Strategy

Flights were categorized using:

- Time of Day
- Route Type
- Haul Type
- Geographic Region

This approach creates a reusable lookup model that can be applied to future schedules.

---

# Outputs

The task generated:

- Lounge Eligibility Lookup Table
- Passenger Eligibility Estimates
- Supporting Business Justification

---

# Business Impact

The lookup model enables British Airways to:

- Estimate lounge demand
- Plan airport capacity
- Allocate operational resources
- Evaluate future lounge investments