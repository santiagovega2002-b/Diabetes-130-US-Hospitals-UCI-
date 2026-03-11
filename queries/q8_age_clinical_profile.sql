SELECT 
    age,
    COUNT(*) as n_encounters,
    ROUND(AVG(readmitted_binary) * 100, 1) as readmission_rate_pct,
    ROUND(AVG(num_medications), 1) as avg_meds,
    ROUND(AVG(number_diagnoses), 1) as avg_diagnoses,
    ROUND(AVG(total_prior_visits), 2) as avg_prior_visits,
    ROUND(AVG(time_in_hospital), 1) as avg_stay,
    ROUND(SUM(CASE WHEN diabetes_as_primary = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as pct_diabetes_primary,
    ROUND(SUM(CASE WHEN insulin = 'No' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as pct_no_insulin
FROM encounters
GROUP BY age
ORDER BY age