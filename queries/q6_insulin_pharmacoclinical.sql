SELECT 
    insulin,
    COUNT(*) as n_encounters,
    ROUND(AVG(readmitted_binary) * 100, 1) as readmission_rate_pct,
    ROUND(SUM(CASE WHEN A1Cresult = '>8' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as pct_A1C_gt8,
    ROUND(SUM(CASE WHEN was_A1C_measured = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as pct_A1C_measured,
    ROUND(AVG(num_medications), 1) as avg_meds,
    ROUND(AVG(time_in_hospital), 1) as avg_stay,
    ROUND(AVG(number_emergency), 2) as avg_prior_emergency
FROM encounters
GROUP BY insulin
ORDER BY readmission_rate_pct DESC