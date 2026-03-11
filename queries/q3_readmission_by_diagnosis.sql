SELECT 
    diag_1_cat,
    COUNT(*) as n_encounters,
    ROUND(AVG(readmitted_binary) * 100, 1) as readmission_rate_pct,
    ROUND(AVG(time_in_hospital), 1) as avg_stay,
    ROUND(AVG(num_medications), 1) as avg_meds
FROM encounters
WHERE diag_1_cat NOT IN ('Unknown', 'External', 'Supplementary')
GROUP BY diag_1_cat
HAVING COUNT(*) > 200
ORDER BY readmission_rate_pct DESC