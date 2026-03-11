SELECT 
    number_inpatient,
    COUNT(*) as n_encounters,
    SUM(readmitted_binary) as n_readmitted,
    ROUND(AVG(readmitted_binary) * 100, 1) as readmission_rate_pct,
    ROUND(AVG(time_in_hospital), 1) as avg_stay_days
FROM encounters
WHERE number_inpatient <= 10
GROUP BY number_inpatient
ORDER BY number_inpatient