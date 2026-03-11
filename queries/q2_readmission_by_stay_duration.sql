SELECT 
    time_in_hospital,
    COUNT(*) as n_encounters,
    ROUND(AVG(readmitted_binary) * 100, 1) as readmission_rate_pct,
    ROUND(AVG(num_medications), 1) as avg_medications,
    ROUND(AVG(number_diagnoses), 1) as avg_diagnoses
FROM encounters
GROUP BY time_in_hospital
ORDER BY time_in_hospital