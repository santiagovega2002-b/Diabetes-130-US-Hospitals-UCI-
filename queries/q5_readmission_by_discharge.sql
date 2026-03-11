SELECT 
    discharge_disposition_id,
    COUNT(*) as n_encounters,
    ROUND(AVG(readmitted_binary) * 100, 1) as readmission_rate_pct,
    ROUND(AVG(time_in_hospital), 1) as avg_stay
FROM encounters
GROUP BY discharge_disposition_id
HAVING COUNT(*) > 300
ORDER BY readmission_rate_pct DESC