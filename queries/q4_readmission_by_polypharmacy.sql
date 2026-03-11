SELECT 
    CASE 
        WHEN num_medications <= 5  THEN '1-5 meds'
        WHEN num_medications <= 10 THEN '6-10 meds'
        WHEN num_medications <= 15 THEN '11-15 meds'
        WHEN num_medications <= 20 THEN '16-20 meds'
        WHEN num_medications <= 25 THEN '21-25 meds'
        ELSE '26+ meds'
    END as med_group,
    COUNT(*) as n_encounters,
    ROUND(AVG(readmitted_binary) * 100, 1) as readmission_rate_pct,
    ROUND(AVG(num_medications), 1) as avg_meds,
    ROUND(AVG(time_in_hospital), 1) as avg_stay
FROM encounters
GROUP BY med_group
ORDER BY avg_meds