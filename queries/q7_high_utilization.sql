SELECT 
    total_prior_visits,
    COUNT(*) as n_encounters,
    ROUND(AVG(readmitted_binary) * 100, 1) as readmission_rate_pct,
    ROUND(AVG(num_medications), 1) as avg_meds,
    ROUND(AVG(time_in_hospital), 1) as avg_stay
FROM encounters
WHERE total_prior_visits <= 15
GROUP BY total_prior_visits
ORDER BY total_prior_visits