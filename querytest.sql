SELECT category, AVG(ABS(sales - sales_prediction)) AS avg_error
FROM sales_prediction
GROUP BY category;