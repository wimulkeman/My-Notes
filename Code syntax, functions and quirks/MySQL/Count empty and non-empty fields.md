Count empty and non-empty fields
================================

```mysql
SELECT COUNT(*) AS `total`, SUM(IF(`text` <> "",1,0)) AS `non_empty` FROM `table`
```
