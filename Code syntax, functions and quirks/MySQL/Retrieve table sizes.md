Retrieve table sizes
====================

This command can be used to retrieve the
size of all the tables within the databases
available on a server.

```mysql
SELECT 
     table_schema as `Database`, 
     table_name AS `Table`, 
     round(((data_length + index_length) / 1024 / 1024), 2) `Size in MB` 
FROM information_schema.TABLES 
ORDER BY (data_length + index_length) DESC;
```

Or to retrieve the complete size of database itself.

```mysql
SELECT 
     table_schema as `Database`, 
     table_name AS `Table`, 
     round((SUM(data_length + index_length) / 1024 / 1024 / 1024), 2) `Size in GB` 
FROM information_schema.TABLES 
GROUP BY table_schema
ORDER BY (data_length + index_length) DESC;
```
