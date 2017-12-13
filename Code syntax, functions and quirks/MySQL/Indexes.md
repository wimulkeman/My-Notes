# Indexes

## Get index usage

To see how many times an index is used, you can use this query

```mysql
select * from performance_schema.table_io_waits_summary_by_index_usage
where object_schema = 'your_schema'
```

This is available since MySQL 5.5.3