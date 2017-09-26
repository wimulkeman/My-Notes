Free disk space for a InnoDB table
==================================

After deleting records from a InnoDB table,
the disk space is not freed.

To free the disk space, the file_per_table
setting needs to be enabled (MySQL 5.6).
When enabled, the following query can be
run.

```mysql
OPTIMIZE TABLE `table`;
```
