#MySQL Dumps

Use the following steps to create a MySQL dump, compress it and (after moving to the other server) insert it into another database

Retrieving the dump and compress it in one run

```sh
mysqldump --skip-lock-tables -u root -p DATABASE | gzip -v9 > DATABASE.sql.gz
```

Read the content of the compressed file and import it in the database

```sh
nohup zcat DEBITEURENBEHEER.sql.gz | mysql -u USER -pPASSWORD DATABASE &
```

This command is composed of the following parts:

```sh
nohup .... &
```

`nohup` ensures that the command is run under *No Hangup*. This keeps the process running even when you close your terminal.
The `&` at the end sends the output of the commands to `stdout` which enables you to see feedback when send.

`zcat` reads the contents of a compressed file and sends the output to `stdout`. There it is passed into the pipeline `|` where
it is inserted into the `mysql` command to update the data in de database.
