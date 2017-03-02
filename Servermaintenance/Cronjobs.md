Cronjobs
========

# Running cronjobs on changing times

Runs the task every 20 minutes between 5 en 59 minutes over
the hour.

```bash
5-59/20 * * * * #(Runs 2, 25, 45)
10-59/25 * * * * #(Runs 10, 35)
1-59/2 * * * * #(Runs every odd minute)
```

- Stackoverflow [Run Cron job every N minutes plus offset
](http://stackoverflow.com/a/19204734)