Postfix commands
================

## Display a list of queued mail, deferred and pending

```bash
mailq
```

Alternative:

```bash
postqueue -p
```

## View message (contents, header and body) in Postfix queue

```bash
postcat -vq XXXmessageIdXXX
```

## Tell Postfix to process the Queue now

```bash
postqueue -f
```

Alternative:

```bash
postfix flush
```

## Delete queued mail

Remove all mails in the queue.

```bash
postsuper -d ALL
```

Remove the deferred mails. These are the mails the server
has listed to retry sending to the other mailserver.

```bash
postsuper -d ALL deferred
```

## Remove mails from the mailqueue with conditions

This command deletes the mails from a queue which are
send to or are send from a specific domain.

```bash
mailq | tail +2 | awk 'BEGIN { RS = "" } / falko@example\.com$/ { print $1 }' | tr -d '*!' | postsuper -d -
```
