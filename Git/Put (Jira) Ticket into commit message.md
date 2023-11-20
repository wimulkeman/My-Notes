With the following `prepare-commit-msg` template you can add the ticket code of a branchname into the commit message before finalizing the commit by git.

```bash
#!/bin/sh
#
# A hook script to prepare the commit log message.
# If the branch name it's a jira Ticket.
# It adds the branch name to the commit message, if it is not already part of it.

branchPath="$(git symbolic-ref -q HEAD)" #Something like refs/heads/myBranchName
branchName=${branchPath##*/}      #Get text behind the last / of the branch path

regex="([A-Za-z0-9]{1,10}-[0-9]{1,10})[-_]*"

if [[ $branchName =~ $regex ]]
then
    # Get the captured portion of the branch name.
    jiraTicketName=$(echo "${BASH_REMATCH[1]^}" | tr 'a-z' 'A-Z')
    originalMessage=`cat $1`

    # If the message already begins with PROJECTNAME-#, do not edit the commit message.
    if [[ $originalMessage == $jiraTicketName* ]]
        then
        exit
    fi

    sed -i '.bak' "1s/^/$jiraTicketName: /" $1 #Insert branch name at the start of the commit message file
fi
```
