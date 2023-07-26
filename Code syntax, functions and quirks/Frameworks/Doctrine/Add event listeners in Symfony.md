# Add event listeners in Symfony (6.3)

## Lifecycle event listeners

The following code can be used to autoconfigure an event listener in Symfony 6.3
combined with Doctrine ORM 2.15.

```php
<?php

namespace App\Listener;

use Doctrine\Bundle\DoctrineBundle\Attribute\AsDoctrineListener;
use Doctrine\ORM\Event\PostFlushEventArgs;
use Doctrine\ORM\Event\PostLoadEventArgs;
use Doctrine\ORM\Event\PostPersistEventArgs;
use Doctrine\ORM\Event\PostUpdateEventArgs;
use Doctrine\ORM\Event\PreFlushEventArgs;
use Doctrine\ORM\Event\PrePersistEventArgs;
use Doctrine\ORM\Event\PreUpdateEventArgs;
use Doctrine\ORM\Events;

# The events are in the sequence in which they are possibly fired
#[AsDoctrineListener(event: Events::postLoad)] # Triggered when an existing entity is fetched
#[AsDoctrineListener(event: Events::preFlush)] # Is always called when adding an entity to the Unit of Work
#[AsDoctrineListener(event: Events::prePersist)] # Is only called when the added entity is new (INSERT)
#[AsDoctrineListener(event: Events::postPersist)] # Is only called when the added entity is new (INSERT)
#[AsDoctrineListener(event: Events::preUpdate)] # Is only called when an existing entity was modified and saved (UPDATE)
#[AsDoctrineListener(event: Events::postUpdate)] # Is only called when an existing entity was modified and saved (UPDATE)
#[AsDoctrineListener(event: Events::postFlush)] # Is always called when adding an entity to the Unit of Work
class DoctrineLifecycleListener
{
    public function preUpdate(PreUpdateEventArgs $args): void
    {
        $entity = $args->getObject();
        
        echo 'preUpdate'.PHP_EOL;
    }

    public function postUpdate(PostUpdateEventArgs $args): void
    {
        $entity = $args->getObject();
        
        echo 'postUpdate'.PHP_EOL;
    }

    public function preFlush(PreFlushEventArgs $args): void
    {
        echo 'preFlush'.PHP_EOL;
    }

    public function postFlush(PostFlushEventArgs $args): void
    {
        echo 'postFlush'.PHP_EOL;
    }

    public function prePersist(PrePersistEventArgs $args): void
    {
        $entity = $args->getObject();
        
        echo 'prePersist'.PHP_EOL;
    }

    public function postPersist(PostPersistEventArgs $args): void
    {
        $entity = $args->getObject();
        
        echo 'postPersist'.PHP_EOL;
    }

    public function postLoad(PostLoadEventArgs $args): void
    {
        $entity = $args->getObject();
        
        echo 'postLoad'.PHP_EOL;
    }
}

```

## Resources

- [Doctrine](https://www.doctrine-project.org/projects/doctrine-orm/en/current/reference/events.html#events-overview)
- [Symfony Doctrine Events](https://symfony.com/doc/current/doctrine/events.html#doctrine-lifecycle-listeners)