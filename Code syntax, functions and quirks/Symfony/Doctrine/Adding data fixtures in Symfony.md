Adding data fixtures in Symfony
===============================

First make sure the Doctrine fixtures bundles are available in the project

```bash
composer require --dev doctrine/doctrine-fixtures-bundle
```

Then register the fixtures bundle in app/AppKernel.php

```php
class AppKernel extends Kernel
{
    public function registerBundles()
    {
        // ...
        if (in_array($this->getEnvironment(), array('dev', 'test'))) {
            $bundles[] = new Doctrine\Bundle\FixturesBundle\DoctrineFixturesBundle();
        }

        return $bundles
    }

    // ...
}
```

After that, its time to create the fixtures. Those are placed in the bundle
where the entities are available on which the data is added. For the AcmeBundle
this could be

```bash
src/AcmeBundle/DataFixtures/ORM/Load[Entity]Data.php
```

## References
- [Symfony docs on Doctrine fixtures](http://symfony.com/doc/current/bundles/DoctrineFixturesBundle/index.html)
- [Blog about Data Fixtures in Symfony2](https://www.sitepoint.com/data-fixtures-symfony2/)
- [AliceBundle: fixtures with yml](https://github.com/hautelook/AliceBundle)