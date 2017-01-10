Twig
====

# For loop with if with and

```twig
<div class="row">
    {% for product in products %}
        <div class="span4">
            {# ... #}
        </div>

        {% if loop.index is divisibleby(3) and not loop.last %}
            </div><div class="row">
        {% endif %}
    {% endfor %}
</div>
```

# For loop with inline if check

```twig
<div class="row">
    {% for product in products if product.name is not empty %}
        <div class="span4">
            {{ product.name }}
        </div>
    {% endfor %}
</div>
```

# For loop with the usage of bash (array_chunk)

```twig
 <div class="grid">
    {% for rows in items | batch(4) %}
        <div class="grid__item">
            <ul class="nav__items">
                {% for colum in rows if colum.name is not empty %}
                    <li class="nav__item">
                        {{ column.name }}
                    </li>
                {% endfor %}
            </ul>
        </div>
    {% endfor %}
</div>
```

# Check if argument is a even number

```twig
{% if products|length is even %}
    <div class="row">
        <div class="span12">
            There is an even number of products! OMG!
        </div>
    </div>
{% endif %}
```

# A else check for when the for loop is not run

```twig
{% for product in products %}
    {# ... #}
{% else %}
    <div class="alert alert-error span12">
        It looks like we're out of really awesome-looking penguin clothes :/.
    </div>
{% endfor %}
```

# Inline if syntax

```twig
<div class="well" style="background-color: {{ backgroundColor is defined ? backgroundColor : 'lightBlue' }};">
    {# ... #}
</div>
```

# Usage of a variable within a string

```twig
{# Concat the icon variable into the iconClass string #}
{% set iconClass = "icon-${icon}"; %}
{% set foo = bar["static#{dynamicVar}"]; %}
```

# Resources
- Knp University - [The for “loop” and inline “if” Syntax](https://knpuniversity.com/screencast/twig/for-loop-inline-if)