MD5 hash differs between two tables
===================================

TL/DR; Two different hashes for the same string because of two different encodings.

On a checkup between two tables, the generated MD5 hash of
the same string value was different.

On debugging the issue, it turned out the encoding of the two tables were different.
One table was set on UTF-8, and the other on UTF-16.