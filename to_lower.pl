use strict;
use utf8;

binmode(STDIN, ":encoding(utf-8)");
binmode(STDOUT, ":encoding(utf-8)");

while (<STDIN>) {
  print lc($_);
}
