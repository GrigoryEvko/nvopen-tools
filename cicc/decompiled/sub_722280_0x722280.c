// Function: sub_722280
// Address: 0x722280
//
char *__fastcall sub_722280(char *s)
{
  char *v2; // rax

  if ( (*s != 45 || s[1]) && (v2 = sub_722110(s)) != 0 )
    return v2 + 1;
  else
    return s;
}
