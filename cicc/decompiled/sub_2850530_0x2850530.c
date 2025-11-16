// Function: sub_2850530
// Address: 0x2850530
//
char *__fastcall sub_2850530(char *a1)
{
  char v1; // dl
  char *result; // rax

  v1 = *a1;
  if ( (unsigned __int8)*a1 <= 0x1Cu )
  {
    result = 0;
    if ( v1 == 5 && *((_WORD *)a1 + 1) == 13 )
      return a1;
  }
  else
  {
    result = 0;
    if ( v1 == 42 )
      return a1;
  }
  return result;
}
