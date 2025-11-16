// Function: sub_FFECA0
// Address: 0xffeca0
//
char *__fastcall sub_FFECA0(char *a1)
{
  char v1; // dl
  char *result; // rax

  v1 = *a1;
  if ( (unsigned __int8)*a1 <= 0x1Cu )
  {
    result = 0;
    if ( v1 == 5 && *((_WORD *)a1 + 1) == 47 )
      return a1;
  }
  else
  {
    result = 0;
    if ( v1 == 76 )
      return a1;
  }
  return result;
}
