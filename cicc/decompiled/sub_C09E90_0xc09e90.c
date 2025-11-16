// Function: sub_C09E90
// Address: 0xc09e90
//
char *__fastcall sub_C09E90(char **a1, __int64 (__fastcall *a2)(__int64, _QWORD), __int64 a3)
{
  char *v3; // r15
  char *result; // rax
  char *v6; // rbx
  char i; // al

  v3 = a1[1];
  result = *a1;
  if ( v3 )
  {
    v6 = *a1;
    for ( i = a2(a3, (unsigned int)**a1); ; i = a2(a3, (unsigned int)*v6) )
    {
      if ( !i )
        return *a1;
      ++v6;
      if ( !--v3 )
        break;
    }
    return *a1;
  }
  return result;
}
