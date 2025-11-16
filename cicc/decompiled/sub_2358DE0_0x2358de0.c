// Function: sub_2358DE0
// Address: 0x2358de0
//
unsigned __int64 __fastcall sub_2358DE0(unsigned __int64 *a1, unsigned __int64 **a2)
{
  unsigned __int64 *v2; // r13
  unsigned __int64 *i; // rbx
  unsigned __int64 result; // rax
  char *v5; // rsi
  unsigned __int64 *v6; // rdx

  v2 = a2[1];
  for ( i = *a2; v2 != i; a1[1] = (unsigned __int64)(v5 + 8) )
  {
    while ( 1 )
    {
      v5 = (char *)a1[1];
      if ( v5 != (char *)a1[2] )
        break;
      v6 = i++;
      result = sub_2275C60(a1, v5, v6);
      if ( v2 == i )
        return result;
    }
    if ( v5 )
    {
      result = *i;
      *(_QWORD *)v5 = *i;
      *i = 0;
      v5 = (char *)a1[1];
    }
    ++i;
  }
  return result;
}
