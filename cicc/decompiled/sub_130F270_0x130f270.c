// Function: sub_130F270
// Address: 0x130f270
//
__int64 __fastcall sub_130F270(unsigned int *a1)
{
  __int64 result; // rax
  int v2; // r13d
  const char *v3; // r14
  int i; // r12d

  if ( *((_BYTE *)a1 + 29) )
  {
    *((_BYTE *)a1 + 29) = 0;
    return result;
  }
  if ( *((_BYTE *)a1 + 28) )
  {
    result = sub_130F0B0((__int64)a1, ",");
    if ( *a1 == 1 )
      return result;
  }
  else if ( *a1 == 1 )
  {
    return result;
  }
  sub_130F0B0((__int64)a1, "\n");
  result = *a1;
  v2 = a1[6];
  v3 = "\t";
  if ( (_DWORD)result )
  {
    v2 *= 2;
    v3 = " ";
  }
  if ( v2 > 0 )
  {
    for ( i = 0; i != v2; ++i )
      result = sub_130F0B0((__int64)a1, "%s", v3);
  }
  return result;
}
