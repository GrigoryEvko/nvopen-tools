// Function: sub_15FAC00
// Address: 0x15fac00
//
__int64 __fastcall sub_15FAC00(int *a1, int a2)
{
  unsigned int v2; // r8d
  int i; // eax
  int v4; // edx

  v2 = sub_15FAB40(a1, a2);
  if ( !(_BYTE)v2 || a2 <= 0 )
    return v2;
  for ( i = a2 - 1; ; --i )
  {
    v4 = *a1;
    if ( *a1 != -1 && v4 != i && v4 != a2 + i )
      break;
    ++a1;
    if ( i == 0 )
      return v2;
  }
  return 0;
}
