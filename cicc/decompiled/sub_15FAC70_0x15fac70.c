// Function: sub_15FAC70
// Address: 0x15fac70
//
__int64 __fastcall sub_15FAC70(int *a1, int a2)
{
  unsigned int v2; // r8d
  __int64 v3; // rcx

  v2 = sub_15FAB40(a1, a2);
  if ( !(_BYTE)v2 || a2 <= 0 )
    return v2;
  v3 = (__int64)&a1[a2 - 1 + 1];
  while ( (unsigned int)(*a1 + 1) <= 1 || *a1 == a2 )
  {
    if ( (int *)v3 == ++a1 )
      return v2;
  }
  return 0;
}
