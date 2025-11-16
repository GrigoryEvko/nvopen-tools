// Function: sub_B4EE20
// Address: 0xb4ee20
//
__int64 __fastcall sub_B4EE20(int *a1, __int64 a2, int a3)
{
  unsigned int v6; // r8d
  __int64 v7; // rcx

  if ( a3 != a2 )
    return 0;
  v6 = sub_B4ED30(a1, (unsigned int)a3, a3);
  if ( (_BYTE)v6 )
  {
    if ( a3 <= 0 )
      return v6;
    v7 = (__int64)&a1[a3 - 1 + 1];
    while ( (unsigned int)(*a1 + 1) <= 1 || *a1 == a3 )
    {
      if ( (int *)v7 == ++a1 )
        return v6;
    }
  }
  return 0;
}
