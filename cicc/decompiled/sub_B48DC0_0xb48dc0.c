// Function: sub_B48DC0
// Address: 0xb48dc0
//
__int64 __fastcall sub_B48DC0(__int64 a1)
{
  __int64 *v1; // rsi
  int v2; // ecx
  __int64 v3; // r8
  unsigned int i; // edx
  __int64 v5; // rax

  v1 = *(__int64 **)(a1 - 8);
  v2 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  v3 = *v1;
  if ( v2 != 1 )
  {
    for ( i = 1; v2 != i; ++i )
    {
      v5 = v1[4 * i];
      if ( v5 )
      {
        if ( v3 == v5 || v5 == a1 )
          continue;
        if ( v3 != a1 )
          return 0;
      }
      else
      {
        if ( !v3 )
          continue;
        if ( v3 != a1 )
          return 0;
      }
      v3 = v1[4 * i];
    }
  }
  if ( a1 == v3 )
    return sub_ACADE0(*(__int64 ***)(a1 + 8));
  else
    return v3;
}
