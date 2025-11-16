// Function: sub_15F5600
// Address: 0x15f5600
//
__int64 __fastcall sub_15F5600(__int64 a1)
{
  unsigned int v1; // ecx
  __int64 *v2; // rsi
  __int64 v3; // r8
  unsigned int i; // edx
  __int64 v5; // rax

  v1 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v2 = *(__int64 **)(a1 - 8);
  else
    v2 = (__int64 *)(a1 - 24LL * v1);
  v3 = *v2;
  if ( v1 != 1 )
  {
    for ( i = 1; v1 != i; ++i )
    {
      v5 = v2[3 * i];
      if ( v5 )
      {
        if ( v5 == v3 || a1 == v5 )
          continue;
        if ( a1 != v3 )
          return 0;
      }
      else
      {
        if ( !v3 )
          continue;
        if ( a1 != v3 )
          return 0;
      }
      v3 = v2[3 * i];
    }
  }
  if ( a1 == v3 )
    return sub_1599EF0(*(__int64 ***)a1);
  else
    return v3;
}
