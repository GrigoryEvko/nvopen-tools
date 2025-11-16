// Function: sub_2E45000
// Address: 0x2e45000
//
__int64 __fastcall sub_2E45000(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 v5; // r13
  __int64 v6; // rbx
  char v7; // al
  unsigned int v8; // edx
  unsigned int v9; // esi

  v4 = *(_QWORD *)(a2 + 32);
  v5 = v4 + 40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF);
  v6 = v4 + 40LL * (unsigned int)sub_2E88FE0(a2);
  if ( v5 == v6 )
    return 0;
  while ( 1 )
  {
    if ( a3 != v6 && !*(_BYTE *)v6 )
    {
      v7 = *(_BYTE *)(v6 + 3);
      if ( (v7 & 0x20) != 0 && (v7 & 0x10) == 0 )
      {
        v8 = *(_DWORD *)(v6 + 8);
        v9 = *(_DWORD *)(a3 + 8);
        if ( v9 == v8 || v9 - 1 <= 0x3FFFFFFE && v8 - 1 <= 0x3FFFFFFE && (unsigned __int8)sub_E92070(*a1, v9, v8) )
          break;
      }
    }
    v6 += 40;
    if ( v5 == v6 )
      return 0;
  }
  return 1;
}
