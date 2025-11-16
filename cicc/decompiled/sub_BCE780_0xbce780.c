// Function: sub_BCE780
// Address: 0xbce780
//
__int64 __fastcall sub_BCE780(__int64 a1)
{
  unsigned __int64 v1; // r12
  __int64 *v2; // r14
  __int64 v3; // r13
  __int64 *v4; // rax
  __int64 *v6; // rax
  int v7; // r12d
  __int64 v8; // rax
  int v9; // r12d
  __int64 *v10; // rax

  v1 = *(_QWORD *)(a1 + 32);
  v2 = *(__int64 **)a1;
  v3 = *(_QWORD *)(a1 + 24);
  if ( v1 == 11 )
  {
    if ( *(_QWORD *)v3 == 0x6D492E7672697073LL && *(_WORD *)(v3 + 8) == 26465 && *(_BYTE *)(v3 + 10) == 101
      || *(_DWORD *)v3 == 1919512691 && *(_WORD *)(v3 + 4) == 11894 )
    {
      return sub_BCE3C0(v2, 0);
    }
LABEL_4:
    if ( memcmp((const void *)v3, "dx.", 3u) )
    {
      if ( v1 == 20
        && !(*(_QWORD *)v3 ^ 0x6E2E6E6367646D61LL | *(_QWORD *)(v3 + 8) ^ 0x7261622E64656D61LL)
        && *(_DWORD *)(v3 + 16) == 1919248754 )
      {
        v4 = (__int64 *)sub_BCB2D0(v2);
        return sub_BCDA70(v4, 4);
      }
      return sub_BCB120(v2);
    }
    return sub_BCE3C0(v2, 0);
  }
  if ( v1 <= 5 )
  {
    if ( v1 > 2 )
      goto LABEL_4;
    return sub_BCB120(v2);
  }
  if ( !memcmp(*(const void **)(a1 + 24), "spirv.", 6u) )
    return sub_BCE3C0(v2, 0);
  if ( v1 == 15 )
  {
    if ( *(_QWORD *)v3 == 0x2E34366863726161LL
      && *(_DWORD *)(v3 + 8) == 1868789363
      && *(_WORD *)(v3 + 12) == 28277
      && *(_BYTE *)(v3 + 14) == 116 )
    {
      v6 = (__int64 *)sub_BCB2A0(v2);
      return sub_BCDE10(v6, 16);
    }
LABEL_20:
    if ( *(_WORD *)v3 == 30820 && *(_BYTE *)(v3 + 2) == 46 )
      return sub_BCE3C0(v2, 0);
    return sub_BCB120(v2);
  }
  if ( v1 != 18 )
    goto LABEL_4;
  if ( *(_QWORD *)v3 ^ 0x65762E7663736972LL | *(_QWORD *)(v3 + 8) ^ 0x7075742E726F7463LL || *(_WORD *)(v3 + 16) != 25964 )
    goto LABEL_20;
  v7 = 8;
  v8 = **(_QWORD **)(a1 + 16);
  if ( *(_DWORD *)(v8 + 32) >= 8u )
    v7 = *(_DWORD *)(v8 + 32);
  v9 = **(_DWORD **)(a1 + 40) * v7;
  v10 = (__int64 *)sub_BCB2B0(v2);
  return sub_BCDE10(v10, v9);
}
