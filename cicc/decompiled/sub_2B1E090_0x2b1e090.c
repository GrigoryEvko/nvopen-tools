// Function: sub_2B1E090
// Address: 0x2b1e090
//
__int64 __fastcall sub_2B1E090(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // r14
  __int64 v4; // r13
  unsigned int v6; // r8d
  int v8; // eax
  int v9; // esi
  unsigned int v10; // eax
  unsigned int v11; // eax

  v3 = a2;
  v4 = a2;
  if ( (_BYTE)qword_5010508 && *(_BYTE *)(a2 + 8) == 17 )
    v4 = **(_QWORD **)(a2 + 16);
  if ( !(unsigned __int8)sub_BCBCB0(v4) || (*(_BYTE *)(v4 + 8) & 0xFD) == 4 )
    goto LABEL_6;
  v8 = *(unsigned __int8 *)(a2 + 8);
  if ( (_BYTE)v8 == 17 )
  {
    v9 = a3 * *(_DWORD *)(a2 + 32);
  }
  else
  {
    v9 = a3;
    if ( (unsigned int)(v8 - 17) > 1 )
      goto LABEL_12;
  }
  v3 = **(_QWORD **)(v3 + 16);
LABEL_12:
  sub_BCDA70((__int64 *)v3, v9);
  v10 = sub_DFDB60(a1);
  v6 = v10;
  if ( !v10 || a3 <= v10 )
  {
LABEL_6:
    v6 = 1;
    if ( a3 > 1 )
    {
      _BitScanReverse(&a3, a3 - 1);
      return (unsigned int)(1 << (32 - (a3 ^ 0x1F)));
    }
    return v6;
  }
  v11 = (a3 != 0) + (a3 - (a3 != 0)) / v10;
  if ( v11 <= 1 )
    return v6;
  _BitScanReverse(&v11, v11 - 1);
  return v6 << (32 - (v11 ^ 0x1F));
}
