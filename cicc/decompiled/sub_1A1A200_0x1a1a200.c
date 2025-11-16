// Function: sub_1A1A200
// Address: 0x1a1a200
//
__int64 __fastcall sub_1A1A200(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // ecx
  __int64 v5; // rsi
  unsigned int v6; // eax

  v4 = *(unsigned __int16 *)(a1 + 18);
  if ( *(_BYTE *)(a1 + 16) != 54 )
  {
    v6 = 1 << (v4 >> 1) >> 1;
    v5 = **(_QWORD **)(a1 - 48);
    if ( v6 )
      return -(a2 | v6) & (a2 | v6);
LABEL_5:
    v6 = sub_15A9FE0(a3, v5);
    return -(a2 | v6) & (a2 | v6);
  }
  v5 = *(_QWORD *)a1;
  v6 = 1 << (v4 >> 1) >> 1;
  if ( !v6 )
    goto LABEL_5;
  return -(a2 | v6) & (a2 | v6);
}
