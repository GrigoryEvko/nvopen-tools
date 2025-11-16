// Function: sub_134B450
// Address: 0x134b450
//
__int64 __fastcall sub_134B450(__int64 a1)
{
  __int64 v1; // rax
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rax
  unsigned int v5; // eax
  char v6; // cl
  unsigned int v7; // eax
  __int64 v8; // rdx

  v1 = *(_QWORD *)(a1 + 104);
  if ( !v1 )
    return 126 - ((*(_BYTE *)(a1 + 16) == 0) - 1LL);
  v3 = sub_130F9A0((*(_QWORD *)(a1 + 176) - v1) << 12);
  if ( v3 > 0x7000000000000000LL )
  {
    v8 = 398;
  }
  else
  {
    _BitScanReverse64(&v4, v3);
    v5 = v4 - ((((v3 - 1) & v3) == 0) - 1);
    if ( v5 < 0xE )
      v5 = 14;
    v6 = v5 - 3;
    v7 = v5 - 14;
    if ( !v7 )
      v6 = 12;
    v8 = 2 * ((((v3 - 1) >> v6) & 3) + 4 * v7);
  }
  return v8 + (*(_BYTE *)(a1 + 16) ^ 1u);
}
