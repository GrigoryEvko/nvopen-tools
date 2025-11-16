// Function: sub_336E870
// Address: 0x336e870
//
__int16 __fastcall sub_336E870(unsigned int *a1, __int64 a2)
{
  unsigned int v2; // edx
  unsigned int v3; // eax
  int v4; // edx
  unsigned int v5; // eax
  unsigned int v6; // edx
  unsigned int v7; // eax
  unsigned int v8; // edx
  unsigned int v9; // eax
  unsigned int v10; // edx

  v2 = *a1 & 0xFFFFFFDF;
  if ( (*(_BYTE *)(a2 + 1) & 4) != 0 )
    v2 = *a1 | 0x20;
  v3 = v2;
  *a1 = v2;
  v4 = v2 | 0x40;
  v5 = v3 & 0xFFFFFFBF;
  if ( (*(_BYTE *)(a2 + 1) & 8) != 0 )
    v5 = v4;
  v6 = v5;
  *a1 = v5;
  LOBYTE(v5) = v5 | 0x80;
  LOBYTE(v6) = v6 & 0x7F;
  if ( (*(_BYTE *)(a2 + 1) & 0x10) != 0 )
    v6 = v5;
  v7 = v6;
  *a1 = v6;
  BYTE1(v6) |= 1u;
  BYTE1(v7) &= ~1u;
  if ( (*(_BYTE *)(a2 + 1) & 0x20) != 0 )
    v7 = v6;
  v8 = v7;
  *a1 = v7;
  BYTE1(v7) |= 2u;
  BYTE1(v8) &= ~2u;
  if ( (*(_BYTE *)(a2 + 1) & 0x40) != 0 )
    v8 = v7;
  v9 = v8;
  *a1 = v8;
  BYTE1(v8) |= 4u;
  BYTE1(v9) &= ~4u;
  if ( *(char *)(a2 + 1) < 0 )
    v9 = v8;
  v10 = v9;
  *a1 = v9;
  BYTE1(v9) |= 8u;
  BYTE1(v10) &= ~8u;
  if ( (*(_BYTE *)(a2 + 1) & 2) != 0 )
    v10 = v9;
  *a1 = v10;
  return v9;
}
