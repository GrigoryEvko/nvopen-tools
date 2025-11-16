// Function: sub_2D00D50
// Address: 0x2d00d50
//
__int64 __fastcall sub_2D00D50(unsigned int *a1, __int64 a2)
{
  unsigned __int64 v2; // rbx
  __int64 v3; // rax
  unsigned int v4; // r8d
  __int16 v5; // dx

  v2 = 1LL << (*(_WORD *)(a2 + 2) >> 1);
  LODWORD(v3) = sub_2D00C30(a1, *(_BYTE **)(a2 - 32));
  _BitScanReverse64(&v2, v2);
  v4 = 0;
  if ( (unsigned int)v3 > (unsigned int)(0x8000000000000000LL >> ((unsigned __int8)v2 ^ 0x3Fu))
    && (_DWORD)v3 != *a1
    && (_DWORD)v3 != a1[1] )
  {
    v5 = 510;
    if ( (_DWORD)v3 )
    {
      _BitScanReverse64((unsigned __int64 *)&v3, (unsigned int)v3);
      v5 = (2 * (63 - (v3 ^ 0x3F))) & 0x1FE;
    }
    v4 = 1;
    *(_WORD *)(a2 + 2) = *(_WORD *)(a2 + 2) & 0xFF81 | v5;
  }
  return v4;
}
