// Function: sub_2B08900
// Address: 0x2b08900
//
__int64 __fastcall sub_2B08900(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rsi
  unsigned __int64 v4; // rax
  unsigned int v5; // r8d
  unsigned __int64 v6; // rax
  int v7; // eax

  v2 = a1 + 8;
  v3 = a1 + 8 * a2;
  _BitScanReverse64(&v4, 1LL << (*(_WORD *)(*(_QWORD *)a1 + 2LL) >> 1));
  v5 = 63 - (v4 ^ 0x3F);
  if ( v3 != a1 + 8 )
  {
    do
    {
      _BitScanReverse64(&v6, 1LL << (*(_WORD *)(*(_QWORD *)v2 + 2LL) >> 1));
      v7 = v6 ^ 0x3F;
      if ( (unsigned __int8)v5 > (unsigned __int8)(63 - v7) )
        v5 = 63 - v7;
      v2 += 8;
    }
    while ( v3 != v2 );
  }
  return v5;
}
