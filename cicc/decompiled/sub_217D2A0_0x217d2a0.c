// Function: sub_217D2A0
// Address: 0x217d2a0
//
__int64 __fastcall sub_217D2A0(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5, __int64 a6)
{
  int v6; // eax
  int v8; // eax
  unsigned int v9; // edx

  v6 = *(__int16 *)(a1 + 24);
  if ( *(_WORD *)(a1 + 24) == 119 )
    return sub_217BB50(a1, a2, *(_DWORD *)(a6 + 252), a3, a4, a5);
  if ( (v6 & 0x8000u) == 0 )
    return 0;
  v8 = ~v6;
  if ( v8 == 3243 )
    return sub_217B110(a1, a2, a3, a4, a5);
  if ( v8 <= 3243 )
  {
    if ( (unsigned int)(v8 - 164) > 1 )
      return 0;
    return sub_217B110(a1, a2, a3, a4, a5);
  }
  if ( (unsigned int)(v8 - 4449) > 1 )
    return 0;
  v9 = *(_DWORD *)(a6 + 252);
  if ( *(__int16 *)(a1 + 24) < 0 )
    return sub_217C7E0(a1, a2, v9, a3, a4, a5);
  else
    return 0;
}
