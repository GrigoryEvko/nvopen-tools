// Function: sub_7A0000
// Address: 0x7a0000
//
__int64 __fastcall sub_7A0000(__int64 a1, __int64 a2, unsigned __int64 a3, char *a4, __m128i *a5)
{
  char v5; // al
  __int64 result; // rax

  v5 = *(_BYTE *)(a2 + 173);
  if ( v5 != 1 )
  {
    if ( v5 == 3 )
    {
      result = 0;
      if ( (*(_BYTE *)(a2 + 171) & 4) == 0 )
        goto LABEL_9;
      return result;
    }
    return sub_79CCD0(a1, a2, a3, a4, a5);
  }
  if ( (*(_BYTE *)(a2 + 168) & 8) != 0 )
    return sub_79CCD0(a1, a2, a3, a4, a5);
  result = 0;
  if ( (*(_BYTE *)(a2 + 171) & 4) == 0 )
  {
LABEL_9:
    *(__m128i *)a3 = _mm_loadu_si128((const __m128i *)(a2 + 176));
    return 1;
  }
  return result;
}
