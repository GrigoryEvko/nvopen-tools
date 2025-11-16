// Function: sub_1E34340
// Address: 0x1e34340
//
__int64 __fastcall sub_1E34340(__m128i *a1, const __m128i *a2)
{
  unsigned int v2; // edx
  __int64 result; // rax
  __int16 v4; // ax

  v2 = (unsigned int)(1 << a2[2].m128i_i16[1]) >> 1;
  result = (unsigned int)(1 << a1[2].m128i_i16[1]) >> 1;
  if ( (unsigned int)result <= v2 )
  {
    v4 = 0;
    if ( v2 )
    {
      _BitScanReverse(&v2, v2);
      v4 = 31 - (v2 ^ 0x1F) + 1;
    }
    a1[2].m128i_i16[1] = v4;
    *a1 = _mm_loadu_si128(a2);
    result = a2[1].m128i_i64[0];
    a1[1].m128i_i64[0] = result;
  }
  return result;
}
