// Function: sub_23B8640
// Address: 0x23b8640
//
__int64 __fastcall sub_23B8640(__m128i *a1, const __m128i *a2)
{
  __int64 result; // rax

  a1[1].m128i_i64[1] = 0;
  result = a2[1].m128i_i64[1];
  a1[1].m128i_i64[1] = result;
  if ( (result & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    if ( (result & 2) != 0 )
    {
      if ( (result & 4) != 0 )
      {
        (*(void (**)(void))((result & 0xFFFFFFFFFFFFFFF8LL) + 8))();
        result = (*(__int64 (__fastcall **)(const __m128i *))((a1[1].m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) + 16))(a2);
      }
      else
      {
        *a1 = _mm_loadu_si128(a2);
        result = a2[1].m128i_i64[0];
        a1[1].m128i_i64[0] = result;
      }
      a2[1].m128i_i64[1] = 0;
    }
    else
    {
      *a1 = _mm_loadu_si128(a2);
      result = a2[1].m128i_i64[0];
      a1[1].m128i_i64[0] = result;
      a2[1].m128i_i64[1] = 0;
    }
  }
  return result;
}
