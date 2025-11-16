// Function: sub_1347EF0
// Address: 0x1347ef0
//
__int64 __fastcall sub_1347EF0(__m128i *a1, __int64 a2, const __m128i *a3)
{
  __int64 result; // rax

  if ( (unsigned __int8)sub_130AF40((__int64)a1[7].m128i_i64) )
    return 1;
  result = sub_130AF40((__int64)a1);
  if ( (_BYTE)result )
    return 1;
  a1[15].m128i_i64[0] = a2;
  a1[14].m128i_i64[0] = 0;
  a1[14].m128i_i64[1] = 0;
  a1[15].m128i_i64[1] = 0;
  a1[16] = _mm_loadu_si128(a3);
  a1[17] = _mm_loadu_si128(a3 + 1);
  a1[18] = _mm_loadu_si128(a3 + 2);
  a1[19].m128i_i64[0] = a3[3].m128i_i64[0];
  return result;
}
