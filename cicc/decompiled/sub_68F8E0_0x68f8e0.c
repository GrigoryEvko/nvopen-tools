// Function: sub_68F8E0
// Address: 0x68f8e0
//
__int64 __fastcall sub_68F8E0(__m128i *a1, const __m128i *a2)
{
  __int64 result; // rax

  *a1 = _mm_loadu_si128(a2);
  a1[1] = _mm_loadu_si128(a2 + 1);
  a1[2] = _mm_loadu_si128(a2 + 2);
  a1[3] = _mm_loadu_si128(a2 + 3);
  a1[4] = _mm_loadu_si128(a2 + 4);
  a1[5] = _mm_loadu_si128(a2 + 5);
  a1[6] = _mm_loadu_si128(a2 + 6);
  a1[7] = _mm_loadu_si128(a2 + 7);
  a1[8] = _mm_loadu_si128(a2 + 8);
  result = a2[1].m128i_u8[0];
  if ( (_BYTE)result == 2 )
  {
    a1[9] = _mm_loadu_si128(a2 + 9);
    a1[10] = _mm_loadu_si128(a2 + 10);
    a1[11] = _mm_loadu_si128(a2 + 11);
    a1[12] = _mm_loadu_si128(a2 + 12);
    a1[13] = _mm_loadu_si128(a2 + 13);
    a1[14] = _mm_loadu_si128(a2 + 14);
    a1[15] = _mm_loadu_si128(a2 + 15);
    a1[16] = _mm_loadu_si128(a2 + 16);
    a1[17] = _mm_loadu_si128(a2 + 17);
    a1[18] = _mm_loadu_si128(a2 + 18);
    a1[19] = _mm_loadu_si128(a2 + 19);
    a1[20] = _mm_loadu_si128(a2 + 20);
    a1[21] = _mm_loadu_si128(a2 + 21);
  }
  else if ( (_BYTE)result == 5 || (_BYTE)result == 1 )
  {
    result = a2[9].m128i_i64[0];
    a1[9].m128i_i64[0] = result;
  }
  return result;
}
