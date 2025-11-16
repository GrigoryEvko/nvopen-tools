// Function: sub_68AD60
// Address: 0x68ad60
//
const __m128i *__fastcall sub_68AD60(__m128i *a1)
{
  const __m128i *result; // rax
  unsigned int v2; // eax
  _BYTE v3[4]; // [rsp+4h] [rbp-1Ch] BYREF
  _BYTE v4[16]; // [rsp+8h] [rbp-18h] BYREF

  if ( dword_4F077BC
    && (dword_4F077C4 != 2 || unk_4F07778 <= 202001)
    && (unsigned int)sub_8D2A90(a1->m128i_i64[0])
    && (a1[1].m128i_i8[1] != 1 || (unsigned int)sub_6ED0A0(a1)) )
  {
    v2 = sub_6E92D0();
    sub_6E68E0(v2, a1);
    result = (const __m128i *)a1[1].m128i_u8[0];
    if ( (_BYTE)result != 3 )
      goto LABEL_5;
  }
  else
  {
    result = (const __m128i *)a1[1].m128i_u8[0];
    if ( (_BYTE)result != 3 )
    {
LABEL_5:
      if ( (_BYTE)result == 1 )
      {
        result = (const __m128i *)a1[9].m128i_i64[0];
        if ( result[1].m128i_i8[8] == 1 && result[3].m128i_i8[8] == 116 )
        {
          result = (const __m128i *)result[4].m128i_i64[1];
          if ( result[1].m128i_i8[8] == 2 )
          {
            result = (const __m128i *)result[3].m128i_i64[1];
            if ( result[10].m128i_i8[13] == 12 && !result[11].m128i_i8[0] )
            {
              a1[1].m128i_i16[0] = 514;
              a1[9] = _mm_loadu_si128(result);
              a1[10] = _mm_loadu_si128(result + 1);
              a1[11] = _mm_loadu_si128(result + 2);
              a1[12] = _mm_loadu_si128(result + 3);
              a1[13] = _mm_loadu_si128(result + 4);
              a1[14] = _mm_loadu_si128(result + 5);
              a1[15] = _mm_loadu_si128(result + 6);
              a1[16] = _mm_loadu_si128(result + 7);
              a1[17] = _mm_loadu_si128(result + 8);
              a1[18] = _mm_loadu_si128(result + 9);
              a1[19] = _mm_loadu_si128(result + 10);
              a1[20] = _mm_loadu_si128(result + 11);
              a1[21] = _mm_loadu_si128(result + 12);
            }
          }
        }
      }
      return result;
    }
  }
  if ( (a1[1].m128i_i8[3] & 8) != 0 )
  {
    result = (const __m128i *)a1[8].m128i_i64[1];
    if ( result[5].m128i_i8[0] != 17 )
      return (const __m128i *)sub_6F6860(a1, 0, v4, v3);
  }
  return result;
}
