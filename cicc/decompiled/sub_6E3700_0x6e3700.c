// Function: sub_6E3700
// Address: 0x6e3700
//
__m128i *__fastcall sub_6E3700(const __m128i *a1, __m128i *a2)
{
  __int64 v2; // rcx
  __m128i *result; // rax
  __int8 v4; // dl
  __int64 v5; // rbx
  __int64 v6; // rax
  __int8 v7; // al

  if ( a2 )
  {
    v2 = a2[8].m128i_i64[0];
    result = a2;
    *a2 = _mm_loadu_si128(a1);
    a2[1] = _mm_loadu_si128(a1 + 1);
    a2[2] = _mm_loadu_si128(a1 + 2);
    a2[3] = _mm_loadu_si128(a1 + 3);
    a2[4] = _mm_loadu_si128(a1 + 4);
    a2[5] = _mm_loadu_si128(a1 + 5);
    a2[6] = _mm_loadu_si128(a1 + 6);
    a2[7] = _mm_loadu_si128(a1 + 7);
    a2[8] = _mm_loadu_si128(a1 + 8);
    v4 = a1[1].m128i_i8[0];
    if ( v4 != 2 )
    {
      if ( v4 == 5 || v4 == 1 )
        a2[9].m128i_i64[0] = a1[9].m128i_i64[0];
      goto LABEL_5;
    }
LABEL_10:
    result[9] = _mm_loadu_si128(a1 + 9);
    result[10] = _mm_loadu_si128(a1 + 10);
    result[11] = _mm_loadu_si128(a1 + 11);
    result[12] = _mm_loadu_si128(a1 + 12);
    result[13] = _mm_loadu_si128(a1 + 13);
    result[14] = _mm_loadu_si128(a1 + 14);
    result[15] = _mm_loadu_si128(a1 + 15);
    result[16] = _mm_loadu_si128(a1 + 16);
    result[17] = _mm_loadu_si128(a1 + 17);
    result[18] = _mm_loadu_si128(a1 + 18);
    result[19] = _mm_loadu_si128(a1 + 19);
    result[20] = _mm_loadu_si128(a1 + 20);
    result[21] = _mm_loadu_si128(a1 + 21);
LABEL_5:
    result[22].m128i_i8[0] = *(_BYTE *)(qword_4D03C50 + 16LL);
    if ( v2 )
      result[8].m128i_i64[0] = v2;
    return result;
  }
  v5 = sub_823970(384);
  sub_6E2E50(0, v5);
  *(_BYTE *)(v5 + 352) = 4;
  *(_DWORD *)(v5 + 364) = 0;
  *(_QWORD *)(v5 + 376) = 0;
  v6 = *(_QWORD *)&dword_4F077C8;
  *(_QWORD *)(v5 + 356) = *(_QWORD *)&dword_4F077C8;
  *(_QWORD *)(v5 + 368) = v6;
  *(__m128i *)v5 = _mm_loadu_si128(a1);
  *(__m128i *)(v5 + 16) = _mm_loadu_si128(a1 + 1);
  *(__m128i *)(v5 + 32) = _mm_loadu_si128(a1 + 2);
  *(__m128i *)(v5 + 48) = _mm_loadu_si128(a1 + 3);
  *(__m128i *)(v5 + 64) = _mm_loadu_si128(a1 + 4);
  *(__m128i *)(v5 + 80) = _mm_loadu_si128(a1 + 5);
  *(__m128i *)(v5 + 96) = _mm_loadu_si128(a1 + 6);
  *(__m128i *)(v5 + 112) = _mm_loadu_si128(a1 + 7);
  *(__m128i *)(v5 + 128) = _mm_loadu_si128(a1 + 8);
  v7 = a1[1].m128i_i8[0];
  if ( v7 == 2 )
  {
    v2 = 0;
    result = (__m128i *)v5;
    goto LABEL_10;
  }
  if ( v7 == 5 || v7 == 1 )
    *(_QWORD *)(v5 + 144) = a1[9].m128i_i64[0];
  *(_BYTE *)(v5 + 352) = *(_BYTE *)(qword_4D03C50 + 16LL);
  return (__m128i *)v5;
}
