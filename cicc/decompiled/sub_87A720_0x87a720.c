// Function: sub_87A720
// Address: 0x87a720
//
__m128i *__fastcall sub_87A720(__int64 a1, __m128i *a2, __int64 *a3, __int64 a4)
{
  __m128i *result; // rax
  __int64 v5; // rdx
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // r15
  int v9; // r14d
  size_t v10; // rax
  __int64 v11; // rax

  result = xmmword_4F06660;
  *a2 = _mm_loadu_si128(xmmword_4F06660);
  a2[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
  a2[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
  a2[3] = _mm_loadu_si128(&xmmword_4F06660[3]);
  v5 = *a3;
  a2->m128i_i64[1] = v5;
  if ( (_BYTE)a1 )
  {
    v6 = qword_4D04A60[(unsigned __int8)a1];
    if ( !v6 )
    {
      v7 = sub_877070(a1, a2, v5, a4);
      qword_4D04A60[(unsigned __int8)a1] = v7;
      v6 = v7;
      v8 = qword_4F064C0[(unsigned __int8)a1];
      v9 = dword_4F05DC0[*(char *)(v8 + 1) + 128];
      v10 = (v9 != 0) + strlen((const char *)v8);
      *(_QWORD *)(v6 + 16) = v10 + 8;
      v11 = sub_7279A0(v10 + 9);
      *(_QWORD *)(v6 + 8) = v11;
      *(_QWORD *)v11 = 0x726F74617265706FLL;
      if ( v9 )
        *(_BYTE *)(v11 + 8) = 32;
      result = (__m128i *)strcpy((char *)(v11 + (v9 != 0) + 8), (const char *)v8);
      *(_BYTE *)(v6 + 72) = a1;
    }
    a2->m128i_i64[0] = v6;
    a2[3].m128i_i8[8] = a1;
    a2[1].m128i_i8[0] |= 8u;
  }
  else
  {
    *a2 = _mm_loadu_si128(xmmword_4F06660);
    a2[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
    a2[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
    result = *(__m128i **)dword_4F07508;
    a2[3] = _mm_loadu_si128(&xmmword_4F06660[3]);
    a2[1].m128i_i8[1] |= 0x20u;
    a2->m128i_i64[1] = (__int64)result;
  }
  return result;
}
