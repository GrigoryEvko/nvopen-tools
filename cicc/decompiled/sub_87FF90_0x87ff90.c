// Function: sub_87FF90
// Address: 0x87ff90
//
_QWORD *__fastcall sub_87FF90(__int64 a1, _QWORD *a2)
{
  __int64 v2; // r12
  __int64 v3; // rbx
  __m128i si128; // xmm1
  __m128i v5; // xmm2
  __m128i v6; // xmm3
  __m128i v7; // xmm4
  __m128i v8; // xmm5
  __m128i v9; // xmm6
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rcx
  _QWORD *result; // rax
  __int64 v14; // r12
  __int64 v15; // rax

  v2 = a1;
  v3 = qword_4F5FED0;
  if ( qword_4F5FED0 )
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_4F60050);
    v5 = _mm_load_si128((const __m128i *)&xmmword_4F60060);
    v6 = _mm_load_si128((const __m128i *)&xmmword_4F60070);
    v7 = _mm_load_si128((const __m128i *)&xmmword_4F60080);
    *(__m128i *)qword_4F5FED0 = _mm_load_si128((const __m128i *)&xmmword_4F60040);
    v8 = _mm_load_si128((const __m128i *)&xmmword_4F60090);
    v9 = _mm_load_si128((const __m128i *)&xmmword_4F600A0);
    *(__m128i *)(v3 + 16) = si128;
    v10 = qword_4F04C68[0];
    *(__m128i *)(v3 + 32) = v5;
    *(__m128i *)(v3 + 48) = v6;
    *(__m128i *)(v3 + 64) = v7;
    *(__m128i *)(v3 + 80) = v8;
    *(__m128i *)(v3 + 96) = v9;
    if ( v10 && *(int *)(v10 + 776LL * dword_4F04C64 + 200) > 0 )
      *(_BYTE *)(v3 + 85) |= 1u;
    sub_87E690(v3, 13);
    if ( !a1 )
    {
      v2 = qword_4F600F0;
      if ( !qword_4F600F0 )
      {
        v14 = sub_877070(v3, 13, v11, v12);
        qword_4F600F0 = v14;
        v15 = sub_7279A0(8);
        *(_DWORD *)v15 = 1920099644;
        *(_WORD *)(v15 + 4) = 29295;
        *(_BYTE *)(v15 + 6) = 62;
        *(_BYTE *)(v15 + 7) = 0;
        *(_QWORD *)(v14 + 8) = v15;
        *(_BYTE *)(v14 + 73) &= ~1u;
        *(_QWORD *)(v14 + 16) = 7;
        v2 = qword_4F600F0;
      }
    }
    *(_QWORD *)v3 = v2;
    *(_QWORD *)(v3 + 48) = *a2;
    return (_QWORD *)qword_4F5FED0;
  }
  else
  {
    result = sub_87EBB0(0xDu, a1, a2);
    qword_4F5FED0 = (__int64)result;
  }
  return result;
}
