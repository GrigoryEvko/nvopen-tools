// Function: sub_87EBB0
// Address: 0x87ebb0
//
_QWORD *__fastcall sub_87EBB0(unsigned __int8 a1, __int64 a2, _QWORD *a3)
{
  __int64 v4; // rbx
  __int64 v5; // rax
  __m128i si128; // xmm1
  __m128i v7; // xmm2
  __m128i v8; // xmm3
  _QWORD *v9; // r12
  __m128i v10; // xmm4
  __m128i v11; // xmm5
  __m128i v12; // xmm6
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v16; // rbx
  __int64 v17; // rax

  v4 = a2;
  v5 = sub_823970(112);
  si128 = _mm_load_si128((const __m128i *)&xmmword_4F60050);
  v7 = _mm_load_si128((const __m128i *)&xmmword_4F60060);
  v8 = _mm_load_si128((const __m128i *)&xmmword_4F60070);
  v9 = (_QWORD *)v5;
  v10 = _mm_load_si128((const __m128i *)&xmmword_4F60080);
  v11 = _mm_load_si128((const __m128i *)&xmmword_4F60090);
  *(__m128i *)v5 = _mm_load_si128((const __m128i *)&xmmword_4F60040);
  v12 = _mm_load_si128((const __m128i *)&xmmword_4F600A0);
  *(__m128i *)(v5 + 16) = si128;
  *(__m128i *)(v5 + 32) = v7;
  *(__m128i *)(v5 + 48) = v8;
  *(__m128i *)(v5 + 64) = v10;
  *(__m128i *)(v5 + 80) = v11;
  *(__m128i *)(v5 + 96) = v12;
  if ( qword_4F04C68[0] && *(int *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 200) > 0 )
    *(_BYTE *)(v5 + 85) |= 1u;
  sub_87E690(v5, a1);
  if ( !a2 )
  {
    v4 = qword_4F600F0;
    if ( !qword_4F600F0 )
    {
      v16 = sub_877070(v9, a1, v13, v14);
      qword_4F600F0 = v16;
      v17 = sub_7279A0(8);
      *(_DWORD *)v17 = 1920099644;
      *(_WORD *)(v17 + 4) = 29295;
      *(_BYTE *)(v17 + 6) = 62;
      *(_BYTE *)(v17 + 7) = 0;
      *(_QWORD *)(v16 + 8) = v17;
      *(_BYTE *)(v16 + 73) &= ~1u;
      *(_QWORD *)(v16 + 16) = 7;
      v4 = qword_4F600F0;
    }
  }
  *v9 = v4;
  v9[6] = *a3;
  return v9;
}
