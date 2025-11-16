// Function: sub_1095E80
// Address: 0x1095e80
//
__int64 __fastcall sub_1095E80(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  _BYTE *v4; // rax
  _BYTE *v5; // rax
  _BYTE *v6; // rsi
  char v7; // dl
  _BYTE *v8; // rax
  _BYTE *v9; // rcx
  __int64 v10; // rax
  __int64 v12; // rax
  __m128i v13; // xmm0
  __m128i v14; // xmm0
  __m128i v15; // xmm0
  unsigned __int8 *v16; // r8
  unsigned __int8 *v18; // rax
  unsigned __int8 *v19; // rdx
  __int64 v20; // rax
  __m128i si128; // xmm0
  __int64 v22; // rax
  __m128i v23; // xmm0
  __int64 v24; // [rsp+8h] [rbp-48h] BYREF
  _QWORD v25[2]; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v26[6]; // [rsp+20h] [rbp-30h] BYREF

  v4 = *(_BYTE **)(a2 + 152);
  if ( *v4 != 46 || (v16 = v4 + 1, *(_QWORD *)(a2 + 152) = v4 + 1, word_3F64060[(unsigned __int8)v4[1]] == 0xFFFF) )
  {
    if ( a3 )
    {
LABEL_18:
      v24 = 84;
      v25[0] = v26;
      v20 = sub_22409D0(v25, &v24, 0);
      v25[0] = v20;
      v26[0] = v24;
      *(__m128i *)v20 = _mm_load_si128((const __m128i *)&xmmword_3F900A0);
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F900B0);
      *(_DWORD *)(v20 + 80) = 1953064809;
      *(__m128i *)(v20 + 16) = si128;
      *(__m128i *)(v20 + 32) = _mm_load_si128((const __m128i *)&xmmword_3F900C0);
      *(__m128i *)(v20 + 48) = _mm_load_si128((const __m128i *)&xmmword_3F900D0);
      *(__m128i *)(v20 + 64) = _mm_load_si128((const __m128i *)&xmmword_3F900E0);
      goto LABEL_12;
    }
  }
  else
  {
    v18 = v4 + 2;
    do
    {
      v19 = v18;
      *(_QWORD *)(a2 + 152) = v18++;
    }
    while ( word_3F64060[*v19] != 0xFFFF );
    if ( (a3 & (v16 == v19)) != 0 )
      goto LABEL_18;
  }
  v5 = *(_BYTE **)(a2 + 152);
  if ( (*v5 & 0xDF) == 0x50 )
  {
    v6 = v5 + 1;
    *(_QWORD *)(a2 + 152) = v5 + 1;
    v7 = v5[1];
    if ( ((v7 - 43) & 0xFD) == 0 )
    {
      v6 = v5 + 2;
      *(_QWORD *)(a2 + 152) = v5 + 2;
      v7 = v5[2];
    }
    v8 = v6 + 1;
    if ( (unsigned __int8)(v7 - 48) <= 9u )
    {
      do
      {
        v9 = v8;
        *(_QWORD *)(a2 + 152) = v8++;
      }
      while ( (unsigned __int8)(*v9 - 48) <= 9u );
      if ( v9 != v6 )
      {
        v10 = *(_QWORD *)(a2 + 104);
        *(_DWORD *)a1 = 6;
        *(_DWORD *)(a1 + 32) = 64;
        *(_QWORD *)(a1 + 8) = v10;
        *(_QWORD *)(a1 + 16) = &v9[-v10];
        *(_QWORD *)(a1 + 24) = 0;
        return a1;
      }
    }
    v24 = 81;
    v25[0] = v26;
    v22 = sub_22409D0(v25, &v24, 0);
    v25[0] = v22;
    v26[0] = v24;
    *(__m128i *)v22 = _mm_load_si128((const __m128i *)&xmmword_3F900A0);
    v23 = _mm_load_si128((const __m128i *)&xmmword_3F900B0);
    *(_BYTE *)(v22 + 80) = 116;
    *(__m128i *)(v22 + 16) = v23;
    *(__m128i *)(v22 + 32) = _mm_load_si128((const __m128i *)&xmmword_3F900C0);
    *(__m128i *)(v22 + 48) = _mm_load_si128((const __m128i *)&xmmword_3F900D0);
    *(__m128i *)(v22 + 64) = _mm_load_si128((const __m128i *)&xmmword_3F90100);
  }
  else
  {
    v24 = 71;
    v25[0] = v26;
    v12 = sub_22409D0(v25, &v24, 0);
    v25[0] = v12;
    v26[0] = v24;
    *(__m128i *)v12 = _mm_load_si128((const __m128i *)&xmmword_3F900A0);
    v13 = _mm_load_si128((const __m128i *)&xmmword_3F900B0);
    *(_DWORD *)(v12 + 64) = 544502369;
    *(__m128i *)(v12 + 16) = v13;
    v14 = _mm_load_si128((const __m128i *)&xmmword_3F900C0);
    *(_WORD *)(v12 + 68) = 28711;
    *(__m128i *)(v12 + 32) = v14;
    v15 = _mm_load_si128((const __m128i *)&xmmword_3F900F0);
    *(_BYTE *)(v12 + 70) = 39;
    *(__m128i *)(v12 + 48) = v15;
  }
LABEL_12:
  v25[1] = v24;
  *(_BYTE *)(v25[0] + v24) = 0;
  sub_1095C00(a1, a2, *(_QWORD *)(a2 + 104), (__int64)v25);
  if ( (_QWORD *)v25[0] == v26 )
    return a1;
  j_j___libc_free_0(v25[0], v26[0] + 1LL);
  return a1;
}
