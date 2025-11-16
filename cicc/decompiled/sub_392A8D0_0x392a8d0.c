// Function: sub_392A8D0
// Address: 0x392a8d0
//
__int64 __fastcall sub_392A8D0(__int64 a1, _QWORD *a2, unsigned __int8 a3)
{
  _BYTE *v7; // rdx
  _BYTE *v8; // rax
  _BYTE *v9; // rsi
  char v10; // dl
  _BYTE *v11; // rax
  _BYTE *v12; // rcx
  __int64 v13; // rax
  __int64 v15; // rax
  __m128i v16; // xmm0
  __m128i v17; // xmm0
  __m128i v18; // xmm0
  _BYTE *v19; // rdi
  _BYTE *i; // rdx
  __int64 v21; // rax
  __m128i si128; // xmm0
  __int64 v23; // rax
  __m128i v24; // xmm0
  unsigned __int64 v25; // [rsp+8h] [rbp-48h] BYREF
  unsigned __int64 v26[2]; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v27[6]; // [rsp+20h] [rbp-30h] BYREF

  v7 = (_BYTE *)a2[18];
  if ( *v7 != 46 )
  {
    if ( !a3 )
      goto LABEL_3;
LABEL_18:
    v25 = 84;
    v26[0] = (unsigned __int64)v27;
    v21 = sub_22409D0((__int64)v26, &v25, 0);
    v26[0] = v21;
    v27[0] = v25;
    *(__m128i *)v21 = _mm_load_si128((const __m128i *)&xmmword_3F900A0);
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F900B0);
    *(_DWORD *)(v21 + 80) = 1953064809;
    *(__m128i *)(v21 + 16) = si128;
    *(__m128i *)(v21 + 32) = _mm_load_si128((const __m128i *)&xmmword_3F900C0);
    *(__m128i *)(v21 + 48) = _mm_load_si128((const __m128i *)&xmmword_3F900D0);
    *(__m128i *)(v21 + 64) = _mm_load_si128((const __m128i *)&xmmword_3F900E0);
    goto LABEL_12;
  }
  v19 = v7 + 1;
  a2[18] = v7 + 1;
  for ( i = v7 + 1; (unsigned __int8)(*i - 48) <= 9u || (unsigned __int8)((*i & 0xDF) - 65) <= 5u; a2[18] = i )
    ++i;
  if ( (a3 & (v19 == i)) != 0 )
    goto LABEL_18;
LABEL_3:
  v8 = (_BYTE *)a2[18];
  if ( (*v8 & 0xDF) == 0x50 )
  {
    v9 = v8 + 1;
    a2[18] = v8 + 1;
    v10 = v8[1];
    if ( ((v10 - 43) & 0xFD) == 0 )
    {
      v9 = v8 + 2;
      a2[18] = v8 + 2;
      v10 = v8[2];
    }
    v11 = v9 + 1;
    if ( (unsigned __int8)(v10 - 48) <= 9u )
    {
      do
      {
        v12 = v11;
        a2[18] = v11++;
      }
      while ( (unsigned __int8)(*v12 - 48) <= 9u );
      if ( v12 != v9 )
      {
        v13 = a2[13];
        *(_DWORD *)a1 = 6;
        *(_DWORD *)(a1 + 32) = 64;
        *(_QWORD *)(a1 + 8) = v13;
        *(_QWORD *)(a1 + 16) = &v12[-v13];
        *(_QWORD *)(a1 + 24) = 0;
        return a1;
      }
    }
    v25 = 81;
    v26[0] = (unsigned __int64)v27;
    v23 = sub_22409D0((__int64)v26, &v25, 0);
    v26[0] = v23;
    v27[0] = v25;
    *(__m128i *)v23 = _mm_load_si128((const __m128i *)&xmmword_3F900A0);
    v24 = _mm_load_si128((const __m128i *)&xmmword_3F900B0);
    *(_BYTE *)(v23 + 80) = 116;
    *(__m128i *)(v23 + 16) = v24;
    *(__m128i *)(v23 + 32) = _mm_load_si128((const __m128i *)&xmmword_3F900C0);
    *(__m128i *)(v23 + 48) = _mm_load_si128((const __m128i *)&xmmword_3F900D0);
    *(__m128i *)(v23 + 64) = _mm_load_si128((const __m128i *)&xmmword_3F90100);
  }
  else
  {
    v25 = 71;
    v26[0] = (unsigned __int64)v27;
    v15 = sub_22409D0((__int64)v26, &v25, 0);
    v26[0] = v15;
    v27[0] = v25;
    *(__m128i *)v15 = _mm_load_si128((const __m128i *)&xmmword_3F900A0);
    v16 = _mm_load_si128((const __m128i *)&xmmword_3F900B0);
    *(_DWORD *)(v15 + 64) = 544502369;
    *(__m128i *)(v15 + 16) = v16;
    v17 = _mm_load_si128((const __m128i *)&xmmword_3F900C0);
    *(_WORD *)(v15 + 68) = 28711;
    *(__m128i *)(v15 + 32) = v17;
    v18 = _mm_load_si128((const __m128i *)&xmmword_3F900F0);
    *(_BYTE *)(v15 + 70) = 39;
    *(__m128i *)(v15 + 48) = v18;
  }
LABEL_12:
  v26[1] = v25;
  *(_BYTE *)(v26[0] + v25) = 0;
  sub_392A760(a1, a2, a2[13], v26);
  if ( (_QWORD *)v26[0] == v27 )
    return a1;
  j_j___libc_free_0(v26[0]);
  return a1;
}
