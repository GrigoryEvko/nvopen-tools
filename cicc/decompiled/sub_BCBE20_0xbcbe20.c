// Function: sub_BCBE20
// Address: 0xbcbe20
//
__int64 __fastcall sub_BCBE20(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  char v7; // al
  unsigned int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // r13
  unsigned int v12; // ebx
  __int64 v13; // rax
  __m128i v14; // xmm0
  __int64 v15; // rcx
  unsigned int v16; // eax
  __int64 v17; // rdx
  __int64 v18; // rax
  __m128i v19; // xmm0
  __m128i v20; // xmm0
  __int64 v21; // rdi
  unsigned int v22; // eax
  __int64 v23; // rdx
  __int64 v24; // rax
  __m128i si128; // xmm0
  __int64 v26; // rax
  __int64 v27; // [rsp+18h] [rbp-58h] BYREF
  _QWORD v28[2]; // [rsp+20h] [rbp-50h] BYREF
  _QWORD v29[8]; // [rsp+30h] [rbp-40h] BYREF

  v5 = *(_QWORD *)(a2 + 32);
  v6 = *(_QWORD *)(a2 + 24);
  switch ( v5 )
  {
    case 15LL:
      if ( *(_QWORD *)v6 == 0x2E34366863726161LL
        && *(_DWORD *)(v6 + 8) == 1868789363
        && *(_WORD *)(v6 + 12) == 28277
        && *(_BYTE *)(v6 + 14) == 116 )
      {
        v21 = *(unsigned int *)(a2 + 12);
        if ( (_DWORD)v21 || *(_DWORD *)(a2 + 8) >> 8 )
        {
          v22 = sub_C63BB0(v21, a2, v6, a4);
          v27 = 63;
          v11 = v23;
          v28[0] = v29;
          v12 = v22;
          v24 = sub_22409D0(v28, &v27, 0);
          v28[0] = v24;
          v29[0] = v27;
          *(__m128i *)v24 = _mm_load_si128((const __m128i *)&xmmword_3F568A0);
          si128 = _mm_load_si128((const __m128i *)&xmmword_3F568B0);
          qmemcpy((void *)(v24 + 48), "e no parameters", 15);
          *(__m128i *)(v24 + 16) = si128;
          *(__m128i *)(v24 + 32) = _mm_load_si128((const __m128i *)&xmmword_3F568C0);
          break;
        }
      }
LABEL_3:
      v7 = *(_BYTE *)(a1 + 8);
      *(_QWORD *)a1 = a2;
      *(_BYTE *)(a1 + 8) = v7 & 0xFC | 2;
      return a1;
    case 18LL:
      if ( *(_QWORD *)v6 ^ 0x65762E7663736972LL | *(_QWORD *)(v6 + 8) ^ 0x7075742E726F7463LL
        || *(_WORD *)(v6 + 16) != 25964
        || *(_DWORD *)(a2 + 12) == 1 && *(_DWORD *)(a2 + 8) >> 8 == 1 )
      {
        goto LABEL_3;
      }
      v9 = sub_C63BB0(a1, a2, v6, 0);
      v27 = 97;
      v11 = v10;
      v28[0] = v29;
      v12 = v9;
      v13 = sub_22409D0(v28, &v27, 0);
      v28[0] = v13;
      v29[0] = v27;
      *(__m128i *)v13 = _mm_load_si128((const __m128i *)&xmmword_3F568A0);
      v14 = _mm_load_si128((const __m128i *)&xmmword_3F568D0);
      *(_BYTE *)(v13 + 96) = 114;
      *(__m128i *)(v13 + 16) = v14;
      *(__m128i *)(v13 + 32) = _mm_load_si128((const __m128i *)&xmmword_3F568E0);
      *(__m128i *)(v13 + 48) = _mm_load_si128((const __m128i *)&xmmword_3F568F0);
      *(__m128i *)(v13 + 64) = _mm_load_si128((const __m128i *)&xmmword_3F56900);
      *(__m128i *)(v13 + 80) = _mm_load_si128((const __m128i *)&xmmword_3F56910);
      break;
    case 20LL:
      if ( *(_QWORD *)v6 ^ 0x6E2E6E6367646D61LL | *(_QWORD *)(v6 + 8) ^ 0x7261622E64656D61LL )
        goto LABEL_3;
      if ( *(_DWORD *)(v6 + 16) != 1919248754 )
        goto LABEL_3;
      v15 = *(unsigned int *)(a2 + 12);
      if ( !(_DWORD)v15 && *(_DWORD *)(a2 + 8) >> 8 == 1 )
        goto LABEL_3;
      v16 = sub_C63BB0(a1, a2, v6, v15);
      v27 = 99;
      v11 = v17;
      v28[0] = v29;
      v12 = v16;
      v18 = sub_22409D0(v28, &v27, 0);
      v28[0] = v18;
      v29[0] = v27;
      *(__m128i *)v18 = _mm_load_si128((const __m128i *)&xmmword_3F568A0);
      v19 = _mm_load_si128((const __m128i *)&xmmword_3F56920);
      *(_WORD *)(v18 + 96) = 25972;
      *(__m128i *)(v18 + 16) = v19;
      v20 = _mm_load_si128((const __m128i *)&xmmword_3F56930);
      *(_BYTE *)(v18 + 98) = 114;
      *(__m128i *)(v18 + 32) = v20;
      *(__m128i *)(v18 + 48) = _mm_load_si128((const __m128i *)&xmmword_3F56940);
      *(__m128i *)(v18 + 64) = _mm_load_si128((const __m128i *)&xmmword_3F56950);
      *(__m128i *)(v18 + 80) = _mm_load_si128((const __m128i *)&xmmword_3F56960);
      break;
    default:
      goto LABEL_3;
  }
  v28[1] = v27;
  *(_BYTE *)(v28[0] + v27) = 0;
  sub_C63F00(&v27, v28, v12, v11);
  if ( (_QWORD *)v28[0] != v29 )
    j_j___libc_free_0(v28[0], v29[0] + 1LL);
  v26 = v27;
  *(_BYTE *)(a1 + 8) |= 3u;
  *(_QWORD *)a1 = v26 & 0xFFFFFFFFFFFFFFFELL;
  return a1;
}
