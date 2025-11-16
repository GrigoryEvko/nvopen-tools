// Function: sub_39D6B60
// Address: 0x39d6b60
//
unsigned __int64 *__fastcall sub_39D6B60(unsigned __int64 *a1, _QWORD *a2, const __m128i *a3)
{
  _QWORD *v4; // r14
  _QWORD *v6; // rbx
  unsigned __int64 v7; // rcx
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rdx
  bool v10; // cf
  unsigned __int64 v11; // rax
  __int64 v12; // r8
  __int64 v13; // r15
  bool v14; // zf
  __m128i *v15; // r8
  __m128i *v16; // r13
  _BYTE *v17; // rsi
  __m128i v18; // xmm7
  __int64 v19; // rdx
  __m128i v20; // xmm7
  _BYTE *v21; // rsi
  __int64 v22; // rdx
  __m128i v23; // xmm7
  __int8 v24; // dl
  _BYTE *v25; // rsi
  _BYTE *v26; // rsi
  __int64 v27; // rdx
  _BYTE *v28; // rsi
  __int64 v29; // rdx
  unsigned __int64 v30; // r13
  __int64 i; // r15
  char v32; // dl
  __m128i v33; // xmm6
  __m128i v34; // xmm7
  __m128i v35; // xmm4
  _BYTE *v36; // rsi
  __int64 v37; // rdx
  __m128i v38; // xmm5
  _BYTE *v39; // rsi
  __int64 v40; // rdx
  __m128i v41; // xmm6
  char v42; // dl
  _BYTE *v43; // rsi
  _BYTE *v44; // rsi
  __m128i v45; // xmm0
  __int64 v46; // rdx
  _BYTE *v47; // rsi
  __m128i v48; // xmm1
  __int64 v49; // rdx
  __int64 *v50; // rdi
  _QWORD *j; // r12
  unsigned __int64 v52; // rdi
  unsigned __int64 v53; // rdi
  unsigned __int64 v54; // rdi
  unsigned __int64 v55; // rdi
  unsigned __int64 v56; // rdi
  unsigned __int64 v58; // r15
  __int64 v59; // rax
  const __m128i *v60; // [rsp+0h] [rbp-60h]
  const __m128i *v61; // [rsp+8h] [rbp-58h]
  unsigned __int64 v62; // [rsp+10h] [rbp-50h]
  __int64 v64; // [rsp+20h] [rbp-40h]
  unsigned __int64 v65; // [rsp+28h] [rbp-38h]

  v4 = a2;
  v6 = (_QWORD *)a1[1];
  v7 = *a1;
  v65 = *a1;
  v8 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)((__int64)v6 - *a1) >> 6);
  if ( v8 == 0x66666666666666LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v9 = 1;
  if ( v8 )
    v9 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)((__int64)v6 - v7) >> 6);
  v10 = __CFADD__(v9, v8);
  v11 = v9 - 0x3333333333333333LL * ((__int64)((__int64)v6 - v7) >> 6);
  v12 = (__int64)a2 - v65;
  if ( v10 )
  {
    v58 = 0x7FFFFFFFFFFFFF80LL;
  }
  else
  {
    if ( !v11 )
    {
      v62 = 0;
      v13 = 320;
      v64 = 0;
      goto LABEL_7;
    }
    if ( v11 > 0x66666666666666LL )
      v11 = 0x66666666666666LL;
    v58 = 320 * v11;
  }
  v60 = a3;
  v59 = sub_22077B0(v58);
  v12 = (__int64)a2 - v65;
  a3 = v60;
  v64 = v59;
  v62 = v59 + v58;
  v13 = v59 + 320;
LABEL_7:
  v14 = v64 + v12 == 0;
  v15 = (__m128i *)(v64 + v12);
  v16 = v15;
  if ( !v14 )
  {
    v17 = (_BYTE *)a3[1].m128i_i64[1];
    v61 = a3;
    v18 = _mm_loadu_si128(a3);
    v15[1].m128i_i64[0] = a3[1].m128i_i64[0];
    v15[1].m128i_i64[1] = (__int64)&v15[2].m128i_i64[1];
    v19 = a3[2].m128i_i64[0];
    *v15 = v18;
    sub_39CF630(&v15[1].m128i_i64[1], v17, (__int64)&v17[v19]);
    v20 = _mm_loadu_si128((const __m128i *)((char *)v61 + 56));
    v21 = (_BYTE *)v61[6].m128i_i64[1];
    v16[4].m128i_i32[2] = v61[4].m128i_i32[2];
    v22 = v61[5].m128i_i64[0];
    *(__m128i *)((char *)v16 + 56) = v20;
    v16[5].m128i_i64[0] = v22;
    v16[5].m128i_i64[1] = v61[5].m128i_i64[1];
    v16[6].m128i_i32[0] = v61[6].m128i_i32[0];
    v16[6].m128i_i8[4] = v61[6].m128i_i8[4];
    v16[6].m128i_i64[1] = (__int64)&v16[7].m128i_i64[1];
    sub_39CF630(&v16[6].m128i_i64[1], v21, (__int64)&v21[v61[7].m128i_i64[0]]);
    v23 = _mm_loadu_si128((const __m128i *)((char *)v61 + 136));
    v16[9].m128i_i8[8] = v61[9].m128i_i8[8];
    v24 = v61[10].m128i_i8[8];
    *(__m128i *)((char *)v16 + 136) = v23;
    v16[10].m128i_i8[8] = v24;
    if ( v24 )
      v16[10].m128i_i64[0] = v61[10].m128i_i64[0];
    v25 = (_BYTE *)v61[11].m128i_i64[0];
    v16[11].m128i_i64[0] = (__int64)v16[12].m128i_i64;
    sub_39CF630(v16[11].m128i_i64, v25, (__int64)&v25[v61[11].m128i_i64[1]]);
    v16[14].m128i_i64[0] = (__int64)v16[15].m128i_i64;
    v26 = (_BYTE *)v61[14].m128i_i64[0];
    v27 = v61[14].m128i_i64[1];
    v16[13] = _mm_loadu_si128(v61 + 13);
    sub_39CF630(v16[14].m128i_i64, v26, (__int64)&v26[v27]);
    v16[17].m128i_i64[0] = (__int64)v16[18].m128i_i64;
    v28 = (_BYTE *)v61[17].m128i_i64[0];
    v29 = v61[17].m128i_i64[1];
    v16[16] = _mm_loadu_si128(v61 + 16);
    sub_39CF630(v16[17].m128i_i64, v28, (__int64)&v28[v29]);
    v16[19] = _mm_loadu_si128(v61 + 19);
  }
  v30 = v65;
  if ( a2 != (_QWORD *)v65 )
  {
    for ( i = v64; ; i += 320 )
    {
      if ( i )
      {
        *(__m128i *)i = _mm_loadu_si128((const __m128i *)v30);
        *(_QWORD *)(i + 16) = *(_QWORD *)(v30 + 16);
        *(_QWORD *)(i + 24) = i + 40;
        sub_39CF630((__int64 *)(i + 24), *(_BYTE **)(v30 + 24), *(_QWORD *)(v30 + 24) + *(_QWORD *)(v30 + 32));
        *(__m128i *)(i + 56) = _mm_loadu_si128((const __m128i *)(v30 + 56));
        *(_DWORD *)(i + 72) = *(_DWORD *)(v30 + 72);
        *(_QWORD *)(i + 80) = *(_QWORD *)(v30 + 80);
        *(_QWORD *)(i + 88) = *(_QWORD *)(v30 + 88);
        *(_DWORD *)(i + 96) = *(_DWORD *)(v30 + 96);
        *(_BYTE *)(i + 100) = *(_BYTE *)(v30 + 100);
        *(_QWORD *)(i + 104) = i + 120;
        sub_39CF630((__int64 *)(i + 104), *(_BYTE **)(v30 + 104), *(_QWORD *)(v30 + 104) + *(_QWORD *)(v30 + 112));
        *(__m128i *)(i + 136) = _mm_loadu_si128((const __m128i *)(v30 + 136));
        *(_BYTE *)(i + 152) = *(_BYTE *)(v30 + 152);
        v32 = *(_BYTE *)(v30 + 168);
        *(_BYTE *)(i + 168) = v32;
        if ( v32 )
          *(_QWORD *)(i + 160) = *(_QWORD *)(v30 + 160);
        *(_QWORD *)(i + 176) = i + 192;
        sub_39CF630((__int64 *)(i + 176), *(_BYTE **)(v30 + 176), *(_QWORD *)(v30 + 176) + *(_QWORD *)(v30 + 184));
        v33 = _mm_loadu_si128((const __m128i *)(v30 + 208));
        *(_QWORD *)(i + 224) = i + 240;
        *(__m128i *)(i + 208) = v33;
        sub_39CF630((__int64 *)(i + 224), *(_BYTE **)(v30 + 224), *(_QWORD *)(v30 + 224) + *(_QWORD *)(v30 + 232));
        v34 = _mm_loadu_si128((const __m128i *)(v30 + 256));
        *(_QWORD *)(i + 272) = i + 288;
        *(__m128i *)(i + 256) = v34;
        sub_39CF630((__int64 *)(i + 272), *(_BYTE **)(v30 + 272), *(_QWORD *)(v30 + 272) + *(_QWORD *)(v30 + 280));
        *(__m128i *)(i + 304) = _mm_loadu_si128((const __m128i *)(v30 + 304));
      }
      v30 += 320LL;
      if ( a2 == (_QWORD *)v30 )
        break;
    }
    v13 = i + 640;
  }
  if ( a2 != v6 )
  {
    do
    {
      v35 = _mm_loadu_si128((const __m128i *)v4);
      v36 = (_BYTE *)v4[3];
      *(_QWORD *)(v13 + 16) = v4[2];
      *(_QWORD *)(v13 + 24) = v13 + 40;
      v37 = v4[4];
      *(__m128i *)v13 = v35;
      sub_39CF630((__int64 *)(v13 + 24), v36, (__int64)&v36[v37]);
      v38 = _mm_loadu_si128((const __m128i *)(v4 + 7));
      v39 = (_BYTE *)v4[13];
      *(_DWORD *)(v13 + 72) = *((_DWORD *)v4 + 18);
      v40 = v4[10];
      *(__m128i *)(v13 + 56) = v38;
      *(_QWORD *)(v13 + 80) = v40;
      *(_QWORD *)(v13 + 88) = v4[11];
      *(_DWORD *)(v13 + 96) = *((_DWORD *)v4 + 24);
      *(_BYTE *)(v13 + 100) = *((_BYTE *)v4 + 100);
      *(_QWORD *)(v13 + 104) = v13 + 120;
      sub_39CF630((__int64 *)(v13 + 104), v39, (__int64)&v39[v4[14]]);
      v41 = _mm_loadu_si128((const __m128i *)(v4 + 17));
      *(_BYTE *)(v13 + 152) = *((_BYTE *)v4 + 152);
      v42 = *((_BYTE *)v4 + 168);
      *(__m128i *)(v13 + 136) = v41;
      *(_BYTE *)(v13 + 168) = v42;
      if ( v42 )
        *(_QWORD *)(v13 + 160) = v4[20];
      v43 = (_BYTE *)v4[22];
      v4 += 40;
      *(_QWORD *)(v13 + 176) = v13 + 192;
      sub_39CF630((__int64 *)(v13 + 176), v43, (__int64)&v43[*(v4 - 17)]);
      v44 = (_BYTE *)*(v4 - 12);
      v45 = _mm_loadu_si128((const __m128i *)v4 - 7);
      *(_QWORD *)(v13 + 224) = v13 + 240;
      v46 = *(v4 - 11);
      *(__m128i *)(v13 + 208) = v45;
      sub_39CF630((__int64 *)(v13 + 224), v44, (__int64)&v44[v46]);
      v47 = (_BYTE *)*(v4 - 6);
      v48 = _mm_loadu_si128((const __m128i *)v4 - 4);
      *(_QWORD *)(v13 + 272) = v13 + 288;
      v49 = *(v4 - 5);
      v50 = (__int64 *)(v13 + 272);
      v13 += 320;
      *(__m128i *)(v13 - 64) = v48;
      sub_39CF630(v50, v47, (__int64)&v47[v49]);
      *(__m128i *)(v13 - 16) = _mm_loadu_si128((const __m128i *)v4 - 1);
    }
    while ( v6 != v4 );
  }
  for ( j = (_QWORD *)v65; j != v6; j += 40 )
  {
    v52 = j[34];
    if ( (_QWORD *)v52 != j + 36 )
      j_j___libc_free_0(v52);
    v53 = j[28];
    if ( (_QWORD *)v53 != j + 30 )
      j_j___libc_free_0(v53);
    v54 = j[22];
    if ( (_QWORD *)v54 != j + 24 )
      j_j___libc_free_0(v54);
    v55 = j[13];
    if ( (_QWORD *)v55 != j + 15 )
      j_j___libc_free_0(v55);
    v56 = j[3];
    if ( (_QWORD *)v56 != j + 5 )
      j_j___libc_free_0(v56);
  }
  if ( v65 )
    j_j___libc_free_0(v65);
  *a1 = v64;
  a1[1] = v13;
  a1[2] = v62;
  return a1;
}
