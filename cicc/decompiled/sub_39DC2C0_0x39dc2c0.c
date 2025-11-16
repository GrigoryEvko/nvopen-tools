// Function: sub_39DC2C0
// Address: 0x39dc2c0
//
void __fastcall sub_39DC2C0(__int64 *a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // r12
  _QWORD *v4; // r13
  _QWORD *v5; // r15
  __int64 v6; // r14
  __int64 v7; // rdx
  unsigned __int64 v8; // rax
  bool v9; // cf
  unsigned __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // r14
  char v13; // al
  __m128i v14; // xmm3
  __m128i v15; // xmm4
  _QWORD *v16; // r14
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rdx
  __int64 v23; // rax
  unsigned __int64 v24; // [rsp-50h] [rbp-50h]
  unsigned __int64 v25; // [rsp-50h] [rbp-50h]
  unsigned __int64 v26; // [rsp-48h] [rbp-48h]
  __int64 v27; // [rsp-40h] [rbp-40h]

  if ( !a2 )
    return;
  v2 = a2;
  v4 = (_QWORD *)a1[1];
  v5 = (_QWORD *)*a1;
  v6 = (__int64)v4 - *a1;
  v26 = 0xCCCCCCCCCCCCCCCDLL * (v6 >> 6);
  if ( 0xCCCCCCCCCCCCCCCDLL * ((a1[2] - (__int64)v4) >> 6) >= a2 )
  {
    v7 = a1[1];
    do
    {
      if ( v7 )
      {
        memset((void *)v7, 0, 0x140u);
        *(_BYTE *)(v7 + 152) = 1;
        *(_QWORD *)(v7 + 24) = v7 + 40;
        *(_QWORD *)(v7 + 104) = v7 + 120;
        *(_QWORD *)(v7 + 176) = v7 + 192;
        *(_QWORD *)(v7 + 224) = v7 + 240;
        *(_QWORD *)(v7 + 272) = v7 + 288;
      }
      v7 += 320;
      --a2;
    }
    while ( a2 );
    a1[1] = (__int64)&v4[40 * v2];
    return;
  }
  if ( 0x66666666666666LL - v26 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v8 = 0xCCCCCCCCCCCCCCCDLL * ((a1[1] - *a1) >> 6);
  if ( a2 >= v26 )
    v8 = a2;
  v9 = __CFADD__(v26, v8);
  v10 = v26 + v8;
  if ( v9 )
  {
    v22 = 0x7FFFFFFFFFFFFF80LL;
  }
  else
  {
    if ( !v10 )
    {
      v24 = 0;
      v27 = 0;
      goto LABEL_14;
    }
    if ( v10 > 0x66666666666666LL )
      v10 = 0x66666666666666LL;
    v22 = 320 * v10;
  }
  v25 = v22;
  v23 = sub_22077B0(v22);
  v4 = (_QWORD *)a1[1];
  v27 = v23;
  v5 = (_QWORD *)*a1;
  v24 = v23 + v25;
LABEL_14:
  v11 = v27 + v6;
  do
  {
    if ( v11 )
    {
      memset((void *)v11, 0, 0x140u);
      *(_BYTE *)(v11 + 152) = 1;
      *(_QWORD *)(v11 + 24) = v11 + 40;
      *(_QWORD *)(v11 + 104) = v11 + 120;
      *(_QWORD *)(v11 + 176) = v11 + 192;
      *(_QWORD *)(v11 + 224) = v11 + 240;
      *(_QWORD *)(v11 + 272) = v11 + 288;
    }
    v11 += 320;
    --a2;
  }
  while ( a2 );
  if ( v5 != v4 )
  {
    v12 = v27;
    do
    {
      if ( v12 )
      {
        *(__m128i *)v12 = _mm_loadu_si128((const __m128i *)v5);
        *(_QWORD *)(v12 + 16) = v5[2];
        *(_QWORD *)(v12 + 24) = v12 + 40;
        sub_39CF630((__int64 *)(v12 + 24), (_BYTE *)v5[3], v5[3] + v5[4]);
        *(__m128i *)(v12 + 56) = _mm_loadu_si128((const __m128i *)(v5 + 7));
        *(_DWORD *)(v12 + 72) = *((_DWORD *)v5 + 18);
        *(_QWORD *)(v12 + 80) = v5[10];
        *(_QWORD *)(v12 + 88) = v5[11];
        *(_DWORD *)(v12 + 96) = *((_DWORD *)v5 + 24);
        *(_BYTE *)(v12 + 100) = *((_BYTE *)v5 + 100);
        *(_QWORD *)(v12 + 104) = v12 + 120;
        sub_39CF630((__int64 *)(v12 + 104), (_BYTE *)v5[13], v5[13] + v5[14]);
        *(__m128i *)(v12 + 136) = _mm_loadu_si128((const __m128i *)(v5 + 17));
        *(_BYTE *)(v12 + 152) = *((_BYTE *)v5 + 152);
        v13 = *((_BYTE *)v5 + 168);
        *(_BYTE *)(v12 + 168) = v13;
        if ( v13 )
          *(_QWORD *)(v12 + 160) = v5[20];
        *(_QWORD *)(v12 + 176) = v12 + 192;
        sub_39CF630((__int64 *)(v12 + 176), (_BYTE *)v5[22], v5[22] + v5[23]);
        v14 = _mm_loadu_si128((const __m128i *)v5 + 13);
        *(_QWORD *)(v12 + 224) = v12 + 240;
        *(__m128i *)(v12 + 208) = v14;
        sub_39CF630((__int64 *)(v12 + 224), (_BYTE *)v5[28], v5[28] + v5[29]);
        v15 = _mm_loadu_si128((const __m128i *)v5 + 16);
        *(_QWORD *)(v12 + 272) = v12 + 288;
        *(__m128i *)(v12 + 256) = v15;
        sub_39CF630((__int64 *)(v12 + 272), (_BYTE *)v5[34], v5[34] + v5[35]);
        *(__m128i *)(v12 + 304) = _mm_loadu_si128((const __m128i *)v5 + 19);
      }
      v5 += 40;
      v12 += 320;
    }
    while ( v5 != v4 );
    v16 = (_QWORD *)a1[1];
    v4 = (_QWORD *)*a1;
    if ( v16 != (_QWORD *)*a1 )
    {
      do
      {
        v17 = v4[34];
        if ( (_QWORD *)v17 != v4 + 36 )
          j_j___libc_free_0(v17);
        v18 = v4[28];
        if ( (_QWORD *)v18 != v4 + 30 )
          j_j___libc_free_0(v18);
        v19 = v4[22];
        if ( (_QWORD *)v19 != v4 + 24 )
          j_j___libc_free_0(v19);
        v20 = v4[13];
        if ( (_QWORD *)v20 != v4 + 15 )
          j_j___libc_free_0(v20);
        v21 = v4[3];
        if ( (_QWORD *)v21 != v4 + 5 )
          j_j___libc_free_0(v21);
        v4 += 40;
      }
      while ( v16 != v4 );
      v4 = (_QWORD *)*a1;
    }
  }
  if ( v4 )
    j_j___libc_free_0((unsigned __int64)v4);
  *a1 = v27;
  a1[1] = v27 + 320 * (v26 + v2);
  a1[2] = v24;
}
