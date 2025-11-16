// Function: sub_2F1CAB0
// Address: 0x2f1cab0
//
void __fastcall sub_2F1CAB0(unsigned __int64 *a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rbx
  __int64 v3; // r12
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // rax
  bool v6; // cf
  unsigned __int64 v7; // rax
  __int64 v8; // r12
  unsigned __int64 v9; // r14
  unsigned __int64 v10; // r13
  __int64 v11; // r15
  unsigned __int64 v12; // r12
  unsigned __int64 v13; // rbx
  __m128i v14; // xmm1
  __int64 v15; // rsi
  __m128i v16; // xmm2
  __m128i v17; // xmm3
  __int64 v18; // rsi
  __m128i v19; // xmm4
  __int64 v20; // rsi
  __m128i v21; // xmm5
  __int64 v22; // rsi
  __m128i v23; // xmm6
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rcx
  __int64 v30; // rsi
  unsigned __int64 v31; // r13
  unsigned __int64 v32; // [rsp+0h] [rbp-60h]
  unsigned __int64 v33; // [rsp+8h] [rbp-58h]
  __int64 v34; // [rsp+10h] [rbp-50h]
  unsigned __int64 v36; // [rsp+20h] [rbp-40h]
  unsigned __int64 v37; // [rsp+28h] [rbp-38h]

  v36 = a2;
  if ( !a2 )
    return;
  v2 = *a1;
  v37 = a1[1];
  v3 = v37 - *a1;
  v33 = 0xCCCCCCCCCCCCCCCDLL * (v3 >> 6);
  if ( a2 <= 0xCCCCCCCCCCCCCCCDLL * ((__int64)(a1[2] - v37) >> 6) )
  {
    v4 = a1[1];
    do
    {
      if ( v4 )
      {
        memset((void *)v4, 0, 0x140u);
        *(_BYTE *)(v4 + 152) = 1;
        *(_QWORD *)(v4 + 24) = v4 + 40;
        *(_QWORD *)(v4 + 104) = v4 + 120;
        *(_QWORD *)(v4 + 176) = v4 + 192;
        *(_QWORD *)(v4 + 224) = v4 + 240;
        *(_QWORD *)(v4 + 272) = v4 + 288;
      }
      v4 += 320LL;
      --a2;
    }
    while ( a2 );
    a1[1] = 320 * v36 + v37;
    return;
  }
  if ( 0x66666666666666LL - v33 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v5 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v37 - *a1) >> 6);
  if ( a2 >= v33 )
    v5 = a2;
  v6 = __CFADD__(v33, v5);
  v7 = v33 + v5;
  if ( v6 )
  {
    v31 = 0x7FFFFFFFFFFFFF80LL;
  }
  else
  {
    if ( !v7 )
    {
      v32 = 0;
      v34 = 0;
      goto LABEL_15;
    }
    if ( v7 > 0x66666666666666LL )
      v7 = 0x66666666666666LL;
    v31 = 320 * v7;
  }
  v34 = sub_22077B0(v31);
  v2 = *a1;
  v32 = v34 + v31;
  v37 = a1[1];
LABEL_15:
  v8 = v34 + v3;
  do
  {
    if ( v8 )
    {
      memset((void *)v8, 0, 0x140u);
      *(_BYTE *)(v8 + 152) = 1;
      *(_QWORD *)(v8 + 24) = v8 + 40;
      *(_QWORD *)(v8 + 104) = v8 + 120;
      *(_QWORD *)(v8 + 176) = v8 + 192;
      *(_QWORD *)(v8 + 224) = v8 + 240;
      *(_QWORD *)(v8 + 272) = v8 + 288;
    }
    v8 += 320;
    --a2;
  }
  while ( a2 );
  if ( v2 != v37 )
  {
    v9 = v2 + 40;
    v10 = v2 + 120;
    v11 = v34;
    v12 = v2 + 192;
    v13 = v2 + 240;
    while ( 1 )
    {
      v29 = v9 + 248;
      if ( v11 )
      {
        *(__m128i *)v11 = _mm_loadu_si128((const __m128i *)(v9 - 40));
        *(_QWORD *)(v11 + 16) = *(_QWORD *)(v9 - 24);
        *(_QWORD *)(v11 + 24) = v11 + 40;
        v30 = *(_QWORD *)(v9 - 16);
        if ( v30 == v9 )
        {
          *(__m128i *)(v11 + 40) = _mm_loadu_si128((const __m128i *)v9);
        }
        else
        {
          *(_QWORD *)(v11 + 24) = v30;
          *(_QWORD *)(v11 + 40) = *(_QWORD *)v9;
        }
        *(_QWORD *)(v11 + 32) = *(_QWORD *)(v9 - 8);
        v14 = _mm_loadu_si128((const __m128i *)(v9 + 16));
        *(_QWORD *)(v9 - 16) = v9;
        *(_QWORD *)(v9 - 8) = 0;
        *(_BYTE *)v9 = 0;
        *(__m128i *)(v11 + 56) = v14;
        *(_DWORD *)(v11 + 72) = *(_DWORD *)(v9 + 32);
        *(_QWORD *)(v11 + 80) = *(_QWORD *)(v9 + 40);
        *(_QWORD *)(v11 + 88) = *(_QWORD *)(v9 + 48);
        *(_WORD *)(v11 + 96) = *(_WORD *)(v9 + 56);
        *(_DWORD *)(v11 + 100) = *(_DWORD *)(v9 + 60);
        *(_QWORD *)(v11 + 104) = v11 + 120;
        v15 = *(_QWORD *)(v9 + 64);
        if ( v15 == v10 )
        {
          *(__m128i *)(v11 + 120) = _mm_loadu_si128((const __m128i *)(v9 + 80));
        }
        else
        {
          *(_QWORD *)(v11 + 104) = v15;
          *(_QWORD *)(v11 + 120) = *(_QWORD *)(v9 + 80);
        }
        *(_QWORD *)(v11 + 112) = *(_QWORD *)(v9 + 72);
        v16 = _mm_loadu_si128((const __m128i *)(v9 + 96));
        *(_QWORD *)(v9 + 64) = v10;
        *(_QWORD *)(v9 + 72) = 0;
        *(_BYTE *)(v9 + 80) = 0;
        *(__m128i *)(v11 + 136) = v16;
        *(_BYTE *)(v11 + 152) = *(_BYTE *)(v9 + 112);
        v17 = _mm_loadu_si128((const __m128i *)(v9 + 120));
        *(_QWORD *)(v11 + 176) = v11 + 192;
        *(__m128i *)(v11 + 160) = v17;
        v18 = *(_QWORD *)(v9 + 136);
        if ( v18 == v12 )
        {
          *(__m128i *)(v11 + 192) = _mm_loadu_si128((const __m128i *)(v9 + 152));
        }
        else
        {
          *(_QWORD *)(v11 + 176) = v18;
          *(_QWORD *)(v11 + 192) = *(_QWORD *)(v9 + 152);
        }
        *(_QWORD *)(v11 + 184) = *(_QWORD *)(v9 + 144);
        v19 = _mm_loadu_si128((const __m128i *)(v9 + 168));
        *(_QWORD *)(v9 + 136) = v12;
        *(_QWORD *)(v9 + 144) = 0;
        *(_BYTE *)(v9 + 152) = 0;
        *(_QWORD *)(v11 + 224) = v11 + 240;
        *(__m128i *)(v11 + 208) = v19;
        v20 = *(_QWORD *)(v9 + 184);
        if ( v20 == v13 )
        {
          *(__m128i *)(v11 + 240) = _mm_loadu_si128((const __m128i *)(v9 + 200));
        }
        else
        {
          *(_QWORD *)(v11 + 224) = v20;
          *(_QWORD *)(v11 + 240) = *(_QWORD *)(v9 + 200);
        }
        *(_QWORD *)(v11 + 232) = *(_QWORD *)(v9 + 192);
        v21 = _mm_loadu_si128((const __m128i *)(v9 + 216));
        *(_QWORD *)(v9 + 184) = v13;
        *(_QWORD *)(v9 + 192) = 0;
        *(_BYTE *)(v9 + 200) = 0;
        *(_QWORD *)(v11 + 272) = v11 + 288;
        *(__m128i *)(v11 + 256) = v21;
        v22 = *(_QWORD *)(v9 + 232);
        if ( v22 == v29 )
        {
          *(__m128i *)(v11 + 288) = _mm_loadu_si128((const __m128i *)(v9 + 248));
        }
        else
        {
          *(_QWORD *)(v11 + 272) = v22;
          *(_QWORD *)(v11 + 288) = *(_QWORD *)(v9 + 248);
        }
        *(_QWORD *)(v11 + 280) = *(_QWORD *)(v9 + 240);
        v23 = _mm_loadu_si128((const __m128i *)(v9 + 264));
        *(_QWORD *)(v9 + 232) = v29;
        *(_QWORD *)(v9 + 240) = 0;
        *(_BYTE *)(v9 + 248) = 0;
        *(__m128i *)(v11 + 304) = v23;
      }
      v24 = *(_QWORD *)(v9 + 232);
      if ( v24 != v29 )
        j_j___libc_free_0(v24);
      v25 = *(_QWORD *)(v9 + 184);
      if ( v25 != v13 )
        j_j___libc_free_0(v25);
      v26 = *(_QWORD *)(v9 + 136);
      if ( v26 != v12 )
        j_j___libc_free_0(v26);
      v27 = *(_QWORD *)(v9 + 64);
      if ( v27 != v10 )
        j_j___libc_free_0(v27);
      v28 = *(_QWORD *)(v9 - 16);
      if ( v28 != v9 )
        j_j___libc_free_0(v28);
      v11 += 320;
      v10 += 320LL;
      v12 += 320LL;
      v13 += 320LL;
      if ( v37 == v9 + 280 )
        break;
      v9 += 320LL;
    }
    v37 = *a1;
  }
  if ( v37 )
    j_j___libc_free_0(v37);
  *a1 = v34;
  a1[1] = v34 + 320 * (v33 + v36);
  a1[2] = v32;
}
