// Function: sub_2F09880
// Address: 0x2f09880
//
void __fastcall sub_2F09880(unsigned __int64 *a1, unsigned __int64 *a2)
{
  unsigned __int64 v2; // rcx
  unsigned __int64 v4; // rbx
  unsigned __int64 v5; // r15
  unsigned __int64 v6; // rsi
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // r12
  __int64 v9; // rdx
  unsigned __int64 v10; // rsi
  unsigned __int64 *v11; // r14
  unsigned __int64 *v12; // rbx
  unsigned __int64 v13; // r13
  __m128i v14; // xmm0
  unsigned __int64 *v15; // rsi
  unsigned __int64 *v16; // rdi
  unsigned __int64 v17; // rdi
  __int64 v18; // r15
  __int64 v19; // rax
  __int64 v20; // r15
  __int64 i; // r14
  unsigned __int64 v22; // rbx
  unsigned __int64 v23; // r14
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // r13
  unsigned __int64 *v26; // r15
  unsigned __int64 *v27; // rbx
  __m128i v28; // xmm2
  unsigned __int64 *v29; // rsi
  unsigned __int64 *v30; // rdi
  unsigned __int64 v31; // rbx
  unsigned __int64 v32; // [rsp+8h] [rbp-48h]
  unsigned __int64 v33; // [rsp+8h] [rbp-48h]
  unsigned __int64 v34; // [rsp+8h] [rbp-48h]
  __int64 v35; // [rsp+10h] [rbp-40h]

  if ( a2 != a1 )
  {
    v2 = a2[1];
    v4 = *a2;
    v5 = *a1;
    v6 = v2 - *a2;
    v7 = a1[2] - *a1;
    v35 = v6;
    if ( v7 < v6 )
    {
      if ( v6 )
      {
        if ( v6 > 0x7FFFFFFFFFFFFFD0LL )
          sub_4261EA(a1, v6, v7);
        v32 = v2;
        v19 = sub_22077B0(v6);
        v2 = v32;
        v20 = v19;
      }
      else
      {
        v20 = 0;
      }
      for ( i = v20; v2 != v4; i += 80 )
      {
        if ( i )
        {
          v33 = v2;
          *(__m128i *)i = _mm_loadu_si128((const __m128i *)v4);
          *(_QWORD *)(i + 16) = *(_QWORD *)(v4 + 16);
          *(_QWORD *)(i + 24) = i + 40;
          sub_2F07250((__int64 *)(i + 24), *(_BYTE **)(v4 + 24), *(_QWORD *)(v4 + 24) + *(_QWORD *)(v4 + 32));
          v2 = v33;
          *(__m128i *)(i + 56) = _mm_loadu_si128((const __m128i *)(v4 + 56));
          *(_WORD *)(i + 72) = *(_WORD *)(v4 + 72);
          *(_BYTE *)(i + 74) = *(_BYTE *)(v4 + 74);
        }
        v4 += 80LL;
      }
      v22 = a1[1];
      v23 = *a1;
      if ( v22 != *a1 )
      {
        do
        {
          v24 = *(_QWORD *)(v23 + 24);
          if ( v24 != v23 + 40 )
            j_j___libc_free_0(v24);
          v23 += 80LL;
        }
        while ( v22 != v23 );
        v23 = *a1;
      }
      if ( v23 )
        j_j___libc_free_0(v23);
      *a1 = v20;
      v18 = v6 + v20;
      a1[2] = v18;
      goto LABEL_13;
    }
    v8 = a1[1];
    v9 = v8 - v5;
    v10 = v8 - v5;
    if ( v35 > v8 - v5 )
    {
      v25 = 0xCCCCCCCCCCCCCCCDLL * (v9 >> 4);
      if ( v9 > 0 )
      {
        v26 = (unsigned __int64 *)(v5 + 24);
        v27 = (unsigned __int64 *)(v4 + 24);
        do
        {
          v28 = _mm_loadu_si128((const __m128i *)(v27 - 3));
          v29 = v27;
          v30 = v26;
          v27 += 10;
          v26 += 10;
          *(__m128i *)(v26 - 13) = v28;
          *(v26 - 11) = *(v27 - 11);
          sub_2240AE0(v30, v29);
          *((__m128i *)v26 - 3) = _mm_loadu_si128((const __m128i *)v27 - 3);
          *((_WORD *)v26 - 16) = *((_WORD *)v27 - 16);
          *((_BYTE *)v26 - 30) = *((_BYTE *)v27 - 30);
          --v25;
        }
        while ( v25 );
        v2 = a2[1];
        v4 = *a2;
        v8 = a1[1];
        v5 = *a1;
        v10 = v8 - *a1;
      }
      v31 = v10 + v4;
      v18 = v35 + v5;
      if ( v31 == v2 )
        goto LABEL_13;
      do
      {
        if ( v8 )
        {
          v34 = v2;
          *(__m128i *)v8 = _mm_loadu_si128((const __m128i *)v31);
          *(_QWORD *)(v8 + 16) = *(_QWORD *)(v31 + 16);
          *(_QWORD *)(v8 + 24) = v8 + 40;
          sub_2F07250((__int64 *)(v8 + 24), *(_BYTE **)(v31 + 24), *(_QWORD *)(v31 + 24) + *(_QWORD *)(v31 + 32));
          v2 = v34;
          *(__m128i *)(v8 + 56) = _mm_loadu_si128((const __m128i *)(v31 + 56));
          *(_WORD *)(v8 + 72) = *(_WORD *)(v31 + 72);
          *(_BYTE *)(v8 + 74) = *(_BYTE *)(v31 + 74);
        }
        v31 += 80LL;
        v8 += 80LL;
      }
      while ( v31 != v2 );
    }
    else
    {
      if ( v35 <= 0 )
        goto LABEL_11;
      v11 = (unsigned __int64 *)(v5 + 24);
      v12 = (unsigned __int64 *)(v4 + 24);
      v13 = 0xCCCCCCCCCCCCCCCDLL * (v35 >> 4);
      do
      {
        v14 = _mm_loadu_si128((const __m128i *)(v12 - 3));
        v15 = v12;
        v16 = v11;
        v12 += 10;
        v11 += 10;
        *(__m128i *)(v11 - 13) = v14;
        *(v11 - 11) = *(v12 - 11);
        sub_2240AE0(v16, v15);
        *((__m128i *)v11 - 3) = _mm_loadu_si128((const __m128i *)v12 - 3);
        *((_WORD *)v11 - 16) = *((_WORD *)v12 - 16);
        *((_BYTE *)v11 - 30) = *((_BYTE *)v12 - 30);
        --v13;
      }
      while ( v13 );
      v5 += v35;
      while ( v8 != v5 )
      {
        v17 = *(_QWORD *)(v5 + 24);
        if ( v17 != v5 + 40 )
          j_j___libc_free_0(v17);
        v5 += 80LL;
LABEL_11:
        ;
      }
    }
    v18 = *a1 + v35;
LABEL_13:
    a1[1] = v18;
  }
}
