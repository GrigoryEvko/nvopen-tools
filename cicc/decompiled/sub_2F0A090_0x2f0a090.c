// Function: sub_2F0A090
// Address: 0x2f0a090
//
void __fastcall sub_2F0A090(__int64 a1, __m128i **a2)
{
  __m128i *v2; // rcx
  __m128i *v4; // rbx
  unsigned __int64 *v5; // r14
  signed __int64 v6; // rsi
  unsigned __int64 v7; // rdx
  __m128i *v8; // r13
  __int64 v9; // rdx
  unsigned __int64 *v10; // r15
  unsigned __int64 v11; // r12
  __m128i v12; // xmm0
  unsigned __int64 *v13; // rsi
  unsigned __int64 *v14; // rdi
  unsigned __int64 v15; // rdi
  char *v16; // r14
  __int64 v17; // rax
  __int64 v18; // r14
  __m128i *i; // r12
  __m128i v20; // xmm6
  unsigned __int64 *v21; // rbx
  unsigned __int64 *v22; // r12
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // r12
  __m128i v25; // xmm2
  unsigned __int64 *v26; // rsi
  unsigned __int64 *v27; // rdi
  __m128i *v28; // rbx
  __m128i v29; // xmm4
  __m128i *v30; // [rsp+8h] [rbp-48h]
  __m128i *v31; // [rsp+8h] [rbp-48h]
  __m128i *v32; // [rsp+8h] [rbp-48h]
  signed __int64 v33; // [rsp+10h] [rbp-40h]

  if ( a2 != (__m128i **)a1 )
  {
    v2 = a2[1];
    v4 = *a2;
    v5 = *(unsigned __int64 **)a1;
    v6 = (char *)v2 - (char *)*a2;
    v7 = *(_QWORD *)(a1 + 16) - *(_QWORD *)a1;
    v33 = v6;
    if ( v7 < v6 )
    {
      if ( v6 )
      {
        if ( (unsigned __int64)v6 > 0x7FFFFFFFFFFFFFE0LL )
          sub_4261EA(a1, v6, v7);
        v30 = v2;
        v17 = sub_22077B0(v6);
        v2 = v30;
        v18 = v17;
      }
      else
      {
        v18 = 0;
      }
      for ( i = (__m128i *)v18; v2 != v4; i += 6 )
      {
        if ( i )
        {
          v31 = v2;
          i->m128i_i64[0] = (__int64)i[1].m128i_i64;
          sub_2F07250(i->m128i_i64, v4->m128i_i64[0], v4->m128i_i64[0] + v4->m128i_i64[1]);
          v20 = _mm_loadu_si128(v4 + 2);
          i[3].m128i_i64[0] = (__int64)i[4].m128i_i64;
          i[2] = v20;
          sub_2F07250(i[3].m128i_i64, (_BYTE *)v4[3].m128i_i64[0], v4[3].m128i_i64[0] + v4[3].m128i_i64[1]);
          v2 = v31;
          i[5] = _mm_loadu_si128(v4 + 5);
        }
        v4 += 6;
      }
      v21 = *(unsigned __int64 **)(a1 + 8);
      v22 = *(unsigned __int64 **)a1;
      if ( v21 != *(unsigned __int64 **)a1 )
      {
        do
        {
          v23 = v22[6];
          if ( (unsigned __int64 *)v23 != v22 + 8 )
            j_j___libc_free_0(v23);
          if ( (unsigned __int64 *)*v22 != v22 + 2 )
            j_j___libc_free_0(*v22);
          v22 += 12;
        }
        while ( v21 != v22 );
        v22 = *(unsigned __int64 **)a1;
      }
      if ( v22 )
        j_j___libc_free_0((unsigned __int64)v22);
      *(_QWORD *)a1 = v18;
      v16 = (char *)(v6 + v18);
      *(_QWORD *)(a1 + 16) = v16;
      goto LABEL_15;
    }
    v8 = *(__m128i **)(a1 + 8);
    v9 = (char *)v8 - (char *)v5;
    if ( v6 > (unsigned __int64)((char *)v8 - (char *)v5) )
    {
      v24 = 0xAAAAAAAAAAAAAAABLL * (v9 >> 5);
      if ( v9 > 0 )
      {
        do
        {
          sub_2240AE0(v5, (unsigned __int64 *)v4);
          v25 = _mm_loadu_si128(v4 + 2);
          v26 = (unsigned __int64 *)&v4[3];
          v27 = v5 + 6;
          v4 += 6;
          v5 += 12;
          *((__m128i *)v5 - 4) = v25;
          sub_2240AE0(v27, v26);
          *((__m128i *)v5 - 1) = _mm_loadu_si128(v4 - 1);
          --v24;
        }
        while ( v24 );
        v2 = a2[1];
        v4 = *a2;
        v8 = *(__m128i **)(a1 + 8);
        v5 = *(unsigned __int64 **)a1;
        v9 = (__int64)v8->m128i_i64 - *(_QWORD *)a1;
      }
      v28 = (__m128i *)((char *)v4 + v9);
      if ( v28 == v2 )
      {
        v16 = (char *)v5 + v33;
        goto LABEL_15;
      }
      do
      {
        if ( v8 )
        {
          v32 = v2;
          v8->m128i_i64[0] = (__int64)v8[1].m128i_i64;
          sub_2F07250(v8->m128i_i64, v28->m128i_i64[0], v28->m128i_i64[0] + v28->m128i_i64[1]);
          v29 = _mm_loadu_si128(v28 + 2);
          v8[3].m128i_i64[0] = (__int64)v8[4].m128i_i64;
          v8[2] = v29;
          sub_2F07250(v8[3].m128i_i64, (_BYTE *)v28[3].m128i_i64[0], v28[3].m128i_i64[0] + v28[3].m128i_i64[1]);
          v2 = v32;
          v8[5] = _mm_loadu_si128(v28 + 5);
        }
        v28 += 6;
        v8 += 6;
      }
      while ( v2 != v28 );
    }
    else
    {
      if ( v6 <= 0 )
        goto LABEL_13;
      v10 = *(unsigned __int64 **)a1;
      v11 = 0xAAAAAAAAAAAAAAABLL * (v6 >> 5);
      do
      {
        sub_2240AE0(v10, (unsigned __int64 *)v4);
        v12 = _mm_loadu_si128(v4 + 2);
        v13 = (unsigned __int64 *)&v4[3];
        v14 = v10 + 6;
        v4 += 6;
        v10 += 12;
        *((__m128i *)v10 - 4) = v12;
        sub_2240AE0(v14, v13);
        *((__m128i *)v10 - 1) = _mm_loadu_si128(v4 - 1);
        --v11;
      }
      while ( v11 );
      v5 = (unsigned __int64 *)((char *)v5 + v33);
      while ( v8 != (__m128i *)v5 )
      {
        v15 = v5[6];
        if ( (unsigned __int64 *)v15 != v5 + 8 )
          j_j___libc_free_0(v15);
        if ( (unsigned __int64 *)*v5 != v5 + 2 )
          j_j___libc_free_0(*v5);
        v5 += 12;
LABEL_13:
        ;
      }
    }
    v16 = (char *)(*(_QWORD *)a1 + v33);
LABEL_15:
    *(_QWORD *)(a1 + 8) = v16;
  }
}
