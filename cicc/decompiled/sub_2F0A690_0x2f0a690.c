// Function: sub_2F0A690
// Address: 0x2f0a690
//
void __fastcall sub_2F0A690(__int64 a1, __m128i **a2)
{
  __m128i *v2; // r8
  __m128i *v4; // rbx
  __m128i *v5; // r14
  signed __int64 v6; // rcx
  unsigned __int64 v7; // rdx
  __m128i *v8; // r12
  __int64 v9; // rdx
  __m128i *v10; // r15
  unsigned __int64 v11; // r13
  __m128i v12; // xmm2
  unsigned __int64 *v13; // rsi
  unsigned __int64 *v14; // rdi
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  __int8 *v18; // r14
  __int64 v19; // rax
  __int64 v20; // r14
  __m128i *i; // r13
  __m128i v22; // xmm4
  __m128i v23; // xmm5
  __m128i v24; // xmm6
  unsigned __int64 *v25; // rbx
  unsigned __int64 *v26; // r13
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rdi
  unsigned __int64 v30; // r13
  __m128i v31; // xmm6
  unsigned __int64 *v32; // rsi
  unsigned __int64 *v33; // rdi
  __m128i *v34; // rbx
  __m128i v35; // xmm0
  __m128i v36; // xmm1
  __m128i v37; // xmm2
  __m128i *v38; // [rsp+8h] [rbp-48h]
  __m128i *v39; // [rsp+8h] [rbp-48h]
  __m128i *v40; // [rsp+8h] [rbp-48h]
  signed __int64 v41; // [rsp+10h] [rbp-40h]

  if ( a2 != (__m128i **)a1 )
  {
    v2 = a2[1];
    v4 = *a2;
    v5 = *(__m128i **)a1;
    v6 = (char *)v2 - (char *)*a2;
    v7 = *(_QWORD *)(a1 + 16) - *(_QWORD *)a1;
    v41 = v6;
    if ( v7 < v6 )
    {
      if ( v6 )
      {
        if ( (unsigned __int64)v6 > 0x7FFFFFFFFFFFFF80LL )
          sub_4261EA(a1, a2, v7);
        v38 = a2[1];
        v19 = sub_22077B0(v6);
        v2 = v38;
        v20 = v19;
      }
      else
      {
        v20 = 0;
      }
      for ( i = (__m128i *)v20; v2 != v4; i += 12 )
      {
        if ( i )
        {
          v39 = v2;
          i->m128i_i64[0] = (__int64)i[1].m128i_i64;
          sub_2F07250(i->m128i_i64, v4->m128i_i64[0], v4->m128i_i64[0] + v4->m128i_i64[1]);
          v22 = _mm_loadu_si128(v4 + 2);
          i[3].m128i_i64[0] = (__int64)i[4].m128i_i64;
          i[2] = v22;
          sub_2F07250(i[3].m128i_i64, (_BYTE *)v4[3].m128i_i64[0], v4[3].m128i_i64[0] + v4[3].m128i_i64[1]);
          v23 = _mm_loadu_si128(v4 + 5);
          i[6].m128i_i64[0] = (__int64)i[7].m128i_i64;
          i[5] = v23;
          sub_2F07250(i[6].m128i_i64, (_BYTE *)v4[6].m128i_i64[0], v4[6].m128i_i64[0] + v4[6].m128i_i64[1]);
          v24 = _mm_loadu_si128(v4 + 8);
          i[9].m128i_i64[0] = (__int64)i[10].m128i_i64;
          i[8] = v24;
          sub_2F07250(i[9].m128i_i64, (_BYTE *)v4[9].m128i_i64[0], v4[9].m128i_i64[0] + v4[9].m128i_i64[1]);
          v2 = v39;
          i[11] = _mm_loadu_si128(v4 + 11);
        }
        v4 += 12;
      }
      v25 = *(unsigned __int64 **)(a1 + 8);
      v26 = *(unsigned __int64 **)a1;
      if ( v25 != *(unsigned __int64 **)a1 )
      {
        do
        {
          v27 = v26[18];
          if ( (unsigned __int64 *)v27 != v26 + 20 )
            j_j___libc_free_0(v27);
          v28 = v26[12];
          if ( (unsigned __int64 *)v28 != v26 + 14 )
            j_j___libc_free_0(v28);
          v29 = v26[6];
          if ( (unsigned __int64 *)v29 != v26 + 8 )
            j_j___libc_free_0(v29);
          if ( (unsigned __int64 *)*v26 != v26 + 2 )
            j_j___libc_free_0(*v26);
          v26 += 24;
        }
        while ( v25 != v26 );
        v26 = *(unsigned __int64 **)a1;
      }
      if ( v26 )
        j_j___libc_free_0((unsigned __int64)v26);
      *(_QWORD *)a1 = v20;
      v18 = (__int8 *)(v41 + v20);
      *(_QWORD *)(a1 + 16) = v18;
      goto LABEL_19;
    }
    v8 = *(__m128i **)(a1 + 8);
    v9 = (char *)v8 - (char *)v5;
    if ( v6 > (unsigned __int64)((char *)v8 - (char *)v5) )
    {
      v30 = 0xAAAAAAAAAAAAAAABLL * (v9 >> 6);
      if ( v9 > 0 )
      {
        do
        {
          sub_2240AE0((unsigned __int64 *)v5, (unsigned __int64 *)v4);
          v5[2] = _mm_loadu_si128(v4 + 2);
          sub_2240AE0((unsigned __int64 *)&v5[3], (unsigned __int64 *)&v4[3]);
          v5[5] = _mm_loadu_si128(v4 + 5);
          sub_2240AE0((unsigned __int64 *)&v5[6], (unsigned __int64 *)&v4[6]);
          v31 = _mm_loadu_si128(v4 + 8);
          v32 = (unsigned __int64 *)&v4[9];
          v33 = (unsigned __int64 *)&v5[9];
          v4 += 12;
          v5 += 12;
          v5[-4] = v31;
          sub_2240AE0(v33, v32);
          v5[-1] = _mm_loadu_si128(v4 - 1);
          --v30;
        }
        while ( v30 );
        v2 = a2[1];
        v4 = *a2;
        v8 = *(__m128i **)(a1 + 8);
        v5 = *(__m128i **)a1;
        v9 = (__int64)v8->m128i_i64 - *(_QWORD *)a1;
      }
      v34 = (__m128i *)((char *)v4 + v9);
      v18 = &v5->m128i_i8[v41];
      if ( v34 == v2 )
        goto LABEL_19;
      do
      {
        if ( v8 )
        {
          v40 = v2;
          v8->m128i_i64[0] = (__int64)v8[1].m128i_i64;
          sub_2F07250(v8->m128i_i64, v34->m128i_i64[0], v34->m128i_i64[0] + v34->m128i_i64[1]);
          v35 = _mm_loadu_si128(v34 + 2);
          v8[3].m128i_i64[0] = (__int64)v8[4].m128i_i64;
          v8[2] = v35;
          sub_2F07250(v8[3].m128i_i64, (_BYTE *)v34[3].m128i_i64[0], v34[3].m128i_i64[0] + v34[3].m128i_i64[1]);
          v36 = _mm_loadu_si128(v34 + 5);
          v8[6].m128i_i64[0] = (__int64)v8[7].m128i_i64;
          v8[5] = v36;
          sub_2F07250(v8[6].m128i_i64, (_BYTE *)v34[6].m128i_i64[0], v34[6].m128i_i64[0] + v34[6].m128i_i64[1]);
          v37 = _mm_loadu_si128(v34 + 8);
          v8[9].m128i_i64[0] = (__int64)v8[10].m128i_i64;
          v8[8] = v37;
          sub_2F07250(v8[9].m128i_i64, (_BYTE *)v34[9].m128i_i64[0], v34[9].m128i_i64[0] + v34[9].m128i_i64[1]);
          v2 = v40;
          v8[11] = _mm_loadu_si128(v34 + 11);
        }
        v34 += 12;
        v8 += 12;
      }
      while ( v34 != v2 );
    }
    else
    {
      if ( v6 <= 0 )
        goto LABEL_17;
      v10 = *(__m128i **)a1;
      v11 = 0xAAAAAAAAAAAAAAABLL * (v6 >> 6);
      do
      {
        sub_2240AE0((unsigned __int64 *)v10, (unsigned __int64 *)v4);
        v10[2] = _mm_loadu_si128(v4 + 2);
        sub_2240AE0((unsigned __int64 *)&v10[3], (unsigned __int64 *)&v4[3]);
        v10[5] = _mm_loadu_si128(v4 + 5);
        sub_2240AE0((unsigned __int64 *)&v10[6], (unsigned __int64 *)&v4[6]);
        v12 = _mm_loadu_si128(v4 + 8);
        v13 = (unsigned __int64 *)&v4[9];
        v14 = (unsigned __int64 *)&v10[9];
        v4 += 12;
        v10 += 12;
        v10[-4] = v12;
        sub_2240AE0(v14, v13);
        v10[-1] = _mm_loadu_si128(v4 - 1);
        --v11;
      }
      while ( v11 );
      v5 = (__m128i *)((char *)v5 + v41);
      while ( v8 != v5 )
      {
        v15 = v5[9].m128i_u64[0];
        if ( (__m128i *)v15 != &v5[10] )
          j_j___libc_free_0(v15);
        v16 = v5[6].m128i_u64[0];
        if ( (__m128i *)v16 != &v5[7] )
          j_j___libc_free_0(v16);
        v17 = v5[3].m128i_u64[0];
        if ( (__m128i *)v17 != &v5[4] )
          j_j___libc_free_0(v17);
        if ( (__m128i *)v5->m128i_i64[0] != &v5[1] )
          j_j___libc_free_0(v5->m128i_i64[0]);
        v5 += 12;
LABEL_17:
        ;
      }
    }
    v18 = (__int8 *)(*(_QWORD *)a1 + v41);
LABEL_19:
    *(_QWORD *)(a1 + 8) = v18;
  }
}
