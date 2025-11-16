// Function: sub_3938290
// Address: 0x3938290
//
void __fastcall sub_3938290(__int64 a1, const __m128i ***a2)
{
  const __m128i ***v2; // rcx
  const __m128i **v4; // rbx
  const __m128i **v5; // rdx
  __m128i **v6; // r14
  unsigned __int64 v7; // rsi
  char *v8; // r12
  __int64 v9; // rsi
  __m128i **v10; // rbx
  unsigned __int64 v11; // r15
  __m128i *v12; // rbx
  unsigned __int64 v13; // rdi
  char *v14; // r14
  __int64 v15; // rax
  _QWORD *v16; // r15
  const __m128i **i; // r12
  const __m128i *j; // r14
  __m128i *v19; // rax
  __m128i **v20; // rbx
  __m128i **v21; // r12
  __m128i **v22; // r15
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // r15
  const __m128i **v25; // r15
  const __m128i *k; // r14
  __m128i *v27; // rax
  const __m128i **v28; // [rsp-50h] [rbp-50h]
  const __m128i ***v29; // [rsp-50h] [rbp-50h]
  const __m128i **v30; // [rsp-48h] [rbp-48h]
  _QWORD *v31; // [rsp-48h] [rbp-48h]
  const __m128i **v32; // [rsp-48h] [rbp-48h]
  signed __int64 v33; // [rsp-40h] [rbp-40h]

  if ( a2 != (const __m128i ***)a1 )
  {
    v2 = a2;
    v4 = a2[1];
    v5 = *a2;
    v6 = *(__m128i ***)a1;
    v7 = *(_QWORD *)(a1 + 16) - *(_QWORD *)a1;
    v33 = (char *)v4 - (char *)v5;
    if ( v7 < (char *)v4 - (char *)v5 )
    {
      if ( v4 == v5 )
      {
        v31 = 0;
      }
      else
      {
        if ( (unsigned __int64)((char *)v4 - (char *)v5) > 0x7FFFFFFFFFFFFFF8LL )
          sub_4261EA(a1, v7, v5);
        v28 = v5;
        v15 = sub_22077B0(v33);
        v5 = v28;
        v31 = (_QWORD *)v15;
      }
      v16 = v31;
      for ( i = v5; v4 != i; v16 += 3 )
      {
        if ( v16 )
        {
          v16[1] = v16;
          *v16 = v16;
          v16[2] = 0;
          for ( j = *i; j != (const __m128i *)i; j = (const __m128i *)j->m128i_i64[0] )
          {
            v19 = (__m128i *)sub_22077B0(0x20u);
            v19[1] = _mm_loadu_si128(j + 1);
            sub_2208C80(v19, (__int64)v16);
            ++v16[2];
          }
        }
        i += 3;
      }
      v20 = *(__m128i ***)(a1 + 8);
      v21 = *(__m128i ***)a1;
      if ( v20 != *(__m128i ***)a1 )
      {
        do
        {
          v22 = (__m128i **)*v21;
          if ( v21 != (__m128i **)*v21 )
          {
            do
            {
              v23 = (unsigned __int64)v22;
              v22 = (__m128i **)*v22;
              j_j___libc_free_0(v23);
            }
            while ( v21 != v22 );
          }
          v21 += 3;
        }
        while ( v20 != v21 );
        v21 = *(__m128i ***)a1;
      }
      if ( v21 )
        j_j___libc_free_0((unsigned __int64)v21);
      v14 = (char *)v31 + v33;
      *(_QWORD *)a1 = v31;
      *(_QWORD *)(a1 + 16) = (char *)v31 + v33;
      goto LABEL_15;
    }
    v8 = *(char **)(a1 + 8);
    v9 = v8 - (char *)v6;
    if ( v33 > (unsigned __int64)(v8 - (char *)v6) )
    {
      v24 = 0xAAAAAAAAAAAAAAABLL * ((v8 - (char *)v6) >> 3);
      if ( v9 > 0 )
      {
        do
        {
          if ( v5 != (const __m128i **)v6 )
          {
            v29 = v2;
            v32 = v5;
            sub_3937D80(v6, *v5, (const __m128i *)v5);
            v2 = v29;
            v5 = v32;
          }
          v5 += 3;
          v6 += 3;
          --v24;
        }
        while ( v24 );
        v8 = *(char **)(a1 + 8);
        v6 = *(__m128i ***)a1;
        v4 = v2[1];
        v5 = *v2;
        v9 = (__int64)&v8[-*(_QWORD *)a1];
      }
      v25 = (const __m128i **)((char *)v5 + v9);
      v14 = (char *)v6 + v33;
      if ( (const __m128i **)((char *)v5 + v9) == v4 )
        goto LABEL_15;
      do
      {
        if ( v8 )
        {
          *((_QWORD *)v8 + 1) = v8;
          *(_QWORD *)v8 = v8;
          *((_QWORD *)v8 + 2) = 0;
          for ( k = *v25; v25 != (const __m128i **)k; k = (const __m128i *)k->m128i_i64[0] )
          {
            v27 = (__m128i *)sub_22077B0(0x20u);
            v27[1] = _mm_loadu_si128(k + 1);
            sub_2208C80(v27, (__int64)v8);
            ++*((_QWORD *)v8 + 2);
          }
        }
        v25 += 3;
        v8 += 24;
      }
      while ( v4 != v25 );
    }
    else
    {
      if ( v33 > 0 )
      {
        v10 = *(__m128i ***)a1;
        v11 = 0xAAAAAAAAAAAAAAABLL * (v33 >> 3);
        do
        {
          if ( v5 != (const __m128i **)v10 )
          {
            v30 = v5;
            sub_3937D80(v10, *v5, (const __m128i *)v5);
            v5 = v30;
          }
          v5 += 3;
          v10 += 3;
          --v11;
        }
        while ( v11 );
        v6 = (__m128i **)((char *)v6 + v33);
      }
      for ( ; v8 != (char *)v6; v6 += 3 )
      {
        v12 = *v6;
        while ( v12 != (__m128i *)v6 )
        {
          v13 = (unsigned __int64)v12;
          v12 = (__m128i *)v12->m128i_i64[0];
          j_j___libc_free_0(v13);
        }
      }
    }
    v14 = (char *)(*(_QWORD *)a1 + v33);
LABEL_15:
    *(_QWORD *)(a1 + 8) = v14;
  }
}
