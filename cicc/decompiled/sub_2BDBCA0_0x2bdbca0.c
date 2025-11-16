// Function: sub_2BDBCA0
// Address: 0x2bdbca0
//
void __fastcall sub_2BDBCA0(unsigned __int64 *a1, const __m128i **a2, __int64 a3)
{
  const __m128i *v4; // r14
  const __m128i *v5; // rbx
  unsigned __int64 v6; // rdi
  __int64 v7; // r13
  __m128i *v8; // rax
  __int64 v9; // rcx
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  unsigned __int64 v12; // r13
  __int64 v13; // rax
  __m128i *v14; // r15
  __m128i *i; // rax
  unsigned __int64 v16; // rdx
  __int64 v17; // rax
  const __m128i *v18; // rbx

  if ( a2 != (const __m128i **)a1 )
  {
    v4 = a2[1];
    v5 = *a2;
    v6 = *a1;
    v7 = (char *)v4 - (char *)*a2;
    if ( v7 > a1[2] - v6 )
    {
      if ( v7 )
      {
        if ( (unsigned __int64)v7 > 0x7FFFFFFFFFFFFFF8LL )
          sub_4261EA(v6, a2, a3);
        v13 = sub_22077B0((char *)a2[1] - (char *)*a2);
        v6 = *a1;
        v14 = (__m128i *)v13;
      }
      else
      {
        v14 = 0;
      }
      for ( i = v14; v4 != v5; i = (__m128i *)((char *)i + 24) )
      {
        if ( i )
        {
          *i = _mm_loadu_si128(v5);
          i[1].m128i_i64[0] = v5[1].m128i_i64[0];
        }
        v5 = (const __m128i *)((char *)v5 + 24);
      }
      if ( v6 )
        j_j___libc_free_0(v6);
      v12 = (unsigned __int64)v14->m128i_u64 + v7;
      *a1 = (unsigned __int64)v14;
      a1[2] = v12;
      goto LABEL_8;
    }
    v8 = (__m128i *)a1[1];
    v9 = (__int64)v8->m128i_i64 - v6;
    if ( v7 > (unsigned __int64)v8 - v6 )
    {
      v16 = 0xAAAAAAAAAAAAAAABLL * (v9 >> 3);
      if ( v9 > 0 )
      {
        do
        {
          v17 = v5->m128i_i64[0];
          v6 += 24LL;
          v5 = (const __m128i *)((char *)v5 + 24);
          *(_QWORD *)(v6 - 24) = v17;
          *(_QWORD *)(v6 - 16) = v5[-1].m128i_i64[0];
          *(_BYTE *)(v6 - 8) = v5[-1].m128i_i8[8];
          --v16;
        }
        while ( v16 );
        v8 = (__m128i *)a1[1];
        v6 = *a1;
        v4 = a2[1];
        v5 = *a2;
        v9 = (__int64)v8->m128i_i64 - *a1;
      }
      v18 = (const __m128i *)((char *)v5 + v9);
      if ( v18 != v4 )
      {
        do
        {
          if ( v8 )
          {
            *v8 = _mm_loadu_si128(v18);
            v8[1].m128i_i64[0] = v18[1].m128i_i64[0];
          }
          v18 = (const __m128i *)((char *)v18 + 24);
          v8 = (__m128i *)((char *)v8 + 24);
        }
        while ( v18 != v4 );
        goto LABEL_7;
      }
    }
    else if ( v7 > 0 )
    {
      v10 = 0xAAAAAAAAAAAAAAABLL * (v7 >> 3);
      do
      {
        v11 = v5->m128i_i64[0];
        v6 += 24LL;
        v5 = (const __m128i *)((char *)v5 + 24);
        *(_QWORD *)(v6 - 24) = v11;
        *(_QWORD *)(v6 - 16) = v5[-1].m128i_i64[0];
        *(_BYTE *)(v6 - 8) = v5[-1].m128i_i8[8];
        --v10;
      }
      while ( v10 );
LABEL_7:
      v12 = *a1 + v7;
LABEL_8:
      a1[1] = v12;
      return;
    }
    v12 = v6 + v7;
    goto LABEL_8;
  }
}
