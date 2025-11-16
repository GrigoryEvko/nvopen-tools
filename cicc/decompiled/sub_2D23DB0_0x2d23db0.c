// Function: sub_2D23DB0
// Address: 0x2d23db0
//
unsigned __int64 __fastcall sub_2D23DB0(
        __int64 a1,
        __m128i *a2,
        unsigned __int64 a3,
        const __m128i *a4,
        __int64 a5,
        __int64 a6)
{
  __int64 v9; // rcx
  unsigned __int64 v10; // r8
  unsigned __int64 result; // rax
  unsigned __int64 v12; // rdx
  __int64 v13; // rsi
  __m128i *v14; // rbx
  __int64 v15; // r9
  size_t v16; // rdx
  unsigned __int64 v17; // r10
  unsigned int v18; // ecx
  __m128i *j; // rdx
  unsigned __int64 v20; // rdx
  __m128i *v21; // rdx
  __int64 v22; // r10
  __int64 v23; // r11
  __int64 v24; // r8
  unsigned __int64 v25; // rax
  const __m128i *v26; // rcx
  __m128i *i; // r10
  const void *v28; // rsi
  __int8 *v29; // r13
  const void *v30; // rsi
  __int8 *v31; // r13
  unsigned __int64 v32; // [rsp+8h] [rbp-58h]
  __int64 v33; // [rsp+10h] [rbp-50h]
  unsigned __int64 v34; // [rsp+20h] [rbp-40h]
  __int64 v35; // [rsp+20h] [rbp-40h]
  size_t v36; // [rsp+28h] [rbp-38h]
  __int64 v37; // [rsp+28h] [rbp-38h]
  __int8 *v38; // [rsp+28h] [rbp-38h]
  __int64 v39; // [rsp+28h] [rbp-38h]

  v9 = *(unsigned int *)(a1 + 8);
  v10 = *(_QWORD *)a1;
  result = *(unsigned int *)(a1 + 12);
  v12 = v9 + a3;
  v13 = 24 * v9;
  v14 = (__m128i *)(*(_QWORD *)a1 + 24 * v9);
  if ( a2 == v14 )
  {
    if ( v12 > result )
    {
      v30 = (const void *)(a1 + 16);
      if ( v10 > (unsigned __int64)a4 || v14 <= a4 )
      {
        sub_C8D5F0(a1, v30, v12, 0x18u, v10, a6);
        v9 = *(unsigned int *)(a1 + 8);
        result = *(_QWORD *)a1;
        v14 = (__m128i *)(*(_QWORD *)a1 + 24 * v9);
      }
      else
      {
        v31 = &a4->m128i_i8[-v10];
        sub_C8D5F0(a1, v30, v12, 0x18u, v10, a6);
        v9 = *(unsigned int *)(a1 + 8);
        result = *(_QWORD *)a1;
        a4 = (const __m128i *)&v31[*(_QWORD *)a1];
        v14 = (__m128i *)(*(_QWORD *)a1 + 24 * v9);
      }
    }
    if ( a3 )
    {
      result = a3;
      do
      {
        if ( v14 )
        {
          *v14 = _mm_loadu_si128(a4);
          v14[1].m128i_i64[0] = a4[1].m128i_i64[0];
        }
        v14 = (__m128i *)((char *)v14 + 24);
        --result;
      }
      while ( result );
      LODWORD(v9) = *(_DWORD *)(a1 + 8);
    }
    *(_DWORD *)(a1 + 8) = a3 + v9;
  }
  else
  {
    v15 = (__int64)a2->m128i_i64 - v10;
    if ( v12 > result )
    {
      v38 = &a2->m128i_i8[-v10];
      v28 = (const void *)(a1 + 16);
      if ( v10 > (unsigned __int64)a4 || v14 <= a4 )
      {
        result = sub_C8D5F0(a1, v28, v12, 0x18u, v10, v15);
        v10 = *(_QWORD *)a1;
      }
      else
      {
        v29 = &a4->m128i_i8[-v10];
        result = sub_C8D5F0(a1, v28, v12, 0x18u, v10, v15);
        v10 = *(_QWORD *)a1;
        a4 = (const __m128i *)&v29[*(_QWORD *)a1];
      }
      v9 = *(unsigned int *)(a1 + 8);
      v15 = (__int64)v38;
      a2 = (__m128i *)&v38[v10];
      v13 = 24 * v9;
      v14 = (__m128i *)(v10 + 24 * v9);
    }
    v16 = v13 - v15;
    v17 = 0xAAAAAAAAAAAAAAABLL * ((v13 - v15) >> 3);
    if ( a3 <= v17 )
    {
      v21 = v14;
      v22 = 24 * a3;
      v23 = 24 * (v9 - a3);
      v24 = v23 + v10;
      v25 = 0xAAAAAAAAAAAAAAABLL * ((v13 - v23) >> 3);
      if ( v25 + v9 > *(unsigned int *)(a1 + 12) )
      {
        v32 = 0xAAAAAAAAAAAAAAABLL * ((v13 - v23) >> 3);
        v33 = 24 * (v9 - a3);
        v35 = v24;
        v39 = v15;
        sub_C8D5F0(a1, (const void *)(a1 + 16), v25 + v9, 0x18u, v24, v15);
        v9 = *(unsigned int *)(a1 + 8);
        v25 = v32;
        v23 = v33;
        v22 = 24 * a3;
        v24 = v35;
        v15 = v39;
        v21 = (__m128i *)(*(_QWORD *)a1 + 24 * v9);
      }
      if ( (__m128i *)v24 != v14 )
      {
        v26 = (const __m128i *)v24;
        do
        {
          if ( v21 )
          {
            *v21 = _mm_loadu_si128(v26);
            v21[1].m128i_i64[0] = v26[1].m128i_i64[0];
          }
          v26 = (const __m128i *)((char *)v26 + 24);
          v21 = (__m128i *)((char *)v21 + 24);
        }
        while ( v26 != v14 );
        v9 = *(unsigned int *)(a1 + 8);
      }
      result = v9 + v25;
      *(_DWORD *)(a1 + 8) = result;
      if ( (__m128i *)v24 != a2 )
      {
        v37 = v22;
        result = (unsigned __int64)memmove((char *)v14 - (v23 - v15), a2, v23 - v15);
        v22 = v37;
      }
      if ( a4 >= a2 )
      {
        result = (unsigned __int64)a4->m128i_u64 + v22;
        if ( (unsigned __int64)a4 < *(_QWORD *)a1 + 24 * (unsigned __int64)*(unsigned int *)(a1 + 8) )
          a4 = (const __m128i *)((char *)a4 + v22);
      }
      if ( a3 )
      {
        for ( i = (__m128i *)((char *)a2 + v22); i != a2; a2[-1].m128i_i64[1] = result )
        {
          a2 = (__m128i *)((char *)a2 + 24);
          *(__m128i *)((char *)a2 - 24) = _mm_loadu_si128(a4);
          result = a4[1].m128i_u64[0];
        }
      }
    }
    else
    {
      v18 = a3 + v9;
      *(_DWORD *)(a1 + 8) = v18;
      if ( v14 != a2 )
      {
        v34 = 0xAAAAAAAAAAAAAAABLL * ((v13 - v15) >> 3);
        v36 = v13 - v15;
        result = (unsigned __int64)memcpy((void *)(v10 + 24LL * v18 - v16), a2, v16);
        v17 = v34;
        v16 = v36;
      }
      if ( a4 >= a2 )
      {
        result = *(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8);
        if ( (unsigned __int64)a4 < result )
        {
          result = 3 * a3;
          a4 = (const __m128i *)((char *)a4 + 24 * a3);
        }
      }
      if ( v17 )
      {
        for ( j = (__m128i *)((char *)a2 + v16); j != a2; a2[-1].m128i_i64[1] = result )
        {
          a2 = (__m128i *)((char *)a2 + 24);
          *(__m128i *)((char *)a2 - 24) = _mm_loadu_si128(a4);
          result = a4[1].m128i_u64[0];
        }
      }
      v20 = a3 - v17;
      do
      {
        if ( v14 )
        {
          *v14 = _mm_loadu_si128(a4);
          result = a4[1].m128i_u64[0];
          v14[1].m128i_i64[0] = result;
        }
        v14 = (__m128i *)((char *)v14 + 24);
        --v20;
      }
      while ( v20 );
    }
  }
  return result;
}
