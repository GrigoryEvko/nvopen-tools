// Function: sub_2BDEEE0
// Address: 0x2bdeee0
//
void __fastcall sub_2BDEEE0(unsigned __int64 *a1, unsigned __int64 a2, const __m128i *a3)
{
  unsigned __int64 v5; // rdi
  __m128i *v6; // rcx
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rsi
  __m128i *v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v11; // r8
  unsigned __int64 v12; // rsi
  unsigned __int64 v13; // rax
  __int64 v14; // r13
  __int64 v15; // rax
  unsigned __int64 v16; // rsi
  __int64 v17; // rdx
  __m128i v18; // xmm1

  v5 = *a1;
  if ( 0xAAAAAAAAAAAAAAABLL * ((__int64)(a1[2] - v5) >> 3) < a2 )
  {
    if ( a2 > 0x555555555555555LL )
      sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
    v14 = 24 * a2;
    if ( a2 )
    {
      v15 = sub_22077B0(24 * a2);
      v16 = v15;
      v17 = v15 + v14;
      do
      {
        if ( v15 )
        {
          v18 = _mm_loadu_si128(a3);
          *(_QWORD *)(v15 + 16) = a3[1].m128i_i64[0];
          *(__m128i *)v15 = v18;
        }
        v15 += 24;
      }
      while ( v17 != v15 );
      v5 = *a1;
    }
    else
    {
      v16 = 0;
      v17 = 0;
    }
    *a1 = v16;
    a1[1] = v17;
    a1[2] = v17;
    if ( v5 )
      j_j___libc_free_0(v5);
  }
  else
  {
    v6 = (__m128i *)a1[1];
    v7 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)v6->m128i_i64 - v5) >> 3);
    if ( a2 <= v7 )
    {
      if ( a2 )
      {
        v11 = 24 * a2;
        v12 = v5 + 24 * a2;
        if ( v5 != v12 )
        {
          v13 = v5;
          do
          {
            v13 += 24LL;
            *(_QWORD *)(v13 - 24) = a3->m128i_i64[0];
            *(_QWORD *)(v13 - 16) = a3->m128i_i64[1];
            *(_BYTE *)(v13 - 8) = a3[1].m128i_i8[0];
          }
          while ( v12 != v13 );
          v6 = (__m128i *)a1[1];
          v5 += v11;
        }
      }
      if ( (__m128i *)v5 != v6 )
        a1[1] = v5;
    }
    else
    {
      if ( (__m128i *)v5 != v6 )
      {
        do
        {
          v5 += 24LL;
          *(_QWORD *)(v5 - 24) = a3->m128i_i64[0];
          *(_QWORD *)(v5 - 16) = a3->m128i_i64[1];
          *(_BYTE *)(v5 - 8) = a3[1].m128i_i8[0];
        }
        while ( v6 != (__m128i *)v5 );
        v6 = (__m128i *)a1[1];
        v7 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)v6->m128i_i64 - *a1) >> 3);
      }
      v8 = a2 - v7;
      if ( v8 )
      {
        v9 = v6;
        v10 = v8;
        do
        {
          if ( v9 )
          {
            *v9 = _mm_loadu_si128(a3);
            v9[1].m128i_i64[0] = a3[1].m128i_i64[0];
          }
          v9 = (__m128i *)((char *)v9 + 24);
          --v10;
        }
        while ( v10 );
        v6 = (__m128i *)((char *)v6 + 24 * v8);
      }
      a1[1] = (unsigned __int64)v6;
    }
  }
}
