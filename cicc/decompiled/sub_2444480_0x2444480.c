// Function: sub_2444480
// Address: 0x2444480
//
__m128i *__fastcall sub_2444480(__m128i *src, __m128i *a2, __m128i *a3)
{
  char *v3; // r12
  __m128i *v4; // rbx
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // r8
  const __m128i *v8; // rcx
  __m128i *v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // r10
  __int64 v12; // r9
  __m128i v13; // xmm0
  __int64 v14; // rdx
  __int64 v15; // rcx
  __m128i *v16; // rcx
  __m128i *v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // r10
  __int64 v20; // r9
  __m128i v21; // xmm1
  const __m128i *v23; // rax
  __m128i v24; // xmm2
  __int64 v25; // rcx
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // r15
  __int64 v29; // r14
  __int64 v30; // r13
  __int64 *v31; // rbx
  __m128i *v32; // rax
  __int64 v33; // r14
  __int64 v34; // r13

  v3 = (char *)a3;
  if ( src != a2 )
  {
    v4 = src;
    if ( a2 == a3 )
    {
      return src;
    }
    else
    {
      v3 = &src->m128i_i8[(char *)a3 - (char *)a2];
      v5 = a3 - src;
      v6 = a2 - src;
      if ( v6 == v5 - v6 )
      {
        v23 = a2;
        do
        {
          v24 = _mm_loadu_si128(v23);
          v25 = v4->m128i_i64[0];
          ++v4;
          ++v23;
          v26 = v4[-1].m128i_i64[1];
          v4[-1] = v24;
          v23[-1].m128i_i64[0] = v25;
          v23[-1].m128i_i64[1] = v26;
        }
        while ( a2 != v4 );
        return a2;
      }
      else
      {
        while ( 1 )
        {
          v7 = v5 - v6;
          if ( v6 < v5 - v6 )
            break;
LABEL_12:
          v15 = v5;
          if ( v7 == 1 )
          {
            v32 = &v4[v15 - 1];
            v33 = v32->m128i_i64[0];
            v34 = v32->m128i_i64[1];
            if ( v4 != v32 )
              memmove(&v4[1], v4, v15 * 16 - 16);
            v4->m128i_i64[0] = v33;
            v4->m128i_i64[1] = v34;
            return (__m128i *)v3;
          }
          v16 = &v4[v15];
          v4 = &v16[-v7];
          if ( v6 > 0 )
          {
            v17 = &v16[-v7];
            v18 = 0;
            do
            {
              v19 = v17[-1].m128i_i64[0];
              v20 = v17[-1].m128i_i64[1];
              ++v18;
              --v17;
              v21 = _mm_loadu_si128(--v16);
              *v17 = v21;
              v16->m128i_i64[0] = v19;
              v16->m128i_i64[1] = v20;
            }
            while ( v6 != v18 );
            v4 -= v6;
          }
          v6 = v5 % v7;
          if ( !(v5 % v7) )
            return (__m128i *)v3;
          v5 = v7;
        }
        while ( v6 != 1 )
        {
          v8 = &v4[v6];
          if ( v7 > 0 )
          {
            v9 = v4;
            v10 = 0;
            do
            {
              v11 = v9->m128i_i64[0];
              v12 = v9->m128i_i64[1];
              ++v10;
              ++v9;
              v13 = _mm_loadu_si128(v8++);
              v9[-1] = v13;
              v8[-1].m128i_i64[0] = v11;
              v8[-1].m128i_i64[1] = v12;
            }
            while ( v7 != v10 );
            v4 += v7;
          }
          v14 = v5 % v6;
          if ( !(v5 % v6) )
            return (__m128i *)v3;
          v5 = v6;
          v6 -= v14;
          v7 = v5 - v6;
          if ( v6 >= v5 - v6 )
            goto LABEL_12;
        }
        v27 = 16 * v5;
        v28 = v4->m128i_i64[0];
        v29 = v4->m128i_i64[1];
        v30 = v27 - 16;
        if ( &v4[1] != &v4[(unsigned __int64)v27 / 0x10] )
          memmove(v4, &v4[1], v27 - 16);
        v31 = (__int64 *)((char *)v4->m128i_i64 + v30);
        *v31 = v28;
        v31[1] = v29;
      }
    }
  }
  return (__m128i *)v3;
}
