// Function: sub_343F700
// Address: 0x343f700
//
unsigned __int64 __fastcall sub_343F700(const __m128i *a1, const __m128i *a2, const __m128i *a3)
{
  const __m128i *v3; // r12
  __m128i *v4; // r11
  signed __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // r8
  const __m128i *v8; // rcx
  __int64 *v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // r10
  __int64 v12; // r9
  __m128i v13; // xmm0
  __int64 *v14; // rcx
  __m128i *v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // r10
  __int64 v18; // r9
  __m128i v19; // xmm1
  const __m128i *v21; // rdx
  const __m128i *v22; // rax
  __int64 v23; // rdi
  __m128i v24; // xmm2
  __int64 v25; // rcx

  v3 = a3;
  if ( a1 == a2 )
    return (unsigned __int64)v3;
  v4 = (__m128i *)a1;
  if ( a2 != a3 )
  {
    v3 = (const __m128i *)((char *)a1 + (char *)a3 - (char *)a2);
    v5 = 0xAAAAAAAAAAAAAAABLL * (((char *)a3 - (char *)a1) >> 3);
    v6 = 0xAAAAAAAAAAAAAAABLL * (((char *)a2 - (char *)a1) >> 3);
    if ( v6 == v5 - v6 )
    {
      v21 = a2;
      v22 = v4;
      do
      {
        v23 = v22->m128i_i64[0];
        v24 = _mm_loadu_si128(v21);
        v22 = (const __m128i *)((char *)v22 + 24);
        v21 = (const __m128i *)((char *)v21 + 24);
        v25 = v22[-1].m128i_i64[0];
        *(__m128i *)((char *)v22 - 24) = v24;
        v21[-2].m128i_i64[1] = v23;
        LODWORD(v23) = v21[-1].m128i_i32[2];
        v21[-1].m128i_i64[0] = v25;
        LODWORD(v25) = v22[-1].m128i_i32[2];
        v22[-1].m128i_i32[2] = v23;
        v21[-1].m128i_i32[2] = v25;
      }
      while ( a2 != v22 );
      return (unsigned __int64)&v4[1].m128i_u64[((unsigned __int64)((char *)&a2[-2].m128i_u64[1] - (char *)v4) >> 3) + 1];
    }
    else
    {
      v7 = v5 - v6;
      if ( v6 >= v5 - v6 )
        goto LABEL_12;
      while ( 1 )
      {
        v8 = (__m128i *)((char *)v4 + 24 * v6);
        if ( v7 > 0 )
        {
          v9 = (__int64 *)v4;
          v10 = 0;
          do
          {
            v11 = *v9;
            v12 = v9[1];
            ++v10;
            v9 += 3;
            v13 = _mm_loadu_si128(v8);
            v8 = (const __m128i *)((char *)v8 + 24);
            *(__m128i *)(v9 - 3) = v13;
            v8[-2].m128i_i64[1] = v11;
            LODWORD(v11) = v8[-1].m128i_i32[2];
            v8[-1].m128i_i64[0] = v12;
            LODWORD(v12) = *((_DWORD *)v9 - 2);
            *((_DWORD *)v9 - 2) = v11;
            v8[-1].m128i_i32[2] = v12;
          }
          while ( v7 != v10 );
          v4 = (__m128i *)((char *)v4 + 24 * v7);
        }
        if ( !(v5 % v6) )
          break;
        v7 = v6;
        v6 -= v5 % v6;
        while ( 1 )
        {
          v5 = v7;
          v7 -= v6;
          if ( v6 < v7 )
            break;
LABEL_12:
          v14 = &v4->m128i_i64[3 * v5];
          v4 = (__m128i *)&v14[-3 * v7];
          if ( v6 > 0 )
          {
            v15 = (__m128i *)&v14[-3 * v7];
            v16 = 0;
            do
            {
              v17 = v15[-2].m128i_i64[1];
              v18 = v15[-1].m128i_i64[0];
              ++v16;
              v15 = (__m128i *)((char *)v15 - 24);
              v19 = _mm_loadu_si128((const __m128i *)(v14 - 3));
              v14 -= 3;
              *v15 = v19;
              *v14 = v17;
              LODWORD(v17) = *((_DWORD *)v14 + 4);
              v14[1] = v18;
              LODWORD(v18) = v15[1].m128i_i32[0];
              v15[1].m128i_i32[0] = v17;
              *((_DWORD *)v14 + 4) = v18;
            }
            while ( v6 != v16 );
            v4 = (__m128i *)((char *)v4 - 24 * v6);
          }
          v6 = v5 % v7;
          if ( !(v5 % v7) )
            return (unsigned __int64)v3;
        }
      }
    }
    return (unsigned __int64)v3;
  }
  return (unsigned __int64)a1;
}
