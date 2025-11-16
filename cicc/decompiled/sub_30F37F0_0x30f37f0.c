// Function: sub_30F37F0
// Address: 0x30f37f0
//
unsigned __int64 __fastcall sub_30F37F0(const __m128i *a1, const __m128i *a2, const __m128i *a3)
{
  __int64 v3; // rbx
  __int64 v4; // rbp
  const __m128i *v5; // r11
  const __m128i *v6; // r10
  __int64 v7; // r11
  signed __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // r8
  __int64 *v11; // rcx
  const __m128i *v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // r9
  __int64 v15; // rbx
  __m128i v16; // xmm0
  __int64 *v17; // rcx
  __int64 *v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // r9
  __int64 v21; // rbx
  __m128i v22; // xmm0
  __int64 *v24; // rdx
  const __m128i *v25; // rax
  __int64 v26; // rcx
  __int64 v27; // rdi
  __m128i v28; // xmm0
  __int64 v29; // [rsp-10h] [rbp-10h]
  __int64 v30; // [rsp-8h] [rbp-8h]

  v5 = a3;
  if ( a1 == a2 )
    return (unsigned __int64)v5;
  v6 = a1;
  if ( a2 == a3 )
    return (unsigned __int64)a1;
  v30 = v4;
  v7 = (__int64)a1->m128i_i64 + (char *)a3 - (char *)a2;
  v8 = 0xAAAAAAAAAAAAAAABLL * (((char *)a3 - (char *)a1) >> 3);
  v29 = v3;
  v9 = 0xAAAAAAAAAAAAAAABLL * (((char *)a2 - (char *)a1) >> 3);
  if ( v9 == v8 - v9 )
  {
    v24 = (__int64 *)a2;
    v25 = v6;
    do
    {
      v26 = v25->m128i_i64[0];
      v27 = *v24;
      v25 = (const __m128i *)((char *)v25 + 24);
      v24 += 3;
      v25[-2].m128i_i64[1] = v27;
      *(v24 - 3) = v26;
      v28 = _mm_loadu_si128(v25 - 1);
      v25[-1].m128i_i64[0] = *(v24 - 2);
      v25[-1].m128i_i32[2] = *((_DWORD *)v24 - 2);
      *((__m128i *)&v30 - 2) = v28;
      *(v24 - 2) = v28.m128i_i64[0];
      *((_DWORD *)v24 - 2) = *((_DWORD *)&v30 - 6);
    }
    while ( a2 != v25 );
    return (unsigned __int64)&v6[1].m128i_u64[((unsigned __int64)((char *)&a2[-2].m128i_u64[1] - (char *)v6) >> 3) + 1];
  }
  else
  {
    v10 = v8 - v9;
    if ( v9 >= v8 - v9 )
      goto LABEL_12;
    while ( 1 )
    {
      v11 = &v6->m128i_i64[3 * v9];
      if ( v10 > 0 )
      {
        v12 = v6;
        v13 = 0;
        do
        {
          v14 = v12->m128i_i64[0];
          v15 = *v11;
          ++v13;
          v12 = (const __m128i *)((char *)v12 + 24);
          v11 += 3;
          v12[-2].m128i_i64[1] = v15;
          *(v11 - 3) = v14;
          v16 = _mm_loadu_si128(v12 - 1);
          v12[-1].m128i_i64[0] = *(v11 - 2);
          v12[-1].m128i_i32[2] = *((_DWORD *)v11 - 2);
          *((__m128i *)&v30 - 3) = v16;
          *(v11 - 2) = v16.m128i_i64[0];
          *((_DWORD *)v11 - 2) = *((_DWORD *)&v30 - 10);
        }
        while ( v10 != v13 );
        v6 = (const __m128i *)((char *)v6 + 24 * v10);
      }
      if ( !(v8 % v9) )
        break;
      v10 = v9;
      v9 -= v8 % v9;
      while ( 1 )
      {
        v8 = v10;
        v10 -= v9;
        if ( v9 < v10 )
          break;
LABEL_12:
        v17 = &v6->m128i_i64[3 * v8];
        v6 = (const __m128i *)&v17[-3 * v10];
        if ( v9 > 0 )
        {
          v18 = &v17[-3 * v10];
          v19 = 0;
          do
          {
            v20 = *(v18 - 3);
            v21 = *(v17 - 3);
            ++v19;
            v17 -= 3;
            v18 -= 3;
            *v18 = v21;
            *v17 = v20;
            v22 = _mm_loadu_si128((const __m128i *)(v18 + 1));
            v18[1] = v17[1];
            *((_DWORD *)v18 + 4) = *((_DWORD *)v17 + 4);
            *((__m128i *)&v30 - 4) = v22;
            v17[1] = v22.m128i_i64[0];
            *((_DWORD *)v17 + 4) = *((_DWORD *)&v30 - 14);
          }
          while ( v9 != v19 );
          v6 = (const __m128i *)((char *)v6 - 24 * v9);
        }
        v9 = v8 % v10;
        if ( !(v8 % v10) )
          return v7;
      }
    }
  }
  return v7;
}
