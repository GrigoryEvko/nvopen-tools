// Function: sub_396B9E0
// Address: 0x396b9e0
//
unsigned __int64 __fastcall sub_396B9E0(const __m128i *a1, const __m128i *a2, const __m128i *a3)
{
  const __m128i *v3; // r12
  __m128i *v4; // r11
  signed __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // r8
  const __m128i *v8; // rcx
  __int32 *v9; // rsi
  __int64 v10; // rdx
  __m128i v11; // xmm0
  __int32 v12; // ebx
  __int64 v13; // r10
  __int64 v14; // r9
  __int32 *v15; // rcx
  __m128i *v16; // rsi
  __int64 v17; // rdx
  __m128i v18; // xmm1
  __int32 v19; // ebx
  __int64 v20; // r10
  __int64 v21; // r9
  const __m128i *v23; // rdx
  const __m128i *v24; // rax
  __m128i v25; // xmm2
  __int32 v26; // r8d
  __int64 v27; // rdi
  __int64 v28; // rcx

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
      v23 = a2;
      v24 = v4;
      do
      {
        v25 = _mm_loadu_si128(v23);
        v26 = v24->m128i_i32[0];
        v24 = (const __m128i *)((char *)v24 + 24);
        v23 = (const __m128i *)((char *)v23 + 24);
        v27 = v24[-1].m128i_i64[0];
        v28 = v24[-1].m128i_i64[1];
        *(__m128i *)((char *)v24 - 24) = v25;
        v24[-1].m128i_i64[1] = v23[-1].m128i_i64[1];
        v23[-2].m128i_i32[2] = v26;
        v23[-1].m128i_i64[0] = v27;
        v23[-1].m128i_i64[1] = v28;
      }
      while ( a2 != v24 );
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
          v9 = (__int32 *)v4;
          v10 = 0;
          do
          {
            v11 = _mm_loadu_si128(v8);
            v12 = *v9;
            ++v10;
            v9 += 6;
            v13 = *((_QWORD *)v9 - 2);
            v14 = *((_QWORD *)v9 - 1);
            v8 = (const __m128i *)((char *)v8 + 24);
            *(__m128i *)(v9 - 6) = v11;
            *((_QWORD *)v9 - 1) = v8[-1].m128i_i64[1];
            v8[-2].m128i_i32[2] = v12;
            v8[-1].m128i_i64[0] = v13;
            v8[-1].m128i_i64[1] = v14;
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
          v15 = &v4->m128i_i32[6 * v5];
          v4 = (__m128i *)&v15[-6 * v7];
          if ( v6 > 0 )
          {
            v16 = (__m128i *)&v15[-6 * v7];
            v17 = 0;
            do
            {
              v18 = _mm_loadu_si128((const __m128i *)(v15 - 6));
              v19 = v16[-2].m128i_i32[2];
              ++v17;
              v16 = (__m128i *)((char *)v16 - 24);
              v20 = v16->m128i_i64[1];
              v21 = v16[1].m128i_i64[0];
              v15 -= 6;
              *v16 = v18;
              v16[1].m128i_i64[0] = *((_QWORD *)v15 + 2);
              *v15 = v19;
              *((_QWORD *)v15 + 1) = v20;
              *((_QWORD *)v15 + 2) = v21;
            }
            while ( v6 != v17 );
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
