// Function: sub_24FDF00
// Address: 0x24fdf00
//
void __fastcall sub_24FDF00(const __m128i *a1, const __m128i *a2)
{
  __int64 v2; // rbp
  const __m128i *v3; // r9
  const __m128i *v4; // rsi
  __m128i *v5; // r8
  __int64 v6; // rcx
  __int64 *v7; // rdi
  __m128i v8; // xmm2
  __int64 v9; // rax
  __int64 v10; // rdx
  __m128i v11; // xmm0
  __int64 v12; // rdx
  __int64 v13; // rax
  __m128i v14; // xmm3
  __m128i v15; // [rsp-28h] [rbp-28h] BYREF
  __m128i v16; // [rsp-18h] [rbp-18h]
  __int64 v17; // [rsp-8h] [rbp-8h]

  if ( a1 != a2 )
  {
    v3 = a2;
    v4 = a1 + 2;
    v5 = (__m128i *)a1;
    if ( v3 != &a1[2] )
    {
      v17 = v2;
      do
      {
        v6 = v4->m128i_i64[0];
        v7 = (__int64 *)v4;
        v4 += 2;
        if ( v6 >= v5->m128i_i64[0] )
        {
          sub_24FD750(v7);
        }
        else
        {
          v8 = _mm_loadu_si128(v4 - 1);
          v15 = _mm_loadu_si128(v4 - 2);
          v16 = v8;
          v9 = ((char *)v7 - (char *)v5) >> 5;
          if ( (char *)v7 - (char *)v5 > 0 )
          {
            do
            {
              v10 = *(v7 - 4);
              v11 = _mm_loadu_si128((const __m128i *)(v7 - 3));
              v7 -= 4;
              v7[4] = v10;
              v12 = v7[3];
              *(__m128i *)(v7 + 5) = v11;
              v7[7] = v12;
              --v9;
            }
            while ( v9 );
          }
          v13 = v16.m128i_i64[1];
          v14 = _mm_loadu_si128((const __m128i *)&v15.m128i_u64[1]);
          v5->m128i_i64[0] = v6;
          v5[1].m128i_i64[1] = v13;
          *(__m128i *)((char *)v5 + 8) = v14;
        }
      }
      while ( v3 != v4 );
    }
  }
}
