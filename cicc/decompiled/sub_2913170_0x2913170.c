// Function: sub_2913170
// Address: 0x2913170
//
void __fastcall sub_2913170(__m128i *src, const __m128i *a2)
{
  const __m128i *v2; // rcx
  __m128i v5; // xmm1
  const __m128i *v6; // r14
  __m128i v7; // xmm2
  unsigned __int64 v8; // rdi
  __int64 v9; // rsi
  unsigned __int64 v10; // r8
  __m128i v11; // xmm3
  __m128i *v12; // rax
  __m128i v13; // xmm0
  __int64 v14; // rdx
  __m128i v15; // xmm4
  __m128i v16; // [rsp-48h] [rbp-48h] BYREF
  __int64 v17; // [rsp-38h] [rbp-38h]

  if ( src != a2 )
  {
    v2 = (__m128i *)((char *)src + 24);
    if ( a2 != (const __m128i *)&src[1].m128i_u64[1] )
    {
      while ( 1 )
      {
        v8 = v2->m128i_i64[0];
        if ( v2->m128i_i64[0] < (unsigned __int64)src->m128i_i64[0] )
          goto LABEL_6;
        v9 = (v2[1].m128i_i64[0] >> 2) & 1;
        if ( v8 > src->m128i_i64[0] )
          goto LABEL_11;
        if ( ((src[1].m128i_i64[0] >> 2) & 1) == (_BYTE)v9 )
        {
          v10 = v2->m128i_u64[1];
          if ( v10 <= src->m128i_i64[1] )
            goto LABEL_12;
LABEL_6:
          v5 = _mm_loadu_si128(v2);
          v6 = (const __m128i *)((char *)v2 + 24);
          v17 = v2[1].m128i_i64[0];
          v16 = v5;
          if ( src != v2 )
            memmove(&src[1].m128i_u64[1], src, (char *)v2 - (char *)src);
          v7 = _mm_loadu_si128(&v16);
          v2 = v6;
          src[1].m128i_i64[0] = v17;
          *src = v7;
          if ( a2 == v6 )
            return;
        }
        else
        {
          if ( !(_BYTE)v9 )
            goto LABEL_6;
LABEL_11:
          v10 = v2->m128i_u64[1];
LABEL_12:
          v11 = _mm_loadu_si128(v2);
          v17 = v2[1].m128i_i64[0];
          v12 = (__m128i *)v2;
          v16 = v11;
          while ( 1 )
          {
            if ( v8 < v12[-2].m128i_i64[1] )
              goto LABEL_15;
            if ( v8 > v12[-2].m128i_i64[1] )
              goto LABEL_18;
            if ( ((v12[-1].m128i_i64[1] >> 2) & 1) == (_BYTE)v9 )
              break;
            if ( (_BYTE)v9 )
              goto LABEL_18;
LABEL_15:
            v13 = _mm_loadu_si128((__m128i *)((char *)v12 - 24));
            v14 = v12[-1].m128i_i64[1];
            v12 = (__m128i *)((char *)v12 - 24);
            v12[2].m128i_i64[1] = v14;
            *(__m128i *)((char *)v12 + 24) = v13;
          }
          if ( v12[-1].m128i_i64[0] < v10 )
            goto LABEL_15;
LABEL_18:
          v15 = _mm_loadu_si128(&v16);
          v2 = (const __m128i *)((char *)v2 + 24);
          v12[1].m128i_i64[0] = v17;
          *v12 = v15;
          if ( a2 == v2 )
            return;
        }
      }
    }
  }
}
