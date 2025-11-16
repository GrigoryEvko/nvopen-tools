// Function: sub_3440410
// Address: 0x3440410
//
void __fastcall sub_3440410(__int64 a1, __int64 a2)
{
  const __m128i *v2; // rcx
  __int64 i; // r9
  unsigned int v5; // edi
  __int64 v6; // rdx
  const __m128i *v7; // rax
  unsigned int v8; // ebx
  __m128i v9; // xmm2
  __int64 v10; // rbx
  unsigned __int64 v11; // rdx
  __m128i v12; // xmm0
  __int32 v13; // r12d
  __m128i v14; // xmm3
  __m128i v15; // xmm4
  __int64 v16; // rdx
  __m128i *v17; // rax
  __m128i v18; // xmm1
  __int32 v19; // edx
  __m128i v20; // xmm5
  __int64 v21; // [rsp-8h] [rbp-8h] BYREF

  if ( a1 != a2 )
  {
    v2 = (const __m128i *)(a1 + 24);
    if ( a2 != a1 + 24 )
    {
      for ( i = a1 + 48; ; i += 24 )
      {
        v5 = v2[1].m128i_u32[0];
        v6 = *(unsigned int *)(a1 + 16);
        v7 = v2;
        if ( v5 > 6 )
          break;
        v8 = dword_44E2140[v5];
        if ( (unsigned int)v6 > 6 )
          break;
        if ( v8 <= dword_44E2140[v6] )
        {
          v15 = _mm_loadu_si128(v2);
          v16 = v2[-1].m128i_u32[2];
          *(&v21 - 4) = v2[1].m128i_i64[0];
          *((__m128i *)&v21 - 3) = v15;
          if ( (unsigned int)v16 > 6 )
            break;
          v17 = (__m128i *)v2;
          while ( v8 > dword_44E2140[v16] )
          {
            v18 = _mm_loadu_si128((__m128i *)((char *)v17 - 24));
            v19 = v17[-1].m128i_i32[2];
            v17 = (__m128i *)((char *)v17 - 24);
            *(__m128i *)((char *)v17 + 24) = v18;
            v17[2].m128i_i32[2] = v19;
            v16 = v17[-1].m128i_u32[2];
            if ( (unsigned int)v16 > 6 )
              goto LABEL_18;
          }
          v20 = _mm_loadu_si128((const __m128i *)&v21 - 3);
          v17[1].m128i_i32[0] = v5;
          v10 = i;
          *v17 = v20;
        }
        else
        {
          v9 = _mm_loadu_si128(v2);
          v10 = i;
          *(&v21 - 4) = v2[1].m128i_i64[0];
          *((__m128i *)&v21 - 3) = v9;
          v11 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v2->m128i_i64 - a1) >> 3);
          if ( (__int64)v2->m128i_i64 - a1 > 0 )
          {
            do
            {
              v12 = _mm_loadu_si128((const __m128i *)((char *)v7 - 24));
              v13 = v7[-1].m128i_i32[2];
              v7 = (const __m128i *)((char *)v7 - 24);
              *(__m128i *)((char *)v7 + 24) = v12;
              v7[2].m128i_i32[2] = v13;
              --v11;
            }
            while ( v11 );
          }
          v14 = _mm_loadu_si128((const __m128i *)&v21 - 3);
          *(_DWORD *)(a1 + 16) = v5;
          *(__m128i *)a1 = v14;
        }
        v2 = (const __m128i *)((char *)v2 + 24);
        if ( a2 == v10 )
          return;
      }
LABEL_18:
      BUG();
    }
  }
}
