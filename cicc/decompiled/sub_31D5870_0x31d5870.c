// Function: sub_31D5870
// Address: 0x31d5870
//
void __fastcall sub_31D5870(__int64 a1, __m128i *a2)
{
  __m128i *v2; // r13
  unsigned int v3; // r15d
  __int64 v4; // r9
  __m128i *v5; // r12
  __int64 v6; // r9
  __m128i v7; // xmm1
  unsigned __int64 v8; // rdx
  __m128i *v9; // rax
  __m128i v10; // xmm0
  __int32 v11; // ecx
  __m128i v12; // xmm2
  size_t v13; // rcx
  size_t v14; // r10
  size_t v15; // rdx
  int v16; // eax
  size_t v17; // [rsp-70h] [rbp-70h]
  size_t v18; // [rsp-68h] [rbp-68h]
  __m128i v19; // [rsp-58h] [rbp-58h] BYREF
  __int64 v20; // [rsp-48h] [rbp-48h]

  if ( (__m128i *)a1 != a2 && a2 != (__m128i *)(a1 + 24) )
  {
    v2 = (__m128i *)(a1 + 48);
    do
    {
      while ( 1 )
      {
        v3 = v2[-1].m128i_u32[2];
        v4 = (__int64)&v2[-2].m128i_i64[1];
        v5 = v2;
        if ( v3 <= *(_DWORD *)(a1 + 16) )
          break;
LABEL_7:
        v6 = v4 - a1;
        v7 = _mm_loadu_si128((__m128i *)((char *)v2 - 24));
        v20 = v2[-1].m128i_i64[1];
        v8 = 0xAAAAAAAAAAAAAAABLL * (v6 >> 3);
        v19 = v7;
        v9 = v2;
        if ( v6 > 0 )
        {
          do
          {
            v10 = _mm_loadu_si128(v9 - 3);
            v11 = v9[-2].m128i_i32[0];
            v9 = (__m128i *)((char *)v9 - 24);
            *v9 = v10;
            v9[1].m128i_i32[0] = v11;
            --v8;
          }
          while ( v8 );
        }
        v12 = _mm_loadu_si128(&v19);
        *(_DWORD *)(a1 + 16) = v3;
        v2 = (__m128i *)((char *)v2 + 24);
        *(__m128i *)a1 = v12;
        if ( a2 == v5 )
          return;
      }
      if ( v3 == *(_DWORD *)(a1 + 16) )
      {
        v13 = v2[-1].m128i_u64[0];
        v14 = *(_QWORD *)(a1 + 8);
        v15 = v13;
        if ( v14 <= v13 )
          v15 = *(_QWORD *)(a1 + 8);
        if ( v15
          && (v17 = v2[-1].m128i_u64[0],
              v18 = *(_QWORD *)(a1 + 8),
              v16 = memcmp((const void *)v2[-2].m128i_i64[1], *(const void **)a1, v15),
              v4 = (__int64)&v2[-2].m128i_i64[1],
              v14 = v18,
              v13 = v17,
              v16) )
        {
          if ( v16 < 0 )
            goto LABEL_7;
        }
        else if ( v14 > v13 )
        {
          goto LABEL_7;
        }
      }
      v2 = (__m128i *)((char *)v2 + 24);
      sub_31D57E0(v4);
    }
    while ( a2 != v5 );
  }
}
