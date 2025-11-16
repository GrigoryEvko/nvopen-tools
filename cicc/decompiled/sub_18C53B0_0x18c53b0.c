// Function: sub_18C53B0
// Address: 0x18c53b0
//
void __fastcall sub_18C53B0(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // rax
  __int64 v7; // r14
  const __m128i *v8; // rcx
  unsigned __int64 v9; // rsi
  const __m128i *v10; // r12
  const __m128i *v11; // rax
  __m128i *v12; // rsi
  __int64 m128i_i64; // rcx
  __m128i *v14; // rdx
  const __m128i *v15; // rdi
  __int64 v16; // rdi
  const __m128i *v17; // r15
  const __m128i *v18; // rdi

  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation", 1u);
  v3 = ((((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
      | (*(unsigned int *)(a1 + 12) + 2LL)
      | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4;
  v4 = ((v3
       | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 8)
     | v3
     | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
     | (*(unsigned int *)(a1 + 12) + 2LL)
     | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1);
  v5 = a2;
  v6 = (v4 | (v4 >> 16) | HIDWORD(v4)) + 1;
  if ( v6 >= a2 )
    v5 = v6;
  if ( v5 > 0xFFFFFFFF )
    v5 = 0xFFFFFFFFLL;
  v7 = malloc(v5 << 6);
  if ( !v7 )
    sub_16BD1C0("Allocation failed", 1u);
  v8 = *(const __m128i **)a1;
  v9 = (unsigned __int64)*(unsigned int *)(a1 + 8) << 6;
  v10 = (const __m128i *)(*(_QWORD *)a1 + v9);
  if ( *(const __m128i **)a1 != v10 )
  {
    v11 = v8 + 3;
    v12 = (__m128i *)(v7 + v9);
    m128i_i64 = (__int64)v8[1].m128i_i64;
    v14 = (__m128i *)v7;
    do
    {
      if ( v14 )
      {
        v14->m128i_i64[0] = (__int64)v14[1].m128i_i64;
        v16 = v11[-3].m128i_i64[0];
        if ( v16 == m128i_i64 )
        {
          v14[1] = _mm_loadu_si128(v11 - 2);
        }
        else
        {
          v14->m128i_i64[0] = v16;
          v14[1].m128i_i64[0] = v11[-2].m128i_i64[0];
        }
        v14->m128i_i64[1] = v11[-3].m128i_i64[1];
        v11[-3].m128i_i64[0] = m128i_i64;
        v11[-3].m128i_i64[1] = 0;
        v11[-2].m128i_i8[0] = 0;
        v14[2].m128i_i64[0] = (__int64)v14[3].m128i_i64;
        v15 = (const __m128i *)v11[-1].m128i_i64[0];
        if ( v15 == v11 )
        {
          v14[3] = _mm_loadu_si128(v11);
        }
        else
        {
          v14[2].m128i_i64[0] = (__int64)v15;
          v14[3].m128i_i64[0] = v11->m128i_i64[0];
        }
        v14[2].m128i_i64[1] = v11[-1].m128i_i64[1];
        v11[-1].m128i_i64[0] = (__int64)v11;
        v11[-1].m128i_i64[1] = 0;
        v11->m128i_i8[0] = 0;
      }
      v14 += 4;
      v11 += 4;
      m128i_i64 += 64;
    }
    while ( v14 != v12 );
    v17 = *(const __m128i **)a1;
    v10 = (const __m128i *)(*(_QWORD *)a1 + ((unsigned __int64)*(unsigned int *)(a1 + 8) << 6));
    if ( *(const __m128i **)a1 != v10 )
    {
      do
      {
        v10 -= 4;
        v18 = (const __m128i *)v10[2].m128i_i64[0];
        if ( v18 != &v10[3] )
          j_j___libc_free_0(v18, v10[3].m128i_i64[0] + 1);
        if ( (const __m128i *)v10->m128i_i64[0] != &v10[1] )
          j_j___libc_free_0(v10->m128i_i64[0], v10[1].m128i_i64[0] + 1);
      }
      while ( v10 != v17 );
      v10 = *(const __m128i **)a1;
    }
  }
  if ( v10 != (const __m128i *)(a1 + 16) )
    _libc_free((unsigned __int64)v10);
  *(_QWORD *)a1 = v7;
  *(_DWORD *)(a1 + 12) = v5;
}
