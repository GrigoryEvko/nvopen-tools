// Function: sub_1740340
// Address: 0x1740340
//
void __fastcall sub_1740340(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // rax
  __int64 v7; // r13
  const __m128i *v8; // rdx
  const __m128i *v9; // r14
  const __m128i *v10; // rax
  unsigned __int64 v11; // rcx
  __m128i *v12; // rdx
  __m128i *v13; // rsi
  __int64 v14; // rcx
  const __m128i *v15; // rcx
  const __m128i *v16; // r15
  __int64 v17; // rdi

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
  v7 = malloc(56 * v5);
  if ( !v7 )
    sub_16BD1C0("Allocation failed", 1u);
  v8 = *(const __m128i **)a1;
  v9 = (const __m128i *)(*(_QWORD *)a1 + 56LL * *(unsigned int *)(a1 + 8));
  if ( *(const __m128i **)a1 != v9 )
  {
    v10 = v8 + 1;
    v11 = 7
        * (((0xDB6DB6DB6DB6DB7LL * ((unsigned __int64)((char *)v9 - (char *)v8 - 56) >> 3)) & 0x1FFFFFFFFFFFFFFFLL) + 1);
    v12 = (__m128i *)v7;
    v13 = (__m128i *)(v7 + 8 * v11);
    do
    {
      if ( v12 )
      {
        v12->m128i_i64[0] = (__int64)v12[1].m128i_i64;
        v15 = (const __m128i *)v10[-1].m128i_i64[0];
        if ( v15 == v10 )
        {
          v12[1] = _mm_loadu_si128(v10);
        }
        else
        {
          v12->m128i_i64[0] = (__int64)v15;
          v12[1].m128i_i64[0] = v10->m128i_i64[0];
        }
        v12->m128i_i64[1] = v10[-1].m128i_i64[1];
        v14 = v10[1].m128i_i64[0];
        v10[-1].m128i_i64[0] = (__int64)v10;
        v10[-1].m128i_i64[1] = 0;
        v10->m128i_i8[0] = 0;
        v12[2].m128i_i64[0] = v14;
        v12[2].m128i_i64[1] = v10[1].m128i_i64[1];
        v12[3].m128i_i64[0] = v10[2].m128i_i64[0];
        v10[2].m128i_i64[0] = 0;
        v10[1].m128i_i64[1] = 0;
        v10[1].m128i_i64[0] = 0;
      }
      v12 = (__m128i *)((char *)v12 + 56);
      v10 = (const __m128i *)((char *)v10 + 56);
    }
    while ( v12 != v13 );
    v16 = *(const __m128i **)a1;
    v9 = (const __m128i *)(*(_QWORD *)a1 + 56LL * *(unsigned int *)(a1 + 8));
    if ( *(const __m128i **)a1 != v9 )
    {
      do
      {
        v17 = v9[-2].m128i_i64[1];
        v9 = (const __m128i *)((char *)v9 - 56);
        if ( v17 )
          j_j___libc_free_0(v17, v9[3].m128i_i64[0] - v17);
        if ( (const __m128i *)v9->m128i_i64[0] != &v9[1] )
          j_j___libc_free_0(v9->m128i_i64[0], v9[1].m128i_i64[0] + 1);
      }
      while ( v9 != v16 );
      v9 = *(const __m128i **)a1;
    }
  }
  if ( v9 != (const __m128i *)(a1 + 16) )
    _libc_free((unsigned __int64)v9);
  *(_QWORD *)a1 = v7;
  *(_DWORD *)(a1 + 12) = v5;
}
