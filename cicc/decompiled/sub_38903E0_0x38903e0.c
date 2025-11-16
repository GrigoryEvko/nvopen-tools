// Function: sub_38903E0
// Address: 0x38903e0
//
void __fastcall sub_38903E0(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // rax
  __int64 v7; // r13
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // r14
  const __m128i *v10; // rdx
  unsigned __int64 v11; // rcx
  __int64 v12; // rax
  unsigned __int64 v13; // rsi
  const __m128i *v14; // rcx
  unsigned __int64 v15; // r15
  unsigned __int64 v16; // rdi

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
  v8 = *(_QWORD *)a1;
  v9 = *(_QWORD *)a1 + 56LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v9 )
  {
    v10 = (const __m128i *)(v8 + 40);
    v11 = 7 * (((0xDB6DB6DB6DB6DB7LL * ((v9 - v8 - 56) >> 3)) & 0x1FFFFFFFFFFFFFFFLL) + 1);
    v12 = v7;
    v13 = v7 + 8 * v11;
    do
    {
      if ( v12 )
      {
        *(_QWORD *)v12 = v10[-3].m128i_i64[1];
        *(_QWORD *)(v12 + 8) = v10[-2].m128i_i64[0];
        *(_QWORD *)(v12 + 16) = v10[-2].m128i_i64[1];
        *(_QWORD *)(v12 + 24) = v12 + 40;
        v14 = (const __m128i *)v10[-1].m128i_i64[0];
        if ( v14 == v10 )
        {
          *(__m128i *)(v12 + 40) = _mm_loadu_si128(v10);
        }
        else
        {
          *(_QWORD *)(v12 + 24) = v14;
          *(_QWORD *)(v12 + 40) = v10->m128i_i64[0];
        }
        *(_QWORD *)(v12 + 32) = v10[-1].m128i_i64[1];
        v10[-1].m128i_i64[0] = (__int64)v10;
        v10[-1].m128i_i64[1] = 0;
        v10->m128i_i8[0] = 0;
      }
      v12 += 56;
      v10 = (const __m128i *)((char *)v10 + 56);
    }
    while ( v12 != v13 );
    v15 = *(_QWORD *)a1;
    v9 = *(_QWORD *)a1 + 56LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v9 )
    {
      do
      {
        v9 -= 56LL;
        v16 = *(_QWORD *)(v9 + 24);
        if ( v16 != v9 + 40 )
          j_j___libc_free_0(v16);
      }
      while ( v9 != v15 );
      v9 = *(_QWORD *)a1;
    }
  }
  if ( v9 != a1 + 16 )
    _libc_free(v9);
  *(_QWORD *)a1 = v7;
  *(_DWORD *)(a1 + 12) = v5;
}
