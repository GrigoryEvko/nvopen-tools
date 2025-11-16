// Function: sub_1EF9AB0
// Address: 0x1ef9ab0
//
void __fastcall sub_1EF9AB0(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // rax
  __int64 v7; // r13
  const __m128i *v8; // rax
  unsigned __int64 v9; // r14
  __m128i *v10; // rdx
  const __m128i *v11; // r15
  unsigned __int64 v12; // rdi

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
  v7 = malloc(40 * v5);
  if ( !v7 )
    sub_16BD1C0("Allocation failed", 1u);
  v8 = *(const __m128i **)a1;
  v9 = *(_QWORD *)a1 + 40LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v9 )
  {
    v10 = (__m128i *)v7;
    do
    {
      if ( v10 )
      {
        v10->m128i_i64[0] = v8->m128i_i64[0];
        v10->m128i_i32[2] = v8->m128i_i32[2];
        v10->m128i_i32[3] = v8->m128i_i32[3];
        v10[1] = _mm_loadu_si128(v8 + 1);
        v10[2].m128i_i32[0] = v8[2].m128i_i32[0];
        v8[1].m128i_i64[0] = 0;
        v8[1].m128i_i64[1] = 0;
        v8[2].m128i_i32[0] = 0;
      }
      v8 = (const __m128i *)((char *)v8 + 40);
      v10 = (__m128i *)((char *)v10 + 40);
    }
    while ( (const __m128i *)v9 != v8 );
    v11 = *(const __m128i **)a1;
    v9 = *(_QWORD *)a1 + 40LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v9 )
    {
      do
      {
        v12 = *(_QWORD *)(v9 - 24);
        v9 -= 40LL;
        _libc_free(v12);
      }
      while ( (const __m128i *)v9 != v11 );
      v9 = *(_QWORD *)a1;
    }
  }
  if ( v9 != a1 + 16 )
    _libc_free(v9);
  *(_QWORD *)a1 = v7;
  *(_DWORD *)(a1 + 12) = v5;
}
