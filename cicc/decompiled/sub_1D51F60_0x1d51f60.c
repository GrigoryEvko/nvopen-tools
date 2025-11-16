// Function: sub_1D51F60
// Address: 0x1d51f60
//
void __fastcall sub_1D51F60(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rbx
  unsigned __int64 v5; // rax
  __int64 v6; // r13
  unsigned __int64 v7; // rdi
  const __m128i *v8; // rdx
  __int64 v9; // rsi
  __m128i *v10; // rax

  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation", 1u);
  v3 = (((((((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
         | (*(unsigned int *)(a1 + 12) + 2LL)
         | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4)
       | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 8)
     | (((((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4)
     | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
     | (*(unsigned int *)(a1 + 12) + 2LL)
     | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1);
  v4 = a2;
  v5 = (v3 | (v3 >> 16) | HIDWORD(v3)) + 1;
  if ( v5 >= a2 )
    v4 = v5;
  if ( v4 > 0xFFFFFFFF )
    v4 = 0xFFFFFFFFLL;
  v6 = malloc(56 * v4);
  if ( !v6 )
    sub_16BD1C0("Allocation failed", 1u);
  v7 = *(_QWORD *)a1;
  v8 = *(const __m128i **)a1;
  v9 = *(_QWORD *)a1 + 56LL * *(unsigned int *)(a1 + 8);
  v10 = (__m128i *)v6;
  if ( *(_QWORD *)a1 != v9 )
  {
    do
    {
      if ( v10 )
      {
        *v10 = _mm_loadu_si128(v8);
        v10[1] = _mm_loadu_si128(v8 + 1);
        v10[2].m128i_i64[1] = v8[2].m128i_i64[1];
        v10[3].m128i_i8[0] = v8[3].m128i_i8[0];
        v10[2].m128i_i64[0] = (__int64)&unk_49F9A38;
      }
      v8 = (const __m128i *)((char *)v8 + 56);
      v10 = (__m128i *)((char *)v10 + 56);
    }
    while ( (const __m128i *)v9 != v8 );
  }
  if ( v7 != a1 + 16 )
    _libc_free(v7);
  *(_QWORD *)a1 = v6;
  *(_DWORD *)(a1 + 12) = v4;
}
