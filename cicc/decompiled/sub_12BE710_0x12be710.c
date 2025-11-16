// Function: sub_12BE710
// Address: 0x12be710
//
__int64 __fastcall sub_12BE710(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // r13
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rbx
  unsigned __int64 v10; // rax
  __int64 v11; // r13
  __int64 v12; // rsi
  const __m128i *v13; // r15
  const __m128i *v14; // rax
  __m128i *v15; // rdx
  const __m128i *v16; // rcx
  const __m128i *v17; // r14
  __int64 result; // rax

  v6 = a2;
  if ( a2 > 0xFFFFFFFF )
  {
    a2 = 1;
    sub_16BD1C0("SmallVector capacity overflow during allocation");
  }
  v7 = ((((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
      | (*(unsigned int *)(a1 + 12) + 2LL)
      | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4;
  v8 = ((v7
       | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 8)
     | v7
     | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
     | (*(unsigned int *)(a1 + 12) + 2LL)
     | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1);
  v9 = v6;
  v10 = (v8 | (v8 >> 16) | HIDWORD(v8)) + 1;
  if ( v10 >= v6 )
    v9 = v10;
  if ( v9 > 0xFFFFFFFF )
    v9 = 0xFFFFFFFFLL;
  v11 = malloc(32 * v9, a2, v7, a4, a5, a6);
  if ( !v11 )
    sub_16BD1C0("Allocation failed");
  v12 = 32LL * *(unsigned int *)(a1 + 8);
  v13 = (const __m128i *)(*(_QWORD *)a1 + v12);
  if ( *(const __m128i **)a1 != v13 )
  {
    v14 = (const __m128i *)(*(_QWORD *)a1 + 16LL);
    v12 += v11;
    v15 = (__m128i *)v11;
    do
    {
      if ( v15 )
      {
        v15->m128i_i64[0] = (__int64)v15[1].m128i_i64;
        v16 = (const __m128i *)v14[-1].m128i_i64[0];
        if ( v16 == v14 )
        {
          v15[1] = _mm_loadu_si128(v14);
        }
        else
        {
          v15->m128i_i64[0] = (__int64)v16;
          v15[1].m128i_i64[0] = v14->m128i_i64[0];
        }
        v15->m128i_i64[1] = v14[-1].m128i_i64[1];
        v14[-1].m128i_i64[0] = (__int64)v14;
        v14[-1].m128i_i64[1] = 0;
        v14->m128i_i8[0] = 0;
      }
      v15 += 2;
      v14 += 2;
    }
    while ( v15 != (__m128i *)v12 );
    v13 = *(const __m128i **)a1;
    v17 = (const __m128i *)(*(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8));
    if ( v17 != *(const __m128i **)a1 )
    {
      do
      {
        v17 -= 2;
        if ( (const __m128i *)v17->m128i_i64[0] != &v17[1] )
        {
          v12 = v17[1].m128i_i64[0] + 1;
          j_j___libc_free_0(v17->m128i_i64[0], v12);
        }
      }
      while ( v13 != v17 );
      v13 = *(const __m128i **)a1;
    }
  }
  result = a1 + 16;
  if ( v13 != (const __m128i *)(a1 + 16) )
    result = _libc_free(v13, v12);
  *(_QWORD *)a1 = v11;
  *(_DWORD *)(a1 + 12) = v9;
  return result;
}
