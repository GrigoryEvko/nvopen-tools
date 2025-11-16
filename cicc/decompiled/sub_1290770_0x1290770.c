// Function: sub_1290770
// Address: 0x1290770
//
__int64 __fastcall sub_1290770(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rbx
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // r12
  const __m128i *v14; // rdx
  __int64 v15; // rax
  const __m128i *v16; // rsi
  __int64 v17; // r15
  __int64 v18; // rdi
  __int64 result; // rax

  v7 = ((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2;
  v8 = ((((v7 | (*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4)
       | v7
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 8)
     | ((v7 | (*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4)
     | v7
     | (*(unsigned int *)(a1 + 12) + 2LL)
     | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1);
  v9 = (v8 | (v8 >> 16) | HIDWORD(v8)) + 1;
  if ( v9 > 0xFFFFFFFF )
    v9 = 0xFFFFFFFFLL;
  v10 = malloc(48 * v9, a2, v7, a4, a5, a6);
  if ( !v10 )
  {
    a2 = 1;
    sub_16BD1C0("Allocation failed");
  }
  v11 = *(_QWORD *)a1;
  v12 = *(_QWORD *)a1 + 48LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v12 )
  {
    v13 = v12 - v11;
    v14 = (const __m128i *)(v11 + 24);
    v15 = v10;
    do
    {
      if ( v15 )
      {
        *(_DWORD *)v15 = v14[-2].m128i_i32[2];
        *(_QWORD *)(v15 + 8) = v15 + 24;
        v16 = (const __m128i *)v14[-1].m128i_i64[0];
        if ( v14 == v16 )
        {
          *(__m128i *)(v15 + 24) = _mm_loadu_si128(v14);
        }
        else
        {
          *(_QWORD *)(v15 + 8) = v16;
          *(_QWORD *)(v15 + 24) = v14->m128i_i64[0];
        }
        *(_QWORD *)(v15 + 16) = v14[-1].m128i_i64[1];
        a2 = v14[1].m128i_u32[0];
        v14[-1].m128i_i64[0] = (__int64)v14;
        v14[-1].m128i_i64[1] = 0;
        v14->m128i_i8[0] = 0;
        *(_DWORD *)(v15 + 40) = a2;
      }
      v15 += 48;
      v14 += 3;
    }
    while ( v10 + 16 * (3 * ((0xAAAAAAAAAAAAAABLL * ((unsigned __int64)(v13 - 48) >> 4)) & 0xFFFFFFFFFFFFFFFLL) + 3) != v15 );
    v17 = *(_QWORD *)a1;
    v12 = *(_QWORD *)a1 + 48LL * *(unsigned int *)(a1 + 8);
    if ( v12 != *(_QWORD *)a1 )
    {
      do
      {
        v12 -= 48;
        v18 = *(_QWORD *)(v12 + 8);
        if ( v18 != v12 + 24 )
        {
          a2 = *(_QWORD *)(v12 + 24) + 1LL;
          j_j___libc_free_0(v18, a2);
        }
      }
      while ( v12 != v17 );
      v12 = *(_QWORD *)a1;
    }
  }
  result = a1 + 16;
  if ( v12 != a1 + 16 )
    result = _libc_free(v12, a2);
  *(_QWORD *)a1 = v10;
  *(_DWORD *)(a1 + 12) = v9;
  return result;
}
