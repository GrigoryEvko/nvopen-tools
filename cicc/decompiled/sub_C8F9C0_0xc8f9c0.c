// Function: sub_C8F9C0
// Address: 0xc8f9c0
//
__int64 __fastcall sub_C8F9C0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r13
  __int64 v8; // rsi
  __int64 v10; // rax
  const __m128i *v11; // rdx
  __int64 v12; // r12
  __int64 result; // rax
  __int64 v14; // r14
  __int64 v15; // r14
  const __m128i *v16; // rax
  __m128i *v17; // rdx
  __m128i v18; // xmm0
  const __m128i *v19; // rcx
  const __m128i *v20; // r15
  __int64 v21; // rdi
  int v22; // r15d
  unsigned __int64 v23[7]; // [rsp+8h] [rbp-38h] BYREF

  v7 = a1 + 16;
  v8 = a1 + 16;
  v10 = sub_C8D7D0(a1, a1 + 16, a2, 0x30u, v23, a6);
  v11 = *(const __m128i **)a1;
  v12 = v10;
  result = *(unsigned int *)(a1 + 8);
  v14 = *(_QWORD *)a1 + 48 * result;
  if ( *(_QWORD *)a1 != v14 )
  {
    v15 = v14 - (_QWORD)v11;
    v16 = v11 + 2;
    v17 = (__m128i *)v12;
    v8 = v12 + 16 * (3 * ((0xAAAAAAAAAAAAAABLL * ((unsigned __int64)(v15 - 48) >> 4)) & 0xFFFFFFFFFFFFFFFLL) + 3);
    do
    {
      if ( v17 )
      {
        v18 = _mm_loadu_si128(v16 - 2);
        v17[1].m128i_i64[0] = (__int64)v17[2].m128i_i64;
        *v17 = v18;
        v19 = (const __m128i *)v16[-1].m128i_i64[0];
        if ( v19 == v16 )
        {
          v17[2] = _mm_loadu_si128(v16);
        }
        else
        {
          v17[1].m128i_i64[0] = (__int64)v19;
          v17[2].m128i_i64[0] = v16->m128i_i64[0];
        }
        v17[1].m128i_i64[1] = v16[-1].m128i_i64[1];
        v16[-1].m128i_i64[0] = (__int64)v16;
        v16[-1].m128i_i64[1] = 0;
        v16->m128i_i8[0] = 0;
      }
      v17 += 3;
      v16 += 3;
    }
    while ( v17 != (__m128i *)v8 );
    result = *(unsigned int *)(a1 + 8);
    v20 = *(const __m128i **)a1;
    v14 = *(_QWORD *)a1 + 48 * result;
    if ( *(_QWORD *)a1 != v14 )
    {
      do
      {
        v14 -= 48;
        v21 = *(_QWORD *)(v14 + 16);
        result = v14 + 32;
        if ( v21 != v14 + 32 )
        {
          v8 = *(_QWORD *)(v14 + 32) + 1LL;
          result = j_j___libc_free_0(v21, v8);
        }
      }
      while ( v20 != (const __m128i *)v14 );
      v14 = *(_QWORD *)a1;
    }
  }
  v22 = v23[0];
  if ( v7 != v14 )
    result = _libc_free(v14, v8);
  *(_QWORD *)a1 = v12;
  *(_DWORD *)(a1 + 12) = v22;
  return result;
}
