// Function: sub_E7F1B0
// Address: 0xe7f1b0
//
__int64 __fastcall sub_E7F1B0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r13
  __int64 v8; // rsi
  __int64 v10; // r12
  __int64 result; // rax
  __int64 v12; // r14
  __int64 v13; // r14
  const __m128i *v14; // rdx
  __m128i *v15; // rax
  const __m128i *v16; // rsi
  __int64 v17; // r15
  __int64 v18; // rdi
  int v19; // r15d
  unsigned __int64 v20[7]; // [rsp+8h] [rbp-38h] BYREF

  v7 = a1 + 16;
  v8 = a1 + 16;
  v10 = sub_C8D7D0(a1, a1 + 16, a2, 0x30u, v20, a6);
  result = *(_QWORD *)a1;
  v12 = *(_QWORD *)a1 + 48LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v12 )
  {
    v13 = v12 - result;
    v14 = (const __m128i *)(result + 32);
    v15 = (__m128i *)v10;
    do
    {
      if ( v15 )
      {
        v15->m128i_i32[0] = v14[-2].m128i_i32[0];
        v15->m128i_i32[1] = v14[-2].m128i_i32[1];
        v15->m128i_i32[2] = v14[-2].m128i_i32[2];
        v15[1].m128i_i64[0] = (__int64)v15[2].m128i_i64;
        v16 = (const __m128i *)v14[-1].m128i_i64[0];
        if ( v14 == v16 )
        {
          v15[2] = _mm_loadu_si128(v14);
        }
        else
        {
          v15[1].m128i_i64[0] = (__int64)v16;
          v15[2].m128i_i64[0] = v14->m128i_i64[0];
        }
        v8 = v14[-1].m128i_i64[1];
        v15[1].m128i_i64[1] = v8;
        v14[-1].m128i_i64[0] = (__int64)v14;
        v14[-1].m128i_i64[1] = 0;
        v14->m128i_i8[0] = 0;
      }
      v15 += 3;
      v14 += 3;
    }
    while ( (__m128i *)(v10
                      + 16
                      * (3 * ((0xAAAAAAAAAAAAAABLL * ((unsigned __int64)(v13 - 48) >> 4)) & 0xFFFFFFFFFFFFFFFLL) + 3)) != v15 );
    result = *(unsigned int *)(a1 + 8);
    v17 = *(_QWORD *)a1;
    v12 = *(_QWORD *)a1 + 48 * result;
    if ( *(_QWORD *)a1 != v12 )
    {
      do
      {
        v12 -= 48;
        v18 = *(_QWORD *)(v12 + 16);
        result = v12 + 32;
        if ( v18 != v12 + 32 )
        {
          v8 = *(_QWORD *)(v12 + 32) + 1LL;
          result = j_j___libc_free_0(v18, v8);
        }
      }
      while ( v12 != v17 );
      v12 = *(_QWORD *)a1;
    }
  }
  v19 = v20[0];
  if ( v7 != v12 )
    result = _libc_free(v12, v8);
  *(_QWORD *)a1 = v10;
  *(_DWORD *)(a1 + 12) = v19;
  return result;
}
