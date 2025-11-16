// Function: sub_92FBC0
// Address: 0x92fbc0
//
__int64 __fastcall sub_92FBC0(__int64 a1, __int64 a2)
{
  __int64 v3; // r14
  __int64 v4; // rsi
  __int64 v6; // r13
  __int64 result; // rax
  __int64 v8; // r12
  __int64 v9; // r12
  const __m128i *v10; // rdx
  __int64 v11; // rax
  const __m128i *v12; // rsi
  __int64 v13; // r15
  __int64 v14; // rdi
  int v15; // r15d
  _QWORD v16[7]; // [rsp+8h] [rbp-38h] BYREF

  v3 = a1 + 16;
  v4 = a1 + 16;
  v6 = sub_C8D7D0(a1, a1 + 16, a2, 48, v16);
  result = *(_QWORD *)a1;
  v8 = *(_QWORD *)a1 + 48LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v8 )
  {
    v9 = v8 - result;
    v10 = (const __m128i *)(result + 24);
    v11 = v6;
    do
    {
      if ( v11 )
      {
        *(_DWORD *)v11 = v10[-2].m128i_i32[2];
        *(_QWORD *)(v11 + 8) = v11 + 24;
        v12 = (const __m128i *)v10[-1].m128i_i64[0];
        if ( v10 == v12 )
        {
          *(__m128i *)(v11 + 24) = _mm_loadu_si128(v10);
        }
        else
        {
          *(_QWORD *)(v11 + 8) = v12;
          *(_QWORD *)(v11 + 24) = v10->m128i_i64[0];
        }
        *(_QWORD *)(v11 + 16) = v10[-1].m128i_i64[1];
        v4 = v10[1].m128i_u32[0];
        v10[-1].m128i_i64[0] = (__int64)v10;
        v10[-1].m128i_i64[1] = 0;
        v10->m128i_i8[0] = 0;
        *(_DWORD *)(v11 + 40) = v4;
      }
      v11 += 48;
      v10 += 3;
    }
    while ( v6 + 16 * (3 * ((0xAAAAAAAAAAAAAABLL * ((unsigned __int64)(v9 - 48) >> 4)) & 0xFFFFFFFFFFFFFFFLL) + 3) != v11 );
    result = *(unsigned int *)(a1 + 8);
    v13 = *(_QWORD *)a1;
    v8 = *(_QWORD *)a1 + 48 * result;
    if ( *(_QWORD *)a1 != v8 )
    {
      do
      {
        v8 -= 48;
        v14 = *(_QWORD *)(v8 + 8);
        result = v8 + 24;
        if ( v14 != v8 + 24 )
        {
          v4 = *(_QWORD *)(v8 + 24) + 1LL;
          result = j_j___libc_free_0(v14, v4);
        }
      }
      while ( v8 != v13 );
      v8 = *(_QWORD *)a1;
    }
  }
  v15 = v16[0];
  if ( v3 != v8 )
    result = _libc_free(v8, v4);
  *(_QWORD *)a1 = v6;
  *(_DWORD *)(a1 + 12) = v15;
  return result;
}
