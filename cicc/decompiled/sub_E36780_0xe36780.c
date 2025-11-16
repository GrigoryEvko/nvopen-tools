// Function: sub_E36780
// Address: 0xe36780
//
const __m128i *__fastcall sub_E36780(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r13
  __int64 v8; // rsi
  __int64 v10; // r12
  const __m128i *result; // rax
  __int64 v12; // rdi
  __m128i *v13; // rdx
  int v14; // r14d
  unsigned __int64 v15[5]; // [rsp+8h] [rbp-28h] BYREF

  v7 = a1 + 16;
  v8 = a1 + 16;
  v10 = sub_C8D7D0(a1, a1 + 16, a2, 0x28u, v15, a6);
  result = *(const __m128i **)a1;
  v12 = *(_QWORD *)a1 + 40LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v12 )
  {
    v13 = (__m128i *)v10;
    do
    {
      if ( v13 )
      {
        *v13 = _mm_loadu_si128(result);
        v13[1] = _mm_loadu_si128(result + 1);
        v13[2].m128i_i64[0] = result[2].m128i_i64[0];
      }
      result = (const __m128i *)((char *)result + 40);
      v13 = (__m128i *)((char *)v13 + 40);
    }
    while ( (const __m128i *)v12 != result );
    v12 = *(_QWORD *)a1;
  }
  v14 = v15[0];
  if ( v7 != v12 )
    result = (const __m128i *)_libc_free(v12, v8);
  *(_QWORD *)a1 = v10;
  *(_DWORD *)(a1 + 12) = v14;
  return result;
}
