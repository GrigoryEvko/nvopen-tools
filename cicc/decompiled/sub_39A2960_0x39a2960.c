// Function: sub_39A2960
// Address: 0x39a2960
//
__m128i *__fastcall sub_39A2960(__int64 a1, __int64 a2, __int64 a3, int a4, int a5, int a6)
{
  __int64 v6; // rax
  __m128i *result; // rax
  __int64 v8; // rdx
  __m128i v9; // [rsp+0h] [rbp-30h] BYREF
  __int64 v10; // [rsp+10h] [rbp-20h]

  v9.m128i_i64[0] = a2;
  v6 = *(unsigned int *)(a1 + 400);
  v9.m128i_i64[1] = a3;
  LODWORD(v10) = a4;
  if ( (unsigned int)v6 >= *(_DWORD *)(a1 + 404) )
  {
    sub_16CD150(a1 + 392, (const void *)(a1 + 408), 0, 24, a5, a6);
    v6 = *(unsigned int *)(a1 + 400);
  }
  result = (__m128i *)(*(_QWORD *)(a1 + 392) + 24 * v6);
  v8 = v10;
  *result = _mm_loadu_si128(&v9);
  result[1].m128i_i64[0] = v8;
  ++*(_DWORD *)(a1 + 400);
  return result;
}
