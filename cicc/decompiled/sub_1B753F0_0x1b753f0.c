// Function: sub_1B753F0
// Address: 0x1b753f0
//
__m128i *__fastcall sub_1B753F0(__int64 *a1, __int64 a2, __int64 a3, int a4, int a5, int a6)
{
  __int64 v6; // rbx
  __int64 v7; // rax
  __m128i *result; // rax
  __int64 v9; // rdx
  __m128i v10; // [rsp+0h] [rbp-30h] BYREF
  __int64 v11; // [rsp+10h] [rbp-20h]

  v6 = *a1;
  v10.m128i_i64[1] = a2;
  v11 = a3;
  v7 = *(unsigned int *)(v6 + 80);
  v10.m128i_i32[0] = v10.m128i_i32[0] & 0x80000000 | (4 * a4) & 0x7FFFFFFC | 2;
  if ( (unsigned int)v7 >= *(_DWORD *)(v6 + 84) )
  {
    sub_16CD150(v6 + 72, (const void *)(v6 + 88), 0, 24, a5, a6);
    v7 = *(unsigned int *)(v6 + 80);
  }
  result = (__m128i *)(*(_QWORD *)(v6 + 72) + 24 * v7);
  v9 = v11;
  *result = _mm_loadu_si128(&v10);
  result[1].m128i_i64[0] = v9;
  ++*(_DWORD *)(v6 + 80);
  return result;
}
