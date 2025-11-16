// Function: sub_FC7A00
// Address: 0xfc7a00
//
__m128i *__fastcall sub_FC7A00(__int64 *a1, __int64 a2, __int64 a3, int a4, __int64 a5, __int64 a6)
{
  const __m128i *v6; // r12
  __int64 v7; // rbx
  unsigned __int64 v8; // rdx
  __int64 v9; // rax
  unsigned __int64 v10; // r8
  __m128i *result; // rax
  __int64 v12; // rdi
  const void *v13; // rsi
  char *v14; // r12
  unsigned int v15; // [rsp+0h] [rbp-30h] BYREF
  __int64 v16; // [rsp+8h] [rbp-28h]
  __int64 v17; // [rsp+10h] [rbp-20h]

  v6 = (const __m128i *)&v15;
  v7 = *a1;
  v17 = a3;
  v16 = a2;
  v8 = *(_QWORD *)(v7 + 72);
  v9 = *(unsigned int *)(v7 + 80);
  v15 = v15 & 0x80000000 | (4 * a4) & 0x7FFFFFFC | 2;
  v10 = v9 + 1;
  if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(v7 + 84) )
  {
    v12 = v7 + 72;
    v13 = (const void *)(v7 + 88);
    if ( v8 > (unsigned __int64)&v15 || (unsigned __int64)&v15 >= v8 + 24 * v9 )
    {
      sub_C8D5F0(v12, v13, v10, 0x18u, v10, a6);
      v8 = *(_QWORD *)(v7 + 72);
      v9 = *(unsigned int *)(v7 + 80);
    }
    else
    {
      v14 = (char *)&v15 - v8;
      sub_C8D5F0(v12, v13, v10, 0x18u, v10, a6);
      v8 = *(_QWORD *)(v7 + 72);
      v9 = *(unsigned int *)(v7 + 80);
      v6 = (const __m128i *)&v14[v8];
    }
  }
  result = (__m128i *)(v8 + 24 * v9);
  *result = _mm_loadu_si128(v6);
  result[1].m128i_i64[0] = v6[1].m128i_i64[0];
  ++*(_DWORD *)(v7 + 80);
  return result;
}
