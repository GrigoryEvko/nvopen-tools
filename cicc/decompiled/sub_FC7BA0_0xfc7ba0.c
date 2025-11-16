// Function: sub_FC7BA0
// Address: 0xfc7ba0
//
__m128i *__fastcall sub_FC7BA0(__int64 *a1, __int64 a2, int a3, __int64 a4, __int64 a5, __int64 a6)
{
  const __m128i *v6; // r12
  __int64 v7; // rbx
  unsigned __int64 v8; // rcx
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // r8
  __m128i *result; // rax
  __int64 v13; // rdi
  const void *v14; // rsi
  char *v15; // r12
  unsigned int v16; // [rsp+0h] [rbp-30h] BYREF
  __int64 v17; // [rsp+8h] [rbp-28h]

  v6 = (const __m128i *)&v16;
  v7 = *a1;
  v17 = a2;
  v8 = *(unsigned int *)(v7 + 84);
  v9 = *(unsigned int *)(v7 + 80);
  v16 = v16 & 0x80000000 | (4 * a3) & 0x7FFFFFFC | 3;
  v10 = *(_QWORD *)(v7 + 72);
  v11 = v9 + 1;
  if ( v9 + 1 > v8 )
  {
    v13 = v7 + 72;
    v14 = (const void *)(v7 + 88);
    if ( v10 > (unsigned __int64)&v16 || (unsigned __int64)&v16 >= v10 + 24 * v9 )
    {
      sub_C8D5F0(v13, v14, v11, 0x18u, v11, a6);
      v10 = *(_QWORD *)(v7 + 72);
      v9 = *(unsigned int *)(v7 + 80);
    }
    else
    {
      v15 = (char *)&v16 - v10;
      sub_C8D5F0(v13, v14, v11, 0x18u, v11, a6);
      v10 = *(_QWORD *)(v7 + 72);
      v9 = *(unsigned int *)(v7 + 80);
      v6 = (const __m128i *)&v15[v10];
    }
  }
  result = (__m128i *)(v10 + 24 * v9);
  *result = _mm_loadu_si128(v6);
  result[1].m128i_i64[0] = v6[1].m128i_i64[0];
  ++*(_DWORD *)(v7 + 80);
  return result;
}
