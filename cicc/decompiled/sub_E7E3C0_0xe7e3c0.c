// Function: sub_E7E3C0
// Address: 0xe7e3c0
//
__m128i *__fastcall sub_E7E3C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rax
  unsigned __int64 v10; // rcx
  unsigned __int64 v11; // rdx
  __int64 v12; // rcx
  const __m128i *v13; // rdx
  __m128i *result; // rax
  unsigned __int64 v15; // r13
  __int64 v16; // rdi
  const void *v17; // rsi
  _QWORD v18[8]; // [rsp+0h] [rbp-40h] BYREF

  v18[0] = a2;
  v6 = sub_E7DDE0(a1);
  v9 = *(unsigned int *)(v6 + 96);
  v18[1] = a3;
  v10 = *(unsigned int *)(v6 + 100);
  v18[2] = a4;
  v11 = v9 + 1;
  if ( v9 + 1 > v10 )
  {
    v15 = *(_QWORD *)(v6 + 88);
    v16 = v6 + 88;
    v17 = (const void *)(v6 + 104);
    if ( v15 > (unsigned __int64)v18 || (unsigned __int64)v18 >= v15 + 24 * v9 )
    {
      sub_C8D5F0(v16, v17, v11, 0x18u, v7, v8);
      v12 = *(_QWORD *)(v6 + 88);
      v9 = *(unsigned int *)(v6 + 96);
      v13 = (const __m128i *)v18;
    }
    else
    {
      sub_C8D5F0(v16, v17, v11, 0x18u, v7, v8);
      v12 = *(_QWORD *)(v6 + 88);
      v9 = *(unsigned int *)(v6 + 96);
      v13 = (const __m128i *)((char *)v18 + v12 - v15);
    }
  }
  else
  {
    v12 = *(_QWORD *)(v6 + 88);
    v13 = (const __m128i *)v18;
  }
  result = (__m128i *)(v12 + 24 * v9);
  *result = _mm_loadu_si128(v13);
  result[1].m128i_i64[0] = v13[1].m128i_i64[0];
  ++*(_DWORD *)(v6 + 96);
  return result;
}
