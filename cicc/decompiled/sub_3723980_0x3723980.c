// Function: sub_3723980
// Address: 0x3723980
//
__m128i *__fastcall sub_3723980(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  const __m128i *v6; // r12
  __int64 v8; // rax
  unsigned __int64 v9; // rcx
  unsigned __int64 v10; // rdx
  __int64 v11; // rax
  unsigned __int64 v12; // r8
  __m128i *result; // rax
  __int64 v14; // rdi
  const void *v15; // rsi
  char *v16; // r12
  __int64 v17; // [rsp+0h] [rbp-30h] BYREF
  char v18; // [rsp+8h] [rbp-28h]
  int v19; // [rsp+10h] [rbp-20h]

  v6 = (const __m128i *)&v17;
  v8 = *(_QWORD *)(a2 + 392);
  v18 = 1;
  v9 = *(unsigned int *)(a1 + 220);
  v10 = *(_QWORD *)(a1 + 208);
  v17 = v8;
  v19 = *(_DWORD *)(a2 + 72);
  v11 = *(unsigned int *)(a1 + 216);
  v12 = v11 + 1;
  if ( v11 + 1 > v9 )
  {
    v14 = a1 + 208;
    v15 = (const void *)(a1 + 224);
    if ( v10 > (unsigned __int64)&v17 || (unsigned __int64)&v17 >= v10 + 24 * v11 )
    {
      sub_C8D5F0(v14, v15, v12, 0x18u, v12, a6);
      v10 = *(_QWORD *)(a1 + 208);
      v11 = *(unsigned int *)(a1 + 216);
    }
    else
    {
      v16 = (char *)&v17 - v10;
      sub_C8D5F0(v14, v15, v12, 0x18u, v12, a6);
      v10 = *(_QWORD *)(a1 + 208);
      v11 = *(unsigned int *)(a1 + 216);
      v6 = (const __m128i *)&v16[v10];
    }
  }
  result = (__m128i *)(v10 + 24 * v11);
  *result = _mm_loadu_si128(v6);
  result[1].m128i_i64[0] = v6[1].m128i_i64[0];
  ++*(_DWORD *)(a1 + 216);
  return result;
}
