// Function: sub_E7E2A0
// Address: 0xe7e2a0
//
__m128i *__fastcall sub_E7E2A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  __int64 v8; // rbx
  __int64 v9; // r9
  _QWORD *v10; // rax
  __int64 v11; // rax
  unsigned __int64 v12; // rcx
  const __m128i *v13; // r12
  unsigned __int64 v14; // r8
  unsigned __int64 v15; // rdx
  __m128i *result; // rax
  __int64 v17; // rdi
  const void *v18; // rsi
  char *v19; // r12
  _QWORD v20[4]; // [rsp+10h] [rbp-60h] BYREF
  char v21; // [rsp+30h] [rbp-40h]

  v8 = sub_E7DDE0(a1);
  v10 = *(_QWORD **)(a1 + 264);
  if ( v10 )
    v10 = (_QWORD *)*v10;
  v20[0] = v10;
  v11 = *(unsigned int *)(v8 + 216);
  v12 = *(unsigned int *)(v8 + 220);
  v20[2] = a3;
  v13 = (const __m128i *)v20;
  v14 = v11 + 1;
  v20[1] = a2;
  v15 = *(_QWORD *)(v8 + 208);
  v20[3] = a4;
  v21 = a5;
  if ( v11 + 1 > v12 )
  {
    v17 = v8 + 208;
    v18 = (const void *)(v8 + 224);
    if ( v15 > (unsigned __int64)v20 || (unsigned __int64)v20 >= v15 + 40 * v11 )
    {
      sub_C8D5F0(v17, v18, v14, 0x28u, v14, v9);
      v15 = *(_QWORD *)(v8 + 208);
      v11 = *(unsigned int *)(v8 + 216);
    }
    else
    {
      v19 = (char *)v20 - v15;
      sub_C8D5F0(v17, v18, v14, 0x28u, v14, v9);
      v15 = *(_QWORD *)(v8 + 208);
      v11 = *(unsigned int *)(v8 + 216);
      v13 = (const __m128i *)&v19[v15];
    }
  }
  result = (__m128i *)(v15 + 40 * v11);
  *result = _mm_loadu_si128(v13);
  result[1] = _mm_loadu_si128(v13 + 1);
  result[2].m128i_i64[0] = v13[2].m128i_i64[0];
  ++*(_DWORD *)(v8 + 216);
  return result;
}
