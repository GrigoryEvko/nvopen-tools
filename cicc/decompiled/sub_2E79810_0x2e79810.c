// Function: sub_2E79810
// Address: 0x2e79810
//
__m128i *__fastcall sub_2E79810(__int64 a1, __int64 a2, __int64 a3, int a4, __int64 a5, __int64 a6)
{
  const __m128i *v6; // r12
  __int64 v8; // rax
  unsigned __int64 v9; // rcx
  unsigned __int64 v10; // r8
  unsigned __int64 v11; // rdx
  __m128i *result; // rax
  __int64 v13; // rdi
  const void *v14; // rsi
  char *v15; // r12
  _QWORD v16[2]; // [rsp+0h] [rbp-30h] BYREF
  int v17; // [rsp+10h] [rbp-20h]

  v6 = (const __m128i *)v16;
  v8 = *(unsigned int *)(a1 + 912);
  v17 = a4;
  v9 = *(unsigned int *)(a1 + 916);
  v16[1] = a3;
  v10 = v8 + 1;
  v16[0] = a2;
  v11 = *(_QWORD *)(a1 + 904);
  if ( v8 + 1 > v9 )
  {
    v13 = a1 + 904;
    v14 = (const void *)(a1 + 920);
    if ( v11 > (unsigned __int64)v16 || (unsigned __int64)v16 >= v11 + 20 * v8 )
    {
      sub_C8D5F0(v13, v14, v10, 0x14u, v10, a6);
      v11 = *(_QWORD *)(a1 + 904);
      v8 = *(unsigned int *)(a1 + 912);
    }
    else
    {
      v15 = (char *)v16 - v11;
      sub_C8D5F0(v13, v14, v10, 0x14u, v10, a6);
      v11 = *(_QWORD *)(a1 + 904);
      v8 = *(unsigned int *)(a1 + 912);
      v6 = (const __m128i *)&v15[v11];
    }
  }
  result = (__m128i *)(v11 + 20 * v8);
  *result = _mm_loadu_si128(v6);
  result[1].m128i_i32[0] = v6[1].m128i_i32[0];
  ++*(_DWORD *)(a1 + 912);
  return result;
}
