// Function: sub_2507260
// Address: 0x2507260
//
__m128i *__fastcall sub_2507260(__int64 a1, const __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbp
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rcx
  int v9; // edx
  __m128i *result; // rax
  __m128i v11; // xmm0
  __m128i v12; // [rsp-28h] [rbp-28h] BYREF
  __int64 v13; // [rsp-10h] [rbp-10h]
  __int64 v14; // [rsp-8h] [rbp-8h]

  v7 = *(unsigned int *)(a1 + 8);
  v8 = *(unsigned int *)(a1 + 12);
  if ( v7 >= v8 )
  {
    v11 = _mm_loadu_si128(a2);
    if ( v8 < v7 + 1 )
    {
      v14 = v6;
      v13 = a1;
      v12 = v11;
      sub_C8D5F0(a1, (const void *)(a1 + 16), v7 + 1, 0x10u, a5, a6);
      result = (__m128i *)(*(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8));
      *result = _mm_load_si128(&v12);
    }
    else
    {
      result = (__m128i *)(*(_QWORD *)a1 + 16 * v7);
      *result = v11;
    }
    ++*(_DWORD *)(a1 + 8);
  }
  else
  {
    v9 = *(_DWORD *)(a1 + 8);
    result = (__m128i *)(*(_QWORD *)a1 + 16 * v7);
    if ( result )
    {
      *result = _mm_loadu_si128(a2);
      v9 = *(_DWORD *)(a1 + 8);
    }
    *(_DWORD *)(a1 + 8) = v9 + 1;
  }
  return result;
}
