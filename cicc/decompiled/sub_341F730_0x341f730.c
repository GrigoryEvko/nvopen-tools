// Function: sub_341F730
// Address: 0x341f730
//
__m128i *__fastcall sub_341F730(__int64 **a1, const __m128i *a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 *v7; // r12
  __m128i v8; // xmm0
  __m128i *result; // rax
  __int64 v10; // rdi
  __int64 v11; // rcx
  char v12; // dl
  __int64 v13; // rdx
  const __m128i *v14; // r13
  const __m128i *v15; // rbx
  __int64 v16; // rax
  bool v17; // zf
  __int64 v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rdi
  __m128i v21; // [rsp+0h] [rbp-40h] BYREF
  _OWORD v22[3]; // [rsp+10h] [rbp-30h] BYREF

  v6 = a2->m128i_i64[0];
  v7 = *a1;
  v8 = _mm_loadu_si128(a2);
  result = (__m128i *)(*(_QWORD *)(a2->m128i_i64[0] + 48) + 16LL * a2->m128i_u32[2]);
  if ( result->m128i_i16[0] != 1 || *(_DWORD *)(v6 + 24) == 1 )
    return result;
  v10 = *v7;
  if ( !*(_BYTE *)(*v7 + 28) )
    goto LABEL_9;
  result = *(__m128i **)(v10 + 8);
  a4 = *(unsigned int *)(v10 + 20);
  a3 = &result->m128i_i64[a4];
  if ( result == (__m128i *)a3 )
  {
LABEL_16:
    if ( (unsigned int)a4 < *(_DWORD *)(v10 + 16) )
    {
      v11 = (unsigned int)(a4 + 1);
      *(_DWORD *)(v10 + 20) = v11;
      *a3 = v6;
      ++*(_QWORD *)v10;
LABEL_10:
      if ( *(_DWORD *)(v6 + 24) == 2 )
      {
        result = *(__m128i **)(v6 + 40);
        v13 = 5LL * *(unsigned int *)(v6 + 64);
        v14 = (__m128i *)((char *)result + 40 * *(unsigned int *)(v6 + 64));
        if ( result != v14 )
        {
          v15 = *(const __m128i **)(v6 + 40);
          do
          {
            v16 = v7[1];
            v17 = *(_QWORD *)(v16 + 16) == 0;
            v22[0] = _mm_loadu_si128(v15);
            if ( v17 )
              sub_4263D6(v10, a2, v13);
            v15 = (const __m128i *)((char *)v15 + 40);
            a2 = (const __m128i *)v22;
            v10 = v16;
            result = (__m128i *)(*(__int64 (__fastcall **)(__int64, _OWORD *, __int64, __int64, __int64, __int64, double, __int64, __int64))(v16 + 24))(
                                  v16,
                                  v22,
                                  v13,
                                  v11,
                                  a5,
                                  a6,
                                  *(double *)v8.m128i_i64,
                                  v21.m128i_i64[0],
                                  v21.m128i_i64[1]);
          }
          while ( v14 != v15 );
        }
      }
      else
      {
        v18 = v7[2];
        v19 = *(unsigned int *)(v18 + 8);
        if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(v18 + 12) )
        {
          v20 = v7[2];
          v21 = v8;
          sub_C8D5F0(v20, (const void *)(v18 + 16), v19 + 1, 0x10u, a5, a6);
          v19 = *(unsigned int *)(v18 + 8);
          v8 = _mm_load_si128(&v21);
        }
        result = (__m128i *)(*(_QWORD *)v18 + 16 * v19);
        *result = v8;
        ++*(_DWORD *)(v18 + 8);
      }
      return result;
    }
LABEL_9:
    a2 = (const __m128i *)a2->m128i_i64[0];
    v21 = v8;
    result = (__m128i *)sub_C8CC70(v10, (__int64)a2, (__int64)a3, a4, a5, a6);
    v8 = _mm_load_si128(&v21);
    if ( !v12 )
      return result;
    goto LABEL_10;
  }
  while ( v6 != result->m128i_i64[0] )
  {
    result = (__m128i *)((char *)result + 8);
    if ( a3 == (__int64 *)result )
      goto LABEL_16;
  }
  return result;
}
