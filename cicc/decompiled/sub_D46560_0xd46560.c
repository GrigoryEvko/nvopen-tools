// Function: sub_D46560
// Address: 0xd46560
//
__int64 __fastcall sub_D46560(__int64 a1, __int64 *a2, const __m128i *a3, const __m128i *a4, __int64 a5, __int64 a6)
{
  __int64 v10; // rdx
  int v11; // eax
  __m128i *v12; // rdx
  __int64 result; // rax
  __int64 v14; // r15
  __int64 v15; // rsi
  __int64 v16; // rdi
  __m128i *v17; // rax
  const __m128i *v18; // rdi
  __m128i *v19; // rdx
  int v20; // r12d
  unsigned __int64 v21[7]; // [rsp+18h] [rbp-38h] BYREF

  v10 = *(unsigned int *)(a1 + 8);
  if ( *(_DWORD *)(a1 + 12) <= (unsigned int)v10 )
  {
    v14 = a1 + 16;
    v15 = sub_C8D7D0(a1, a1 + 16, 0, 0x28u, v21, a6);
    v16 = 40LL * *(unsigned int *)(a1 + 8);
    v17 = (__m128i *)(v16 + v15);
    if ( v16 + v15 )
    {
      *v17 = _mm_loadu_si128(a4);
      v17[1] = _mm_loadu_si128(a3);
      v17[2].m128i_i64[0] = *a2;
      v16 = 40LL * *(unsigned int *)(a1 + 8);
    }
    result = *(_QWORD *)a1;
    v18 = (const __m128i *)(*(_QWORD *)a1 + v16);
    if ( *(const __m128i **)a1 != v18 )
    {
      v19 = (__m128i *)v15;
      do
      {
        if ( v19 )
        {
          *v19 = _mm_loadu_si128((const __m128i *)result);
          v19[1] = _mm_loadu_si128((const __m128i *)(result + 16));
          v19[2].m128i_i64[0] = *(_QWORD *)(result + 32);
        }
        result += 40;
        v19 = (__m128i *)((char *)v19 + 40);
      }
      while ( v18 != (const __m128i *)result );
      v18 = *(const __m128i **)a1;
    }
    v20 = v21[0];
    if ( (const __m128i *)v14 != v18 )
      result = _libc_free(v18, v15);
    *(_DWORD *)(a1 + 12) = v20;
    ++*(_DWORD *)(a1 + 8);
    *(_QWORD *)a1 = v15;
  }
  else
  {
    v11 = *(_DWORD *)(a1 + 8);
    v12 = (__m128i *)(*(_QWORD *)a1 + 40 * v10);
    if ( v12 )
    {
      *v12 = _mm_loadu_si128(a4);
      v12[1] = _mm_loadu_si128(a3);
      v12[2].m128i_i64[0] = *a2;
      v11 = *(_DWORD *)(a1 + 8);
    }
    result = (unsigned int)(v11 + 1);
    *(_DWORD *)(a1 + 8) = result;
  }
  return result;
}
