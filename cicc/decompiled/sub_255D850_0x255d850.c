// Function: sub_255D850
// Address: 0x255d850
//
__int64 __fastcall sub_255D850(
        const __m128i *a1,
        const __m128i *a2,
        const __m128i *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  const __m128i *v7; // rbx
  const __m128i *v8; // r14
  __int64 (__fastcall *v9)(const __m128i *, __int64); // r12
  char i; // al
  const __m128i *v11; // rdi
  __int64 v12; // rax
  __m128i v13; // xmm0
  __int64 v14; // r12
  __int64 v15; // rax
  __m128i si128; // xmm0
  __m128i v18; // [rsp+0h] [rbp-60h] BYREF
  const void *v19; // [rsp+18h] [rbp-48h]
  __m128i v20[4]; // [rsp+20h] [rbp-40h] BYREF

  v7 = a1;
  v20[0].m128i_i64[0] = a4;
  if ( a1 != a2 )
  {
    v8 = a3;
    if ( a3 != (const __m128i *)a4 )
    {
      v9 = (__int64 (__fastcall *)(const __m128i *, __int64))a6;
      v19 = (const void *)(a5 + 16);
      for ( i = ((__int64 (__fastcall *)(const __m128i *, const __m128i *))a6)(a1, a3); ; i = v9(v7, (__int64)v8) )
      {
        if ( i )
        {
          v12 = *(unsigned int *)(a5 + 8);
          v13 = _mm_loadu_si128(v7);
          if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
          {
            v18 = v13;
            sub_C8D5F0(a5, v19, v12 + 1, 0x10u, a5, a6);
            v12 = *(unsigned int *)(a5 + 8);
            v13 = _mm_load_si128(&v18);
          }
          ++v7;
          *(__m128i *)(*(_QWORD *)a5 + 16 * v12) = v13;
          ++*(_DWORD *)(a5 + 8);
          if ( a2 == v7 )
            break;
        }
        else
        {
          v11 = v8++;
          a6 = (unsigned int)v9(v11, (__int64)v7);
          if ( !(_BYTE)a6 )
            ++v7;
          if ( a2 == v7 )
            break;
        }
        if ( (const __m128i *)v20[0].m128i_i64[0] == v8 )
          break;
      }
    }
  }
  v14 = a2 - v7;
  if ( (char *)a2 - (char *)v7 > 0 )
  {
    v15 = *(unsigned int *)(a5 + 8);
    do
    {
      si128 = _mm_loadu_si128(v7);
      if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
      {
        v20[0] = si128;
        sub_C8D5F0(a5, (const void *)(a5 + 16), v15 + 1, 0x10u, a5, a6);
        v15 = *(unsigned int *)(a5 + 8);
        si128 = _mm_load_si128(v20);
      }
      ++v7;
      *(__m128i *)(*(_QWORD *)a5 + 16 * v15) = si128;
      v15 = (unsigned int)(*(_DWORD *)(a5 + 8) + 1);
      *(_DWORD *)(a5 + 8) = v15;
      --v14;
    }
    while ( v14 );
  }
  return a5;
}
