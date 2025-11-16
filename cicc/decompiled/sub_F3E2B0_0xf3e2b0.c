// Function: sub_F3E2B0
// Address: 0xf3e2b0
//
__int64 __fastcall sub_F3E2B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  bool v5; // zf
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 i; // rdx
  __int64 v9; // rsi
  __m128i *v10; // rax
  __m128i *v11; // [rsp+8h] [rbp-28h] BYREF

  v4 = a2;
  v5 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v5 )
  {
    result = *(_QWORD *)(a1 + 16);
    v7 = 40LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    result = a1 + 16;
    v7 = 160;
  }
  for ( i = result + v7; i != result; result += 40 )
  {
    if ( result )
    {
      *(_QWORD *)result = 0;
      *(_BYTE *)(result + 24) = 0;
      *(_QWORD *)(result + 32) = 0;
    }
  }
  if ( a2 != a3 )
  {
    do
    {
      while ( *(_QWORD *)v4
           || *(_BYTE *)(v4 + 24) && (*(_QWORD *)(v4 + 8) || *(_QWORD *)(v4 + 16))
           || *(_QWORD *)(v4 + 32) )
      {
        v9 = v4;
        v4 += 40;
        sub_F38D60(a1, v9, (__int64 *)&v11);
        v10 = v11;
        *v11 = _mm_loadu_si128((const __m128i *)(v4 - 40));
        v10[1] = _mm_loadu_si128((const __m128i *)(v4 - 24));
        v10[2].m128i_i64[0] = *(_QWORD *)(v4 - 8);
        result = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1u;
        *(_DWORD *)(a1 + 8) = result;
        if ( a3 == v4 )
          return result;
      }
      v4 += 40;
    }
    while ( a3 != v4 );
  }
  return result;
}
