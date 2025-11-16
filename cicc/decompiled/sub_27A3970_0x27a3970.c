// Function: sub_27A3970
// Address: 0x27a3970
//
__int64 __fastcall sub_27A3970(__int64 *a1, __m128i *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // r14
  __int64 v5; // r13
  __int64 v6; // rdx
  __m128i v7; // xmm3
  __int64 v8; // rcx
  __int64 v9; // rax
  __m128i v10; // xmm0
  __m128i v11; // xmm1
  __int64 v12; // rax
  __int32 v13; // ecx
  __int64 v14; // rcx

  result = 0x3FFFFFFFFFFFFFFLL;
  *a1 = a3;
  a1[1] = 0;
  a1[2] = 0;
  if ( a3 <= 0x3FFFFFFFFFFFFFFLL )
    result = a3;
  if ( a3 > 0 )
  {
    v4 = result;
    while ( 1 )
    {
      v5 = 32 * v4;
      result = sub_2207800(32 * v4);
      v6 = result;
      if ( result )
        break;
      v4 >>= 1;
      if ( !v4 )
        return result;
    }
    v7 = _mm_loadu_si128(a2 + 1);
    v8 = result + v5;
    v9 = result + 32;
    *(__m128i *)(v9 - 32) = _mm_loadu_si128(a2);
    *(__m128i *)(v9 - 16) = v7;
    if ( v8 == v9 )
    {
      v12 = v6;
    }
    else
    {
      do
      {
        v10 = _mm_loadu_si128((const __m128i *)(v9 - 32));
        v11 = _mm_loadu_si128((const __m128i *)(v9 - 16));
        v9 += 32;
        *(__m128i *)(v9 - 32) = v10;
        *(__m128i *)(v9 - 16) = v11;
      }
      while ( v8 != v9 );
      v12 = v6 + v5 - 32;
    }
    v13 = *(_DWORD *)v12;
    a1[2] = v6;
    a1[1] = v4;
    a2->m128i_i32[0] = v13;
    a2->m128i_i64[1] = *(_QWORD *)(v12 + 8);
    v14 = *(_QWORD *)(v12 + 16);
    result = *(_QWORD *)(v12 + 24);
    a2[1].m128i_i64[0] = v14;
    a2[1].m128i_i64[1] = result;
  }
  return result;
}
