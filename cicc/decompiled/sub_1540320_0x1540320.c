// Function: sub_1540320
// Address: 0x1540320
//
__int64 __fastcall sub_1540320(__int64 *a1, __m128i *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // r14
  __int64 v5; // r12
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rax
  __m128i v9; // xmm0
  __int64 v10; // rax
  __int64 v11; // rcx

  result = 0x7FFFFFFFFFFFFFFLL;
  *a1 = a3;
  a1[1] = 0;
  a1[2] = 0;
  if ( a3 <= 0x7FFFFFFFFFFFFFFLL )
    result = a3;
  if ( a3 > 0 )
  {
    v4 = result;
    while ( 1 )
    {
      v5 = 16 * v4;
      result = sub_2207800(16 * v4, &unk_435FF63);
      v6 = result;
      if ( result )
        break;
      v4 >>= 1;
      if ( !v4 )
        return result;
    }
    v7 = result + v5;
    v8 = result + 16;
    *(__m128i *)(v8 - 16) = _mm_loadu_si128(a2);
    if ( v7 == v8 )
    {
      v10 = v6;
    }
    else
    {
      do
      {
        v9 = _mm_loadu_si128((const __m128i *)(v8 - 16));
        v8 += 16;
        *(__m128i *)(v8 - 16) = v9;
      }
      while ( v7 != v8 );
      v10 = v6 + v5 - 16;
    }
    v11 = *(_QWORD *)v10;
    result = *(unsigned int *)(v10 + 8);
    a1[2] = v6;
    a1[1] = v4;
    a2->m128i_i64[0] = v11;
    a2->m128i_i32[2] = result;
  }
  return result;
}
