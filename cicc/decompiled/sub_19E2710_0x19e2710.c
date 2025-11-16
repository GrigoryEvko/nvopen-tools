// Function: sub_19E2710
// Address: 0x19e2710
//
__int64 __fastcall sub_19E2710(__m128i *a1)
{
  __int64 result; // rax
  signed int v2; // edx
  signed int v3; // ecx
  __int64 v4; // rsi
  unsigned __int64 v5; // r8
  __m128i v6; // xmm0
  __m128i v7; // xmm1

  result = a1->m128i_u32[0];
  v2 = a1->m128i_i32[1];
  v3 = a1->m128i_i32[2];
  v4 = a1[1].m128i_i64[0];
  v5 = a1[1].m128i_u64[1];
  while ( a1[-2].m128i_i32[0] > (int)result
       || a1[-2].m128i_i32[0] == (_DWORD)result
       && (a1[-2].m128i_i32[1] > v2
        || a1[-2].m128i_i32[1] == v2
        && (a1[-2].m128i_i32[2] > v3
         || a1[-2].m128i_i32[2] == v3
         && (a1[-1].m128i_i64[0] > v4 || a1[-1].m128i_i64[0] == v4 && a1[-1].m128i_i64[1] > v5))) )
  {
    v6 = _mm_loadu_si128(a1 - 2);
    v7 = _mm_loadu_si128(a1 - 1);
    a1 -= 2;
    a1[2] = v6;
    a1[3] = v7;
  }
  a1->m128i_i32[0] = result;
  a1->m128i_i32[1] = v2;
  a1->m128i_i32[2] = v3;
  a1[1].m128i_i64[0] = v4;
  a1[1].m128i_i64[1] = v5;
  return result;
}
