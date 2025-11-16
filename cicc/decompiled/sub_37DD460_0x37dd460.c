// Function: sub_37DD460
// Address: 0x37dd460
//
__int64 __fastcall sub_37DD460(const __m128i *a1, unsigned __int8 (__fastcall *a2)(__m128i *, const __m128i *))
{
  const __m128i *v2; // rbx
  __int64 v3; // rax
  __int64 result; // rax
  __m128i v5; // [rsp+0h] [rbp-30h] BYREF

  v2 = a1 - 1;
  v5 = _mm_loadu_si128(a1);
  while ( a2(&v5, v2) )
  {
    v3 = v2->m128i_i64[0];
    --v2;
    v2[2].m128i_i64[0] = v3;
    v2[2].m128i_i32[2] = v2[1].m128i_i32[2];
  }
  v2[1].m128i_i64[0] = v5.m128i_i64[0];
  result = v5.m128i_u32[2];
  v2[1].m128i_i32[2] = v5.m128i_i32[2];
  return result;
}
