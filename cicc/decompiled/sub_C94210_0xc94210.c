// Function: sub_C94210
// Address: 0xc94210
//
char __fastcall sub_C94210(const __m128i *a1, unsigned int a2, unsigned __int64 *a3)
{
  char result; // al
  __m128i v4; // [rsp+0h] [rbp-10h] BYREF

  v4 = _mm_loadu_si128(a1);
  result = sub_C93CF0(&v4, a2, a3);
  if ( !result )
    return v4.m128i_i64[1] != 0;
  return result;
}
