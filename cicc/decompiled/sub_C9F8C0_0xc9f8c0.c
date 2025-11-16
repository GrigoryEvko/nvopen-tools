// Function: sub_C9F8C0
// Address: 0xc9f8c0
//
__int64 __fastcall sub_C9F8C0(const __m128i *a1)
{
  __int64 v2; // rdi
  const __m128i *v3; // rdi
  const __m128i *v4; // rdi
  __int64 result; // rax

  v2 = a1[9].m128i_i64[1];
  if ( v2 )
    sub_C9F770(v2, a1);
  v3 = (const __m128i *)a1[7].m128i_i64[0];
  if ( v3 != &a1[8] )
    j_j___libc_free_0(v3, a1[8].m128i_i64[0] + 1);
  v4 = (const __m128i *)a1[5].m128i_i64[0];
  result = (__int64)a1[6].m128i_i64;
  if ( v4 != &a1[6] )
    return j_j___libc_free_0(v4, a1[6].m128i_i64[0] + 1);
  return result;
}
