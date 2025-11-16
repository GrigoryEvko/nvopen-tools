// Function: sub_B6E100
// Address: 0xb6e100
//
__int64 __fastcall sub_B6E100(__int64 *a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  __m128i v7; // [rsp+0h] [rbp-30h] BYREF
  __int64 v8; // [rsp+10h] [rbp-20h] BYREF

  sub_B6E0E0(&v7, a2, a3, a4, a1, a5);
  v5 = sub_BA8CB0(a1, v7.m128i_i64[0], v7.m128i_i64[1]);
  if ( (__int64 *)v7.m128i_i64[0] != &v8 )
    j_j___libc_free_0(v7.m128i_i64[0], v8 + 1);
  return v5;
}
