// Function: sub_20E9F00
// Address: 0x20e9f00
//
_QWORD *__fastcall sub_20E9F00(const char *a1)
{
  _QWORD *result; // rax
  const char *v2; // [rsp+0h] [rbp-60h] BYREF
  char v3; // [rsp+10h] [rbp-50h]
  char v4; // [rsp+11h] [rbp-4Fh]
  __int16 v5; // [rsp+30h] [rbp-30h]
  __m128i v6; // [rsp+40h] [rbp-20h] BYREF
  _QWORD v7[2]; // [rsp+50h] [rbp-10h] BYREF

  v5 = 257;
  v4 = 1;
  v2 = "EdgeBundles";
  v3 = 3;
  sub_20E9B20(&v6, a1, (__int64)&v2);
  if ( v6.m128i_i64[1] )
    sub_16BED90(v6.m128i_i64[0], v6.m128i_i64[1], 0, 0);
  result = v7;
  if ( (_QWORD *)v6.m128i_i64[0] != v7 )
    return (_QWORD *)j_j___libc_free_0(v6.m128i_i64[0], v7[0] + 1LL);
  return result;
}
