// Function: sub_1DE2C60
// Address: 0x1de2c60
//
_QWORD *__fastcall sub_1DE2C60(__int64 a1, __int64 a2, char a3)
{
  _QWORD *result; // rax
  __int64 v4; // [rsp+8h] [rbp-48h] BYREF
  _BYTE v5[16]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v6; // [rsp+20h] [rbp-30h]
  __m128i v7; // [rsp+30h] [rbp-20h] BYREF
  _QWORD v8[2]; // [rsp+40h] [rbp-10h] BYREF

  v4 = a1;
  v6 = 257;
  sub_1DE2730(&v7, &v4, a2, a3, (__int64)v5);
  if ( v7.m128i_i64[1] )
    sub_16BED90(v7.m128i_i64[0], v7.m128i_i64[1], 0, 0);
  result = v8;
  if ( (_QWORD *)v7.m128i_i64[0] != v8 )
    return (_QWORD *)j_j___libc_free_0(v7.m128i_i64[0], v8[0] + 1LL);
  return result;
}
