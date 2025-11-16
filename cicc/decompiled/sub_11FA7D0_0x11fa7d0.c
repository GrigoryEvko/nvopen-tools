// Function: sub_11FA7D0
// Address: 0x11fa7d0
//
_QWORD *__fastcall sub_11FA7D0(__int64 **a1, void **a2, char a3, void **a4, int a5)
{
  _QWORD *v7; // rdi
  _QWORD *result; // rax
  _BYTE *v10; // [rsp+10h] [rbp-70h] BYREF
  __int64 v11; // [rsp+18h] [rbp-68h]
  _QWORD v12[2]; // [rsp+20h] [rbp-60h] BYREF
  __m128i v13; // [rsp+30h] [rbp-50h] BYREF
  _QWORD v14[8]; // [rsp+40h] [rbp-40h] BYREF

  v13.m128i_i64[0] = (__int64)v14;
  sub_11F33F0(v13.m128i_i64, byte_3F871B3, (__int64)byte_3F871B3);
  sub_11FA1D0((__int64)&v10, a1, a2, a3, a4, &v13);
  if ( (_QWORD *)v13.m128i_i64[0] != v14 )
    j_j___libc_free_0(v13.m128i_i64[0], v14[0] + 1LL);
  v7 = v10;
  if ( v11 )
  {
    sub_C67930(v10, v11, 0, a5);
    v7 = v10;
  }
  result = v12;
  if ( v7 != v12 )
    return (_QWORD *)j_j___libc_free_0(v7, v12[0] + 1LL);
  return result;
}
