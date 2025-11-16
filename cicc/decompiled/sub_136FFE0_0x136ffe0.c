// Function: sub_136FFE0
// Address: 0x136ffe0
//
_QWORD *__fastcall sub_136FFE0(__int64 *a1)
{
  _QWORD *result; // rax
  __int64 *v2; // [rsp+8h] [rbp-68h] BYREF
  const char *v3; // [rsp+10h] [rbp-60h] BYREF
  char v4; // [rsp+20h] [rbp-50h]
  char v5; // [rsp+21h] [rbp-4Fh]
  _BYTE v6[16]; // [rsp+30h] [rbp-40h] BYREF
  __int16 v7; // [rsp+40h] [rbp-30h]
  __m128i v8; // [rsp+50h] [rbp-20h] BYREF
  _QWORD v9[2]; // [rsp+60h] [rbp-10h] BYREF

  v7 = 257;
  v2 = a1;
  v5 = 1;
  v3 = "BlockFrequencyDAGs";
  v4 = 3;
  sub_136FB20(&v8, &v2, (__int64)&v3, 0, (__int64)v6);
  if ( v8.m128i_i64[1] )
    sub_16BED90(v8.m128i_i64[0], v8.m128i_i64[1], 0, 0);
  result = v9;
  if ( (_QWORD *)v8.m128i_i64[0] != v9 )
    return (_QWORD *)j_j___libc_free_0(v8.m128i_i64[0], v9[0] + 1LL);
  return result;
}
