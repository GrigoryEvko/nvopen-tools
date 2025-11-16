// Function: sub_254E230
// Address: 0x254e230
//
_BYTE *__fastcall sub_254E230(__int64 *a1, unsigned __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // r13
  __m128i v4; // rax
  _BYTE *result; // rax
  __int64 v6; // rdx
  char v7; // [rsp+Fh] [rbp-41h] BYREF
  __m128i v8; // [rsp+10h] [rbp-40h] BYREF
  _BYTE *v9; // [rsp+20h] [rbp-30h]
  __int64 v10; // [rsp+28h] [rbp-28h]

  v2 = *a1;
  v3 = a1[1];
  v7 = 0;
  v4.m128i_i64[0] = sub_250D2C0(a2, 0);
  v8 = v4;
  result = sub_2527570(v2, &v8, v3, &v7);
  v10 = v6;
  v9 = result;
  if ( !(_BYTE)v6 || !result )
    return (_BYTE *)a2;
  return result;
}
