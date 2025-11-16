// Function: sub_F54ED0
// Address: 0xf54ed0
//
__int64 __fastcall sub_F54ED0(unsigned __int8 *a1)
{
  __int64 *v1; // rsi
  __int64 result; // rax
  __int64 *v3; // [rsp+0h] [rbp-60h] BYREF
  __int64 v4; // [rsp+8h] [rbp-58h]
  _BYTE v5[16]; // [rsp+10h] [rbp-50h] BYREF
  __int64 *v6; // [rsp+20h] [rbp-40h] BYREF
  __int64 v7; // [rsp+28h] [rbp-38h]
  _BYTE v8[48]; // [rsp+30h] [rbp-30h] BYREF

  v4 = 0x100000000LL;
  v7 = 0x100000000LL;
  v3 = (__int64 *)v5;
  v6 = (__int64 *)v8;
  sub_AE7A50((__int64)&v3, (__int64)a1, (__int64)&v6);
  v1 = v3;
  result = (__int64)sub_F54050(a1, v3, (unsigned int)v4, v6, (unsigned int)v7);
  if ( v6 != (__int64 *)v8 )
    result = _libc_free(v6, v1);
  if ( v3 != (__int64 *)v5 )
    return _libc_free(v3, v1);
  return result;
}
