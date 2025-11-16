// Function: sub_A7A4C0
// Address: 0xa7a4c0
//
__int64 __fastcall sub_A7A4C0(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v5; // [rsp+8h] [rbp-E8h]
  _BYTE v6[8]; // [rsp+10h] [rbp-E0h] BYREF
  char *v7; // [rsp+18h] [rbp-D8h]
  char v8; // [rsp+28h] [rbp-C8h] BYREF
  _BYTE v9[8]; // [rsp+70h] [rbp-80h] BYREF
  char *v10; // [rsp+78h] [rbp-78h]
  char v11; // [rsp+88h] [rbp-68h] BYREF

  result = a3;
  if ( *a1 )
  {
    if ( a3 )
    {
      sub_A74940((__int64)v6, (__int64)a2, *a1);
      sub_A74940((__int64)v9, (__int64)a2, a3);
      sub_A776F0((__int64)v6, (__int64)v9);
      if ( v10 != &v11 )
        _libc_free(v10, v9);
      result = sub_A7A280(a2, (__int64)v6);
      if ( v7 != &v8 )
      {
        v5 = result;
        _libc_free(v7, v6);
        return v5;
      }
    }
    else
    {
      return *a1;
    }
  }
  return result;
}
