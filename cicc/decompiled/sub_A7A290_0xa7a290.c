// Function: sub_A7A290
// Address: 0xa7a290
//
__int64 __fastcall sub_A7A290(__int64 *a1, __int64 *a2, const void *a3, size_t a4)
{
  __int64 result; // rax
  __int64 v7; // [rsp+8h] [rbp-98h]
  _BYTE v8[8]; // [rsp+10h] [rbp-90h] BYREF
  char *v9; // [rsp+18h] [rbp-88h]
  char v10; // [rsp+28h] [rbp-78h] BYREF

  if ( !(unsigned __int8)sub_A73380(a1, a3, a4) )
    return *a1;
  sub_A74940((__int64)v8, (__int64)a2, *a1);
  sub_A77740((__int64)v8, a3, a4);
  result = sub_A7A280(a2, (__int64)v8);
  if ( v9 != &v10 )
  {
    v7 = result;
    _libc_free(v9, v8);
    return v7;
  }
  return result;
}
