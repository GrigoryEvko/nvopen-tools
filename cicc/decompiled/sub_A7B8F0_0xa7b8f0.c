// Function: sub_A7B8F0
// Address: 0xa7b8f0
//
__int64 __fastcall sub_A7B8F0(__int64 *a1, __int64 *a2, int a3)
{
  __int64 v5; // rbx
  _BYTE v6[8]; // [rsp+0h] [rbp-80h] BYREF
  char *v7; // [rsp+8h] [rbp-78h]
  char v8; // [rsp+18h] [rbp-68h] BYREF

  if ( !(unsigned __int8)sub_A73170(a1, a3) )
    return *a1;
  sub_A74940((__int64)v6, (__int64)a2, *a1);
  sub_A77390((__int64)v6, a3);
  v5 = sub_A7A280(a2, (__int64)v6);
  if ( v7 != &v8 )
    _libc_free(v7, v6);
  return v5;
}
