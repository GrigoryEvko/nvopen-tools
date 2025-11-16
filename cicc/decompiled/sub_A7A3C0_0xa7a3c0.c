// Function: sub_A7A3C0
// Address: 0xa7a3c0
//
__int64 __fastcall sub_A7A3C0(__int64 *a1, __int64 *a2, __int64 a3)
{
  _BYTE *v5; // rsi
  __int64 v6; // r12
  _BYTE v8[8]; // [rsp+0h] [rbp-80h] BYREF
  char *v9; // [rsp+8h] [rbp-78h]
  char v10; // [rsp+18h] [rbp-68h] BYREF

  sub_A74940((__int64)v8, (__int64)a2, *a1);
  v5 = (_BYTE *)a3;
  if ( sub_A74BD0((__int64)v8, a3) )
  {
    sub_A74A10((__int64)v8, a3);
    v5 = v8;
    v6 = sub_A7A280(a2, (__int64)v8);
  }
  else
  {
    v6 = *a1;
  }
  if ( v9 != &v10 )
    _libc_free(v9, v5);
  return v6;
}
