// Function: sub_A7B440
// Address: 0xa7b440
//
unsigned __int64 __fastcall sub_A7B440(__int64 *a1, __int64 *a2, int a3, __int64 a4)
{
  unsigned __int64 v5; // r12
  __int64 *v7; // [rsp+0h] [rbp-90h] BYREF
  _BYTE *v8; // [rsp+8h] [rbp-88h]
  __int64 v9; // [rsp+10h] [rbp-80h]
  _BYTE v10[120]; // [rsp+18h] [rbp-78h] BYREF

  v7 = a2;
  v8 = v10;
  v9 = 0x800000000LL;
  sub_A77670((__int64)&v7, a4);
  v5 = sub_A7B2C0(a1, a2, a3, (__int64)&v7);
  if ( v8 != v10 )
    _libc_free(v8, a2);
  return v5;
}
