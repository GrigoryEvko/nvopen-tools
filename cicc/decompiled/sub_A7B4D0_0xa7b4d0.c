// Function: sub_A7B4D0
// Address: 0xa7b4d0
//
unsigned __int64 __fastcall sub_A7B4D0(__int64 *a1, _QWORD *a2, __int64 a3)
{
  unsigned __int64 v3; // r12
  _QWORD *v5; // [rsp+0h] [rbp-80h] BYREF
  _BYTE *v6; // [rsp+8h] [rbp-78h]
  __int64 v7; // [rsp+10h] [rbp-70h]
  _BYTE v8[104]; // [rsp+18h] [rbp-68h] BYREF

  v5 = a2;
  v6 = v8;
  v7 = 0x800000000LL;
  sub_A78C10(&v5, a3);
  v3 = sub_A7B2C0(a1, a2, 0, (__int64)&v5);
  if ( v6 != v8 )
    _libc_free(v6, a2);
  return v3;
}
