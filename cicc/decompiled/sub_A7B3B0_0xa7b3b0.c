// Function: sub_A7B3B0
// Address: 0xa7b3b0
//
unsigned __int64 __fastcall sub_A7B3B0(
        __int64 *a1,
        _QWORD *a2,
        int a3,
        const void *a4,
        size_t a5,
        __int64 a6,
        const void *a7,
        size_t a8)
{
  unsigned __int64 v9; // r12
  _QWORD *v11; // [rsp+0h] [rbp-90h] BYREF
  _BYTE *v12; // [rsp+8h] [rbp-88h]
  __int64 v13; // [rsp+10h] [rbp-80h]
  _BYTE v14[120]; // [rsp+18h] [rbp-78h] BYREF

  v13 = 0x800000000LL;
  v11 = a2;
  v12 = v14;
  sub_A78980(&v11, a4, a5, a7, a8);
  v9 = sub_A7B2C0(a1, a2, a3, (__int64)&v11);
  if ( v12 != v14 )
    _libc_free(v12, a2);
  return v9;
}
