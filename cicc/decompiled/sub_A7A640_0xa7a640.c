// Function: sub_A7A640
// Address: 0xa7a640
//
__int64 __fastcall sub_A7A640(__int64 *a1, _QWORD *a2, const void *a3, size_t a4, const void *a5, size_t a6)
{
  __int64 v6; // rax
  __int64 v7; // r12
  _QWORD *v9; // [rsp+0h] [rbp-80h] BYREF
  _BYTE *v10; // [rsp+8h] [rbp-78h]
  __int64 v11; // [rsp+10h] [rbp-70h]
  _BYTE v12[104]; // [rsp+18h] [rbp-68h] BYREF

  v9 = a2;
  v10 = v12;
  v11 = 0x800000000LL;
  sub_A78980(&v9, a3, a4, a5, a6);
  v6 = sub_A7A280(a2, (__int64)&v9);
  v7 = sub_A7A4C0(a1, a2, v6);
  if ( v10 != v12 )
    _libc_free(v10, a2);
  return v7;
}
