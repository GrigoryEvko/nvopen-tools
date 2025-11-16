// Function: sub_390D200
// Address: 0x390d200
//
void __fastcall sub_390D200(__int64 a1)
{
  _BYTE v1[8]; // [rsp+0h] [rbp-D0h] BYREF
  char *v2; // [rsp+8h] [rbp-C8h]
  char v3; // [rsp+18h] [rbp-B8h] BYREF
  unsigned __int64 v4; // [rsp+A0h] [rbp-30h]

  sub_38CF370((__int64)v1, a1);
  sub_390CE40(a1, (__int64)v1);
  (*(void (__fastcall **)(_QWORD, __int64, _BYTE *))(**(_QWORD **)(a1 + 24) + 72LL))(*(_QWORD *)(a1 + 24), a1, v1);
  j___libc_free_0(v4);
  if ( v2 != &v3 )
    _libc_free((unsigned __int64)v2);
}
