// Function: sub_1C3EFF0
// Address: 0x1c3eff0
//
void __fastcall sub_1C3EFF0(__int64 a1, _BYTE *a2, __int64 a3)
{
  __int64 v3[2]; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v4[4]; // [rsp+10h] [rbp-20h] BYREF

  v3[0] = (__int64)v4;
  sub_CEB5A0(v3, a2, (__int64)&a2[a3]);
  sub_1C3EFD0((__int64)v3, 0);
  if ( (_QWORD *)v3[0] != v4 )
    j_j___libc_free_0(v3[0], v4[0] + 1LL);
}
