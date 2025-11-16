// Function: sub_16E2F40
// Address: 0x16e2f40
//
__int64 __fastcall sub_16E2F40(__int64 a1, __int64 a2)
{
  _QWORD v3[4]; // [rsp+0h] [rbp-40h] BYREF
  int v4; // [rsp+20h] [rbp-20h]
  __int64 v5; // [rsp+28h] [rbp-18h]

  v5 = a2;
  v4 = 1;
  v3[0] = &unk_49EFC48;
  memset(&v3[1], 0, 24);
  sub_16E7A40(v3, 0, 0, 0);
  sub_16E2CE0(a1, (__int64)v3);
  v3[0] = &unk_49EFD28;
  return sub_16E7960(v3);
}
