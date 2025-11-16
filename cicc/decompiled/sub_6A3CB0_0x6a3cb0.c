// Function: sub_6A3CB0
// Address: 0x6a3cb0
//
__int64 __fastcall sub_6A3CB0(char a1)
{
  __int64 v1; // r12
  __int64 v3; // [rsp+8h] [rbp-B8h] BYREF
  _BYTE v4[176]; // [rsp+10h] [rbp-B0h] BYREF

  sub_6E1DD0(&v3);
  sub_6E1E00(4, v4, 1, 0);
  sub_6E2170(v3);
  *(_BYTE *)(qword_4D03C50 + 20LL) = *(_BYTE *)(qword_4D03C50 + 20LL) & 0xFD | (2 * (a1 & 1));
  v1 = sub_6A2C00(1, 0);
  sub_6E2B30(1, 0);
  sub_6E1DF0(v3);
  return v1;
}
