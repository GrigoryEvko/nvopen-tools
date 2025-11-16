// Function: sub_692720
// Address: 0x692720
//
__int64 __fastcall sub_692720(__int64 a1)
{
  __int64 v1; // r12
  __int64 v3; // [rsp+8h] [rbp-218h] BYREF
  _BYTE v4[160]; // [rsp+10h] [rbp-210h] BYREF
  _QWORD v5[46]; // [rsp+B0h] [rbp-170h] BYREF

  sub_6E1DD0(&v3);
  sub_6E1E00(5, v4, 0, 1);
  sub_6E70E0(a1, v5);
  sub_688FA0(v5);
  v1 = sub_6F6F40(v5, 0);
  sub_6E2B30(v5, 0);
  sub_6E1DF0(v3);
  return v1;
}
