// Function: sub_390F830
// Address: 0x390f830
//
__int64 __fastcall sub_390F830(__int64 **a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r14
  __int64 v5; // r13
  __int64 v6; // rax
  unsigned __int64 v7; // rax
  _QWORD v9[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = **a1;
  v5 = sub_38CF310(a2, 0, v4, 0);
  v6 = sub_38CF310(a3, 0, v4, 0);
  v7 = sub_38CB1F0(17, v6, v5, v4, 0);
  sub_38CF260(v7, v9, a1);
  return LODWORD(v9[0]);
}
