// Function: sub_CB2F00
// Address: 0xcb2f00
//
__int64 __fastcall sub_CB2F00(int *a1, __int64 a2, __int64 a3)
{
  int v3; // xmm0_4
  _QWORD v5[2]; // [rsp+0h] [rbp-20h] BYREF
  int v6; // [rsp+10h] [rbp-10h]

  v3 = *a1;
  v5[1] = "%g";
  v6 = v3;
  v5[0] = &unk_49DB348;
  return sub_CB6620(a3, v5);
}
