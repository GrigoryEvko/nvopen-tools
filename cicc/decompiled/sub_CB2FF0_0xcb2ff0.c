// Function: sub_CB2FF0
// Address: 0xcb2ff0
//
__int64 __fastcall sub_CB2FF0(int *a1, __int64 a2, __int64 a3)
{
  int v3; // eax
  _QWORD v5[2]; // [rsp+0h] [rbp-20h] BYREF
  int v6; // [rsp+10h] [rbp-10h]

  v3 = *a1;
  v5[1] = "0x%X";
  v6 = v3;
  v5[0] = &unk_49DD0F8;
  return sub_CB6620(a3, v5);
}
