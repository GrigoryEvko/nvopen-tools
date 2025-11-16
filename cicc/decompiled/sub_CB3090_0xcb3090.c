// Function: sub_CB3090
// Address: 0xcb3090
//
__int64 __fastcall sub_CB3090(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  _QWORD v5[4]; // [rsp+0h] [rbp-20h] BYREF

  v3 = *a1;
  v5[1] = "0x%lX";
  v5[2] = v3;
  v5[0] = &unk_49DC5C0;
  return sub_CB6620(a3, v5);
}
