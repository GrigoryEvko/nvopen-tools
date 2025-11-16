// Function: sub_6B9640
// Address: 0x6b9640
//
__int64 __fastcall sub_6B9640(__int64 a1)
{
  _BYTE v2[160]; // [rsp+0h] [rbp-210h] BYREF
  _QWORD v3[46]; // [rsp+A0h] [rbp-170h] BYREF

  sub_6E1E00(5, v2, 0, 0);
  sub_69ED20((__int64)v3, 0, 0, 1);
  sub_6F69D0(v3, 3);
  *(_QWORD *)(a1 + 88) = v3[0];
  return sub_6E2B30(v3, 3);
}
