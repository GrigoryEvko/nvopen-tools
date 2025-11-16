// Function: sub_24853D0
// Address: 0x24853d0
//
__int64 __fastcall sub_24853D0(__int64 a1, int a2, int a3)
{
  __int64 v5; // [rsp+8h] [rbp-7B8h] BYREF
  _BYTE v6[1920]; // [rsp+10h] [rbp-7B0h] BYREF
  _BYTE *v7; // [rsp+790h] [rbp-30h]

  sub_CC1970((__int64)v6);
  v5 = a1;
  v7 = v6;
  v6[1912] = 1;
  sub_CC19D0((__int64)v6, &v5, 8u);
  LODWORD(v5) = a2;
  sub_CC19D0((__int64)v7, &v5, 4u);
  LODWORD(v5) = a3;
  sub_CC19D0((__int64)v7, &v5, 4u);
  sub_CC21A0((__int64)v7, &v5, 8u);
  return v5;
}
