// Function: sub_14D1B70
// Address: 0x14d1b70
//
__int64 __fastcall sub_14D1B70(__int64 a1, __int64 a2, float a3)
{
  __int64 v3; // r13
  _BYTE v5[64]; // [rsp+10h] [rbp-40h] BYREF

  v3 = sub_1698270(a1, a2);
  sub_169D3B0(v5, a3);
  sub_169E320(a1 + 8, v5, v3);
  return sub_1698460(v5);
}
