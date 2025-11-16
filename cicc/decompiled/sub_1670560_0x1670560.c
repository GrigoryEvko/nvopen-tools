// Function: sub_1670560
// Address: 0x1670560
//
bool __fastcall sub_1670560(__int64 a1, __int64 a2)
{
  _BYTE v3[32]; // [rsp+0h] [rbp-60h] BYREF
  _BYTE v4[64]; // [rsp+20h] [rbp-40h] BYREF

  if ( a2 == sub_16704E0() || a2 == sub_16704F0() )
    return a2 == a1;
  sub_1670480((__int64)v3, a2);
  sub_1670480((__int64)v4, a1);
  return sub_16704A0((__int64)v4, (__int64)v3);
}
