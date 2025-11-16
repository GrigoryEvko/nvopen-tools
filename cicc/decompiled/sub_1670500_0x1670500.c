// Function: sub_1670500
// Address: 0x1670500
//
bool __fastcall sub_1670500(__int64 a1, __int64 a2)
{
  _BYTE v3[64]; // [rsp+0h] [rbp-40h] BYREF

  if ( a2 == sub_16704E0() || a2 == sub_16704F0() )
    return 0;
  sub_1670480((__int64)v3, a2);
  return sub_16704A0(a1, (__int64)v3);
}
