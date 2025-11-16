// Function: sub_1061B40
// Address: 0x1061b40
//
bool __fastcall sub_1061B40(__int64 a1, __int64 a2)
{
  _BYTE v3[32]; // [rsp+0h] [rbp-60h] BYREF
  _BYTE v4[64]; // [rsp+20h] [rbp-40h] BYREF

  if ( a2 == sub_1061AC0() || a2 == sub_1061AD0() )
    return a2 == a1;
  sub_1061A60((__int64)v3, a2);
  sub_1061A60((__int64)v4, a1);
  return sub_1061A80((__int64)v4, (__int64)v3);
}
