// Function: sub_1061AE0
// Address: 0x1061ae0
//
bool __fastcall sub_1061AE0(__int64 a1, __int64 a2)
{
  _BYTE v3[64]; // [rsp+0h] [rbp-40h] BYREF

  if ( a2 == sub_1061AC0() || a2 == sub_1061AD0() )
    return 0;
  sub_1061A60((__int64)v3, a2);
  return sub_1061A80(a1, (__int64)v3);
}
