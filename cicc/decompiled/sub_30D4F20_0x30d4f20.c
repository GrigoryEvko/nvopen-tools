// Function: sub_30D4F20
// Address: 0x30d4f20
//
__int64 __fastcall sub_30D4F20(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // [rsp+8h] [rbp-18h]
  __int64 v4; // [rsp+8h] [rbp-18h]

  v3 = sub_30D4EB0(a2, "call-threshold-bonus", 0x14u);
  if ( BYTE4(v3) )
    *(_DWORD *)(a1 + 704) += v3;
  v4 = sub_30D4EB0(a2, "call-inline-cost", 0x10u);
  result = 1;
  if ( BYTE4(v4) )
  {
    sub_30D0F50(a1, (int)v4);
    return 0;
  }
  return result;
}
