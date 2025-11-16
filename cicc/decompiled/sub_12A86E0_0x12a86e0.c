// Function: sub_12A86E0
// Address: 0x12a86e0
//
__int64 __fastcall sub_12A86E0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 result; // rax
  __int64 v5; // rsi
  _QWORD v6[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = *a1;
  if ( *a1 )
  {
    v6[0] = *a1;
    result = sub_1623A60(v6, v3, 2);
    if ( *(_QWORD *)(a2 + 48) )
      result = sub_161E7C0(a2 + 48);
    v5 = v6[0];
    *(_QWORD *)(a2 + 48) = v6[0];
    if ( v5 )
      return sub_1623210(v6, v5, a2 + 48);
  }
  return result;
}
