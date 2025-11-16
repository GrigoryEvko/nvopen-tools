// Function: sub_15AC0B0
// Address: 0x15ac0b0
//
__int64 __fastcall sub_15AC0B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 result; // rax
  __int64 v5; // rsi
  _QWORD v6[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = sub_15BA070(a2, a3, 1);
  result = sub_15C7080(v6, v3);
  if ( *(_QWORD *)(a1 + 48) )
    result = sub_161E7C0(a1 + 48);
  v5 = v6[0];
  *(_QWORD *)(a1 + 48) = v6[0];
  if ( v5 )
    return sub_1623210(v6, v5, a1 + 48);
  return result;
}
