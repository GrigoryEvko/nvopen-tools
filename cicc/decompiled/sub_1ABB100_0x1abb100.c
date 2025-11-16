// Function: sub_1ABB100
// Address: 0x1abb100
//
bool __fastcall sub_1ABB100(__int64 a1, __int64 *a2)
{
  __int64 v2; // r14
  __int64 v3; // rbx
  __int64 v5; // rsi
  __int64 v6; // r12
  __int64 v7; // rsi
  unsigned __int8 *v8; // rsi
  __int64 v10[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = a1 + 40;
  v3 = *(_QWORD *)(a1 + 48);
  if ( v3 == a1 + 40 )
    return 0;
  while ( 1 )
  {
    if ( !v3 )
      BUG();
    v5 = *(_QWORD *)(v3 + 24);
    if ( v5 )
      break;
    v3 = *(_QWORD *)(v3 + 8);
    if ( v2 == v3 )
      return 0;
  }
  v6 = *a2;
  v10[0] = *(_QWORD *)(v3 + 24);
  sub_1623A60((__int64)v10, v5, 2);
  if ( (__int64 *)(v6 + 48) == v10 )
  {
    if ( v10[0] )
      sub_161E7C0(v6 + 48, v10[0]);
  }
  else
  {
    v7 = *(_QWORD *)(v6 + 48);
    if ( v7 )
      sub_161E7C0(v6 + 48, v7);
    v8 = (unsigned __int8 *)v10[0];
    *(_QWORD *)(v6 + 48) = v10[0];
    if ( v8 )
      sub_1623210((__int64)v10, v8, v6 + 48);
  }
  return v2 != v3;
}
