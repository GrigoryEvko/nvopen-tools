// Function: sub_1705150
// Address: 0x1705150
//
unsigned __int64 __fastcall sub_1705150(__int64 a1)
{
  unsigned __int64 result; // rax
  __int64 *v2; // r12
  __int64 v3; // rdx
  __int64 v4; // rsi
  __int64 v5; // rsi
  unsigned __int8 *v6; // rsi
  __int64 v7; // rsi
  __int64 *v8; // r12
  __int64 v9; // rsi
  unsigned __int8 *v10; // rsi
  __int64 v11; // rsi
  __int64 v12[5]; // [rsp+8h] [rbp-28h] BYREF

  result = *(_QWORD *)(a1 + 8);
  v2 = *(__int64 **)a1;
  v3 = *(_QWORD *)(a1 + 16);
  if ( !result )
  {
    v2[1] = 0;
    v2[2] = 0;
    goto LABEL_15;
  }
  v2[1] = result;
  result += 40LL;
  v2[2] = v3;
  if ( v3 == result )
    goto LABEL_15;
  if ( !v3 )
    BUG();
  v4 = *(_QWORD *)(v3 + 24);
  v12[0] = v4;
  if ( !v4 )
  {
    v5 = *v2;
    if ( !*v2 )
      goto LABEL_15;
    goto LABEL_6;
  }
  result = sub_1623A60((__int64)v12, v4, 2);
  v5 = *v2;
  if ( *v2 )
LABEL_6:
    result = sub_161E7C0((__int64)v2, v5);
  v6 = (unsigned __int8 *)v12[0];
  *v2 = v12[0];
  if ( v6 )
  {
    result = sub_1623210((__int64)v12, v6, (__int64)v2);
    v7 = *(_QWORD *)(a1 + 24);
    v8 = *(__int64 **)a1;
    v12[0] = v7;
    if ( !v7 )
      goto LABEL_9;
    goto LABEL_16;
  }
  if ( v12[0] )
    result = sub_161E7C0((__int64)v12, v12[0]);
LABEL_15:
  v7 = *(_QWORD *)(a1 + 24);
  v8 = *(__int64 **)a1;
  v12[0] = v7;
  if ( !v7 )
  {
LABEL_9:
    if ( v8 == v12 )
      return result;
    v9 = *v8;
    if ( !*v8 )
      goto LABEL_19;
    goto LABEL_11;
  }
LABEL_16:
  result = sub_1623A60((__int64)v12, v7, 2);
  if ( v8 == v12 )
    goto LABEL_19;
  v9 = *v8;
  if ( *v8 )
  {
LABEL_11:
    result = sub_161E7C0((__int64)v8, v9);
    v10 = (unsigned __int8 *)v12[0];
    *v8 = v12[0];
    if ( v10 )
      goto LABEL_12;
    goto LABEL_19;
  }
  v10 = (unsigned __int8 *)v12[0];
  *v8 = v12[0];
  if ( v10 )
  {
LABEL_12:
    result = sub_1623210((__int64)v12, v10, (__int64)v8);
    v11 = *(_QWORD *)(a1 + 24);
    if ( !v11 )
      return result;
    return sub_161E7C0(a1 + 24, v11);
  }
LABEL_19:
  if ( v12[0] )
    result = sub_161E7C0((__int64)v12, v12[0]);
  v11 = *(_QWORD *)(a1 + 24);
  if ( v11 )
    return sub_161E7C0(a1 + 24, v11);
  return result;
}
