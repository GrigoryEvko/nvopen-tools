// Function: sub_2F83750
// Address: 0x2f83750
//
__int64 __fastcall sub_2F83750(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  char v3; // bl
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // rax
  _QWORD v10[6]; // [rsp+0h] [rbp-30h] BYREF

  v2 = *(_QWORD *)(a2 + 72);
  v3 = sub_AE5020(a1, v2);
  v4 = sub_9208B0(a1, v2);
  v10[1] = v5;
  v10[0] = ((1LL << v3) + ((unsigned __int64)(v4 + 7) >> 3) - 1) >> v3 << v3;
  v6 = sub_CA1930(v10);
  if ( !(unsigned __int8)sub_B4CE70(a2) )
    return v6;
  v7 = *(_QWORD *)(a2 - 32);
  if ( *(_BYTE *)v7 == 17 )
  {
    if ( *(_DWORD *)(v7 + 32) <= 0x40u )
      v8 = *(_QWORD *)(v7 + 24);
    else
      v8 = **(_QWORD **)(v7 + 24);
    v6 *= v8;
    return v6;
  }
  return 0;
}
