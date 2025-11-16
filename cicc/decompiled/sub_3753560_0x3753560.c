// Function: sub_3753560
// Address: 0x3753560
//
__int64 __fastcall sub_3753560(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // rax
  unsigned __int8 *v4; // rsi
  __int64 v5; // r15
  __int64 v6; // rdx
  __int64 v7; // r12
  unsigned __int8 *v9[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = *(_QWORD *)(a2 + 32);
  v3 = sub_B0D220(*(_QWORD **)(a2 + 40));
  v4 = *(unsigned __int8 **)(a2 + 48);
  v5 = v3;
  v9[0] = v4;
  if ( v4 )
    sub_B96E90((__int64)v9, (__int64)v4, 1);
  sub_2E8FEC0(*(_QWORD **)a1, v9, *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) - 560LL, 0, 0, v2, v5);
  v7 = v6;
  if ( v9[0] )
    sub_B91220((__int64)v9, (__int64)v9[0]);
  return v7;
}
