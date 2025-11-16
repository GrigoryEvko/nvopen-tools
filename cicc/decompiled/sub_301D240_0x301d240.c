// Function: sub_301D240
// Address: 0x301d240
//
_QWORD *__fastcall sub_301D240(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int8 *v7; // rsi
  bool v8; // zf
  _QWORD *v9; // r13
  __int64 v10; // rbx
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v14; // rdx
  __int64 v15; // rax
  unsigned __int8 *v16[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = *(unsigned __int8 **)a3;
  v8 = (*(_BYTE *)(a2 + 44) & 4) == 0;
  v9 = *(_QWORD **)(a1 + 32);
  v16[0] = *(unsigned __int8 **)a3;
  if ( v8 )
  {
    if ( v7 )
      sub_B96E90((__int64)v16, (__int64)v7, 1);
    v10 = (__int64)sub_2E7B380(v9, a4, v16, 0);
    if ( v16[0] )
      sub_B91220((__int64)v16, (__int64)v16[0]);
    sub_2E31040((__int64 *)(a1 + 40), v10);
    v14 = *(_QWORD *)a2;
    v15 = *(_QWORD *)v10;
    *(_QWORD *)(v10 + 8) = a2;
    v14 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v10 = v14 | v15 & 7;
    *(_QWORD *)(v14 + 8) = v10;
    *(_QWORD *)a2 = v10 | *(_QWORD *)a2 & 7LL;
    v11 = *(_QWORD *)(a3 + 8);
    if ( v11 )
      goto LABEL_7;
  }
  else
  {
    if ( v7 )
      sub_B96E90((__int64)v16, (__int64)v7, 1);
    v10 = (__int64)sub_2E7B380(v9, a4, v16, 0);
    if ( v16[0] )
      sub_B91220((__int64)v16, (__int64)v16[0]);
    sub_2E326B0(a1, (__int64 *)a2, v10);
    v11 = *(_QWORD *)(a3 + 8);
    if ( v11 )
LABEL_7:
      sub_2E882B0(v10, (__int64)v9, v11);
  }
  v12 = *(_QWORD *)(a3 + 16);
  if ( v12 )
    sub_2E88680(v10, (__int64)v9, v12);
  return v9;
}
