// Function: sub_28514E0
// Address: 0x28514e0
//
void __fastcall sub_28514E0(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // r13
  __int64 *v5; // r14
  __int64 v6; // r13
  __int64 *v7; // rax
  __int64 v8; // rax
  __int64 v9; // rsi
  unsigned __int8 *v10; // rsi
  __int64 *v11; // r15
  __int64 v12; // r12
  __int64 *v13; // rax
  __int64 v14; // rax
  __int64 v15; // rsi
  unsigned __int8 *v16; // rsi
  _QWORD v17[7]; // [rsp+8h] [rbp-38h] BYREF

  v4 = sub_B98A20(a2, a2);
  sub_B91340(a1 + 40, 0);
  *(_QWORD *)(a1 + 40) = v4;
  sub_B96F50(a1 + 40, 0);
  v5 = *(__int64 **)a3;
  v6 = *(unsigned int *)(a3 + 8);
  v7 = (__int64 *)sub_B141C0(a1);
  v8 = sub_B0D000(v7, v5, v6, 0, 1);
  sub_B11F20(v17, v8);
  v9 = *(_QWORD *)(a1 + 80);
  if ( v9 )
    sub_B91220(a1 + 80, v9);
  v10 = (unsigned __int8 *)v17[0];
  *(_QWORD *)(a1 + 80) = v17[0];
  if ( v10 )
    sub_B976B0((__int64)v17, v10, a1 + 80);
  v11 = *(__int64 **)a3;
  v12 = *(unsigned int *)(a3 + 8);
  v13 = (__int64 *)sub_B141C0(a1);
  v14 = sub_B0D000(v13, v11, v12, 0, 1);
  sub_B11F20(v17, v14);
  v15 = *(_QWORD *)(a1 + 80);
  if ( v15 )
    sub_B91220(a1 + 80, v15);
  v16 = (unsigned __int8 *)v17[0];
  *(_QWORD *)(a1 + 80) = v17[0];
  if ( v16 )
    sub_B976B0((__int64)v17, v16, a1 + 80);
}
