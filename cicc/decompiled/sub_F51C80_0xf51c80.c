// Function: sub_F51C80
// Address: 0xf51c80
//
void __fastcall sub_F51C80(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  __int64 v6; // r15
  __int64 v7; // r13
  __int64 v8; // r9
  __int64 v9; // rax
  _QWORD *v10; // r15
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // rsi
  __int64 v15; // [rsp+0h] [rbp-50h]
  __int64 v16; // [rsp+8h] [rbp-48h]
  _QWORD v17[7]; // [rsp+18h] [rbp-38h] BYREF

  v16 = sub_B12000(a1 + 72);
  v4 = sub_B11F60(a1 + 80);
  v5 = a1;
  v6 = *(_QWORD *)(a2 - 64);
  v7 = v4;
  sub_AE7AF0((__int64)v17, a1);
  if ( sub_AF4770(v7) || !sub_AF4730(v7) && (v5 = a1, (unsigned __int8)sub_F50590(*(_QWORD *)(v6 + 8), a1)) )
  {
    sub_F4EE60(a3, v6, v16, v7, (__int64)v17, v8, a2 + 24, 0);
    v13 = v17[0];
    if ( !v17[0] )
      return;
    goto LABEL_6;
  }
  v9 = sub_ACADE0(*(__int64 ***)(v6 + 8));
  v10 = sub_B98A20(v9, v5);
  v15 = sub_B10CD0((__int64)v17);
  v11 = sub_22077B0(96);
  v12 = v11;
  if ( v11 )
    sub_B12150(v11, (__int64)v10, v16, v7, v15, 1);
  sub_AA8770(*(_QWORD *)(a2 + 40), v12, a2 + 24, 0);
  v13 = v17[0];
  if ( v17[0] )
LABEL_6:
    sub_B91220((__int64)v17, v13);
}
