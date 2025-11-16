// Function: sub_2A41400
// Address: 0x2a41400
//
unsigned __int64 __fastcall sub_2A41400(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // r13d
  __int64 *v5; // rdi
  __int64 *v6; // rax
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // r13
  __int64 *v9; // rbx
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 *v12; // rbx
  _QWORD *v13; // rdi
  __int64 v15; // [rsp+0h] [rbp-60h] BYREF
  __int64 v16; // [rsp+8h] [rbp-58h]
  __int16 v17; // [rsp+20h] [rbp-40h]

  v4 = *(_DWORD *)(a1 + 320);
  v5 = *(__int64 **)a1;
  v16 = a3;
  v15 = a2;
  v17 = 261;
  v6 = (__int64 *)sub_BCB120(v5);
  v7 = sub_BCF640(v6, 0);
  v8 = sub_B2CE20(v7, 7, v4, (__int64)&v15, a1);
  sub_B2CD30(v8, 41);
  sub_2A3ED80((__int64 **)a1, v8, "_ZTSFvvE", 8);
  v9 = *(__int64 **)a1;
  v17 = 257;
  v10 = sub_22077B0(0x50u);
  v11 = v10;
  if ( v10 )
    sub_AA4D50(v10, (__int64)v9, (__int64)&v15, v8, 0);
  v12 = *(__int64 **)a1;
  sub_B43C20((__int64)&v15, v11);
  v13 = sub_BD2C40(72, 0);
  if ( v13 )
    sub_B4BB80((__int64)v13, (__int64)v12, 0, 0, v15, v16);
  v15 = v8;
  sub_2A413E0((__int64 **)a1, (unsigned __int64 *)&v15, 1);
  return v8;
}
