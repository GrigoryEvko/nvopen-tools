// Function: sub_23DC480
// Address: 0x23dc480
//
_QWORD *__fastcall sub_23DC480(__int64 *a1)
{
  __int64 v2; // r12
  _QWORD *v3; // rdi
  __int64 *v4; // rax
  unsigned __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // r15
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 v12; // r14
  _QWORD *v13; // rax
  _QWORD *v14; // r12
  const char *v16; // [rsp+0h] [rbp-60h] BYREF
  __int64 v17; // [rsp+8h] [rbp-58h]
  __int16 v18; // [rsp+20h] [rbp-40h]

  v2 = *a1;
  v3 = (_QWORD *)a1[5];
  v16 = "asan.module_dtor";
  v18 = 259;
  v4 = (__int64 *)sub_BCB120(v3);
  v5 = sub_BCF640(v4, 0);
  v6 = sub_B2CE20(v5, 7, 0, (__int64)&v16, v2);
  a1[33] = v6;
  sub_B2CD30(v6, 41);
  v7 = *a1;
  v16 = (const char *)a1[33];
  sub_2A413E0(v7, &v16, 1);
  v8 = a1[33];
  v18 = 257;
  v9 = a1[5];
  v10 = sub_22077B0(0x50u);
  v11 = v10;
  if ( v10 )
    sub_AA4D50(v10, v9, (__int64)&v16, v8, 0);
  v12 = a1[5];
  sub_B43C20((__int64)&v16, v11);
  v13 = sub_BD2C40(72, 0);
  v14 = v13;
  if ( v13 )
    sub_B4BB80((__int64)v13, v12, 0, 0, (__int64)v16, v17);
  return v14;
}
