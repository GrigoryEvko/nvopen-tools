// Function: sub_15A4230
// Address: 0x15a4230
//
__int64 __fastcall sub_15A4230(_QWORD *a1)
{
  __int64 *v2; // rax
  __int64 v3; // rdi
  __int64 v4; // r12
  __int64 **v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 *v10; // r14
  __int64 v11; // rax
  unsigned __int64 v12; // r12
  __int64 **v13; // rax
  _BYTE v15[8]; // [rsp+8h] [rbp-78h] BYREF
  __int64 *v16; // [rsp+10h] [rbp-70h] BYREF
  __int64 v17; // [rsp+18h] [rbp-68h]
  _QWORD v18[12]; // [rsp+20h] [rbp-60h] BYREF

  v2 = (__int64 *)sub_1643320(*a1);
  v3 = *v2;
  v18[0] = v2;
  v16 = v18;
  v18[1] = a1;
  v17 = 0x800000002LL;
  v4 = sub_1645600(v3, v18, 2, 0);
  v5 = (__int64 **)sub_1647190(v4, 0);
  v8 = sub_15A06D0(v5, 0, v6, v7);
  v9 = sub_1643360(*a1);
  v10 = (__int64 *)sub_159C470(v9, 0, 0);
  v11 = sub_1643350(*a1);
  v16 = v10;
  v17 = sub_159C470(v11, 1, 0);
  v15[4] = 0;
  v12 = sub_15A2E80(v4, v8, &v16, 2u, 0, (__int64)v15, 0);
  v13 = (__int64 **)sub_1643360(*a1);
  return sub_15A4180(v12, v13, 0);
}
