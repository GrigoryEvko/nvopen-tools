// Function: sub_1F2B760
// Address: 0x1f2b760
//
__int64 __fastcall sub_1F2B760(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // r14
  _QWORD *v4; // r15
  _QWORD *v5; // rax
  __int64 v6; // r12
  _QWORD *v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 *v10; // rax
  __int64 v11; // rax
  __int64 v12; // r15
  _QWORD *v13; // rax
  _QWORD *v14; // rbx
  unsigned __int64 *v15; // r15
  __int64 v16; // rax
  unsigned __int64 v17; // rcx
  __int64 v18; // rsi
  unsigned __int8 *v19; // rsi
  __int64 *v21; // rax
  __int64 v22; // rax
  __int64 v23; // r15
  __int64 v24; // rdi
  char *v25; // rax
  signed __int64 v26; // rdx
  __int64 v27; // rbx
  __int64 v28; // rax
  __int64 v29; // rdi
  __int64 v30; // [rsp+10h] [rbp-F0h]
  __int64 v31; // [rsp+18h] [rbp-E8h]
  __int64 v32; // [rsp+18h] [rbp-E8h]
  _BYTE v33[8]; // [rsp+28h] [rbp-D8h] BYREF
  __int64 *v34[2]; // [rsp+30h] [rbp-D0h] BYREF
  _QWORD v35[2]; // [rsp+40h] [rbp-C0h] BYREF
  char v36; // [rsp+50h] [rbp-B0h]
  char v37; // [rsp+51h] [rbp-AFh]
  char *v38; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v39; // [rsp+68h] [rbp-98h]
  _QWORD v40[2]; // [rsp+70h] [rbp-90h] BYREF
  const char *v41; // [rsp+80h] [rbp-80h] BYREF
  __int64 v42; // [rsp+88h] [rbp-78h]
  unsigned __int64 *v43; // [rsp+90h] [rbp-70h]
  _QWORD *v44; // [rsp+98h] [rbp-68h]
  __int64 v45; // [rsp+A0h] [rbp-60h]
  int v46; // [rsp+A8h] [rbp-58h]
  __int64 v47; // [rsp+B0h] [rbp-50h]
  __int64 v48; // [rsp+B8h] [rbp-48h]

  v2 = sub_15E0530(*(_QWORD *)(a1 + 232));
  v3 = *(_QWORD *)(a1 + 232);
  v4 = (_QWORD *)v2;
  LOWORD(v43) = 259;
  v41 = "CallStackCheckFailBlk";
  v5 = (_QWORD *)sub_22077B0(64);
  v6 = (__int64)v5;
  if ( v5 )
    sub_157FB60(v5, (__int64)v4, (__int64)&v41, v3, 0);
  v7 = (_QWORD *)sub_157E9C0(v6);
  v8 = *(_QWORD *)(a1 + 232);
  v42 = v6;
  v44 = v7;
  v41 = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v43 = (unsigned __int64 *)(v6 + 40);
  v9 = sub_1626D20(v8);
  sub_15C7110(&v38, 0, 0, v9, 0);
  if ( v41 )
    sub_161E7C0((__int64)&v41, (__int64)v41);
  v41 = v38;
  if ( v38 )
    sub_1623210((__int64)&v38, (unsigned __int8 *)v38, (__int64)&v41);
  if ( *(_DWORD *)(a1 + 220) == 13 )
  {
    v30 = *(_QWORD *)(a1 + 240);
    v32 = sub_16471D0(v4, 0);
    v21 = (__int64 *)sub_1643270(v4);
    v38 = (char *)v40;
    v40[0] = v32;
    v39 = 0x100000001LL;
    v22 = sub_1644EA0(v21, v40, 1, 0);
    v23 = sub_1632080(v30, (__int64)"__stack_smash_handler", 21, v22, 0);
    if ( v38 != (char *)v40 )
      _libc_free((unsigned __int64)v38);
    v37 = 1;
    LOWORD(v40[0]) = 257;
    v24 = *(_QWORD *)(a1 + 232);
    v35[0] = "SSH";
    v36 = 3;
    v25 = (char *)sub_1649960(v24);
    v27 = sub_15E70A0((__int64)&v41, v25, v26, (__int64)v35, 0);
    v28 = sub_1643350(v44);
    v34[0] = (__int64 *)sub_159C470(v28, 0, 0);
    v34[1] = v34[0];
    v29 = *(_QWORD *)(v27 + 24);
    v33[4] = 0;
    v34[0] = (__int64 *)sub_15A2E80(v29, v27, v34, 2u, 1u, (__int64)v33, 0);
    sub_1285290((__int64 *)&v41, *(_QWORD *)(*(_QWORD *)v23 + 24LL), v23, (int)v34, 1, (__int64)&v38, 0);
  }
  else
  {
    v31 = *(_QWORD *)(a1 + 240);
    v10 = (__int64 *)sub_1643270(v4);
    v38 = (char *)v40;
    v39 = 0;
    v11 = sub_1644EA0(v10, v40, 0, 0);
    v12 = sub_1632080(v31, (__int64)"__stack_chk_fail", 16, v11, 0);
    if ( v38 != (char *)v40 )
      _libc_free((unsigned __int64)v38);
    LOWORD(v40[0]) = 257;
    sub_1285290((__int64 *)&v41, *(_QWORD *)(*(_QWORD *)v12 + 24LL), v12, 0, 0, (__int64)&v38, 0);
  }
  LOWORD(v40[0]) = 257;
  v13 = sub_1648A60(56, 0);
  v14 = v13;
  if ( v13 )
    sub_15F82A0((__int64)v13, (__int64)v44, 0);
  if ( v42 )
  {
    v15 = v43;
    sub_157E9D0(v42 + 40, (__int64)v14);
    v16 = v14[3];
    v17 = *v15;
    v14[4] = v15;
    v17 &= 0xFFFFFFFFFFFFFFF8LL;
    v14[3] = v17 | v16 & 7;
    *(_QWORD *)(v17 + 8) = v14 + 3;
    *v15 = *v15 & 7 | (unsigned __int64)(v14 + 3);
  }
  sub_164B780((__int64)v14, (__int64 *)&v38);
  if ( v41 )
  {
    v35[0] = v41;
    sub_1623A60((__int64)v35, (__int64)v41, 2);
    v18 = v14[6];
    if ( v18 )
      sub_161E7C0((__int64)(v14 + 6), v18);
    v19 = (unsigned __int8 *)v35[0];
    v14[6] = v35[0];
    if ( v19 )
      sub_1623210((__int64)v35, v19, (__int64)(v14 + 6));
    if ( v41 )
      sub_161E7C0((__int64)&v41, (__int64)v41);
  }
  return v6;
}
