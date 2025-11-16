// Function: sub_1815CC0
// Address: 0x1815cc0
//
__int64 __fastcall sub_1815CC0(__int64 a1, _QWORD *a2)
{
  int v3; // r15d
  __int64 v4; // rax
  _QWORD *v5; // rdi
  __int64 v6; // r14
  __int64 *v7; // rax
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 *v15; // rdi
  __int64 v16; // rax
  _QWORD *v17; // rdi
  __int64 v18; // rax
  _QWORD *v19; // rdi
  __int64 *v20; // rax
  __int64 v21; // rax
  _QWORD *v22; // rdi
  __int64 v23; // rax
  _QWORD *v24; // rdi
  __int64 *v25; // rax
  __int64 v26; // rax
  _QWORD *v27; // rdi
  __int64 *v28; // rax
  __int64 v29; // rax
  _QWORD *v30; // rdi
  __int64 v31; // rax
  _QWORD *v32; // rdi
  __int64 *v33; // rax
  __int64 v34; // rax
  bool v35; // zf
  __int64 *v36; // rax
  __int64 *v37; // rax
  __int64 *v38; // rax
  __int64 **v39; // r13
  unsigned __int64 v40; // rax
  __int64 *v41; // rax
  __int64 *v42; // rax
  __int64 **v43; // r13
  unsigned __int64 v44; // rax
  __int64 v45; // rax
  __int64 *v46; // rdi
  bool v48; // [rsp+Eh] [rbp-C2h]
  bool v49; // [rsp+Fh] [rbp-C1h]
  __int64 v50; // [rsp+18h] [rbp-B8h] BYREF
  __int64 *v51[2]; // [rsp+20h] [rbp-B0h] BYREF
  _QWORD v52[2]; // [rsp+30h] [rbp-A0h] BYREF
  _QWORD v53[2]; // [rsp+40h] [rbp-90h] BYREF
  __int64 v54; // [rsp+50h] [rbp-80h]
  __int64 *v55; // [rsp+60h] [rbp-70h] BYREF
  __int64 v56; // [rsp+70h] [rbp-60h] BYREF
  int v57; // [rsp+80h] [rbp-50h]

  LOWORD(v54) = 260;
  v53[0] = a2 + 30;
  sub_16E1010((__int64)&v55, (__int64)v53);
  v3 = v57;
  if ( v57 == 12 )
  {
    v48 = 1;
  }
  else
  {
    v49 = 1;
    v48 = v57 == 13;
    if ( v57 == 3 )
      goto LABEL_3;
  }
  v49 = v57 == 4;
LABEL_3:
  v4 = sub_1632FA0((__int64)a2);
  *(_QWORD *)(a1 + 160) = a2;
  v5 = (_QWORD *)*a2;
  v6 = v4;
  *(_QWORD *)(a1 + 168) = *a2;
  v7 = (__int64 *)sub_1644900(v5, 0x10u);
  *(_QWORD *)(a1 + 176) = v7;
  v8 = sub_1646BA0(v7, 0);
  v9 = *(_QWORD *)(a1 + 168);
  *(_QWORD *)(a1 + 184) = v8;
  v10 = sub_15A9620(v6, v9, 0);
  v11 = *(_QWORD *)(a1 + 176);
  *(_QWORD *)(a1 + 192) = v10;
  v12 = sub_159C580(v11, 0);
  v13 = *(_QWORD *)(a1 + 192);
  *(_QWORD *)(a1 + 200) = v12;
  *(_QWORD *)(a1 + 216) = sub_159C580(v13, 2);
  if ( v3 == 32 )
  {
    *(_QWORD *)(a1 + 208) = sub_159C580(*(_QWORD *)(a1 + 192), 0xFFFF8FFFFFFFFFFFLL);
  }
  else if ( v48 )
  {
    *(_QWORD *)(a1 + 208) = sub_159C580(*(_QWORD *)(a1 + 192), 0xFFFFFF0FFFFFFFFFLL);
  }
  else
  {
    if ( !v49 )
      sub_16BD130("unsupported triple", 1u);
    *(_BYTE *)(a1 + 528) = 1;
  }
  v51[0] = *(__int64 **)(a1 + 176);
  v51[1] = v51[0];
  v14 = sub_1644EA0(v51[0], v51, 2, 0);
  v15 = *(__int64 **)(a1 + 176);
  *(_QWORD *)(a1 + 280) = v14;
  v52[0] = *(_QWORD *)(a1 + 184);
  v52[1] = *(_QWORD *)(a1 + 192);
  v16 = sub_1644EA0(v15, v52, 2, 0);
  v17 = *(_QWORD **)(a1 + 168);
  *(_QWORD *)(a1 + 288) = v16;
  v18 = sub_16471D0(v17, 0);
  v19 = *(_QWORD **)(a1 + 168);
  v53[0] = v18;
  v20 = (__int64 *)sub_1643270(v19);
  v21 = sub_1644EA0(v20, v53, 1, 0);
  v22 = *(_QWORD **)(a1 + 168);
  *(_QWORD *)(a1 + 296) = v21;
  v53[0] = *(_QWORD *)(a1 + 176);
  v23 = sub_16471D0(v22, 0);
  v24 = *(_QWORD **)(a1 + 168);
  v53[1] = v23;
  v54 = *(_QWORD *)(a1 + 192);
  v25 = (__int64 *)sub_1643270(v24);
  v26 = sub_1644EA0(v25, v53, 3, 0);
  v27 = *(_QWORD **)(a1 + 168);
  *(_QWORD *)(a1 + 304) = v26;
  v28 = (__int64 *)sub_1643270(v27);
  v29 = sub_1644EA0(v28, 0, 0, 0);
  v30 = *(_QWORD **)(a1 + 168);
  *(_QWORD *)(a1 + 312) = v29;
  v31 = sub_16471D0(v30, 0);
  v32 = *(_QWORD **)(a1 + 168);
  v50 = v31;
  v33 = (__int64 *)sub_1643270(v32);
  v34 = sub_1644EA0(v33, &v50, 1, 0);
  v35 = *(_QWORD *)(a1 + 240) == 0;
  *(_QWORD *)(a1 + 320) = v34;
  if ( !v35 )
  {
    v36 = sub_1645D80(*(__int64 **)(a1 + 176), 64);
    *(_QWORD *)(a1 + 224) = 0;
    v37 = (__int64 *)sub_1646BA0(v36, 0);
    v38 = (__int64 *)sub_16453E0(v37, 0);
    v39 = (__int64 **)sub_1646BA0(v38, 0);
    v40 = sub_159C470(*(_QWORD *)(a1 + 192), *(_QWORD *)(a1 + 240), 0);
    *(_QWORD *)(a1 + 256) = sub_15A3BA0(v40, v39, 0);
  }
  if ( *(_QWORD *)(a1 + 248) )
  {
    *(_QWORD *)(a1 + 232) = 0;
    v41 = (__int64 *)sub_1646BA0(*(__int64 **)(a1 + 176), 0);
    v42 = (__int64 *)sub_16453E0(v41, 0);
    v43 = (__int64 **)sub_1646BA0(v42, 0);
    v44 = sub_159C470(*(_QWORD *)(a1 + 192), *(_QWORD *)(a1 + 248), 0);
    *(_QWORD *)(a1 + 264) = sub_15A3BA0(v44, v43, 0);
  }
  v50 = *(_QWORD *)(a1 + 168);
  v45 = sub_161BE60(&v50, 1u, 0x3E8u);
  v46 = v55;
  *(_QWORD *)(a1 + 384) = v45;
  if ( v46 != &v56 )
    j_j___libc_free_0(v46, v56 + 1);
  return 1;
}
