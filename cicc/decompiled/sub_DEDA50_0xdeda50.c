// Function: sub_DEDA50
// Address: 0xdeda50
//
__int64 __fastcall sub_DEDA50(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, char a6, bool a7, char a8)
{
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r9
  __int64 v13; // r8
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v20; // rax
  __int64 v21; // rax
  bool v22; // al
  __int64 v23; // r8
  _QWORD *v24; // rax
  __int64 v25; // rcx
  __int64 v26; // r14
  __int64 v27; // rax
  _QWORD *v28; // rax
  _QWORD *v29; // rax
  _QWORD *v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  int v34; // eax
  int v35; // eax
  __int64 v36; // rax
  _QWORD *v37; // rbx
  _QWORD *v38; // rbx
  int v39; // eax
  _QWORD *v40; // rax
  bool v41; // al
  __int64 v42; // r9
  __int64 v43; // rcx
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rax
  int v47; // eax
  int v48; // eax
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // r9
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 v57; // r9
  char v58; // al
  __int128 v59; // [rsp-10h] [rbp-110h]
  bool v60; // [rsp+0h] [rbp-100h]
  unsigned int v61; // [rsp+8h] [rbp-F8h]
  _QWORD *v62; // [rsp+8h] [rbp-F8h]
  __int64 v63; // [rsp+8h] [rbp-F8h]
  __int64 v65; // [rsp+18h] [rbp-E8h]
  __int64 v66; // [rsp+18h] [rbp-E8h]
  __int64 *v67; // [rsp+20h] [rbp-E0h]
  unsigned int v69; // [rsp+28h] [rbp-D8h]
  unsigned int v70; // [rsp+28h] [rbp-D8h]
  __int64 v71[2]; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v72[2]; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v73; // [rsp+50h] [rbp-B0h] BYREF
  int v74; // [rsp+58h] [rbp-A8h]
  __int64 v75[2]; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v76; // [rsp+70h] [rbp-90h] BYREF
  int v77; // [rsp+78h] [rbp-88h]
  __int64 v78; // [rsp+80h] [rbp-80h] BYREF
  int v79; // [rsp+88h] [rbp-78h]
  _BYTE *v80; // [rsp+90h] [rbp-70h] BYREF
  __int64 v81; // [rsp+98h] [rbp-68h]
  _BYTE v82[96]; // [rsp+A0h] [rbp-60h] BYREF

  v60 = a7;
  v80 = v82;
  v81 = 0x600000000LL;
  if ( !sub_DADE90((__int64)a2, a4, a5) )
    goto LABEL_4;
  v13 = a3;
  if ( *(_WORD *)(a3 + 24) == 8 )
  {
    if ( a5 != *(_QWORD *)(a3 + 48) )
    {
LABEL_4:
      v14 = sub_D970F0((__int64)a2);
      sub_D97F80(a1, v14, v15, v16, v17, v18);
      goto LABEL_5;
    }
  }
  else
  {
    if ( !a8 )
      goto LABEL_4;
    v20 = sub_DEAB30((__int64)a2, a3, a5, (__int64)&v80, a3, v12);
    v13 = v20;
    if ( !v20 || a5 != *(_QWORD *)(v20 + 48) )
      goto LABEL_4;
  }
  if ( *(_QWORD *)(v13 + 40) != 2 )
    goto LABEL_4;
  if ( a6 )
  {
    if ( a7 )
      v60 = (*(_WORD *)(v13 + 28) & 4) != 0;
    v61 = 38;
  }
  else
  {
    if ( a7 )
      v60 = (*(_WORD *)(v13 + 28) & 2) != 0;
    v61 = 34;
  }
  v65 = v13;
  v21 = sub_D33D80((_QWORD *)v13, (__int64)a2, v10, v11, v13);
  v67 = sub_DCAF50(a2, v21, 0);
  if ( !(unsigned __int8)sub_DBEDC0((__int64)a2, (__int64)v67) )
    goto LABEL_4;
  v22 = sub_D96900((__int64)v67);
  v23 = v65;
  if ( !v22 && !v60 )
  {
    v58 = sub_DD29F0(a2, a4, (__int64)v67, a6);
    v23 = v65;
    if ( v58 )
      goto LABEL_4;
  }
  v66 = **(_QWORD **)(v23 + 32);
  v24 = sub_DC7ED0(a2, v66, (__int64)v67, 0, 0);
  if ( (unsigned __int8)sub_DDD5B0(a2, a5, v61, (__int64)v24, a4) )
    goto LABEL_38;
  if ( !a6 )
  {
    if ( !(unsigned __int8)sub_DDD5B0(a2, a5, 35, v66, a4) )
    {
      v26 = (__int64)sub_DCEE80(a2, a4, v66, 0);
      goto LABEL_23;
    }
    goto LABEL_38;
  }
  if ( (unsigned __int8)sub_DDD5B0(a2, a5, 39, v66, a4) )
  {
LABEL_38:
    v26 = a4;
    goto LABEL_23;
  }
  v26 = sub_DCE160(a2, a4, v66, v25);
LABEL_23:
  if ( *(_BYTE *)(sub_D95540(v66) + 8) == 14 )
  {
    v66 = sub_DD3750((__int64)a2, v66);
    if ( sub_D96A50(v66) )
    {
      v14 = v66;
      sub_D97F80(a1, v66, v50, v51, v52, v53);
      goto LABEL_5;
    }
  }
  if ( *(_BYTE *)(sub_D95540(v26) + 8) == 14 )
  {
    v26 = sub_DD3750((__int64)a2, v26);
    if ( sub_D96A50(v26) )
    {
      v14 = v26;
      sub_D97F80(a1, v26, v54, v55, v56, v57);
      goto LABEL_5;
    }
  }
  v27 = sub_D95540((__int64)v67);
  v28 = sub_DA2C50((__int64)a2, v27, 1, 0);
  v62 = sub_DCC810(a2, (__int64)v67, (__int64)v28, 0, 0);
  v29 = sub_DCC810(a2, v66, v26, 0, 0);
  v30 = sub_DC7ED0(a2, (__int64)v29, (__int64)v62, 0, 0);
  v63 = sub_DCB270((__int64)a2, (__int64)v30, (__int64)v67);
  if ( !a6 )
  {
    v44 = sub_DBB9F0((__int64)a2, v66, 0, 0);
    sub_AB0910((__int64)v71, v44);
    v45 = sub_DBB9F0((__int64)a2, (__int64)v67, 0, 0);
    sub_AB0A00((__int64)v72, v45);
    v46 = sub_D95540(a3);
    v70 = sub_D97050((__int64)a2, v46);
    sub_9865C0((__int64)&v76, (__int64)v72);
    sub_C46F20((__int64)&v76, 1u);
    v47 = v77;
    v77 = 0;
    v79 = v47;
    v78 = v76;
    sub_9691E0((__int64)v75, v70, 0, 0, 0);
    sub_C45EE0((__int64)&v78, v75);
    v48 = v79;
    v79 = 0;
    v74 = v48;
    v73 = v78;
    sub_969240(v75);
    sub_969240(&v78);
    sub_969240(&v76);
    v49 = sub_DBB9F0((__int64)a2, a4, 0, 0);
    sub_AB0A00((__int64)&v78, v49);
    if ( (int)sub_C49970((__int64)&v78, (unsigned __int64 *)&v73) > 0 )
      goto LABEL_27;
LABEL_49:
    sub_9865C0((__int64)v75, (__int64)&v73);
    goto LABEL_28;
  }
  v31 = sub_DBB9F0((__int64)a2, v66, 1u, 0);
  sub_AB13A0((__int64)v71, v31);
  v32 = sub_DBB9F0((__int64)a2, (__int64)v67, 1u, 0);
  sub_AB14C0((__int64)v72, v32);
  v33 = sub_D95540(a3);
  v69 = sub_D97050((__int64)a2, v33);
  sub_9865C0((__int64)&v76, (__int64)v72);
  sub_C46F20((__int64)&v76, 1u);
  v34 = v77;
  v77 = 0;
  v79 = v34;
  v78 = v76;
  sub_986680((__int64)v75, v69);
  sub_C45EE0((__int64)&v78, v75);
  v35 = v79;
  v79 = 0;
  v74 = v35;
  v73 = v78;
  sub_969240(v75);
  sub_969240(&v78);
  sub_969240(&v76);
  v36 = sub_DBB9F0((__int64)a2, a4, 1u, 0);
  sub_AB14C0((__int64)&v78, v36);
  if ( (int)sub_C4C880((__int64)&v78, (__int64)&v73) <= 0 )
    goto LABEL_49;
LABEL_27:
  sub_9865C0((__int64)v75, (__int64)&v78);
LABEL_28:
  sub_969240(&v78);
  v37 = (_QWORD *)v63;
  if ( *(_WORD *)(v63 + 24) )
  {
    v38 = sub_DA26C0(a2, (__int64)v72);
    sub_9865C0((__int64)&v76, (__int64)v71);
    sub_C46B40((__int64)&v76, v75);
    v39 = v77;
    v77 = 0;
    v79 = v39;
    v78 = v76;
    v40 = sub_DA26C0(a2, (__int64)&v78);
    v37 = sub_DD21F0(a2, (__int64)v40, (__int64)v38);
    sub_969240(&v78);
    sub_969240(&v76);
  }
  if ( sub_D96A50((__int64)v37) )
    v37 = (_QWORD *)v63;
  v41 = sub_D96A50(v63);
  v43 = v63;
  *((_QWORD *)&v59 + 1) = (unsigned int)v81;
  if ( v41 )
    v43 = (__int64)v37;
  *(_QWORD *)&v59 = v80;
  v14 = v63;
  sub_D97FA0(a1, v63, (__int64)v37, v43, 0, v42, v59);
  sub_969240(v75);
  sub_969240(&v73);
  sub_969240(v72);
  sub_969240(v71);
LABEL_5:
  if ( v80 != v82 )
    _libc_free(v80, v14);
  return a1;
}
