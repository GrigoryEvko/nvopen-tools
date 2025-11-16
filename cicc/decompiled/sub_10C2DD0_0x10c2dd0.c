// Function: sub_10C2DD0
// Address: 0x10c2dd0
//
__int64 __fastcall sub_10C2DD0(__int64 a1, __int64 a2, __int64 a3, char a4, char a5)
{
  __int16 v8; // r9
  __int16 v9; // r10
  __int64 v10; // r11
  __int64 v11; // r15
  unsigned int v12; // r14d
  __int64 v13; // rax
  __int64 v14; // r8
  char v16; // al
  char v17; // al
  unsigned __int8 v18; // al
  int v19; // eax
  __int64 v20; // rdi
  int v21; // edx
  _BYTE *v22; // rax
  __int64 v23; // rax
  __int64 *v24; // rax
  __int64 *v25; // rsi
  char v26; // r9
  char v27; // r9
  char v28; // r10
  char v29; // r9
  char v30; // r10
  char v31; // r9
  char v32; // r10
  unsigned int v33; // eax
  unsigned int v34; // edx
  int v35; // r14d
  int v36; // ebx
  __int64 v37; // rdi
  int v38; // edx
  __int64 v39; // r12
  __int64 v40; // rax
  __int64 v41; // rax
  int v42; // edx
  __int64 v43; // r11
  __int64 v44; // rax
  __int64 *v45; // rax
  int v46; // edx
  __int64 v47; // rax
  int v48; // eax
  int v49; // edx
  unsigned int v50; // ebx
  __int64 v51; // r12
  __int64 v52; // rax
  bool v53; // zf
  bool v54; // al
  char v55; // r9
  char v56; // r10
  int v57; // ebx
  __int64 v58; // rax
  __int64 v59; // r13
  __int64 v60; // r12
  __int64 v61; // rax
  __int64 *v62; // rax
  __int64 *v63; // rdx
  unsigned int v64; // eax
  bool v65; // al
  int v66; // [rsp+8h] [rbp-108h]
  char v67; // [rsp+10h] [rbp-100h]
  char v68; // [rsp+10h] [rbp-100h]
  __int64 *v69; // [rsp+10h] [rbp-100h]
  bool v70; // [rsp+10h] [rbp-100h]
  __int64 v71; // [rsp+18h] [rbp-F8h]
  char v72; // [rsp+18h] [rbp-F8h]
  __int64 v73; // [rsp+18h] [rbp-F8h]
  char v74; // [rsp+18h] [rbp-F8h]
  __int16 v75; // [rsp+20h] [rbp-F0h]
  __int64 v76; // [rsp+20h] [rbp-F0h]
  char v77; // [rsp+20h] [rbp-F0h]
  char v78; // [rsp+20h] [rbp-F0h]
  char v79; // [rsp+20h] [rbp-F0h]
  char v80; // [rsp+20h] [rbp-F0h]
  char v81; // [rsp+20h] [rbp-F0h]
  char v82; // [rsp+20h] [rbp-F0h]
  char v83; // [rsp+20h] [rbp-F0h]
  __int16 v84; // [rsp+28h] [rbp-E8h]
  _BYTE *v85; // [rsp+28h] [rbp-E8h]
  char v86; // [rsp+28h] [rbp-E8h]
  __int64 *v87; // [rsp+28h] [rbp-E8h]
  char v88; // [rsp+28h] [rbp-E8h]
  char v89; // [rsp+28h] [rbp-E8h]
  char v90; // [rsp+28h] [rbp-E8h]
  __int64 v91; // [rsp+28h] [rbp-E8h]
  char v92; // [rsp+28h] [rbp-E8h]
  __int64 v93; // [rsp+30h] [rbp-E0h]
  char v94; // [rsp+30h] [rbp-E0h]
  __int64 *v95; // [rsp+30h] [rbp-E0h]
  char v96; // [rsp+30h] [rbp-E0h]
  __int64 v97; // [rsp+38h] [rbp-D8h]
  __int64 **v98; // [rsp+38h] [rbp-D8h]
  char v99; // [rsp+38h] [rbp-D8h]
  unsigned __int16 v101; // [rsp+4Eh] [rbp-C2h]
  unsigned int v102; // [rsp+50h] [rbp-C0h]
  unsigned __int16 v104; // [rsp+58h] [rbp-B8h]
  __int64 v105; // [rsp+58h] [rbp-B8h]
  __int64 v106; // [rsp+58h] [rbp-B8h]
  __int64 *v107; // [rsp+58h] [rbp-B8h]
  __int64 *v108; // [rsp+60h] [rbp-B0h] BYREF
  __int64 *v109; // [rsp+68h] [rbp-A8h] BYREF
  __int64 **v110; // [rsp+70h] [rbp-A0h] BYREF
  char v111; // [rsp+78h] [rbp-98h]
  __int64 **v112; // [rsp+80h] [rbp-90h] BYREF
  char v113; // [rsp+88h] [rbp-88h]
  __int64 *v114[4]; // [rsp+90h] [rbp-80h] BYREF
  __int64 *v115[4]; // [rsp+B0h] [rbp-60h] BYREF
  __int16 v116; // [rsp+D0h] [rbp-40h]

  v8 = *(_WORD *)(a2 + 2);
  v9 = *(_WORD *)(a3 + 2);
  v10 = *(_QWORD *)(a2 - 64);
  v11 = *(_QWORD *)(a3 - 64);
  v93 = *(_QWORD *)(a2 - 32);
  v12 = v9 & 0x3F;
  v97 = *(_QWORD *)(a3 - 32);
  v104 = v8 & 0x3F;
  v102 = v8 & 0x3F;
  v101 = v9 & 0x3F;
  if ( v93 == v11 && *(_QWORD *)(a3 - 32) == v10 )
  {
    v105 = *(_QWORD *)(a2 - 64);
    v33 = sub_B52F50(v9 & 0x3F);
    v97 = v11;
    v10 = v105;
    v12 = v33;
LABEL_44:
    v106 = v10;
    v34 = v12 & v102;
    v35 = v102 | v12;
    if ( !a4 )
      v34 = v35;
    v36 = *(_BYTE *)(a3 + 1) >> 1;
    v37 = v34;
    if ( v36 == 127 )
      v36 = -1;
    v38 = *(_BYTE *)(a2 + 1) >> 1;
    if ( v38 != 127 )
      v36 &= v38;
    v39 = *(_QWORD *)(a1 + 32);
    v14 = sub_11FAF40(v37, *(_QWORD *)(v10 + 8), &v112);
    if ( !v14 )
    {
      LODWORD(v114[0]) = v36;
      BYTE4(v114[0]) = 1;
      v116 = 257;
      return sub_B35C90(v39, (unsigned int)v112, v106, v97, (__int64)v115, 0, (__int64)v114[0], 0);
    }
    return v14;
  }
  if ( v11 == v10 && v93 == v97 )
    goto LABEL_44;
  if ( a5 )
    goto LABEL_5;
  if ( v104 == 7 )
  {
    if ( v12 != 7 )
      goto LABEL_5;
    if ( !a4 )
      goto LABEL_6;
LABEL_13:
    if ( *(_QWORD *)(v11 + 8) != *(_QWORD *)(v10 + 8) )
      return 0;
    v71 = *(_QWORD *)(a2 - 64);
    v75 = *(_WORD *)(a3 + 2);
    v84 = *(_WORD *)(a2 + 2);
    v114[0] = 0;
    v16 = sub_10069D0(v114, v93);
    LOBYTE(v8) = v84;
    LOBYTE(v9) = v75;
    v10 = v71;
    if ( v16 )
    {
      v115[0] = 0;
      v17 = sub_10069D0(v115, v97);
      LOBYTE(v8) = v84;
      LOBYTE(v9) = v75;
      v10 = v71;
      if ( v17 )
      {
        v18 = *(_BYTE *)(a3 + 1);
        BYTE4(v114[0]) = 1;
        v19 = v18 >> 1;
        v20 = *(_QWORD *)(a1 + 32);
        v116 = 257;
        if ( v19 == 127 )
          v19 = -1;
        v21 = *(_BYTE *)(a2 + 1) >> 1;
        if ( v21 != 127 )
          v19 &= v21;
        LODWORD(v114[0]) = v19;
        return sub_B35C90(v20, v102, v71, v11, (__int64)v115, 0, (__int64)v114[0], 0);
      }
    }
LABEL_5:
    if ( !a4 )
      goto LABEL_6;
    goto LABEL_21;
  }
  if ( v12 != 8 || v104 != 8 )
    goto LABEL_5;
  if ( !a4 )
    goto LABEL_13;
LABEL_21:
  v67 = v9;
  v72 = v8;
  v76 = v10;
  v85 = sub_10BA570((char *)v10);
  v22 = sub_10BA570((char *)v11);
  v10 = v76;
  LOBYTE(v8) = v72;
  LOBYTE(v9) = v67;
  if ( v85 == v22 )
  {
    v14 = sub_10B86C0(*(_QWORD *)(a1 + 32), a2, a3);
    if ( v14 )
      return v14;
    v14 = sub_10B86C0(*(_QWORD *)(a1 + 32), a3, a2);
    if ( v14 )
      return v14;
    LOBYTE(v8) = v72;
    LOBYTE(v9) = v67;
    v10 = v76;
  }
LABEL_6:
  v13 = *(_QWORD *)(a2 + 16);
  if ( !v13 || *(_QWORD *)(v13 + 8) )
    return 0;
  v23 = *(_QWORD *)(a3 + 16);
  if ( !v23 || *(_QWORD *)(v23 + 8) )
  {
    if ( v10 == v11 )
      goto LABEL_28;
    return 0;
  }
  v73 = v10;
  v80 = v9;
  v90 = v8;
  v40 = sub_B43CB0(a3);
  v41 = sub_98A030(v12, v40, v11, v97, 1);
  LOBYTE(v8) = v90;
  LOBYTE(v9) = v80;
  v66 = v42;
  v43 = v73;
  v69 = (__int64 *)v41;
  if ( v41 )
  {
    v74 = v80;
    v81 = v90;
    v91 = v43;
    v44 = sub_B43CB0(a2);
    v45 = (__int64 *)sub_98A030(v102, v44, v91, v93, 1);
    v43 = v91;
    LOBYTE(v8) = v81;
    LOBYTE(v9) = v74;
    if ( v45 == v69 )
    {
      v114[0] = v45;
      v116 = 257;
      v107 = v45;
      HIDWORD(v112) = 0;
      v48 = v46 & v66;
      v49 = v66 | v46;
      if ( !a4 )
        v48 = v49;
      v50 = v48;
      v51 = *(_QWORD *)(a1 + 32);
      v52 = sub_BCB2D0(*(_QWORD **)(v51 + 72));
      v114[1] = (__int64 *)sub_ACD640(v52, v50, 0);
      v110 = (__int64 **)v107[1];
      return sub_B33D10(v51, 0xCFu, (__int64)&v110, 1, (int)v114, 2, (__int64)v112, (__int64)v115);
    }
  }
  if ( v43 != v11 )
    return 0;
  v47 = *(_QWORD *)(a2 + 16);
  if ( !v47 || *(_QWORD *)(v47 + 8) )
    return 0;
  v23 = *(_QWORD *)(a3 + 16);
LABEL_28:
  if ( !v23 )
    return 0;
  if ( *(_QWORD *)(v23 + 8) )
    return 0;
  v77 = v9;
  v86 = v8;
  if ( (unsigned int)sub_B52F50(v102) != v12 )
    return 0;
  v111 = 1;
  v110 = &v108;
  if ( !(unsigned __int8)sub_9940E0((__int64)&v110, v93) )
    return 0;
  v94 = v86;
  v112 = &v109;
  v113 = 1;
  if ( !(unsigned __int8)sub_9940E0((__int64)&v112, v97) )
    return 0;
  v87 = v109;
  v98 = (__int64 **)v108;
  v24 = (__int64 *)sub_C33340();
  v25 = v87;
  v26 = v94;
  v95 = v24;
  if ( (__int64 *)*v87 == v24 )
  {
    v92 = v26;
    sub_C3C790(v114, (_QWORD **)v25);
    v28 = v77;
    v27 = v92;
  }
  else
  {
    v88 = v26;
    sub_C33EB0(v114, v25);
    v27 = v88;
    v28 = v77;
  }
  v78 = v28;
  v89 = v27;
  if ( v114[0] == v95 )
  {
    sub_C3CCB0((__int64)v114);
    v30 = v78;
    v29 = v89;
  }
  else
  {
    sub_C34440((unsigned __int8 *)v114);
    v29 = v89;
    v30 = v78;
  }
  v68 = v30;
  v79 = v29;
  if ( v114[0] == v95 )
  {
    sub_C3C840(v115, v114);
    v32 = v68;
    v31 = v79;
  }
  else
  {
    sub_C338E0((__int64)v115, (__int64)v114);
    v31 = v79;
    v32 = v68;
  }
  if ( *v98 != v115[0] )
  {
    sub_91D830(v115);
    sub_91D830(v114);
    return 0;
  }
  v53 = *v98 == v95;
  v82 = v32;
  v96 = v31;
  if ( v53 )
  {
    v65 = sub_C3E590((__int64)v98, (__int64)v115);
    v56 = v82;
    v70 = v65;
    v55 = v96;
  }
  else
  {
    v54 = sub_C33D00((__int64)v98, (__int64)v115);
    v55 = v96;
    v56 = v82;
    v70 = v54;
  }
  v99 = v56;
  v83 = v55;
  sub_91D830(v115);
  sub_91D830(v114);
  if ( !v70 )
    return 0;
  v14 = 0;
  if ( !a4 )
  {
    if ( v104 > 5u )
    {
      if ( (unsigned __int16)(v104 - 12) > 1u )
        goto LABEL_72;
    }
    else if ( (v83 & 0x3C) == 0 )
    {
      goto LABEL_72;
    }
    v63 = v109;
    v109 = v108;
    v64 = v12;
    v12 = v102;
    v108 = v63;
    v102 = v64;
    goto LABEL_72;
  }
  if ( v101 > 5u )
  {
    if ( (unsigned __int16)(v101 - 12) > 1u )
      goto LABEL_71;
LABEL_80:
    v62 = v108;
    v102 = v12;
    v108 = v109;
    v109 = v62;
    goto LABEL_72;
  }
  if ( (v99 & 0x3C) != 0 )
    goto LABEL_80;
LABEL_71:
  v12 = v102;
LABEL_72:
  if ( v12 > 5 )
  {
    if ( v12 - 12 > 1 )
      return v14;
  }
  else if ( v12 <= 3 )
  {
    return v14;
  }
  v57 = sub_B45210(a2);
  if ( !a5 )
    v57 |= sub_B45210(a3);
  LODWORD(v114[0]) = v57;
  BYTE4(v114[0]) = 1;
  v116 = 257;
  v58 = sub_B33BC0(*(_QWORD *)(a1 + 32), 0xAAu, v11, (__int64)v114[0], (__int64)v115);
  v59 = *(_QWORD *)(a1 + 32);
  v116 = 257;
  v60 = v58;
  v61 = sub_AD8F10(*(_QWORD *)(v11 + 8), v108);
  LODWORD(v114[0]) = v57;
  BYTE4(v114[0]) = 1;
  return sub_B35C90(v59, v102, v60, v61, (__int64)v115, 0, (__int64)v114[0], 0);
}
