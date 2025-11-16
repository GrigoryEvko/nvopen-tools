// Function: sub_11B5E90
// Address: 0x11b5e90
//
__int64 *__fastcall sub_11B5E90(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r10
  __int64 *v7; // r13
  __int64 *v8; // r12
  __int64 v9; // r11
  unsigned __int8 *v10; // rbx
  unsigned __int8 *v11; // r14
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v14; // rax
  __int64 *v15; // rdi
  bool v17; // al
  int v18; // r13d
  __int64 v19; // r14
  int v20; // r13d
  __int64 v21; // rax
  __int64 v22; // r11
  __int64 *v23; // r10
  __int64 v24; // r15
  __int64 v25; // rsi
  __int64 v26; // r13
  __int64 v27; // rsi
  unsigned __int8 *v28; // rsi
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // r13
  __int64 *v34; // r12
  const char *v35; // rdx
  unsigned __int8 *v36; // rax
  unsigned __int8 *v37; // r12
  __int64 v38; // rsi
  __int64 v39; // rsi
  __int64 v40; // rdx
  unsigned __int8 *v41; // rsi
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 v47; // r9
  __int64 v48; // r12
  __int64 v49; // rsi
  __int64 v50; // r14
  __int64 v51; // rsi
  unsigned __int8 *v52; // rsi
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 v57; // r9
  int v58; // eax
  int v59; // eax
  unsigned int v60; // edx
  __int64 v61; // rax
  __int64 v62; // rdx
  __int64 v63; // rdx
  __int64 v64; // rdx
  unsigned __int8 *v65; // r11
  _QWORD *v66; // rax
  __int64 v67; // r11
  __int64 v68; // r8
  const char *v69; // rsi
  __int64 v70; // rsi
  __int64 v71; // rdx
  unsigned __int8 *v72; // rsi
  __int64 v73; // rax
  __int64 v74; // rdx
  __int64 v75; // rcx
  __int64 v76; // r8
  __int64 v77; // r9
  __int16 v78; // dx
  __int64 *v79; // rbx
  __int64 v80; // r14
  __int64 *v81; // r12
  __int64 v82; // r13
  __int64 *v83; // [rsp+0h] [rbp-100h]
  __int64 v84; // [rsp+8h] [rbp-F8h]
  __int64 v85; // [rsp+10h] [rbp-F0h]
  __int64 v86; // [rsp+10h] [rbp-F0h]
  __int64 v87; // [rsp+10h] [rbp-F0h]
  __int64 v88; // [rsp+10h] [rbp-F0h]
  __int64 v89; // [rsp+18h] [rbp-E8h]
  __int64 v90; // [rsp+18h] [rbp-E8h]
  __int64 v91; // [rsp+18h] [rbp-E8h]
  char v92; // [rsp+18h] [rbp-E8h]
  __int64 v93; // [rsp+20h] [rbp-E0h]
  __int64 v94; // [rsp+28h] [rbp-D8h]
  __int64 v95; // [rsp+30h] [rbp-D0h]
  __int64 v96; // [rsp+38h] [rbp-C8h]
  __int64 *v97; // [rsp+40h] [rbp-C0h]
  __int64 *v98; // [rsp+48h] [rbp-B8h]
  __int64 v99; // [rsp+48h] [rbp-B8h]
  __int64 v100; // [rsp+48h] [rbp-B8h]
  char v101; // [rsp+48h] [rbp-B8h]
  __int64 v102; // [rsp+50h] [rbp-B0h]
  __int64 *v103; // [rsp+50h] [rbp-B0h]
  __int64 *v104; // [rsp+50h] [rbp-B0h]
  __int64 *v105; // [rsp+50h] [rbp-B0h]
  __int64 v106; // [rsp+50h] [rbp-B0h]
  __int64 v107; // [rsp+50h] [rbp-B0h]
  __int64 v108; // [rsp+58h] [rbp-A8h]
  __int64 *v109; // [rsp+58h] [rbp-A8h]
  __int64 *v110; // [rsp+60h] [rbp-A0h]
  __int64 v111; // [rsp+60h] [rbp-A0h]
  __int64 v112; // [rsp+60h] [rbp-A0h]
  __int64 v113; // [rsp+60h] [rbp-A0h]
  __int64 v114; // [rsp+60h] [rbp-A0h]
  __int64 v115; // [rsp+60h] [rbp-A0h]
  __int64 *v116; // [rsp+60h] [rbp-A0h]
  __int64 *v118; // [rsp+68h] [rbp-98h]
  unsigned __int8 *v119; // [rsp+78h] [rbp-88h] BYREF
  __int64 *v120; // [rsp+80h] [rbp-80h] BYREF
  __int64 v121; // [rsp+88h] [rbp-78h]
  _BYTE v122[16]; // [rsp+90h] [rbp-70h] BYREF
  const char *v123[4]; // [rsp+A0h] [rbp-60h] BYREF
  __int16 v124; // [rsp+C0h] [rbp-40h]

  v6 = (__int64 *)v122;
  v7 = *(__int64 **)(a3 + 16);
  v120 = (__int64 *)v122;
  v121 = 0x200000000LL;
  if ( !v7 )
    return v7;
  v8 = a2;
  v9 = a3;
  v10 = 0;
  do
  {
    while ( 1 )
    {
      v11 = (unsigned __int8 *)v7[3];
      if ( *v11 == 90 )
        break;
      if ( v10 )
        goto LABEL_14;
      v7 = (__int64 *)v7[1];
      v10 = v11;
      if ( !v7 )
        goto LABEL_10;
    }
    if ( *(v8 - 4) != *((_QWORD *)v11 - 4) )
    {
LABEL_14:
      v15 = v120;
      v7 = 0;
      goto LABEL_15;
    }
    v12 = (unsigned int)v121;
    a4 = HIDWORD(v121);
    v13 = (unsigned int)v121 + 1LL;
    if ( v13 > HIDWORD(v121) )
    {
      a2 = v6;
      v108 = v9;
      v110 = v6;
      sub_C8D5F0((__int64)&v120, v6, v13, 8u, a5, a6);
      v12 = (unsigned int)v121;
      v9 = v108;
      v6 = v110;
    }
    a3 = (__int64)v120;
    v120[v12] = (__int64)v11;
    LODWORD(v121) = v121 + 1;
    v7 = (__int64 *)v7[1];
  }
  while ( v7 );
LABEL_10:
  if ( !v10
    || (v14 = *((_QWORD *)v10 + 2)) == 0
    || *(_QWORD *)(v14 + 8)
    || (v111 = v9, v9 != *(_QWORD *)(v14 + 24))
    || (unsigned int)*v10 - 42 > 0x11
    || (a2 = (__int64 *)*(v8 - 4), v109 = v6, v17 = sub_11AF2C0((__int64)v10, (__int64)a2, a3, a4), v6 = v109, !v17) )
  {
    v15 = v120;
    if ( v120 != v6 )
      goto LABEL_16;
    return v7;
  }
  v18 = *(_DWORD *)(v111 + 4);
  v124 = 257;
  v19 = v111 + 24;
  v20 = v18 & 0x7FFFFFF;
  v102 = v8[1];
  v21 = sub_BD2DA0(80);
  v22 = v111;
  v23 = v109;
  v24 = v21;
  if ( v21 )
  {
    sub_B44260(v21, v102, 55, 0x8000000u, 0, 0);
    *(_DWORD *)(v24 + 72) = v20;
    sub_BD6B50((unsigned __int8 *)v24, v123);
    sub_BD2A10(v24, *(_DWORD *)(v24 + 72), 1);
    v22 = v111;
    v23 = v109;
  }
  v25 = *(_QWORD *)(v22 + 48);
  v119 = (unsigned __int8 *)v25;
  if ( v25 )
  {
    v26 = v24 + 48;
    v103 = v23;
    v112 = v22;
    sub_B96E90((__int64)&v119, v25, 1);
    v27 = *(_QWORD *)(v24 + 48);
    v22 = v112;
    v23 = v103;
    if ( !v27 )
      goto LABEL_26;
    goto LABEL_25;
  }
  v27 = *(_QWORD *)(v24 + 48);
  v26 = v24 + 48;
  if ( v27 )
  {
LABEL_25:
    v104 = v23;
    v113 = v22;
    sub_B91220(v26, v27);
    v23 = v104;
    v22 = v113;
LABEL_26:
    v28 = v119;
    *(_QWORD *)(v24 + 48) = v119;
    if ( v28 )
    {
      v105 = v23;
      v114 = v22;
      sub_B976B0((__int64)&v119, v28, v26);
      v22 = v114;
      v23 = v105;
    }
  }
  v98 = v23;
  v106 = v22;
  sub_B44220((_QWORD *)v24, v19, 0);
  a2 = (__int64 *)&v119;
  v119 = (unsigned __int8 *)v24;
  sub_11B4E60(*(_QWORD *)(a1 + 40) + 2096LL, (__int64 *)&v119, v29, v30, v31, v32);
  v115 = 0;
  v6 = v98;
  if ( (*(_DWORD *)(v106 + 4) & 0x7FFFFFF) == 0 )
    goto LABEL_73;
  v97 = v8;
  v33 = v106;
  v83 = v98;
  do
  {
    v64 = *(_QWORD *)(v33 - 8);
    v65 = *(unsigned __int8 **)(v64 + 32 * v115);
    v107 = *(_QWORD *)(v64 + 32LL * *(unsigned int *)(v33 + 72) + 8 * v115);
    if ( v65 && v65 == v10 )
    {
      v85 = *(v97 - 4);
      v99 = (__int64)(v10 + 24);
      v34 = (__int64 *)&v10[32 * (v33 == *((_QWORD *)v10 - 8) && *((_QWORD *)v10 - 8) != 0) - 64];
      v123[0] = sub_BD5D20(*v34);
      v124 = 773;
      v123[1] = v35;
      v123[2] = ".Elt";
      v89 = *v34;
      v36 = (unsigned __int8 *)sub_BD2C40(72, 2u);
      v37 = v36;
      if ( v36 )
        sub_B4DE80((__int64)v36, v89, v85, (__int64)v123, 0, 0);
      v38 = *((_QWORD *)v10 + 6);
      v119 = (unsigned __int8 *)v38;
      if ( v38 )
      {
        sub_B96E90((__int64)&v119, v38, 1);
        v39 = *((_QWORD *)v37 + 6);
        v40 = (__int64)(v37 + 48);
        if ( !v39 )
          goto LABEL_35;
      }
      else
      {
        v39 = *((_QWORD *)v37 + 6);
        v40 = (__int64)(v37 + 48);
        if ( !v39 )
          goto LABEL_37;
      }
      v90 = v40;
      sub_B91220(v40, v39);
      v40 = v90;
LABEL_35:
      v41 = v119;
      *((_QWORD *)v37 + 6) = v119;
      if ( v41 )
        sub_B976B0((__int64)&v119, v41, v40);
LABEL_37:
      v42 = v96;
      LOWORD(v42) = 0;
      sub_B44220(v37, v99, v42);
      v119 = v37;
      v96 = *(_QWORD *)(a1 + 40);
      sub_11B4E60(v96 + 2096, (__int64 *)&v119, v43, v44, v45, v46);
      v124 = 257;
      v47 = v95;
      LOWORD(v47) = 0;
      v48 = sub_B504D0((unsigned int)*v10 - 29, v24, (__int64)v37, (__int64)v123, 0, v47);
      sub_B45260((unsigned __int8 *)v48, (__int64)v10, 1);
      v49 = *((_QWORD *)v10 + 6);
      v119 = (unsigned __int8 *)v49;
      if ( v49 )
      {
        v50 = v48 + 48;
        sub_B96E90((__int64)&v119, v49, 1);
        v51 = *(_QWORD *)(v48 + 48);
        if ( !v51 )
          goto LABEL_40;
      }
      else
      {
        v51 = *(_QWORD *)(v48 + 48);
        v50 = v48 + 48;
        if ( !v51 )
          goto LABEL_42;
      }
      sub_B91220(v50, v51);
LABEL_40:
      v52 = v119;
      *(_QWORD *)(v48 + 48) = v119;
      if ( v52 )
        sub_B976B0((__int64)&v119, v52, v50);
LABEL_42:
      v53 = v94;
      LOWORD(v53) = 0;
      sub_B44220((_QWORD *)v48, v99, v53);
      v119 = (unsigned __int8 *)v48;
      sub_11B4E60(*(_QWORD *)(a1 + 40) + 2096LL, (__int64 *)&v119, v54, v55, v56, v57);
      v58 = *(_DWORD *)(v24 + 4) & 0x7FFFFFF;
      if ( v58 == *(_DWORD *)(v24 + 72) )
        goto LABEL_63;
      goto LABEL_43;
    }
    v91 = *(_QWORD *)(v64 + 32 * v115);
    v100 = *(v97 - 4);
    v124 = 257;
    v66 = sub_BD2C40(72, 2u);
    v67 = v91;
    v48 = (__int64)v66;
    if ( v66 )
    {
      sub_B4DE80((__int64)v66, v91, v100, (__int64)v123, 0, 0);
      v67 = v91;
    }
    if ( *(_BYTE *)v67 <= 0x1Cu || *(_BYTE *)v67 == 84 )
    {
      v68 = sub_AA5190(v107);
      if ( !v68 )
LABEL_81:
        BUG();
      v92 = v78;
      v101 = HIBYTE(v78);
    }
    else
    {
      v101 = 0;
      v68 = *(_QWORD *)(v67 + 32);
      v92 = 0;
    }
    if ( !v68 )
      goto LABEL_81;
    v69 = *(const char **)(v68 + 24);
    v123[0] = v69;
    if ( v69 )
    {
      v86 = v68;
      sub_B96E90((__int64)v123, (__int64)v69, 1);
      v70 = *(_QWORD *)(v48 + 48);
      v71 = v48 + 48;
      v68 = v86;
      if ( !v70 )
        goto LABEL_60;
    }
    else
    {
      v70 = *(_QWORD *)(v48 + 48);
      v71 = v48 + 48;
      if ( !v70 )
        goto LABEL_62;
    }
    v84 = v68;
    v87 = v71;
    sub_B91220(v71, v70);
    v68 = v84;
    v71 = v87;
LABEL_60:
    v72 = (unsigned __int8 *)v123[0];
    *(const char **)(v48 + 48) = v123[0];
    if ( v72 )
    {
      v88 = v68;
      sub_B976B0((__int64)v123, v72, v71);
      v68 = v88;
    }
LABEL_62:
    v73 = v93;
    LOBYTE(v73) = v92;
    BYTE1(v73) = v101;
    sub_B44220((_QWORD *)v48, v68, v73);
    v123[0] = (const char *)v48;
    sub_11B4E60(*(_QWORD *)(a1 + 40) + 2096LL, (__int64 *)v123, v74, v75, v76, v77);
    v58 = *(_DWORD *)(v24 + 4) & 0x7FFFFFF;
    if ( v58 == *(_DWORD *)(v24 + 72) )
    {
LABEL_63:
      sub_B48D90(v24);
      v58 = *(_DWORD *)(v24 + 4) & 0x7FFFFFF;
    }
LABEL_43:
    v59 = (v58 + 1) & 0x7FFFFFF;
    v60 = v59 | *(_DWORD *)(v24 + 4) & 0xF8000000;
    v61 = *(_QWORD *)(v24 - 8) + 32LL * (unsigned int)(v59 - 1);
    *(_DWORD *)(v24 + 4) = v60;
    if ( *(_QWORD *)v61 )
    {
      v62 = *(_QWORD *)(v61 + 8);
      **(_QWORD **)(v61 + 16) = v62;
      if ( v62 )
        *(_QWORD *)(v62 + 16) = *(_QWORD *)(v61 + 16);
    }
    *(_QWORD *)v61 = v48;
    v63 = *(_QWORD *)(v48 + 16);
    *(_QWORD *)(v61 + 8) = v63;
    if ( v63 )
      *(_QWORD *)(v63 + 16) = v61 + 8;
    *(_QWORD *)(v61 + 16) = v48 + 16;
    *(_QWORD *)(v48 + 16) = v61;
    a2 = (__int64 *)v107;
    *(_QWORD *)(*(_QWORD *)(v24 - 8)
              + 32LL * *(unsigned int *)(v24 + 72)
              + 8LL * ((*(_DWORD *)(v24 + 4) & 0x7FFFFFFu) - 1)) = v107;
    ++v115;
  }
  while ( (*(_DWORD *)(v33 + 4) & 0x7FFFFFFu) > (unsigned int)v115 );
  v8 = v97;
  v6 = v83;
LABEL_73:
  v15 = v120;
  v79 = &v120[(unsigned int)v121];
  if ( v79 != v120 )
  {
    v116 = v6;
    v80 = a1;
    v118 = v8;
    v81 = v120;
    do
    {
      v82 = *v81++;
      sub_F162A0(v80, v82, v24);
      a2 = (__int64 *)v82;
      sub_F15FC0(*(_QWORD *)(v80 + 40), v82);
    }
    while ( v79 != v81 );
    v8 = v118;
    v6 = v116;
    v15 = v120;
  }
  v7 = v8;
LABEL_15:
  if ( v15 != v6 )
LABEL_16:
    _libc_free(v15, a2);
  return v7;
}
