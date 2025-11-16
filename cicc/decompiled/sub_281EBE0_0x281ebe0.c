// Function: sub_281EBE0
// Address: 0x281ebe0
//
void __fastcall sub_281EBE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r14
  const char *v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rax
  const char **v16; // r14
  const char *v17; // rsi
  __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // r13
  unsigned int v22; // r14d
  const char *v23; // rsi
  const char **v24; // r13
  __int64 v25; // r14
  __int64 v26; // r9
  __int64 v27; // rax
  __int64 v28; // r8
  unsigned int v29; // r13d
  __int64 v30; // rax
  __int64 v31; // r9
  _QWORD *v32; // r14
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 *v35; // r14
  unsigned __int64 v36; // rbx
  __int64 v37; // r13
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rbx
  _QWORD *v41; // r15
  __int64 v42; // r15
  int v43; // edx
  int v44; // edx
  unsigned int v45; // ecx
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // rcx
  int v49; // edx
  int v50; // edx
  unsigned int v51; // ecx
  __int64 v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // rcx
  __int64 *v55; // rcx
  __int16 v56; // si
  bool v57; // zf
  __int64 v58; // rdx
  __int64 v59; // rdx
  __int64 v60; // rax
  __int64 v61; // rdx
  __int64 v62; // rdx
  __int64 v63; // rsi
  unsigned __int8 *v64; // rsi
  __int64 v65; // rsi
  unsigned __int8 *v66; // rsi
  __int64 v67; // rsi
  unsigned __int8 *v68; // rsi
  _QWORD **v69; // rdx
  int v70; // ecx
  int v71; // eax
  __int64 *v72; // rax
  __int64 v73; // rax
  __int64 v74; // r9
  __int64 v75; // r8
  unsigned int *v76; // r15
  unsigned int *v77; // r13
  __int64 v78; // rdx
  unsigned int v79; // esi
  __int64 v80; // r12
  unsigned int *v81; // r12
  unsigned int *v82; // rbx
  __int64 v83; // rdx
  unsigned int v84; // esi
  unsigned int *v85; // rbx
  unsigned int *v86; // r14
  __int64 v87; // rdx
  unsigned int v88; // esi
  __int64 v89; // [rsp+8h] [rbp-198h]
  __int64 v90; // [rsp+10h] [rbp-190h]
  __int64 v91; // [rsp+10h] [rbp-190h]
  __int64 v92; // [rsp+30h] [rbp-170h]
  unsigned __int64 v93; // [rsp+30h] [rbp-170h]
  __int64 v94; // [rsp+30h] [rbp-170h]
  __int64 v95; // [rsp+38h] [rbp-168h]
  __int64 v96; // [rsp+40h] [rbp-160h]
  __int64 v97; // [rsp+40h] [rbp-160h]
  char *v98; // [rsp+48h] [rbp-158h]
  __int64 v99; // [rsp+48h] [rbp-158h]
  __int64 v100; // [rsp+48h] [rbp-158h]
  __int64 v101; // [rsp+58h] [rbp-148h]
  __int64 *v103; // [rsp+68h] [rbp-138h]
  __int64 v104; // [rsp+68h] [rbp-138h]
  __int64 v105; // [rsp+70h] [rbp-130h] BYREF
  __int64 v106; // [rsp+78h] [rbp-128h] BYREF
  unsigned int v107[8]; // [rsp+80h] [rbp-120h] BYREF
  __int16 v108; // [rsp+A0h] [rbp-100h]
  const char *v109[2]; // [rsp+B0h] [rbp-F0h] BYREF
  void (__fastcall *v110)(const char **, const char **, __int64); // [rsp+C0h] [rbp-E0h]
  __int16 v111; // [rsp+D0h] [rbp-D0h]
  unsigned int *v112; // [rsp+E0h] [rbp-C0h] BYREF
  __int64 v113; // [rsp+E8h] [rbp-B8h]
  _BYTE v114[32]; // [rsp+F0h] [rbp-B0h] BYREF
  __int64 v115; // [rsp+110h] [rbp-90h]
  __int64 v116; // [rsp+118h] [rbp-88h]
  __int64 v117; // [rsp+120h] [rbp-80h]
  __int64 v118; // [rsp+128h] [rbp-78h]
  void **v119; // [rsp+130h] [rbp-70h]
  void **v120; // [rsp+138h] [rbp-68h]
  __int64 v121; // [rsp+140h] [rbp-60h]
  int v122; // [rsp+148h] [rbp-58h]
  __int16 v123; // [rsp+14Ch] [rbp-54h]
  char v124; // [rsp+14Eh] [rbp-52h]
  __int64 v125; // [rsp+150h] [rbp-50h]
  __int64 v126; // [rsp+158h] [rbp-48h]
  void *v127; // [rsp+160h] [rbp-40h] BYREF
  void *v128; // [rsp+168h] [rbp-38h] BYREF

  v95 = sub_D4B130(*(_QWORD *)a1);
  v7 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v7 == a2 + 48 )
    goto LABEL_115;
  if ( !v7 )
    BUG();
  v8 = v7 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v7 - 24) - 30 > 0xA )
  {
LABEL_115:
    v89 = -96;
    v8 = 0;
  }
  else
  {
    v89 = v7 - 120;
  }
  v9 = sub_BD5C60(v8);
  v124 = 7;
  v118 = v9;
  v119 = &v127;
  v120 = &v128;
  v112 = (unsigned int *)v114;
  v127 = &unk_49DA100;
  v113 = 0x200000000LL;
  v123 = 512;
  LOWORD(v117) = 0;
  v128 = &unk_49DA0B0;
  v121 = 0;
  v122 = 0;
  v125 = 0;
  v126 = 0;
  v115 = 0;
  v116 = 0;
  sub_D5F1F0((__int64)&v112, v8);
  v10 = *(_QWORD *)(a5 + 8);
  v107[1] = 0;
  v105 = a5;
  v106 = v10;
  v111 = 257;
  v11 = sub_B33D10((__int64)&v112, 0x42u, (__int64)&v106, 1, (int)&v105, 1, v107[0], (__int64)v109);
  v12 = *(const char **)(a3 + 48);
  v109[0] = v12;
  if ( !v12 )
  {
    v13 = v11 + 48;
    if ( (const char **)(v11 + 48) == v109 )
      goto LABEL_9;
    v67 = *(_QWORD *)(v11 + 48);
    if ( !v67 )
      goto LABEL_9;
LABEL_91:
    v104 = v13;
    sub_B91220(v13, v67);
    v13 = v104;
    goto LABEL_92;
  }
  sub_B96E90((__int64)v109, (__int64)v12, 1);
  v13 = v11 + 48;
  if ( (const char **)(v11 + 48) == v109 )
  {
    if ( v109[0] )
      sub_B91220((__int64)v109, (__int64)v109[0]);
    goto LABEL_9;
  }
  v67 = *(_QWORD *)(v11 + 48);
  if ( v67 )
    goto LABEL_91;
LABEL_92:
  v68 = (unsigned __int8 *)v109[0];
  *(const char **)(v11 + 48) = v109[0];
  if ( v68 )
    sub_B976B0((__int64)v109, v68, v13);
LABEL_9:
  v14 = *(_QWORD *)(a4 + 8);
  v111 = 257;
  v15 = sub_A830B0(&v112, v11, v14, (__int64)v109);
  v103 = (__int64 *)v15;
  if ( v15 == v11 )
    goto LABEL_14;
  v16 = (const char **)(v15 + 48);
  v17 = *(const char **)(a3 + 48);
  v109[0] = v17;
  if ( !v17 )
  {
    if ( v16 == v109 )
      goto LABEL_14;
    v65 = *(_QWORD *)(v15 + 48);
    if ( !v65 )
      goto LABEL_14;
LABEL_87:
    sub_B91220((__int64)v16, v65);
    goto LABEL_88;
  }
  sub_B96E90((__int64)v109, (__int64)v17, 1);
  if ( v16 == v109 )
  {
    if ( v109[0] )
      sub_B91220((__int64)v109, (__int64)v109[0]);
    goto LABEL_14;
  }
  v65 = v103[6];
  if ( v65 )
    goto LABEL_87;
LABEL_88:
  v66 = (unsigned __int8 *)v109[0];
  v103[6] = (__int64)v109[0];
  if ( v66 )
    sub_B976B0((__int64)v109, v66, (__int64)v16);
LABEL_14:
  v18 = *(_QWORD *)(a4 - 8);
  v19 = 0x1FFFFFFFE0LL;
  if ( (*(_DWORD *)(a4 + 4) & 0x7FFFFFF) != 0 )
  {
    v20 = 0;
    do
    {
      if ( v95 == *(_QWORD *)(v18 + 32LL * *(unsigned int *)(a4 + 72) + 8 * v20) )
      {
        v19 = 32 * v20;
        goto LABEL_19;
      }
      ++v20;
    }
    while ( (*(_DWORD *)(a4 + 4) & 0x7FFFFFF) != (_DWORD)v20 );
    v19 = 0x1FFFFFFFE0LL;
  }
LABEL_19:
  v21 = *(_QWORD *)(v18 + v19);
  if ( *(_BYTE *)v21 == 17 )
  {
    v22 = *(_DWORD *)(v21 + 32);
    if ( v22 <= 0x40 )
    {
      v101 = (__int64)v103;
      if ( !*(_QWORD *)(v21 + 24) )
        goto LABEL_27;
    }
    else
    {
      v101 = (__int64)v103;
      if ( v22 == (unsigned int)sub_C444A0(v21 + 24) )
        goto LABEL_27;
    }
  }
  v108 = 257;
  v101 = (*((__int64 (__fastcall **)(void **, __int64, __int64 *, __int64, _QWORD, _QWORD))*v119 + 4))(
           v119,
           13,
           v103,
           v21,
           0,
           0);
  if ( !v101 )
  {
    v111 = 257;
    v101 = sub_B504D0(13, (__int64)v103, v21, (__int64)v109, 0, 0);
    (*((void (__fastcall **)(void **, __int64, unsigned int *, __int64, __int64))*v120 + 2))(
      v120,
      v101,
      v107,
      v116,
      v117);
    if ( v112 != &v112[4 * (unsigned int)v113] )
    {
      v100 = v8;
      v85 = v112;
      v86 = &v112[4 * (unsigned int)v113];
      do
      {
        v87 = *((_QWORD *)v85 + 1);
        v88 = *v85;
        v85 += 4;
        sub_B99FD0(v101, v88, v87);
      }
      while ( v86 != v85 );
      v8 = v100;
    }
  }
  v23 = *(const char **)(a3 + 48);
  v24 = (const char **)(v101 + 48);
  v109[0] = v23;
  if ( !v23 )
  {
    if ( v24 == v109 )
      goto LABEL_27;
    v63 = *(_QWORD *)(v101 + 48);
    if ( !v63 )
      goto LABEL_27;
LABEL_82:
    sub_B91220((__int64)v24, v63);
    goto LABEL_83;
  }
  sub_B96E90((__int64)v109, (__int64)v23, 1);
  if ( v24 == v109 )
  {
    if ( v109[0] )
      sub_B91220((__int64)v109, (__int64)v109[0]);
    goto LABEL_27;
  }
  v63 = *(_QWORD *)(v101 + 48);
  if ( v63 )
    goto LABEL_82;
LABEL_83:
  v64 = (unsigned __int8 *)v109[0];
  *(const char **)(v101 + 48) = v109[0];
  if ( v64 )
    sub_B976B0((__int64)v109, v64, (__int64)v24);
LABEL_27:
  v25 = *(_QWORD *)(v8 - 96);
  v98 = (char *)v25;
  v26 = sub_AD64C0(v103[1], 0, 0);
  v27 = *(_QWORD *)(v25 - 64);
  if ( !v27 || (v28 = (__int64)v103, v27 != a5) )
  {
    v28 = v26;
    v26 = (__int64)v103;
  }
  v90 = v26;
  v108 = 257;
  v92 = v28;
  v29 = *(_WORD *)(v25 + 2) & 0x3F;
  v30 = (*((__int64 (__fastcall **)(void **, _QWORD, __int64, __int64))*v119 + 7))(v119, v29, v28, v26);
  v31 = v90;
  v32 = (_QWORD *)v30;
  if ( !v30 )
  {
    v91 = v92;
    v94 = v31;
    v111 = 257;
    v32 = sub_BD2C40(72, unk_3F10FD0);
    if ( v32 )
    {
      v69 = *(_QWORD ***)(v91 + 8);
      v70 = *((unsigned __int8 *)v69 + 8);
      if ( (unsigned int)(v70 - 17) > 1 )
      {
        v73 = sub_BCB2A0(*v69);
        v75 = v91;
        v74 = v94;
      }
      else
      {
        v71 = *((_DWORD *)v69 + 8);
        BYTE4(v106) = (_BYTE)v70 == 18;
        LODWORD(v106) = v71;
        v72 = (__int64 *)sub_BCB2A0(*v69);
        v73 = sub_BCE1B0(v72, v106);
        v74 = v94;
        v75 = v91;
      }
      sub_B523C0((__int64)v32, v73, 53, v29, v75, v74, (__int64)v109, 0, 0, 0);
    }
    (*((void (__fastcall **)(void **, _QWORD *, unsigned int *, __int64, __int64))*v120 + 2))(
      v120,
      v32,
      v107,
      v116,
      v117);
    v76 = v112;
    v77 = &v112[4 * (unsigned int)v113];
    if ( v112 != v77 )
    {
      do
      {
        v78 = *((_QWORD *)v76 + 1);
        v79 = *v76;
        v76 += 4;
        sub_B99FD0((__int64)v32, v79, v78);
      }
      while ( v77 != v76 );
    }
    if ( !*(_QWORD *)(v8 - 96) || (v33 = *(_QWORD *)(v8 - 88), (**(_QWORD **)(v8 - 80) = v33) == 0) )
    {
LABEL_33:
      *(_QWORD *)(v8 - 96) = v32;
      if ( !v32 )
        goto LABEL_37;
      goto LABEL_34;
    }
LABEL_32:
    *(_QWORD *)(v33 + 16) = *(_QWORD *)(v8 - 80);
    goto LABEL_33;
  }
  if ( *(_QWORD *)(v8 - 96) )
  {
    v33 = *(_QWORD *)(v8 - 88);
    **(_QWORD **)(v8 - 80) = v33;
    if ( v33 )
      goto LABEL_32;
  }
  *(_QWORD *)(v8 - 96) = v32;
LABEL_34:
  v34 = v32[2];
  *(_QWORD *)(v8 - 88) = v34;
  if ( v34 )
    *(_QWORD *)(v34 + 16) = v8 - 88;
  *(_QWORD *)(v8 - 80) = v32 + 2;
  v32[2] = v89;
LABEL_37:
  v110 = 0;
  sub_F5CAB0(v98, *(__int64 **)(a1 + 40), 0, (__int64)v109);
  if ( v110 )
    v110(v109, v109, 3);
  v35 = **(__int64 ***)(*(_QWORD *)a1 + 32LL);
  v36 = v35[6] & 0xFFFFFFFFFFFFFFF8LL;
  v93 = v36;
  if ( (__int64 *)v36 == v35 + 6 )
    goto LABEL_129;
  if ( !v36 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v36 - 24) - 30 > 0xA )
LABEL_129:
    BUG();
  v37 = *(_QWORD *)(v36 - 120);
  v38 = v103[1];
  v111 = 259;
  v99 = v38;
  v109[0] = "tcphi";
  v39 = sub_BD2DA0(80);
  v40 = v39;
  if ( v39 )
  {
    v41 = (_QWORD *)v39;
    sub_B44260(v39, v99, 55, 0x8000000u, 0, 0);
    *(_DWORD *)(v40 + 72) = 2;
    sub_BD6B50((unsigned __int8 *)v40, v109);
    sub_BD2A10(v40, *(_DWORD *)(v40 + 72), 1);
  }
  else
  {
    v41 = 0;
  }
  sub_B44220(v41, v35[7], 1);
  sub_D5F1F0((__int64)&v112, v37);
  *(_QWORD *)v107 = "tcdec";
  v108 = 259;
  v96 = sub_AD64C0(v99, 1, 0);
  v42 = (*((__int64 (__fastcall **)(void **, __int64, __int64, __int64, _QWORD, __int64))*v119 + 4))(
          v119,
          15,
          v40,
          v96,
          0,
          1);
  if ( v42 )
  {
    v43 = *(_DWORD *)(v40 + 4) & 0x7FFFFFF;
    if ( v43 != *(_DWORD *)(v40 + 72) )
      goto LABEL_46;
LABEL_114:
    sub_B48D90(v40);
    v43 = *(_DWORD *)(v40 + 4) & 0x7FFFFFF;
    goto LABEL_46;
  }
  v111 = 257;
  v42 = sub_B504D0(15, v40, v96, (__int64)v109, 0, 0);
  (*((void (__fastcall **)(void **, __int64, unsigned int *, __int64, __int64))*v120 + 2))(v120, v42, v107, v116, v117);
  v80 = 4LL * (unsigned int)v113;
  if ( v112 != &v112[v80] )
  {
    v97 = v40;
    v81 = &v112[v80];
    v82 = v112;
    do
    {
      v83 = *((_QWORD *)v82 + 1);
      v84 = *v82;
      v82 += 4;
      sub_B99FD0(v42, v84, v83);
    }
    while ( v81 != v82 );
    v40 = v97;
  }
  sub_B44850((unsigned __int8 *)v42, 1);
  v43 = *(_DWORD *)(v40 + 4) & 0x7FFFFFF;
  if ( v43 == *(_DWORD *)(v40 + 72) )
    goto LABEL_114;
LABEL_46:
  v44 = (v43 + 1) & 0x7FFFFFF;
  v45 = v44 | *(_DWORD *)(v40 + 4) & 0xF8000000;
  v46 = *(_QWORD *)(v40 - 8) + 32LL * (unsigned int)(v44 - 1);
  *(_DWORD *)(v40 + 4) = v45;
  if ( *(_QWORD *)v46 )
  {
    v47 = *(_QWORD *)(v46 + 8);
    **(_QWORD **)(v46 + 16) = v47;
    if ( v47 )
      *(_QWORD *)(v47 + 16) = *(_QWORD *)(v46 + 16);
  }
  *(_QWORD *)v46 = v103;
  v48 = v103[2];
  *(_QWORD *)(v46 + 8) = v48;
  if ( v48 )
    *(_QWORD *)(v48 + 16) = v46 + 8;
  *(_QWORD *)(v46 + 16) = v103 + 2;
  v103[2] = v46;
  *(_QWORD *)(*(_QWORD *)(v40 - 8) + 32LL * *(unsigned int *)(v40 + 72)
                                   + 8LL * ((*(_DWORD *)(v40 + 4) & 0x7FFFFFFu) - 1)) = v95;
  v49 = *(_DWORD *)(v40 + 4) & 0x7FFFFFF;
  if ( v49 == *(_DWORD *)(v40 + 72) )
  {
    sub_B48D90(v40);
    v49 = *(_DWORD *)(v40 + 4) & 0x7FFFFFF;
  }
  v50 = (v49 + 1) & 0x7FFFFFF;
  v51 = v50 | *(_DWORD *)(v40 + 4) & 0xF8000000;
  v52 = *(_QWORD *)(v40 - 8) + 32LL * (unsigned int)(v50 - 1);
  *(_DWORD *)(v40 + 4) = v51;
  if ( *(_QWORD *)v52 )
  {
    v53 = *(_QWORD *)(v52 + 8);
    **(_QWORD **)(v52 + 16) = v53;
    if ( v53 )
      *(_QWORD *)(v53 + 16) = *(_QWORD *)(v52 + 16);
  }
  *(_QWORD *)v52 = v42;
  if ( v42 )
  {
    v54 = *(_QWORD *)(v42 + 16);
    *(_QWORD *)(v52 + 8) = v54;
    if ( v54 )
      *(_QWORD *)(v54 + 16) = v52 + 8;
    *(_QWORD *)(v52 + 16) = v42 + 16;
    *(_QWORD *)(v42 + 16) = v52;
  }
  *(_QWORD *)(*(_QWORD *)(v40 - 8) + 32LL * *(unsigned int *)(v40 + 72)
                                   + 8LL * ((*(_DWORD *)(v40 + 4) & 0x7FFFFFFu) - 1)) = v35;
  v55 = *(__int64 **)(v93 - 56);
  if ( v35 != v55 || (v56 = 34, !v55) )
    v56 = 41;
  v57 = *(_QWORD *)(v37 - 64) == 0;
  *(_WORD *)(v37 + 2) = v56 | *(_WORD *)(v37 + 2) & 0xFFC0;
  if ( !v57 )
  {
    v58 = *(_QWORD *)(v37 - 56);
    **(_QWORD **)(v37 - 48) = v58;
    if ( v58 )
      *(_QWORD *)(v58 + 16) = *(_QWORD *)(v37 - 48);
  }
  *(_QWORD *)(v37 - 64) = v42;
  if ( v42 )
  {
    v59 = *(_QWORD *)(v42 + 16);
    *(_QWORD *)(v37 - 56) = v59;
    if ( v59 )
      *(_QWORD *)(v59 + 16) = v37 - 56;
    *(_QWORD *)(v37 - 48) = v42 + 16;
    *(_QWORD *)(v42 + 16) = v37 - 64;
  }
  v60 = sub_AD64C0(v99, 0, 0);
  if ( *(_QWORD *)(v37 - 32) )
  {
    v61 = *(_QWORD *)(v37 - 24);
    **(_QWORD **)(v37 - 16) = v61;
    if ( v61 )
      *(_QWORD *)(v61 + 16) = *(_QWORD *)(v37 - 16);
  }
  *(_QWORD *)(v37 - 32) = v60;
  if ( v60 )
  {
    v62 = *(_QWORD *)(v60 + 16);
    *(_QWORD *)(v37 - 24) = v62;
    if ( v62 )
      *(_QWORD *)(v62 + 16) = v37 - 24;
    *(_QWORD *)(v37 - 16) = v60 + 16;
    *(_QWORD *)(v60 + 16) = v37 - 32;
  }
  sub_BD7E80((unsigned __int8 *)a3, (unsigned __int8 *)v101, v35);
  sub_DAC210(*(_QWORD *)(a1 + 32), *(_QWORD *)a1);
  nullsub_61();
  v127 = &unk_49DA100;
  nullsub_63();
  if ( v112 != (unsigned int *)v114 )
    _libc_free((unsigned __int64)v112);
}
