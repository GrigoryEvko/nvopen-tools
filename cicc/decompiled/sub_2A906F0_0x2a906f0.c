// Function: sub_2A906F0
// Address: 0x2a906f0
//
__int64 __fastcall sub_2A906F0(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 *a4, __int64 a5, int a6)
{
  _QWORD *v8; // r12
  unsigned int v9; // eax
  unsigned int v10; // ebx
  unsigned __int8 *v11; // r13
  unsigned __int8 *v12; // r14
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 *v18; // rax
  _QWORD *v19; // rsi
  __int64 v20; // rax
  unsigned int v21; // edx
  char *v22; // r9
  bool v23; // al
  unsigned int v24; // r14d
  char *v25; // r13
  bool v26; // r12
  char **v27; // rsi
  unsigned int v28; // eax
  unsigned int v29; // eax
  char *v30; // rax
  unsigned int v31; // eax
  unsigned __int64 v32; // rax
  unsigned __int8 v33; // dl
  __int64 v34; // rax
  __int64 v35; // rax
  unsigned __int8 *v36; // rdx
  __int64 v37; // rax
  unsigned __int8 **v38; // rcx
  unsigned __int8 **v39; // r8
  __int64 v40; // r14
  int v41; // eax
  unsigned __int8 **v42; // rbx
  unsigned __int8 **v43; // r13
  unsigned __int64 v44; // rdi
  unsigned __int64 v45; // rax
  __int64 v46; // r14
  unsigned __int8 v47; // dl
  unsigned __int64 v48; // rdi
  unsigned __int64 v49; // rdx
  __int64 v50; // rax
  unsigned __int8 v51; // al
  unsigned int v52; // eax
  unsigned int v53; // eax
  unsigned __int64 v54; // rax
  __int64 v55; // rax
  unsigned __int64 v56; // rax
  bool v57; // zf
  unsigned __int64 v58; // rax
  unsigned __int8 v59; // dl
  unsigned __int8 *v60; // r14
  unsigned __int8 v61; // al
  __int64 v62; // rdx
  _BYTE **v63; // rax
  _BYTE *v64; // r13
  __int64 v65; // rax
  __int64 *v66; // r14
  __int64 *v67; // rax
  _QWORD *v68; // r14
  __int64 v69; // r14
  __int64 *v70; // rax
  unsigned int v71; // r10d
  __int64 v72; // rdx
  __int64 v73; // r8
  unsigned __int64 v74; // r12
  __int64 v75; // rdi
  unsigned int v76; // eax
  __int64 v77; // rdx
  unsigned int v78; // r12d
  __int64 v79; // rax
  unsigned int v80; // eax
  __int64 v81; // rbx
  __int64 v82; // rdx
  _BYTE *v83; // rcx
  __int64 v84; // rsi
  __int64 v85; // rax
  bool v86; // al
  __int64 v87; // r14
  __int64 v88; // r12
  unsigned __int8 v89; // al
  __int64 v90; // r15
  __int64 v91; // rsi
  _QWORD *v92; // r14
  __int64 v93; // rcx
  __int64 v94; // r15
  __int64 v95; // r9
  __int64 v96; // rax
  __int64 v97; // r9
  __int64 v98; // r15
  __int64 v99; // rsi
  __int64 v100; // rax
  __int64 v101; // r8
  __int64 v102; // rax
  __int64 v103; // r10
  _QWORD *v104; // [rsp+10h] [rbp-170h]
  __int64 v105; // [rsp+18h] [rbp-168h]
  __int64 v106; // [rsp+20h] [rbp-160h]
  unsigned __int64 v107; // [rsp+28h] [rbp-158h]
  unsigned int v108; // [rsp+40h] [rbp-140h]
  bool v109; // [rsp+47h] [rbp-139h]
  char v110; // [rsp+48h] [rbp-138h]
  unsigned int v111; // [rsp+4Ch] [rbp-134h]
  char *v112; // [rsp+50h] [rbp-130h]
  unsigned int v113; // [rsp+50h] [rbp-130h]
  __int64 v114; // [rsp+50h] [rbp-130h]
  unsigned int v115; // [rsp+58h] [rbp-128h]
  bool v116; // [rsp+58h] [rbp-128h]
  __int64 v117; // [rsp+58h] [rbp-128h]
  __int64 v118; // [rsp+58h] [rbp-128h]
  __int64 *v119; // [rsp+60h] [rbp-120h]
  unsigned __int8 v120; // [rsp+60h] [rbp-120h]
  __int64 *v121; // [rsp+68h] [rbp-118h]
  __int64 v122; // [rsp+68h] [rbp-118h]
  unsigned __int8 *v124; // [rsp+70h] [rbp-110h]
  char v125; // [rsp+70h] [rbp-110h]
  __int64 v126; // [rsp+70h] [rbp-110h]
  bool v127; // [rsp+70h] [rbp-110h]
  unsigned int v129; // [rsp+90h] [rbp-F0h]
  unsigned __int8 *v130; // [rsp+90h] [rbp-F0h]
  unsigned __int8 **v131; // [rsp+90h] [rbp-F0h]
  unsigned int v132; // [rsp+90h] [rbp-F0h]
  unsigned __int8 *v133; // [rsp+90h] [rbp-F0h]
  __int64 v134; // [rsp+90h] [rbp-F0h]
  int v135; // [rsp+90h] [rbp-F0h]
  char *v137; // [rsp+A0h] [rbp-E0h] BYREF
  unsigned int v138; // [rsp+A8h] [rbp-D8h]
  char *v139; // [rsp+B0h] [rbp-D0h] BYREF
  unsigned int v140; // [rsp+B8h] [rbp-C8h]
  char *v141; // [rsp+C0h] [rbp-C0h] BYREF
  signed __int64 v142; // [rsp+C8h] [rbp-B8h]
  char *v143; // [rsp+D0h] [rbp-B0h] BYREF
  unsigned int v144; // [rsp+D8h] [rbp-A8h]
  __int64 v145; // [rsp+E0h] [rbp-A0h] BYREF
  unsigned int v146; // [rsp+E8h] [rbp-98h]
  __int64 v147; // [rsp+F0h] [rbp-90h] BYREF
  unsigned int v148; // [rsp+F8h] [rbp-88h]
  char v149; // [rsp+100h] [rbp-80h]
  char *v150; // [rsp+110h] [rbp-70h] BYREF
  unsigned int v151; // [rsp+118h] [rbp-68h]
  __int64 v152; // [rsp+120h] [rbp-60h] BYREF
  unsigned int v153; // [rsp+128h] [rbp-58h]
  unsigned __int64 v154; // [rsp+130h] [rbp-50h] BYREF
  __int64 v155; // [rsp+138h] [rbp-48h]
  char *v156; // [rsp+140h] [rbp-40h] BYREF
  unsigned int v157; // [rsp+148h] [rbp-38h]

  v8 = (_QWORD *)a2;
  v9 = sub_AE43F0(*(_QWORD *)(a2 + 48), *(_QWORD *)(a3 + 8));
  v138 = v9;
  v10 = v9;
  if ( v9 > 0x40 )
  {
    sub_C43690((__int64)&v137, 0, 0);
    v140 = v10;
    sub_C43690((__int64)&v139, 0, 0);
  }
  else
  {
    v140 = v9;
    v137 = 0;
    v139 = 0;
  }
  v11 = sub_BD45C0((unsigned __int8 *)a3, *(_QWORD *)(a2 + 48), (__int64)&v137, 0, 0, 0, 0, 0);
  v12 = sub_BD45C0(a4, *(_QWORD *)(a2 + 48), (__int64)&v139, 0, 0, 0, 0, 0);
  v13 = sub_9208B0(*(_QWORD *)(a2 + 48), *((_QWORD *)v11 + 1));
  v155 = v14;
  v154 = (v13 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  v129 = sub_CA1930(&v154);
  v15 = sub_9208B0(*(_QWORD *)(a2 + 48), *((_QWORD *)v12 + 1));
  v155 = v16;
  v154 = (v15 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( sub_CA1930(&v154) != v129 )
    goto LABEL_4;
  sub_C44B10((__int64)&v154, &v137, v129);
  if ( v138 > 0x40 && v137 )
    j_j___libc_free_0_0((unsigned __int64)v137);
  v137 = (char *)v154;
  v138 = v155;
  sub_C44B10((__int64)&v154, &v139, v129);
  if ( v140 > 0x40 && v139 )
    j_j___libc_free_0_0((unsigned __int64)v139);
  v139 = (char *)v154;
  v140 = v155;
  if ( v12 == v11 )
  {
    v148 = v155;
    if ( (unsigned int)v155 > 0x40 )
      sub_C43780((__int64)&v147, (const void **)&v139);
    else
      v147 = v154;
    sub_C46B40((__int64)&v147, (__int64 *)&v137);
    v31 = v148;
    v148 = 0;
    v151 = v31;
    v150 = (char *)v147;
    sub_C44B10((__int64)&v154, &v150, v10);
    *(_DWORD *)(a1 + 8) = v155;
    v32 = v154;
    *(_BYTE *)(a1 + 16) = 1;
    *(_QWORD *)a1 = v32;
    sub_969240((__int64 *)&v150);
    sub_969240(&v147);
    goto LABEL_5;
  }
  v119 = *(__int64 **)(a2 + 32);
  v121 = sub_DD8400((__int64)v119, (__int64)v11);
  v18 = sub_DD8400(*(_QWORD *)(a2 + 32), (__int64)v12);
  v19 = sub_DCC810(v119, (__int64)v18, (__int64)v121, 0, 0);
  if ( v19 != (_QWORD *)sub_D970F0(v8[4]) )
  {
    v20 = sub_DBB9F0(v8[4], (__int64)v19, 1u, 0);
    LODWORD(v155) = *(_DWORD *)(v20 + 8);
    if ( (unsigned int)v155 > 0x40 )
    {
      v122 = v20;
      sub_C43780((__int64)&v154, (const void **)v20);
      v20 = v122;
    }
    else
    {
      v154 = *(_QWORD *)v20;
    }
    v157 = *(_DWORD *)(v20 + 24);
    if ( v157 > 0x40 )
      sub_C43780((__int64)&v156, (const void **)(v20 + 16));
    else
      v156 = *(char **)(v20 + 16);
    v151 = v155;
    if ( (unsigned int)v155 > 0x40 )
      sub_C43780((__int64)&v150, (const void **)&v154);
    else
      v150 = (char *)v154;
    sub_C46A40((__int64)&v150, 1);
    v21 = v151;
    v22 = v150;
    v151 = 0;
    v148 = v21;
    v147 = (__int64)v150;
    if ( v157 <= 0x40 )
    {
      v23 = v156 == v150;
    }
    else
    {
      v112 = v150;
      v115 = v21;
      v23 = sub_C43C50((__int64)&v156, (const void **)&v147);
      v22 = v112;
      v21 = v115;
    }
    if ( v21 > 0x40 )
    {
      if ( v22 )
      {
        v116 = v23;
        j_j___libc_free_0_0((unsigned __int64)v22);
        v23 = v116;
        if ( v151 > 0x40 )
        {
          if ( v150 )
          {
            j_j___libc_free_0_0((unsigned __int64)v150);
            v23 = v116;
          }
        }
      }
    }
    if ( v23 )
    {
      v151 = v155;
      if ( (unsigned int)v155 > 0x40 )
        sub_C43780((__int64)&v150, (const void **)&v154);
      else
        v150 = (char *)v154;
      sub_C46A40((__int64)&v150, 1);
      v24 = v151;
      v25 = v150;
      v151 = 0;
      v148 = v24;
      v147 = (__int64)v150;
      if ( v157 <= 0x40 )
        v26 = v156 == v150;
      else
        v26 = sub_C43C50((__int64)&v156, (const void **)&v147);
      if ( v24 > 0x40 )
      {
        if ( v25 )
        {
          j_j___libc_free_0_0((unsigned __int64)v25);
          if ( v151 > 0x40 )
          {
            if ( v150 )
              j_j___libc_free_0_0((unsigned __int64)v150);
          }
        }
      }
      v27 = 0;
      if ( v26 )
        v27 = (char **)&v154;
      sub_C44B10((__int64)&v141, v27, v129);
      v144 = v140;
      if ( v140 > 0x40 )
        sub_C43780((__int64)&v143, (const void **)&v139);
      else
        v143 = v139;
      sub_C46B40((__int64)&v143, (__int64 *)&v137);
      v28 = v144;
      v144 = 0;
      v146 = v28;
      v145 = (__int64)v143;
      sub_C45EE0((__int64)&v145, (__int64 *)&v141);
      v29 = v146;
      v146 = 0;
      v148 = v29;
      v147 = v145;
      sub_C44B10((__int64)&v150, (char **)&v147, v10);
      *(_DWORD *)(a1 + 8) = v151;
      v30 = v150;
      *(_BYTE *)(a1 + 16) = 1;
      *(_QWORD *)a1 = v30;
      sub_969240(&v147);
      sub_969240(&v145);
      sub_969240((__int64 *)&v143);
      sub_969240((__int64 *)&v141);
      sub_969240((__int64 *)&v156);
      sub_969240((__int64 *)&v154);
      goto LABEL_5;
    }
    sub_969240((__int64 *)&v156);
    sub_969240((__int64 *)&v154);
  }
  v33 = *v12;
  if ( *v11 != 63 )
  {
    if ( v33 == 63 || a6 == 3 || *v11 != 86 || v33 != 86 )
      goto LABEL_4;
    if ( *((_QWORD *)v11 - 12) != *((_QWORD *)v12 - 12)
      || (sub_2A906F0(&v150, v8, *((_QWORD *)v11 - 8), *((_QWORD *)v12 - 8), a5), !(_BYTE)v152) )
    {
      v149 = 0;
      goto LABEL_83;
    }
    sub_2A906F0(&v154, v8, *((_QWORD *)v11 - 4), *((_QWORD *)v12 - 4), a5);
    if ( (_BYTE)v152 != (_BYTE)v156 )
    {
      if ( !(_BYTE)v156 )
        goto LABEL_140;
      goto LABEL_146;
    }
    if ( !(_BYTE)v152 )
    {
      v149 = 0;
LABEL_151:
      if ( (_BYTE)v152 )
      {
        LOBYTE(v152) = 0;
        sub_969240((__int64 *)&v150);
      }
      goto LABEL_83;
    }
    v78 = v151;
    if ( v151 <= 0x40 )
    {
      if ( v150 == (char *)v154 )
      {
LABEL_150:
        v149 = 1;
        v148 = v78;
        v147 = (__int64)v150;
        v151 = 0;
        LOBYTE(v156) = 0;
        sub_969240((__int64 *)&v154);
        goto LABEL_151;
      }
    }
    else if ( sub_C43C50((__int64)&v150, (const void **)&v154) )
    {
      goto LABEL_150;
    }
LABEL_146:
    LOBYTE(v156) = 0;
    sub_969240((__int64 *)&v154);
LABEL_140:
    if ( (_BYTE)v152 )
    {
      LOBYTE(v152) = 0;
      sub_969240((__int64 *)&v150);
    }
    goto LABEL_4;
  }
  if ( v33 != 63 )
    goto LABEL_4;
  v34 = *((_DWORD *)v11 + 1) & 0x7FFFFFF;
  if ( (_DWORD)v34 != (*((_DWORD *)v12 + 1) & 0x7FFFFFF) || *(_QWORD *)&v11[-32 * v34] != *(_QWORD *)&v12[-32 * v34] )
    goto LABEL_4;
  if ( (v11[7] & 0x40) != 0 )
    v130 = (unsigned __int8 *)*((_QWORD *)v11 - 1);
  else
    v130 = &v11[-32 * v34];
  v35 = sub_BB5290((__int64)v11);
  v141 = (char *)(v130 + 32);
  v142 = v35 & 0xFFFFFFFFFFFFFFF9LL | 4;
  if ( (v12[7] & 0x40) != 0 )
    v36 = (unsigned __int8 *)*((_QWORD *)v12 - 1);
  else
    v36 = &v12[-32 * (*((_DWORD *)v12 + 1) & 0x7FFFFFF)];
  v124 = v36;
  v131 = (unsigned __int8 **)(v36 + 32);
  v37 = sub_BB5290((__int64)v12);
  v38 = (unsigned __int8 **)v141;
  v39 = v131;
  v40 = v37 & 0xFFFFFFFFFFFFFFF9LL | 4;
  v41 = *((_DWORD *)v11 + 1) & 0x7FFFFFF;
  if ( v41 == 2 )
  {
    v43 = v131;
    goto LABEL_113;
  }
  v132 = v10;
  v42 = v39;
  v43 = (unsigned __int8 **)&v124[32 * (v41 - 3) + 64];
  do
  {
    if ( *v38 != *v42 )
      goto LABEL_4;
    v48 = v142 & 0xFFFFFFFFFFFFFFF8LL;
    v49 = v142 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v142 )
      goto LABEL_95;
    v50 = (v142 >> 1) & 3;
    if ( v50 == 2 )
    {
      if ( v48 )
        goto LABEL_74;
LABEL_95:
      v55 = sub_BCBAE0(v48, *v38, v49);
      v38 = (unsigned __int8 **)v141;
      v49 = v55;
      goto LABEL_74;
    }
    if ( v50 != 1 || !v48 )
      goto LABEL_95;
    v49 = *(_QWORD *)(v48 + 24);
LABEL_74:
    v51 = *(_BYTE *)(v49 + 8);
    if ( v51 == 16 )
    {
      v142 = *(_QWORD *)(v49 + 24) & 0xFFFFFFFFFFFFFFF9LL | 4;
    }
    else
    {
      v49 &= 0xFFFFFFFFFFFFFFF9LL;
      if ( (unsigned int)v51 - 17 > 1 )
      {
        v57 = v51 == 15;
        v58 = 0;
        if ( v57 )
          v58 = v49;
        v142 = v58;
      }
      else
      {
        v49 |= 2u;
        v142 = v49;
      }
    }
    v38 += 4;
    v44 = v40 & 0xFFFFFFFFFFFFFFF8LL;
    v141 = (char *)v38;
    v45 = v40 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v40 )
      goto LABEL_94;
    v46 = (v40 >> 1) & 3;
    if ( v46 == 2 )
    {
      if ( v44 )
        goto LABEL_67;
LABEL_94:
      v45 = sub_BCBAE0(v44, *v42, v49);
      v38 = (unsigned __int8 **)v141;
      goto LABEL_67;
    }
    if ( v46 != 1 || !v44 )
      goto LABEL_94;
    v45 = *(_QWORD *)(v44 + 24);
LABEL_67:
    v47 = *(_BYTE *)(v45 + 8);
    if ( v47 == 16 )
    {
      v40 = *(_QWORD *)(v45 + 24) & 0xFFFFFFFFFFFFFFF9LL | 4;
    }
    else
    {
      v56 = v45 & 0xFFFFFFFFFFFFFFF9LL;
      if ( (unsigned int)v47 - 17 > 1 )
      {
        v40 = 0;
        if ( v47 == 15 )
          v40 = v56;
      }
      else
      {
        v40 = v56 | 2;
      }
    }
    v42 += 4;
  }
  while ( v42 != v43 );
  v10 = v132;
LABEL_113:
  v59 = **v38;
  if ( v59 <= 0x1Cu )
    goto LABEL_4;
  v60 = *v43;
  v61 = **v43;
  if ( v61 <= 0x1Cu )
    goto LABEL_4;
  if ( v59 != v61 )
    goto LABEL_4;
  if ( *((_QWORD *)*v38 + 1) != *((_QWORD *)v60 + 1) )
    goto LABEL_4;
  v133 = *v38;
  v154 = sub_9914A0((__int64)&v141, v8[6]);
  v155 = v62;
  v107 = sub_CA1930(&v154);
  v120 = *v133;
  if ( (unsigned __int8)(*v133 - 68) > 1u )
    goto LABEL_4;
  v63 = (_BYTE **)sub_986520((__int64)v60);
  v64 = *v63;
  if ( **v63 <= 0x1Cu )
    goto LABEL_4;
  v65 = *(_QWORD *)sub_986520((__int64)v133);
  v134 = v65;
  if ( *(_QWORD *)(v65 + 8) != *((_QWORD *)v64 + 1) )
    goto LABEL_4;
  v66 = sub_DD8400(v8[4], v65);
  v67 = sub_DD8400(v8[4], (__int64)v64);
  v68 = sub_DCC810((__int64 *)v8[4], (__int64)v67, (__int64)v66, 0, 0);
  if ( v68 == (_QWORD *)sub_D970F0(v8[4]) )
    goto LABEL_4;
  v69 = sub_DBB9F0(v8[4], (__int64)v68, 1u, 0);
  v151 = *(_DWORD *)(v69 + 8);
  if ( v151 > 0x40 )
    sub_C43780((__int64)&v150, (const void **)v69);
  else
    v150 = *(char **)v69;
  v153 = *(_DWORD *)(v69 + 24);
  if ( v153 > 0x40 )
    sub_C43780((__int64)&v152, (const void **)(v69 + 16));
  else
    v152 = *(_QWORD *)(v69 + 16);
  if ( !sub_9876C0((__int64 *)&v150) )
  {
    v149 = 0;
    goto LABEL_135;
  }
  v125 = v120 == 69;
  v70 = sub_9876C0((__int64 *)&v150);
  sub_9865C0((__int64)&v143, (__int64)v70);
  if ( *v64 != 42 )
  {
LABEL_127:
    v71 = sub_BCB060(*(_QWORD *)(v134 + 8));
    goto LABEL_128;
  }
  v79 = *(_QWORD *)(sub_986520((__int64)v64) + 32);
  if ( *(_BYTE *)v79 != 17
    || (v85 = sub_2A8A780(*(_QWORD *)(v79 + 24), *(_DWORD *)(v79 + 32)), sub_AAD930((__int64)&v143, v85)) )
  {
    if ( *(_BYTE *)v134 <= 0x1Cu )
      goto LABEL_127;
  }
  else
  {
    v86 = sub_2A8A3A0((__int64)v64, v125);
    if ( *(_BYTE *)v134 <= 0x1Cu )
    {
      if ( v86 )
        goto LABEL_133;
      goto LABEL_127;
    }
    if ( v86 )
      goto LABEL_133;
  }
  if ( *(_BYTE *)v134 != 42 )
    goto LABEL_127;
  v110 = v120 == 69;
  if ( !sub_2A8A3A0(v134, v125) )
    goto LABEL_127;
  v109 = sub_2A8A3A0((__int64)v64, v125);
  if ( !v109 )
    goto LABEL_127;
  v127 = 0;
  v108 = v10;
  v154 = 0x100000000LL;
  v118 = 0;
  v114 = (__int64)v143;
  v104 = v8;
  v111 = v144;
  while ( 2 )
  {
    v80 = *(_DWORD *)((char *)&v154 + v118);
    v147 = 0x100000000LL;
    v106 = 32LL * v80;
    v81 = 0;
    v105 = 32LL * (v80 != 1);
    while ( 2 )
    {
      if ( !v127 )
      {
        v82 = (*(_BYTE *)(v134 + 7) & 0x40) != 0
            ? *(_QWORD *)(v134 - 8)
            : v134 - 32LL * (*(_DWORD *)(v134 + 4) & 0x7FFFFFF);
        v83 = (v64[7] & 0x40) != 0 ? (_BYTE *)*((_QWORD *)v64 - 1) : &v64[-32 * (*((_DWORD *)v64 + 1) & 0x7FFFFFF)];
        v84 = *(unsigned int *)((char *)&v147 + v81);
        if ( *(_QWORD *)(v82 + v106) == *(_QWORD *)&v83[32 * v84] )
        {
          v87 = *(_QWORD *)(v82 + v105);
          v88 = *(_QWORD *)&v83[32 * ((_DWORD)v84 != 1)];
          v89 = *(_BYTE *)v88;
          if ( *(_BYTE *)v87 <= 0x1Cu )
          {
            v98 = 0;
            if ( v89 != 42 )
              goto LABEL_168;
          }
          else
          {
            if ( v89 <= 0x1Cu )
            {
              v90 = 0;
LABEL_185:
              if ( *(_BYTE *)v87 != 42 )
                goto LABEL_168;
              if ( v120 == 69 )
              {
                if ( !sub_B44900(v87) )
                  goto LABEL_191;
              }
              else if ( !sub_B448F0(v87) )
              {
                goto LABEL_191;
              }
              if ( (*(_BYTE *)(v87 + 7) & 0x40) != 0 )
                v91 = *(_QWORD *)(v87 - 8);
              else
                v91 = v87 - 32LL * (*(_DWORD *)(v87 + 4) & 0x7FFFFFF);
              if ( **(_BYTE **)(v91 + 32) == 17 && v88 == *(_QWORD *)v91 )
              {
                v102 = sub_2A8A780(v114, v111);
                if ( !(v103 + v102) )
                {
LABEL_218:
                  v127 = v109;
                  goto LABEL_168;
                }
              }
LABEL_191:
              if ( v90 && *(_BYTE *)v90 == 42 )
              {
                if ( sub_2A8A3A0(v87, v110) && sub_2A8A3A0(v90, v110) )
                {
                  if ( (*(_BYTE *)(v87 + 7) & 0x40) != 0 )
                    v92 = *(_QWORD **)(v87 - 8);
                  else
                    v92 = (_QWORD *)(v87 - 32LL * (*(_DWORD *)(v87 + 4) & 0x7FFFFFF));
                  v93 = v92[4];
                  if ( *(_BYTE *)v93 == 17 )
                  {
                    v94 = (*(_BYTE *)(v90 + 7) & 0x40) != 0
                        ? *(_QWORD *)(v90 - 8)
                        : v90 - 32LL * (*(_DWORD *)(v90 + 4) & 0x7FFFFFF);
                    if ( **(_BYTE **)(v94 + 32) == 17 )
                    {
                      sub_2A8A780(*(_QWORD *)(v93 + 24), *(_DWORD *)(v93 + 32));
                      sub_2A8A780(*(_QWORD *)(v95 + 24), *(_DWORD *)(v95 + 32));
                      if ( *v92 == *(_QWORD *)v94 )
                      {
                        v96 = sub_2A8A780(v114, v111);
                        v127 = v97 == v96;
                      }
                    }
                  }
                }
                else
                {
                  v127 = 0;
                }
              }
              goto LABEL_168;
            }
            if ( v89 != 42 )
            {
LABEL_184:
              v90 = v88;
              goto LABEL_185;
            }
            v98 = *(_QWORD *)(v82 + v105);
          }
          if ( v120 == 69 )
          {
            if ( sub_B44900(v88) )
              goto LABEL_209;
          }
          else if ( sub_B448F0(v88) )
          {
LABEL_209:
            if ( (*(_BYTE *)(v88 + 7) & 0x40) != 0 )
              v99 = *(_QWORD *)(v88 - 8);
            else
              v99 = v88 - 32LL * (*(_DWORD *)(v88 + 4) & 0x7FFFFFF);
            if ( **(_BYTE **)(v99 + 32) == 17 && v87 == *(_QWORD *)v99 )
            {
              v100 = sub_2A8A780(v114, v111);
              if ( v100 == v101 )
                goto LABEL_218;
            }
          }
          v87 = v98;
          if ( !v98 )
            goto LABEL_168;
          goto LABEL_184;
        }
      }
LABEL_168:
      v81 += 4;
      if ( v81 != 8 )
        continue;
      break;
    }
    v118 += 4;
    if ( v118 != 8 )
      continue;
    break;
  }
  v10 = v108;
  v8 = v104;
  v71 = sub_BCB060(*(_QWORD *)(v134 + 8));
  if ( v127 )
  {
LABEL_133:
    sub_9865C0((__int64)&v154, (__int64)&v143);
    sub_C47170((__int64)&v154, v107);
    v76 = v155;
    v149 = 1;
    LODWORD(v155) = 0;
    v148 = v76;
    v147 = v154;
    sub_969240((__int64 *)&v154);
    goto LABEL_134;
  }
LABEL_128:
  v113 = v71;
  sub_9878D0((__int64)&v154, v71);
  v72 = v8[3];
  v73 = v8[2];
  v74 = v8[6];
  v117 = v72;
  v126 = v73;
  v75 = (__int64)v64;
  if ( !sub_986F30((__int64)&v143, 0) )
    v75 = v134;
  sub_9AC1B0(v75, &v154, v74, 0, v126, a5, v117, 1);
  sub_C449B0((__int64)&v145, (const void **)&v154, v144);
  if ( v120 == 69 )
  {
    v77 = ~(1LL << ((unsigned __int8)v113 - 1));
    if ( v146 > 0x40 )
      *(_QWORD *)(v145 + 8LL * ((v113 - 1) >> 6)) &= v77;
    else
      v145 &= v77;
  }
  sub_9692E0((__int64)&v147, (__int64 *)&v143);
  v135 = sub_C49970((__int64)&v145, (unsigned __int64 *)&v147);
  sub_969240(&v147);
  if ( v135 >= 0 )
  {
    sub_969240(&v145);
    sub_969240((__int64 *)&v156);
    sub_969240((__int64 *)&v154);
    goto LABEL_133;
  }
  v149 = 0;
  sub_969240(&v145);
  sub_969240((__int64 *)&v156);
  sub_969240((__int64 *)&v154);
LABEL_134:
  sub_969240((__int64 *)&v143);
LABEL_135:
  sub_969240(&v152);
  sub_969240((__int64 *)&v150);
LABEL_83:
  if ( !v149 )
  {
LABEL_4:
    *(_BYTE *)(a1 + 16) = 0;
    goto LABEL_5;
  }
  sub_C44830((__int64)&v145, &v147, v140);
  sub_9865C0((__int64)&v141, (__int64)&v139);
  sub_C46B40((__int64)&v141, (__int64 *)&v137);
  v52 = v142;
  LODWORD(v142) = 0;
  v144 = v52;
  v143 = v141;
  sub_C45EE0((__int64)&v145, (__int64 *)&v143);
  v53 = v146;
  v146 = 0;
  v151 = v53;
  v150 = (char *)v145;
  sub_C44B10((__int64)&v154, &v150, v10);
  *(_DWORD *)(a1 + 8) = v155;
  v54 = v154;
  *(_BYTE *)(a1 + 16) = 1;
  *(_QWORD *)a1 = v54;
  sub_969240((__int64 *)&v150);
  sub_969240((__int64 *)&v143);
  sub_969240((__int64 *)&v141);
  sub_969240(&v145);
  if ( v149 )
  {
    v149 = 0;
    sub_969240(&v147);
  }
LABEL_5:
  if ( v140 > 0x40 && v139 )
    j_j___libc_free_0_0((unsigned __int64)v139);
  if ( v138 > 0x40 && v137 )
    j_j___libc_free_0_0((unsigned __int64)v137);
  return a1;
}
