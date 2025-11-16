// Function: sub_1145730
// Address: 0x1145730
//
__int64 __fastcall sub_1145730(__int64 a1, __int64 a2)
{
  __int64 v4; // r13
  int v5; // eax
  __int64 *v6; // rdx
  __int64 v7; // rsi
  __int64 v8; // rdi
  unsigned int v9; // r14d
  __int64 v10; // rdi
  __int16 v11; // ax
  __int64 *v12; // r8
  __int64 v13; // rsi
  __int64 v14; // rcx
  __int64 v15; // rdx
  _QWORD *v16; // r9
  __int64 v17; // rdi
  __int64 v18; // r15
  unsigned int v19; // eax
  unsigned int v20; // r15d
  unsigned int v21; // edx
  __int64 v22; // rax
  __int64 v24; // rdx
  char v25; // r15
  char v26; // r15
  char v27; // cl
  unsigned int v28; // r15d
  int v29; // eax
  unsigned __int16 v30; // ax
  __int64 v31; // rax
  _BYTE *v32; // rax
  unsigned int v33; // r15d
  int v34; // eax
  __int64 v35; // r12
  __int64 v36; // rax
  unsigned __int64 v39; // rax
  unsigned int v40; // eax
  unsigned __int64 v41; // rdx
  unsigned __int64 v42; // rdx
  unsigned __int64 v43; // rsi
  unsigned __int64 v44; // rax
  unsigned int v45; // eax
  unsigned __int64 v46; // rdx
  unsigned __int64 v47; // rdx
  unsigned __int64 v48; // rsi
  __int64 v49; // rax
  unsigned int v50; // eax
  char v51; // bl
  unsigned int v52; // eax
  __int64 v53; // rdi
  __int64 v54; // r12
  _QWORD *v55; // rax
  unsigned __int64 v56; // rax
  int v57; // edx
  _QWORD *v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rcx
  __int64 v61; // r8
  __int64 v62; // rdx
  unsigned int v63; // ebx
  int v64; // r14d
  int v65; // eax
  __int64 v66; // r12
  _QWORD *v67; // rax
  unsigned int v68; // eax
  unsigned __int64 v69; // rdx
  unsigned __int64 v70; // rdx
  unsigned int v71; // eax
  unsigned __int64 v72; // rdx
  unsigned __int64 v73; // rdx
  __int64 v76; // rsi
  __int64 v77; // rdx
  __int64 v78; // rdx
  __int64 v79; // rsi
  _QWORD *v80; // rax
  __int16 v81; // ax
  unsigned int v82; // eax
  char v83; // bl
  char *v84; // rbx
  __int64 v85; // rax
  unsigned int v86; // eax
  __int64 v87; // rdx
  __int64 v88; // rcx
  __int64 v89; // r8
  unsigned int v90; // edx
  __int64 *v91; // r8
  _QWORD *v92; // rax
  int v93; // eax
  __int64 v94; // rax
  _QWORD *v95; // rax
  unsigned int v96; // eax
  char v97; // bl
  unsigned int v98; // eax
  const void **v99; // rbx
  __int64 v100; // rax
  __int64 v101; // r12
  _QWORD *v102; // rax
  unsigned int v103; // eax
  int v104; // eax
  unsigned int v105; // eax
  char v106; // bl
  unsigned int v107; // eax
  __int16 v108; // bx
  __int64 v109; // r12
  unsigned int v110; // eax
  const void **v111; // rsi
  unsigned int v112; // ebx
  unsigned __int64 v113; // rax
  int v114; // eax
  unsigned int v115; // edx
  __int64 v116; // rax
  __int64 v117; // r12
  int v118; // eax
  __int64 *v119; // [rsp+8h] [rbp-218h]
  __int64 v120; // [rsp+18h] [rbp-208h]
  int v121; // [rsp+30h] [rbp-1F0h]
  int v122; // [rsp+30h] [rbp-1F0h]
  unsigned __int16 v123; // [rsp+38h] [rbp-1E8h]
  int v124; // [rsp+48h] [rbp-1D8h]
  __int16 v125; // [rsp+4Ch] [rbp-1D4h]
  int v126; // [rsp+4Ch] [rbp-1D4h]
  __int64 v127; // [rsp+50h] [rbp-1D0h]
  char *v128; // [rsp+58h] [rbp-1C8h]
  __int64 v129; // [rsp+70h] [rbp-1B0h]
  int v130; // [rsp+70h] [rbp-1B0h]
  unsigned int v131; // [rsp+70h] [rbp-1B0h]
  __int64 v132; // [rsp+78h] [rbp-1A8h]
  char *v133; // [rsp+80h] [rbp-1A0h] BYREF
  __int64 v134; // [rsp+88h] [rbp-198h] BYREF
  __int64 v135; // [rsp+90h] [rbp-190h] BYREF
  _DWORD v136[6]; // [rsp+98h] [rbp-188h] BYREF
  __int64 v137; // [rsp+B0h] [rbp-170h] BYREF
  unsigned int v138; // [rsp+B8h] [rbp-168h]
  __int64 v139; // [rsp+C0h] [rbp-160h] BYREF
  unsigned int v140; // [rsp+C8h] [rbp-158h]
  __int64 v141; // [rsp+D0h] [rbp-150h] BYREF
  unsigned int v142; // [rsp+D8h] [rbp-148h]
  __int64 v143; // [rsp+E0h] [rbp-140h] BYREF
  unsigned int v144; // [rsp+E8h] [rbp-138h]
  unsigned __int64 v145; // [rsp+F0h] [rbp-130h] BYREF
  int v146; // [rsp+F8h] [rbp-128h]
  __int64 v147[2]; // [rsp+100h] [rbp-120h] BYREF
  unsigned __int64 v148; // [rsp+110h] [rbp-110h] BYREF
  int v149; // [rsp+118h] [rbp-108h]
  const void **v150; // [rsp+120h] [rbp-100h] BYREF
  unsigned int v151; // [rsp+128h] [rbp-F8h]
  __int64 v152; // [rsp+130h] [rbp-F0h] BYREF
  unsigned int v153; // [rsp+138h] [rbp-E8h]
  _QWORD *v154; // [rsp+140h] [rbp-E0h] BYREF
  unsigned int v155; // [rsp+148h] [rbp-D8h]
  _QWORD *v156; // [rsp+150h] [rbp-D0h] BYREF
  unsigned int v157; // [rsp+158h] [rbp-C8h]
  _QWORD *v158; // [rsp+160h] [rbp-C0h] BYREF
  unsigned int v159; // [rsp+168h] [rbp-B8h]
  _QWORD *v160; // [rsp+170h] [rbp-B0h] BYREF
  unsigned int v161; // [rsp+178h] [rbp-A8h]
  __int64 v162; // [rsp+180h] [rbp-A0h] BYREF
  unsigned int v163; // [rsp+188h] [rbp-98h]
  __int64 v164; // [rsp+190h] [rbp-90h] BYREF
  int v165; // [rsp+198h] [rbp-88h]
  unsigned __int64 v166; // [rsp+1A0h] [rbp-80h] BYREF
  __int64 *v167; // [rsp+1A8h] [rbp-78h] BYREF
  __int64 v168; // [rsp+1B0h] [rbp-70h] BYREF
  __int64 v169; // [rsp+1B8h] [rbp-68h]
  __int64 v170; // [rsp+1C0h] [rbp-60h]
  __int64 v171; // [rsp+1C8h] [rbp-58h]
  __int64 v172; // [rsp+1D0h] [rbp-50h]
  __int64 v173; // [rsp+1D8h] [rbp-48h]
  __int16 v174; // [rsp+1E0h] [rbp-40h]

  v4 = *(_QWORD *)(*(_QWORD *)(a2 - 64) + 8LL);
  v128 = *(char **)(a2 - 64);
  v127 = *(_QWORD *)(a2 - 32);
  v125 = *(_WORD *)(a2 + 2);
  v5 = *(unsigned __int8 *)(v4 + 8);
  if ( (unsigned int)(v5 - 17) <= 1 )
  {
    v6 = *(__int64 **)(v4 + 16);
    v7 = *v6;
    if ( *(_BYTE *)(*v6 + 8) != 12 )
    {
      v8 = *(_QWORD *)(a1 + 88);
      if ( v5 == 18 )
        goto LABEL_4;
      goto LABEL_58;
    }
LABEL_20:
    v132 = 0;
    v9 = sub_BCB060(v4);
    if ( !v9 )
      return v132;
    goto LABEL_5;
  }
  if ( (_BYTE)v5 == 12 )
    goto LABEL_20;
  v8 = *(_QWORD *)(a1 + 88);
LABEL_58:
  v7 = v4;
  if ( v5 == 17 )
    v7 = **(_QWORD **)(v4 + 16);
LABEL_4:
  v132 = 0;
  v9 = sub_AE43A0(v8, v7);
  if ( !v9 )
    return v132;
LABEL_5:
  v155 = v9;
  if ( v9 > 0x40 )
  {
    sub_C43690((__int64)&v154, 0, 0);
    v157 = v9;
    sub_C43690((__int64)&v156, 0, 0);
    v159 = v9;
    sub_C43690((__int64)&v158, 0, 0);
    v161 = v9;
    sub_C43690((__int64)&v160, 0, 0);
  }
  else
  {
    v154 = 0;
    v157 = v9;
    v156 = 0;
    v159 = v9;
    v158 = 0;
    v161 = v9;
    v160 = 0;
  }
  v10 = *(_QWORD *)(a1 + 112);
  v11 = *(_WORD *)(a1 + 160);
  v171 = a2;
  v12 = *(__int64 **)(a1 + 104);
  v13 = *(_QWORD *)(a1 + 120);
  v172 = 0;
  v14 = *(_QWORD *)(a1 + 128);
  v15 = *(_QWORD *)(a1 + 152);
  v168 = v10;
  v16 = *(_QWORD **)(a1 + 96);
  v17 = *(_QWORD *)(a2 - 32);
  v167 = v12;
  v169 = v13;
  v166 = (unsigned __int64)v16;
  v18 = v17 + 24;
  v170 = v14;
  v173 = v15;
  v174 = v11;
  v132 = a2;
  if ( *(_BYTE *)v17 != 17 )
  {
    v24 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v17 + 8) + 8LL) - 17;
    if ( (unsigned int)v24 > 1 )
      goto LABEL_23;
    if ( *(_BYTE *)v17 > 0x15u )
      goto LABEL_23;
    v32 = sub_AD7630(v17, 0, v24);
    if ( !v32 )
      goto LABEL_23;
    v18 = (__int64)(v32 + 24);
    if ( *v32 != 17 )
      goto LABEL_23;
  }
  if ( sub_9893F0(*(_WORD *)(a2 + 2) & 0x3F, v18, &v152) )
  {
    v163 = v9;
    v22 = 1LL << ((unsigned __int8)v9 - 1);
    if ( v9 > 0x40 )
    {
      sub_C43690((__int64)&v162, 0, 0);
      v22 = 1LL << ((unsigned __int8)v9 - 1);
      if ( v163 > 0x40 )
      {
        *(_QWORD *)(v162 + 8LL * ((v9 - 1) >> 6)) |= 1LL << ((unsigned __int8)v9 - 1);
        goto LABEL_25;
      }
    }
    else
    {
      v162 = 0;
    }
    goto LABEL_19;
  }
  if ( (*(_WORD *)(a2 + 2) & 0x3F) == 0x22 )
  {
    if ( *(_DWORD *)(v18 + 8) > 0x40u )
    {
      v20 = sub_C445E0(v18);
    }
    else
    {
      v36 = *(_QWORD *)v18;
      v20 = 64;
      _RAX = ~v36;
      __asm { tzcnt   rdx, rax }
      if ( _RAX )
        v20 = _RDX;
    }
  }
  else
  {
    if ( (*(_WORD *)(a2 + 2) & 0x3F) != 0x24 )
    {
LABEL_23:
      v163 = v9;
      if ( v9 > 0x40 )
        sub_C43690((__int64)&v162, -1, 1);
      else
        v162 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v9);
      goto LABEL_25;
    }
    v19 = *(_DWORD *)(v18 + 8);
    if ( v19 <= 0x40 )
    {
      _RDX = *(_QWORD *)v18;
      v20 = 64;
      __asm { tzcnt   rcx, rdx }
      if ( _RDX )
        v20 = _RCX;
      if ( v19 <= v20 )
        v20 = v19;
    }
    else
    {
      v20 = sub_C44590(v18);
    }
  }
  v163 = v9;
  if ( v9 > 0x40 )
  {
    sub_C43690((__int64)&v162, 0, 0);
    v21 = v163;
  }
  else
  {
    v21 = v9;
    v162 = 0;
  }
  if ( v20 != v21 )
  {
    if ( v20 > 0x3F || v21 > 0x40 )
    {
      sub_C43C90(&v162, v20, v21);
      goto LABEL_25;
    }
    v22 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v20 + 64 - (unsigned __int8)v21) << v20;
LABEL_19:
    v162 |= v22;
  }
LABEL_25:
  v25 = sub_11AE940(a1, a2, 0, (unsigned int)&v162, (unsigned int)&v154, 0, (__int64)&v166);
  if ( v163 > 0x40 && v162 )
    j_j___libc_free_0_0(v162);
  if ( v25 )
    goto LABEL_44;
  v163 = v9;
  if ( v9 > 0x40 )
    sub_C43690((__int64)&v162, -1, 1);
  else
    v162 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v9);
  v26 = sub_11AE940(a1, a2, 1, (unsigned int)&v162, (unsigned int)&v158, 0, (__int64)&v166);
  if ( v163 > 0x40 && v162 )
    j_j___libc_free_0_0(v162);
  if ( v26 )
    goto LABEL_44;
  v27 = v125;
  v126 = v125 & 0x3F;
  v123 = v27 & 0x3F;
  if ( (unsigned __int8)*v128 > 0x15u )
  {
    v33 = v155;
    v122 = v155 > 0x40 ? sub_C44630((__int64)&v154) : sub_39FAC40(v154);
    v34 = v157 > 0x40 ? sub_C44630((__int64)&v156) : sub_39FAC40(v156);
    if ( v122 + v34 == v33 )
    {
      v35 = sub_AD6220(v4, (__int64)&v156);
      LOWORD(v170) = 257;
      v132 = (__int64)sub_BD2C40(72, unk_3F10FD0);
      if ( v132 )
        sub_1113300(v132, v126, v35, v127, (__int64)&v166);
      goto LABEL_44;
    }
  }
  if ( *(_BYTE *)v127 > 0x15u )
  {
    v28 = v159;
    v121 = v159 > 0x40 ? sub_C44630((__int64)&v158) : sub_39FAC40(v158);
    v29 = v161 > 0x40 ? sub_C44630((__int64)&v160) : sub_39FAC40(v160);
    if ( v121 + v29 == v28 )
    {
      v66 = sub_AD6220(v4, (__int64)&v160);
      LOWORD(v170) = 257;
      v67 = sub_BD2C40(72, unk_3F10FD0);
      v132 = (__int64)v67;
      if ( v67 )
        sub_1113300((__int64)v67, v126, (__int64)v128, v66, (__int64)&v166);
      goto LABEL_44;
    }
  }
  v30 = sub_B534E0((__int64)&v154, (__int64)&v158, v126);
  if ( HIBYTE(v30) )
  {
    v31 = sub_AD64A0(*(_QWORD *)(a2 + 8), v30);
    v132 = (__int64)sub_F162A0(a1, a2, v31);
    goto LABEL_44;
  }
  v138 = v9;
  if ( v9 > 0x40 )
  {
    sub_C43690((__int64)&v137, 0, 0);
    v140 = v9;
    sub_C43690((__int64)&v139, 0, 0);
    v142 = v9;
    sub_C43690((__int64)&v141, 0, 0);
    v144 = v9;
    sub_C43690((__int64)&v143, 0, 0);
  }
  else
  {
    v137 = 0;
    v140 = v9;
    v139 = 0;
    v142 = v9;
    v141 = 0;
    v144 = v9;
    v143 = 0;
  }
  if ( !sub_B532B0(*(_WORD *)(a2 + 2) & 0x3F) )
  {
    sub_9865C0((__int64)&v166, (__int64)&v156);
    sub_1110A30(&v137, (__int64 *)&v166);
    sub_969240((__int64 *)&v166);
    v68 = v155;
    LODWORD(v167) = v155;
    if ( v155 > 0x40 )
    {
      sub_C43780((__int64)&v166, (const void **)&v154);
      v68 = (unsigned int)v167;
      if ( (unsigned int)v167 > 0x40 )
      {
        sub_C43D10((__int64)&v166);
        v68 = (unsigned int)v167;
        v70 = v166;
        goto LABEL_165;
      }
      v69 = v166;
    }
    else
    {
      v69 = (unsigned __int64)v154;
    }
    v70 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v68) & ~v69;
    if ( !v68 )
      v70 = 0;
LABEL_165:
    v163 = v68;
    v162 = v70;
    sub_1110A30(&v139, &v162);
    sub_969240(&v162);
    sub_9865C0((__int64)&v166, (__int64)&v160);
    sub_1110A30(&v141, (__int64 *)&v166);
    sub_969240((__int64 *)&v166);
    v71 = v159;
    LODWORD(v167) = v159;
    if ( v159 > 0x40 )
    {
      sub_C43780((__int64)&v166, (const void **)&v158);
      v71 = (unsigned int)v167;
      if ( (unsigned int)v167 > 0x40 )
      {
        sub_C43D10((__int64)&v166);
        v71 = (unsigned int)v167;
        v73 = v166;
LABEL_169:
        v163 = v71;
        v162 = v73;
        goto LABEL_107;
      }
      v72 = v166;
    }
    else
    {
      v72 = (unsigned __int64)v158;
    }
    v73 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v71) & ~v72;
    if ( !v71 )
      v73 = 0;
    goto LABEL_169;
  }
  LODWORD(v167) = v157;
  if ( v157 > 0x40 )
    sub_C43780((__int64)&v166, (const void **)&v156);
  else
    v166 = (unsigned __int64)v156;
  v39 = (unsigned __int64)v154;
  if ( v155 > 0x40 )
    v39 = v154[(v155 - 1) >> 6];
  if ( (v39 & (1LL << ((unsigned __int8)v155 - 1))) == 0 )
  {
    v77 = 1LL << ((unsigned __int8)v167 - 1);
    if ( (unsigned int)v167 > 0x40 )
      *(_QWORD *)(v166 + 8LL * ((unsigned int)((_DWORD)v167 - 1) >> 6)) |= v77;
    else
      v166 |= v77;
  }
  sub_1110A30(&v137, (__int64 *)&v166);
  sub_969240((__int64 *)&v166);
  v40 = v155;
  LODWORD(v167) = v155;
  if ( v155 > 0x40 )
  {
    sub_C43780((__int64)&v166, (const void **)&v154);
    v40 = (unsigned int)v167;
    if ( (unsigned int)v167 > 0x40 )
    {
      sub_C43D10((__int64)&v166);
      v40 = (unsigned int)v167;
      v42 = v166;
      goto LABEL_92;
    }
    v41 = v166;
  }
  else
  {
    v41 = (unsigned __int64)v154;
  }
  v42 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v40) & ~v41;
  if ( !v40 )
    v42 = 0;
LABEL_92:
  v163 = v40;
  v162 = v42;
  v43 = (unsigned __int64)v156;
  if ( v157 > 0x40 )
    v43 = v156[(v157 - 1) >> 6];
  if ( (v43 & (1LL << ((unsigned __int8)v157 - 1))) == 0 )
  {
    v79 = ~(1LL << ((unsigned __int8)v40 - 1));
    if ( v40 > 0x40 )
      *(_QWORD *)(v42 + 8LL * ((v40 - 1) >> 6)) &= v79;
    else
      v162 = v42 & v79;
  }
  sub_1110A30(&v139, &v162);
  sub_969240(&v162);
  LODWORD(v167) = v161;
  if ( v161 > 0x40 )
    sub_C43780((__int64)&v166, (const void **)&v160);
  else
    v166 = (unsigned __int64)v160;
  v44 = (unsigned __int64)v158;
  if ( v159 > 0x40 )
    v44 = v158[(v159 - 1) >> 6];
  if ( (v44 & (1LL << ((unsigned __int8)v159 - 1))) == 0 )
  {
    v78 = 1LL << ((unsigned __int8)v167 - 1);
    if ( (unsigned int)v167 > 0x40 )
      *(_QWORD *)(v166 + 8LL * ((unsigned int)((_DWORD)v167 - 1) >> 6)) |= v78;
    else
      v166 |= v78;
  }
  sub_1110A30(&v141, (__int64 *)&v166);
  sub_969240((__int64 *)&v166);
  v45 = v159;
  LODWORD(v167) = v159;
  if ( v159 <= 0x40 )
  {
    v46 = (unsigned __int64)v158;
    goto LABEL_102;
  }
  sub_C43780((__int64)&v166, (const void **)&v158);
  v45 = (unsigned int)v167;
  if ( (unsigned int)v167 <= 0x40 )
  {
    v46 = v166;
LABEL_102:
    v47 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v45) & ~v46;
    if ( !v45 )
      v47 = 0;
    goto LABEL_104;
  }
  sub_C43D10((__int64)&v166);
  v45 = (unsigned int)v167;
  v47 = v166;
LABEL_104:
  v163 = v45;
  v162 = v47;
  v48 = (unsigned __int64)v160;
  if ( v161 > 0x40 )
    v48 = v160[(v161 - 1) >> 6];
  if ( (v48 & (1LL << ((unsigned __int8)v161 - 1))) == 0 )
  {
    v76 = ~(1LL << ((unsigned __int8)v45 - 1));
    if ( v45 > 0x40 )
      *(_QWORD *)(v47 + 8LL * ((v45 - 1) >> 6)) &= v76;
    else
      v162 = v47 & v76;
  }
LABEL_107:
  sub_1110A30(&v143, &v162);
  sub_969240(&v162);
  v49 = *(_QWORD *)(a2 + 16);
  if ( v49 )
  {
    if ( !*(_QWORD *)(v49 + 8) )
    {
      v56 = sub_99AEC0(*(_BYTE **)(v49 + 24), &v162, (__int64 *)&v166, 0, 0);
      *(_QWORD *)&v136[3] = v56;
      v136[5] = v57;
      if ( (unsigned int)(v56 - 7) > 1 && (_DWORD)v56 && (sub_1111940(v128) || sub_1111940((char *)v127)) )
        goto LABEL_144;
    }
  }
  if ( v123 == 38 )
  {
    if ( sub_AAD8B0((__int64)&v143, &v137) )
      goto LABEL_196;
    LOBYTE(v167) = 0;
    v166 = (unsigned __int64)&v150;
    if ( !(unsigned __int8)sub_991580((__int64)&v166, v127) )
      goto LABEL_114;
    sub_9865C0((__int64)&v162, (__int64)&v139);
    sub_C46F20((__int64)&v162, 1u);
    v82 = v163;
    v163 = 0;
    LODWORD(v167) = v82;
    v166 = v162;
    v83 = sub_AAD8B0((__int64)v150, &v166);
    sub_969240((__int64 *)&v166);
    sub_969240(&v162);
    if ( !v83 )
      goto LABEL_114;
    goto LABEL_202;
  }
  if ( v123 > 0x26u )
  {
    if ( v123 != 40 )
      goto LABEL_144;
    if ( sub_AAD8B0((__int64)&v141, &v139) )
      goto LABEL_196;
    LOBYTE(v167) = 0;
    v166 = (unsigned __int64)&v150;
    if ( !(unsigned __int8)sub_991580((__int64)&v166, v127) )
      goto LABEL_114;
    sub_9865C0((__int64)&v162, (__int64)&v137);
    sub_C46A40((__int64)&v162, 1);
    v50 = v163;
    v163 = 0;
    LODWORD(v167) = v50;
    v166 = v162;
    v51 = sub_AAD8B0((__int64)v150, &v166);
    sub_969240((__int64 *)&v166);
    sub_969240(&v162);
    if ( !v51 )
      goto LABEL_114;
    sub_9865C0((__int64)&v152, (__int64)v150);
    sub_C46F20((__int64)&v152, 1u);
LABEL_135:
    v52 = v153;
    v153 = 0;
    v163 = v52;
    v162 = v152;
    v53 = *(_QWORD *)(v127 + 8);
    goto LABEL_136;
  }
  if ( v123 == 34 )
  {
    if ( sub_AAD8B0((__int64)&v143, &v137) )
    {
      LOWORD(v170) = 257;
      v132 = (__int64)sub_BD2C40(72, unk_3F10FD0);
      if ( v132 )
        sub_1113300(v132, 33, (__int64)v128, v127, (__int64)&v166);
      goto LABEL_120;
    }
    LOBYTE(v167) = 0;
    v166 = (unsigned __int64)&v150;
    if ( !(unsigned __int8)sub_991580((__int64)&v166, v127) )
      goto LABEL_114;
    sub_9865C0((__int64)&v162, (__int64)&v139);
    sub_C46F20((__int64)&v162, 1u);
    v96 = v163;
    v163 = 0;
    LODWORD(v167) = v96;
    v166 = v162;
    v97 = sub_AAD8B0((__int64)v150, &v166);
    sub_969240((__int64 *)&v166);
    sub_969240(&v162);
    if ( !v97 )
    {
      v98 = sub_10E00E0((__int64)&v154);
      v99 = v150;
      if ( *((_DWORD *)v99 + 2) - (unsigned int)sub_9871A0((__int64)v150) <= v98 )
      {
        v100 = sub_AD6530(*(_QWORD *)(v127 + 8), (__int64)&v166);
        LOWORD(v170) = 257;
        v101 = v100;
        v102 = sub_BD2C40(72, unk_3F10FD0);
        v132 = (__int64)v102;
        if ( v102 )
          sub_1113300((__int64)v102, 33, (__int64)v128, v101, (__int64)&v166);
        goto LABEL_120;
      }
      goto LABEL_114;
    }
LABEL_202:
    sub_9865C0((__int64)&v152, (__int64)v150);
    sub_C46A40((__int64)&v152, 1);
    goto LABEL_135;
  }
  if ( v123 == 36 )
  {
    if ( !sub_AAD8B0((__int64)&v141, &v139) )
    {
      LOBYTE(v167) = 0;
      v166 = (unsigned __int64)&v150;
      if ( !(unsigned __int8)sub_991580((__int64)&v166, v127) )
      {
LABEL_114:
        if ( (sub_B532B0(*(_WORD *)(a2 + 2) & 0x3F)
           || sub_B532A0(*(_WORD *)(a2 + 2) & 0x3F) && (*(_BYTE *)(a2 + 1) & 2) == 0)
          && (sub_986C60((__int64 *)&v154, v155 - 1) && sub_986C60((__int64 *)&v158, v159 - 1)
           || sub_986C60((__int64 *)&v156, v157 - 1) && sub_986C60((__int64 *)&v160, v161 - 1)) )
        {
          v81 = sub_B52EF0(*(_WORD *)(a2 + 2) & 0x3F);
          *(_BYTE *)(a2 + 1) |= 2u;
          *(_WORD *)(a2 + 2) = v81 | *(_WORD *)(a2 + 2) & 0xFFC0;
        }
        else
        {
          v132 = 0;
        }
        goto LABEL_120;
      }
      sub_9865C0((__int64)&v162, (__int64)&v137);
      sub_C46A40((__int64)&v162, 1);
      v105 = v163;
      v163 = 0;
      LODWORD(v167) = v105;
      v166 = v162;
      v106 = sub_AAD8B0((__int64)v150, &v166);
      sub_969240((__int64 *)&v166);
      sub_969240(&v162);
      if ( !v106 )
      {
        v110 = sub_10E00E0((__int64)&v154);
        v111 = v150;
        v112 = v110;
        LODWORD(v167) = *((_DWORD *)v150 + 2);
        if ( (unsigned int)v167 > 0x40 )
          sub_C43780((__int64)&v166, v150);
        else
          v166 = (unsigned __int64)*v150;
        sub_C46E90((__int64)&v166);
        if ( (unsigned int)v167 > 0x40 )
        {
          v130 = (int)v167;
          v118 = sub_C444A0((__int64)&v166);
          v115 = v130 - v118;
          if ( v166 )
          {
            v131 = v130 - v118;
            j_j___libc_free_0_0(v166);
            v115 = v131;
          }
        }
        else
        {
          _BitScanReverse64(&v113, v166);
          v114 = v113 ^ 0x3F;
          if ( !v166 )
            v114 = 64;
          v115 = 64 - v114;
        }
        if ( v115 <= v112 )
        {
          v116 = sub_AD6530(*(_QWORD *)(v127 + 8), (__int64)v111);
          LOWORD(v170) = 257;
          v117 = v116;
          v132 = (__int64)sub_BD2C40(72, unk_3F10FD0);
          if ( v132 )
            sub_1113300(v132, 32, (__int64)v128, v117, (__int64)&v166);
          goto LABEL_120;
        }
        goto LABEL_114;
      }
      sub_9865C0((__int64)&v152, (__int64)v150);
      sub_C46F20((__int64)&v152, 1u);
      v107 = v153;
      v153 = 0;
      v163 = v107;
      v162 = v152;
      v53 = *(_QWORD *)(v127 + 8);
LABEL_136:
      v54 = sub_AD8D80(v53, (__int64)&v162);
      LOWORD(v170) = 257;
      v55 = sub_BD2C40(72, unk_3F10FD0);
      v132 = (__int64)v55;
      if ( v55 )
        sub_1113300((__int64)v55, 32, (__int64)v128, v54, (__int64)&v166);
      sub_969240(&v162);
      sub_969240(&v152);
      goto LABEL_120;
    }
LABEL_196:
    LOWORD(v170) = 257;
    v80 = sub_BD2C40(72, unk_3F10FD0);
    v132 = (__int64)v80;
    if ( v80 )
      sub_1113300((__int64)v80, 33, (__int64)v128, v127, (__int64)&v166);
    goto LABEL_120;
  }
LABEL_144:
  switch ( v123 )
  {
    case ' ':
    case '!':
      sub_9865C0((__int64)&v166, (__int64)&v154);
      sub_987160((__int64)&v166, (__int64)&v154, v59, v60, v61);
      v146 = (int)v167;
      v145 = v166;
      if ( !sub_986760((__int64)&v158) )
        goto LABEL_151;
      LOBYTE(v168) = 0;
      v166 = (unsigned __int64)&v133;
      v167 = &v134;
      v133 = 0;
      if ( *v128 == 57
        && *((_QWORD *)v128 - 8)
        && (v133 = (char *)*((_QWORD *)v128 - 8), (unsigned __int8)sub_991580((__int64)&v167, *((_QWORD *)v128 - 4)))
        && sub_AAD8B0(v134, &v145) )
      {
        v84 = v133;
      }
      else
      {
        v133 = v128;
        v84 = v128;
      }
      v166 = (unsigned __int64)v136;
      v167 = &v135;
      if ( *v84 != 54 )
        goto LABEL_151;
      if ( !(unsigned __int8)sub_1007280((_QWORD **)&v166, *((_QWORD *)v84 - 8), v62) )
        goto LABEL_151;
      v85 = *((_QWORD *)v84 - 4);
      if ( !v85 )
        goto LABEL_151;
      *v167 = v85;
      v120 = *(_QWORD *)(v135 + 8);
      v124 = sub_D949C0(*(__int64 *)v136);
      sub_9865C0((__int64)v147, (__int64)&v145);
      v119 = *(__int64 **)v136;
      sub_9865C0((__int64)&v150, *(__int64 *)v136);
      sub_C46F20((__int64)&v150, 1u);
      v86 = v151;
      v151 = 0;
      v153 = v86;
      v152 = (__int64)v150;
      sub_987160((__int64)&v152, 1, v87, v88, v89);
      v90 = v153;
      v153 = 0;
      v91 = v119;
      v163 = v90;
      v162 = v152;
      if ( v90 > 0x40 )
      {
        sub_C43B90(&v162, v147);
        v90 = v163;
        v92 = (_QWORD *)v162;
        v91 = v119;
      }
      else
      {
        v92 = (_QWORD *)(v147[0] & v152);
        v162 = v147[0] & v152;
      }
      LODWORD(v167) = v90;
      v166 = (unsigned __int64)v92;
      v163 = 0;
      sub_C45EE0((__int64)&v166, v91);
      v149 = (int)v167;
      v148 = v166;
      sub_969240(&v162);
      sub_969240(&v152);
      sub_969240((__int64 *)&v150);
      if ( !sub_986BA0((__int64)&v148) )
      {
        sub_969240((__int64 *)&v148);
        sub_969240(v147);
LABEL_151:
        v63 = v159;
        if ( v159 > 0x40 )
          v64 = sub_C44630((__int64)&v158);
        else
          v64 = sub_39FAC40(v158);
        if ( v161 > 0x40 )
          v65 = sub_C44630((__int64)&v160);
        else
          v65 = sub_39FAC40(v160);
        if ( v64 + v65 == v63 && sub_986BA0((__int64)&v160) )
        {
          sub_9865C0((__int64)&v162, (__int64)&v154);
          sub_9865C0((__int64)&v164, (__int64)&v156);
          sub_C7BCF0((__int64)&v162, (__int64 *)&v158);
          v103 = v163;
          v163 = 0;
          LODWORD(v167) = v103;
          v166 = v162;
          v104 = v165;
          v165 = 0;
          LODWORD(v169) = v104;
          v168 = v164;
          if ( sub_AAD8B0((__int64)&v166, &v154) && sub_AAD8B0((__int64)&v168, &v156) )
          {
            sub_969240(&v168);
            sub_969240((__int64 *)&v166);
            sub_969240(&v164);
            sub_969240(&v162);
            v108 = sub_B52870(v126);
            v109 = sub_AD6530(*(_QWORD *)(v127 + 8), (__int64)&v156);
            LOWORD(v170) = 257;
            v132 = (__int64)sub_BD2C40(72, unk_3F10FD0);
            if ( v132 )
              sub_1113300(v132, v108, (__int64)v128, v109, (__int64)&v166);
            goto LABEL_221;
          }
          sub_969240(&v168);
          sub_969240((__int64 *)&v166);
          sub_969240(&v164);
          sub_969240(&v162);
        }
        sub_969240((__int64 *)&v145);
        goto LABEL_114;
      }
      v93 = sub_D949C0((__int64)&v148);
      v94 = sub_AD64C0(v120, (unsigned int)(v93 - v124), 0);
      LOWORD(v170) = 257;
      v129 = v94;
      v95 = sub_BD2C40(72, unk_3F10FD0);
      v132 = (__int64)v95;
      if ( v95 )
        sub_1113300((__int64)v95, (v123 != 32) + 35, v135, v129, (__int64)&v166);
      sub_969240((__int64 *)&v148);
      sub_969240(v147);
LABEL_221:
      sub_969240((__int64 *)&v145);
LABEL_120:
      sub_969240(&v143);
      sub_969240(&v141);
      sub_969240(&v139);
      sub_969240(&v137);
LABEL_44:
      if ( v161 > 0x40 && v160 )
        j_j___libc_free_0_0(v160);
      if ( v159 > 0x40 && v158 )
        j_j___libc_free_0_0(v158);
      if ( v157 > 0x40 && v156 )
        j_j___libc_free_0_0(v156);
      if ( v155 > 0x40 && v154 )
        j_j___libc_free_0_0(v154);
      return v132;
    case '#':
    case '\'':
      if ( !sub_AAD8B0((__int64)&v141, &v139) )
        goto LABEL_114;
      goto LABEL_146;
    case '%':
    case ')':
      if ( !sub_AAD8B0((__int64)&v143, &v137) )
        goto LABEL_114;
LABEL_146:
      LOWORD(v170) = 257;
      v58 = sub_BD2C40(72, unk_3F10FD0);
      v132 = (__int64)v58;
      if ( v58 )
        sub_1113300((__int64)v58, 32, (__int64)v128, v127, (__int64)&v166);
      goto LABEL_120;
    default:
      goto LABEL_114;
  }
}
