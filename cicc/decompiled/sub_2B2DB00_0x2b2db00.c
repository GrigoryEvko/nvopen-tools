// Function: sub_2B2DB00
// Address: 0x2b2db00
//
__int64 __fastcall sub_2B2DB00(__int64 a1, char a2, __int64 a3, __int64 a4)
{
  int v4; // eax
  unsigned int v7; // esi
  unsigned int v8; // r15d
  unsigned __int64 v9; // r13
  __int64 ***v10; // rbx
  char v11; // al
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // r13
  __int64 ***v15; // r13
  __int64 **v16; // rax
  __int64 **v17; // rax
  __int64 **v18; // rax
  __int64 **v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  unsigned int v22; // edx
  _QWORD *v23; // r8
  __int64 v24; // r9
  _QWORD *v25; // r14
  _QWORD *v26; // r13
  unsigned __int64 v27; // rsi
  _QWORD *v28; // rax
  _QWORD *v29; // rdi
  __int64 v30; // rcx
  __int64 v31; // rdx
  __int64 v32; // rax
  _QWORD *v33; // rdi
  __int64 v34; // rcx
  __int64 v35; // rdx
  _QWORD *v36; // r14
  _QWORD *v37; // r13
  unsigned __int64 v38; // rsi
  _QWORD *v39; // rax
  _QWORD *v40; // rdi
  __int64 v41; // rcx
  __int64 v42; // rdx
  __int64 v43; // rax
  _QWORD *v44; // r8
  __int64 ***v45; // r14
  unsigned __int64 v46; // rax
  __int64 ***v47; // r13
  __int64 **v48; // rdi
  __int64 *v49; // r9
  __int64 v50; // r10
  __int64 v51; // r10
  __int64 ***v52; // r10
  __int64 **v53; // rax
  __int64 **v54; // rax
  __int64 **v55; // rax
  __int64 v56; // rax
  __int64 v58; // r13
  __int64 **v59; // rax
  __int64 *v60; // r13
  __int64 v61; // rax
  __int64 v62; // rsi
  __int64 ***v63; // r9
  __int64 v64; // r10
  __int64 v65; // rax
  __int64 v66; // rsi
  __int64 ***v67; // rsi
  __int64 **v68; // rdi
  __int64 *v69; // r11
  __int64 *v70; // rax
  __int64 v71; // r8
  __int64 v72; // rdx
  bool v73; // cl
  __int64 **v74; // rdi
  __int64 ***v75; // r13
  __int64 *v76; // r11
  __int64 *v77; // rax
  __int64 v78; // r8
  __int64 v79; // rdx
  bool v80; // cl
  __int64 **v81; // rdi
  __int64 *v82; // r11
  __int64 **v83; // rdi
  __int64 *v84; // r11
  __int64 v85; // rcx
  __int64 *v86; // rdi
  __int64 *v87; // rax
  __int64 v88; // r8
  __int64 v89; // rdx
  bool v90; // cl
  __int64 *v91; // rax
  __int64 v92; // r8
  __int64 v93; // rdx
  bool v94; // cl
  __int64 v95; // rdx
  __int64 v96; // rcx
  unsigned int v97; // edx
  _DWORD *v98; // rax
  unsigned int v99; // esi
  int v100; // ebx
  __int64 *v101; // r13
  _DWORD *v102; // rax
  int v103; // esi
  __int64 v104; // rax
  int v105; // edx
  __int64 v106; // rcx
  __int64 *v107; // rax
  __int64 v108; // r8
  __int64 **v109; // rsi
  __int64 *v110; // r15
  __int64 *v111; // r8
  __int64 v112; // r10
  __int64 *v113; // rax
  __int64 v114; // rdi
  __int64 **v115; // rsi
  __int64 *v116; // r8
  __int64 *v117; // rax
  __int64 **v118; // rsi
  __int64 *v119; // r8
  __int64 *v120; // rax
  signed __int64 v121; // rax
  __int64 *v122; // rax
  __int64 v123; // rdi
  __int64 *v124; // rax
  __int64 v125; // rdi
  __int64 **v126; // rax
  __int64 **v127; // rax
  __int64 *v128; // rdx
  __int64 *v129; // rcx
  char v130; // dl
  __int64 *v131; // r15
  __int64 *v132; // rax
  __int64 **v133; // rax
  bool v134; // al
  __int64 *v135; // r13
  __int64 *v136; // r13
  _BYTE **v137; // rax
  __int64 v138; // r8
  __int64 v139; // rdx
  bool v140; // si
  __int64 ***v141; // [rsp+0h] [rbp-60h]
  __int64 *v142; // [rsp+8h] [rbp-58h]
  __int64 *v143; // [rsp+8h] [rbp-58h]
  __int64 *v144; // [rsp+8h] [rbp-58h]
  __int64 *v145; // [rsp+8h] [rbp-58h]
  __int64 v146; // [rsp+8h] [rbp-58h]
  __int64 v147; // [rsp+8h] [rbp-58h]
  __int64 v148; // [rsp+8h] [rbp-58h]
  char v149; // [rsp+1Fh] [rbp-41h] BYREF
  _QWORD *v150; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v151; // [rsp+28h] [rbp-38h]

  v4 = *(_DWORD *)(a1 + 8);
  if ( !v4 )
    return 1;
  if ( v4 == 2 )
  {
    a3 = ****(_QWORD ****)a1;
    if ( *(_BYTE *)a3 == 91 )
    {
      v21 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
      if ( *(_DWORD *)(v21 + 104) == 3 )
      {
        v22 = *(_DWORD *)(v21 + 120);
        if ( !v22 )
          v22 = *(_DWORD *)(v21 + 8);
        if ( v22 <= 2
          || !sub_2B08550(*(unsigned __int8 ***)v21, *(unsigned int *)(v21 + 8))
          && !sub_2B0D880(v23, v24, (unsigned __int8 (__fastcall *)(_QWORD))sub_2B0D8B0) )
        {
          return 1;
        }
      }
    }
  }
  if ( a2 )
  {
LABEL_4:
    v7 = *(_DWORD *)(a1 + 8);
    goto LABEL_5;
  }
  v25 = sub_C52410();
  v26 = v25 + 1;
  v27 = sub_C959E0();
  v28 = (_QWORD *)v25[2];
  if ( v28 )
  {
    v29 = v25 + 1;
    do
    {
      while ( 1 )
      {
        v30 = v28[2];
        v31 = v28[3];
        if ( v27 <= v28[4] )
          break;
        v28 = (_QWORD *)v28[3];
        if ( !v31 )
          goto LABEL_32;
      }
      v29 = v28;
      v28 = (_QWORD *)v28[2];
    }
    while ( v30 );
LABEL_32:
    if ( v26 != v29 && v27 >= v29[4] )
      v26 = v29;
  }
  if ( v26 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v32 = v26[7];
    if ( v32 )
    {
      v33 = v26 + 6;
      do
      {
        while ( 1 )
        {
          v34 = *(_QWORD *)(v32 + 16);
          v35 = *(_QWORD *)(v32 + 24);
          if ( *(_DWORD *)(v32 + 32) >= dword_50103A8 )
            break;
          v32 = *(_QWORD *)(v32 + 24);
          if ( !v35 )
            goto LABEL_41;
        }
        v33 = (_QWORD *)v32;
        v32 = *(_QWORD *)(v32 + 16);
      }
      while ( v34 );
LABEL_41:
      if ( v33 != v26 + 6 && dword_50103A8 >= *((_DWORD *)v33 + 8) && *((_DWORD *)v33 + 9) )
        goto LABEL_44;
    }
  }
  v61 = *(unsigned int *)(a1 + 8);
  if ( !(_DWORD)v61 )
    goto LABEL_44;
  v62 = 8 * v61;
  v63 = *(__int64 ****)a1;
  v64 = *(_QWORD *)a1 + 8 * v61;
  v65 = (8 * v61) >> 3;
  v66 = v62 >> 5;
  if ( !v66 )
    goto LABEL_135;
  v67 = &v63[4 * v66];
  do
  {
    v68 = *v63;
    v69 = (*v63)[52];
    if ( *((_DWORD *)*v63 + 26) != 3 )
      goto LABEL_101;
    if ( !v69 || !v68[53] || *(_BYTE *)v69 != 90 )
    {
      v70 = *v68;
      v71 = (__int64)&(*v68)[*((unsigned int *)v68 + 2)];
      if ( *v68 == (__int64 *)v71 )
        goto LABEL_113;
      v72 = 0;
      do
      {
        v73 = *(_BYTE *)*v70++ == 90;
        v72 += v73;
      }
      while ( (__int64 *)v71 != v70 );
      if ( v72 <= 4 )
        goto LABEL_113;
LABEL_101:
      if ( !v69 || !v68[53] )
        goto LABEL_104;
    }
    if ( *(_BYTE *)v69 != 84 )
      goto LABEL_104;
LABEL_113:
    v74 = v63[1];
    v75 = v63 + 1;
    v76 = v74[52];
    if ( *((_DWORD *)v74 + 26) != 3 )
      goto LABEL_114;
    if ( !v76 || !v74[53] || *(_BYTE *)v76 != 90 )
    {
      v77 = *v74;
      v78 = (__int64)&(*v74)[*((unsigned int *)v74 + 2)];
      if ( *v74 == (__int64 *)v78 )
        goto LABEL_125;
      v79 = 0;
      do
      {
        v80 = *(_BYTE *)*v77++ == 90;
        v79 += v80;
      }
      while ( (__int64 *)v78 != v77 );
      if ( v79 <= 4 )
        goto LABEL_125;
LABEL_114:
      if ( !v76 || !v74[53] )
      {
LABEL_117:
        v63 = v75;
        goto LABEL_104;
      }
    }
    if ( *(_BYTE *)v76 != 84 )
      goto LABEL_117;
LABEL_125:
    v81 = v63[2];
    v75 = v63 + 2;
    v82 = v81[52];
    if ( *((_DWORD *)v81 + 26) != 3 )
      goto LABEL_126;
    if ( !v82 || !v81[53] || *(_BYTE *)v82 != 90 )
    {
      v87 = *v81;
      v88 = (__int64)&(*v81)[*((unsigned int *)v81 + 2)];
      if ( *v81 == (__int64 *)v88 )
        goto LABEL_129;
      v89 = 0;
      do
      {
        v90 = *(_BYTE *)*v87++ == 90;
        v89 += v90;
      }
      while ( (__int64 *)v88 != v87 );
      if ( v89 <= 4 )
        goto LABEL_129;
LABEL_126:
      if ( !v82 || !v81[53] )
        goto LABEL_117;
    }
    if ( *(_BYTE *)v82 != 84 )
      goto LABEL_117;
LABEL_129:
    v83 = v63[3];
    v75 = v63 + 3;
    v84 = v83[52];
    if ( *((_DWORD *)v83 + 26) != 3 )
      goto LABEL_130;
    if ( !v84 || !v83[53] || *(_BYTE *)v84 != 90 )
    {
      v91 = *v83;
      v92 = (__int64)&(*v83)[*((unsigned int *)v83 + 2)];
      if ( *v83 == (__int64 *)v92 )
        goto LABEL_133;
      v93 = 0;
      do
      {
        v94 = *(_BYTE *)*v91++ == 90;
        v93 += v94;
      }
      while ( (__int64 *)v92 != v91 );
      if ( v93 <= 4 )
        goto LABEL_133;
LABEL_130:
      if ( !v84 || !v83[53] )
        goto LABEL_117;
    }
    if ( *(_BYTE *)v84 != 84 )
      goto LABEL_117;
LABEL_133:
    v63 += 4;
  }
  while ( v67 != v63 );
  v65 = (v64 - (__int64)v63) >> 3;
LABEL_135:
  if ( v65 == 2 )
  {
LABEL_258:
    if ( !sub_2B0E4E0((__int64 *)v63) )
      goto LABEL_104;
    ++v63;
LABEL_138:
    v85 = (__int64)*v63;
    v86 = (*v63)[52];
    if ( *((_DWORD *)*v63 + 26) == 3 )
    {
      if ( v86 && *(_QWORD *)(v85 + 424) && *(_BYTE *)v86 == 90 )
      {
LABEL_141:
        if ( *(_BYTE *)v86 == 84 )
          return 1;
        goto LABEL_104;
      }
      v137 = *(_BYTE ***)v85;
      v138 = *(_QWORD *)v85 + 8LL * *(unsigned int *)(v85 + 8);
      if ( *(_QWORD *)v85 == v138 )
        return 1;
      v139 = 0;
      do
      {
        v140 = **v137++ == 90;
        v139 += v140;
      }
      while ( (_BYTE **)v138 != v137 );
      if ( v139 <= 4 )
        return 1;
    }
    if ( !v86 || !*(_QWORD *)(v85 + 424) )
      goto LABEL_104;
    goto LABEL_141;
  }
  if ( v65 != 3 )
  {
    if ( v65 == 1 )
      goto LABEL_138;
    return 1;
  }
  if ( sub_2B0E4E0((__int64 *)v63) )
  {
    ++v63;
    goto LABEL_258;
  }
LABEL_104:
  if ( (__int64 ***)v64 == v63 )
    return 1;
LABEL_44:
  v36 = sub_C52410();
  v37 = v36 + 1;
  v38 = sub_C959E0();
  v39 = (_QWORD *)v36[2];
  if ( v39 )
  {
    v40 = v36 + 1;
    do
    {
      while ( 1 )
      {
        v41 = v39[2];
        v42 = v39[3];
        if ( v38 <= v39[4] )
          break;
        v39 = (_QWORD *)v39[3];
        if ( !v42 )
          goto LABEL_49;
      }
      v40 = v39;
      v39 = (_QWORD *)v39[2];
    }
    while ( v41 );
LABEL_49:
    if ( v40 != v37 && v38 >= v40[4] )
      v37 = v40;
  }
  if ( v37 == (_QWORD *)((char *)sub_C52410() + 8) )
    goto LABEL_4;
  v43 = v37[7];
  v7 = *(_DWORD *)(a1 + 8);
  if ( !v43 )
    goto LABEL_5;
  v44 = v37 + 6;
  do
  {
    while ( 1 )
    {
      a4 = *(_QWORD *)(v43 + 16);
      a3 = *(_QWORD *)(v43 + 24);
      if ( *(_DWORD *)(v43 + 32) >= dword_50103A8 )
        break;
      v43 = *(_QWORD *)(v43 + 24);
      if ( !a3 )
        goto LABEL_58;
    }
    v44 = (_QWORD *)v43;
    v43 = *(_QWORD *)(v43 + 16);
  }
  while ( a4 );
LABEL_58:
  if ( v44 == v37 + 6 )
    goto LABEL_5;
  if ( dword_50103A8 < *((_DWORD *)v44 + 8) )
    goto LABEL_5;
  a3 = *((unsigned int *)v44 + 9);
  if ( !(_DWORD)a3 || v7 > 4 )
    goto LABEL_5;
  v45 = *(__int64 ****)a1;
  v150 = (_QWORD *)a1;
  v46 = v7;
  v47 = &v45[v46];
  if ( !((v46 * 8) >> 5) )
  {
    v110 = (__int64 *)v45;
LABEL_220:
    v121 = (char *)v47 - (char *)v110;
    if ( (char *)v47 - (char *)v110 != 16 )
    {
      if ( v121 != 24 )
      {
        if ( v121 != 8 )
        {
          v7 = *(_DWORD *)(a1 + 8);
          goto LABEL_198;
        }
        goto LABEL_266;
      }
      if ( !sub_2B232B0(&v150, v110, a3, a4) )
        goto LABEL_196;
      ++v110;
    }
    if ( !sub_2B232B0(&v150, v110, a3, a4) )
      goto LABEL_196;
    ++v110;
LABEL_266:
    v134 = sub_2B232B0(&v150, v110, a3, a4);
    v7 = *(_DWORD *)(a1 + 8);
    if ( !v134 )
      goto LABEL_197;
    goto LABEL_198;
  }
  v48 = *v45;
  v49 = (*v45)[52];
  if ( *((_DWORD *)*v45 + 26) != 3 )
    goto LABEL_64;
  if ( v49 && v48[53] && *(_BYTE *)v49 == 90 )
    goto LABEL_66;
  v107 = *v48;
  a3 = *((unsigned int *)v48 + 2);
  v108 = (__int64)&(*v48)[a3];
  if ( *v48 != (__int64 *)v108 )
  {
    a3 = 0;
    do
    {
      LOBYTE(a4) = *(_BYTE *)*v107++ == 90;
      a4 = (unsigned __int8)a4;
      a3 += (unsigned __int8)a4;
    }
    while ( (__int64 *)v108 != v107 );
    if ( a3 > 4 )
    {
LABEL_64:
      if ( !v49 || !v48[53] )
      {
LABEL_68:
        if ( v45 != v47 )
          goto LABEL_5;
        v50 = v7;
        a3 = (__int64)&v45[v50];
        v51 = (v50 * 8) >> 5;
        goto LABEL_70;
      }
LABEL_66:
      if ( *(_BYTE *)v49 == 91 )
        goto LABEL_191;
      if ( *(_BYTE *)v49 == 84 )
      {
        v131 = &(*v48)[*((unsigned int *)v48 + 2)];
        if ( v131 == sub_2B22F90(*v48, (__int64)v131, (_QWORD *)a1, a4) )
          goto LABEL_191;
        v7 = *(_DWORD *)(a1 + 8);
        v110 = (__int64 *)v45;
        goto LABEL_197;
      }
      goto LABEL_68;
    }
  }
LABEL_191:
  v109 = v45[1];
  v110 = (__int64 *)(v45 + 1);
  v111 = v109[52];
  if ( *((_DWORD *)v109 + 26) != 3 )
    goto LABEL_192;
  if ( v111 && v109[53] && *(_BYTE *)v111 == 90 )
  {
LABEL_194:
    if ( *(_BYTE *)v111 == 91 )
      goto LABEL_207;
    if ( *(_BYTE *)v111 == 84 )
    {
      v148 = (__int64)&(*v109)[*((unsigned int *)v109 + 2)];
      v132 = sub_2B22F90(*v109, v148, v150, v148);
      a4 = v148;
      if ( (__int64 *)v148 == v132 )
        goto LABEL_207;
    }
    goto LABEL_196;
  }
  v113 = *v109;
  a3 = *((unsigned int *)v109 + 2);
  v114 = (__int64)&(*v109)[a3];
  if ( *v109 != (__int64 *)v114 )
  {
    a3 = 0;
    do
    {
      LOBYTE(a4) = *(_BYTE *)*v113++ == 90;
      a4 = (unsigned __int8)a4;
      a3 += (unsigned __int8)a4;
    }
    while ( (__int64 *)v114 != v113 );
    if ( a3 > 4 )
    {
LABEL_192:
      if ( !v111 || !v109[53] )
        goto LABEL_196;
      goto LABEL_194;
    }
  }
LABEL_207:
  v115 = v45[2];
  v110 = (__int64 *)(v45 + 2);
  v116 = v115[52];
  if ( *((_DWORD *)v115 + 26) != 3 )
    goto LABEL_208;
  if ( v116 && v115[53] && *(_BYTE *)v116 == 90 )
  {
LABEL_210:
    if ( *(_BYTE *)v116 == 91 )
      goto LABEL_213;
    if ( *(_BYTE *)v116 == 84 )
    {
      v146 = (__int64)&(*v115)[*((unsigned int *)v115 + 2)];
      v117 = sub_2B22F90(*v115, v146, v150, v146);
      a4 = v146;
      if ( (__int64 *)v146 == v117 )
        goto LABEL_213;
    }
    goto LABEL_196;
  }
  v122 = *v115;
  a3 = *((unsigned int *)v115 + 2);
  v123 = (__int64)&(*v115)[a3];
  if ( *v115 != (__int64 *)v123 )
  {
    a3 = 0;
    do
    {
      LOBYTE(a4) = *(_BYTE *)*v122++ == 90;
      a4 = (unsigned __int8)a4;
      a3 += (unsigned __int8)a4;
    }
    while ( (__int64 *)v123 != v122 );
    if ( a3 > 4 )
    {
LABEL_208:
      if ( !v116 || !v115[53] )
        goto LABEL_196;
      goto LABEL_210;
    }
  }
LABEL_213:
  v118 = v45[3];
  v110 = (__int64 *)(v45 + 3);
  v119 = v118[52];
  if ( *((_DWORD *)v118 + 26) != 3 )
    goto LABEL_214;
  if ( v119 && v118[53] && *(_BYTE *)v119 == 90 )
    goto LABEL_293;
  v124 = *v118;
  a3 = *((unsigned int *)v118 + 2);
  v125 = (__int64)&(*v118)[a3];
  if ( *v118 == (__int64 *)v125 )
    goto LABEL_219;
  a3 = 0;
  do
  {
    LOBYTE(a4) = *(_BYTE *)*v124++ == 90;
    a4 = (unsigned __int8)a4;
    a3 += (unsigned __int8)a4;
  }
  while ( (__int64 *)v125 != v124 );
  if ( a3 <= 4 )
  {
LABEL_219:
    v110 = (__int64 *)(v45 + 4);
    goto LABEL_220;
  }
LABEL_214:
  if ( v119 && v118[53] )
  {
LABEL_293:
    if ( *(_BYTE *)v119 == 91 )
      goto LABEL_219;
    if ( *(_BYTE *)v119 == 84 )
    {
      v147 = (__int64)&(*v118)[*((unsigned int *)v118 + 2)];
      v120 = sub_2B22F90(*v118, v147, v150, v147);
      a4 = v147;
      if ( (__int64 *)v147 == v120 )
        goto LABEL_219;
    }
  }
LABEL_196:
  v7 = *(_DWORD *)(a1 + 8);
LABEL_197:
  if ( v47 != (__int64 ***)v110 )
    goto LABEL_5;
LABEL_198:
  v45 = *(__int64 ****)a1;
  v112 = 8LL * v7;
  a3 = *(_QWORD *)a1 + v112;
  v51 = v112 >> 5;
  if ( v51 )
  {
LABEL_70:
    v52 = &v45[4 * v51];
    while ( *((_DWORD *)*v45 + 26) || *(_BYTE *)(*v45)[52] != 84 )
    {
      v53 = v45[1];
      if ( !*((_DWORD *)v53 + 26) && *(_BYTE *)v53[52] == 84 )
      {
        ++v45;
        goto LABEL_85;
      }
      v54 = v45[2];
      if ( !*((_DWORD *)v54 + 26) && *(_BYTE *)v54[52] == 84 )
      {
        v45 += 2;
        goto LABEL_85;
      }
      v55 = v45[3];
      if ( !*((_DWORD *)v55 + 26) && *(_BYTE *)v55[52] == 84 )
      {
        v45 += 3;
        goto LABEL_85;
      }
      v45 += 4;
      if ( v52 == v45 )
        goto LABEL_80;
    }
    goto LABEL_85;
  }
LABEL_80:
  v56 = a3 - (_QWORD)v45;
  if ( a3 - (_QWORD)v45 != 16 )
  {
    if ( v56 != 24 )
    {
      if ( v56 == 8 )
      {
LABEL_83:
        if ( !*((_DWORD *)*v45 + 26) && *(_BYTE *)(*v45)[52] == 84 )
          goto LABEL_85;
      }
      goto LABEL_5;
    }
    if ( !*((_DWORD *)*v45 + 26) && *(_BYTE *)(*v45)[52] == 84 )
      goto LABEL_85;
    ++v45;
  }
  if ( *((_DWORD *)*v45 + 26) || *(_BYTE *)(*v45)[52] != 84 )
  {
    ++v45;
    goto LABEL_83;
  }
LABEL_85:
  if ( v45 != (__int64 ***)a3 )
    return 1;
LABEL_5:
  if ( (unsigned int)qword_500FB68 <= v7 )
    return 0;
  v8 = sub_2B2D940(a1, a2, a3, (_QWORD *)a4);
  if ( (_BYTE)v8 )
    return 0;
  v9 = *(unsigned int *)(a1 + 8);
  v10 = *(__int64 ****)a1;
  v11 = 1;
  if ( v9 > 1 )
    goto LABEL_8;
  if ( v9 != 1 )
  {
    v9 = 0;
    goto LABEL_91;
  }
  v127 = *v10;
  v128 = (*v10)[52];
  if ( !v128 || (v129 = v127[53], v129 != v128) || !v129 || (v130 = *(_BYTE *)v128, v130 == 84) || v130 == 63 )
  {
LABEL_91:
    v58 = v9;
    v149 = 0;
    v141 = &v10[v58];
    v13 = (v58 * 8) >> 3;
LABEL_92:
    if ( v13 != 3 )
    {
      if ( v13 != 1 )
        goto LABEL_17;
      goto LABEL_94;
    }
    v133 = *v10;
    if ( *((_DWORD *)*v10 + 26) == 3 )
    {
      v136 = &(*v133)[*((unsigned int *)v133 + 2)];
      if ( v136 == sub_2B1DD60(*v133, (__int64)v136, &v149) )
        goto LABEL_16;
    }
    ++v10;
LABEL_242:
    v126 = *v10;
    if ( *((_DWORD *)*v10 + 26) == 3 )
    {
      v135 = &(*v126)[*((unsigned int *)v126 + 2)];
      if ( v135 == sub_2B1DD60(*v126, (__int64)v135, &v149) )
        goto LABEL_16;
    }
    ++v10;
LABEL_94:
    v59 = *v10;
    if ( *((_DWORD *)*v10 + 26) == 3 )
    {
      v60 = &(*v59)[*((unsigned int *)v59 + 2)];
      if ( v60 == sub_2B1DD60(*v59, (__int64)v60, &v149) )
        goto LABEL_16;
    }
    goto LABEL_17;
  }
  v11 = sub_2B17600((_BYTE **)*v127, *((unsigned int *)v127 + 2));
  v10 = *(__int64 ****)a1;
  v9 = *(unsigned int *)(a1 + 8);
LABEL_8:
  v12 = 8 * v9;
  v149 = v11;
  v141 = &v10[(unsigned __int64)v12 / 8];
  v13 = v12 >> 3;
  v14 = v12 >> 5;
  if ( !v14 )
  {
LABEL_241:
    if ( v13 == 2 )
      goto LABEL_242;
    goto LABEL_92;
  }
  v15 = &v10[4 * v14];
  while ( 1 )
  {
    v19 = *v10;
    if ( *((_DWORD *)*v10 + 26) == 3 )
    {
      v142 = &(*v19)[*((unsigned int *)v19 + 2)];
      if ( v142 == sub_2B1DD60(*v19, (__int64)v142, &v149) )
        break;
    }
    v16 = v10[1];
    if ( *((_DWORD *)v16 + 26) == 3 )
    {
      v143 = &(*v16)[*((unsigned int *)v16 + 2)];
      if ( v143 == sub_2B1DD60(*v16, (__int64)v143, &v149) )
      {
        ++v10;
        break;
      }
    }
    v17 = v10[2];
    if ( *((_DWORD *)v17 + 26) == 3 )
    {
      v144 = &(*v17)[*((unsigned int *)v17 + 2)];
      if ( v144 == sub_2B1DD60(*v17, (__int64)v144, &v149) )
      {
        v10 += 2;
        break;
      }
    }
    v18 = v10[3];
    if ( *((_DWORD *)v18 + 26) == 3 )
    {
      v145 = &(*v18)[*((unsigned int *)v18 + 2)];
      if ( v145 == sub_2B1DD60(*v18, (__int64)v145, &v149) )
      {
        v10 += 3;
        break;
      }
    }
    v10 += 4;
    if ( v15 == v10 )
    {
      v13 = v141 - v10;
      goto LABEL_241;
    }
  }
LABEL_16:
  if ( v141 != v10 )
    return 0;
LABEL_17:
  v20 = *(_QWORD *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - 8);
  if ( *(_DWORD *)(v20 + 104) != 3 )
    return 1;
  v95 = *(_QWORD *)(v20 + 416);
  if ( !v95 )
    return 1;
  v96 = *(_QWORD *)(v20 + 424);
  if ( !v96 || v95 == v96 )
    return 1;
  v97 = *(_DWORD *)(v20 + 120);
  if ( !v97 )
    v97 = *(_DWORD *)(v20 + 8);
  if ( v97 <= 2 )
    return 1;
  if ( !(unsigned __int8)sub_2B17600(*(_BYTE ***)v20, *(unsigned int *)(v20 + 8)) )
    return 1;
  v98 = *(_DWORD **)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - 8);
  if ( (unsigned __int8)(*(_BYTE *)(*(_QWORD *)(**(_QWORD **)v98 + 8LL) + 8LL) - 17) <= 1u )
    return 1;
  v99 = v98[30];
  v100 = qword_5010428;
  v101 = *(__int64 **)(a1 + 3296);
  if ( !v99 )
    v99 = v98[2];
  sub_9691E0((__int64)&v150, v99, -1, 1u, 0);
  v102 = *(_DWORD **)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - 8);
  v103 = v102[30];
  if ( !v103 )
    v103 = v102[2];
  v104 = sub_2B08680(*(_QWORD *)(**(_QWORD **)v102 + 8LL), v103);
  v106 = sub_DFAAD0(v101, v104, (__int64)&v150, 1u, 0);
  if ( v105 )
    LOBYTE(v100) = v105 > 0;
  else
    LOBYTE(v100) = -v100 < v106;
  if ( (_BYTE)v100 )
  {
    if ( v151 <= 0x40 )
      return v8;
  }
  else if ( v151 <= 0x40 )
  {
    return 1;
  }
  if ( v150 )
    j_j___libc_free_0_0((unsigned __int64)v150);
  return v100 ^ 1u;
}
