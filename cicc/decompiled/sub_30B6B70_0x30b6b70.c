// Function: sub_30B6B70
// Address: 0x30b6b70
//
void __fastcall sub_30B6B70(__int64 *a1, __int64 a2, unsigned int *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  bool v7; // zf
  __int64 v8; // rdx
  __int64 v9; // r14
  _QWORD *v10; // r14
  __int64 *v11; // rax
  unsigned __int64 v12; // rdx
  int v13; // eax
  _BYTE *v14; // r8
  _BYTE *v15; // rcx
  __int64 v16; // rdi
  unsigned __int16 v17; // dx
  __int64 *v18; // r12
  __int64 v19; // rbx
  unsigned int v20; // eax
  _BYTE *v21; // rcx
  __int64 v22; // rdi
  unsigned __int16 v23; // dx
  __int64 *v24; // r15
  __int64 *v25; // rbx
  char v26; // r14
  __int64 v27; // rax
  unsigned __int64 v28; // rdx
  unsigned int *v29; // r13
  __int64 v30; // rdi
  unsigned int v31; // eax
  _BYTE *v32; // r8
  __int64 v33; // rcx
  __int64 v34; // rdi
  unsigned __int16 v35; // dx
  __int64 v36; // rax
  __int64 v37; // rdx
  unsigned __int64 v38; // rcx
  __int64 v39; // r8
  __int64 *v40; // r13
  __int64 *v41; // r15
  __int64 v42; // rbx
  __int64 *v43; // rax
  __int16 v44; // ax
  __int64 v45; // rax
  unsigned __int64 v46; // rdx
  __int64 *v47; // r14
  __int64 v48; // r14
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // r8
  __int64 *v54; // r14
  __int64 v55; // rbx
  __int64 *v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rax
  __int64 v59; // r15
  _QWORD *v60; // r12
  __int64 v61; // rax
  __int64 v62; // rax
  unsigned __int64 v63; // rdx
  __int64 v64; // r8
  unsigned int *v65; // r14
  __int64 v66; // rax
  signed __int64 v67; // r13
  __int64 v68; // r12
  __int64 *v69; // r11
  unsigned int v70; // r12d
  size_t v71; // rax
  __int64 *v72; // rdi
  __int64 *v73; // rdi
  signed __int64 v74; // r15
  __int64 v75; // rbx
  __int64 *v76; // r9
  unsigned int v77; // r15d
  size_t v78; // r10
  unsigned __int64 v79; // rdx
  __int64 v80; // rax
  __int64 v81; // rdx
  unsigned __int64 v82; // rcx
  __int64 v83; // r8
  __int64 v84; // r9
  __int64 *v85; // r12
  __int64 *v86; // r13
  __int64 v87; // rbx
  __int64 *v88; // rax
  __int64 *v89; // r14
  __int64 v90; // rax
  unsigned __int64 v91; // r15
  __int64 *v92; // r12
  char v93; // bl
  __int64 v94; // rax
  unsigned __int64 v95; // rdx
  __int64 v96; // rdi
  char v97; // r10
  __int64 v98; // rax
  unsigned __int64 v99; // rdx
  __int64 v100; // rbx
  __int64 *v101; // rax
  __int64 *v102; // r14
  __int64 v103; // rax
  __int64 v104; // rax
  unsigned __int64 v105; // rdx
  __int64 *v106; // rdi
  __int64 v107; // rbx
  __int64 v108; // r8
  __int64 v109; // r9
  __int64 *v110; // r14
  __int64 v111; // rax
  __int64 *v112; // rdi
  __int64 *src; // [rsp+0h] [rbp-340h]
  __int64 v114; // [rsp+10h] [rbp-330h]
  __int64 v116; // [rsp+38h] [rbp-308h]
  __int64 v117; // [rsp+40h] [rbp-300h]
  __int64 v118; // [rsp+40h] [rbp-300h]
  __int64 v119; // [rsp+48h] [rbp-2F8h]
  __int64 *v120; // [rsp+48h] [rbp-2F8h]
  __int64 *v121; // [rsp+48h] [rbp-2F8h]
  __int64 *v122; // [rsp+50h] [rbp-2F0h]
  __int64 *v124; // [rsp+68h] [rbp-2D8h]
  __int64 v125; // [rsp+68h] [rbp-2D8h]
  unsigned int *v126; // [rsp+78h] [rbp-2C8h] BYREF
  _QWORD v127[2]; // [rsp+80h] [rbp-2C0h] BYREF
  unsigned int *v128; // [rsp+90h] [rbp-2B0h] BYREF
  __int64 *v129; // [rsp+98h] [rbp-2A8h]
  __int64 *v130; // [rsp+A0h] [rbp-2A0h] BYREF
  __int64 v131; // [rsp+A8h] [rbp-298h]
  _BYTE v132[32]; // [rsp+B0h] [rbp-290h] BYREF
  unsigned int **v133; // [rsp+D0h] [rbp-270h] BYREF
  _BYTE *v134; // [rsp+D8h] [rbp-268h] BYREF
  __int64 v135; // [rsp+E0h] [rbp-260h]
  _BYTE v136[64]; // [rsp+E8h] [rbp-258h] BYREF
  __int64 v137; // [rsp+128h] [rbp-218h] BYREF
  __int64 *v138; // [rsp+130h] [rbp-210h]
  __int64 v139; // [rsp+138h] [rbp-208h]
  int v140; // [rsp+140h] [rbp-200h]
  char v141; // [rsp+144h] [rbp-1FCh]
  __int64 v142; // [rsp+148h] [rbp-1F8h] BYREF
  unsigned int **v143; // [rsp+190h] [rbp-1B0h] BYREF
  __int64 v144; // [rsp+198h] [rbp-1A8h] BYREF
  __int64 v145; // [rsp+1A0h] [rbp-1A0h] BYREF
  _QWORD v146[8]; // [rsp+1A8h] [rbp-198h] BYREF
  __int64 v147; // [rsp+1E8h] [rbp-158h] BYREF
  __int64 *v148; // [rsp+1F0h] [rbp-150h]
  __int64 v149; // [rsp+1F8h] [rbp-148h]
  int v150; // [rsp+200h] [rbp-140h]
  char v151; // [rsp+204h] [rbp-13Ch]
  __int64 v152; // [rsp+208h] [rbp-138h] BYREF
  __int64 *p_dest; // [rsp+250h] [rbp-F0h] BYREF
  __int64 v154; // [rsp+258h] [rbp-E8h] BYREF
  __int64 dest; // [rsp+260h] [rbp-E0h] BYREF
  _BYTE v156[64]; // [rsp+268h] [rbp-D8h] BYREF
  __int64 v157; // [rsp+2A8h] [rbp-98h] BYREF
  __int64 *v158; // [rsp+2B0h] [rbp-90h]
  __int64 v159; // [rsp+2B8h] [rbp-88h]
  int v160; // [rsp+2C0h] [rbp-80h]
  char v161; // [rsp+2C4h] [rbp-7Ch]
  __int64 v162; // [rsp+2C8h] [rbp-78h] BYREF

  v131 = 0x400000000LL;
  v133 = (unsigned int **)v127;
  v134 = v136;
  v135 = 0x800000000LL;
  v138 = &v142;
  v139 = 0x100000008LL;
  v6 = 0;
  v7 = *(_WORD *)(a2 + 24) == 8;
  v130 = (__int64 *)v132;
  v127[0] = a1;
  v127[1] = &v130;
  v140 = 0;
  v141 = 1;
  v142 = a2;
  v137 = 1;
  if ( v7 )
  {
    v8 = *(_QWORD *)(a2 + 40);
    v9 = *(_QWORD *)(a2 + 32);
    if ( v8 == 2 )
    {
      v10 = *(_QWORD **)(v9 + 8);
      v11 = (__int64 *)v132;
LABEL_4:
      *v11 = (__int64)v10;
      v6 = (unsigned int)v135;
      LODWORD(v131) = v131 + 1;
      v12 = (unsigned int)v135 + 1LL;
      if ( HIDWORD(v135) < v12 )
      {
        sub_C8D5F0((__int64)&v134, v136, v12, 8u, a5, a6);
        v6 = (unsigned int)v135;
      }
      goto LABEL_6;
    }
    v74 = 8 * v8 - 8;
    v143 = (unsigned int **)&v145;
    v75 = v74 >> 3;
    v125 = *(_QWORD *)(a2 + 48);
    v144 = 0x300000000LL;
    if ( (unsigned __int64)v74 > 0x18 )
    {
      sub_C8D5F0((__int64)&v143, &v145, (8 * v8 - 8) >> 3, 8u, a5, a6);
      v112 = (__int64 *)&v143[(unsigned int)v144];
    }
    else
    {
      v76 = &v145;
      if ( 8 * v8 == 8 )
        goto LABEL_122;
      v112 = &v145;
    }
    memcpy(v112, (const void *)(v9 + 8), v74);
    v76 = (__int64 *)v143;
    LODWORD(v74) = v144;
LABEL_122:
    v77 = v75 + v74;
    LODWORD(v144) = v77;
    p_dest = &dest;
    v78 = 8LL * v77;
    v154 = 0x400000000LL;
    if ( v77 > 4uLL )
    {
      v121 = v76;
      sub_C8D5F0((__int64)&p_dest, &dest, v77, 8u, a5, (__int64)v76);
      v78 = 8LL * v77;
      v76 = v121;
      v106 = &p_dest[(unsigned int)v154];
    }
    else
    {
      if ( !v78 )
      {
LABEL_124:
        LODWORD(v154) = v78 + v77;
        v10 = sub_DBFF60((__int64)a1, (unsigned int *)&p_dest, v125, 0);
        if ( p_dest != &dest )
          _libc_free((unsigned __int64)p_dest);
        if ( v143 != (unsigned int **)&v145 )
          _libc_free((unsigned __int64)v143);
        v79 = (unsigned int)v131 + 1LL;
        if ( v79 > HIDWORD(v131) )
          sub_C8D5F0((__int64)&v130, v132, v79, 8u, a5, a6);
        v11 = &v130[(unsigned int)v131];
        goto LABEL_4;
      }
      v106 = &dest;
    }
    memcpy(v106, v76, v78);
    LODWORD(v78) = v154;
    goto LABEL_124;
  }
LABEL_6:
  *(_QWORD *)&v134[8 * v6] = a2;
  v13 = v135 + 1;
  LODWORD(v135) = v135 + 1;
  while ( 1 )
  {
    v14 = v134;
    v15 = &v134[8 * v13];
    if ( !v13 )
      break;
    while ( 1 )
    {
      v16 = *((_QWORD *)v15 - 1);
      LODWORD(v135) = --v13;
      v17 = *(_WORD *)(v16 + 24);
      if ( v17 > 0xEu )
      {
        if ( v17 != 15 )
LABEL_184:
          BUG();
        goto LABEL_10;
      }
      if ( v17 > 1u )
        break;
LABEL_10:
      v15 -= 8;
      if ( !v13 )
        goto LABEL_11;
    }
    v50 = sub_D960E0(v16);
    if ( v50 != v50 + 8 * v51 )
    {
      v124 = (__int64 *)(v50 + 8 * v51);
      v54 = (__int64 *)v50;
      while ( 1 )
      {
        v55 = *v54;
        if ( !v141 )
          goto LABEL_89;
        v56 = v138;
        v52 = HIDWORD(v139);
        v51 = (__int64)&v138[HIDWORD(v139)];
        if ( v138 != (__int64 *)v51 )
        {
          while ( v55 != *v56 )
          {
            if ( (__int64 *)v51 == ++v56 )
              goto LABEL_100;
          }
          goto LABEL_87;
        }
LABEL_100:
        if ( HIDWORD(v139) < (unsigned int)v139 )
        {
          ++HIDWORD(v139);
          *(_QWORD *)v51 = v55;
          ++v137;
LABEL_90:
          if ( *(_WORD *)(v55 + 24) == 8 )
          {
            v57 = *(_QWORD *)(v55 + 40);
            v58 = *(_QWORD *)(v55 + 32);
            v59 = (__int64)v133[1];
            if ( v57 == 2 )
            {
              v60 = *(_QWORD **)(v58 + 8);
LABEL_93:
              v61 = *(unsigned int *)(v59 + 8);
              if ( v61 + 1 > (unsigned __int64)*(unsigned int *)(v59 + 12) )
              {
                sub_C8D5F0(v59, (const void *)(v59 + 16), v61 + 1, 8u, v53, a6);
                v61 = *(unsigned int *)(v59 + 8);
              }
              *(_QWORD *)(*(_QWORD *)v59 + 8 * v61) = v60;
              ++*(_DWORD *)(v59 + 8);
              goto LABEL_96;
            }
            v67 = 8 * v57 - 8;
            v119 = (__int64)*v133;
            v68 = v67 >> 3;
            v117 = *(_QWORD *)(v55 + 48);
            v143 = (unsigned int **)&v145;
            v144 = 0x300000000LL;
            if ( (unsigned __int64)v67 > 0x18 )
            {
              v114 = v58;
              sub_C8D5F0((__int64)&v143, &v145, (8 * v57 - 8) >> 3, 8u, v53, a6);
              v58 = v114;
              v73 = (__int64 *)&v143[(unsigned int)v144];
            }
            else
            {
              v69 = &v145;
              if ( 8 * v57 == 8 )
                goto LABEL_108;
              v73 = &v145;
            }
            memcpy(v73, (const void *)(v58 + 8), v67);
            v69 = (__int64 *)v143;
            LODWORD(v67) = v144;
LABEL_108:
            v70 = v67 + v68;
            LODWORD(v144) = v70;
            p_dest = &dest;
            v71 = 8LL * v70;
            v154 = 0x400000000LL;
            if ( v70 > 4uLL )
            {
              src = v69;
              sub_C8D5F0((__int64)&p_dest, &dest, v70, 8u, v53, a6);
              v71 = 8LL * v70;
              v69 = src;
              v72 = &p_dest[(unsigned int)v154];
            }
            else
            {
              if ( !v71 )
              {
LABEL_110:
                LODWORD(v154) = v70 + v71;
                v60 = sub_DBFF60(v119, (unsigned int *)&p_dest, v117, 0);
                if ( p_dest != &dest )
                  _libc_free((unsigned __int64)p_dest);
                if ( v143 != (unsigned int **)&v145 )
                  _libc_free((unsigned __int64)v143);
                goto LABEL_93;
              }
              v72 = &dest;
            }
            memcpy(v72, v69, v71);
            LODWORD(v71) = v154;
            goto LABEL_110;
          }
LABEL_96:
          v62 = (unsigned int)v135;
          v52 = HIDWORD(v135);
          v63 = (unsigned int)v135 + 1LL;
          if ( v63 > HIDWORD(v135) )
          {
            sub_C8D5F0((__int64)&v134, v136, v63, 8u, v53, a6);
            v62 = (unsigned int)v135;
          }
          v51 = (__int64)v134;
          ++v54;
          *(_QWORD *)&v134[8 * v62] = v55;
          LODWORD(v135) = v135 + 1;
          if ( v124 == v54 )
            break;
        }
        else
        {
LABEL_89:
          sub_C8CC70((__int64)&v137, *v54, v51, v52, v53, a6);
          if ( (_BYTE)v51 )
            goto LABEL_90;
LABEL_87:
          if ( v124 == ++v54 )
            break;
        }
      }
    }
    v13 = v135;
  }
LABEL_11:
  if ( !v141 )
  {
    _libc_free((unsigned __int64)v138);
    v14 = v134;
  }
  if ( v14 != v136 )
    _libc_free((unsigned __int64)v14);
  v18 = v130;
  v122 = &v130[(unsigned int)v131];
  if ( v122 != v130 )
  {
    do
    {
      v19 = *v18;
      v150 = 0;
      v14 = v146;
      v151 = 1;
      v126 = a3;
      v144 = (__int64)v146;
      v143 = &v126;
      v145 = 0x800000000LL;
      v152 = v19;
      v148 = &v152;
      v149 = 0x100000008LL;
      v147 = 1;
      if ( (*(_WORD *)(v19 + 24) & 0xFFFD) != 4 && *(_WORD *)(v19 + 24) != 15 )
      {
        v146[0] = v19;
        v20 = 1;
        LODWORD(v145) = 1;
        goto LABEL_19;
      }
      LOBYTE(p_dest) = 0;
      sub_30B6900(v19, &p_dest);
      if ( !(_BYTE)p_dest )
      {
        v65 = v126;
        v66 = v126[2];
        if ( v66 + 1 > (unsigned __int64)v126[3] )
        {
          sub_C8D5F0((__int64)v126, v126 + 4, v66 + 1, 8u, v64, a6);
          v66 = v65[2];
        }
        *(_QWORD *)(*(_QWORD *)v65 + 8 * v66) = v19;
        ++v65[2];
        v14 = (_BYTE *)v144;
        v20 = v145;
        goto LABEL_19;
      }
LABEL_63:
      while ( 1 )
      {
        v14 = (_BYTE *)v144;
        v20 = v145;
LABEL_19:
        v21 = &v14[8 * v20];
        if ( !v20 )
          break;
        while ( 1 )
        {
          v22 = *((_QWORD *)v21 - 1);
          LODWORD(v145) = --v20;
          v23 = *(_WORD *)(v22 + 24);
          if ( v23 <= 0xEu )
            break;
          if ( v23 != 15 )
            goto LABEL_184;
LABEL_22:
          v21 -= 8;
          if ( !v20 )
            goto LABEL_23;
        }
        if ( v23 <= 1u )
          goto LABEL_22;
        v36 = sub_D960E0(v22);
        v40 = (__int64 *)(v36 + 8 * v37);
        v41 = (__int64 *)v36;
        if ( (__int64 *)v36 != v40 )
        {
          while ( 1 )
          {
            while ( 1 )
            {
              v42 = *v41;
              if ( v151 )
              {
                v43 = v148;
                v38 = HIDWORD(v149);
                v37 = (__int64)&v148[HIDWORD(v149)];
                if ( v148 != (__int64 *)v37 )
                {
                  while ( v42 != *v43 )
                  {
                    if ( (__int64 *)v37 == ++v43 )
                      goto LABEL_71;
                  }
                  goto LABEL_62;
                }
LABEL_71:
                if ( HIDWORD(v149) < (unsigned int)v149 )
                  break;
              }
              sub_C8CC70((__int64)&v147, *v41, v37, v38, v39, a6);
              if ( (_BYTE)v37 )
              {
                v44 = *(_WORD *)(v42 + 24);
                if ( (v44 & 0xFFFD) == 4 )
                  goto LABEL_73;
LABEL_66:
                if ( v44 == 15 )
                  goto LABEL_73;
                v45 = (unsigned int)v145;
                v38 = HIDWORD(v145);
                v46 = (unsigned int)v145 + 1LL;
                if ( v46 > HIDWORD(v145) )
                {
                  sub_C8D5F0((__int64)&v144, v146, v46, 8u, v39, a6);
                  v45 = (unsigned int)v145;
                }
                v37 = v144;
                ++v41;
                *(_QWORD *)(v144 + 8 * v45) = v42;
                LODWORD(v145) = v145 + 1;
                if ( v40 == v41 )
                  goto LABEL_63;
              }
              else
              {
LABEL_62:
                if ( v40 == ++v41 )
                  goto LABEL_63;
              }
            }
            ++HIDWORD(v149);
            *(_QWORD *)v37 = v42;
            ++v147;
            v44 = *(_WORD *)(v42 + 24);
            if ( (v44 & 0xFFFD) != 4 )
              goto LABEL_66;
LABEL_73:
            LOBYTE(p_dest) = 0;
            v47 = (__int64 *)v143;
            sub_30B6900(v42, &p_dest);
            if ( (_BYTE)p_dest )
              goto LABEL_62;
            v48 = *v47;
            v49 = *(unsigned int *)(v48 + 8);
            v38 = *(unsigned int *)(v48 + 12);
            if ( v49 + 1 > v38 )
            {
              sub_C8D5F0(v48, (const void *)(v48 + 16), v49 + 1, 8u, v39, a6);
              v49 = *(unsigned int *)(v48 + 8);
            }
            v37 = *(_QWORD *)v48;
            ++v41;
            *(_QWORD *)(*(_QWORD *)v48 + 8 * v49) = v42;
            ++*(_DWORD *)(v48 + 8);
            if ( v40 == v41 )
              goto LABEL_63;
          }
        }
      }
LABEL_23:
      if ( !v151 )
      {
        _libc_free((unsigned __int64)v148);
        v14 = (_BYTE *)v144;
      }
      if ( v14 != (_BYTE *)v146 )
        _libc_free((unsigned __int64)v14);
      ++v18;
    }
    while ( v122 != v18 );
  }
  v161 = 1;
  v160 = 0;
  v128 = a3;
  v157 = 1;
  v129 = a1;
  p_dest = (__int64 *)&v128;
  v154 = (__int64)v156;
  dest = 0x800000000LL;
  v158 = &v162;
  v159 = 0x100000008LL;
  v7 = *(_WORD *)(a2 + 24) == 6;
  v162 = a2;
  if ( !v7 )
  {
    v104 = 0;
LABEL_170:
    *(_QWORD *)(v154 + 8 * v104) = a2;
    v31 = dest + 1;
    LODWORD(dest) = dest + 1;
    goto LABEL_43;
  }
  v144 = 0;
  v143 = (unsigned int **)&v145;
  v24 = *(__int64 **)(a2 + 32);
  v25 = &v24[*(_QWORD *)(a2 + 40)];
  if ( v24 == v25 )
  {
LABEL_173:
    v104 = (unsigned int)dest;
    v105 = (unsigned int)dest + 1LL;
    if ( v105 > HIDWORD(dest) )
    {
      sub_C8D5F0((__int64)&v154, v156, v105, 8u, (__int64)v14, a6);
      v104 = (unsigned int)dest;
    }
    goto LABEL_170;
  }
  v26 = 0;
  do
  {
    while ( 1 )
    {
      v29 = (unsigned int *)*v24;
      if ( *(_WORD *)(*v24 + 24) == 15 )
        break;
      v30 = *v24++;
      LOBYTE(v126) = 0;
      v133 = &v126;
      sub_30B66A0(v30, (_BYTE **)&v133);
      v26 |= (unsigned __int8)v126;
      if ( v25 == v24 )
        goto LABEL_38;
    }
    if ( **((_BYTE **)v29 - 1) == 85 )
    {
      v26 = 1;
    }
    else
    {
      v27 = (unsigned int)v144;
      v28 = (unsigned int)v144 + 1LL;
      if ( v28 > HIDWORD(v144) )
      {
        sub_C8D5F0((__int64)&v143, &v145, v28, 8u, (__int64)v14, a6);
        v27 = (unsigned int)v144;
      }
      v143[v27] = v29;
      LODWORD(v144) = v144 + 1;
    }
    ++v24;
  }
  while ( v25 != v24 );
LABEL_38:
  if ( !(_DWORD)v144 )
  {
    if ( v143 != (unsigned int **)&v145 )
      _libc_free((unsigned __int64)v143);
    goto LABEL_173;
  }
  if ( v26 )
  {
    v107 = (__int64)v128;
    v110 = sub_DC8BD0(v129, (__int64)&v143, 0, 0);
    v111 = *(unsigned int *)(v107 + 8);
    if ( v111 + 1 > (unsigned __int64)*(unsigned int *)(v107 + 12) )
    {
      sub_C8D5F0(v107, (const void *)(v107 + 16), v111 + 1, 8u, v108, v109);
      v111 = *(unsigned int *)(v107 + 8);
    }
    *(_QWORD *)(*(_QWORD *)v107 + 8 * v111) = v110;
    ++*(_DWORD *)(v107 + 8);
  }
  if ( v143 != (unsigned int **)&v145 )
    _libc_free((unsigned __int64)v143);
  v31 = dest;
LABEL_43:
  while ( 1 )
  {
    v32 = (_BYTE *)v154;
    v33 = v154 + 8LL * v31;
    if ( !v31 )
      break;
    while ( 1 )
    {
      v34 = *(_QWORD *)(v33 - 8);
      LODWORD(dest) = --v31;
      v35 = *(_WORD *)(v34 + 24);
      if ( v35 <= 0xEu )
        break;
      if ( v35 != 15 )
        goto LABEL_184;
LABEL_46:
      v33 -= 8;
      if ( !v31 )
        goto LABEL_47;
    }
    if ( v35 <= 1u )
      goto LABEL_46;
    v80 = sub_D960E0(v34);
    v85 = (__int64 *)(v80 + 8 * v81);
    if ( (__int64 *)v80 != v85 )
    {
      v86 = (__int64 *)v80;
      while ( 2 )
      {
        v87 = *v86;
        if ( v161 )
        {
          v88 = v158;
          v82 = HIDWORD(v159);
          v81 = (__int64)&v158[HIDWORD(v159)];
          if ( v158 != (__int64 *)v81 )
          {
            while ( v87 != *v88 )
            {
              if ( (__int64 *)v81 == ++v88 )
                goto LABEL_157;
            }
            goto LABEL_140;
          }
LABEL_157:
          if ( HIDWORD(v159) < (unsigned int)v159 )
          {
            v82 = (unsigned int)++HIDWORD(v159);
            *(_QWORD *)v81 = v87;
            ++v157;
            if ( *(_WORD *)(v87 + 24) != 6 )
              goto LABEL_159;
            goto LABEL_144;
          }
        }
        sub_C8CC70((__int64)&v157, *v86, v81, v82, v83, v84);
        if ( (_BYTE)v81 )
        {
          if ( *(_WORD *)(v87 + 24) != 6 )
            goto LABEL_159;
LABEL_144:
          v144 = 0;
          v143 = (unsigned int **)&v145;
          v89 = *(__int64 **)(v87 + 32);
          v90 = *(_QWORD *)(v87 + 40);
          v84 = (__int64)&v89[v90];
          if ( v89 == (__int64 *)v84 )
            goto LABEL_159;
          v120 = v85;
          v91 = (unsigned __int64)p_dest;
          v92 = &v89[v90];
          v118 = v87;
          v93 = 0;
          do
          {
            while ( 1 )
            {
              v83 = *v89;
              if ( *(_WORD *)(*v89 + 24) == 15 )
                break;
              v96 = *v89++;
              LOBYTE(v126) = 0;
              v133 = &v126;
              sub_30B66A0(v96, (_BYTE **)&v133);
              v93 |= (unsigned __int8)v126;
              if ( v92 == v89 )
                goto LABEL_153;
            }
            if ( **(_BYTE **)(v83 - 8) == 85 )
            {
              v93 = 1;
            }
            else
            {
              v94 = (unsigned int)v144;
              v82 = HIDWORD(v144);
              v95 = (unsigned int)v144 + 1LL;
              if ( v95 > HIDWORD(v144) )
              {
                v116 = *v89;
                sub_C8D5F0((__int64)&v143, &v145, v95, 8u, v83, v84);
                v94 = (unsigned int)v144;
                v83 = v116;
              }
              v81 = (__int64)v143;
              v143[v94] = (unsigned int *)v83;
              LODWORD(v144) = v144 + 1;
            }
            ++v89;
          }
          while ( v92 != v89 );
LABEL_153:
          v97 = v93;
          v85 = v120;
          v87 = v118;
          if ( (_DWORD)v144 )
          {
            if ( v97 )
            {
              v100 = *(_QWORD *)v91;
              v101 = sub_DC8BD0(*(__int64 **)(v91 + 8), (__int64)&v143, 0, 0);
              v82 = *(unsigned int *)(v100 + 12);
              v102 = v101;
              v103 = *(unsigned int *)(v100 + 8);
              if ( v103 + 1 > v82 )
              {
                sub_C8D5F0(v100, (const void *)(v100 + 16), v103 + 1, 8u, v83, v84);
                v103 = *(unsigned int *)(v100 + 8);
              }
              v81 = *(_QWORD *)v100;
              *(_QWORD *)(*(_QWORD *)v100 + 8 * v103) = v102;
              ++*(_DWORD *)(v100 + 8);
            }
            if ( v143 != (unsigned int **)&v145 )
              _libc_free((unsigned __int64)v143);
          }
          else
          {
            if ( v143 != (unsigned int **)&v145 )
              _libc_free((unsigned __int64)v143);
LABEL_159:
            v98 = (unsigned int)dest;
            v82 = HIDWORD(dest);
            v99 = (unsigned int)dest + 1LL;
            if ( v99 > HIDWORD(dest) )
            {
              sub_C8D5F0((__int64)&v154, v156, v99, 8u, v83, v84);
              v98 = (unsigned int)dest;
            }
            v81 = v154;
            *(_QWORD *)(v154 + 8 * v98) = v87;
            LODWORD(dest) = dest + 1;
          }
        }
LABEL_140:
        if ( v85 == ++v86 )
          break;
        continue;
      }
    }
    v31 = dest;
  }
LABEL_47:
  if ( !v161 )
  {
    _libc_free((unsigned __int64)v158);
    v32 = (_BYTE *)v154;
  }
  if ( v32 != v156 )
    _libc_free((unsigned __int64)v32);
  if ( v130 != (__int64 *)v132 )
    _libc_free((unsigned __int64)v130);
}
