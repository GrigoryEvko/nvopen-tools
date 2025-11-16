// Function: sub_110B960
// Address: 0x110b960
//
unsigned __int8 *__fastcall sub_110B960(__int64 *a1, _BYTE *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _BYTE *v6; // r10
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  __int64 v9; // r12
  __int64 *v10; // rsi
  __int64 v11; // r9
  unsigned int v12; // eax
  _BYTE *v13; // r10
  __int64 v14; // rcx
  __int64 v15; // rdx
  unsigned __int8 **v16; // r8
  __int64 v17; // rdx
  unsigned __int8 **v18; // rbx
  unsigned __int8 **v19; // rax
  __int64 v20; // r8
  unsigned __int8 **v21; // r12
  unsigned __int8 *v22; // r13
  unsigned __int8 v23; // al
  char v24; // al
  __int64 *v25; // r14
  __int64 v26; // rax
  _BYTE *v27; // r15
  __int64 *v28; // r8
  __int64 *v29; // r13
  __int64 v30; // r14
  __int64 v31; // rbx
  char *v32; // r12
  unsigned __int8 v33; // al
  unsigned __int8 *v34; // r12
  __int64 v35; // r13
  char v36; // bl
  __int64 v37; // rax
  __int64 v38; // rdx
  unsigned __int64 v39; // rbx
  char v40; // r14
  __int64 v41; // rax
  __int64 v42; // rdx
  unsigned __int64 v43; // rax
  __int64 v44; // rbx
  __int64 v45; // r15
  char v46; // r14
  __int64 v47; // rax
  __int64 v48; // rdx
  unsigned __int64 v49; // rax
  unsigned __int64 v50; // r14
  char v51; // r15
  __int64 v52; // rax
  __int64 v53; // rdx
  unsigned __int64 v54; // rax
  __int64 v55; // rsi
  _BYTE *v57; // rax
  __int64 v58; // rax
  __int64 v59; // rax
  unsigned __int64 v60; // r13
  unsigned __int64 v61; // rdx
  __int64 *v62; // rax
  bool v63; // al
  __int64 *v64; // rax
  __int64 v65; // rdi
  __int64 v66; // r14
  int v67; // r15d
  __int64 v68; // rax
  int v69; // r15d
  __int64 v70; // rbx
  __int64 v71; // r10
  __int64 v72; // r15
  __int64 v73; // r14
  __int64 v74; // rdx
  unsigned int v75; // esi
  __int64 *v76; // rax
  __int64 v77; // rdx
  __int64 v78; // r13
  __int64 v79; // r15
  __int64 v80; // rbx
  __int64 v81; // rax
  __int64 v82; // rcx
  __int64 v83; // rax
  __int64 v84; // r14
  int v85; // eax
  int v86; // eax
  __int64 v87; // rax
  __int64 v88; // r14
  char v89; // si
  __int64 *v90; // rax
  int v91; // eax
  __int64 *v92; // r9
  __int64 v93; // r10
  __int64 v94; // rax
  __int64 v95; // rax
  __int64 v96; // rcx
  int v97; // r9d
  int v98; // edx
  __int64 v99; // r14
  __int64 v100; // r12
  __int64 *v101; // r13
  char v102; // al
  __int64 v103; // rbx
  __int64 v104; // r15
  __int64 *v105; // rdx
  __int64 v106; // rax
  __int64 v107; // rax
  unsigned __int8 *v108; // rax
  __int64 v109; // rax
  __int64 v110; // rdx
  int v111; // r8d
  __int64 v112; // rcx
  __int64 v113; // rbx
  __int64 *v114; // rax
  __int64 v115; // r13
  __int64 v116; // r12
  __int64 *v117; // r15
  __int64 v118; // rdx
  unsigned int v119; // esi
  __int64 *v120; // rax
  __int64 v121; // rdx
  int v122; // r15d
  _BYTE *v123; // [rsp+10h] [rbp-1E0h]
  __int64 v124; // [rsp+18h] [rbp-1D8h]
  int v125; // [rsp+18h] [rbp-1D8h]
  __int64 v126; // [rsp+18h] [rbp-1D8h]
  __int64 v127; // [rsp+20h] [rbp-1D0h]
  _BYTE *v128; // [rsp+20h] [rbp-1D0h]
  __int64 v129; // [rsp+28h] [rbp-1C8h]
  _BYTE *v130; // [rsp+28h] [rbp-1C8h]
  __int64 *v131; // [rsp+30h] [rbp-1C0h]
  __int64 v132; // [rsp+30h] [rbp-1C0h]
  __int64 *v133; // [rsp+30h] [rbp-1C0h]
  __int64 *v134; // [rsp+30h] [rbp-1C0h]
  __int64 v135; // [rsp+38h] [rbp-1B8h]
  __int64 *v136; // [rsp+38h] [rbp-1B8h]
  __int64 v137; // [rsp+40h] [rbp-1B0h]
  _BYTE *v138; // [rsp+48h] [rbp-1A8h]
  _BYTE *v139; // [rsp+48h] [rbp-1A8h]
  __int64 *v140; // [rsp+48h] [rbp-1A8h]
  __int64 *v141; // [rsp+48h] [rbp-1A8h]
  __int64 v142; // [rsp+48h] [rbp-1A8h]
  unsigned __int8 *v143; // [rsp+48h] [rbp-1A8h]
  __int64 v144; // [rsp+48h] [rbp-1A8h]
  _BYTE *v145; // [rsp+50h] [rbp-1A0h]
  __int64 v146; // [rsp+50h] [rbp-1A0h]
  __int64 *v147; // [rsp+50h] [rbp-1A0h]
  __int64 v148; // [rsp+50h] [rbp-1A0h]
  __int64 *v149; // [rsp+50h] [rbp-1A0h]
  __int64 *v150; // [rsp+50h] [rbp-1A0h]
  __int64 v151; // [rsp+58h] [rbp-198h]
  __int64 v152; // [rsp+60h] [rbp-190h]
  __int64 v153; // [rsp+60h] [rbp-190h]
  __int64 v155; // [rsp+70h] [rbp-180h]
  __int64 v156[2]; // [rsp+78h] [rbp-178h] BYREF
  __int64 v157; // [rsp+88h] [rbp-168h] BYREF
  __int64 v158[4]; // [rsp+90h] [rbp-160h] BYREF
  __int16 v159; // [rsp+B0h] [rbp-140h]
  const char *v160[4]; // [rsp+C0h] [rbp-130h] BYREF
  __int16 v161; // [rsp+E0h] [rbp-110h]
  _QWORD *v162; // [rsp+F0h] [rbp-100h] BYREF
  __int64 v163; // [rsp+F8h] [rbp-F8h]
  _QWORD v164[4]; // [rsp+100h] [rbp-F0h] BYREF
  __int64 v165; // [rsp+120h] [rbp-D0h] BYREF
  __int64 *v166; // [rsp+128h] [rbp-C8h]
  __int64 v167; // [rsp+130h] [rbp-C0h]
  __int64 v168; // [rsp+138h] [rbp-B8h]
  _BYTE *v169; // [rsp+140h] [rbp-B0h]
  __int64 v170; // [rsp+148h] [rbp-A8h]
  _BYTE v171[32]; // [rsp+150h] [rbp-A0h] BYREF
  unsigned __int64 v172; // [rsp+170h] [rbp-80h] BYREF
  __int64 v173; // [rsp+178h] [rbp-78h]
  __int64 v174; // [rsp+180h] [rbp-70h] BYREF
  unsigned int v175; // [rsp+188h] [rbp-68h]
  char v176; // [rsp+1C0h] [rbp-30h] BYREF

  v6 = a2;
  v7 = *((_QWORD *)a2 + 2);
  v156[0] = a3;
  if ( !v7 )
    return 0;
  while ( 1 )
  {
    v8 = *(_QWORD *)(v7 + 24);
    if ( *(_BYTE *)v8 != 62 )
      break;
    v7 = *(_QWORD *)(v7 + 8);
    if ( !v7 )
      return 0;
  }
  v9 = *((_QWORD *)a2 + 1);
  v155 = *(_QWORD *)(*((_QWORD *)a2 - 4) + 8LL);
  if ( (_BYTE)qword_4F90288 )
  {
    v35 = a1[11];
    v36 = sub_AE5020(v35, v9);
    v37 = sub_9208B0(v35, v9);
    v173 = v38;
    v172 = ((1LL << v36) + ((unsigned __int64)(v37 + 7) >> 3) - 1) >> v36 << v36;
    v39 = sub_CA1930(&v172);
    v152 = a1[11];
    v40 = sub_AE5020(v152, v155);
    v41 = sub_9208B0(v152, v155);
    v173 = v42;
    v172 = ((1LL << v40) + ((unsigned __int64)(v41 + 7) >> 3) - 1) >> v40 << v40;
    v43 = sub_CA1930(&v172);
    v6 = a2;
    if ( v39 > 3 || v39 >= v43 )
    {
      if ( (unsigned int)*(unsigned __int8 *)(v9 + 8) - 17 > 1 )
        goto LABEL_4;
      v44 = *(_QWORD *)(v9 + 24);
      v45 = a1[11];
      v46 = sub_AE5020(v45, v44);
      v47 = sub_9208B0(v45, v44);
      v173 = v48;
      v172 = ((1LL << v46) + ((unsigned __int64)(v47 + 7) >> 3) - 1) >> v46 << v46;
      v49 = sub_CA1930(&v172);
      v6 = a2;
      v50 = v49;
      if ( v49 > 3 )
        goto LABEL_4;
      a4 = v155;
      v8 = *(unsigned __int8 *)(v155 + 8);
      if ( (unsigned int)(v8 - 17) > 1 )
      {
        if ( (unsigned __int8)v8 > 3u && (_BYTE)v8 != 5 )
        {
          if ( (unsigned __int8)v8 > 0x14u )
            goto LABEL_4;
          v95 = 1070160;
          if ( !_bittest64(&v95, v8) )
            goto LABEL_4;
        }
      }
      else
      {
        v151 = *(_QWORD *)(v155 + 24);
        v153 = a1[11];
        v51 = sub_AE5020(v153, v151);
        v52 = sub_9208B0(v153, v151);
        v173 = v53;
        v172 = ((1LL << v51) + ((unsigned __int64)(v52 + 7) >> 3) - 1) >> v51 << v51;
        v54 = sub_CA1930(&v172);
        v6 = a2;
        if ( v50 >= v54 )
          goto LABEL_4;
      }
    }
    return 0;
  }
LABEL_4:
  v145 = v6;
  v10 = v156;
  v162 = v164;
  v169 = v171;
  v170 = 0x400000000LL;
  v165 = 0;
  v164[0] = v156[0];
  v166 = 0;
  v167 = 0;
  v168 = 0;
  v163 = 0x400000001LL;
  sub_110B2F0((__int64)&v165, v156, v8, a4, a5, a6);
  v12 = v163;
  v13 = v145;
  while ( v12 )
  {
    v14 = v12--;
    v15 = v162[v14 - 1];
    LODWORD(v163) = v12;
    if ( (*(_BYTE *)(v15 + 7) & 0x40) != 0 )
    {
      v18 = *(unsigned __int8 ***)(v15 - 8);
      v17 = 32LL * (*(_DWORD *)(v15 + 4) & 0x7FFFFFF);
      v16 = (unsigned __int8 **)((char *)v18 + v17);
    }
    else
    {
      v16 = (unsigned __int8 **)v15;
      v17 = 32LL * (*(_DWORD *)(v15 + 4) & 0x7FFFFFF);
      v18 = (unsigned __int8 **)((char *)v16 - v17);
    }
    if ( v16 != v18 )
    {
      v19 = v16;
      v20 = v9;
      v21 = v19;
      do
      {
        v22 = *v18;
        v23 = **v18;
        if ( v23 > 0x15u )
        {
          if ( v23 == 61 )
          {
            v57 = (_BYTE *)*((_QWORD *)v22 - 4);
            if ( !v57 )
              BUG();
            if ( v13 == v57
              || *v57 == 61
              || *(_BYTE *)(v20 + 8) == 10
              || (v58 = *((_QWORD *)v22 + 2)) == 0
              || *(_QWORD *)(v58 + 8)
              || (v139 = v13, v148 = v20, sub_B46500(*v18))
              || (v22[2] & 1) != 0 )
            {
LABEL_34:
              v27 = v169;
              v34 = 0;
              goto LABEL_35;
            }
            v20 = v148;
            v13 = v139;
          }
          else if ( v23 == 84 )
          {
            v10 = (__int64 *)&v172;
            v172 = (unsigned __int64)*v18;
            v138 = v13;
            v146 = v20;
            v24 = sub_110B2F0((__int64)&v165, (__int64 *)&v172, v17, v14, v20, v11);
            v20 = v146;
            v13 = v138;
            if ( v24 )
            {
              v59 = (unsigned int)v163;
              v14 = HIDWORD(v163);
              v60 = v172;
              v61 = (unsigned int)v163 + 1LL;
              if ( v61 > HIDWORD(v163) )
              {
                v10 = v164;
                sub_C8D5F0((__int64)&v162, v164, v61, 8u, v146, v11);
                v59 = (unsigned int)v163;
                v13 = v138;
                v20 = v146;
              }
              v17 = (__int64)v162;
              v162[v59] = v60;
              LODWORD(v163) = v163 + 1;
            }
          }
          else if ( *v22 != 78 || *(_QWORD *)(*((_QWORD *)v22 - 4) + 8LL) != v20 || *((_QWORD *)v22 + 1) != v155 )
          {
            goto LABEL_34;
          }
        }
        v18 += 4;
      }
      while ( v21 != v18 );
      v12 = v163;
      v9 = v20;
    }
  }
  v25 = (__int64 *)v169;
  v127 = (unsigned int)v170;
  v26 = 8LL * (unsigned int)v170;
  v27 = v169;
  v147 = (__int64 *)&v169[v26];
  if ( &v169[v26] != v169 )
  {
    v135 = v9;
    v28 = (__int64 *)v169;
    v124 = v26 >> 3;
    v123 = v13;
    v131 = (__int64 *)v169;
    v29 = (__int64 *)&v169[32 * (v26 >> 5)];
    v30 = v26 >> 5;
    v129 = (v26 - 32 * (v26 >> 5)) >> 3;
    while ( 1 )
    {
      v137 = *v131;
      if ( *(_QWORD *)(*v131 + 16) )
        break;
LABEL_68:
      if ( v147 == ++v131 )
      {
        v9 = v135;
        v13 = v123;
        v25 = v28;
        goto LABEL_70;
      }
    }
    v31 = *(_QWORD *)(*v131 + 16);
    while ( 1 )
    {
      v32 = *(char **)(v31 + 24);
      v33 = *v32;
      if ( (unsigned __int8)*v32 <= 0x1Cu )
      {
LABEL_24:
        v34 = 0;
        goto LABEL_35;
      }
      switch ( v33 )
      {
        case '>':
          v141 = v28;
          if ( sub_B46500(*(unsigned __int8 **)(v31 + 24)) )
            goto LABEL_24;
          if ( (v32[2] & 1) != 0 )
            goto LABEL_24;
          v94 = *((_QWORD *)v32 - 8);
          if ( !v94 )
            goto LABEL_24;
          v28 = v141;
          if ( v137 != v94 )
            goto LABEL_24;
          break;
        case 'N':
          if ( *(_QWORD *)(*((_QWORD *)v32 - 4) + 8LL) != v155 || *((_QWORD *)v32 + 1) != v135 )
            goto LABEL_24;
          break;
        case 'T':
          if ( (_DWORD)v167 )
          {
            v10 = v166;
            if ( !(_DWORD)v168 )
              goto LABEL_24;
            v91 = (v168 - 1) & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
            v92 = &v166[v91];
            v93 = *v92;
            if ( v32 != (char *)*v92 )
            {
              v97 = 1;
              while ( v93 != -4096 )
              {
                v98 = v97 + 1;
                v91 = (v168 - 1) & (v91 + v97);
                v92 = &v166[v91];
                v93 = *v92;
                if ( v32 == (char *)*v92 )
                  goto LABEL_105;
                v97 = v98;
              }
              goto LABEL_24;
            }
LABEL_105:
            v63 = &v166[(unsigned int)v168] != v92;
            goto LABEL_66;
          }
          if ( v30 )
          {
            v62 = v28;
            while ( 1 )
            {
              if ( v32 == (char *)*v62 )
                goto LABEL_65;
              if ( v32 == (char *)v62[1] )
              {
                v63 = v147 != v62 + 1;
                goto LABEL_66;
              }
              if ( v32 == (char *)v62[2] )
              {
                v63 = v147 != v62 + 2;
                goto LABEL_66;
              }
              if ( v32 == (char *)v62[3] )
                break;
              v62 += 4;
              if ( v29 == v62 )
              {
                v96 = v129;
                goto LABEL_123;
              }
            }
            v63 = v147 != v62 + 3;
LABEL_66:
            if ( !v63 )
              goto LABEL_24;
            break;
          }
          v96 = v124;
          v62 = v28;
LABEL_123:
          if ( v96 == 2 )
            goto LABEL_130;
          if ( v96 != 3 )
          {
            if ( v96 != 1 )
              goto LABEL_24;
            goto LABEL_126;
          }
          if ( v32 != (char *)*v62 )
          {
            ++v62;
LABEL_130:
            if ( v32 != (char *)*v62 )
            {
              ++v62;
LABEL_126:
              if ( v32 != (char *)*v62 )
                goto LABEL_24;
            }
          }
LABEL_65:
          v63 = v147 != v62;
          goto LABEL_66;
        default:
          goto LABEL_24;
      }
      v31 = *(_QWORD *)(v31 + 8);
      if ( !v31 )
        goto LABEL_68;
    }
  }
LABEL_70:
  v172 = 0;
  v64 = &v174;
  v173 = 1;
  do
  {
    *v64 = -4096;
    v64 += 2;
  }
  while ( v64 != (__int64 *)&v176 );
  if ( v147 != v25 )
  {
    v140 = v25;
    v128 = v13;
    do
    {
      v65 = a1[4];
      v157 = *v140;
      sub_D5F1F0(v65, v157);
      v159 = 257;
      v66 = a1[4];
      v67 = *(_DWORD *)(v157 + 4);
      v161 = 257;
      v68 = sub_BD2DA0(80);
      v69 = v67 & 0x7FFFFFF;
      v70 = v68;
      if ( v68 )
      {
        v132 = v68;
        sub_B44260(v68, v9, 55, 0x8000000u, 0, 0);
        *(_DWORD *)(v70 + 72) = v69;
        sub_BD6B50((unsigned __int8 *)v70, v160);
        sub_BD2A10(v70, *(_DWORD *)(v70 + 72), 1);
        v71 = v132;
      }
      else
      {
        v71 = 0;
      }
      if ( (unsigned __int8)sub_920620(v71) )
      {
        v121 = *(_QWORD *)(v66 + 96);
        v122 = *(_DWORD *)(v66 + 104);
        if ( v121 )
          sub_B99FD0(v70, 3u, v121);
        sub_B45150(v70, v122);
      }
      (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v66 + 88) + 16LL))(
        *(_QWORD *)(v66 + 88),
        v70,
        v158,
        *(_QWORD *)(v66 + 56),
        *(_QWORD *)(v66 + 64));
      v72 = *(_QWORD *)v66;
      v73 = *(_QWORD *)v66 + 16LL * *(unsigned int *)(v66 + 8);
      while ( v73 != v72 )
      {
        v74 = *(_QWORD *)(v72 + 8);
        v75 = *(_DWORD *)v72;
        v72 += 16;
        sub_B99FD0(v70, v75, v74);
      }
      v10 = &v157;
      ++v140;
      *sub_1104E10((__int64)&v172, &v157) = v70;
    }
    while ( v147 != v140 );
    v13 = v128;
    v25 = (__int64 *)v169;
    v127 = (unsigned int)v170;
  }
  v133 = &v25[v127];
  if ( v133 != v25 )
  {
    v149 = v25;
    v130 = v13;
    do
    {
      v10 = v158;
      v158[0] = *v149;
      v76 = sub_1104E10((__int64)&v172, v158);
      v77 = v158[0];
      v78 = *v76;
      if ( (*(_DWORD *)(v158[0] + 4) & 0x7FFFFFF) != 0 )
      {
        v79 = 0;
        v80 = 8LL * (*(_DWORD *)(v158[0] + 4) & 0x7FFFFFF);
        while ( 1 )
        {
          v83 = *(_QWORD *)(v77 - 8);
          v88 = *(_QWORD *)(v83 + 4 * v79);
          v89 = *(_BYTE *)v88;
          if ( *(_BYTE *)v88 <= 0x15u )
          {
            v81 = sub_AD4C90(*(_QWORD *)(v83 + 4 * v79), (__int64 **)v9, 0);
            v77 = v158[0];
            v82 = v81;
            v83 = *(_QWORD *)(v158[0] - 8);
          }
          else if ( v89 == 61 )
          {
            sub_D5F1F0(a1[4], v88);
            v161 = 257;
            v144 = sub_114A4F0(a1, v88, v9, v160);
            v109 = sub_ACADE0(*(__int64 ***)(v88 + 8));
            sub_F162A0((__int64)a1, v88, v109);
            sub_F207A0((__int64)a1, (__int64 *)v88);
            v77 = v158[0];
            v82 = v144;
            v83 = *(_QWORD *)(v158[0] - 8);
          }
          else if ( v89 == 78 )
          {
            v82 = *(_QWORD *)(v88 - 32);
          }
          else
          {
            v82 = 0;
            if ( v89 == 84 )
            {
              v160[0] = *(const char **)(v83 + 4 * v79);
              v90 = sub_1104E10((__int64)&v172, (__int64 *)v160);
              v77 = v158[0];
              v82 = *v90;
            }
            v83 = *(_QWORD *)(v77 - 8);
          }
          v84 = *(_QWORD *)(32LL * *(unsigned int *)(v77 + 72) + v83 + v79);
          v85 = *(_DWORD *)(v78 + 4) & 0x7FFFFFF;
          if ( v85 == *(_DWORD *)(v78 + 72) )
          {
            v142 = v82;
            sub_B48D90(v78);
            v82 = v142;
            v85 = *(_DWORD *)(v78 + 4) & 0x7FFFFFF;
          }
          v86 = (v85 + 1) & 0x7FFFFFF;
          v10 = (__int64 *)(v86 | *(_DWORD *)(v78 + 4) & 0xF8000000);
          v87 = *(_QWORD *)(v78 - 8) + 32LL * (unsigned int)(v86 - 1);
          *(_DWORD *)(v78 + 4) = (_DWORD)v10;
          if ( *(_QWORD *)v87 )
          {
            v10 = *(__int64 **)(v87 + 8);
            **(_QWORD **)(v87 + 16) = v10;
            if ( v10 )
              v10[2] = *(_QWORD *)(v87 + 16);
          }
          *(_QWORD *)v87 = v82;
          if ( v82 )
          {
            v10 = *(__int64 **)(v82 + 16);
            *(_QWORD *)(v87 + 8) = v10;
            if ( v10 )
              v10[2] = v87 + 8;
            *(_QWORD *)(v87 + 16) = v82 + 16;
            *(_QWORD *)(v82 + 16) = v87;
          }
          v79 += 8;
          *(_QWORD *)(*(_QWORD *)(v78 - 8)
                    + 32LL * *(unsigned int *)(v78 + 72)
                    + 8LL * ((*(_DWORD *)(v78 + 4) & 0x7FFFFFFu) - 1)) = v84;
          if ( v80 == v79 )
            break;
          v77 = v158[0];
        }
      }
      ++v149;
    }
    while ( v133 != v149 );
    v13 = v130;
    v25 = (__int64 *)v169;
    v127 = (unsigned int)v170;
  }
  v136 = &v25[v127];
  if ( v136 != v25 )
  {
    v150 = v25;
    v143 = 0;
    v134 = (__int64 *)v13;
    while ( 1 )
    {
      v10 = &v157;
      v157 = *v150;
      v99 = *sub_1104E10((__int64)&v172, &v157);
      v100 = *(_QWORD *)(v157 + 16);
      if ( v100 )
        break;
LABEL_162:
      if ( v136 == ++v150 )
      {
        v34 = v143;
        goto LABEL_164;
      }
    }
    while ( 1 )
    {
      v101 = *(__int64 **)(v100 + 24);
      v102 = *(_BYTE *)v101;
      if ( *(_BYTE *)v101 <= 0x1Cu )
LABEL_187:
        BUG();
      v100 = *(_QWORD *)(v100 + 8);
      if ( v102 != 62 )
      {
        if ( v102 == 78 )
        {
          v10 = v101;
          v108 = sub_F162A0((__int64)a1, (__int64)v101, v99);
          if ( v134 != v101 )
            v108 = v143;
          v143 = v108;
        }
        else if ( v102 != 84 )
        {
          goto LABEL_187;
        }
        goto LABEL_146;
      }
      sub_D5F1F0(a1[4], (__int64)v101);
      v103 = a1[4];
      v159 = 257;
      if ( v155 == *(_QWORD *)(v99 + 8) )
        break;
      v104 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(v103 + 80) + 120LL))(
               *(_QWORD *)(v103 + 80),
               49,
               v99);
      if ( v104 )
        goto LABEL_151;
      v161 = 257;
      v104 = sub_B51D30(49, v99, v155, (__int64)v160, 0, 0);
      if ( (unsigned __int8)sub_920620(v104) )
      {
        v110 = *(_QWORD *)(v103 + 96);
        v111 = *(_DWORD *)(v103 + 104);
        if ( v110 )
        {
          v125 = *(_DWORD *)(v103 + 104);
          sub_B99FD0(v104, 3u, v110);
          v111 = v125;
        }
        sub_B45150(v104, v111);
      }
      (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v103 + 88) + 16LL))(
        *(_QWORD *)(v103 + 88),
        v104,
        v158,
        *(_QWORD *)(v103 + 56),
        *(_QWORD *)(v103 + 64));
      v112 = *(_QWORD *)v103 + 16LL * *(unsigned int *)(v103 + 8);
      v113 = *(_QWORD *)v103;
      if ( v113 != v112 )
      {
        v114 = v101;
        v126 = v100;
        v115 = v104;
        v116 = v112;
        v117 = v114;
        do
        {
          v118 = *(_QWORD *)(v113 + 8);
          v119 = *(_DWORD *)v113;
          v113 += 16;
          sub_B99FD0(v115, v119, v118);
        }
        while ( v116 != v113 );
        v120 = v117;
        v100 = v126;
        v104 = v115;
        v101 = v120;
      }
      v105 = v101 - 8;
      if ( *(v101 - 8) )
      {
        v106 = *(v101 - 7);
        *(_QWORD *)*(v101 - 6) = v106;
        if ( v106 )
          goto LABEL_153;
      }
LABEL_154:
      *(v101 - 8) = v104;
      if ( v104 )
        goto LABEL_155;
LABEL_158:
      v10 = v101;
      sub_F15FC0(a1[5], (__int64)v101);
LABEL_146:
      if ( !v100 )
        goto LABEL_162;
    }
    v104 = v99;
LABEL_151:
    v105 = v101 - 8;
    if ( !*(v101 - 8) || (v106 = *(v101 - 7), (*(_QWORD *)*(v101 - 6) = v106) == 0) )
    {
      *(v101 - 8) = v104;
LABEL_155:
      v107 = *(_QWORD *)(v104 + 16);
      *(v101 - 7) = v107;
      if ( v107 )
        *(_QWORD *)(v107 + 16) = v101 - 7;
      *(v101 - 6) = v104 + 16;
      *(_QWORD *)(v104 + 16) = v105;
      goto LABEL_158;
    }
LABEL_153:
    *(_QWORD *)(v106 + 16) = *(v101 - 6);
    goto LABEL_154;
  }
  v34 = 0;
LABEL_164:
  if ( (v173 & 1) == 0 )
  {
    v10 = (__int64 *)(16LL * v175);
    sub_C7D6A0(v174, (__int64)v10, 8);
  }
  v27 = v169;
LABEL_35:
  if ( v27 != v171 )
    _libc_free(v27, v10);
  v55 = 8LL * (unsigned int)v168;
  sub_C7D6A0((__int64)v166, v55, 8);
  if ( v162 != v164 )
    _libc_free(v162, v55);
  return v34;
}
