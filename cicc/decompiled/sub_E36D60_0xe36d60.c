// Function: sub_E36D60
// Address: 0xe36d60
//
__int64 __fastcall sub_E36D60(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // r13
  __int64 v5; // rax
  unsigned __int64 v6; // rax
  int v7; // eax
  unsigned __int64 v8; // rcx
  unsigned __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r9
  __int64 v16; // rsi
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // r9
  _BYTE *v21; // rdi
  __int64 *v22; // r15
  __int64 v23; // r12
  unsigned int v24; // edx
  __int64 v25; // r8
  _QWORD *v26; // rbx
  __int64 v27; // rcx
  _QWORD *v28; // rdi
  __int64 v29; // rbx
  __int64 v30; // r12
  int v31; // ecx
  _BYTE *v32; // r15
  int v33; // ecx
  unsigned int v34; // edx
  _QWORD *v35; // rax
  _BYTE *v36; // r10
  __int64 v37; // r8
  __int64 v38; // rax
  unsigned __int64 v39; // rdx
  unsigned __int64 v40; // rax
  __int64 v41; // r14
  __int64 v42; // rax
  __int64 v43; // r12
  __int64 v44; // rdx
  unsigned int v45; // eax
  __int64 v46; // r10
  int v47; // r11d
  __int64 *v48; // rbx
  unsigned int v49; // edx
  _QWORD *v50; // r8
  __int64 v51; // rax
  __int64 **v52; // r10
  __int64 v53; // rax
  _QWORD *v54; // rbx
  __int64 *v55; // r12
  __int64 **v56; // r9
  __int64 **v57; // r10
  __int64 **v58; // r9
  __int64 *v59; // r13
  __int64 **v60; // r9
  __int64 **v61; // r10
  __int64 v62; // rsi
  __int64 v63; // rax
  _QWORD *v64; // rbx
  _QWORD *v65; // r12
  _QWORD *v66; // rdi
  int v68; // eax
  _QWORD *v69; // r12
  __int64 v70; // rax
  __int64 v71; // rdx
  __int64 v72; // rcx
  __int64 v73; // r13
  unsigned int v74; // edx
  __int64 v75; // rdx
  __int64 v76; // rcx
  __int64 v77; // rdx
  unsigned __int64 v78; // r8
  __int64 *v79; // rcx
  __int64 v80; // rdx
  __int64 v81; // rcx
  __int64 v82; // r8
  unsigned int v83; // edx
  __int64 *v84; // r10
  __int64 v85; // rdi
  __int64 *v86; // r15
  int v87; // ebx
  __int64 v88; // r9
  __int64 v89; // r14
  __int64 v90; // r12
  __int64 v91; // rdx
  __int64 v92; // rax
  __int64 *v93; // r10
  int v94; // r13d
  __int64 *v95; // rcx
  __int64 *v96; // rax
  int v97; // eax
  int v98; // edi
  __int64 **v99; // r11
  unsigned int v100; // edx
  __int64 *v101; // r11
  __int64 *v102; // rcx
  unsigned int v103; // r13d
  int v104; // r8d
  __int64 v105; // rdx
  int v106; // r8d
  unsigned int v107; // r8d
  __int64 v108; // [rsp+8h] [rbp-BF8h]
  __int64 v109; // [rsp+10h] [rbp-BF0h]
  __int64 v110; // [rsp+18h] [rbp-BE8h]
  __int64 v111; // [rsp+30h] [rbp-BD0h]
  _BYTE *v113; // [rsp+40h] [rbp-BC0h]
  _QWORD *v114; // [rsp+50h] [rbp-BB0h]
  _QWORD *v115; // [rsp+58h] [rbp-BA8h]
  __int64 v116; // [rsp+68h] [rbp-B98h]
  __int64 v117; // [rsp+68h] [rbp-B98h]
  __int64 v118; // [rsp+70h] [rbp-B90h]
  __int64 v119; // [rsp+70h] [rbp-B90h]
  __int64 v120; // [rsp+78h] [rbp-B88h]
  __int64 v121; // [rsp+78h] [rbp-B88h]
  const void *v122; // [rsp+80h] [rbp-B80h]
  __int64 v123; // [rsp+80h] [rbp-B80h]
  __int64 v124; // [rsp+80h] [rbp-B80h]
  int v125; // [rsp+8Ch] [rbp-B74h]
  __int64 *v126; // [rsp+90h] [rbp-B70h]
  __int64 v127; // [rsp+90h] [rbp-B70h]
  unsigned __int64 v128; // [rsp+98h] [rbp-B68h]
  __int64 v129; // [rsp+98h] [rbp-B68h]
  unsigned int v130; // [rsp+98h] [rbp-B68h]
  __int64 v131[4]; // [rsp+A0h] [rbp-B60h] BYREF
  __int64 v132; // [rsp+C0h] [rbp-B40h] BYREF
  _QWORD *v133; // [rsp+C8h] [rbp-B38h]
  __int64 v134; // [rsp+D0h] [rbp-B30h]
  unsigned int v135; // [rsp+D8h] [rbp-B28h]
  __int64 v136; // [rsp+E0h] [rbp-B20h] BYREF
  __int64 v137; // [rsp+E8h] [rbp-B18h]
  __int64 v138; // [rsp+F0h] [rbp-B10h]
  unsigned int v139; // [rsp+F8h] [rbp-B08h]
  _BYTE *v140; // [rsp+100h] [rbp-B00h] BYREF
  __int64 v141; // [rsp+108h] [rbp-AF8h]
  _BYTE v142[64]; // [rsp+110h] [rbp-AF0h] BYREF
  _QWORD v143[54]; // [rsp+150h] [rbp-AB0h] BYREF
  __int64 v144; // [rsp+300h] [rbp-900h] BYREF
  __int64 *v145; // [rsp+308h] [rbp-8F8h]
  int v146; // [rsp+310h] [rbp-8F0h]
  int v147; // [rsp+314h] [rbp-8ECh]
  int v148; // [rsp+318h] [rbp-8E8h]
  char v149; // [rsp+31Ch] [rbp-8E4h]
  __int64 v150; // [rsp+320h] [rbp-8E0h] BYREF
  unsigned __int64 *v151; // [rsp+360h] [rbp-8A0h]
  __int64 v152; // [rsp+368h] [rbp-898h]
  unsigned __int64 v153; // [rsp+370h] [rbp-890h] BYREF
  int v154; // [rsp+378h] [rbp-888h]
  unsigned __int64 v155; // [rsp+380h] [rbp-880h]
  int v156; // [rsp+388h] [rbp-878h]
  __int64 v157; // [rsp+390h] [rbp-870h]
  char v158[8]; // [rsp+4B0h] [rbp-750h] BYREF
  __int64 v159; // [rsp+4B8h] [rbp-748h]
  char v160; // [rsp+4CCh] [rbp-734h]
  char *v161; // [rsp+510h] [rbp-6F0h]
  char v162; // [rsp+520h] [rbp-6E0h] BYREF
  char v163[8]; // [rsp+660h] [rbp-5A0h] BYREF
  __int64 v164; // [rsp+668h] [rbp-598h]
  char v165; // [rsp+67Ch] [rbp-584h]
  char *v166; // [rsp+6C0h] [rbp-540h]
  char v167; // [rsp+6D0h] [rbp-530h] BYREF
  _QWORD *v168; // [rsp+810h] [rbp-3F0h] BYREF
  __int64 v169; // [rsp+818h] [rbp-3E8h]
  _BYTE v170[80]; // [rsp+820h] [rbp-3E0h] BYREF
  char *v171; // [rsp+870h] [rbp-390h]
  char v172; // [rsp+880h] [rbp-380h] BYREF
  __int64 *v173; // [rsp+9C0h] [rbp-240h] BYREF
  __int64 v174; // [rsp+9C8h] [rbp-238h]
  __int64 v175; // [rsp+9D0h] [rbp-230h] BYREF
  __int64 v176; // [rsp+9D8h] [rbp-228h]
  char *v177; // [rsp+A20h] [rbp-1E0h]
  char v178; // [rsp+A30h] [rbp-1D0h] BYREF

  v3 = *(_QWORD *)(a1 + 144);
  v132 = 0;
  v133 = 0;
  v134 = 0;
  v135 = 0;
  v136 = 0;
  v137 = 0;
  v138 = 0;
  v139 = 0;
  sub_E3D0C0(a1 + 48, v3);
  v131[2] = (__int64)&v136;
  v140 = v142;
  memset(v143, 0, sizeof(v143));
  v131[1] = a1;
  v143[12] = &v143[14];
  HIDWORD(v143[13]) = 8;
  v4 = *(_QWORD *)(v3 + 80);
  v143[1] = &v143[4];
  v152 = 0x800000000LL;
  if ( v4 )
    v4 -= 24;
  v151 = &v153;
  v145 = &v150;
  v5 = *(_QWORD *)(v4 + 48);
  v141 = 0x800000000LL;
  v131[0] = a2;
  v6 = v5 & 0xFFFFFFFFFFFFFFF8LL;
  LODWORD(v143[2]) = 8;
  BYTE4(v143[3]) = 1;
  v146 = 8;
  v148 = 0;
  v149 = 1;
  v147 = 1;
  v150 = v4;
  v144 = 1;
  if ( v6 == v4 + 48 )
    goto LABEL_162;
  if ( !v6 )
LABEL_157:
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v6 - 24) - 30 > 0xA )
  {
LABEL_162:
    v7 = 0;
    v9 = 0;
    v8 = 0;
  }
  else
  {
    v128 = v6 - 24;
    v7 = sub_B46E30(v6 - 24);
    v8 = v128;
    v9 = v128;
  }
  v157 = v4;
  v155 = v8;
  v153 = v9;
  v154 = v7;
  v156 = 0;
  LODWORD(v152) = 1;
  sub_E36520((__int64)&v144);
  sub_E36940((__int64)&v168, (__int64)v143, (__int64)&v140, v10, v11, v12);
  sub_E36820((__int64)&v173, (__int64)&v168);
  sub_E36940((__int64)v158, (__int64)&v144, v13, v14, (__int64)v158, v15);
  sub_E36820((__int64)v163, (__int64)v158);
  v16 = (__int64)&v173;
  sub_E36A10((__int64)v163, (__int64)&v173, (__int64)&v140, v17, v18, v19);
  if ( v166 != &v167 )
    _libc_free(v166, &v173);
  if ( !v165 )
    _libc_free(v164, &v173);
  if ( v161 != &v162 )
    _libc_free(v161, &v173);
  if ( !v160 )
    _libc_free(v159, &v173);
  if ( v177 != &v178 )
    _libc_free(v177, &v173);
  if ( !BYTE4(v176) )
    _libc_free(v174, &v173);
  if ( v171 != &v172 )
    _libc_free(v171, &v173);
  if ( !v170[12] )
    _libc_free(v169, &v173);
  if ( v151 != &v153 )
    _libc_free(v151, &v173);
  if ( !v149 )
    _libc_free(v145, &v173);
  if ( (_QWORD *)v143[12] != &v143[14] )
    _libc_free(v143[12], &v173);
  if ( !BYTE4(v143[3]) )
    _libc_free(v143[1], &v173);
  v168 = v170;
  v169 = 0x800000000LL;
  v113 = v140;
  v21 = &v140[8 * (unsigned int)v141];
  if ( v140 != v21 )
  {
    v111 = a2;
    v22 = (__int64 *)&v173;
    v115 = v21 - 8;
    while ( 1 )
    {
      v16 = (__int64)v133;
      v23 = *v115;
      v114 = v115;
      LODWORD(v169) = 0;
      if ( v135 )
      {
        v24 = (v135 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
        v25 = 5LL * v24;
        v26 = &v133[11 * v24];
        v27 = *v26;
        if ( v23 == *v26 )
        {
LABEL_35:
          if ( v26 != &v133[11 * v135] )
          {
            v16 = (__int64)(v26 + 1);
            sub_E34390((__int64)&v168, (char **)v26 + 1, 5LL * v135, v27, v25, v20);
            v28 = (_QWORD *)v26[1];
            if ( v28 != v26 + 3 )
              _libc_free(v28, v16);
            *v26 = -8192;
            LODWORD(v134) = v134 - 1;
            ++HIDWORD(v134);
          }
        }
        else
        {
          v106 = 1;
          while ( v27 != -4096 )
          {
            v20 = (unsigned int)(v106 + 1);
            v24 = (v135 - 1) & (v106 + v24);
            v25 = 5LL * v24;
            v26 = &v133[11 * v24];
            v27 = *v26;
            if ( v23 == *v26 )
              goto LABEL_35;
            v106 = v20;
          }
        }
      }
      v29 = v23 + 48;
      if ( v23 + 48 != *(_QWORD *)(v23 + 56) )
      {
        v126 = v22;
        v129 = v23;
        v30 = *(_QWORD *)(v23 + 56);
        do
        {
          v31 = *(_DWORD *)(a1 + 184);
          v32 = 0;
          if ( v30 )
            v32 = (_BYTE *)(v30 - 24);
          v16 = *(_QWORD *)(a1 + 168);
          if ( v31 )
          {
            v33 = v31 - 1;
            v34 = v33 & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
            v35 = (_QWORD *)(v16 + 16LL * v34);
            v36 = (_BYTE *)*v35;
            if ( v32 == (_BYTE *)*v35 )
            {
LABEL_45:
              v16 = v35[1];
              if ( v16 )
                sub_E35A10(v131, v16, (__int64)v32, (__int64)&v168);
            }
            else
            {
              v97 = 1;
              while ( v36 != (_BYTE *)-4096LL )
              {
                v98 = v97 + 1;
                v34 = v33 & (v97 + v34);
                v35 = (_QWORD *)(v16 + 16LL * v34);
                v36 = (_BYTE *)*v35;
                if ( v32 == (_BYTE *)*v35 )
                  goto LABEL_45;
                v97 = v98;
              }
            }
          }
          if ( (unsigned int)sub_E345B0(v32) != 3 )
          {
            v38 = (unsigned int)v169;
            v39 = (unsigned int)v169 + 1LL;
            if ( v39 > HIDWORD(v169) )
            {
              v16 = (__int64)v170;
              sub_C8D5F0((__int64)&v168, v170, v39, 8u, v37, v20);
              v38 = (unsigned int)v169;
            }
            v168[v38] = v32;
            LODWORD(v169) = v169 + 1;
          }
          v30 = *(_QWORD *)(v30 + 8);
        }
        while ( v29 != v30 );
        v23 = v129;
        v22 = v126;
      }
      v40 = *(_QWORD *)(v23 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v29 != v40 )
      {
        if ( !v40 )
          goto LABEL_157;
        v127 = v40 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v40 - 24) - 30 <= 0xA )
        {
          v125 = sub_B46E30(v40 - 24);
          if ( v125 )
            break;
        }
      }
LABEL_76:
      --v115;
      if ( v113 == (_BYTE *)v114 )
      {
        if ( v168 != (_QWORD *)v170 )
          _libc_free(v168, v16);
        v21 = v140;
        goto LABEL_80;
      }
    }
    v41 = v111;
    v130 = 0;
    while ( 1 )
    {
      v42 = sub_B46EC0(v127, v130);
      v43 = v42;
      if ( v42 )
      {
        v44 = (unsigned int)(*(_DWORD *)(v42 + 44) + 1);
        v45 = *(_DWORD *)(v42 + 44) + 1;
      }
      else
      {
        v44 = 0;
        v45 = 0;
      }
      v46 = 0;
      if ( v45 < *(_DWORD *)(v41 + 32) )
        v46 = *(_QWORD *)(*(_QWORD *)(v41 + 24) + 8 * v44);
      v16 = v135;
      if ( v135 )
      {
        v47 = 1;
        v48 = 0;
        v49 = (v135 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
        v50 = &v133[11 * v49];
        v51 = *v50;
        if ( v43 == *v50 )
        {
LABEL_64:
          v52 = (__int64 **)v50[1];
          v53 = *((unsigned int *)v50 + 4);
          v20 = (__int64)&v52[v53];
          if ( v52 == (__int64 **)v20 )
          {
            v99 = &v52[v53];
          }
          else
          {
            do
            {
              while ( 1 )
              {
                v54 = v168;
                v55 = *v52;
                v173 = *v52;
                v16 = (__int64)&v168[(unsigned int)v169];
                if ( (_QWORD *)v16 == sub_E342D0(v168, v16, v22) )
                  break;
                v52 = v57 + 1;
                v99 = v52;
                if ( v56 == v52 )
                  goto LABEL_73;
              }
              v58 = v56 - 1;
              if ( v58 == v57 )
                break;
              while ( 1 )
              {
                v59 = *v58;
                v173 = *v58;
                if ( (_QWORD *)v16 != sub_E342D0(v54, v16, v22) )
                  break;
                v58 = v60 - 1;
                if ( v58 == v61 )
                  goto LABEL_73;
              }
              *v61 = v59;
              v52 = v61 + 1;
              *v60 = v55;
              v99 = v52;
            }
            while ( v60 != v52 );
LABEL_73:
            v20 = v50[1];
          }
          *((_DWORD *)v50 + 4) = ((__int64)v99 - v20) >> 3;
          goto LABEL_75;
        }
        while ( v51 != -4096 )
        {
          if ( !v48 && v51 == -8192 )
            v48 = v50;
          v20 = (unsigned int)(v47 + 1);
          v49 = (v135 - 1) & (v47 + v49);
          v50 = &v133[11 * v49];
          v51 = *v50;
          if ( v43 == *v50 )
            goto LABEL_64;
          ++v47;
        }
        if ( !v48 )
          v48 = v50;
        ++v132;
        v68 = v134 + 1;
        if ( 4 * ((int)v134 + 1) < 3 * v135 )
        {
          if ( v135 - HIDWORD(v134) - v68 <= v135 >> 3 )
          {
            v124 = v46;
            sub_E362D0((__int64)&v132, v135);
            if ( !v135 )
            {
LABEL_173:
              LODWORD(v134) = v134 + 1;
              BUG();
            }
            v20 = v135 - 1;
            v16 = (__int64)v133;
            v102 = 0;
            v46 = v124;
            v103 = v20 & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
            v104 = 1;
            v48 = &v133[11 * v103];
            v105 = *v48;
            v68 = v134 + 1;
            if ( v43 != *v48 )
            {
              while ( v105 != -4096 )
              {
                if ( !v102 && v105 == -8192 )
                  v102 = v48;
                v103 = v20 & (v103 + v104);
                v48 = &v133[11 * v103];
                v105 = *v48;
                if ( v43 == *v48 )
                  goto LABEL_101;
                ++v104;
              }
              if ( v102 )
                v48 = v102;
            }
          }
          goto LABEL_101;
        }
      }
      else
      {
        ++v132;
      }
      v123 = v46;
      sub_E362D0((__int64)&v132, 2 * v135);
      if ( !v135 )
        goto LABEL_173;
      v46 = v123;
      v100 = (v135 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
      v48 = &v133[11 * v100];
      v16 = *v48;
      v68 = v134 + 1;
      if ( v43 != *v48 )
      {
        v20 = 1;
        v101 = 0;
        while ( v16 != -4096 )
        {
          if ( v16 == -8192 && !v101 )
            v101 = v48;
          v107 = v20 + 1;
          v100 = (v135 - 1) & (v100 + v20);
          v20 = 5LL * v100;
          v48 = &v133[11 * v100];
          v16 = *v48;
          if ( v43 == *v48 )
            goto LABEL_101;
          v20 = v107;
        }
        if ( v101 )
          v48 = v101;
      }
LABEL_101:
      LODWORD(v134) = v68;
      if ( *v48 != -4096 )
        --HIDWORD(v134);
      *v48 = v43;
      v48[1] = (__int64)(v48 + 3);
      v122 = v48 + 3;
      v48[2] = 0x800000000LL;
      v69 = v168;
      if ( &v168[(unsigned int)v169] == v168 )
        goto LABEL_75;
      v70 = *v168;
      v20 = (__int64)&v168[(unsigned int)v169];
      v71 = *(_QWORD *)(*v168 + 40LL);
      if ( v71 )
      {
        while ( 1 )
        {
          v72 = (unsigned int)(*(_DWORD *)(v71 + 44) + 1);
          if ( (unsigned int)(*(_DWORD *)(v71 + 44) + 1) < *(_DWORD *)(v41 + 32) )
            break;
LABEL_122:
          if ( v46 )
            goto LABEL_75;
LABEL_118:
          v77 = *((unsigned int *)v48 + 4);
          v78 = v77 + 1;
          if ( v77 + 1 > (unsigned __int64)*((unsigned int *)v48 + 5) )
            goto LABEL_129;
LABEL_119:
          ++v69;
          *(_QWORD *)(v48[1] + 8 * v77) = v70;
          ++*((_DWORD *)v48 + 4);
          if ( (_QWORD *)v20 == v69 )
            goto LABEL_75;
          v70 = *v69;
          v71 = *(_QWORD *)(*v69 + 40LL);
          if ( !v71 )
            goto LABEL_121;
        }
      }
      else
      {
LABEL_121:
        v72 = 0;
        if ( !*(_DWORD *)(v41 + 32) )
          goto LABEL_122;
      }
      v73 = *(_QWORD *)(*(_QWORD *)(v41 + 24) + 8 * v72);
      if ( v46 == v73 || !v46 )
        goto LABEL_118;
      if ( !v73 )
        goto LABEL_75;
      if ( v73 == *(_QWORD *)(v46 + 8) )
        goto LABEL_118;
      if ( v46 != *(_QWORD *)(v73 + 8) && *(_DWORD *)(v73 + 16) < *(_DWORD *)(v46 + 16) )
      {
        if ( !*(_BYTE *)(v41 + 112) )
        {
          v74 = *(_DWORD *)(v41 + 116) + 1;
          *(_DWORD *)(v41 + 116) = v74;
          if ( v74 <= 0x20 )
          {
            v16 = *(unsigned int *)(v73 + 16);
            v75 = v46;
            do
            {
              v76 = v75;
              v75 = *(_QWORD *)(v75 + 8);
            }
            while ( v75 && (unsigned int)v16 <= *(_DWORD *)(v75 + 16) );
            if ( v73 != v76 )
              goto LABEL_75;
            goto LABEL_118;
          }
          v80 = *(_QWORD *)(v41 + 96);
          HIDWORD(v174) = 32;
          v173 = &v175;
          if ( v80 )
          {
            v81 = *(_QWORD *)(v80 + 24);
            v175 = v80;
            v82 = (__int64)v69;
            LODWORD(v174) = 1;
            v176 = v81;
            *(_DWORD *)(v80 + 72) = 0;
            v83 = 1;
            v119 = v46;
            v84 = &v175;
            v85 = (__int64)v22;
            v86 = v48;
            v117 = v20;
            v87 = 1;
            v88 = v41;
            v89 = v73;
            v121 = v70;
            do
            {
              v94 = v87++;
              v95 = &v84[2 * v83 - 2];
              v16 = *v95;
              v96 = (__int64 *)v95[1];
              if ( v96 == (__int64 *)(*(_QWORD *)(*v95 + 24) + 8LL * *(unsigned int *)(*v95 + 32)) )
              {
                --v83;
                *(_DWORD *)(v16 + 76) = v94;
                LODWORD(v174) = v83;
              }
              else
              {
                v90 = *v96;
                v95[1] = (__int64)(v96 + 1);
                v91 = (unsigned int)v174;
                v92 = *(_QWORD *)(v90 + 24);
                if ( (unsigned __int64)(unsigned int)v174 + 1 > HIDWORD(v174) )
                {
                  v16 = (__int64)&v175;
                  v108 = v88;
                  v109 = v82;
                  v110 = *(_QWORD *)(v90 + 24);
                  sub_C8D5F0(v85, &v175, (unsigned int)v174 + 1LL, 0x10u, v82, v88);
                  v84 = v173;
                  v91 = (unsigned int)v174;
                  v88 = v108;
                  v82 = v109;
                  v92 = v110;
                }
                v93 = &v84[2 * v91];
                *v93 = v90;
                v93[1] = v92;
                v83 = v174 + 1;
                LODWORD(v174) = v174 + 1;
                *(_DWORD *)(v90 + 72) = v94;
                v84 = v173;
              }
            }
            while ( v83 );
            v48 = v86;
            v22 = (__int64 *)v85;
            v73 = v89;
            v79 = v84;
            v41 = v88;
            v70 = v121;
            v69 = (_QWORD *)v82;
            *(_DWORD *)(v88 + 116) = 0;
            v46 = v119;
            *(_BYTE *)(v88 + 112) = 1;
            v20 = v117;
            if ( v79 != &v175 )
            {
              _libc_free(v79, v16);
              v70 = v121;
              v46 = v119;
              v20 = v117;
            }
          }
        }
        if ( *(_DWORD *)(v46 + 72) >= *(_DWORD *)(v73 + 72) && *(_DWORD *)(v46 + 76) <= *(_DWORD *)(v73 + 76) )
        {
          v77 = *((unsigned int *)v48 + 4);
          v78 = v77 + 1;
          if ( v77 + 1 <= (unsigned __int64)*((unsigned int *)v48 + 5) )
            goto LABEL_119;
LABEL_129:
          v16 = (__int64)v122;
          v116 = v20;
          v118 = v46;
          v120 = v70;
          sub_C8D5F0((__int64)(v48 + 1), v122, v78, 8u, v78, v20);
          v77 = *((unsigned int *)v48 + 4);
          v20 = v116;
          v46 = v118;
          v70 = v120;
          goto LABEL_119;
        }
      }
LABEL_75:
      if ( v125 == ++v130 )
        goto LABEL_76;
    }
  }
LABEL_80:
  if ( v21 != v142 )
    _libc_free(v21, v16);
  v62 = 16LL * v139;
  sub_C7D6A0(v137, v62, 8);
  v63 = v135;
  if ( v135 )
  {
    v64 = v133;
    v65 = &v133[11 * v135];
    do
    {
      if ( *v64 != -8192 && *v64 != -4096 )
      {
        v66 = (_QWORD *)v64[1];
        if ( v66 != v64 + 3 )
          _libc_free(v66, v62);
      }
      v64 += 11;
    }
    while ( v65 != v64 );
    v63 = v135;
  }
  return sub_C7D6A0((__int64)v133, 88 * v63, 8);
}
