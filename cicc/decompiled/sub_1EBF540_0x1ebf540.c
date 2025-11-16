// Function: sub_1EBF540
// Address: 0x1ebf540
//
__int64 __fastcall sub_1EBF540(__int64 a1, __int64 a2, unsigned int a3, char a4, __int64 a5)
{
  __int64 v5; // r15
  __int64 v6; // r14
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 (*v15)(); // rcx
  __int64 v16; // rdx
  unsigned int v17; // edx
  int v18; // r8d
  __int64 v19; // r9
  unsigned __int64 v20; // r15
  int v21; // eax
  __int64 v22; // rbx
  __int64 v23; // r12
  _BYTE *v24; // r14
  int v25; // eax
  __int64 v26; // rbx
  __int64 v27; // rax
  __int64 *v28; // rcx
  __int64 v29; // rax
  __int64 v30; // rdi
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // rdi
  unsigned int v34; // r15d
  __int64 v35; // rax
  __int64 *v36; // rdi
  __int64 v37; // rax
  __int64 v38; // r10
  __int64 v39; // r8
  __int64 v40; // r9
  int v41; // eax
  char *v42; // r14
  __int64 v43; // rax
  __int64 v44; // rax
  unsigned int *v45; // rbx
  int v46; // eax
  __int64 v47; // r15
  __int64 *v48; // rsi
  __int64 v49; // rax
  __int64 v50; // rdi
  __int64 v51; // r8
  __int64 v52; // rcx
  __int64 v53; // rdx
  __int64 v54; // r9
  int v55; // eax
  unsigned int v56; // r12d
  __int64 v57; // rax
  char *v58; // rdx
  __int64 v59; // r8
  __int64 v60; // rsi
  __int64 v61; // rdi
  __int64 v62; // rax
  __int64 *v63; // rcx
  __int64 v64; // rax
  __int64 v65; // rdi
  __int64 v66; // rax
  __int64 v67; // rdi
  signed __int64 v68; // r9
  int v69; // r8d
  int v70; // r9d
  unsigned __int64 v71; // r12
  unsigned __int64 v72; // rax
  unsigned int v73; // eax
  _QWORD *v74; // rdx
  unsigned int v75; // ecx
  __int64 v76; // r14
  __int64 v77; // rbx
  __int64 v78; // rdx
  _DWORD *v79; // rax
  unsigned int v80; // ecx
  __int64 v81; // r12
  int v82; // r11d
  unsigned __int64 v83; // rcx
  unsigned int v84; // edx
  __int64 v85; // r15
  __int64 v86; // rax
  unsigned int v87; // ebx
  __int64 v88; // rsi
  __int64 v89; // rdx
  __int64 v90; // rcx
  unsigned __int64 v91; // rsi
  unsigned __int64 v92; // r10
  int v93; // r12d
  _QWORD *v94; // rcx
  _QWORD *v95; // rax
  _QWORD *v97; // rdi
  _QWORD *v98; // rdx
  __int64 v99; // rcx
  unsigned __int64 v100; // rsi
  unsigned __int64 v101; // rax
  int v102; // r12d
  __int64 v103; // rdx
  __int64 v104; // rdx
  _QWORD *v105; // rcx
  _QWORD *i; // rax
  unsigned __int64 v107; // r12
  __int64 v108; // r15
  int v109; // r8d
  __int64 v110; // rax
  __int64 v111; // rdi
  __int64 v112; // rbx
  int v113; // r8d
  __int64 v114; // rax
  __int64 v115; // rdi
  __int64 v116; // rax
  _QWORD *v117; // rcx
  _QWORD *v118; // rax
  __int64 v119; // rax
  int v120; // [rsp+Ch] [rbp-1C4h]
  __int64 v121; // [rsp+18h] [rbp-1B8h]
  __int64 v122; // [rsp+18h] [rbp-1B8h]
  __int64 v123; // [rsp+18h] [rbp-1B8h]
  unsigned int v124; // [rsp+20h] [rbp-1B0h]
  unsigned int v125; // [rsp+24h] [rbp-1ACh]
  unsigned int v126; // [rsp+24h] [rbp-1ACh]
  unsigned int v127; // [rsp+24h] [rbp-1ACh]
  int v128; // [rsp+24h] [rbp-1ACh]
  unsigned int v129; // [rsp+28h] [rbp-1A8h]
  unsigned int v130; // [rsp+28h] [rbp-1A8h]
  __int64 v131; // [rsp+28h] [rbp-1A8h]
  __int64 v132; // [rsp+28h] [rbp-1A8h]
  unsigned __int64 v133; // [rsp+28h] [rbp-1A8h]
  unsigned __int64 v134; // [rsp+28h] [rbp-1A8h]
  __int64 v135; // [rsp+30h] [rbp-1A0h]
  __int64 v136; // [rsp+30h] [rbp-1A0h]
  unsigned int v137; // [rsp+30h] [rbp-1A0h]
  const void *v138; // [rsp+30h] [rbp-1A0h]
  __int64 v139; // [rsp+38h] [rbp-198h]
  __int64 v140; // [rsp+38h] [rbp-198h]
  unsigned __int64 v141; // [rsp+40h] [rbp-190h]
  char *v142; // [rsp+40h] [rbp-190h]
  __int64 v143; // [rsp+48h] [rbp-188h]
  unsigned int v144; // [rsp+48h] [rbp-188h]
  unsigned __int8 v145; // [rsp+58h] [rbp-178h]
  int v146; // [rsp+58h] [rbp-178h]
  __int64 v147; // [rsp+58h] [rbp-178h]
  __int64 v148; // [rsp+58h] [rbp-178h]
  _BYTE *v149; // [rsp+60h] [rbp-170h] BYREF
  __int64 v150; // [rsp+68h] [rbp-168h]
  _BYTE v151[32]; // [rsp+70h] [rbp-160h] BYREF
  unsigned __int64 v152[2]; // [rsp+90h] [rbp-140h] BYREF
  _BYTE v153[32]; // [rsp+A0h] [rbp-130h] BYREF
  _QWORD v154[2]; // [rsp+C0h] [rbp-110h] BYREF
  __int64 v155; // [rsp+D0h] [rbp-100h]
  __int64 v156; // [rsp+D8h] [rbp-F8h]
  __int64 v157; // [rsp+E0h] [rbp-F0h]
  __int64 v158; // [rsp+E8h] [rbp-E8h]
  __int64 v159; // [rsp+F0h] [rbp-E0h]
  __int64 v160; // [rsp+F8h] [rbp-D8h]
  unsigned int v161; // [rsp+100h] [rbp-D0h]
  char v162; // [rsp+104h] [rbp-CCh]
  __int64 v163; // [rsp+108h] [rbp-C8h]
  __int64 v164; // [rsp+110h] [rbp-C0h]
  _BYTE *v165; // [rsp+118h] [rbp-B8h]
  _BYTE *v166; // [rsp+120h] [rbp-B0h]
  __int64 v167; // [rsp+128h] [rbp-A8h]
  int v168; // [rsp+130h] [rbp-A0h]
  _BYTE v169[32]; // [rsp+138h] [rbp-98h] BYREF
  __int64 v170; // [rsp+158h] [rbp-78h]
  _BYTE *v171; // [rsp+160h] [rbp-70h]
  _BYTE *v172; // [rsp+168h] [rbp-68h]
  __int64 v173; // [rsp+170h] [rbp-60h]
  int v174; // [rsp+178h] [rbp-58h]
  _BYTE v175[80]; // [rsp+180h] [rbp-50h] BYREF

  v5 = a1 + 376;
  v6 = a1 + 672;
  v149 = v151;
  v10 = *(_QWORD *)(a1 + 680);
  v11 = *(_QWORD *)(a1 + 256);
  v150 = 0x800000000LL;
  v12 = *(_QWORD *)(a1 + 264);
  v155 = a5;
  v154[1] = a2;
  v154[0] = &unk_4A00C10;
  v13 = *(_QWORD *)(v10 + 40);
  v157 = v12;
  v158 = v11;
  v156 = v13;
  v14 = *(_QWORD *)(v10 + 16);
  v15 = *(__int64 (**)())(*(_QWORD *)v14 + 40LL);
  v16 = 0;
  if ( v15 != sub_1D00B00 )
  {
    v148 = a5;
    v116 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v15)(v14, a2, 0);
    a5 = v148;
    v16 = v116;
    v13 = v156;
  }
  v159 = v16;
  v17 = *(_DWORD *)(a5 + 8);
  v160 = v6;
  v161 = v17;
  v165 = v169;
  v166 = v169;
  v163 = v5;
  v171 = v175;
  v172 = v175;
  v162 = 0;
  v164 = 0;
  v167 = 4;
  v168 = 0;
  v170 = 0;
  v173 = 4;
  v174 = 0;
  *(_QWORD *)(v13 + 8) = v154;
  sub_1F15B50(*(_QWORD *)(a1 + 992), v154, (unsigned int)dword_4FC99C0);
  v20 = *(unsigned int *)(*(_QWORD *)(a1 + 840) + 288LL);
  *(_DWORD *)(a1 + 27264) = 0;
  v21 = v20;
  if ( *(_DWORD *)(a1 + 27268) < (unsigned int)v20 )
  {
    sub_16CD150(a1 + 27256, (const void *)(a1 + 27272), v20, 4, v18, v19);
    v21 = v20;
  }
  *(_DWORD *)(a1 + 27264) = v21;
  if ( 4 * v20 )
    memset(*(void **)(a1 + 27256), 255, 4 * v20);
  if ( a3 == -1
    || (v108 = *(_QWORD *)(a1 + 24168) + 96LL * a3, !(unsigned int)sub_1EBBD70(v108, (_QWORD *)(a1 + 27256), a3)) )
  {
    if ( !a4 )
      goto LABEL_9;
    goto LABEL_121;
  }
  v110 = (unsigned int)v150;
  if ( (unsigned int)v150 >= HIDWORD(v150) )
  {
    sub_16CD150((__int64)&v149, v151, 0, 4, v109, v19);
    v110 = (unsigned int)v150;
  }
  *(_DWORD *)&v149[4 * v110] = a3;
  v111 = *(_QWORD *)(a1 + 992);
  LODWORD(v150) = v150 + 1;
  *(_DWORD *)(v108 + 4) = sub_1F15650(v111);
  if ( a4 )
  {
LABEL_121:
    v112 = *(_QWORD *)(a1 + 24168);
    if ( (unsigned int)sub_1EBBD70(v112, (_QWORD *)(a1 + 27256), 0) )
    {
      v114 = (unsigned int)v150;
      if ( (unsigned int)v150 >= HIDWORD(v150) )
      {
        sub_16CD150((__int64)&v149, v151, 0, 4, v113, v19);
        v114 = (unsigned int)v150;
      }
      *(_DWORD *)&v149[4 * v114] = 0;
      v115 = *(_QWORD *)(a1 + 992);
      LODWORD(v150) = v150 + 1;
      *(_DWORD *)(v112 + 4) = sub_1F15650(v115);
    }
  }
LABEL_9:
  v22 = *(_QWORD *)(a1 + 984);
  v141 = (unsigned __int64)v149;
  v143 = (unsigned int)v150;
  v124 = *(_DWORD *)(v155 + 8) - v161;
  v120 = *(_DWORD *)(*(_QWORD *)(v22 + 40) + 112LL);
  v23 = *(_QWORD *)(a1 + 280)
      + 24LL
      * *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 248) + 24LL)
                                                    + 16LL * (v120 & 0x7FFFFFFF))
                                        & 0xFFFFFFFFFFFFFFF8LL)
                            + 24LL);
  if ( *(_DWORD *)(a1 + 288) != *(_DWORD *)v23 )
  {
    sub_1ED7890(a1 + 280);
    v22 = *(_QWORD *)(a1 + 984);
  }
  v24 = *(_BYTE **)(v22 + 280);
  v145 = *(_BYTE *)(v23 + 8);
  v25 = *(_DWORD *)(v22 + 288);
  if ( v25 )
  {
    v26 = (__int64)&v24[40 * (v25 - 1) + 40];
    while ( 1 )
    {
      v32 = 0;
      v34 = *(_DWORD *)(*(_QWORD *)v24 + 48LL);
      if ( !v24[32]
        || (v27 = *(unsigned int *)(*(_QWORD *)(a1 + 27256)
                                  + 4LL * *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(a1 + 840) + 240LL) + 8LL * v34)),
            (_DWORD)v27 == -1) )
      {
        v31 = 0;
        if ( !v24[33] )
          goto LABEL_54;
      }
      else
      {
        v28 = &qword_4FCF930;
        v29 = *(_QWORD *)(a1 + 24168) + 96 * v27;
        v30 = *(_QWORD *)(v29 + 8);
        v31 = *(unsigned int *)(v29 + 4);
        if ( v30 )
        {
          v28 = (__int64 *)(*(_QWORD *)(v30 + 512) + 24LL * v34);
          if ( *(_DWORD *)v28 != *(_DWORD *)(v30 + 4) )
          {
            v129 = *(_DWORD *)(v29 + 4);
            v135 = v29;
            sub_20F85B0(v30, v34, v31, v28, 24LL * v34, v19);
            v31 = v129;
            v29 = v135;
            v28 = (__int64 *)(*(_QWORD *)(v30 + 512) + 24LL * v34);
          }
        }
        *(_QWORD *)(v29 + 16) = v28;
        v32 = v28[1];
        if ( !v24[33] )
        {
LABEL_18:
          if ( !(_DWORD)v31 )
            goto LABEL_54;
          v33 = *(_QWORD *)(a1 + 992);
          goto LABEL_20;
        }
      }
      v35 = *(unsigned int *)(*(_QWORD *)(a1 + 27256)
                            + 4LL * *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(a1 + 840) + 240LL) + 4LL * (2 * v34 + 1)));
      if ( (_DWORD)v35 == -1 )
        goto LABEL_18;
      v36 = &qword_4FCF930;
      v37 = *(_QWORD *)(a1 + 24168) + 96 * v35;
      v38 = *(_QWORD *)(v37 + 8);
      v39 = *(unsigned int *)(v37 + 4);
      if ( v38 )
      {
        v36 = (__int64 *)(*(_QWORD *)(v38 + 512) + 24LL * v34);
        if ( *(_DWORD *)v36 != *(_DWORD *)(v38 + 4) )
        {
          v121 = v32;
          v125 = v31;
          v130 = *(_DWORD *)(v37 + 4);
          v136 = v37;
          v139 = *(_QWORD *)(v37 + 8);
          sub_20F85B0(v38, v34, v31, v32, v39, 24LL * v34);
          v32 = v121;
          v31 = v125;
          v39 = v130;
          v37 = v136;
          v36 = (__int64 *)(*(_QWORD *)(v139 + 512) + 24LL * v34);
        }
      }
      *(_QWORD *)(v37 + 16) = v36;
      v40 = v36[2];
      if ( !((unsigned int)v31 | (unsigned int)v39) )
      {
LABEL_54:
        if ( (unsigned __int8)sub_1F15AD0(*(_QWORD *)(a1 + 984), v24, v145) )
          sub_1F203D0(*(_QWORD *)(a1 + 992), v24);
        goto LABEL_21;
      }
      v33 = *(_QWORD *)(a1 + 992);
      if ( !(_DWORD)v31 )
      {
        sub_1F1FDE0(v33, v24, (unsigned int)v39, v40);
        goto LABEL_21;
      }
      if ( (_DWORD)v39 )
      {
        v24 += 40;
        sub_1F200F0(v33, v34, v31, v32, v39, v40);
        if ( (_BYTE *)v26 == v24 )
        {
LABEL_32:
          v22 = *(_QWORD *)(a1 + 984);
          break;
        }
      }
      else
      {
LABEL_20:
        sub_1F20540(v33, v24, v31, v32);
LABEL_21:
        v24 += 40;
        if ( (_BYTE *)v26 == v24 )
          goto LABEL_32;
      }
    }
  }
  v41 = *(_DWORD *)(v22 + 640);
  v42 = 0;
  if ( v41 )
  {
    v107 = 8LL * ((unsigned int)(v41 + 63) >> 6);
    v42 = (char *)malloc(v107);
    if ( !v42 )
    {
      if ( v107 || (v119 = malloc(1u)) == 0 )
        sub_16BD1C0("Allocation failed", 1u);
      else
        v42 = (char *)v119;
    }
    memcpy(v42, *(const void **)(v22 + 624), v107);
  }
  v146 = 0;
  v43 = 0;
  if ( v143 )
  {
    do
    {
      v44 = *(_QWORD *)(a1 + 24168) + 96LL * *(unsigned int *)(v141 + 4 * v43);
      v45 = *(unsigned int **)(v44 + 48);
      v46 = *(_DWORD *)(v44 + 56);
      if ( v46 )
      {
        v47 = (__int64)&v45[v46 - 1 + 1];
        do
        {
          v56 = *v45;
          v57 = 1LL << *v45;
          v58 = &v42[8 * (*v45 >> 6)];
          if ( (*(_QWORD *)v58 & v57) != 0 )
          {
            v59 = 2 * v56;
            v60 = *(_QWORD *)(a1 + 27256);
            v54 = 0;
            v52 = 0;
            *(_QWORD *)v58 &= ~v57;
            v53 = 0;
            v61 = *(_QWORD *)(*(_QWORD *)(a1 + 840) + 240LL);
            v62 = *(unsigned int *)(v60 + 4LL * *(unsigned int *)(v61 + 4 * v59));
            if ( (_DWORD)v62 != -1 )
            {
              v63 = &qword_4FCF930;
              v64 = *(_QWORD *)(a1 + 24168) + 96 * v62;
              v65 = *(_QWORD *)(v64 + 8);
              v53 = *(unsigned int *)(v64 + 4);
              if ( v65 )
              {
                v63 = (__int64 *)(*(_QWORD *)(v65 + 512) + 24LL * v56);
                if ( *(_DWORD *)v63 != *(_DWORD *)(v65 + 4) )
                {
                  v127 = *(_DWORD *)(v64 + 4);
                  v132 = v64;
                  sub_20F85B0(v65, v56, v53, v63, v59, 0);
                  v54 = 0;
                  v53 = v127;
                  v64 = v132;
                  LODWORD(v59) = 2 * v56;
                  v63 = (__int64 *)(*(_QWORD *)(v65 + 512) + 24LL * v56);
                }
              }
              *(_QWORD *)(v64 + 16) = v63;
              v52 = v63[1];
              v60 = *(_QWORD *)(a1 + 27256);
              v61 = *(_QWORD *)(*(_QWORD *)(a1 + 840) + 240LL);
            }
            v66 = *(unsigned int *)(v60 + 4LL * *(unsigned int *)(v61 + 4LL * (unsigned int)(v59 + 1)));
            if ( (_DWORD)v66 == -1 )
            {
              v55 = v53;
              v51 = 0;
            }
            else
            {
              v48 = &qword_4FCF930;
              v49 = *(_QWORD *)(a1 + 24168) + 96 * v66;
              v50 = *(_QWORD *)(v49 + 8);
              v51 = *(unsigned int *)(v49 + 4);
              if ( v50 )
              {
                v48 = (__int64 *)(*(_QWORD *)(v50 + 512) + 24LL * v56);
                if ( *(_DWORD *)v48 != *(_DWORD *)(v50 + 4) )
                {
                  v122 = v52;
                  v126 = *(_DWORD *)(v49 + 4);
                  v131 = v49;
                  v137 = v53;
                  sub_20F85B0(v50, v56, v53, v52, v51, 24LL * v56);
                  v52 = v122;
                  v51 = v126;
                  v49 = v131;
                  v53 = v137;
                  v48 = (__int64 *)(*(_QWORD *)(v50 + 512) + 24LL * v56);
                }
              }
              *(_QWORD *)(v49 + 16) = v48;
              v54 = v48[2];
              v55 = v53 | v51;
            }
            if ( v55 )
              sub_1F200F0(*(_QWORD *)(a1 + 992), v56, v53, v52, v51, v54);
          }
          ++v45;
        }
        while ( v45 != (unsigned int *)v47 );
      }
      v43 = (unsigned int)++v146;
    }
    while ( v143 != v146 );
  }
  v67 = *(_QWORD *)(a1 + 992);
  v152[0] = (unsigned __int64)v153;
  v152[1] = 0x800000000LL;
  sub_1F1E080(v67, v152);
  sub_1DADA60(
    *(_QWORD *)(a1 + 856),
    v120,
    *(_QWORD *)v155 + 4LL * v161,
    *(unsigned int *)(v155 + 8) - (unsigned __int64)v161,
    *(_QWORD *)(a1 + 264),
    v68);
  v71 = *(unsigned int *)(*(_QWORD *)(a1 + 248) + 32LL);
  v140 = a1 + 920;
  v72 = *(unsigned int *)(a1 + 928);
  if ( v71 < v72 )
    goto LABEL_114;
  if ( v71 > v72 )
  {
    if ( v71 > *(unsigned int *)(a1 + 932) )
    {
      sub_16CD150(v140, (const void *)(a1 + 936), v71, 8, v69, v70);
      v72 = *(unsigned int *)(a1 + 928);
    }
    v104 = *(_QWORD *)(a1 + 920);
    v105 = (_QWORD *)(v104 + 8 * v71);
    for ( i = (_QWORD *)(v104 + 8 * v72); v105 != i; ++i )
    {
      if ( i )
        *i = *(_QWORD *)(a1 + 936);
    }
LABEL_114:
    *(_DWORD *)(a1 + 928) = v71;
  }
  v73 = v161;
  v144 = *(_DWORD *)(*(_QWORD *)(a1 + 984) + 288LL)
       + *(_DWORD *)(*(_QWORD *)(a1 + 984) + 648LL)
       - *(_DWORD *)(*(_QWORD *)(a1 + 984) + 616LL);
  v74 = (_QWORD *)v155;
  v75 = *(_DWORD *)(v155 + 8) - v161;
  if ( !v75 )
    goto LABEL_85;
  v142 = v42;
  v147 = v75;
  v76 = 0;
  v138 = (const void *)(a1 + 936);
  while ( 1 )
  {
    v81 = *(_QWORD *)(a1 + 264);
    v82 = *(_DWORD *)(*v74 + 4LL * (v73 + (unsigned int)v76));
    v83 = *(unsigned int *)(v81 + 408);
    v84 = v82 & 0x7FFFFFFF;
    v85 = v82 & 0x7FFFFFFF;
    v86 = 8 * v85;
    if ( (v82 & 0x7FFFFFFFu) >= (unsigned int)v83 || (v77 = *(_QWORD *)(*(_QWORD *)(v81 + 400) + 8LL * v84)) == 0 )
    {
      v87 = v84 + 1;
      if ( (unsigned int)v83 < v84 + 1 )
      {
        v89 = v87;
        if ( v87 >= v83 )
        {
          if ( v87 > v83 )
          {
            if ( v87 > (unsigned __int64)*(unsigned int *)(v81 + 412) )
            {
              v123 = 8LL * (v82 & 0x7FFFFFFF);
              v128 = v82;
              sub_16CD150(v81 + 400, (const void *)(v81 + 416), v87, 8, v69, v70);
              v83 = *(unsigned int *)(v81 + 408);
              v86 = v123;
              v82 = v128;
              v89 = v87;
            }
            v88 = *(_QWORD *)(v81 + 400);
            v97 = (_QWORD *)(v88 + 8 * v89);
            v98 = (_QWORD *)(v88 + 8 * v83);
            v99 = *(_QWORD *)(v81 + 416);
            if ( v97 != v98 )
            {
              do
                *v98++ = v99;
              while ( v97 != v98 );
              v88 = *(_QWORD *)(v81 + 400);
            }
            *(_DWORD *)(v81 + 408) = v87;
            goto LABEL_70;
          }
        }
        else
        {
          *(_DWORD *)(v81 + 408) = v87;
        }
      }
      v88 = *(_QWORD *)(v81 + 400);
LABEL_70:
      *(_QWORD *)(v88 + v86) = sub_1DBA290(v82);
      v77 = *(_QWORD *)(*(_QWORD *)(v81 + 400) + 8 * v85);
      sub_1DBB110((_QWORD *)v81, v77);
    }
    v78 = *(_QWORD *)(a1 + 920);
    v79 = (_DWORD *)(v78 + 8LL * (*(_DWORD *)(v77 + 112) & 0x7FFFFFFF));
    if ( *v79 )
      goto LABEL_65;
    v80 = *(_DWORD *)(v152[0] + 4 * v76);
    if ( !v80 )
      break;
    if ( v124 > v80 && v144 <= (unsigned int)sub_1F14EF0(*(_QWORD *)(a1 + 984), v77) )
    {
      v100 = *(unsigned int *)(a1 + 928);
      v101 = *(unsigned int *)(*(_QWORD *)(a1 + 248) + 32LL);
      v102 = *(_DWORD *)(*(_QWORD *)(a1 + 248) + 32LL);
      if ( v101 < v100 )
      {
        *(_DWORD *)(a1 + 928) = v101;
        v103 = *(_QWORD *)(a1 + 920);
      }
      else if ( v101 > v100 )
      {
        if ( v101 > *(unsigned int *)(a1 + 932) )
        {
          v134 = *(unsigned int *)(*(_QWORD *)(a1 + 248) + 32LL);
          sub_16CD150(v140, v138, v134, 8, v69, v70);
          v100 = *(unsigned int *)(a1 + 928);
          v101 = v134;
        }
        v103 = *(_QWORD *)(a1 + 920);
        v117 = (_QWORD *)(v103 + 8 * v101);
        v118 = (_QWORD *)(v103 + 8 * v100);
        if ( v117 != v118 )
        {
          do
          {
            if ( v118 )
              *v118 = *(_QWORD *)(a1 + 936);
            ++v118;
          }
          while ( v117 != v118 );
          v103 = *(_QWORD *)(a1 + 920);
        }
        *(_DWORD *)(a1 + 928) = v102;
      }
      else
      {
        v103 = *(_QWORD *)(a1 + 920);
      }
      *(_DWORD *)(v103 + 8LL * (*(_DWORD *)(v77 + 112) & 0x7FFFFFFF)) = 3;
    }
LABEL_65:
    if ( v147 == ++v76 )
      goto LABEL_84;
LABEL_66:
    v73 = v161;
    v74 = (_QWORD *)v155;
  }
  v90 = *(_QWORD *)(a1 + 248);
  v91 = *(unsigned int *)(a1 + 928);
  v92 = *(unsigned int *)(v90 + 32);
  v93 = *(_DWORD *)(v90 + 32);
  if ( v92 < v91 )
    goto LABEL_82;
  if ( v92 > v91 )
  {
    if ( v92 > *(unsigned int *)(a1 + 932) )
    {
      v133 = *(unsigned int *)(v90 + 32);
      sub_16CD150(v140, v138, v133, 8, v69, v70);
      v78 = *(_QWORD *)(a1 + 920);
      v91 = *(unsigned int *)(a1 + 928);
      v92 = v133;
    }
    v94 = (_QWORD *)(v78 + 8 * v92);
    v95 = (_QWORD *)(v78 + 8 * v91);
    if ( v94 != v95 )
    {
      do
      {
        if ( v95 )
          *v95 = *(_QWORD *)(a1 + 936);
        ++v95;
      }
      while ( v94 != v95 );
      v78 = *(_QWORD *)(a1 + 920);
    }
LABEL_82:
    *(_DWORD *)(a1 + 928) = v93;
    v79 = (_DWORD *)(v78 + 8LL * (*(_DWORD *)(v77 + 112) & 0x7FFFFFFF));
  }
  *v79 = 4;
  if ( v147 != ++v76 )
    goto LABEL_66;
LABEL_84:
  v42 = v142;
LABEL_85:
  if ( byte_4FCF965[0] )
    sub_1E926D0(*(_QWORD *)(a1 + 680), a1, (__int64)"After splitting live range around region", 1);
  if ( (_BYTE *)v152[0] != v153 )
    _libc_free(v152[0]);
  _libc_free((unsigned __int64)v42);
  *(_QWORD *)(v156 + 8) = 0;
  if ( v172 != v171 )
    _libc_free((unsigned __int64)v172);
  if ( v166 != v165 )
    _libc_free((unsigned __int64)v166);
  if ( v149 != v151 )
    _libc_free((unsigned __int64)v149);
  return 0;
}
