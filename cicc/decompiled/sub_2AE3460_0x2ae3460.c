// Function: sub_2AE3460
// Address: 0x2ae3460
//
__int64 __fastcall sub_2AE3460(
        __int64 a1,
        __int64 **a2,
        __int64 a3,
        unsigned int a4,
        _QWORD *a5,
        __int64 a6,
        __int64 a7,
        char a8,
        char *a9)
{
  _QWORD *v9; // r14
  char *v12; // r15
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 *v15; // r15
  __int64 v16; // rax
  __int64 v17; // rbx
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // r15
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 *v30; // rbx
  __int64 *v31; // rax
  __int64 v32; // rdx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // rcx
  __int64 v36; // rdi
  __int64 v37; // rsi
  __int64 *v38; // rax
  __int64 v39; // r15
  __int64 v40; // rdi
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // rax
  __int64 *v46; // rax
  unsigned __int64 v47; // r12
  unsigned int v48; // eax
  unsigned __int64 v49; // rdi
  __int64 v50; // rcx
  unsigned int v51; // eax
  _QWORD *v53; // r13
  _QWORD *v54; // rbx
  __int64 v55; // rdx
  __int64 v56; // rax
  _QWORD *v57; // rdi
  __int64 v58; // rax
  __int64 v59; // rbx
  __int64 v60; // r8
  __int64 v61; // r12
  __int64 v62; // rdx
  __int64 v63; // r13
  __int64 v64; // rbx
  __int64 v65; // r15
  __int64 v66; // rax
  __int64 v67; // rdx
  __int64 v68; // rcx
  __int64 v69; // rbx
  __int64 v70; // r14
  __int64 v71; // r12
  __int64 v72; // rdi
  unsigned int v73; // r9d
  int v74; // edx
  __int64 v75; // rax
  __int64 v76; // r8
  __int64 v77; // rax
  __int64 v78; // r15
  __int64 v79; // rax
  __int64 v80; // rdx
  __int64 v81; // r14
  int v82; // eax
  __int64 *v83; // rax
  __int64 v84; // rdx
  __int64 *v85; // rdi
  __int64 v86; // rsi
  __int64 v87; // rdx
  __int64 *v88; // rdx
  __int64 v89; // rsi
  __int64 v90; // rax
  __int64 v91; // rax
  int v92; // edx
  __int64 v93; // rdi
  __int64 v94; // rdx
  __int64 v95; // rdx
  __int64 *v96; // rax
  __int64 *v97; // rdi
  __int64 *v98; // rax
  __int64 *v99; // r14
  __int64 v100; // rcx
  __int64 v101; // rax
  __int64 v102; // r8
  __int64 v103; // rax
  __int64 v104; // rax
  unsigned int v105; // r8d
  __int64 v106; // r9
  unsigned int v107; // esi
  __int64 *v108; // rdx
  __int64 v109; // r10
  __int64 v110; // r8
  __int64 v111; // rdx
  __int64 v112; // rcx
  __int64 v113; // r9
  __int64 v114; // r9
  __int64 v115; // r9
  int v116; // edx
  __int64 v117; // rax
  __int64 v118; // r9
  __int64 v119; // rdx
  __int64 v120; // rax
  __int64 v121; // rax
  __int64 v122; // r8
  __int64 v123; // rax
  __int64 v124; // rdi
  __int64 v125; // rcx
  __int64 v126; // rdi
  __int64 v127; // rdi
  void *v128; // rax
  void *v129; // rsi
  __int64 v130; // rdx
  __int64 v131; // rdx
  __int64 v132; // rcx
  __int64 v133; // rsi
  __int64 v134; // r8
  __int64 v135; // r9
  __int64 v136; // rax
  _QWORD *v137; // rbx
  _QWORD *v138; // r12
  __int64 v139; // rax
  unsigned __int64 v140; // rax
  int v141; // eax
  __int64 v142; // rsi
  __int64 v143; // rax
  _QWORD *v144; // rax
  __int64 v145; // rdi
  int v146; // eax
  unsigned __int64 v147; // rax
  int v148; // edx
  __int64 v149; // [rsp+0h] [rbp-610h]
  _BYTE *v150; // [rsp+0h] [rbp-610h]
  __int64 v151; // [rsp+8h] [rbp-608h]
  _QWORD *v152; // [rsp+18h] [rbp-5F8h]
  __int64 v153; // [rsp+28h] [rbp-5E8h]
  __int64 *v154; // [rsp+30h] [rbp-5E0h]
  __int64 v155; // [rsp+30h] [rbp-5E0h]
  __int64 v156; // [rsp+38h] [rbp-5D8h]
  __int64 v157; // [rsp+38h] [rbp-5D8h]
  _QWORD *v158; // [rsp+38h] [rbp-5D8h]
  const void *v161; // [rsp+50h] [rbp-5C0h]
  __int64 v162; // [rsp+50h] [rbp-5C0h]
  __int64 v163; // [rsp+50h] [rbp-5C0h]
  int v164; // [rsp+50h] [rbp-5C0h]
  __int64 v165; // [rsp+58h] [rbp-5B8h]
  __int64 v167; // [rsp+78h] [rbp-598h] BYREF
  __int64 v168; // [rsp+80h] [rbp-590h] BYREF
  int v169; // [rsp+88h] [rbp-588h]
  __int64 v170; // [rsp+90h] [rbp-580h] BYREF
  _QWORD v171[2]; // [rsp+98h] [rbp-578h] BYREF
  __int64 v172; // [rsp+A8h] [rbp-568h]
  __int64 v173; // [rsp+B0h] [rbp-560h]
  const char *v174; // [rsp+C0h] [rbp-550h] BYREF
  __int64 v175; // [rsp+C8h] [rbp-548h] BYREF
  const char *v176; // [rsp+D0h] [rbp-540h]
  __int64 v177; // [rsp+D8h] [rbp-538h]
  __int64 v178; // [rsp+E0h] [rbp-530h]
  char v179; // [rsp+FCh] [rbp-514h]
  int v180; // [rsp+100h] [rbp-510h]
  __int16 v181; // [rsp+104h] [rbp-50Ch]
  char v182[8]; // [rsp+140h] [rbp-4D0h] BYREF
  int v183; // [rsp+148h] [rbp-4C8h]
  __int64 v184; // [rsp+1A8h] [rbp-468h]
  _BYTE v185[816]; // [rsp+1B8h] [rbp-458h] BYREF
  __int64 v186; // [rsp+4E8h] [rbp-128h]
  char v187; // [rsp+4F0h] [rbp-120h] BYREF
  void *src; // [rsp+4F8h] [rbp-118h]
  __int64 v189; // [rsp+500h] [rbp-110h]
  unsigned int v190; // [rsp+508h] [rbp-108h]

  v9 = a5;
  v12 = a9;
  sub_2C292C0(a5);
  v13 = sub_AA48A0(*(_QWORD *)(*a2)[4]);
  sub_2C46300(v9, a4, v13);
  if ( LOBYTE(qword_500D260[17]) )
    sub_2C4B640(v9);
  sub_2C37A10(v9, a3, a4, a2[8]);
  sub_2C36780(v9, a2[5][42]);
  sub_2C37F10(v9);
  sub_2C33CD0(v9);
  sub_2BF05E0(
    (unsigned int)v182,
    (unsigned int)a2[4],
    a3,
    a4,
    (unsigned int)a2[1],
    a7,
    a6 + 96,
    a6,
    (__int64)v9,
    **a2,
    a2[5][42]);
  if ( (*(_QWORD *)(*v9 + 112LL) & 0xFFFFFFFFFFFFFFF8LL) != *v9 + 112LL )
    (*(void (__fastcall **)(_QWORD, char *))(*(_QWORD *)*v9 + 16LL))(*v9, v182);
  if ( !*(_QWORD *)(a6 + 360) )
  {
    v142 = v9[25];
    LODWORD(v174) = 0;
    BYTE4(v174) = 0;
    *(_QWORD *)(a6 + 360) = sub_2BFB120(v182, v142, &v174);
  }
  v14 = 0;
  if ( *(_DWORD *)(*v9 + 88LL) == 1 )
    v14 = **(_QWORD **)(*v9 + 80LL);
  if ( !a9 )
    v12 = &v187;
  v184 = (*(__int64 (__fastcall **)(__int64, char *))(*(_QWORD *)a6 + 16LL))(a6, v12);
  if ( a8 )
    sub_2C37F10(v9);
  v15 = *(__int64 **)(*(_QWORD *)(a6 + 376) + 56LL);
  if ( v15 && (v16 = v15[1], v17 = *(unsigned int *)(v16 + 304), (_DWORD)v17) && !*(_BYTE *)(v16 + 376) )
  {
    v154 = *a2;
    v156 = (__int64)a2[1];
    v153 = a2[8][14];
    v161 = *(const void **)(v16 + 296);
    v165 = sub_22077B0(0x138u);
    if ( v165 )
      sub_2A28870(v165, v15, v161, v17, (__int64)v154, v156, a7, v153);
    v186 = v165;
    sub_2A2A680(v165);
  }
  else
  {
    v165 = 0;
  }
  (*(void (__fastcall **)(__int64))(*(_QWORD *)a6 + 24LL))(a6);
  v18 = sub_2AB8740(a6, *(_QWORD *)(a6 + 240));
  sub_2BF1580(v9, *(_QWORD *)(a6 + 360), v18, v182);
  sub_2AB1A90(v14, v184);
  v19 = (__int64)v182;
  sub_2BFC330(v9, v182);
  v20 = v9[1];
  if ( *(_DWORD *)(v20 + 64) != 1 )
    BUG();
  v167 = **(_QWORD **)(**(_QWORD **)(v20 + 56) + 56LL);
  if ( a8 )
  {
    v58 = sub_D4B130((__int64)*a2);
    v59 = *(_QWORD *)(a6 + 456);
    v60 = *(_QWORD *)(v58 + 16);
    v61 = v58;
    if ( !v60 )
      goto LABEL_68;
    do
    {
      v62 = *(_QWORD *)(v60 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v62 - 30) <= 0xAu )
      {
        v162 = a6;
        v63 = *(_QWORD *)(a6 + 456);
        v64 = v60;
        while ( 1 )
        {
          v65 = *(_QWORD *)(v62 + 40);
          v66 = sub_AA5930(v61);
          if ( v67 == v66 )
            goto LABEL_64;
          v68 = v64;
          v19 = (__int64)v9;
          v69 = v66;
          v70 = v61;
          v71 = v67;
          do
          {
            v72 = *(_QWORD *)(v69 - 8);
            v73 = *(_DWORD *)(v69 + 72);
            v74 = *(_DWORD *)(v69 + 4) & 0x7FFFFFF;
            if ( !v74 )
            {
LABEL_102:
              v91 = *(_QWORD *)(v72 + 0x1FFFFFFFE0LL);
              if ( v74 != v73 )
                goto LABEL_94;
LABEL_103:
              v155 = v68;
              v157 = v91;
              sub_B48D90(v69);
              v72 = *(_QWORD *)(v69 - 8);
              v68 = v155;
              v91 = v157;
              v74 = *(_DWORD *)(v69 + 4) & 0x7FFFFFF;
              goto LABEL_94;
            }
            v75 = 0;
            v76 = v72 + 32LL * v73;
            do
            {
              if ( v65 == *(_QWORD *)(v76 + 8 * v75) )
                goto LABEL_59;
              ++v75;
            }
            while ( v74 != (_DWORD)v75 );
            v90 = 0;
            while ( v63 != *(_QWORD *)(v76 + 8 * v90) )
            {
              if ( v74 == (_DWORD)++v90 )
                goto LABEL_102;
            }
            v91 = *(_QWORD *)(v72 + 32 * v90);
            if ( v74 == v73 )
              goto LABEL_103;
LABEL_94:
            v92 = (v74 + 1) & 0x7FFFFFF;
            *(_DWORD *)(v69 + 4) = v92 | *(_DWORD *)(v69 + 4) & 0xF8000000;
            v93 = 32LL * (unsigned int)(v92 - 1) + v72;
            if ( *(_QWORD *)v93 )
            {
              v94 = *(_QWORD *)(v93 + 8);
              **(_QWORD **)(v93 + 16) = v94;
              if ( v94 )
                *(_QWORD *)(v94 + 16) = *(_QWORD *)(v93 + 16);
            }
            *(_QWORD *)v93 = v91;
            if ( v91 )
            {
              v95 = *(_QWORD *)(v91 + 16);
              *(_QWORD *)(v93 + 8) = v95;
              if ( v95 )
                *(_QWORD *)(v95 + 16) = v93 + 8;
              *(_QWORD *)(v93 + 16) = v91 + 16;
              *(_QWORD *)(v91 + 16) = v93;
            }
            *(_QWORD *)(*(_QWORD *)(v69 - 8)
                      + 32LL * *(unsigned int *)(v69 + 72)
                      + 8LL * ((*(_DWORD *)(v69 + 4) & 0x7FFFFFFu) - 1)) = v65;
LABEL_59:
            v77 = *(_QWORD *)(v69 + 32);
            if ( !v77 )
              BUG();
            v69 = 0;
            if ( *(_BYTE *)(v77 - 24) == 84 )
              v69 = v77 - 24;
          }
          while ( v71 != v69 );
          v61 = v70;
          v64 = v68;
          v9 = (_QWORD *)v19;
LABEL_64:
          v64 = *(_QWORD *)(v64 + 8);
          if ( !v64 )
          {
LABEL_67:
            v59 = v63;
            a6 = v162;
            goto LABEL_68;
          }
          while ( 1 )
          {
            v62 = *(_QWORD *)(v64 + 24);
            if ( (unsigned __int8)(*(_BYTE *)v62 - 30) <= 0xAu )
              break;
            v64 = *(_QWORD *)(v64 + 8);
            if ( !v64 )
              goto LABEL_67;
          }
        }
      }
      v60 = *(_QWORD *)(v60 + 8);
    }
    while ( v60 );
LABEL_68:
    v78 = *(_QWORD *)(v167 + 120);
    v163 = v167 + 112;
    if ( v167 + 112 != v78 )
    {
      v152 = v9;
      while ( 1 )
      {
        if ( !v78 )
        {
          sub_2ACA740((__int64)v185, &v167);
          BUG();
        }
        v19 = (__int64)&v167;
        sub_2ACA740((__int64)v185, &v167);
        if ( *(_BYTE *)(v78 - 16) == 4 && *(_BYTE *)(v78 + 136) == 81 )
          break;
LABEL_106:
        v78 = *(_QWORD *)(v78 + 8);
        if ( v163 == v78 )
        {
          v9 = v152;
          goto LABEL_108;
        }
      }
      v79 = **(_QWORD **)(v78 + 24);
      if ( !v79 )
        BUG();
      v80 = *(_QWORD *)(v79 + 56);
      if ( !*(_DWORD *)(v79 - 40) )
        BUG();
      v81 = *(_QWORD *)(**(_QWORD **)(v79 - 48) + 40LL);
      v82 = *(_DWORD *)(v80 + 40);
      if ( (unsigned int)(v82 - 17) <= 1 )
      {
        v81 = *(_QWORD *)(v81 - 64);
      }
      else if ( (unsigned int)(v82 - 19) <= 1 )
      {
        v151 = v80;
        v143 = sub_2AAEBF0(v80);
        if ( *(_BYTE *)v81 == 86 )
        {
          v149 = v143;
          v144 = (_QWORD *)sub_986520(v81);
          v145 = *(_QWORD *)(*v144 + 16LL);
          if ( v145 )
          {
            if ( !*(_QWORD *)(v145 + 8) && v149 == v144[4] )
            {
              v81 = v144[8];
              v150 = (_BYTE *)*v144;
              if ( v81 )
              {
                sub_D68CD0((unsigned __int64 *)&v174, 3u, (_QWORD *)(v151 + 8));
                v170 = 32;
                if ( *v150 == 82 )
                {
                  v147 = sub_B53900((__int64)v150);
                  v168 = sub_B53630(v147, v170);
                  v169 = v148;
                }
                sub_D68D70(&v174);
              }
            }
          }
        }
      }
      v83 = *(__int64 **)(v78 + 88);
      v84 = 8LL * *(unsigned int *)(v78 + 96);
      v85 = &v83[(unsigned __int64)v84 / 8];
      v86 = v84 >> 3;
      v87 = v84 >> 5;
      if ( v87 )
      {
        v88 = &v83[4 * v87];
        while ( 1 )
        {
          v89 = *v83;
          if ( *(_BYTE *)(*v83 - 32) == 4 && *(_BYTE *)(v89 + 120) == 75 )
            goto LABEL_135;
          v89 = v83[1];
          if ( *(_BYTE *)(v89 - 32) == 4 )
          {
            if ( *(_BYTE *)(v89 + 120) == 75 )
              goto LABEL_135;
            v89 = v83[2];
            if ( *(_BYTE *)(v89 - 32) != 4 )
            {
LABEL_82:
              v89 = v83[3];
              if ( *(_BYTE *)(v89 - 32) == 4 )
                goto LABEL_156;
              goto LABEL_83;
            }
          }
          else
          {
            v89 = v83[2];
            if ( *(_BYTE *)(v89 - 32) != 4 )
              goto LABEL_82;
          }
          if ( *(_BYTE *)(v89 + 120) == 75 )
            goto LABEL_135;
          v89 = v83[3];
          if ( *(_BYTE *)(v89 - 32) == 4 )
          {
LABEL_156:
            if ( *(_BYTE *)(v89 + 120) == 75 )
              goto LABEL_135;
          }
LABEL_83:
          v83 += 4;
          if ( v88 == v83 )
          {
            v86 = v85 - v83;
            break;
          }
        }
      }
      if ( v86 != 2 )
      {
        if ( v86 != 3 )
        {
          if ( v86 != 1 )
          {
LABEL_88:
            v89 = *v85;
            if ( !*v85 )
            {
LABEL_136:
              v117 = sub_2BFB640(v182, v89, 1);
              v118 = *(_QWORD *)(v81 - 8);
              v119 = v117;
              if ( (*(_DWORD *)(v81 + 4) & 0x7FFFFFF) != 0 )
              {
                v120 = 0;
                while ( v59 != *(_QWORD *)(v118 + 32LL * *(unsigned int *)(v81 + 72) + 8 * v120) )
                {
                  if ( (*(_DWORD *)(v81 + 4) & 0x7FFFFFF) == (_DWORD)++v120 )
                    goto LABEL_159;
                }
                v121 = 32 * v120;
              }
              else
              {
LABEL_159:
                v121 = 0x1FFFFFFFE0LL;
              }
              v122 = *(_QWORD *)(v118 + v121);
              v19 = *(_DWORD *)(v119 + 4) & 0x7FFFFFF;
              if ( (*(_DWORD *)(v119 + 4) & 0x7FFFFFF) != 0 )
              {
                v123 = 0;
                v19 = 8LL * (unsigned int)v19;
                do
                {
                  v124 = *(_QWORD *)(v119 - 8);
                  if ( v59 == *(_QWORD *)(v124 + 32LL * *(unsigned int *)(v119 + 72) + v123) )
                  {
                    v125 = v124 + 4 * v123;
                    if ( *(_QWORD *)v125 )
                    {
                      v126 = *(_QWORD *)(v125 + 8);
                      **(_QWORD **)(v125 + 16) = v126;
                      if ( v126 )
                        *(_QWORD *)(v126 + 16) = *(_QWORD *)(v125 + 16);
                    }
                    *(_QWORD *)v125 = v122;
                    if ( v122 )
                    {
                      v127 = *(_QWORD *)(v122 + 16);
                      *(_QWORD *)(v125 + 8) = v127;
                      if ( v127 )
                        *(_QWORD *)(v127 + 16) = v125 + 8;
                      *(_QWORD *)(v125 + 16) = v122 + 16;
                      *(_QWORD *)(v122 + 16) = v125;
                    }
                  }
                  v123 += 8;
                }
                while ( v19 != v123 );
              }
              goto LABEL_106;
            }
LABEL_135:
            v89 += 56;
            goto LABEL_136;
          }
LABEL_194:
          v89 = *v83;
          if ( *(_BYTE *)(*v83 - 32) == 4 && *(_BYTE *)(v89 + 120) == 75 )
            goto LABEL_135;
          goto LABEL_88;
        }
        v89 = *v83;
        if ( *(_BYTE *)(*v83 - 32) == 4 && *(_BYTE *)(v89 + 120) == 75 )
          goto LABEL_135;
        ++v83;
      }
      v89 = *v83;
      if ( *(_BYTE *)(*v83 - 32) == 4 && *(_BYTE *)(v89 + 120) == 75 )
        goto LABEL_135;
      ++v83;
      goto LABEL_194;
    }
LABEL_108:
    v96 = a2[5];
    v97 = (__int64 *)v96[20];
    v98 = &v97[11 * *((unsigned int *)v96 + 42)];
    if ( v98 == v97 )
      goto LABEL_18;
    v158 = v9;
    v99 = v98;
    while ( 1 )
    {
      v100 = *v97;
      v101 = 0x1FFFFFFFE0LL;
      v102 = *(_QWORD *)(*v97 - 8);
      if ( (*(_DWORD *)(*v97 + 4) & 0x7FFFFFF) != 0 )
      {
        v103 = 0;
        do
        {
          if ( v61 == *(_QWORD *)(v102 + 32LL * *(unsigned int *)(v100 + 72) + 8 * v103) )
          {
            v101 = 32 * v103;
            goto LABEL_115;
          }
          ++v103;
        }
        while ( (*(_DWORD *)(*v97 + 4) & 0x7FFFFFF) != (_DWORD)v103 );
        v101 = 0x1FFFFFFFE0LL;
      }
LABEL_115:
      v104 = *(_QWORD *)(v102 + v101);
      v105 = *(_DWORD *)(a6 + 448);
      v106 = *(_QWORD *)(a6 + 432);
      if ( !v105 )
        goto LABEL_133;
      v107 = (v105 - 1) & (((unsigned int)v100 >> 9) ^ ((unsigned int)v100 >> 4));
      v108 = (__int64 *)(v106 + 16LL * v107);
      v109 = *v108;
      if ( v100 != *v108 )
        break;
LABEL_117:
      v110 = v108[1];
      v111 = 0;
      v19 = *(_DWORD *)(v104 + 4) & 0x7FFFFFF;
      v112 = 8 * v19;
      if ( (_DWORD)v19 )
      {
        do
        {
          while ( 1 )
          {
            v113 = *(_QWORD *)(v104 - 8);
            v19 = v113 + 32LL * *(unsigned int *)(v104 + 72);
            if ( v59 == *(_QWORD *)(v19 + v111) )
            {
              v19 = v113 + 4 * v111;
              if ( *(_QWORD *)v19 )
              {
                v114 = *(_QWORD *)(v19 + 8);
                **(_QWORD **)(v19 + 16) = v114;
                if ( v114 )
                  *(_QWORD *)(v114 + 16) = *(_QWORD *)(v19 + 16);
              }
              *(_QWORD *)v19 = v110;
              if ( v110 )
                break;
            }
            v111 += 8;
            if ( v111 == v112 )
              goto LABEL_128;
          }
          v115 = *(_QWORD *)(v110 + 16);
          *(_QWORD *)(v19 + 8) = v115;
          if ( v115 )
            *(_QWORD *)(v115 + 16) = v19 + 8;
          v111 += 8;
          *(_QWORD *)(v19 + 16) = v110 + 16;
          *(_QWORD *)(v110 + 16) = v19;
        }
        while ( v111 != v112 );
      }
LABEL_128:
      v97 += 11;
      if ( v99 == v97 )
      {
        v9 = v158;
        goto LABEL_18;
      }
    }
    v116 = 1;
    while ( v109 != -4096 )
    {
      v107 = (v105 - 1) & (v116 + v107);
      v164 = v116 + 1;
      v108 = (__int64 *)(v106 + 16LL * v107);
      v109 = *v108;
      if ( v100 == *v108 )
        goto LABEL_117;
      v116 = v164;
    }
LABEL_133:
    v108 = (__int64 *)(v106 + 16LL * v105);
    goto LABEL_117;
  }
LABEL_18:
  v25 = sub_2BF3F10(v9);
  if ( !v25 )
    goto LABEL_26;
  v26 = sub_D49300((__int64)*a2, v19, v21, v22, v23, v24);
  v175 = 32;
  v174 = "llvm.loop.vectorize.followup_all";
  v176 = "llvm.loop.vectorize.followup_vectorized";
  v177 = 39;
  v27 = sub_F6E0D0(v26, (__int64)&v174, 2, byte_3F871B3, 0);
  v171[0] = v28;
  v170 = v27;
  v29 = sub_2BF04D0(v25);
  v30 = a2[1];
  v168 = v29;
  v31 = sub_2ACA740((__int64)v185, &v168);
  v35 = *((unsigned int *)v30 + 6);
  v36 = v30[1];
  v37 = *v31;
  if ( !(_DWORD)v35 )
    goto LABEL_173;
  v35 = (unsigned int)(v35 - 1);
  v32 = (unsigned int)v35 & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
  v38 = (__int64 *)(v36 + 16 * v32);
  v33 = *v38;
  if ( v37 != *v38 )
  {
    v146 = 1;
    while ( v33 != -4096 )
    {
      v34 = (unsigned int)(v146 + 1);
      v32 = (unsigned int)v35 & (v146 + (_DWORD)v32);
      v38 = (__int64 *)(v36 + 16LL * (unsigned int)v32);
      v33 = *v38;
      if ( v37 == *v38 )
        goto LABEL_21;
      v146 = v34;
    }
LABEL_173:
    v39 = 0;
    goto LABEL_22;
  }
LABEL_21:
  v39 = v38[1];
LABEL_22:
  if ( LOBYTE(v171[0]) )
  {
    sub_D49440(v39, v170, v32, v35, v33, v34);
  }
  else
  {
    v133 = sub_D49300((__int64)*a2, v37, v32, v35, v33, v34);
    if ( v133 )
      sub_D49440(v39, v133, v131, v132, v134, v135);
    sub_31A4FD0(&v174, v39, 1, a2[10], 0);
    sub_31A4950(&v174);
  }
  v179 = 0;
  v181 = 0;
  v40 = (__int64)a2[4];
  v180 = 1;
  sub_DFA030(v40);
  if ( v179 != 1 || a8 )
    sub_2AA9AA0(v39, v39, v41, v42, v43, v44);
LABEL_26:
  sub_2AE2E80(a6, (__int64)v182);
  (*(void (__fastcall **)(__int64))(*(_QWORD *)a6 + 32LL))(a6);
  if ( sub_2BF3F10(v9) )
  {
    v45 = v9[1];
    if ( *(_DWORD *)(v45 + 64) != 1 )
      BUG();
    v170 = **(_QWORD **)(**(_QWORD **)(v45 + 56) + 56LL);
    v46 = sub_2ACA740((__int64)v185, &v170);
    v47 = sub_986580(*v46);
    if ( (*(_DWORD *)(v47 + 4) & 0x7FFFFFF) == 3 )
    {
      v139 = sub_D47930((__int64)*a2);
      v140 = sub_986580(v139);
      if ( (unsigned __int8)sub_BC8700(v140) )
      {
        v141 = v183 * *(_DWORD *)v9[18];
        LODWORD(v174) = 1;
        HIDWORD(v174) = v141 - 1;
        sub_BC8EC0(v47, (unsigned int *)&v174, 2, 0);
      }
    }
  }
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  sub_C7D6A0(0, 0, 8);
  v48 = v190;
  *(_DWORD *)(a1 + 24) = v190;
  if ( v48 )
  {
    v128 = (void *)sub_C7D670(16LL * v48, 8);
    v129 = src;
    *(_QWORD *)(a1 + 8) = v128;
    v130 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = v189;
    memcpy(v128, v129, 16 * v130);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
  }
  if ( v165 )
  {
    sub_C7D6A0(*(_QWORD *)(v165 + 256), 16LL * *(unsigned int *)(v165 + 272), 8);
    sub_C7D6A0(*(_QWORD *)(v165 + 224), 16LL * *(unsigned int *)(v165 + 240), 8);
    sub_C7D6A0(*(_QWORD *)(v165 + 192), 16LL * *(unsigned int *)(v165 + 208), 8);
    v49 = *(_QWORD *)(v165 + 96);
    if ( v49 != v165 + 112 )
      _libc_free(v49);
    if ( *(_BYTE *)(v165 + 80) )
    {
      *(_BYTE *)(v165 + 80) = 0;
      v136 = *(unsigned int *)(v165 + 72);
      if ( (_DWORD)v136 )
      {
        v137 = *(_QWORD **)(v165 + 56);
        v138 = &v137[2 * (unsigned int)v136];
        do
        {
          if ( *v137 != -8192 && *v137 != -4096 )
            sub_9C6650(v137 + 1);
          v137 += 2;
        }
        while ( v138 != v137 );
        v136 = *(unsigned int *)(v165 + 72);
      }
      sub_C7D6A0(*(_QWORD *)(v165 + 56), 16 * v136, 8);
      v50 = v165;
      v51 = *(_DWORD *)(v165 + 40);
      if ( !v51 )
        goto LABEL_36;
    }
    else
    {
      v50 = v165;
      v51 = *(_DWORD *)(v165 + 40);
      if ( !v51 )
      {
LABEL_36:
        sub_C7D6A0(*(_QWORD *)(v165 + 24), (unsigned __int64)v51 << 6, 8);
        j_j___libc_free_0(v165);
        goto LABEL_37;
      }
    }
    v53 = *(_QWORD **)(v50 + 24);
    v171[0] = 2;
    v171[1] = 0;
    v54 = &v53[8 * (unsigned __int64)v51];
    v172 = -4096;
    v170 = (__int64)&unk_49DD7B0;
    v174 = (const char *)&unk_49DD7B0;
    v55 = -4096;
    v173 = 0;
    v175 = 2;
    v176 = 0;
    v177 = -8192;
    v178 = 0;
    while ( 1 )
    {
      v56 = v53[3];
      if ( v56 != v55 && v56 != v177 )
        sub_D68D70(v53 + 5);
      *v53 = &unk_49DB368;
      v57 = v53 + 1;
      v53 += 8;
      sub_D68D70(v57);
      if ( v54 == v53 )
        break;
      v55 = v172;
    }
    v174 = (const char *)&unk_49DB368;
    sub_D68D70(&v175);
    v170 = (__int64)&unk_49DB368;
    sub_D68D70(v171);
    v51 = *(_DWORD *)(v165 + 40);
    goto LABEL_36;
  }
LABEL_37:
  sub_2AB6BC0((__int64)v182);
  return a1;
}
