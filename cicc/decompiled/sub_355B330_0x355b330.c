// Function: sub_355B330
// Address: 0x355b330
//
__int64 __fastcall sub_355B330(__int64 a1, __int64 a2, __int64 *a3, unsigned __int64 *a4)
{
  int v6; // eax
  __int64 **v7; // r13
  __int64 v8; // r14
  __int64 v9; // rbx
  __int64 v10; // r15
  __int64 (*v11)(); // rax
  __int64 v12; // rax
  unsigned int v13; // r8d
  __int64 v14; // rdi
  __int64 (*v15)(); // rax
  __int16 v16; // ax
  __int64 v17; // r8
  char v18; // si
  char v19; // di
  __int64 *v20; // rsi
  _QWORD *v21; // rcx
  _QWORD *v22; // r9
  _QWORD *v23; // rax
  __int64 v24; // r8
  __int64 v25; // rdx
  int v26; // eax
  _QWORD *v27; // r9
  _QWORD *v28; // rax
  __int64 v29; // r8
  __int64 v30; // rdx
  int v31; // eax
  __int64 v32; // rax
  __int64 **v33; // rcx
  __int64 v34; // rdi
  __int64 *v35; // rsi
  unsigned int v36; // r9d
  __int64 v37; // rax
  _QWORD *v38; // rax
  _QWORD *v39; // r11
  __int64 v40; // r10
  __int64 v41; // rdx
  int v42; // eax
  __int64 v43; // rax
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 *v46; // rsi
  bool v47; // r9
  unsigned int v48; // r10d
  _QWORD *v49; // rax
  _QWORD *v50; // r15
  __int64 v51; // rbx
  __int64 v52; // rdx
  int v53; // eax
  unsigned __int64 v54; // rdx
  unsigned __int64 v55; // rax
  __int64 v56; // r8
  _QWORD *v57; // r11
  __int64 v58; // r13
  unsigned __int64 v59; // rsi
  unsigned __int64 v60; // rcx
  unsigned __int64 *v61; // rax
  _QWORD *v62; // r11
  __int64 *v63; // r8
  unsigned __int64 v64; // rax
  unsigned __int64 v65; // rsi
  unsigned __int64 v66; // rcx
  unsigned __int64 *v67; // rax
  unsigned __int64 v68; // rax
  _QWORD *v69; // r11
  __int64 result; // rax
  _QWORD *v71; // rcx
  __int64 *v72; // rdi
  _QWORD *v73; // r9
  _QWORD *v74; // rax
  __int64 v75; // r8
  __int64 v76; // rdx
  int v77; // eax
  _QWORD *v78; // r8
  unsigned int v79; // eax
  _QWORD *v80; // rax
  _QWORD *v81; // r11
  __int64 v82; // r10
  __int64 v83; // rdx
  int v84; // eax
  __int64 v85; // rdx
  __int64 v86; // rax
  int v87; // eax
  bool v88; // cc
  unsigned int v89; // eax
  char v90; // al
  char v91; // al
  __int64 v92; // rdi
  __int64 v93; // rax
  unsigned int v94; // edx
  __int64 *v95; // rsi
  __int64 v96; // r9
  _QWORD *v97; // rdi
  __int64 v98; // rdx
  __int64 v99; // rax
  int v100; // eax
  unsigned int v101; // edi
  char v102; // al
  char v103; // di
  _QWORD *v104; // r9
  _QWORD *v105; // rax
  __int64 v106; // rdx
  int v107; // eax
  unsigned int v108; // eax
  _QWORD *v109; // rdx
  _QWORD *v110; // rax
  __int64 v111; // r9
  __int64 v112; // r8
  int v113; // r10d
  _QWORD *v114; // r11
  int v115; // r9d
  unsigned __int64 v116; // r8
  __int64 v117; // rdx
  __int64 v118; // rax
  int v119; // eax
  unsigned __int64 *v120; // rax
  _QWORD *v121; // rax
  _QWORD *v122; // rcx
  int v123; // esi
  int v124; // r11d
  __int64 v125; // rsi
  unsigned __int64 v126; // rsi
  unsigned __int64 v127; // rcx
  unsigned __int64 *v128; // rax
  _QWORD *v129; // r11
  unsigned __int64 v130; // rax
  unsigned __int64 v131; // rsi
  unsigned __int64 v132; // rcx
  unsigned __int64 *v133; // rax
  unsigned __int64 v134; // rax
  _QWORD *v135; // r11
  _QWORD *v137; // [rsp+0h] [rbp-F0h]
  unsigned __int64 v139; // [rsp+10h] [rbp-E0h]
  __int64 **v140; // [rsp+18h] [rbp-D8h]
  __int64 **v141; // [rsp+20h] [rbp-D0h]
  _QWORD *v142; // [rsp+28h] [rbp-C8h]
  __int64 v143; // [rsp+30h] [rbp-C0h]
  unsigned int v144; // [rsp+38h] [rbp-B8h]
  char v145; // [rsp+3Dh] [rbp-B3h]
  char v146; // [rsp+3Eh] [rbp-B2h]
  char v147; // [rsp+3Fh] [rbp-B1h]
  char v148[4]; // [rsp+40h] [rbp-B0h]
  unsigned int v149; // [rsp+44h] [rbp-ACh]
  int v150; // [rsp+48h] [rbp-A8h]
  __int64 *v151; // [rsp+48h] [rbp-A8h]
  int v152; // [rsp+48h] [rbp-A8h]
  __int64 v153; // [rsp+48h] [rbp-A8h]
  int v154; // [rsp+50h] [rbp-A0h]
  __int64 v155; // [rsp+50h] [rbp-A0h]
  unsigned __int64 v156; // [rsp+58h] [rbp-98h] BYREF
  __int64 *v157; // [rsp+60h] [rbp-90h] BYREF
  unsigned __int64 v158; // [rsp+68h] [rbp-88h]
  unsigned __int64 v159; // [rsp+70h] [rbp-80h]
  unsigned __int64 *v160; // [rsp+78h] [rbp-78h]
  __int64 v161[4]; // [rsp+80h] [rbp-70h] BYREF
  __int64 *v162; // [rsp+A0h] [rbp-50h] BYREF
  unsigned __int64 v163; // [rsp+A8h] [rbp-48h]
  unsigned __int64 v164; // [rsp+B0h] [rbp-40h]
  unsigned __int64 *v165; // [rsp+B8h] [rbp-38h]

  v156 = (unsigned __int64)a3;
  v143 = *a3;
  v6 = sub_3542500(a1, (unsigned __int64)a3);
  v7 = (__int64 **)a4[2];
  v154 = v6;
  v142 = *(_QWORD **)(a2 + 3464);
  v141 = (__int64 **)a4[4];
  v139 = a4[5];
  v140 = (__int64 **)a4[6];
  if ( v7 == v140 )
    goto LABEL_186;
  v149 = 0;
  v8 = a1 + 40;
  *(_DWORD *)v148 = 0;
  v144 = 0;
  v145 = 0;
  v147 = 0;
  v146 = 0;
  do
  {
    v9 = *(_QWORD *)(v143 + 32);
    v10 = v9 + 40LL * (*(_DWORD *)(v143 + 40) & 0xFFFFFF);
    if ( v10 != v9 )
    {
      while ( 1 )
      {
        while ( 1 )
        {
          if ( *(_BYTE *)v9 )
            goto LABEL_30;
          v150 = *(_DWORD *)(v9 + 8);
          if ( v150 >= 0 )
            goto LABEL_30;
          v11 = *(__int64 (**)())(**(_QWORD **)(a1 + 96) + 128LL);
          if ( v11 == sub_2DAC790 )
            BUG();
          v12 = v11();
          v13 = v150;
          v14 = v12;
          v15 = *(__int64 (**)())(*(_QWORD *)v12 + 824LL);
          if ( v15 != sub_2FDC6B0 )
          {
            v91 = ((__int64 (__fastcall *)(__int64, __int64, __int64 *, __int64 **, _QWORD))v15)(
                    v14,
                    v143,
                    v161,
                    &v162,
                    (unsigned int)v150);
            v13 = v150;
            if ( v91 )
            {
              if ( v150 == *(_DWORD *)(*(_QWORD *)(v143 + 32) + 40LL * LODWORD(v161[0]) + 8) )
              {
                v92 = *(_QWORD *)(a2 + 4024);
                v93 = *(unsigned int *)(a2 + 4040);
                if ( (_DWORD)v93 )
                {
                  v94 = (v93 - 1) & (((unsigned int)v156 >> 9) ^ ((unsigned int)v156 >> 4));
                  v95 = (__int64 *)(v92 + 24LL * v94);
                  v96 = *v95;
                  if ( v156 == *v95 )
                  {
LABEL_121:
                    if ( v95 != (__int64 *)(v92 + 24 * v93) && *((_DWORD *)v95 + 2) )
                      v13 = *((_DWORD *)v95 + 2);
                  }
                  else
                  {
                    v123 = 1;
                    while ( v96 != -4096 )
                    {
                      v124 = v123 + 1;
                      v125 = ((_DWORD)v93 - 1) & (v94 + v123);
                      v94 = v125;
                      v95 = (__int64 *)(v92 + 24 * v125);
                      v96 = *v95;
                      if ( v156 == *v95 )
                        goto LABEL_121;
                      v123 = v124;
                    }
                  }
                }
              }
            }
          }
          v16 = sub_2E89D80(**v7, v13, 0);
          v18 = v16;
          v19 = HIBYTE(v16);
          if ( (*(_BYTE *)(v9 + 3) & 0x10) != 0 )
            break;
          v20 = *v7;
          v21 = *(_QWORD **)(a1 + 48);
          if ( HIBYTE(v16) )
          {
            if ( v21 )
            {
              v22 = (_QWORD *)(a1 + 40);
              v23 = *(_QWORD **)(a1 + 48);
              do
              {
                while ( 1 )
                {
                  v24 = v23[2];
                  v25 = v23[3];
                  if ( v23[4] >= (unsigned __int64)v20 )
                    break;
                  v23 = (_QWORD *)v23[3];
                  if ( !v25 )
                    goto LABEL_15;
                }
                v22 = v23;
                v23 = (_QWORD *)v23[2];
              }
              while ( v24 );
LABEL_15:
              v26 = -1;
              if ( v22 != (_QWORD *)v8 && v22[4] <= (unsigned __int64)v20 )
                v26 = (*((_DWORD *)v22 + 10) - *(_DWORD *)(a1 + 80)) / *(_DWORD *)(a1 + 88);
              if ( v154 == v26 )
              {
                v109 = (_QWORD *)(a1 + 40);
                v110 = *(_QWORD **)(a1 + 48);
                do
                {
                  while ( 1 )
                  {
                    v111 = v110[2];
                    v112 = v110[3];
                    if ( v110[4] >= (unsigned __int64)v20 )
                      break;
                    v110 = (_QWORD *)v110[3];
                    if ( !v112 )
                      goto LABEL_159;
                  }
                  v109 = v110;
                  v110 = (_QWORD *)v110[2];
                }
                while ( v111 );
LABEL_159:
                if ( v109 != (_QWORD *)v8 && v109[4] > (unsigned __int64)v20 )
                  v109 = (_QWORD *)(a1 + 40);
                v113 = *(_DWORD *)(a1 + 80);
                v114 = (_QWORD *)(a1 + 40);
                v115 = *(_DWORD *)(a1 + 88);
                v116 = v156;
                v152 = (*((_DWORD *)v109 + 10) - v113) % v115;
                do
                {
                  while ( 1 )
                  {
                    v117 = v21[2];
                    v118 = v21[3];
                    if ( v21[4] >= v156 )
                      break;
                    v21 = (_QWORD *)v21[3];
                    if ( !v118 )
                      goto LABEL_166;
                  }
                  v114 = v21;
                  v21 = (_QWORD *)v21[2];
                }
                while ( v117 );
LABEL_166:
                if ( v114 != (_QWORD *)v8 )
                {
                  v119 = *(_DWORD *)(a1 + 80);
                  if ( v114[4] <= v156 )
                    v119 = *((_DWORD *)v114 + 10);
LABEL_169:
                  if ( (v119 - v113) % v115 != v152 )
                    goto LABEL_170;
                  goto LABEL_191;
                }
                if ( 0 % v115 != v152 )
                  goto LABEL_170;
LABEL_191:
                v121 = (_QWORD *)v20[15];
                v122 = &v121[2 * *((unsigned int *)v20 + 32)];
                if ( v121 == v122 )
                  goto LABEL_148;
                while ( v116 != (*v121 & 0xFFFFFFFFFFFFFFF8LL) )
                {
                  v121 += 2;
                  if ( v122 == v121 )
                    goto LABEL_148;
                }
LABEL_170:
                v9 += 40;
                v147 = v19;
                v144 = v149;
                if ( v10 == v9 )
                  goto LABEL_31;
              }
              else
              {
                v27 = (_QWORD *)(a1 + 40);
                v28 = *(_QWORD **)(a1 + 48);
                do
                {
                  while ( 1 )
                  {
                    v29 = v28[2];
                    v30 = v28[3];
                    if ( v28[4] >= (unsigned __int64)v20 )
                      break;
                    v28 = (_QWORD *)v28[3];
                    if ( !v30 )
                      goto LABEL_23;
                  }
                  v27 = v28;
                  v28 = (_QWORD *)v28[2];
                }
                while ( v29 );
LABEL_23:
                v31 = -1;
                if ( v27 != (_QWORD *)v8 && v27[4] <= (unsigned __int64)v20 )
                  v31 = (*((_DWORD *)v27 + 10) - *(_DWORD *)(a1 + 80)) / *(_DWORD *)(a1 + 88);
                if ( v154 < v31 )
                  goto LABEL_27;
                v104 = (_QWORD *)(a1 + 40);
                v105 = *(_QWORD **)(a1 + 48);
                do
                {
                  while ( 1 )
                  {
                    v17 = v105[2];
                    v106 = v105[3];
                    if ( v105[4] >= (unsigned __int64)v20 )
                      break;
                    v105 = (_QWORD *)v105[3];
                    if ( !v106 )
                      goto LABEL_144;
                  }
                  v104 = v105;
                  v105 = (_QWORD *)v105[2];
                }
                while ( v17 );
LABEL_144:
                v107 = -1;
                if ( v104 != (_QWORD *)v8 && v104[4] <= (unsigned __int64)v20 )
                  v107 = (*((_DWORD *)v104 + 10) - *(_DWORD *)(a1 + 80)) / *(_DWORD *)(a1 + 88);
                if ( v154 <= v107 )
                  goto LABEL_126;
LABEL_148:
                v108 = *(_DWORD *)v148;
                v146 = v19;
                if ( !*(_DWORD *)v148 )
                  v108 = v149;
                v9 += 40;
                *(_DWORD *)v148 = v108;
                if ( v10 == v9 )
                  goto LABEL_31;
              }
            }
            else
            {
              if ( v154 == -1 )
              {
                v119 = *(_DWORD *)(a1 + 80);
                v115 = *(_DWORD *)(a1 + 88);
                v152 = 0;
                v116 = v156;
                v113 = v119;
                goto LABEL_169;
              }
              if ( v154 >= -1 )
                goto LABEL_148;
LABEL_27:
              if ( *(_DWORD *)v148 )
              {
LABEL_174:
                v9 += 40;
                v147 = v19;
                v146 = v19;
                v144 = v149 - 1;
                if ( v10 == v9 )
                  goto LABEL_31;
              }
              else
              {
                if ( v149 )
                {
                  *(_DWORD *)v148 = v149;
                  goto LABEL_174;
                }
                v146 = v19;
LABEL_30:
                v9 += 40;
                if ( v10 == v9 )
                  goto LABEL_31;
              }
            }
          }
          else
          {
            if ( !v21 )
            {
              v100 = -1;
              goto LABEL_133;
            }
LABEL_126:
            v97 = (_QWORD *)(a1 + 40);
            do
            {
              while ( 1 )
              {
                v98 = v21[2];
                v99 = v21[3];
                if ( v21[4] >= (unsigned __int64)v20 )
                  break;
                v21 = (_QWORD *)v21[3];
                if ( !v99 )
                  goto LABEL_130;
              }
              v97 = v21;
              v21 = (_QWORD *)v21[2];
            }
            while ( v98 );
LABEL_130:
            v100 = -1;
            if ( v97 != (_QWORD *)v8 && v97[4] <= (unsigned __int64)v20 )
              v100 = (*((_DWORD *)v97 + 10) - *(_DWORD *)(a1 + 80)) / *(_DWORD *)(a1 + 88);
LABEL_133:
            if ( v154 != v100 )
              goto LABEL_30;
            v101 = *(_DWORD *)v148;
            v102 = sub_3544D50(a1, a2, *v20, v9, v17) & (*(_DWORD *)v148 == 0);
            if ( v102 )
              v101 = v149;
            *(_DWORD *)v148 = v101;
            v103 = v145;
            if ( v102 )
              v103 = v102;
            v9 += 40;
            v145 = v103;
            if ( v10 == v9 )
              goto LABEL_31;
          }
        }
        if ( !(_BYTE)v16 )
          goto LABEL_30;
        v71 = *(_QWORD **)(a1 + 48);
        v72 = *v7;
        if ( v71 )
        {
          v73 = (_QWORD *)(a1 + 40);
          v74 = *(_QWORD **)(a1 + 48);
          do
          {
            while ( 1 )
            {
              v75 = v74[2];
              v76 = v74[3];
              if ( v74[4] >= (unsigned __int64)v72 )
                break;
              v74 = (_QWORD *)v74[3];
              if ( !v76 )
                goto LABEL_85;
            }
            v73 = v74;
            v74 = (_QWORD *)v74[2];
          }
          while ( v75 );
LABEL_85:
          v77 = -1;
          if ( (_QWORD *)v8 != v73 && v73[4] <= (unsigned __int64)v72 )
            v77 = (*((_DWORD *)v73 + 10) - *(_DWORD *)(a1 + 80)) / *(_DWORD *)(a1 + 88);
          v78 = (_QWORD *)(a1 + 40);
          if ( v154 >= v77 )
            goto LABEL_89;
          do
          {
            while ( 1 )
            {
              v85 = v71[2];
              v86 = v71[3];
              if ( v71[4] >= (unsigned __int64)v72 )
                break;
              v71 = (_QWORD *)v71[3];
              if ( !v86 )
                goto LABEL_108;
            }
            v78 = v71;
            v71 = (_QWORD *)v71[2];
          }
          while ( v85 );
LABEL_108:
          v87 = -1;
          if ( (_QWORD *)v8 != v78 && v78[4] <= (unsigned __int64)v72 )
            v87 = (*((_DWORD *)v78 + 10) - *(_DWORD *)(a1 + 80)) / *(_DWORD *)(a1 + 88);
          v88 = v154 < v87;
          v89 = v144;
          if ( v88 )
            v89 = v149;
          v144 = v89;
          v90 = v147;
          if ( v88 )
            v90 = v18;
          v9 += 40;
          v147 = v90;
          if ( v10 == v9 )
            break;
        }
        else
        {
          if ( v154 < -1 )
          {
            v147 = v16;
            v144 = v149;
            goto LABEL_30;
          }
LABEL_89:
          v79 = *(_DWORD *)v148;
          v146 = v18;
          if ( !*(_DWORD *)v148 )
            v79 = v149;
          v9 += 40;
          *(_DWORD *)v148 = v79;
          if ( v10 == v9 )
            break;
        }
      }
    }
LABEL_31:
    v32 = sub_3545E90(v142, v156);
    v33 = *(__int64 ***)v32;
    v34 = *(_QWORD *)v32 + 32LL * *(unsigned int *)(v32 + 8);
    if ( v34 == *(_QWORD *)v32 )
      goto LABEL_50;
    v35 = *v7;
    v36 = *(_DWORD *)v148;
    do
    {
      while ( 1 )
      {
        if ( v35 == *v33 )
        {
          v37 = ((__int64)v33[1] >> 1) & 3;
          if ( v37 == 3 )
          {
            v80 = *(_QWORD **)(a1 + 48);
            if ( v80 )
            {
              v81 = (_QWORD *)(a1 + 40);
              do
              {
                while ( 1 )
                {
                  v82 = v80[2];
                  v83 = v80[3];
                  if ( v80[4] >= (unsigned __int64)v35 )
                    break;
                  v80 = (_QWORD *)v80[3];
                  if ( !v83 )
                    goto LABEL_98;
                }
                v81 = v80;
                v80 = (_QWORD *)v80[2];
              }
              while ( v82 );
LABEL_98:
              v84 = -1;
              if ( (_QWORD *)v8 != v81 && v81[4] <= (unsigned __int64)v35 )
                v84 = (*((_DWORD *)v81 + 10) - *(_DWORD *)(a1 + 80)) / *(_DWORD *)(a1 + 88);
            }
            else
            {
              v84 = -1;
            }
            if ( v154 == v84 )
            {
              v146 = 1;
              if ( v36 > v149 )
                v36 = v149;
            }
            goto LABEL_33;
          }
          if ( v37 == 1 || v37 == 2 )
          {
            v38 = *(_QWORD **)(a1 + 48);
            if ( v38 )
            {
              v39 = (_QWORD *)(a1 + 40);
              do
              {
                while ( 1 )
                {
                  v40 = v38[2];
                  v41 = v38[3];
                  if ( v38[4] >= (unsigned __int64)v35 )
                    break;
                  v38 = (_QWORD *)v38[3];
                  if ( !v41 )
                    goto LABEL_43;
                }
                v39 = v38;
                v38 = (_QWORD *)v38[2];
              }
              while ( v40 );
LABEL_43:
              v42 = -1;
              if ( (_QWORD *)v8 != v39 && v39[4] <= (unsigned __int64)v35 )
                v42 = (*((_DWORD *)v39 + 10) - *(_DWORD *)(a1 + 80)) / *(_DWORD *)(a1 + 88);
            }
            else
            {
              v42 = -1;
            }
            if ( v154 == v42 )
              break;
          }
        }
LABEL_33:
        v33 += 4;
        if ( (__int64 **)v34 == v33 )
          goto LABEL_49;
      }
      v146 = v149 < v36 || v36 == 0;
      if ( !v146 )
      {
        v146 = 1;
        goto LABEL_33;
      }
      v33 += 4;
      v36 = v149;
    }
    while ( (__int64 **)v34 != v33 );
LABEL_49:
    *(_DWORD *)v148 = v36;
LABEL_50:
    v43 = sub_35459D0(v142, v156);
    v44 = *(_QWORD *)v43;
    v45 = *(_QWORD *)v43 + 32LL * *(unsigned int *)(v43 + 8);
    if ( v45 != *(_QWORD *)v43 )
    {
      v46 = *v7;
      v47 = v147;
      v48 = v144;
      do
      {
        while ( v46 != (__int64 *)(*(_QWORD *)(v44 + 8) & 0xFFFFFFFFFFFFFFF8LL)
             || ((*(__int64 *)(v44 + 8) >> 1) & 3) != 1 && ((unsigned int)(*(__int64 *)(v44 + 8) >> 1) & 3) - 2 > 1 )
        {
          v44 += 32;
          if ( v44 == v45 )
            goto LABEL_66;
        }
        v49 = *(_QWORD **)(a1 + 48);
        if ( v49 )
        {
          v50 = (_QWORD *)(a1 + 40);
          do
          {
            while ( 1 )
            {
              v51 = v49[2];
              v52 = v49[3];
              if ( v49[4] >= (unsigned __int64)v46 )
                break;
              v49 = (_QWORD *)v49[3];
              if ( !v52 )
                goto LABEL_60;
            }
            v50 = v49;
            v49 = (_QWORD *)v49[2];
          }
          while ( v51 );
LABEL_60:
          v53 = -1;
          if ( (_QWORD *)v8 != v50 && v50[4] <= (unsigned __int64)v46 )
            v53 = (*((_DWORD *)v50 + 10) - *(_DWORD *)(a1 + 80)) / *(_DWORD *)(a1 + 88);
        }
        else
        {
          v53 = -1;
        }
        if ( v154 == v53 )
        {
          v48 = v149;
          v47 = ((*(__int64 *)(v44 + 8) >> 1) & 3) == 1 || ((unsigned int)(*(__int64 *)(v44 + 8) >> 1) & 3) - 2 <= 1;
        }
        v44 += 32;
      }
      while ( v44 != v45 );
LABEL_66:
      v147 = v47;
      v144 = v48;
    }
    if ( v141 == ++v7 )
    {
      v7 = *(__int64 ***)(v139 + 8);
      v139 += 8LL;
      v141 = v7 + 64;
    }
    ++v149;
  }
  while ( v140 != v7 );
  if ( !v147 || !v146 )
  {
    if ( v145 )
      goto LABEL_181;
    goto LABEL_183;
  }
  if ( v144 == *(_DWORD *)v148 )
    goto LABEL_186;
  if ( !v145 )
    goto LABEL_74;
LABEL_181:
  v146 = (v144 < *(_DWORD *)v148) | v147 ^ 1;
  if ( v144 < *(_DWORD *)v148 && v147 )
  {
LABEL_74:
    v54 = sub_3549CC0(a4 + 6, a4 + 2);
    if ( *(unsigned int *)v148 >= v54 )
      sub_222CF80("deque::_M_range_check: __n (which is %zu)>= this->size() (which is %zu)", *(unsigned int *)v148, v54);
    v162 = (__int64 *)a4[2];
    v163 = a4[3];
    v164 = a4[4];
    v165 = (unsigned __int64 *)a4[5];
    sub_353DF70((__int64 *)&v162, *(unsigned int *)v148);
    v155 = *v162;
    v55 = sub_3549CC0(a4 + 6, a4 + 2);
    if ( v144 >= v55 )
      sub_222CF80("deque::_M_range_check: __n (which is %zu)>= this->size() (which is %zu)", v144, v55);
    v162 = (__int64 *)a4[2];
    v163 = a4[3];
    v164 = a4[4];
    v165 = (unsigned __int64 *)a4[5];
    sub_353DF70((__int64 *)&v162, v144);
    v58 = *v162;
    if ( v144 < *(_DWORD *)v148 )
    {
      v126 = v57[3];
      v153 = v56;
      v127 = v57[4];
      v128 = (unsigned __int64 *)v57[5];
      v157 = (__int64 *)v57[2];
      v158 = v126;
      v159 = v127;
      v160 = v128;
      sub_353DF70((__int64 *)&v157, *(unsigned int *)v148);
      v137 = v129;
      v162 = v157;
      v130 = *v160;
      v165 = v160;
      v163 = v130;
      v164 = v130 + 512;
      sub_355AFE0(v161, v129, (__int64 *)&v162);
      v131 = v137[3];
      v132 = v137[4];
      v133 = (unsigned __int64 *)v137[5];
      v157 = (__int64 *)v137[2];
      v158 = v131;
      v159 = v132;
      v160 = v133;
      sub_353DF70((__int64 *)&v157, v153);
      v162 = v157;
      v134 = *v160;
      v165 = v160;
      v163 = v134;
      v164 = v134 + 512;
      sub_355AFE0(v161, v135, (__int64 *)&v162);
    }
    else
    {
      v59 = a4[3];
      v60 = a4[4];
      v61 = (unsigned __int64 *)a4[5];
      v157 = (__int64 *)a4[2];
      v158 = v59;
      v159 = v60;
      v160 = v61;
      sub_353DF70((__int64 *)&v157, v56);
      v137 = v62;
      v151 = v63;
      v162 = v157;
      v64 = *v160;
      v165 = v160;
      v163 = v64;
      v164 = v64 + 512;
      sub_355AFE0(v63, v62, (__int64 *)&v162);
      v65 = v137[3];
      v66 = v137[4];
      v67 = (unsigned __int64 *)v137[5];
      v157 = (__int64 *)v137[2];
      v158 = v65;
      v159 = v66;
      v160 = v67;
      sub_353DF70((__int64 *)&v157, *(unsigned int *)v148);
      v162 = v157;
      v68 = *v160;
      v165 = v160;
      v163 = v68;
      v164 = v68 + 512;
      sub_355AFE0(v151, v69, (__int64 *)&v162);
    }
    sub_355B330(a1, a2, v155, v137);
    sub_355B330(a1, a2, v156, v137);
    return sub_355B330(a1, a2, v58, v137);
  }
  else
  {
LABEL_183:
    if ( v146 )
    {
      result = a4[2];
      if ( result == a4[3] )
        return sub_354AFF0(a4, &v156);
      *(_QWORD *)(result - 8) = v156;
      a4[2] -= 8LL;
      return result;
    }
LABEL_186:
    v120 = (unsigned __int64 *)a4[6];
    if ( v120 == (unsigned __int64 *)(a4[8] - 8) )
    {
      return sub_354B0D0(a4, &v156);
    }
    else
    {
      if ( v120 )
      {
        *v120 = v156;
        v120 = (unsigned __int64 *)a4[6];
      }
      result = (__int64)(v120 + 1);
      a4[6] = result;
    }
  }
  return result;
}
