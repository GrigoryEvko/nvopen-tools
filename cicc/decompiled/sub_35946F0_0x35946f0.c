// Function: sub_35946F0
// Address: 0x35946f0
//
float *__fastcall sub_35946F0(_QWORD *a1, __int64 a2, float **a3, __int64 a4, __int64 a5, __int64 a6, float a7)
{
  _QWORD *v7; // r14
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // rax
  float v11; // xmm0_4
  float v12; // xmm1_4
  float v13; // xmm2_4
  float v14; // xmm3_4
  float v15; // xmm3_4
  float v16; // xmm3_4
  float v17; // xmm3_4
  float v18; // xmm3_4
  float *result; // rax
  int v20; // edx
  __int64 v21; // r9
  int *v22; // rax
  unsigned int v23; // edi
  int *v24; // r13
  int v25; // ecx
  __int64 v26; // rcx
  int v27; // r12d
  __int64 *v28; // rsi
  __int64 v29; // rax
  __int64 v30; // rcx
  __int64 v31; // rcx
  __int64 v32; // rsi
  int v33; // r11d
  int v34; // r11d
  __int64 v35; // r9
  unsigned int v36; // edx
  int v37; // eax
  int v38; // edi
  int *v39; // rcx
  __int64 *v40; // r12
  __int64 v41; // rax
  __int64 v42; // r15
  __int64 v43; // rsi
  __int64 v44; // rax
  double v45; // xmm0_8
  __int64 v46; // rax
  double v47; // xmm1_8
  double v48; // xmm0_8
  __int64 v49; // rdx
  unsigned __int64 v50; // r12
  unsigned __int64 v51; // rcx
  __int64 v52; // rbx
  __int64 v53; // rax
  __int64 v54; // r13
  __int64 v55; // rsi
  __int64 v56; // rax
  double v57; // xmm0_8
  __int64 v58; // rax
  double v59; // xmm1_8
  int v60; // r15d
  __int64 v61; // rax
  int v62; // eax
  __int64 v63; // rax
  __int64 v64; // rdi
  __int64 v65; // rax
  unsigned __int64 v66; // rcx
  __int64 v67; // r8
  __int64 v68; // r9
  __int64 v69; // rdx
  __int64 v70; // rax
  __int64 v71; // r12
  double *v72; // r14
  __int64 v73; // r15
  char *v74; // rax
  __int64 v75; // rcx
  __int64 v76; // rdi
  __int64 (*v77)(); // rax
  char v78; // al
  bool v79; // zf
  __int16 v80; // ax
  __int16 v81; // ax
  char v82; // r13
  __int64 *v83; // rbx
  __int64 v84; // rax
  __int64 v85; // rax
  double v86; // xmm0_8
  double v87; // xmm2_8
  double v88; // xmm3_8
  float v89; // xmm5_4
  double v90; // xmm1_8
  double v91; // xmm0_8
  __int64 v92; // rax
  __int64 v93; // rdi
  __int64 *v94; // rax
  __int64 v95; // r11
  __int64 v96; // rbx
  __int64 *v97; // r12
  __int64 v98; // r13
  __int64 v99; // r14
  __int64 *v100; // rbx
  _QWORD *v101; // rax
  __int64 v102; // rdx
  __int64 v103; // rbx
  __int64 *v104; // rax
  _DWORD *v105; // rax
  int i; // eax
  int v107; // edi
  int v108; // r11d
  int v109; // r11d
  __int64 v110; // r9
  __int64 v111; // rdx
  int v112; // edi
  __int64 v113; // rax
  _QWORD *v114; // rdi
  __int64 v115; // rax
  __int64 v116; // rcx
  __int64 *v117; // rsi
  unsigned __int64 v118; // rdx
  __int64 v119; // rax
  _QWORD *v120; // r8
  __int64 v121; // rcx
  __int64 *v122; // rsi
  _QWORD *v125; // [rsp+20h] [rbp-170h]
  __int64 v126; // [rsp+28h] [rbp-168h]
  __int64 v127; // [rsp+30h] [rbp-160h]
  int v130; // [rsp+44h] [rbp-14Ch]
  __int64 v131; // [rsp+48h] [rbp-148h]
  int v132; // [rsp+50h] [rbp-140h]
  signed __int64 v133; // [rsp+60h] [rbp-130h]
  signed __int64 v134; // [rsp+68h] [rbp-128h]
  __int64 v135; // [rsp+70h] [rbp-120h]
  __int64 v136; // [rsp+78h] [rbp-118h]
  double v137; // [rsp+80h] [rbp-110h]
  double v138; // [rsp+88h] [rbp-108h]
  double v139; // [rsp+90h] [rbp-100h]
  double v140; // [rsp+98h] [rbp-F8h]
  double v141; // [rsp+A0h] [rbp-F0h]
  float v142; // [rsp+A8h] [rbp-E8h]
  float v143; // [rsp+ACh] [rbp-E4h]
  __int64 v144; // [rsp+B0h] [rbp-E0h]
  __int64 ***v145; // [rsp+B8h] [rbp-D8h]
  double *v146; // [rsp+C0h] [rbp-D0h]
  __int64 v147; // [rsp+C8h] [rbp-C8h]
  double v148; // [rsp+D0h] [rbp-C0h]
  float v149; // [rsp+D0h] [rbp-C0h]
  __int64 **v150; // [rsp+D8h] [rbp-B8h]
  float v151; // [rsp+E0h] [rbp-B0h]
  _QWORD *v152; // [rsp+E0h] [rbp-B0h]
  unsigned __int64 v153; // [rsp+E8h] [rbp-A8h]
  char v154; // [rsp+E8h] [rbp-A8h]
  __int64 v157; // [rsp+100h] [rbp-90h] BYREF
  char *v158; // [rsp+108h] [rbp-88h]
  __int64 v159; // [rsp+110h] [rbp-80h]
  int v160; // [rsp+118h] [rbp-78h]
  char v161; // [rsp+11Ch] [rbp-74h]
  char v162; // [rsp+120h] [rbp-70h] BYREF

  v7 = a1;
  v8 = *(_QWORD *)(a1[4] + 32LL);
  v9 = *(_QWORD *)(v8 + 104);
  v134 = *(_QWORD *)(v8 + 96) & 0xFFFFFFFFFFFFFFF8LL;
  v10 = *(unsigned int *)(a2 + 8);
  v133 = v9 & 0xFFFFFFFFFFFFFFF9LL;
  if ( !(_DWORD)v10 )
  {
    v135 = 0;
    v11 = 0.0;
    v130 = 0;
    v12 = 0.0;
    v13 = 0.0;
    v132 = 0;
    v136 = 0;
    LODWORD(v131) = 0;
    v138 = 0.0;
    v140 = 0.0;
    v142 = 0.0;
    v143 = 0.0;
    v139 = 0.0;
    v141 = 0.0;
    v137 = 0.0;
    goto LABEL_3;
  }
  v136 = 0;
  v143 = 0.0;
  v142 = 0.0;
  v130 = 0;
  v145 = *(__int64 ****)a2;
  v126 = *(_QWORD *)a2 + 8 * v10;
  v127 = (__int64)(a1 + 29);
  v135 = 0x7FFFFFFFFFFFFFFFLL;
  v131 = 0;
  v132 = 0;
  v137 = 0.0;
  v138 = 0.0;
  v139 = 0.0;
  v140 = 0.0;
  v141 = 0.0;
  do
  {
    v26 = v136;
    v27 = *((_DWORD *)*v145 + 28);
    v28 = **v145;
    v150 = *v145;
    v29 = *(unsigned int *)(*(_QWORD *)(v7[2] + 920LL) + 8LL * (v27 & 0x7FFFFFFF));
    v143 = fmaxf(*((float *)*v145 + 29), v143);
    if ( v136 < v29 )
      v26 = *(unsigned int *)(*(_QWORD *)(v7[2] + 920LL) + 8LL * (v27 & 0x7FFFFFFF));
    v136 = v26;
    if ( v135 <= v29 )
      v29 = v135;
    v30 = *v28;
    v135 = v29;
    if ( (*(_DWORD *)((v30 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v30 >> 1) & 3) >= (*(_DWORD *)((v134 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                           | (unsigned int)(v134 >> 1)
                                                                                           & 3) )
      v30 = v134;
    v134 = v30;
    v31 = v28[3 * *((unsigned int *)*v145 + 2) - 2];
    v32 = *((unsigned int *)v7 + 64);
    if ( (*(_DWORD *)((v31 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v31 >> 1) & 3) <= (*(_DWORD *)((v133 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                           | (unsigned int)(v133 >> 1)
                                                                                           & 3) )
      v31 = v133;
    v133 = v31;
    if ( !(_DWORD)v32 )
    {
      ++v7[29];
LABEL_57:
      v32 = (unsigned int)(2 * v32);
      sub_35931D0(v127, v32);
      v33 = *((_DWORD *)v7 + 64);
      if ( !v33 )
        goto LABEL_201;
      v34 = v33 - 1;
      v35 = v7[30];
      v36 = v34 & (37 * v27);
      v37 = *((_DWORD *)v7 + 62) + 1;
      v24 = (int *)(v35 + ((unsigned __int64)v36 << 6));
      v38 = *v24;
      if ( v27 != *v24 )
      {
        v32 = 1;
        v39 = 0;
        while ( v38 != -1 )
        {
          if ( !v39 && v38 == -2 )
            v39 = v24;
          v36 = v34 & (v32 + v36);
          v24 = (int *)(v35 + ((unsigned __int64)v36 << 6));
          v38 = *v24;
          if ( v27 == *v24 )
            goto LABEL_91;
          v32 = (unsigned int)(v32 + 1);
        }
LABEL_61:
        if ( v39 )
          v24 = v39;
        goto LABEL_91;
      }
      goto LABEL_91;
    }
    v20 = 1;
    v21 = v7[30];
    v22 = 0;
    v23 = (v32 - 1) & (37 * v27);
    v24 = (int *)(v21 + ((unsigned __int64)v23 << 6));
    v25 = *v24;
    if ( v27 == *v24 )
      goto LABEL_46;
    while ( v25 != -1 )
    {
      if ( v25 == -2 && !v22 )
        v22 = v24;
      v23 = (v32 - 1) & (v20 + v23);
      v24 = (int *)(v21 + ((unsigned __int64)v23 << 6));
      v25 = *v24;
      if ( v27 == *v24 )
        goto LABEL_46;
      ++v20;
    }
    if ( v22 )
      v24 = v22;
    v62 = *((_DWORD *)v7 + 62);
    ++v7[29];
    v37 = v62 + 1;
    if ( 4 * v37 >= (unsigned int)(3 * v32) )
      goto LABEL_57;
    if ( (int)v32 - *((_DWORD *)v7 + 63) - v37 <= (unsigned int)v32 >> 3 )
    {
      sub_35931D0(v127, v32);
      v108 = *((_DWORD *)v7 + 64);
      if ( !v108 )
      {
LABEL_201:
        ++*((_DWORD *)v7 + 62);
        BUG();
      }
      v109 = v108 - 1;
      v110 = v7[30];
      v39 = 0;
      v32 = 1;
      v37 = *((_DWORD *)v7 + 62) + 1;
      v111 = v109 & (unsigned int)(37 * v27);
      v24 = (int *)(v110 + (v111 << 6));
      v112 = *v24;
      if ( v27 != *v24 )
      {
        while ( v112 != -1 )
        {
          if ( !v39 && v112 == -2 )
            v39 = v24;
          LODWORD(v111) = v109 & (v32 + v111);
          v24 = (int *)(v110 + ((unsigned __int64)(unsigned int)v111 << 6));
          v112 = *v24;
          if ( v27 == *v24 )
            goto LABEL_91;
          v32 = (unsigned int)(v32 + 1);
        }
        goto LABEL_61;
      }
    }
LABEL_91:
    *((_DWORD *)v7 + 62) = v37;
    if ( *v24 != -1 )
      --*((_DWORD *)v7 + 63);
    *v24 = v27;
    *((_QWORD *)v24 + 1) = 0;
    *((_QWORD *)v24 + 2) = 0;
    *((_QWORD *)v24 + 3) = 0;
    *((_QWORD *)v24 + 4) = 0;
    *((_QWORD *)v24 + 5) = 0;
    *((_QWORD *)v24 + 6) = 0;
    v24[14] = 0;
    *((_BYTE *)v24 + 60) = 0;
    v158 = &v162;
    v63 = v7[1];
    v159 = 8;
    v160 = 0;
    v161 = 1;
    v64 = *(_QWORD *)(v63 + 16);
    v157 = 0;
    v65 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v64 + 200LL))(v64);
    v69 = v7[6];
    v125 = (_QWORD *)v65;
    v70 = *((unsigned int *)v150 + 28);
    if ( (int)v70 < 0 )
    {
      v71 = *(_QWORD *)(*(_QWORD *)(v69 + 56) + 16 * (v70 & 0x7FFFFFFF) + 8);
    }
    else
    {
      v69 = *(_QWORD *)(v69 + 304);
      v71 = *(_QWORD *)(v69 + 8 * v70);
    }
    if ( v71 )
    {
      if ( (*(_BYTE *)(v71 + 4) & 8) == 0 )
      {
LABEL_97:
        v152 = v7;
        v72 = (double *)v24;
        while ( 2 )
        {
          while ( 2 )
          {
            v73 = *(_QWORD *)(v71 + 16);
            do
            {
              v71 = *(_QWORD *)(v71 + 32);
              if ( !v71 )
              {
                ++*((_QWORD *)v72 + 6);
                if ( !v161 )
                  goto LABEL_118;
                goto LABEL_102;
              }
            }
            while ( (*(_BYTE *)(v71 + 4) & 8) != 0 || v73 == *(_QWORD *)(v71 + 16) );
            ++*((_QWORD *)v72 + 6);
            if ( !v161 )
              goto LABEL_118;
LABEL_102:
            v74 = v158;
            v66 = HIDWORD(v159);
            v69 = (__int64)&v158[8 * HIDWORD(v159)];
            if ( v158 != (char *)v69 )
            {
              while ( v73 != *(_QWORD *)v74 )
              {
                v74 += 8;
                if ( (char *)v69 == v74 )
                  goto LABEL_153;
              }
LABEL_106:
              if ( v71 )
                continue;
              goto LABEL_107;
            }
            break;
          }
LABEL_153:
          if ( HIDWORD(v159) >= (unsigned int)v159 )
          {
LABEL_118:
            v32 = v73;
            sub_C8CC70((__int64)&v157, v73, v69, v66, v67, v68);
            if ( !(_BYTE)v69 )
              goto LABEL_106;
            v80 = *(_WORD *)(v73 + 68);
            if ( v80 == 20 )
            {
LABEL_155:
              v105 = *(_DWORD **)(v73 + 32);
              if ( v105[12] == v105[2] )
              {
                v69 = *v105 >> 8;
                LOWORD(v69) = v69 & 0xFFF;
                if ( (_WORD)v69 == ((v105[10] >> 8) & 0xFFF) )
                  goto LABEL_106;
              }
LABEL_121:
              v81 = sub_2E89D80(v73, *((_DWORD *)v150 + 28), 0);
              v82 = v81;
              v154 = HIBYTE(v81);
              v83 = (__int64 *)v152[25];
              v84 = sub_2E39EA0(v83, *(_QWORD *)(v73 + 24));
              if ( v84 < 0 )
                v148 = (double)(int)(v84 & 1 | ((unsigned __int64)v84 >> 1))
                     + (double)(int)(v84 & 1 | ((unsigned __int64)v84 >> 1));
              else
                v148 = (double)(int)v84;
              v85 = sub_2E3A080((__int64)v83);
              if ( v85 < 0 )
              {
                v69 = v85 & 1 | ((unsigned __int64)v85 >> 1);
                v86 = (double)(int)v69 + (double)(int)v69;
              }
              else
              {
                v86 = (double)(int)v85;
              }
              v87 = v72[2];
              v88 = v72[3];
              v89 = v148 / v86;
              v90 = v72[1];
              v149 = v89;
              *((float *)v72 + 14) = fmaxf(*((float *)v72 + 14), v89);
              v91 = (float)(0.0 * v89);
              if ( v82 == 1 && !v154 )
              {
                v72[1] = v90 + v89;
              }
              else
              {
                v72[1] = v90 + v91;
                if ( !v82 && v154 )
                {
                  v72[2] = v89 + v87;
                  goto LABEL_130;
                }
              }
              v72[2] = v91 + v87;
              v91 = (float)((float)(unsigned __int8)(v154 & v82) * v89);
LABEL_130:
              v72[3] = v91 + v88;
              v92 = v152[26];
              v93 = *(_QWORD *)(v73 + 24);
              v66 = *(unsigned int *)(v92 + 24);
              v32 = *(_QWORD *)(v92 + 8);
              v144 = v93;
              if ( (_DWORD)v66 )
              {
                v66 = (unsigned int)(v66 - 1);
                v69 = (unsigned int)v66 & (((unsigned int)v93 >> 9) ^ ((unsigned int)v93 >> 4));
                v94 = (__int64 *)(v32 + 16 * v69);
                v95 = *v94;
                if ( v93 != *v94 )
                {
                  for ( i = 1; ; i = v107 )
                  {
                    if ( v95 == -4096 )
                      goto LABEL_142;
                    v107 = i + 1;
                    v69 = (unsigned int)v66 & (i + (_DWORD)v69);
                    v94 = (__int64 *)(v32 + 16LL * (unsigned int)v69);
                    v95 = *v94;
                    if ( v144 == *v94 )
                      break;
                  }
                }
                v96 = v94[1];
                if ( v96 )
                {
                  v69 = *(_QWORD *)(v144 + 112);
                  if ( v69 != v69 + 8LL * *(unsigned int *)(v144 + 120) )
                  {
                    v147 = v71;
                    v97 = *(__int64 **)(v144 + 112);
                    v146 = v72;
                    v98 = v96 + 56;
                    v99 = v94[1];
                    v100 = (__int64 *)(v69 + 8LL * *(unsigned int *)(v144 + 120));
                    while ( 1 )
                    {
                      v32 = *v97;
                      if ( *(_BYTE *)(v99 + 84) )
                      {
                        v101 = *(_QWORD **)(v99 + 64);
                        v69 = (__int64)&v101[*(unsigned int *)(v99 + 76)];
                        if ( v101 == (_QWORD *)v69 )
                          goto LABEL_147;
                        while ( v32 != *v101 )
                        {
                          if ( (_QWORD *)v69 == ++v101 )
                            goto LABEL_147;
                        }
                      }
                      else if ( !sub_C8CA60(v98, v32) )
                      {
LABEL_147:
                        v71 = v147;
                        v72 = v146;
                        if ( v154 )
                        {
                          v102 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v152[4] + 32LL) + 152LL)
                                           + 16LL * *(unsigned int *)(v144 + 24)
                                           + 8);
                          v103 = ((v102 >> 1) & 3) != 0
                               ? v102 & 0xFFFFFFFFFFFFFFF8LL | (2LL * (int)(((v102 >> 1) & 3) - 1))
                               : *(_QWORD *)(v102 & 0xFFFFFFFFFFFFFFF8LL) & 0xFFFFFFFFFFFFFFF8LL | 6;
                          v32 = v103;
                          v104 = (__int64 *)sub_2E09D00((__int64 *)v150, v103);
                          v66 = 3LL * *((unsigned int *)v150 + 2);
                          v69 = (__int64)&(*v150)[3 * *((unsigned int *)v150 + 2)];
                          if ( v104 != (__int64 *)v69 )
                          {
                            v69 = *(_DWORD *)((*v104 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((*v104 >> 1) & 3);
                            v66 = v103 & 0xFFFFFFFFFFFFFFF8LL;
                            if ( (unsigned int)v69 <= (*(_DWORD *)((v103 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                     | (unsigned int)(v103 >> 1) & 3) )
                              v146[4] = v149 + v146[4];
                          }
                        }
                        break;
                      }
                      if ( v100 == ++v97 )
                      {
                        v71 = v147;
                        v72 = v146;
                        break;
                      }
                    }
                  }
                }
              }
LABEL_142:
              if ( *(_WORD *)(v73 + 68) != 20 )
                goto LABEL_106;
              v32 = *((unsigned int *)v150 + 28);
              if ( !(unsigned int)sub_34C7490(v73, v32, v125, v152[6]) )
                goto LABEL_106;
              v72[5] = v149 + v72[5];
              if ( v71 )
                continue;
LABEL_107:
              v24 = (int *)v72;
              v7 = v152;
              goto LABEL_108;
            }
          }
          else
          {
            v66 = (unsigned int)++HIDWORD(v159);
            *(_QWORD *)v69 = v73;
            v80 = *(_WORD *)(v73 + 68);
            ++v157;
            if ( v80 == 20 )
              goto LABEL_155;
          }
          break;
        }
        if ( v80 == 10 )
          goto LABEL_106;
        goto LABEL_121;
      }
      while ( 1 )
      {
        v71 = *(_QWORD *)(v71 + 32);
        if ( !v71 )
          break;
        if ( (*(_BYTE *)(v71 + 4) & 8) == 0 )
          goto LABEL_97;
      }
    }
LABEL_108:
    v75 = 0;
    v76 = *(_QWORD *)(v7[1] + 16LL);
    v77 = *(__int64 (**)())(*(_QWORD *)v76 + 128LL);
    if ( v77 != sub_2DAC790 )
      v75 = ((__int64 (__fastcall *)(__int64, __int64, __int64, _QWORD))v77)(v76, v32, v69, 0);
    v78 = sub_34C7590((__int64)v150, v7[4], v7[5], v75, v67, v68);
    v79 = v161 == 0;
    *((_BYTE *)v24 + 60) = v78;
    if ( v79 )
      _libc_free((unsigned __int64)v158);
    v27 = *((_DWORD *)v150 + 28);
LABEL_46:
    v132 += sub_300C040((_QWORD *)v7[5], v27);
    v141 = v141 + *((double *)v24 + 1);
    v140 = v140 + *((double *)v24 + 2);
    ++v145;
    v130 += *((unsigned __int8 *)v24 + 60);
    v131 += *((_QWORD *)v24 + 6);
    v142 = fmaxf(*((float *)v24 + 14), v142);
    v139 = v139 + *((double *)v24 + 3);
    v138 = v138 + *((double *)v24 + 4);
    v137 = v137 + *((double *)v24 + 5);
  }
  while ( (__int64 ***)v126 != v145 );
  if ( *(_DWORD *)(a2 + 8) )
  {
    v40 = (__int64 *)v7[25];
    v153 = v134 & 0xFFFFFFFFFFFFFFF8LL;
    v41 = *(_QWORD *)((v134 & 0xFFFFFFFFFFFFFFF8LL) + 16);
    v42 = (v134 >> 1) & 3;
    if ( v41 )
    {
      v43 = *(_QWORD *)(v41 + 24);
    }
    else
    {
      v113 = *(_QWORD *)(v7[4] + 32LL);
      v114 = *(_QWORD **)(v113 + 296);
      v115 = *(unsigned int *)(v113 + 304);
      if ( v115 )
      {
        while ( 1 )
        {
          v116 = v115 >> 1;
          v117 = &v114[2 * (v115 >> 1)];
          if ( ((unsigned int)v42 | *(_DWORD *)(v153 + 24)) >= (*(_DWORD *)((*v117 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                              | (unsigned int)(*v117 >> 1) & 3) )
          {
            v114 = v117 + 2;
            v116 = v115 - v116 - 1;
          }
          if ( v116 <= 0 )
            break;
          v115 = v116;
        }
      }
      v43 = *(v114 - 1);
    }
    v44 = sub_2E39EA0(v40, v43);
    if ( v44 < 0 )
      v45 = (double)(int)(v44 & 1 | ((unsigned __int64)v44 >> 1))
          + (double)(int)(v44 & 1 | ((unsigned __int64)v44 >> 1));
    else
      v45 = (double)(int)v44;
    v46 = sub_2E3A080((__int64)v40);
    if ( v46 < 0 )
      v47 = (double)(int)(v46 & 1 | ((unsigned __int64)v46 >> 1))
          + (double)(int)(v46 & 1 | ((unsigned __int64)v46 >> 1));
    else
      v47 = (double)(int)v46;
    v48 = v45 / v47;
    v49 = *(_QWORD *)(v7[4] + 32LL);
    v50 = v133 & 0xFFFFFFFFFFFFFFF8LL;
    v51 = *(_QWORD *)(v49 + 96) & 0xFFFFFFFFFFFFFFF8LL;
    v52 = (v133 >> 1) & 3;
    if ( (*(_DWORD *)((v133 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)v52) >= *(_DWORD *)(v51 + 24) )
    {
      LODWORD(v52) = 0;
      v50 = *(_QWORD *)v51 & 0xFFFFFFFFFFFFFFF8LL;
    }
    v53 = *(_QWORD *)(v50 + 16);
    v54 = v7[25];
    if ( v53 )
    {
      v55 = *(_QWORD *)(v53 + 24);
    }
    else
    {
      v119 = *(unsigned int *)(v49 + 304);
      v120 = *(_QWORD **)(v49 + 296);
      if ( *(_DWORD *)(v49 + 304) )
      {
        while ( 1 )
        {
          v121 = v119 >> 1;
          v122 = &v120[2 * (v119 >> 1)];
          if ( ((unsigned int)v52 | *(_DWORD *)(v50 + 24)) >= (*(_DWORD *)((*v122 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                             | (unsigned int)(*v122 >> 1) & 3) )
          {
            v120 = v122 + 2;
            v121 = v119 - v121 - 1;
          }
          if ( v121 <= 0 )
            break;
          v119 = v121;
        }
      }
      v55 = *(v120 - 1);
    }
    v151 = v48;
    v56 = sub_2E39EA0((__int64 *)v7[25], v55);
    v13 = v151;
    if ( v56 < 0 )
      v57 = (double)(int)(v56 & 1 | ((unsigned __int64)v56 >> 1))
          + (double)(int)(v56 & 1 | ((unsigned __int64)v56 >> 1));
    else
      v57 = (double)(int)v56;
    v58 = sub_2E3A080(v54);
    if ( v58 < 0 )
      v59 = (double)(int)(v58 & 1 | ((unsigned __int64)v58 >> 1))
          + (double)(int)(v58 & 1 | ((unsigned __int64)v58 >> 1));
    else
      v59 = (double)(int)v58;
    v60 = *(_DWORD *)(v153 + 24) | v42;
    v61 = (int)((v52 | *(_DWORD *)(v50 + 24)) - v60);
    v12 = v57 / v59;
    if ( v61 < 0 )
    {
      v118 = (((unsigned __int8)v52 | *(_BYTE *)(v50 + 24)) - (_BYTE)v60) & 1
           | ((unsigned __int64)(int)((v52 | *(_DWORD *)(v50 + 24)) - v60) >> 1);
      v11 = (float)(int)v118 + (float)(int)v118;
    }
    else
    {
      v11 = (float)(int)v61;
    }
  }
  else
  {
    v11 = 0.0;
    v12 = 0.0;
    v13 = 0.0;
  }
LABEL_3:
  *(_QWORD *)(**(_QWORD **)(v7[24] + 24LL) + 8 * a4) = 1;
  if ( (v7[27] & 1) == 0 )
    **a3 = fmaxf(1.0, **a3);
  *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v7[24] + 24LL) + 8LL) + 8 * a4) = *(_DWORD *)(a2 + 8) == 0;
  if ( (v7[27] & 2) == 0 )
    (*a3)[1] = fmaxf((float)(*(_DWORD *)(a2 + 8) == 0), (*a3)[1]);
  *(float *)(*(_QWORD *)(*(_QWORD *)(v7[24] + 24LL) + 16LL) + 4 * a4) = a7;
  if ( (v7[27] & 4) == 0 )
    (*a3)[2] = fmaxf(a7, (*a3)[2]);
  *(float *)(*(_QWORD *)(*(_QWORD *)(v7[24] + 24LL) + 24LL) + 4 * a4) = (float)v132;
  if ( (v7[27] & 8) == 0 )
    (*a3)[3] = fmaxf((float)v132, (*a3)[3]);
  *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v7[24] + 24LL) + 32LL) + 8 * a4) = a5;
  if ( (v7[27] & 0x10) == 0 )
    (*a3)[4] = fmaxf((float)(int)a5, (*a3)[4]);
  *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v7[24] + 24LL) + 40LL) + 8 * a4) = a6;
  if ( (v7[27] & 0x20) == 0 )
    (*a3)[5] = fmaxf((float)(int)a6, (*a3)[5]);
  *(float *)(*(_QWORD *)(*(_QWORD *)(v7[24] + 24LL) + 48LL) + 4 * a4) = (float)v130;
  if ( (v7[27] & 0x40) == 0 )
    (*a3)[6] = fmaxf((float)v130, (*a3)[6]);
  *(float *)(*(_QWORD *)(*(_QWORD *)(v7[24] + 24LL) + 56LL) + 4 * a4) = (float)(int)v131;
  if ( *((char *)v7 + 216) >= 0 )
    (*a3)[7] = fmaxf((float)(int)v131, (*a3)[7]);
  v14 = v141;
  *(float *)(*(_QWORD *)(*(_QWORD *)(v7[24] + 24LL) + 64LL) + 4 * a4) = v14;
  if ( (*((_BYTE *)v7 + 217) & 1) == 0 )
    (*a3)[8] = fmaxf(v14, (*a3)[8]);
  v15 = v140;
  *(float *)(*(_QWORD *)(*(_QWORD *)(v7[24] + 24LL) + 72LL) + 4 * a4) = v15;
  if ( (*((_BYTE *)v7 + 217) & 2) == 0 )
    (*a3)[9] = fmaxf(v15, (*a3)[9]);
  v16 = v139;
  *(float *)(*(_QWORD *)(*(_QWORD *)(v7[24] + 24LL) + 80LL) + 4 * a4) = v16;
  if ( (*((_BYTE *)v7 + 217) & 4) == 0 )
    (*a3)[10] = fmaxf(v16, (*a3)[10]);
  v17 = v138;
  *(float *)(*(_QWORD *)(*(_QWORD *)(v7[24] + 24LL) + 88LL) + 4 * a4) = v17;
  if ( (*((_BYTE *)v7 + 217) & 8) == 0 )
    (*a3)[11] = fmaxf(v17, (*a3)[11]);
  v18 = v137;
  *(float *)(*(_QWORD *)(*(_QWORD *)(v7[24] + 24LL) + 96LL) + 4 * a4) = v18;
  if ( (*((_BYTE *)v7 + 217) & 0x10) == 0 )
    (*a3)[12] = fmaxf(v18, (*a3)[12]);
  *(float *)(*(_QWORD *)(*(_QWORD *)(v7[24] + 24LL) + 104LL) + 4 * a4) = v13;
  if ( (*((_BYTE *)v7 + 217) & 0x20) == 0 )
    (*a3)[13] = fmaxf(v13, (*a3)[13]);
  *(float *)(*(_QWORD *)(*(_QWORD *)(v7[24] + 24LL) + 112LL) + 4 * a4) = v12;
  if ( (*((_BYTE *)v7 + 217) & 0x40) == 0 )
    (*a3)[14] = fmaxf(v12, (*a3)[14]);
  *(float *)(*(_QWORD *)(*(_QWORD *)(v7[24] + 24LL) + 120LL) + 4 * a4) = v142;
  if ( *((char *)v7 + 217) >= 0 )
    (*a3)[15] = fmaxf(v142, (*a3)[15]);
  *(float *)(*(_QWORD *)(*(_QWORD *)(v7[24] + 24LL) + 128LL) + 4 * a4) = v11;
  if ( (*((_BYTE *)v7 + 218) & 1) == 0 )
    (*a3)[16] = fmaxf(v11, (*a3)[16]);
  *(float *)(*(_QWORD *)(*(_QWORD *)(v7[24] + 24LL) + 136LL) + 4 * a4) = v143;
  if ( (*((_BYTE *)v7 + 218) & 2) == 0 )
    (*a3)[17] = fmaxf(v143, (*a3)[17]);
  *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v7[24] + 24LL) + 144LL) + 8 * a4) = v136;
  if ( (*((_BYTE *)v7 + 218) & 4) == 0 )
    (*a3)[18] = fmaxf((float)(int)v136, (*a3)[18]);
  result = *(float **)(*(_QWORD *)(v7[24] + 24LL) + 152LL);
  *(_QWORD *)&result[2 * a4] = v135;
  if ( (*((_BYTE *)v7 + 218) & 8) == 0 )
  {
    result = *a3;
    (*a3)[19] = fmaxf((float)(int)v135, (*a3)[19]);
  }
  return result;
}
