// Function: sub_2C9F570
// Address: 0x2c9f570
//
void __fastcall sub_2C9F570(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  unsigned __int64 v7; // r12
  __int64 *v9; // rax
  __int64 *v10; // r11
  __int64 v11; // r12
  __int64 v12; // r12
  unsigned int v13; // r10d
  __int64 *v14; // r15
  unsigned int v15; // r9d
  _QWORD *v16; // rax
  __int64 v17; // rdi
  __int64 *v18; // rax
  __int64 v19; // rbx
  __int64 v20; // rcx
  __int64 v21; // r8
  unsigned int v22; // r10d
  _QWORD *v23; // rdx
  __int64 v24; // rsi
  int v25; // edi
  __int64 v26; // rsi
  __int64 v27; // rax
  unsigned int v28; // ecx
  __int64 *v29; // rdx
  __int64 v30; // r8
  __int64 *v31; // rax
  unsigned int v32; // r10d
  __int64 *v33; // r11
  unsigned __int64 *v34; // r14
  unsigned __int64 *v35; // rax
  __int64 *v36; // r11
  unsigned __int64 *v37; // r12
  __int64 v38; // r13
  unsigned int i; // r15d
  __int64 v40; // rax
  _QWORD **v41; // rdx
  unsigned __int64 *v42; // rdi
  _BYTE *v43; // rsi
  unsigned int v44; // r12d
  __int64 v45; // rax
  unsigned int v46; // ebx
  unsigned int v47; // r13d
  __int64 v48; // r15
  __int64 v49; // rdx
  __int64 v50; // rax
  unsigned __int64 *v51; // rdi
  _BYTE *v52; // rsi
  unsigned __int64 v53; // rdx
  unsigned __int64 v54; // rsi
  _BYTE *v55; // rsi
  unsigned int v56; // esi
  unsigned __int64 *v57; // rdx
  __int64 v58; // r8
  unsigned int v59; // eax
  unsigned __int64 **v60; // rcx
  unsigned __int64 *v61; // rdi
  unsigned __int64 *v62; // rax
  unsigned __int64 v63; // rdi
  __int64 v64; // r8
  __int64 v65; // rcx
  __int64 *v66; // rdx
  int v67; // ecx
  unsigned int v68; // esi
  __int64 *v69; // rax
  __int64 v70; // r10
  __int64 *v71; // rax
  unsigned __int64 v72; // r11
  __int64 *v73; // r11
  __int64 *v74; // r15
  unsigned int v75; // r11d
  _QWORD *v76; // r14
  unsigned int v77; // r9d
  int v78; // r10d
  unsigned int v79; // edi
  _QWORD *v80; // rdx
  _QWORD *v81; // rax
  _QWORD *v82; // rcx
  int v83; // ebx
  char *v84; // r13
  _QWORD *v85; // r12
  unsigned int v86; // r8d
  _QWORD *v87; // rdx
  __int64 v88; // rdi
  __int64 v89; // rcx
  __int64 v90; // rax
  _QWORD *v91; // rcx
  int v92; // edx
  unsigned int v93; // eax
  __int64 v94; // rdi
  int v95; // r9d
  _QWORD *v96; // r10
  int v97; // r9d
  unsigned int v98; // eax
  __int64 v99; // rdi
  int v100; // ecx
  int v101; // r10d
  unsigned int v102; // r14d
  __int64 v103; // rsi
  _QWORD *v104; // rax
  unsigned int v105; // edx
  __int64 v106; // rbx
  int v107; // r8d
  _QWORD *v108; // rdi
  _QWORD *v109; // r9
  unsigned int v110; // ebx
  int v111; // edx
  _QWORD *v112; // rsi
  int v113; // r11d
  unsigned __int64 **v114; // rbx
  int v115; // eax
  int v116; // ecx
  int v117; // edx
  unsigned __int64 *v118; // rax
  int v119; // eax
  unsigned int v120; // r10d
  _QWORD *v121; // r12
  int v122; // eax
  int v123; // ebx
  int v124; // r11d
  unsigned __int64 **v125; // r10
  int v126; // r9d
  __int64 v127; // [rsp+8h] [rbp-F8h]
  __int64 v128; // [rsp+8h] [rbp-F8h]
  __int64 v130; // [rsp+20h] [rbp-E0h]
  int v131; // [rsp+20h] [rbp-E0h]
  __int64 v132; // [rsp+20h] [rbp-E0h]
  unsigned int v134; // [rsp+30h] [rbp-D0h]
  int v135; // [rsp+38h] [rbp-C8h]
  unsigned int v136; // [rsp+38h] [rbp-C8h]
  __int64 *v138; // [rsp+48h] [rbp-B8h]
  int v139; // [rsp+48h] [rbp-B8h]
  __int64 *v140; // [rsp+50h] [rbp-B0h]
  unsigned int v141; // [rsp+50h] [rbp-B0h]
  unsigned int v142; // [rsp+50h] [rbp-B0h]
  unsigned int v143; // [rsp+50h] [rbp-B0h]
  _QWORD *v145; // [rsp+58h] [rbp-A8h]
  __int64 v146; // [rsp+60h] [rbp-A0h]
  __int64 *v147; // [rsp+60h] [rbp-A0h]
  __int64 *v148; // [rsp+60h] [rbp-A0h]
  _QWORD *v149; // [rsp+60h] [rbp-A0h]
  unsigned __int64 v150; // [rsp+60h] [rbp-A0h]
  _QWORD *v151; // [rsp+60h] [rbp-A0h]
  _QWORD *v152; // [rsp+60h] [rbp-A0h]
  _QWORD *v153; // [rsp+60h] [rbp-A0h]
  _QWORD *v154; // [rsp+60h] [rbp-A0h]
  __int64 v155; // [rsp+60h] [rbp-A0h]
  unsigned int v156; // [rsp+60h] [rbp-A0h]
  unsigned int v157; // [rsp+60h] [rbp-A0h]
  __int64 *v158; // [rsp+60h] [rbp-A0h]
  __int64 v159; // [rsp+68h] [rbp-98h] BYREF
  char v160; // [rsp+77h] [rbp-89h] BYREF
  unsigned __int64 *v161; // [rsp+78h] [rbp-88h] BYREF
  unsigned __int64 **v162; // [rsp+80h] [rbp-80h] BYREF
  unsigned __int64 **v163; // [rsp+88h] [rbp-78h] BYREF
  __int64 v164; // [rsp+90h] [rbp-70h] BYREF
  __int64 v165; // [rsp+98h] [rbp-68h] BYREF
  unsigned __int64 **v166; // [rsp+A0h] [rbp-60h] BYREF
  unsigned __int64 **v167; // [rsp+A8h] [rbp-58h] BYREF
  __int64 v168; // [rsp+B0h] [rbp-50h] BYREF
  __int64 v169; // [rsp+B8h] [rbp-48h]
  __int64 v170; // [rsp+C0h] [rbp-40h]
  unsigned int v171; // [rsp+C8h] [rbp-38h]

  v6 = a3[1];
  v159 = a2;
  v7 = v6 - *a3;
  if ( v6 == *a3 )
  {
    v10 = 0;
    v168 = 0;
    v13 = 0;
    v169 = 0;
    v170 = 0;
    v171 = 0;
    goto LABEL_18;
  }
  if ( v7 > 0x7FFFFFFFFFFFFFF8LL )
    sub_4261EA(a1, *a3, a3);
  v9 = (__int64 *)sub_22077B0(v7);
  v10 = v9;
  v11 = a3[1] - *a3;
  if ( *a3 != a3[1] )
    v10 = (__int64 *)memmove(v9, (const void *)*a3, a3[1] - *a3);
  v12 = v11 >> 3;
  v168 = 0;
  v169 = 0;
  v13 = v12;
  v170 = 0;
  v171 = 0;
  if ( (_DWORD)v12 )
  {
    v14 = v10;
    v140 = v10;
    v138 = v10;
    v146 = (__int64)&v10[(unsigned int)(v12 - 1) + 1];
    while ( 1 )
    {
      v19 = *v14;
      v166 = 0;
      v167 = 0;
      sub_2C9EEF0(a1, *(unsigned int **)v19, *(_QWORD ***)(v19 + 8), (__int64 *)&v166, (__int64 *)&v167, a6, 0);
      v20 = (__int64)v166;
      v21 = (__int64)v167;
      if ( !v171 )
        break;
      v15 = (v171 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
      v16 = (_QWORD *)(v169 + 24LL * v15);
      v17 = *v16;
      if ( v19 == *v16 )
        goto LABEL_8;
      v131 = 1;
      v23 = 0;
      while ( 1 )
      {
        if ( v17 == -4096 )
        {
          if ( !v23 )
            v23 = v16;
          ++v168;
          v25 = v170 + 1;
          if ( 4 * ((int)v170 + 1) < 3 * v171 )
          {
            if ( v171 - HIDWORD(v170) - v25 <= v171 >> 3 )
            {
              v128 = (__int64)v167;
              v132 = (__int64)v166;
              sub_2C93960((__int64)&v168, v171);
              if ( !v171 )
                goto LABEL_223;
              v20 = v132;
              v101 = 1;
              v102 = (v171 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
              v21 = v128;
              v23 = (_QWORD *)(v169 + 24LL * v102);
              v103 = *v23;
              v25 = v170 + 1;
              v104 = 0;
              if ( v19 != *v23 )
              {
                while ( v103 != -4096 )
                {
                  if ( !v104 && v103 == -8192 )
                    v104 = v23;
                  v102 = (v171 - 1) & (v101 + v102);
                  v23 = (_QWORD *)(v169 + 24LL * v102);
                  v103 = *v23;
                  if ( v19 == *v23 )
                    goto LABEL_14;
                  ++v101;
                }
LABEL_131:
                if ( v104 )
                  v23 = v104;
              }
            }
LABEL_14:
            LODWORD(v170) = v25;
            if ( *v23 != -4096 )
              --HIDWORD(v170);
            *v23 = v19;
            v18 = v23 + 1;
            v23[1] = 0;
            v23[2] = 0;
            goto LABEL_9;
          }
LABEL_12:
          v127 = (__int64)v167;
          v130 = (__int64)v166;
          sub_2C93960((__int64)&v168, 2 * v171);
          if ( !v171 )
            goto LABEL_223;
          v20 = v130;
          v21 = v127;
          v22 = (v171 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
          v23 = (_QWORD *)(v169 + 24LL * v22);
          v24 = *v23;
          v25 = v170 + 1;
          if ( v19 != *v23 )
          {
            v124 = 1;
            v104 = 0;
            while ( v24 != -4096 )
            {
              if ( !v104 && v24 == -8192 )
                v104 = v23;
              v22 = (v171 - 1) & (v124 + v22);
              v23 = (_QWORD *)(v169 + 24LL * v22);
              v24 = *v23;
              if ( v19 == *v23 )
                goto LABEL_14;
              ++v124;
            }
            goto LABEL_131;
          }
          goto LABEL_14;
        }
        if ( v17 != -8192 || v23 )
          v16 = v23;
        v15 = (v171 - 1) & (v131 + v15);
        v17 = *(_QWORD *)(v169 + 24LL * v15);
        if ( v19 == v17 )
          break;
        ++v131;
        v23 = v16;
        v16 = (_QWORD *)(v169 + 24LL * v15);
      }
      v16 = (_QWORD *)(v169 + 24LL * v15);
LABEL_8:
      v18 = v16 + 1;
LABEL_9:
      *v18 = v20;
      ++v14;
      v18[1] = v21;
      if ( (__int64 *)v146 == v14 )
      {
        v73 = v138;
        v139 = 0;
        v74 = v73;
        v75 = v12;
        while ( 1 )
        {
          v76 = (_QWORD *)*v140;
          if ( !v171 )
            break;
          v77 = v171 - 1;
          v78 = 1;
          v79 = (v171 - 1) & (((unsigned int)v76 >> 9) ^ ((unsigned int)v76 >> 4));
          v80 = (_QWORD *)(v169 + 24LL * v79);
          v81 = 0;
          v82 = (_QWORD *)*v80;
          if ( v76 == (_QWORD *)*v80 )
            goto LABEL_76;
          while ( 1 )
          {
            if ( v82 == (_QWORD *)-4096LL )
            {
              if ( !v81 )
                v81 = v80;
              ++v168;
              v100 = v170 + 1;
              if ( 4 * ((int)v170 + 1) < 3 * v171 )
              {
                if ( v171 - HIDWORD(v170) - v100 > v171 >> 3 )
                {
LABEL_119:
                  LODWORD(v170) = v100;
                  if ( *v81 != -4096 )
                    --HIDWORD(v170);
                  *v81 = v76;
                  v81[1] = 0;
                  v81[2] = 0;
                  v155 = 0;
                  goto LABEL_77;
                }
                v157 = v75;
                sub_2C93960((__int64)&v168, v171);
                if ( v171 )
                {
                  v109 = 0;
                  v110 = (v171 - 1) & (((unsigned int)v76 >> 9) ^ ((unsigned int)v76 >> 4));
                  v75 = v157;
                  v111 = 1;
                  v100 = v170 + 1;
                  v81 = (_QWORD *)(v169 + 24LL * v110);
                  v112 = (_QWORD *)*v81;
                  if ( v76 != (_QWORD *)*v81 )
                  {
                    while ( v112 != (_QWORD *)-4096LL )
                    {
                      if ( !v109 && v112 == (_QWORD *)-8192LL )
                        v109 = v81;
                      v110 = (v171 - 1) & (v111 + v110);
                      v81 = (_QWORD *)(v169 + 24LL * v110);
                      v112 = (_QWORD *)*v81;
                      if ( v76 == (_QWORD *)*v81 )
                        goto LABEL_119;
                      ++v111;
                    }
                    if ( v109 )
                      v81 = v109;
                  }
                  goto LABEL_119;
                }
LABEL_223:
                LODWORD(v170) = v170 + 1;
                BUG();
              }
LABEL_135:
              v156 = v75;
              sub_2C93960((__int64)&v168, 2 * v171);
              if ( v171 )
              {
                v75 = v156;
                v105 = (v171 - 1) & (((unsigned int)v76 >> 9) ^ ((unsigned int)v76 >> 4));
                v100 = v170 + 1;
                v81 = (_QWORD *)(v169 + 24LL * v105);
                v106 = *v81;
                if ( v76 != (_QWORD *)*v81 )
                {
                  v107 = 1;
                  v108 = 0;
                  while ( v106 != -4096 )
                  {
                    if ( !v108 && v106 == -8192 )
                      v108 = v81;
                    v105 = (v171 - 1) & (v107 + v105);
                    v81 = (_QWORD *)(v169 + 24LL * v105);
                    v106 = *v81;
                    if ( v76 == (_QWORD *)*v81 )
                      goto LABEL_119;
                    ++v107;
                  }
                  if ( v108 )
                    v81 = v108;
                }
                goto LABEL_119;
              }
              goto LABEL_223;
            }
            if ( v81 || v82 != (_QWORD *)-8192LL )
              v80 = v81;
            v119 = v78 + 1;
            v120 = v79 + v78;
            v79 = v77 & v120;
            v121 = (_QWORD *)(v169 + 24LL * (v77 & v120));
            v82 = (_QWORD *)*v121;
            if ( v76 == (_QWORD *)*v121 )
              break;
            v78 = v119;
            v81 = v80;
            v80 = v121;
          }
          v80 = (_QWORD *)(v169 + 24LL * (v77 & v120));
LABEL_76:
          v155 = v80[1];
LABEL_77:
          ++v139;
          while ( 2 )
          {
            v83 = v139;
            if ( v75 == v139 )
            {
              v13 = v75;
              v10 = v74;
              *v140 = (__int64)v76;
              goto LABEL_18;
            }
LABEL_81:
            v84 = (char *)&v74[v83];
            v85 = *(_QWORD **)v84;
            if ( *(_QWORD *)(*(_QWORD *)v84 + 8LL) != *v76 )
              break;
            if ( !v171 )
            {
              ++v168;
              goto LABEL_98;
            }
            v86 = (v171 - 1) & (((unsigned int)v85 >> 4) ^ ((unsigned int)v85 >> 9));
            v87 = (_QWORD *)(v169 + 24LL * v86);
            v88 = *v87;
            if ( v85 != (_QWORD *)*v87 )
            {
              v135 = 1;
              v91 = 0;
              while ( v88 != -4096 )
              {
                if ( !v91 && v88 == -8192 )
                  v91 = v87;
                v86 = (v171 - 1) & (v135 + v86);
                v87 = (_QWORD *)(v169 + 24LL * v86);
                v88 = *v87;
                if ( v85 == (_QWORD *)*v87 )
                  goto LABEL_84;
                ++v135;
              }
              if ( !v91 )
                v91 = v87;
              ++v168;
              v92 = v170 + 1;
              if ( 4 * ((int)v170 + 1) >= 3 * v171 )
              {
LABEL_98:
                v136 = v75;
                sub_2C93960((__int64)&v168, 2 * v171);
                if ( !v171 )
                  goto LABEL_223;
                v75 = v136;
                v93 = (v171 - 1) & (((unsigned int)v85 >> 9) ^ ((unsigned int)v85 >> 4));
                v91 = (_QWORD *)(v169 + 24LL * v93);
                v94 = *v91;
                v92 = v170 + 1;
                if ( v85 != (_QWORD *)*v91 )
                {
                  v95 = 1;
                  v96 = 0;
                  while ( v94 != -4096 )
                  {
                    if ( !v96 && v94 == -8192 )
                      v96 = v91;
                    v93 = (v171 - 1) & (v95 + v93);
                    v91 = (_QWORD *)(v169 + 24LL * v93);
                    v94 = *v91;
                    if ( v85 == (_QWORD *)*v91 )
                      goto LABEL_94;
                    ++v95;
                  }
LABEL_102:
                  if ( v96 )
                    v91 = v96;
                }
              }
              else if ( v171 - HIDWORD(v170) - v92 <= v171 >> 3 )
              {
                v134 = v75;
                sub_2C93960((__int64)&v168, v171);
                if ( !v171 )
                  goto LABEL_223;
                v96 = 0;
                v75 = v134;
                v97 = 1;
                v98 = (v171 - 1) & (((unsigned int)v85 >> 4) ^ ((unsigned int)v85 >> 9));
                v91 = (_QWORD *)(v169 + 24LL * v98);
                v99 = *v91;
                v92 = v170 + 1;
                if ( v85 != (_QWORD *)*v91 )
                {
                  while ( v99 != -4096 )
                  {
                    if ( v99 == -8192 && !v96 )
                      v96 = v91;
                    v98 = (v171 - 1) & (v97 + v98);
                    v91 = (_QWORD *)(v169 + 24LL * v98);
                    v99 = *v91;
                    if ( v85 == (_QWORD *)*v91 )
                      goto LABEL_94;
                    ++v97;
                  }
                  goto LABEL_102;
                }
              }
LABEL_94:
              LODWORD(v170) = v92;
              if ( *v91 != -4096 )
                --HIDWORD(v170);
              *v91 = v85;
              v90 = 0;
              v91[1] = 0;
              v91[2] = 0;
              v89 = 0;
              goto LABEL_85;
            }
LABEL_84:
            v89 = v87[1];
            v90 = v87[2];
LABEL_85:
            if ( v155 == v90 )
            {
              *(_QWORD *)v84 = v76;
              v76 = v85;
              v155 = v89;
              continue;
            }
            break;
          }
          if ( v75 != ++v83 )
            goto LABEL_81;
          *v140++ = (__int64)v76;
        }
        ++v168;
        goto LABEL_135;
      }
    }
    ++v168;
    goto LABEL_12;
  }
LABEL_18:
  v26 = *(_QWORD *)(a4 + 8);
  v27 = *(unsigned int *)(a4 + 24);
  if ( (_DWORD)v27 )
  {
    v28 = (v27 - 1) & (((unsigned int)v159 >> 9) ^ ((unsigned int)v159 >> 4));
    v29 = (__int64 *)(v26 + 16LL * v28);
    v30 = *v29;
    if ( v159 == *v29 )
    {
LABEL_20:
      if ( v29 != (__int64 *)(v26 + 16 * v27) )
      {
        v141 = v13;
        v147 = v10;
        v31 = sub_2C93500(a4, &v159);
        v32 = v141;
        v33 = v147;
        v34 = (unsigned __int64 *)*v31;
        goto LABEL_22;
      }
    }
    else
    {
      v117 = 1;
      while ( v30 != -4096 )
      {
        v123 = v117 + 1;
        v28 = (v27 - 1) & (v117 + v28);
        v29 = (__int64 *)(v26 + 16LL * v28);
        v30 = *v29;
        if ( v159 == *v29 )
          goto LABEL_20;
        v117 = v123;
      }
    }
  }
  v143 = v13;
  v158 = v10;
  v118 = (unsigned __int64 *)sub_22077B0(0x18u);
  v33 = v158;
  v32 = v143;
  v34 = v118;
  if ( v118 )
  {
    *v118 = 0;
    v118[1] = 0;
    v118[2] = 0;
  }
LABEL_22:
  v142 = v32;
  v148 = v33;
  v35 = (unsigned __int64 *)sub_22077B0(0x18u);
  v36 = v148;
  v37 = v35;
  if ( v35 )
  {
    *v35 = 0;
    v35[1] = 0;
    v35[2] = 0;
    v161 = v35;
    if ( !v142 )
      goto LABEL_67;
  }
  else
  {
    v161 = 0;
    if ( !v142 )
    {
LABEL_57:
      v63 = *v34;
      if ( v34[1] == *v34 )
        goto LABEL_70;
      goto LABEL_58;
    }
  }
  v38 = a1;
  for ( i = v142; ; i = v44 )
  {
    v40 = *v148;
    v162 = 0;
    v163 = 0;
    v41 = *(_QWORD ***)(v40 + 8);
    v164 = v40;
    v160 = 0;
    sub_2C9EEF0(v38, *(unsigned int **)v40, v41, (__int64 *)&v162, (__int64 *)&v163, a6, &v160);
    v42 = v161;
    v43 = (_BYTE *)v161[1];
    if ( v43 == (_BYTE *)v161[2] )
    {
      sub_2C908A0((__int64)v161, v43, &v164);
    }
    else
    {
      if ( v43 )
      {
        *(_QWORD *)v43 = v164;
        v43 = (_BYTE *)v42[1];
      }
      v42[1] = (unsigned __int64)(v43 + 8);
    }
    v44 = 0;
    if ( i != 1 )
    {
      v45 = v38;
      v46 = 1;
      v47 = i;
      v48 = v45;
      while ( 1 )
      {
        while ( 1 )
        {
          v50 = v148[v46];
          v165 = v50;
          if ( *(_QWORD *)(v164 + 8) == *(_QWORD *)v50 )
          {
            sub_2C9EEF0(v48, *(unsigned int **)v50, *(_QWORD ***)(v50 + 8), (__int64 *)&v166, (__int64 *)&v167, a6, 0);
            v50 = v165;
            if ( v163 == v166 )
              break;
          }
          v49 = v44++;
          v148[v49] = v50;
LABEL_32:
          if ( ++v46 == v47 )
            goto LABEL_39;
        }
        v162 = v166;
        v51 = v161;
        v164 = v165;
        v52 = (_BYTE *)v161[1];
        v163 = v167;
        if ( v52 == (_BYTE *)v161[2] )
        {
          sub_2C908A0((__int64)v161, v52, &v165);
          goto LABEL_32;
        }
        if ( v52 )
        {
          *(_QWORD *)v52 = v165;
          v52 = (_BYTE *)v51[1];
        }
        ++v46;
        v51[1] = (unsigned __int64)(v52 + 8);
        if ( v46 == v47 )
        {
LABEL_39:
          v38 = v48;
          break;
        }
      }
    }
    v53 = *v161;
    v54 = v161[1] - *v161;
    if ( !v160 )
      break;
    if ( v54 > 8 )
      goto LABEL_47;
LABEL_42:
    if ( v161[1] != v53 )
      v161[1] = v53;
    if ( !v44 )
      goto LABEL_56;
LABEL_45:
    ;
  }
  if ( v54 <= 0x10 )
    goto LABEL_42;
LABEL_47:
  v55 = (_BYTE *)v34[1];
  if ( v55 == (_BYTE *)v34[2] )
  {
    sub_2C90710((__int64)v34, v55, &v161);
  }
  else
  {
    if ( v55 )
    {
      *(_QWORD *)v55 = v161;
      v55 = (_BYTE *)v34[1];
    }
    v34[1] = (unsigned __int64)(v55 + 8);
  }
  v56 = *(_DWORD *)(v38 + 256);
  if ( !v56 )
  {
    ++*(_QWORD *)(v38 + 232);
    v167 = 0;
LABEL_158:
    v56 *= 2;
    goto LABEL_159;
  }
  v57 = v161;
  v58 = *(_QWORD *)(v38 + 240);
  v59 = (v56 - 1) & (((unsigned int)v161 >> 9) ^ ((unsigned int)v161 >> 4));
  v60 = (unsigned __int64 **)(v58 + 8LL * v59);
  v61 = *v60;
  if ( v161 == *v60 )
    goto LABEL_53;
  v113 = 1;
  v114 = 0;
  while ( v61 != (unsigned __int64 *)-4096LL )
  {
    if ( v61 != (unsigned __int64 *)-8192LL || v114 )
      v60 = v114;
    v59 = (v56 - 1) & (v113 + v59);
    v125 = (unsigned __int64 **)(v58 + 8LL * v59);
    v61 = *v125;
    if ( v161 == *v125 )
      goto LABEL_53;
    ++v113;
    v114 = v60;
    v60 = (unsigned __int64 **)(v58 + 8LL * v59);
  }
  v115 = *(_DWORD *)(v38 + 248);
  if ( !v114 )
    v114 = v60;
  ++*(_QWORD *)(v38 + 232);
  v116 = v115 + 1;
  v167 = v114;
  if ( 4 * (v115 + 1) >= 3 * v56 )
    goto LABEL_158;
  if ( v56 - *(_DWORD *)(v38 + 252) - v116 <= v56 >> 3 )
  {
LABEL_159:
    sub_2C92940(v38 + 232, v56);
    sub_2C8FCF0(v38 + 232, (__int64 *)&v161, &v167);
    v57 = v161;
    v114 = v167;
    v116 = *(_DWORD *)(v38 + 248) + 1;
  }
  *(_DWORD *)(v38 + 248) = v116;
  if ( *v114 != (unsigned __int64 *)-4096LL )
    --*(_DWORD *)(v38 + 252);
  *v114 = v57;
LABEL_53:
  v62 = (unsigned __int64 *)sub_22077B0(0x18u);
  if ( v62 )
  {
    *v62 = 0;
    v62[1] = 0;
    v62[2] = 0;
  }
  v161 = v62;
  if ( v44 )
    goto LABEL_45;
LABEL_56:
  v37 = v161;
  v36 = v148;
  if ( !v161 )
    goto LABEL_57;
LABEL_67:
  if ( *v37 )
  {
    v151 = v36;
    j_j___libc_free_0(*v37);
    v36 = v151;
  }
  v152 = v36;
  j_j___libc_free_0((unsigned __int64)v37);
  v36 = v152;
  v63 = *v34;
  if ( v34[1] != *v34 )
  {
LABEL_58:
    v64 = *(_QWORD *)(a4 + 8);
    v65 = *(unsigned int *)(a4 + 24);
    v66 = (__int64 *)(v64 + 16 * v65);
    if ( (_DWORD)v65 )
    {
      v67 = v65 - 1;
      v68 = v67 & (((unsigned int)v159 >> 9) ^ ((unsigned int)v159 >> 4));
      v69 = (__int64 *)(v64 + 16LL * v68);
      v70 = *v69;
      if ( v159 == *v69 )
      {
LABEL_60:
        if ( v66 != v69 )
        {
LABEL_61:
          v149 = v36;
          v71 = sub_2C93500(a4, &v159);
          v72 = (unsigned __int64)v149;
          *v71 = (__int64)v34;
          goto LABEL_62;
        }
      }
      else
      {
        v122 = 1;
        while ( v70 != -4096 )
        {
          v126 = v122 + 1;
          v68 = v67 & (v122 + v68);
          v69 = (__int64 *)(v64 + 16LL * v68);
          v70 = *v69;
          if ( v159 == *v69 )
            goto LABEL_60;
          v122 = v126;
        }
      }
    }
    v145 = v36;
    sub_2C95CA0(a5, &v159);
    v36 = v145;
    goto LABEL_61;
  }
LABEL_70:
  if ( v63 )
  {
    v153 = v36;
    j_j___libc_free_0(v63);
    v36 = v153;
  }
  v154 = v36;
  j_j___libc_free_0((unsigned __int64)v34);
  v72 = (unsigned __int64)v154;
LABEL_62:
  v150 = v72;
  sub_C7D6A0(v169, 24LL * v171, 8);
  if ( v150 )
    j_j___libc_free_0(v150);
}
