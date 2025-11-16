// Function: sub_2063B90
// Address: 0x2063b90
//
__int64 __fastcall sub_2063B90(
        _QWORD *a1,
        _QWORD *a2,
        unsigned int a3,
        unsigned int a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  unsigned int v8; // ebx
  unsigned int v10; // esi
  __int64 v11; // rdi
  __int64 v12; // rcx
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // r8
  __int64 v16; // r13
  __int64 v17; // r8
  int v18; // edx
  unsigned int v19; // esi
  unsigned int v20; // r13d
  __int64 v21; // r9
  __int64 v22; // rbx
  __int64 v23; // rdx
  __int64 v24; // r15
  unsigned int v25; // eax
  __int64 v26; // rax
  const void **v27; // r14
  unsigned int v28; // r14d
  _QWORD *v29; // r15
  __int64 v30; // rcx
  __int64 v31; // r14
  __int64 v32; // rax
  __int64 v33; // r13
  __int64 v34; // r15
  __int64 v35; // rbx
  __int64 *v36; // rsi
  __int64 v37; // r14
  unsigned int v38; // esi
  __int64 v39; // rcx
  unsigned int v40; // edx
  __int64 *v41; // rax
  __int64 v42; // rdi
  __int64 v43; // rdi
  int v44; // esi
  unsigned int v45; // edx
  int v46; // r13d
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // r15
  __int64 v50; // r14
  unsigned __int64 v51; // rbx
  __int64 *v52; // r14
  unsigned int v53; // r14d
  _QWORD *v54; // rax
  __int64 *v55; // rbx
  __int64 *v56; // r13
  __int64 *v57; // rax
  __int64 *v58; // rdx
  __int64 *v59; // r14
  __int64 v60; // r12
  unsigned int v61; // ecx
  __int64 *v62; // rax
  __int64 v63; // rdi
  __int64 *v64; // rax
  int v65; // eax
  __int64 v66; // rax
  __int64 v67; // rdx
  __int32 v68; // eax
  __int64 *v69; // r8
  __int64 v70; // rdx
  __int64 v71; // rbx
  __int64 v72; // rax
  __int64 v73; // rax
  unsigned int v74; // edx
  _QWORD *v75; // rax
  unsigned int v76; // ecx
  __int64 v77; // rsi
  __m128i v78; // xmm0
  char v79; // al
  __int64 v80; // rax
  int v81; // edx
  __int64 v82; // rsi
  unsigned __int64 v83; // r14
  __int64 v85; // r8
  int v86; // edx
  unsigned int v87; // esi
  int v88; // ebx
  __int64 *v89; // r10
  _QWORD *v90; // rdi
  _QWORD *v91; // rcx
  __int64 v92; // r9
  __int64 *v93; // r9
  unsigned int v94; // edx
  __int64 v95; // rcx
  __int64 v96; // r12
  __int64 i; // rbx
  __int64 *v98; // rsi
  __int64 v99; // r9
  __int64 *v100; // r9
  _QWORD *v101; // rdi
  __int64 v102; // r9
  __int64 v103; // r9
  int v104; // r11d
  __int64 *v105; // r10
  int v106; // r11d
  __int64 *v107; // r10
  __int64 v108; // r9
  __int64 *v109; // r10
  int v110; // r11d
  unsigned int v111; // esi
  __int64 *v112; // rsi
  __int64 *v113; // rcx
  int v114; // ebx
  unsigned int v115; // esi
  __int64 *v116; // rbx
  int v117; // r14d
  __int64 *v118; // r8
  int v119; // ecx
  __int64 *v120; // rdx
  __int64 v121; // rdx
  __int64 v122; // r9
  int v123; // edi
  __int64 *v124; // rsi
  int v125; // edi
  __int64 v126; // rdx
  __int64 v127; // r9
  int v128; // r11d
  __int64 *v129; // r10
  unsigned int v132; // [rsp+18h] [rbp-1A8h]
  _QWORD *v134; // [rsp+20h] [rbp-1A0h]
  unsigned int v135; // [rsp+28h] [rbp-198h]
  __int64 v136; // [rsp+28h] [rbp-198h]
  _QWORD *v137; // [rsp+28h] [rbp-198h]
  __int64 v139; // [rsp+30h] [rbp-190h]
  __int64 v140; // [rsp+38h] [rbp-188h]
  unsigned int v141; // [rsp+38h] [rbp-188h]
  __int64 v142; // [rsp+38h] [rbp-188h]
  _QWORD *v143; // [rsp+38h] [rbp-188h]
  __int64 v144; // [rsp+38h] [rbp-188h]
  __int64 v145; // [rsp+38h] [rbp-188h]
  __int64 *v146; // [rsp+38h] [rbp-188h]
  __int64 v147; // [rsp+40h] [rbp-180h]
  unsigned int v148; // [rsp+4Ch] [rbp-174h]
  const void **v149; // [rsp+50h] [rbp-170h]
  __int64 v150; // [rsp+50h] [rbp-170h]
  __int64 v151; // [rsp+50h] [rbp-170h]
  __int64 v152; // [rsp+50h] [rbp-170h]
  unsigned __int64 *v153; // [rsp+50h] [rbp-170h]
  __int64 v154; // [rsp+58h] [rbp-168h] BYREF
  _QWORD *v155; // [rsp+60h] [rbp-160h] BYREF
  unsigned int v156; // [rsp+68h] [rbp-158h]
  __int64 v157; // [rsp+70h] [rbp-150h] BYREF
  unsigned int v158; // [rsp+78h] [rbp-148h]
  __int64 *v159; // [rsp+80h] [rbp-140h] BYREF
  __int64 *v160; // [rsp+88h] [rbp-138h]
  __int64 *v161; // [rsp+90h] [rbp-130h]
  __m128i v162; // [rsp+A0h] [rbp-120h] BYREF
  __int64 v163; // [rsp+B0h] [rbp-110h]
  __int64 v164; // [rsp+C0h] [rbp-100h] BYREF
  __int64 v165; // [rsp+C8h] [rbp-F8h]
  __int64 v166; // [rsp+D0h] [rbp-F0h]
  unsigned int v167; // [rsp+D8h] [rbp-E8h]
  _QWORD *v168; // [rsp+E0h] [rbp-E0h] BYREF
  unsigned int v169; // [rsp+E8h] [rbp-D8h]
  __int64 v170; // [rsp+F0h] [rbp-D0h]
  unsigned int v171; // [rsp+F8h] [rbp-C8h]
  __int64 v172; // [rsp+100h] [rbp-C0h]
  __int64 v173; // [rsp+108h] [rbp-B8h]
  char v174; // [rsp+110h] [rbp-B0h]
  _QWORD *v175; // [rsp+120h] [rbp-A0h] BYREF
  __int64 *v176; // [rsp+128h] [rbp-98h]
  __int64 *v177; // [rsp+130h] [rbp-90h]
  __int64 v178; // [rsp+138h] [rbp-88h]
  int v179; // [rsp+140h] [rbp-80h]
  _BYTE v180[120]; // [rsp+148h] [rbp-78h] BYREF

  v154 = a6;
  v159 = 0;
  v160 = 0;
  v161 = 0;
  v164 = 0;
  v165 = 0;
  v166 = 0;
  v167 = 0;
  if ( a3 <= a4 )
  {
    v8 = a3;
    v10 = 0;
    v11 = 0;
    while ( 1 )
    {
      v16 = *a2 + 40LL * v8;
      if ( !v10 )
        break;
      v12 = *(_QWORD *)(v16 + 24);
      v13 = (v10 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v14 = (__int64 *)(v11 + 16LL * v13);
      v15 = *v14;
      if ( v12 == *v14 )
      {
LABEL_4:
        ++v8;
        *((_DWORD *)v14 + 2) = 0;
        if ( a4 < v8 )
          goto LABEL_13;
        goto LABEL_5;
      }
      v106 = 1;
      v107 = 0;
      while ( v15 != -8 )
      {
        if ( v15 == -16 && !v107 )
          v107 = v14;
        v13 = (v10 - 1) & (v106 + v13);
        v14 = (__int64 *)(v11 + 16LL * v13);
        v15 = *v14;
        if ( v12 == *v14 )
          goto LABEL_4;
        ++v106;
      }
      if ( v107 )
        v14 = v107;
      ++v164;
      v18 = v166 + 1;
      if ( 4 * ((int)v166 + 1) >= 3 * v10 )
        goto LABEL_8;
      if ( v10 - (v18 + HIDWORD(v166)) <= v10 >> 3 )
      {
        sub_20639D0((__int64)&v164, v10);
        if ( !v167 )
        {
LABEL_268:
          LODWORD(v166) = v166 + 1;
          BUG();
        }
        v108 = *(_QWORD *)(v16 + 24);
        v109 = 0;
        v110 = 1;
        v18 = v166 + 1;
        v111 = (v167 - 1) & (((unsigned int)v108 >> 9) ^ ((unsigned int)v108 >> 4));
        v14 = (__int64 *)(v165 + 16LL * v111);
        v12 = *v14;
        if ( v108 != *v14 )
        {
          while ( v12 != -8 )
          {
            if ( !v109 && v12 == -16 )
              v109 = v14;
            v111 = (v167 - 1) & (v110 + v111);
            v14 = (__int64 *)(v165 + 16LL * v111);
            v12 = *v14;
            if ( v108 == *v14 )
              goto LABEL_10;
            ++v110;
          }
          v12 = *(_QWORD *)(v16 + 24);
          if ( v109 )
            v14 = v109;
        }
      }
LABEL_10:
      LODWORD(v166) = v18;
      if ( *v14 != -8 )
        --HIDWORD(v166);
      ++v8;
      *((_DWORD *)v14 + 2) = -1;
      *v14 = v12;
      *((_DWORD *)v14 + 2) = 0;
      if ( a4 < v8 )
      {
LABEL_13:
        v20 = a3;
        v148 = 0;
        v132 = 0;
        while ( 1 )
        {
          v21 = *a2;
          v22 = 40LL * v20;
          v23 = *a2 + v22;
          v24 = *(_QWORD *)(v23 + 16);
          v25 = v148 + *(_DWORD *)(v23 + 32);
          if ( v148 + (unsigned __int64)*(unsigned int *)(v23 + 32) > 0x80000000 )
            v25 = 0x80000000;
          v149 = (const void **)(v24 + 24);
          v148 = v25;
          v26 = *(_QWORD *)(v23 + 8);
          v27 = (const void **)(v26 + 24);
          if ( *(_DWORD *)(v26 + 32) <= 0x40u )
          {
            v91 = *(_QWORD **)(v26 + 24);
            if ( v91 == *(_QWORD **)(v24 + 24) )
            {
              ++v132;
              if ( a3 == v20 )
                goto LABEL_19;
              v92 = *(_QWORD *)(v21 + 40LL * (v20 - 1) + 16);
              v169 = *(_DWORD *)(v26 + 32);
              v93 = (__int64 *)(v92 + 24);
            }
            else
            {
              v132 += 2;
              if ( a3 == v20 )
              {
LABEL_19:
                v169 = *(_DWORD *)(v24 + 32);
                if ( v169 <= 0x40 )
                  goto LABEL_20;
                goto LABEL_115;
              }
              v102 = *(_QWORD *)(v21 + 40LL * (v20 - 1) + 16);
              v169 = *(_DWORD *)(v26 + 32);
              v91 = *(_QWORD **)(v26 + 24);
              v93 = (__int64 *)(v102 + 24);
            }
            v168 = v91;
          }
          else
          {
            v135 = *(_DWORD *)(v26 + 32);
            v140 = *a2;
            if ( sub_16A5220(v26 + 24, v149) )
            {
              ++v132;
              if ( a3 == v20 )
                goto LABEL_19;
              v103 = *(_QWORD *)(v140 + 40LL * (v20 - 1) + 16);
              v169 = v135;
              v100 = (__int64 *)(v103 + 24);
            }
            else
            {
              v132 += 2;
              if ( a3 == v20 )
                goto LABEL_19;
              v99 = *(_QWORD *)(v140 + 40LL * (v20 - 1) + 16);
              v169 = v135;
              v100 = (__int64 *)(v99 + 24);
            }
            v146 = v100;
            sub_16A4FD0((__int64)&v168, v27);
            v93 = v146;
          }
          sub_16A7590((__int64)&v168, v93);
          v94 = v169;
          v169 = 0;
          LODWORD(v176) = v94;
          v175 = v168;
          if ( v94 <= 0x40 )
          {
            v95 = (__int64)v168 - 1;
            goto LABEL_107;
          }
          v137 = v168;
          if ( v94 - (unsigned int)sub_16A57B0((__int64)&v175) > 0x40 )
            break;
          v144 = *v137 - 1LL;
          j_j___libc_free_0_0(v137);
          v95 = v144;
          if ( v169 > 0x40 )
          {
            v101 = v168;
            if ( v168 )
              goto LABEL_123;
          }
LABEL_107:
          if ( !v95 )
            goto LABEL_19;
LABEL_108:
          v143 = a2;
          v96 = v95;
          for ( i = 0; i != v96; ++i )
          {
            while ( 1 )
            {
              v98 = v160;
              if ( v160 != v161 )
                break;
              ++i;
              sub_1D4AF10((__int64)&v159, v160, &v154);
              if ( i == v96 )
                goto LABEL_114;
            }
            if ( v160 )
            {
              *v160 = v154;
              v98 = v160;
            }
            v160 = v98 + 1;
          }
LABEL_114:
          a2 = v143;
          v22 = 40LL * v20;
          v169 = *(_DWORD *)(v24 + 32);
          if ( v169 <= 0x40 )
          {
LABEL_20:
            v168 = *(_QWORD **)(v24 + 24);
            goto LABEL_21;
          }
LABEL_115:
          sub_16A4FD0((__int64)&v168, v149);
LABEL_21:
          sub_16A7590((__int64)&v168, (__int64 *)v27);
          v28 = v169;
          v29 = v168;
          v169 = 0;
          LODWORD(v176) = v28;
          v175 = v168;
          if ( v28 <= 0x40 )
          {
            v30 = (__int64)v168 + 1;
            goto LABEL_23;
          }
          if ( v28 - (unsigned int)sub_16A57B0((__int64)&v175) > 0x40 )
          {
            if ( !v29 || (j_j___libc_free_0_0(v29), v169 <= 0x40) || (v90 = v168, v30 = 0, !v168) )
            {
              v31 = *a2;
LABEL_89:
              v38 = v167;
              v37 = v22 + v31;
              if ( !v167 )
                goto LABEL_90;
              goto LABEL_31;
            }
LABEL_100:
            v152 = v30;
            j_j___libc_free_0_0(v90);
            v30 = v152;
            goto LABEL_23;
          }
          v151 = *v29 + 1LL;
          j_j___libc_free_0_0(v29);
          v30 = v151;
          if ( v169 > 0x40 )
          {
            v90 = v168;
            if ( v168 )
              goto LABEL_100;
          }
LABEL_23:
          v31 = *a2;
          if ( !v30 )
            goto LABEL_89;
          v141 = v20;
          v32 = v31 + v22;
          v33 = v22;
          v34 = 0;
          v35 = v30;
          do
          {
            while ( 1 )
            {
              v36 = v160;
              v37 = v32;
              if ( v160 != v161 )
                break;
              ++v34;
              sub_1D4AF10((__int64)&v159, v160, (_QWORD *)(v32 + 24));
              v32 = v33 + *a2;
              v37 = v32;
              if ( v34 == v35 )
                goto LABEL_30;
            }
            if ( v160 )
            {
              *v160 = *(_QWORD *)(v32 + 24);
              v36 = v160;
              v32 = v33 + *a2;
              v37 = v32;
            }
            ++v34;
            v160 = v36 + 1;
          }
          while ( v34 != v35 );
LABEL_30:
          v38 = v167;
          v20 = v141;
          if ( !v167 )
          {
LABEL_90:
            ++v164;
            goto LABEL_91;
          }
LABEL_31:
          v39 = *(_QWORD *)(v37 + 24);
          v40 = (v38 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
          v41 = (__int64 *)(v165 + 16LL * v40);
          v42 = *v41;
          if ( v39 == *v41 )
          {
            v43 = *((unsigned int *)v41 + 2);
            v44 = *((_DWORD *)v41 + 2);
LABEL_33:
            v45 = v44 + *(_DWORD *)(v37 + 32);
            if ( v43 + (unsigned __int64)*(unsigned int *)(v37 + 32) > 0x80000000 )
              v45 = 0x80000000;
            goto LABEL_35;
          }
          v104 = 1;
          v105 = 0;
          while ( v42 != -8 )
          {
            if ( v105 || v42 != -16 )
              v41 = v105;
            v40 = (v38 - 1) & (v104 + v40);
            v116 = (__int64 *)(v165 + 16LL * v40);
            v42 = *v116;
            if ( v39 == *v116 )
            {
              v43 = *((unsigned int *)v116 + 2);
              v41 = (__int64 *)(v165 + 16LL * v40);
              v44 = *((_DWORD *)v116 + 2);
              goto LABEL_33;
            }
            ++v104;
            v105 = v41;
            v41 = (__int64 *)(v165 + 16LL * v40);
          }
          if ( v105 )
            v41 = v105;
          ++v164;
          v86 = v166 + 1;
          if ( 4 * ((int)v166 + 1) < 3 * v38 )
          {
            if ( v38 - HIDWORD(v166) - v86 > v38 >> 3 )
              goto LABEL_134;
            sub_20639D0((__int64)&v164, v38);
            if ( !v167 )
            {
LABEL_269:
              LODWORD(v166) = v166 + 1;
              BUG();
            }
            v85 = *(_QWORD *)(v37 + 24);
            v89 = 0;
            v86 = v166 + 1;
            v114 = 1;
            v115 = (v167 - 1) & (((unsigned int)v85 >> 9) ^ ((unsigned int)v85 >> 4));
            v41 = (__int64 *)(v165 + 16LL * v115);
            v39 = *v41;
            if ( v85 == *v41 )
              goto LABEL_134;
            while ( v39 != -8 )
            {
              if ( !v89 && v39 == -16 )
                v89 = v41;
              v115 = (v167 - 1) & (v114 + v115);
              v41 = (__int64 *)(v165 + 16LL * v115);
              v39 = *v41;
              if ( v85 == *v41 )
                goto LABEL_134;
              ++v114;
            }
            goto LABEL_95;
          }
LABEL_91:
          sub_20639D0((__int64)&v164, 2 * v38);
          if ( !v167 )
            goto LABEL_269;
          v85 = *(_QWORD *)(v37 + 24);
          v86 = v166 + 1;
          v87 = (v167 - 1) & (((unsigned int)v85 >> 9) ^ ((unsigned int)v85 >> 4));
          v41 = (__int64 *)(v165 + 16LL * v87);
          v39 = *v41;
          if ( v85 == *v41 )
            goto LABEL_134;
          v88 = 1;
          v89 = 0;
          while ( v39 != -8 )
          {
            if ( !v89 && v39 == -16 )
              v89 = v41;
            v87 = (v167 - 1) & (v88 + v87);
            v41 = (__int64 *)(v165 + 16LL * v87);
            v39 = *v41;
            if ( v85 == *v41 )
              goto LABEL_134;
            ++v88;
          }
LABEL_95:
          v39 = v85;
          if ( v89 )
            v41 = v89;
LABEL_134:
          LODWORD(v166) = v86;
          if ( *v41 != -8 )
            --HIDWORD(v166);
          *v41 = v39;
          v45 = 0x80000000;
          *((_DWORD *)v41 + 2) = -1;
LABEL_35:
          *((_DWORD *)v41 + 2) = v45;
          if ( a4 < ++v20 )
          {
            v46 = v166;
            goto LABEL_37;
          }
        }
        if ( !v137 || (j_j___libc_free_0_0(v137), v169 <= 0x40) )
        {
          v95 = -2;
          goto LABEL_108;
        }
        v101 = v168;
        v95 = -2;
        if ( !v168 )
          goto LABEL_108;
LABEL_123:
        v145 = v95;
        j_j___libc_free_0_0(v101);
        v95 = v145;
        goto LABEL_107;
      }
LABEL_5:
      v11 = v165;
      v10 = v167;
    }
    ++v164;
LABEL_8:
    sub_20639D0((__int64)&v164, 2 * v10);
    if ( !v167 )
      goto LABEL_268;
    v17 = *(_QWORD *)(v16 + 24);
    v18 = v166 + 1;
    v19 = (v167 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
    v14 = (__int64 *)(v165 + 16LL * v19);
    v12 = *v14;
    if ( v17 != *v14 )
    {
      v128 = 1;
      v129 = 0;
      while ( v12 != -8 )
      {
        if ( !v129 && v12 == -16 )
          v129 = v14;
        v19 = (v167 - 1) & (v128 + v19);
        v14 = (__int64 *)(v165 + 16LL * v19);
        v12 = *v14;
        if ( v17 == *v14 )
          goto LABEL_10;
        ++v128;
      }
      v12 = *(_QWORD *)(v16 + 24);
      if ( v129 )
        v14 = v129;
    }
    goto LABEL_10;
  }
  v148 = 0;
  v46 = 0;
  v132 = 0;
LABEL_37:
  v47 = a1[69];
  v136 = *(_QWORD *)(v47 + 16);
  v48 = sub_1E0A0C0(*(_QWORD *)(v47 + 32));
  v49 = *(_QWORD *)(*a2 + 40LL * a4 + 16);
  v147 = 40LL * a4;
  v50 = *(_QWORD *)(*a2 + 40LL * a3 + 8);
  v142 = 40LL * a3;
  v51 = 8 * (unsigned int)sub_15A95A0(v48, 0);
  v52 = (__int64 *)(v50 + 24);
  LODWORD(v176) = *(_DWORD *)(v49 + 32);
  if ( (unsigned int)v176 > 0x40 )
    sub_16A4FD0((__int64)&v175, (const void **)(v49 + 24));
  else
    v175 = *(_QWORD **)(v49 + 24);
  sub_16A7590((__int64)&v175, v52);
  v53 = (unsigned int)v176;
  LODWORD(v176) = 0;
  v169 = v53;
  v168 = v175;
  if ( v53 > 0x40 )
  {
    v153 = v175;
    if ( v53 - (unsigned int)sub_16A57B0((__int64)&v168) > 0x40 )
    {
      v83 = -1;
    }
    else
    {
      v83 = *v153;
      if ( *v153 == -1 )
      {
        j_j___libc_free_0_0(v153);
        if ( (unsigned int)v176 <= 0x40 )
          goto LABEL_41;
        goto LABEL_154;
      }
      ++v83;
    }
    if ( !v153 )
      goto LABEL_156;
    j_j___libc_free_0_0(v153);
    if ( (unsigned int)v176 <= 0x40 )
      goto LABEL_156;
LABEL_154:
    if ( v175 )
      j_j___libc_free_0_0(v175);
    goto LABEL_156;
  }
  if ( v175 == (_QWORD *)-1LL )
  {
LABEL_41:
    v139 = *(_QWORD *)(a1[89] + 8LL);
    v54 = sub_1E0B6F0(v139, *(_QWORD *)(a5 + 40));
    v55 = v159;
    v56 = v160;
    v175 = 0;
    v150 = (__int64)v54;
    v57 = (__int64 *)v180;
    v176 = (__int64 *)v180;
    v177 = (__int64 *)v180;
    v178 = 8;
    v179 = 0;
    if ( v160 == v159 )
      goto LABEL_64;
    v134 = a2;
    v58 = (__int64 *)v180;
    while ( 1 )
    {
      v60 = *v55;
      if ( v57 == v58 )
      {
        v59 = &v57[HIDWORD(v178)];
        if ( v59 == v57 )
        {
          v120 = v57;
        }
        else
        {
          do
          {
            if ( v60 == *v57 )
              break;
            ++v57;
          }
          while ( v59 != v57 );
          v120 = v59;
        }
      }
      else
      {
        v59 = &v58[(unsigned int)v178];
        v57 = sub_16CC9F0((__int64)&v175, *v55);
        if ( v60 == *v57 )
        {
          if ( v177 == v176 )
            v120 = &v177[HIDWORD(v178)];
          else
            v120 = &v177[(unsigned int)v178];
        }
        else
        {
          if ( v177 != v176 )
          {
            v57 = &v177[(unsigned int)v178];
LABEL_46:
            if ( v57 != v59 )
              goto LABEL_47;
            goto LABEL_59;
          }
          v120 = &v177[HIDWORD(v178)];
          v57 = v120;
        }
      }
      if ( v57 == v120 )
        goto LABEL_46;
      do
      {
        if ( (unsigned __int64)*v57 < 0xFFFFFFFFFFFFFFFELL )
          goto LABEL_46;
        ++v57;
      }
      while ( v120 != v57 );
      if ( v57 != v59 )
        goto LABEL_47;
LABEL_59:
      if ( v167 )
      {
        v61 = (v167 - 1) & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
        v62 = (__int64 *)(v165 + 16LL * v61);
        v63 = *v62;
        if ( v60 == *v62 )
          goto LABEL_61;
        v117 = 1;
        v118 = 0;
        while ( v63 != -8 )
        {
          if ( !v118 && v63 == -16 )
            v118 = v62;
          v61 = (v167 - 1) & (v117 + v61);
          v62 = (__int64 *)(v165 + 16LL * v61);
          v63 = *v62;
          if ( v60 == *v62 )
            goto LABEL_61;
          ++v117;
        }
        if ( v118 )
          v62 = v118;
        ++v164;
        v119 = v166 + 1;
        if ( 4 * ((int)v166 + 1) < 3 * v167 )
        {
          if ( v167 - HIDWORD(v166) - v119 > v167 >> 3 )
            goto LABEL_206;
          sub_20639D0((__int64)&v164, v167);
          if ( !v167 )
          {
LABEL_267:
            LODWORD(v166) = v166 + 1;
            BUG();
          }
          v125 = 1;
          v124 = 0;
          LODWORD(v126) = (v167 - 1) & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
          v119 = v166 + 1;
          v62 = (__int64 *)(v165 + 16LL * (unsigned int)v126);
          v127 = *v62;
          if ( v60 == *v62 )
            goto LABEL_206;
          while ( v127 != -8 )
          {
            if ( v127 == -16 && !v124 )
              v124 = v62;
            v126 = (v167 - 1) & ((_DWORD)v126 + v125);
            v62 = (__int64 *)(v165 + 16 * v126);
            v127 = *v62;
            if ( v60 == *v62 )
              goto LABEL_206;
            ++v125;
          }
          goto LABEL_223;
        }
      }
      else
      {
        ++v164;
      }
      sub_20639D0((__int64)&v164, 2 * v167);
      if ( !v167 )
        goto LABEL_267;
      LODWORD(v121) = (v167 - 1) & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
      v119 = v166 + 1;
      v62 = (__int64 *)(v165 + 16LL * (unsigned int)v121);
      v122 = *v62;
      if ( v60 == *v62 )
        goto LABEL_206;
      v123 = 1;
      v124 = 0;
      while ( v122 != -8 )
      {
        if ( !v124 && v122 == -16 )
          v124 = v62;
        v121 = (v167 - 1) & ((_DWORD)v121 + v123);
        v62 = (__int64 *)(v165 + 16 * v121);
        v122 = *v62;
        if ( v60 == *v62 )
          goto LABEL_206;
        ++v123;
      }
LABEL_223:
      if ( v124 )
        v62 = v124;
LABEL_206:
      LODWORD(v166) = v119;
      if ( *v62 != -8 )
        --HIDWORD(v166);
      *v62 = v60;
      *((_DWORD *)v62 + 2) = -1;
LABEL_61:
      sub_2052F00((__int64)a1, v150, v60, *((_DWORD *)v62 + 2));
      v64 = v176;
      if ( v177 == v176 )
      {
        v112 = &v176[HIDWORD(v178)];
        if ( v176 != v112 )
        {
          v113 = 0;
          do
          {
            if ( v60 == *v64 )
              goto LABEL_47;
            if ( *v64 == -2 )
              v113 = v64;
            ++v64;
          }
          while ( v112 != v64 );
          if ( v113 )
          {
            *v113 = v60;
            --v179;
            v175 = (_QWORD *)((char *)v175 + 1);
            goto LABEL_47;
          }
        }
        if ( HIDWORD(v178) < (unsigned int)v178 )
        {
          ++HIDWORD(v178);
          *v112 = v60;
          v175 = (_QWORD *)((char *)v175 + 1);
LABEL_47:
          if ( v56 == ++v55 )
            goto LABEL_63;
          goto LABEL_48;
        }
      }
      ++v55;
      sub_16CCBA0((__int64)&v175, v60);
      if ( v56 == v55 )
      {
LABEL_63:
        a2 = v134;
LABEL_64:
        sub_1D96570(*(unsigned int **)(v150 + 112), *(unsigned int **)(v150 + 120));
        v65 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v136 + 1016LL))(v136);
        v66 = sub_1E0BA80(v139, v65);
        v68 = sub_1E0D9C0(v66, (__int64)&v159, v67);
        v162.m128i_i64[1] = v150;
        v162.m128i_i32[1] = v68;
        v162.m128i_i32[0] = -1;
        v163 = 0;
        if ( (*(_BYTE *)(a5 + 23) & 0x40) != 0 )
          v69 = *(__int64 **)(a5 - 8);
        else
          v69 = (__int64 *)(a5 - 24LL * (*(_DWORD *)(a5 + 20) & 0xFFFFFFF));
        v70 = *a2;
        v71 = *v69;
        v72 = *(_QWORD *)(*a2 + v147 + 16);
        v158 = *(_DWORD *)(v72 + 32);
        if ( v158 > 0x40 )
        {
          sub_16A4FD0((__int64)&v157, (const void **)(v72 + 24));
          v70 = *a2;
        }
        else
        {
          v157 = *(_QWORD *)(v72 + 24);
        }
        v73 = *(_QWORD *)(v70 + v142 + 8);
        v74 = *(_DWORD *)(v73 + 32);
        v156 = v74;
        if ( v74 > 0x40 )
        {
          sub_16A4FD0((__int64)&v155, (const void **)(v73 + 24));
          v74 = v156;
          v75 = v155;
        }
        else
        {
          v75 = *(_QWORD **)(v73 + 24);
        }
        v168 = v75;
        v76 = v158;
        v172 = v71;
        v169 = v74;
        v171 = v158;
        v170 = v157;
        v77 = a1[77];
        v173 = 0;
        v174 = 0;
        if ( v77 == a1[78] )
        {
          sub_205AD70(a1 + 76, v77, (__int64)&v168, &v162);
          v80 = a1[77];
          v76 = v171;
        }
        else
        {
          if ( v77 )
          {
            *(_DWORD *)(v77 + 8) = v74;
            v76 = 0;
            *(_QWORD *)v77 = v168;
            v169 = 0;
            *(_DWORD *)(v77 + 24) = v171;
            *(_QWORD *)(v77 + 16) = v170;
            v171 = 0;
            v78 = _mm_loadu_si128(&v162);
            *(_QWORD *)(v77 + 32) = v172;
            *(_QWORD *)(v77 + 40) = v173;
            v79 = v174;
            *(__m128i *)(v77 + 56) = v78;
            *(_BYTE *)(v77 + 48) = v79;
            *(_QWORD *)(v77 + 72) = v163;
            v77 = a1[77];
          }
          v80 = v77 + 80;
          a1[77] = v77 + 80;
        }
        v81 = -858993459 * ((v80 - a1[76]) >> 4) - 1;
        v82 = *(_QWORD *)(*a2 + v147 + 16);
        *(_QWORD *)(a7 + 8) = *(_QWORD *)(*a2 + v142 + 8);
        *(_DWORD *)a7 = 1;
        *(_QWORD *)(a7 + 16) = v82;
        *(_DWORD *)(a7 + 24) = v81;
        *(_DWORD *)(a7 + 32) = v148;
        if ( v76 > 0x40 && v170 )
          j_j___libc_free_0_0(v170);
        if ( v169 > 0x40 && v168 )
          j_j___libc_free_0_0(v168);
        if ( v177 != v176 )
          _libc_free((unsigned __int64)v177);
        LODWORD(v83) = 1;
        goto LABEL_83;
      }
LABEL_48:
      v58 = v177;
      v57 = v176;
    }
  }
  v83 = (unsigned __int64)v175 + 1;
LABEL_156:
  if ( v51 < v83 )
    goto LABEL_41;
  LOBYTE(v83) = v132 > 2 && v46 == 1;
  if ( (_BYTE)v83 )
  {
    LODWORD(v83) = 0;
    goto LABEL_83;
  }
  if ( (v132 <= 4 || v46 != 2) && (v46 != 3 || v132 <= 5) )
    goto LABEL_41;
LABEL_83:
  j___libc_free_0(v165);
  if ( v159 )
    j_j___libc_free_0(v159, (char *)v161 - (char *)v159);
  return (unsigned int)v83;
}
