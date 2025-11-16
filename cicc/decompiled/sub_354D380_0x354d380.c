// Function: sub_354D380
// Address: 0x354d380
//
__int64 __fastcall sub_354D380(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rdx
  void *v3; // rbx
  __int64 (*v4)(); // rax
  void *v5; // rax
  __int64 v6; // rcx
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 v13; // rbx
  unsigned __int64 v14; // r12
  int v15; // esi
  int v16; // r10d
  unsigned int v17; // ecx
  char *v18; // rax
  _QWORD *v19; // rdx
  __int64 v20; // r9
  unsigned int v21; // ebx
  unsigned __int64 v22; // kr08_8
  _BYTE *v23; // r13
  __int64 v24; // r12
  __int64 v25; // r13
  __int64 v26; // rax
  __int64 *v27; // rbx
  __int64 v28; // r14
  void *v29; // rax
  unsigned __int64 v30; // kr00_8
  void *v31; // rax
  unsigned int v32; // r8d
  unsigned int v33; // r10d
  unsigned int v34; // ecx
  void *v35; // rdi
  _BYTE *v36; // r15
  __int64 v37; // r14
  __int64 v38; // rax
  _BYTE *v39; // rsi
  __int64 v40; // r13
  __int64 v41; // r12
  __int64 v42; // rdi
  _WORD *v43; // rcx
  __int64 v44; // rdi
  __int64 v45; // rsi
  unsigned __int16 *v46; // r12
  unsigned __int16 *v47; // rbx
  unsigned __int64 v48; // rsi
  int v49; // r14d
  unsigned int v50; // ecx
  char *v51; // rax
  _QWORD *v52; // rdx
  __int64 v53; // r10
  int *v54; // rdx
  int v55; // eax
  __int64 v56; // rdi
  __int64 (*v57)(); // rax
  __int64 *v58; // rcx
  __int64 *v59; // rdx
  unsigned __int64 v60; // r13
  int v61; // eax
  volatile signed __int32 *v62; // r12
  signed __int32 v63; // eax
  signed __int32 v64; // eax
  volatile signed __int32 *v65; // r12
  signed __int32 v66; // eax
  signed __int32 v67; // eax
  _BYTE *v68; // rax
  __int64 v69; // rbx
  unsigned int v70; // r13d
  __int64 v71; // r15
  void *v72; // r14
  _QWORD *v73; // r12
  __int64 v74; // rdx
  __int64 v75; // rsi
  int v76; // edx
  int v77; // edi
  unsigned int v78; // ecx
  __int64 *v79; // rdx
  __int64 v80; // r8
  __int64 *v81; // r15
  __int64 *v82; // r12
  int v83; // r14d
  unsigned int v84; // r13d
  __int64 v85; // r8
  __int64 v86; // r9
  unsigned __int64 v87; // rdx
  unsigned __int64 v88; // rsi
  int v89; // eax
  __int64 v90; // rsi
  _QWORD *v91; // rcx
  _QWORD *v92; // rdx
  unsigned __int64 v93; // r15
  volatile signed __int32 *v94; // rdi
  signed __int32 v95; // edx
  volatile signed __int32 *v96; // rdi
  signed __int32 v97; // edx
  __int64 v98; // rdi
  __int64 (*v99)(); // rax
  signed __int32 v100; // eax
  signed __int32 v101; // eax
  _BYTE *v102; // rax
  __int64 v103; // rcx
  __int64 v104; // rcx
  __int64 v105; // rbx
  __int64 *v106; // r12
  unsigned __int64 v107; // r13
  volatile signed __int32 *v108; // r14
  signed __int32 v109; // eax
  volatile signed __int32 *v110; // r14
  signed __int32 v111; // eax
  int v113; // ecx
  signed __int32 v114; // eax
  signed __int32 v115; // eax
  int v116; // esi
  __int64 v117; // r13
  __int64 i; // r14
  __int64 v119; // r15
  int v120; // ecx
  _DWORD *v121; // rdx
  int v122; // edx
  int v123; // r9d
  __int64 v124; // rax
  __int64 v125; // r8
  __int64 v126; // r9
  __int64 v127; // rbx
  unsigned __int64 v128; // rsi
  __int64 v129; // [rsp+8h] [rbp-178h]
  _BYTE *v130; // [rsp+10h] [rbp-170h]
  __int64 *v131; // [rsp+10h] [rbp-170h]
  unsigned int v132; // [rsp+1Ch] [rbp-164h]
  unsigned __int64 v133; // [rsp+1Ch] [rbp-164h]
  __int64 v134; // [rsp+20h] [rbp-160h]
  void *v136; // [rsp+30h] [rbp-150h]
  void *v137; // [rsp+30h] [rbp-150h]
  __int64 v138; // [rsp+40h] [rbp-140h]
  __int64 *v139; // [rsp+40h] [rbp-140h]
  __int64 v140; // [rsp+40h] [rbp-140h]
  unsigned int v141; // [rsp+40h] [rbp-140h]
  __int64 v142; // [rsp+40h] [rbp-140h]
  __int64 v143; // [rsp+48h] [rbp-138h]
  __int64 v144; // [rsp+48h] [rbp-138h]
  unsigned int v145; // [rsp+48h] [rbp-138h]
  __int64 v146; // [rsp+50h] [rbp-130h]
  _QWORD *v147; // [rsp+58h] [rbp-128h]
  __int64 v148; // [rsp+60h] [rbp-120h] BYREF
  void *v149; // [rsp+68h] [rbp-118h]
  unsigned __int64 v150; // [rsp+70h] [rbp-110h]
  unsigned int v151; // [rsp+78h] [rbp-108h]
  _QWORD v152[3]; // [rsp+80h] [rbp-100h] BYREF
  void *v153; // [rsp+98h] [rbp-E8h]
  unsigned __int64 v154; // [rsp+A0h] [rbp-E0h]
  unsigned int v155; // [rsp+A8h] [rbp-D8h]
  unsigned __int64 v156; // [rsp+B0h] [rbp-D0h] BYREF
  _BYTE *v157; // [rsp+B8h] [rbp-C8h]
  _BYTE *v158; // [rsp+C0h] [rbp-C0h]
  __int64 v159; // [rsp+C8h] [rbp-B8h]
  void *v160; // [rsp+D0h] [rbp-B0h]
  __int64 v161; // [rsp+D8h] [rbp-A8h]
  void *src; // [rsp+E0h] [rbp-A0h]
  unsigned __int64 v163; // [rsp+E8h] [rbp-98h]
  unsigned int v164; // [rsp+F0h] [rbp-90h]
  __int64 *v165; // [rsp+100h] [rbp-80h] BYREF
  __int64 v166; // [rsp+108h] [rbp-78h]
  __int64 v167; // [rsp+110h] [rbp-70h] BYREF
  void *v168; // [rsp+118h] [rbp-68h]
  unsigned __int64 v169; // [rsp+120h] [rbp-60h]
  unsigned int v170; // [rsp+128h] [rbp-58h]

  v2 = 0;
  v3 = (void *)a1[2];
  v4 = *(__int64 (**)())(*(_QWORD *)v3 + 216LL);
  if ( v4 != sub_2F391C0 )
    v2 = ((__int64 (__fastcall *)(void *, __int64, _QWORD))v4)(v3, a2, 0);
  v5 = v3;
  v146 = v2;
  v147 = v3;
  v148 = 0;
  v6 = a1[4];
  v149 = 0;
  v150 = 0;
  v7 = *(_QWORD *)(v6 + 48);
  v151 = 0;
  v143 = *(_QWORD *)(v6 + 56);
  if ( v143 != v7 )
  {
    while ( 1 )
    {
      v8 = *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v7 + 16LL) + 6LL);
      if ( !v2 )
        break;
      v9 = *(_QWORD *)(v2 + 104);
      if ( !v9 )
        break;
      v10 = v9 + 10 * v8;
      v11 = *(_QWORD *)(v2 + 80);
      v12 = v11 + 24LL * *(unsigned __int16 *)(v10 + 4);
      v13 = v11 + 24LL * *(unsigned __int16 *)(v10 + 2);
      if ( v12 != v13 )
      {
        while ( 1 )
        {
          while ( 1 )
          {
            v14 = *(_QWORD *)(v13 + 8);
            v156 = v14;
            if ( (unsigned int)sub_39FAC40(v14) == 1 )
              break;
            v13 += 24;
            if ( v12 == v13 )
              goto LABEL_14;
          }
          v15 = v151;
          if ( !v151 )
            break;
          v16 = 1;
          v17 = (v151 - 1) & (((0xBF58476D1CE4E5B9LL * v14) >> 31) ^ (484763065 * v14));
          v18 = (char *)v149 + 16 * v17;
          v19 = 0;
          v20 = *(_QWORD *)v18;
          if ( v14 != *(_QWORD *)v18 )
          {
            while ( v20 != -1 )
            {
              if ( !v19 && v20 == -2 )
                v19 = v18;
              v17 = (v151 - 1) & (v16 + v17);
              v18 = (char *)v149 + 16 * v17;
              v20 = *(_QWORD *)v18;
              if ( v14 == *(_QWORD *)v18 )
                goto LABEL_12;
              ++v16;
            }
            if ( !v19 )
              v19 = v18;
            ++v148;
            v120 = v150 + 1;
            v165 = v19;
            if ( 4 * ((int)v150 + 1) < 3 * v151 )
            {
              if ( v151 - HIDWORD(v150) - v120 > v151 >> 3 )
              {
LABEL_182:
                LODWORD(v150) = v120;
                if ( *v19 != -1 )
                  --HIDWORD(v150);
                *v19 = v14;
                v121 = v19 + 1;
                *v121 = 0;
                *v121 = 1;
                goto LABEL_13;
              }
LABEL_187:
              sub_9E25D0((__int64)&v148, v15);
              sub_27B2460((__int64)&v148, (__int64 *)&v156, &v165);
              v14 = v156;
              v19 = v165;
              v120 = v150 + 1;
              goto LABEL_182;
            }
LABEL_186:
            v15 = 2 * v151;
            goto LABEL_187;
          }
LABEL_12:
          ++*((_DWORD *)v18 + 2);
LABEL_13:
          v13 += 24;
          if ( v12 == v13 )
          {
LABEL_14:
            v2 = v146;
            goto LABEL_15;
          }
        }
        ++v148;
        v165 = 0;
        goto LABEL_186;
      }
LABEL_15:
      v7 += 256;
      if ( v143 == v7 )
      {
        v5 = v147;
        goto LABEL_17;
      }
    }
    if ( !v147 || (v42 = *(_QWORD *)(v147[25] + 40LL)) == 0 )
      BUG();
    v43 = (_WORD *)(v42 + 14 * v8);
    if ( (*v43 & 0x1FFF) == 0x1FFF )
      goto LABEL_15;
    v44 = v147[22];
    v45 = (unsigned __int16)v43[1];
    v46 = (unsigned __int16 *)(v44 + 6 * (v45 + (unsigned __int16)v43[2]));
    v47 = (unsigned __int16 *)(v44 + 6 * v45);
    if ( v46 == v47 )
      goto LABEL_15;
    while ( !v47[1] )
    {
LABEL_41:
      v47 += 3;
      if ( v46 == v47 )
      {
        v2 = v146;
        goto LABEL_15;
      }
    }
    v48 = *v47;
    v156 = v48;
    if ( v151 )
    {
      v49 = 1;
      v50 = (v151 - 1) & (((0xBF58476D1CE4E5B9LL * v48) >> 31) ^ (484763065 * v48));
      v51 = (char *)v149 + 16 * v50;
      v52 = 0;
      v53 = *(_QWORD *)v51;
      if ( v48 == *(_QWORD *)v51 )
      {
LABEL_39:
        v54 = (int *)(v51 + 8);
        v55 = *((_DWORD *)v51 + 2) + 1;
LABEL_40:
        *v54 = v55;
        goto LABEL_41;
      }
      while ( v53 != -1 )
      {
        if ( v52 || v53 != -2 )
          v51 = (char *)v52;
        v50 = (v151 - 1) & (v49 + v50);
        v53 = *((_QWORD *)v149 + 2 * v50);
        if ( v48 == v53 )
        {
          v51 = (char *)v149 + 16 * v50;
          goto LABEL_39;
        }
        ++v49;
        v52 = v51;
        v51 = (char *)v149 + 16 * v50;
      }
      if ( !v52 )
        v52 = v51;
      ++v148;
      v113 = v150 + 1;
      v165 = v52;
      if ( 4 * ((int)v150 + 1) < 3 * v151 )
      {
        if ( v151 - HIDWORD(v150) - v113 > v151 >> 3 )
          goto LABEL_147;
        v116 = v151;
        goto LABEL_162;
      }
    }
    else
    {
      ++v148;
      v165 = 0;
    }
    v116 = 2 * v151;
LABEL_162:
    sub_9E25D0((__int64)&v148, v116);
    sub_27B2460((__int64)&v148, (__int64 *)&v156, &v165);
    v48 = v156;
    v52 = v165;
    v113 = v150 + 1;
LABEL_147:
    LODWORD(v150) = v113;
    if ( *v52 != -1 )
      --HIDWORD(v150);
    *v52 = v48;
    v55 = 1;
    v54 = (int *)(v52 + 1);
    *v54 = 0;
    goto LABEL_40;
  }
LABEL_17:
  v159 = v2;
  v160 = v5;
  v156 = 0;
  v157 = 0;
  v158 = 0;
  v161 = 0;
  src = 0;
  v163 = 0;
  v164 = 0;
  sub_C7D6A0(0, 0, 8);
  v164 = v151;
  if ( v151 )
  {
    src = (void *)sub_C7D670(16LL * v151, 8);
    v163 = v150;
    memcpy(src, v149, 16LL * v164);
  }
  else
  {
    src = 0;
    v163 = 0;
  }
  v144 = v159;
  v138 = (__int64)v160;
  sub_C7D6A0(0, 0, 8);
  v21 = v164;
  if ( v164 )
  {
    v134 = 16LL * v164;
    v136 = (void *)sub_C7D670(v134, 8);
    v22 = v163;
    memcpy(v136, src, 16LL * v21);
  }
  else
  {
    v134 = 0;
    v136 = 0;
    v22 = 0;
  }
  v23 = v157;
  v24 = v156;
  sub_C7D6A0(0, 0, 8);
  v25 = (__int64)&v23[-v24];
  if ( v25 > 8 )
  {
    v117 = v25 >> 3;
    for ( i = (v117 - 2) / 2; ; --i )
    {
      v119 = *(_QWORD *)(v24 + 8 * i);
      v167 = 0;
      v165 = (__int64 *)v144;
      v168 = 0;
      v166 = v138;
      v169 = 0;
      v170 = 0;
      sub_C7D6A0(0, 0, 8);
      v170 = v21;
      if ( v21 )
      {
        v168 = (void *)sub_C7D670(v134, 8);
        v169 = v22;
        memcpy(v168, v136, 16LL * v170);
      }
      else
      {
        v168 = 0;
        v169 = 0;
      }
      sub_353F9B0(v24, i, v117, v119, (__int64)&v165);
      sub_C7D6A0((__int64)v168, 16LL * v170, 8);
      if ( !i )
        break;
    }
  }
  sub_C7D6A0((__int64)v136, v134, 8);
  sub_C7D6A0(0, 0, 8);
  v26 = a1[4];
  v27 = *(__int64 **)(v26 + 48);
  v139 = *(__int64 **)(v26 + 56);
  while ( v139 != v27 )
  {
    v38 = *v27;
    v39 = v157;
    v152[0] = *v27;
    if ( v157 == v158 )
    {
      sub_2E26050((__int64)&v156, v157, v152);
    }
    else
    {
      if ( v157 )
      {
        *(_QWORD *)v157 = v38;
        v39 = v157;
      }
      v157 = v39 + 8;
    }
    v40 = v159;
    v41 = (__int64)v160;
    sub_C7D6A0(0, 0, 8);
    v34 = v164;
    if ( v164 )
    {
      v132 = v164;
      v28 = 16LL * v164;
      v29 = (void *)sub_C7D670(v28, 8);
      v30 = v163;
      v31 = memcpy(v29, src, v28);
      v32 = HIDWORD(v30);
      v33 = v30;
      v34 = v132;
      v35 = v31;
    }
    else
    {
      v32 = 0;
      v33 = 0;
      v35 = 0;
    }
    v36 = v157;
    v168 = v35;
    v37 = v156;
    v170 = v34;
    v27 += 32;
    v169 = __PAIR64__(v32, v33);
    v165 = (__int64 *)v40;
    v166 = v41;
    v167 = 1;
    sub_C7D6A0(0, 0, 8);
    sub_353F520(v37, ((__int64)&v36[-v37] >> 3) - 1, 0, *((_QWORD *)v36 - 1), (__int64 *)&v165);
    sub_C7D6A0((__int64)v168, 16LL * v170, 8);
    sub_C7D6A0(0, 0, 8);
  }
  v165 = &v167;
  v166 = 0x800000000LL;
  v56 = a1[3];
  v57 = *(__int64 (**)())(*(_QWORD *)v56 + 1256LL);
  if ( v57 == sub_2FDC7B0 )
  {
    v58 = &v167;
    v59 = v152;
    v152[0] = 0;
  }
  else
  {
    v124 = ((__int64 (__fastcall *)(__int64, _QWORD))v57)(v56, a1[2]);
    v127 = (__int64)v165;
    v60 = v124;
    v128 = (unsigned int)v166 + 1LL;
    v61 = v166;
    v58 = &v165[(unsigned int)v166];
    v152[0] = v60;
    v59 = v152;
    if ( v128 > HIDWORD(v166) )
    {
      if ( v165 > v152 || v58 <= v152 )
      {
        sub_354B1B0((__int64)&v165, v128, (__int64)v152, (__int64)v58, v125, v126);
        v61 = v166;
        v58 = &v165[(unsigned int)v166];
        v59 = v152;
      }
      else
      {
        sub_354B1B0((__int64)&v165, v128, (__int64)v152, (__int64)v58, v125, v126);
        v59 = (_QWORD *)((char *)v152 + (_QWORD)v165 - v127);
        v61 = v166;
        v58 = &v165[(unsigned int)v166];
      }
    }
    if ( !v58 )
      goto LABEL_47;
  }
  *v58 = *v59;
  *v59 = 0;
  v60 = v152[0];
  v61 = v166;
LABEL_47:
  LODWORD(v166) = v61 + 1;
  if ( v60 )
  {
    v62 = *(volatile signed __int32 **)(v60 + 32);
    if ( v62 )
    {
      if ( &_pthread_key_create )
      {
        v63 = _InterlockedExchangeAdd(v62 + 2, 0xFFFFFFFF);
      }
      else
      {
        v63 = *((_DWORD *)v62 + 2);
        *((_DWORD *)v62 + 2) = v63 - 1;
      }
      if ( v63 == 1 )
      {
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v62 + 16LL))(v62);
        if ( &_pthread_key_create )
        {
          v64 = _InterlockedExchangeAdd(v62 + 3, 0xFFFFFFFF);
        }
        else
        {
          v64 = *((_DWORD *)v62 + 3);
          *((_DWORD *)v62 + 3) = v64 - 1;
        }
        if ( v64 == 1 )
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v62 + 24LL))(v62);
      }
    }
    v65 = *(volatile signed __int32 **)(v60 + 16);
    if ( v65 )
    {
      if ( &_pthread_key_create )
      {
        v66 = _InterlockedExchangeAdd(v65 + 2, 0xFFFFFFFF);
      }
      else
      {
        v66 = *((_DWORD *)v65 + 2);
        *((_DWORD *)v65 + 2) = v66 - 1;
      }
      if ( v66 == 1 )
      {
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v65 + 16LL))(v65);
        if ( &_pthread_key_create )
        {
          v67 = _InterlockedExchangeAdd(v65 + 3, 0xFFFFFFFF);
        }
        else
        {
          v67 = *((_DWORD *)v65 + 3);
          *((_DWORD *)v65 + 3) = v67 - 1;
        }
        if ( v67 == 1 )
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v65 + 24LL))(v65);
      }
    }
    j_j___libc_free_0(v60);
  }
LABEL_65:
  v68 = v157;
  while ( (_BYTE *)v156 != v68 )
  {
    v69 = *(_QWORD *)v156;
    v140 = v159;
    v137 = v160;
    sub_C7D6A0(0, 0, 8);
    v70 = v164;
    if ( v164 )
    {
      v71 = 16LL * v164;
      v72 = (void *)sub_C7D670(v71, 8);
      v133 = v163;
      memcpy(v72, src, 16LL * v70);
    }
    else
    {
      v71 = 0;
      v72 = 0;
      v133 = 0;
    }
    v73 = (_QWORD *)v156;
    if ( (__int64)&v157[-v156] > 8 )
    {
      v130 = v157;
      sub_C7D6A0(0, 0, 8);
      v102 = v130;
      v131 = (__int64 *)(v130 - 8);
      v103 = *v131;
      *((_QWORD *)v102 - 1) = *v73;
      v129 = v103;
      v152[0] = v140;
      v152[2] = 0;
      v152[1] = v137;
      v153 = 0;
      v154 = 0;
      v155 = 0;
      sub_C7D6A0(0, 0, 8);
      v155 = v70;
      v104 = v129;
      if ( v70 )
      {
        v153 = (void *)sub_C7D670(v71, 8);
        v154 = v133;
        memcpy(v153, v72, 16LL * v155);
        v104 = v129;
      }
      else
      {
        v153 = 0;
        v154 = 0;
      }
      sub_353F9B0((__int64)v73, 0, v131 - v73, v104, (__int64)v152);
      sub_C7D6A0((__int64)v153, 16LL * v155, 8);
      sub_C7D6A0((__int64)v72, v71, 8);
      v71 = 0;
      v72 = 0;
    }
    sub_C7D6A0((__int64)v72, v71, 8);
    v68 = v157 - 8;
    v157 -= 8;
    if ( *(_WORD *)(v69 + 68) > 0x14u )
    {
      v74 = a1[4];
      v75 = *(_QWORD *)(v74 + 944);
      v76 = *(_DWORD *)(v74 + 960);
      if ( !v76 )
        goto LABEL_211;
      v77 = v76 - 1;
      v78 = (v76 - 1) & (((unsigned int)v69 >> 9) ^ ((unsigned int)v69 >> 4));
      v79 = (__int64 *)(v75 + 16LL * v78);
      v80 = *v79;
      if ( v69 != *v79 )
      {
        v122 = 1;
        while ( v80 != -4096 )
        {
          v123 = v122 + 1;
          v78 = v77 & (v122 + v78);
          v79 = (__int64 *)(v75 + 16LL * v78);
          v80 = *v79;
          if ( v69 == *v79 )
            goto LABEL_73;
          v122 = v123;
        }
LABEL_211:
        BUG();
      }
LABEL_73:
      v81 = v165;
      v82 = &v165[(unsigned int)v166];
      v141 = *(unsigned __int16 *)(v79[1] + 252);
      if ( *(_WORD *)(v79[1] + 252) )
      {
        v83 = 0;
        v84 = 0;
        while ( 1 )
        {
          if ( v82 != v81 )
          {
            while ( !(unsigned __int8)sub_37F0CA0(*v81, v69) )
            {
              if ( ++v81 == v82 )
                goto LABEL_80;
            }
            ++v84;
            sub_37F1A40(*v81, v69);
          }
LABEL_80:
          if ( v141 == v83 + 1 )
            break;
          ++v83;
        }
        if ( v84 < v141 )
        {
          while ( 1 )
          {
            v98 = a1[3];
            v93 = 0;
            v99 = *(__int64 (**)())(*(_QWORD *)v98 + 1256LL);
            if ( v99 != sub_2FDC7B0 )
              v93 = ((__int64 (__fastcall *)(__int64, _QWORD))v99)(v98, a1[2]);
            sub_37F1A40(v93, v69);
            v87 = (unsigned int)v166;
            v152[0] = v93;
            v88 = (unsigned int)v166 + 1LL;
            v89 = v166;
            if ( v88 > HIDWORD(v166) )
            {
              if ( v165 > v152
                || (v87 = (unsigned __int64)&v165[(unsigned int)v166],
                    v142 = (__int64)v165,
                    (unsigned __int64)v152 >= v87) )
              {
                sub_354B1B0((__int64)&v165, v88, v87, (__int64)v152, v85, v86);
                v87 = (unsigned int)v166;
                v90 = (__int64)v165;
                v91 = v152;
                v89 = v166;
              }
              else
              {
                sub_354B1B0((__int64)&v165, v88, v87, (__int64)v152, v85, v86);
                v90 = (__int64)v165;
                v87 = (unsigned int)v166;
                v91 = (_QWORD *)((char *)v152 + (_QWORD)v165 - v142);
                v89 = v166;
              }
            }
            else
            {
              v90 = (__int64)v165;
              v91 = v152;
            }
            v92 = (_QWORD *)(v90 + 8 * v87);
            if ( v92 )
            {
              *v92 = *v91;
              *v91 = 0;
              v93 = v152[0];
              v89 = v166;
            }
            LODWORD(v166) = v89 + 1;
            if ( v93 )
            {
              v94 = *(volatile signed __int32 **)(v93 + 32);
              if ( v94 )
              {
                if ( &_pthread_key_create )
                {
                  v95 = _InterlockedExchangeAdd(v94 + 2, 0xFFFFFFFF);
                }
                else
                {
                  v95 = *((_DWORD *)v94 + 2);
                  *((_DWORD *)v94 + 2) = v95 - 1;
                }
                if ( v95 == 1 )
                {
                  (*(void (**)(void))(*(_QWORD *)v94 + 16LL))();
                  if ( &_pthread_key_create )
                  {
                    v101 = _InterlockedExchangeAdd(v94 + 3, 0xFFFFFFFF);
                  }
                  else
                  {
                    v101 = *((_DWORD *)v94 + 3);
                    *((_DWORD *)v94 + 3) = v101 - 1;
                  }
                  if ( v101 == 1 )
                    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v94 + 24LL))(v94);
                }
              }
              v96 = *(volatile signed __int32 **)(v93 + 16);
              if ( v96 )
              {
                if ( &_pthread_key_create )
                {
                  v97 = _InterlockedExchangeAdd(v96 + 2, 0xFFFFFFFF);
                }
                else
                {
                  v97 = *((_DWORD *)v96 + 2);
                  *((_DWORD *)v96 + 2) = v97 - 1;
                }
                if ( v97 == 1 )
                {
                  (*(void (**)(void))(*(_QWORD *)v96 + 16LL))();
                  if ( &_pthread_key_create )
                  {
                    v100 = _InterlockedExchangeAdd(v96 + 3, 0xFFFFFFFF);
                  }
                  else
                  {
                    v100 = *((_DWORD *)v96 + 3);
                    *((_DWORD *)v96 + 3) = v100 - 1;
                  }
                  if ( v100 == 1 )
                    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v96 + 24LL))(v96);
                }
              }
              j_j___libc_free_0(v93);
            }
            if ( v83 == v84 )
              break;
            ++v84;
          }
        }
        goto LABEL_65;
      }
    }
  }
  v105 = (__int64)v165;
  v106 = &v165[(unsigned int)v166];
  v145 = v166;
  if ( v165 != v106 )
  {
    do
    {
      v107 = *--v106;
      if ( v107 )
      {
        v108 = *(volatile signed __int32 **)(v107 + 32);
        if ( v108 )
        {
          if ( &_pthread_key_create )
          {
            v109 = _InterlockedExchangeAdd(v108 + 2, 0xFFFFFFFF);
          }
          else
          {
            v109 = *((_DWORD *)v108 + 2);
            *((_DWORD *)v108 + 2) = v109 - 1;
          }
          if ( v109 == 1 )
          {
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v108 + 16LL))(v108);
            if ( &_pthread_key_create )
            {
              v115 = _InterlockedExchangeAdd(v108 + 3, 0xFFFFFFFF);
            }
            else
            {
              v115 = *((_DWORD *)v108 + 3);
              *((_DWORD *)v108 + 3) = v115 - 1;
            }
            if ( v115 == 1 )
              (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v108 + 24LL))(v108);
          }
        }
        v110 = *(volatile signed __int32 **)(v107 + 16);
        if ( v110 )
        {
          if ( &_pthread_key_create )
          {
            v111 = _InterlockedExchangeAdd(v110 + 2, 0xFFFFFFFF);
          }
          else
          {
            v111 = *((_DWORD *)v110 + 2);
            *((_DWORD *)v110 + 2) = v111 - 1;
          }
          if ( v111 == 1 )
          {
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v110 + 16LL))(v110);
            if ( &_pthread_key_create )
            {
              v114 = _InterlockedExchangeAdd(v110 + 3, 0xFFFFFFFF);
            }
            else
            {
              v114 = *((_DWORD *)v110 + 3);
              *((_DWORD *)v110 + 3) = v114 - 1;
            }
            if ( v114 == 1 )
              (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v110 + 24LL))(v110);
          }
        }
        j_j___libc_free_0(v107);
      }
    }
    while ( (__int64 *)v105 != v106 );
    v106 = v165;
  }
  if ( v106 != &v167 )
    _libc_free((unsigned __int64)v106);
  sub_C7D6A0((__int64)src, 16LL * v164, 8);
  if ( v156 )
    j_j___libc_free_0(v156);
  sub_C7D6A0((__int64)v149, 16LL * v151, 8);
  return v145;
}
