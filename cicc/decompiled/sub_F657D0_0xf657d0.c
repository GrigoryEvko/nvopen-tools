// Function: sub_F657D0
// Address: 0xf657d0
//
void __fastcall sub_F657D0(__int64 a1, __int64 **a2)
{
  __int64 v2; // rsi
  __int64 v4; // rdi
  __int64 v5; // rdi
  __int64 v6; // rdx
  __int64 v7; // r12
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // r11
  _BYTE *v11; // r14
  __int64 v12; // rdx
  __int64 v13; // rbx
  _QWORD *v14; // rax
  _QWORD *v15; // r13
  int v16; // ebx
  __int64 v17; // rcx
  _QWORD *v18; // rax
  _BYTE *v19; // rdx
  __int64 v20; // rdx
  __int64 *v21; // rax
  __int64 *v22; // r14
  __int64 v23; // rbx
  __int64 v24; // r13
  __int64 v25; // rax
  unsigned __int64 v26; // rax
  __int64 v27; // rdi
  __int64 v28; // rax
  unsigned __int8 **v29; // r8
  unsigned __int8 **v30; // rdi
  unsigned __int8 **v31; // r15
  unsigned __int8 *v32; // r12
  unsigned int v33; // ecx
  unsigned __int8 **v34; // rdx
  unsigned __int8 *v35; // r9
  __int64 v36; // r13
  int v37; // r11d
  unsigned int j; // eax
  _QWORD *v39; // rcx
  unsigned int v40; // eax
  _QWORD *v41; // r10
  int v42; // eax
  unsigned int v43; // edx
  int v44; // r9d
  _QWORD *v45; // rdi
  int v46; // r9d
  unsigned int v47; // edx
  __int64 v48; // rdi
  __int64 v49; // rsi
  __int64 v50; // rsi
  _QWORD *v51; // rax
  _QWORD *v52; // rdx
  char v53; // si
  __int64 v54; // rbx
  __int64 v55; // r14
  __int64 v56; // rax
  unsigned int v57; // eax
  __int64 v58; // rax
  _QWORD *v59; // rbx
  _BYTE *v60; // rax
  __int64 v61; // rdx
  _QWORD *v62; // rax
  _QWORD *v63; // r12
  __int64 v64; // r13
  __int64 *v65; // rbx
  __int64 v66; // rax
  __int64 v67; // r12
  __int64 v68; // r13
  __int64 v69; // rax
  unsigned __int64 v70; // rax
  __int64 v71; // rdi
  __int64 v72; // rax
  __int64 *v73; // r9
  __int64 *v74; // rdi
  __int64 *v75; // r14
  __int64 v76; // r13
  unsigned int v77; // ecx
  _QWORD *v78; // rdx
  __int64 v79; // r10
  _BYTE *v80; // r12
  __int64 v81; // rax
  char v82; // dl
  __int64 v83; // r12
  __m128i v84; // rax
  __int64 *v85; // rdx
  __int64 *v86; // r12
  __int64 *v87; // rbx
  __int64 v88; // r13
  char v89; // al
  char v90; // dl
  __int64 v91; // rax
  _QWORD *v92; // r14
  __int16 v93; // dx
  __int64 *v94; // rbx
  __int64 *v95; // r12
  unsigned __int8 v96; // al
  __int64 v97; // r14
  __int64 v98; // r15
  unsigned __int8 v99; // dl
  __int64 v100; // r8
  int v101; // edx
  int v102; // r8d
  __int64 v103; // rax
  __int64 v104; // r13
  __int64 v105; // rax
  _QWORD *v106; // rdx
  _QWORD *v107; // rdx
  unsigned int v108; // eax
  _QWORD *v109; // rbx
  __int64 v110; // rax
  _QWORD *v111; // r12
  __int64 v112; // rdx
  __int64 v113; // rax
  __int64 v114; // rax
  int i; // edx
  int v116; // r8d
  __int64 v117; // rax
  _QWORD *v118; // rbx
  __int64 v119; // rax
  _QWORD *v120; // r12
  __int64 v121; // rdx
  __int64 v122; // rax
  __int64 v123; // rax
  _QWORD *v124; // r12
  _QWORD *v125; // rbx
  __int64 v126; // rsi
  __int64 v127; // rax
  _QWORD *v128; // r12
  _QWORD *v129; // rbx
  __int64 v130; // rsi
  __int64 v131; // [rsp+20h] [rbp-160h]
  __int64 v132; // [rsp+30h] [rbp-150h]
  unsigned __int8 *v134; // [rsp+40h] [rbp-140h]
  __int64 v136; // [rsp+48h] [rbp-138h]
  __int64 *v137; // [rsp+50h] [rbp-130h]
  int v138; // [rsp+50h] [rbp-130h]
  __int64 v139; // [rsp+50h] [rbp-130h]
  __int64 *v140; // [rsp+50h] [rbp-130h]
  __int64 v141; // [rsp+58h] [rbp-128h]
  unsigned __int64 v142; // [rsp+60h] [rbp-120h]
  __int64 v143; // [rsp+60h] [rbp-120h]
  __int64 v144; // [rsp+68h] [rbp-118h]
  __int64 v145; // [rsp+68h] [rbp-118h]
  __int64 v146; // [rsp+68h] [rbp-118h]
  __int64 *v147; // [rsp+68h] [rbp-118h]
  __int64 v148; // [rsp+90h] [rbp-F0h] BYREF
  _QWORD *v149; // [rsp+98h] [rbp-E8h]
  __m128i v150; // [rsp+A0h] [rbp-E0h] BYREF
  __int64 v151; // [rsp+B0h] [rbp-D0h] BYREF
  __int64 v152; // [rsp+B8h] [rbp-C8h]
  __int64 v153; // [rsp+C0h] [rbp-C0h]
  char *v154; // [rsp+D0h] [rbp-B0h] BYREF
  __int64 v155; // [rsp+D8h] [rbp-A8h] BYREF
  __int64 v156; // [rsp+E0h] [rbp-A0h]
  __int64 v157; // [rsp+E8h] [rbp-98h]
  __int64 *k; // [rsp+F0h] [rbp-90h]
  __int64 v159; // [rsp+F8h] [rbp-88h]
  __int64 v160; // [rsp+100h] [rbp-80h] BYREF
  _QWORD *v161; // [rsp+108h] [rbp-78h]
  __int64 v162; // [rsp+110h] [rbp-70h]
  unsigned int v163; // [rsp+118h] [rbp-68h]
  __int64 *v164; // [rsp+120h] [rbp-60h]
  __int64 v165; // [rsp+128h] [rbp-58h]
  char v166; // [rsp+130h] [rbp-50h] BYREF
  unsigned int v167; // [rsp+138h] [rbp-48h]
  char v168; // [rsp+140h] [rbp-40h]

  v2 = *((unsigned int *)a2 + 2);
  if ( !(_DWORD)v2 )
    return;
  v154 = 0;
  v4 = *(_QWORD *)(a1 + 56);
  v155 = 0;
  v156 = 0;
  LODWORD(v157) = 0;
  v144 = v4;
  v141 = a1 + 48;
  if ( v4 == a1 + 48 )
  {
    v48 = 0;
    v49 = 0;
    goto LABEL_73;
  }
  do
  {
    if ( !v144 )
      BUG();
    v5 = *(_QWORD *)(v144 + 40);
    if ( v5 )
    {
      v7 = sub_B14240(v5);
      if ( v6 != v7 )
      {
        while ( *(_BYTE *)(v7 + 32) )
        {
          v7 = *(_QWORD *)(v7 + 8);
          if ( v7 == v6 )
            goto LABEL_26;
        }
        if ( v6 != v7 )
        {
          v8 = v6;
          while ( 1 )
          {
            v2 = v7;
            sub_B129C0(&v160, v7);
            v9 = v160;
            v10 = (__int64)v161;
            if ( v161 != (_QWORD *)v160 )
              break;
            do
            {
LABEL_25:
              v7 = *(_QWORD *)(v7 + 8);
              if ( v8 == v7 )
                goto LABEL_26;
            }
            while ( *(_BYTE *)(v7 + 32) );
            if ( v8 == v7 )
              goto LABEL_26;
          }
          while ( 1 )
          {
            v13 = v9;
            v14 = (_QWORD *)(v9 & 0xFFFFFFFFFFFFFFF8LL);
            v15 = v14;
            v16 = (v13 >> 2) & 1;
            if ( v16 )
            {
              v11 = *(_BYTE **)(*v14 + 136LL);
              if ( !v11 || *v11 != 84 )
              {
LABEL_23:
                v9 = (unsigned __int64)(v15 + 1) | 4;
                v12 = v9;
                goto LABEL_16;
              }
            }
            else
            {
              v11 = (_BYTE *)v14[17];
              if ( !v11 || *v11 != 84 )
                goto LABEL_15;
            }
            v2 = (unsigned int)v157;
            if ( (_DWORD)v157 )
            {
              v17 = ((_DWORD)v157 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
              v18 = (_QWORD *)(v155 + 16 * v17);
              v19 = (_BYTE *)*v18;
              if ( v11 == (_BYTE *)*v18 )
                goto LABEL_22;
              v138 = 1;
              v41 = 0;
              while ( v19 != (_BYTE *)-4096LL )
              {
                if ( v19 != (_BYTE *)-8192LL || v41 )
                  v18 = v41;
                LODWORD(v17) = (v157 - 1) & (v138 + v17);
                v19 = *(_BYTE **)(v155 + 16LL * (unsigned int)v17);
                if ( v19 == v11 )
                  goto LABEL_22;
                ++v138;
                v41 = v18;
                v18 = (_QWORD *)(v155 + 16LL * (unsigned int)v17);
              }
              if ( !v41 )
                v41 = v18;
              ++v154;
              v42 = v156 + 1;
              if ( 4 * ((int)v156 + 1) < (unsigned int)(3 * v157) )
              {
                if ( (int)v157 - HIDWORD(v156) - v42 > (unsigned int)v157 >> 3 )
                  goto LABEL_53;
                v132 = v10;
                sub_F61420((__int64)&v154, v157);
                if ( !(_DWORD)v157 )
                {
LABEL_264:
                  LODWORD(v156) = v156 + 1;
                  BUG();
                }
                v45 = 0;
                v10 = v132;
                v42 = v156 + 1;
                v46 = 1;
                v47 = (v157 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
                v41 = (_QWORD *)(v155 + 16LL * v47);
                v2 = *v41;
                if ( v11 == (_BYTE *)*v41 )
                  goto LABEL_53;
                while ( v2 != -4096 )
                {
                  if ( !v45 && v2 == -8192 )
                    v45 = v41;
                  v47 = (v157 - 1) & (v46 + v47);
                  v41 = (_QWORD *)(v155 + 16LL * v47);
                  v2 = *v41;
                  if ( (_BYTE *)*v41 == v11 )
                    goto LABEL_53;
                  ++v46;
                }
                goto LABEL_61;
              }
            }
            else
            {
              ++v154;
            }
            v139 = v10;
            sub_F61420((__int64)&v154, 2 * v157);
            if ( !(_DWORD)v157 )
              goto LABEL_264;
            v10 = v139;
            v43 = (v157 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
            v42 = v156 + 1;
            v41 = (_QWORD *)(v155 + 16LL * v43);
            v2 = *v41;
            if ( v11 == (_BYTE *)*v41 )
              goto LABEL_53;
            v44 = 1;
            v45 = 0;
            while ( v2 != -4096 )
            {
              if ( v2 == -8192 && !v45 )
                v45 = v41;
              v43 = (v157 - 1) & (v44 + v43);
              v41 = (_QWORD *)(v155 + 16LL * v43);
              v2 = *v41;
              if ( (_BYTE *)*v41 == v11 )
                goto LABEL_53;
              ++v44;
            }
LABEL_61:
            if ( v45 )
              v41 = v45;
LABEL_53:
            LODWORD(v156) = v42;
            if ( *v41 != -4096 )
              --HIDWORD(v156);
            *v41 = v11;
            v41[1] = v7;
LABEL_22:
            if ( v16 )
              goto LABEL_23;
LABEL_15:
            v12 = (__int64)(v15 + 18);
            v9 = (__int64)(v15 + 18);
            if ( !v15 )
              goto LABEL_23;
LABEL_16:
            if ( v10 == v12 )
              goto LABEL_25;
          }
        }
      }
    }
LABEL_26:
    v144 = *(_QWORD *)(v144 + 8);
  }
  while ( v141 != v144 );
  if ( !(_DWORD)v156 )
  {
    v48 = v155;
    v49 = 16LL * (unsigned int)v157;
LABEL_73:
    sub_C7D6A0(v48, v49, 8);
    goto LABEL_74;
  }
  v160 = 0;
  v164 = (__int64 *)&v166;
  v20 = *((unsigned int *)a2 + 2);
  v21 = *a2;
  v161 = 0;
  v162 = 0;
  v163 = 0;
  v165 = 0;
  v137 = &v21[v20];
  if ( v21 == v137 )
    goto LABEL_206;
  v22 = v21;
  while ( 2 )
  {
    v23 = *v22;
    v24 = *(_QWORD *)(*v22 + 40);
    v25 = sub_AA4FF0(v24);
    if ( !v25 )
      BUG();
    v26 = (unsigned int)*(unsigned __int8 *)(v25 - 24) - 39;
    if ( (unsigned int)v26 <= 0x38 )
    {
      v27 = 0x100060000000001LL;
      if ( _bittest64(&v27, v26) )
        goto LABEL_135;
    }
    v28 = 4LL * (*(_DWORD *)(v23 + 4) & 0x7FFFFFF);
    if ( (*(_BYTE *)(v23 + 7) & 0x40) != 0 )
    {
      v29 = *(unsigned __int8 ***)(v23 - 8);
      v30 = &v29[v28];
    }
    else
    {
      v30 = (unsigned __int8 **)v23;
      v29 = (unsigned __int8 **)(v23 - v28 * 8);
    }
    if ( v29 == v30 )
      goto LABEL_135;
    v145 = v24;
    v31 = v29;
    v142 = (unsigned __int64)(((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4)) << 32;
    while ( 2 )
    {
      v32 = *v31;
      v2 = v155;
      if ( !(_DWORD)v157 )
        goto LABEL_134;
      v33 = (v157 - 1) & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
      v34 = (unsigned __int8 **)(v155 + 16LL * v33);
      v35 = *v34;
      if ( v32 != *v34 )
      {
        for ( i = 1; ; i = v116 )
        {
          if ( v35 == (unsigned __int8 *)-4096LL )
            goto LABEL_134;
          v116 = i + 1;
          v33 = (v157 - 1) & (i + v33);
          v34 = (unsigned __int8 **)(v155 + 16LL * v33);
          v35 = *v34;
          if ( v32 == *v34 )
            break;
        }
      }
      if ( v34 == (unsigned __int8 **)(v155 + 16LL * (unsigned int)v157) )
        goto LABEL_134;
      v36 = (__int64)v34[1];
      if ( !v163 )
        goto LABEL_197;
      v37 = 1;
      for ( j = (v163 - 1)
              & (((0xBF58476D1CE4E5B9LL * (v142 | ((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4))) >> 31)
               ^ (484763065 * (v142 | ((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4)))); ; j = (v163 - 1) & v40 )
      {
        v39 = &v161[3 * j];
        if ( v145 == *v39 && v36 == v39[1] )
          break;
        if ( *v39 == -4096 && v39[1] == -4096 )
          goto LABEL_197;
        v40 = v37 + j;
        ++v37;
      }
      if ( v39 == &v161[3 * v163]
        || (v103 = (__int64)&v164[3 * *((unsigned int *)v39 + 4)], (__int64 *)v103 == &v164[3 * (unsigned int)v165]) )
      {
LABEL_197:
        v114 = sub_B13070(v36);
        v150.m128i_i64[1] = v36;
        v150.m128i_i64[0] = v145;
        v151 = v114;
        v103 = sub_F653C0((__int64)&v160, &v150, &v151);
      }
      v104 = *(_QWORD *)(v103 + 16);
      sub_B129C0(&v150, v104);
      v2 = v150.m128i_i64[1];
      v105 = v150.m128i_i64[0];
      v148 = v150.m128i_i64[0];
      if ( v150.m128i_i64[1] == v150.m128i_i64[0] )
        goto LABEL_132;
      while ( 2 )
      {
        v107 = (_QWORD *)(v105 & 0xFFFFFFFFFFFFFFF8LL);
        if ( (v105 & 4) != 0 )
        {
          if ( v32 == *(unsigned __int8 **)(*v107 + 136LL) )
            break;
          goto LABEL_164;
        }
        if ( v32 != (unsigned __int8 *)v107[17] )
        {
          if ( v107 )
          {
            v106 = v107 + 18;
            v105 = (__int64)v106;
LABEL_165:
            if ( v106 == (_QWORD *)v150.m128i_i64[1] )
              goto LABEL_132;
            continue;
          }
LABEL_164:
          v105 = (unsigned __int64)(v107 + 1) | 4;
          v106 = (_QWORD *)v105;
          goto LABEL_165;
        }
        break;
      }
      v2 = v105;
LABEL_132:
      if ( v2 != v150.m128i_i64[1] )
      {
        v2 = (__int64)v32;
        sub_B13360(v104, v32, (unsigned __int8 *)v23, 0);
      }
LABEL_134:
      v31 += 4;
      if ( v30 != v31 )
        continue;
      break;
    }
LABEL_135:
    if ( v137 != ++v22 )
      continue;
    break;
  }
  v94 = v164;
  v95 = &v164[3 * (unsigned int)v165];
  if ( v164 != v95 )
  {
    do
    {
      v97 = *v94;
      v98 = v94[2];
      v100 = sub_AA5190(*v94);
      if ( v100 )
        v96 = v99;
      else
        v96 = 0;
      v2 = v98;
      v94 += 3;
      sub_AA8770(v97, v98, v100, v96);
    }
    while ( v95 != v94 );
    v95 = v164;
  }
  if ( v95 != (__int64 *)&v166 )
    _libc_free(v95, v2);
LABEL_206:
  sub_C7D6A0((__int64)v161, 24LL * v163, 8);
  sub_C7D6A0(v155, 16LL * (unsigned int)v157, 8);
LABEL_74:
  v50 = 8;
  v160 = 0;
  v163 = 128;
  v51 = (_QWORD *)sub_C7D670(0x2000, 8);
  v162 = 0;
  v161 = v51;
  v155 = 2;
  v52 = &v51[8 * (unsigned __int64)v163];
  v154 = (char *)&unk_49DD7B0;
  v156 = 0;
  v157 = -4096;
  for ( k = 0; v52 != v51; v51 += 8 )
  {
    if ( v51 )
    {
      v53 = v155;
      v51[2] = 0;
      v51[3] = -4096;
      *v51 = &unk_49DD7B0;
      v51[1] = v53 & 6;
      v50 = (__int64)k;
      v51[4] = k;
    }
  }
  v54 = v141;
  v168 = 0;
  v55 = *(_QWORD *)(a1 + 56);
  if ( v55 != v141 )
  {
    while ( 2 )
    {
      while ( 2 )
      {
        if ( !v55 )
          BUG();
        if ( *(_BYTE *)(v55 - 24) != 85 )
          goto LABEL_80;
        v56 = *(_QWORD *)(v55 - 56);
        if ( !v56
          || *(_BYTE *)v56
          || *(_QWORD *)(v56 + 24) != *(_QWORD *)(v55 + 56)
          || (*(_BYTE *)(v56 + 33) & 0x20) == 0 )
        {
          goto LABEL_80;
        }
        v57 = *(_DWORD *)(v56 + 36);
        if ( v57 <= 0x45 )
        {
          if ( v57 > 0x43 )
            goto LABEL_89;
LABEL_80:
          v55 = *(_QWORD *)(v55 + 8);
          if ( v55 == v54 )
            goto LABEL_99;
          continue;
        }
        break;
      }
      if ( v57 != 71 )
        goto LABEL_80;
LABEL_89:
      v50 = v55 - 24;
      sub_B58E30(&v148, v55 - 24);
      v58 = v148;
      if ( v149 == (_QWORD *)v148 )
        goto LABEL_80;
      v146 = v54;
      v59 = v149;
LABEL_94:
      v61 = v58;
      v62 = (_QWORD *)(v58 & 0xFFFFFFFFFFFFFFF8LL);
      v63 = v62;
      LODWORD(v61) = (v61 >> 2) & 1;
      v64 = (unsigned int)v61;
      if ( (_DWORD)v61 )
      {
        v60 = *(_BYTE **)(*v62 + 136LL);
        if ( !v60 || *v60 != 84 )
        {
LABEL_97:
          v58 = (unsigned __int64)(v63 + 1) | 4;
          if ( v59 == (_QWORD *)v58 )
          {
LABEL_98:
            v54 = v146;
            v55 = *(_QWORD *)(v55 + 8);
            if ( v55 == v146 )
            {
LABEL_99:
              if ( (_DWORD)v162 )
                goto LABEL_100;
              if ( v168 )
              {
                v127 = v167;
                v168 = 0;
                if ( v167 )
                {
                  v128 = (_QWORD *)v165;
                  v129 = (_QWORD *)(v165 + 16LL * v167);
                  do
                  {
                    if ( *v128 != -8192 && *v128 != -4096 )
                    {
                      v130 = v128[1];
                      if ( v130 )
                        sub_B91220((__int64)(v128 + 1), v130);
                    }
                    v128 += 2;
                  }
                  while ( v129 != v128 );
                  v127 = v167;
                }
                sub_C7D6A0(v165, 16 * v127, 8);
              }
LABEL_208:
              v117 = v163;
              if ( v163 )
              {
                v118 = v161;
                v150.m128i_i64[1] = 2;
                v151 = 0;
                v119 = -4096;
                v120 = &v161[8 * (unsigned __int64)v163];
                v152 = -4096;
                v150.m128i_i64[0] = (__int64)&unk_49DD7B0;
                v153 = 0;
                v155 = 2;
                v156 = 0;
                v157 = -8192;
                v154 = (char *)&unk_49DD7B0;
                k = 0;
                while ( 1 )
                {
                  v121 = v118[3];
                  if ( v121 != v119 )
                  {
                    v119 = v157;
                    if ( v121 != v157 )
                    {
                      v122 = v118[7];
                      if ( v122 != 0 && v122 != -4096 && v122 != -8192 )
                      {
                        sub_BD60C0(v118 + 5);
                        v121 = v118[3];
                      }
                      v119 = v121;
                    }
                  }
                  *v118 = &unk_49DB368;
                  if ( v119 != 0 && v119 != -4096 && v119 != -8192 )
                    sub_BD60C0(v118 + 1);
                  v118 += 8;
                  if ( v120 == v118 )
                    break;
                  v119 = v152;
                }
                v154 = (char *)&unk_49DB368;
                if ( v157 != -4096 && v157 != 0 && v157 != -8192 )
                  sub_BD60C0(&v155);
                v150.m128i_i64[0] = (__int64)&unk_49DB368;
                if ( v152 != 0 && v152 != -4096 && v152 != -8192 )
                  sub_BD60C0(&v150.m128i_i64[1]);
                v117 = v163;
              }
              sub_C7D6A0((__int64)v161, v117 << 6, 8);
              return;
            }
            continue;
          }
          goto LABEL_94;
        }
      }
      else
      {
        v60 = (_BYTE *)v62[17];
        if ( !v60 || *v60 != 84 )
          goto LABEL_93;
      }
      break;
    }
    v150.m128i_i64[0] = (__int64)v60;
    v150.m128i_i64[1] = 6;
    v151 = 0;
    v152 = v55 - 24;
    if ( v55 != -4072 && v55 != -8168 )
      sub_BD73F0((__int64)&v150.m128i_i64[1]);
    v50 = (__int64)&v160;
    sub_F621C0((__int64)&v154, (__int64)&v160, v150.m128i_i64);
    LOBYTE(v50) = v152 != -4096;
    if ( ((v152 != 0) & (unsigned __int8)v50) != 0 && v152 != -8192 )
      sub_BD60C0(&v150.m128i_i64[1]);
    if ( !v64 )
    {
LABEL_93:
      v58 = (__int64)(v63 + 18);
      if ( v59 == v63 + 18 )
        goto LABEL_98;
      goto LABEL_94;
    }
    goto LABEL_97;
  }
  if ( !(_DWORD)v162 )
    goto LABEL_208;
LABEL_100:
  v154 = 0;
  k = &v160;
  v155 = 0;
  v156 = 0;
  v65 = *a2;
  LODWORD(v157) = 0;
  v66 = *((unsigned int *)a2 + 2);
  v159 = 0;
  v140 = &v65[v66];
  if ( v140 != v65 )
  {
    v147 = v65;
    do
    {
      v67 = *v147;
      v68 = *(_QWORD *)(*v147 + 40);
      v69 = sub_AA4FF0(v68);
      if ( !v69 )
        BUG();
      v70 = (unsigned int)*(unsigned __int8 *)(v69 - 24) - 39;
      if ( (unsigned int)v70 > 0x38 || (v71 = 0x100060000000001LL, !_bittest64(&v71, v70)) )
      {
        v72 = 4LL * (*(_DWORD *)(v67 + 4) & 0x7FFFFFF);
        if ( (*(_BYTE *)(v67 + 7) & 0x40) != 0 )
        {
          v73 = *(__int64 **)(v67 - 8);
          v74 = &v73[v72];
        }
        else
        {
          v74 = (__int64 *)v67;
          v73 = (__int64 *)(v67 - v72 * 8);
        }
        v75 = v73;
        if ( v74 != v73 )
        {
          v134 = (unsigned __int8 *)v67;
          v143 = v68;
          do
          {
            if ( v163 )
            {
              v76 = *v75;
              v50 = (__int64)v161;
              v77 = (v163 - 1) & (((unsigned int)*v75 >> 9) ^ ((unsigned int)*v75 >> 4));
              v78 = &v161[8 * (unsigned __int64)v77];
              v79 = v78[3];
              if ( *v75 == v79 )
              {
LABEL_111:
                if ( v78 != &v161[8 * (unsigned __int64)v163] )
                {
                  v80 = (_BYTE *)v78[7];
                  v150.m128i_i64[0] = v143;
                  v150.m128i_i64[1] = (__int64)v80;
                  v81 = sub_F62A00((__int64)&v154, &v150);
                  if ( v82 )
                  {
                    v136 = v81;
                    v83 = sub_B47F80(v80);
                    *(_QWORD *)(v136 + 16) = v83;
                  }
                  else
                  {
                    v83 = *(_QWORD *)(v81 + 16);
                  }
                  v50 = v83;
                  sub_B58E30(&v150, v83);
                  v84 = v150;
                  v148 = v150.m128i_i64[0];
                  if ( v150.m128i_i64[1] != v150.m128i_i64[0] )
                  {
                    while ( 1 )
                    {
                      while ( 1 )
                      {
                        v85 = (__int64 *)(v84.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL);
                        if ( (v84.m128i_i8[0] & 4) == 0 )
                          break;
                        v50 = *v85;
                        if ( v76 == *(_QWORD *)(*v85 + 136) )
                          goto LABEL_152;
                        v84.m128i_i64[0] = (unsigned __int64)(v85 + 1) | 4;
                        v84.m128i_i64[1] = v84.m128i_i64[0];
                        if ( v150.m128i_i64[1] == v84.m128i_i64[0] )
                          goto LABEL_121;
                      }
                      if ( v76 == v85[17] )
                        break;
                      v84.m128i_i64[1] = (__int64)(v85 + 18);
                      v84.m128i_i64[0] = v84.m128i_i64[1];
                      if ( v150.m128i_i64[1] == v84.m128i_i64[1] )
                        goto LABEL_121;
                    }
LABEL_152:
                    v84.m128i_i64[1] = v84.m128i_i64[0];
                  }
LABEL_121:
                  if ( v84.m128i_i64[1] != v150.m128i_i64[1] )
                  {
                    v50 = v76;
                    sub_B59720(v83, v76, v134);
                  }
                }
              }
              else
              {
                v101 = 1;
                while ( v79 != -4096 )
                {
                  v102 = v101 + 1;
                  v77 = (v163 - 1) & (v101 + v77);
                  v78 = &v161[8 * (unsigned __int64)v77];
                  v79 = v78[3];
                  if ( v76 == v79 )
                    goto LABEL_111;
                  v101 = v102;
                }
              }
            }
            v75 += 4;
          }
          while ( v74 != v75 );
        }
      }
      ++v147;
    }
    while ( v140 != v147 );
    v86 = k;
    v87 = &k[3 * (unsigned int)v159];
    if ( v87 != k )
    {
      v88 = v131;
      do
      {
        v92 = (_QWORD *)v86[2];
        v50 = sub_AA5190(*v86);
        if ( v50 )
        {
          v89 = v93;
          v90 = HIBYTE(v93);
        }
        else
        {
          v90 = 0;
          v89 = 0;
        }
        LOBYTE(v88) = v89;
        v86 += 3;
        v91 = v88;
        BYTE1(v91) = v90;
        sub_B44220(v92, v50, v91);
      }
      while ( v87 != v86 );
      v86 = k;
    }
    if ( v86 != &v160 )
      _libc_free(v86, v50);
  }
  sub_C7D6A0(v155, 24LL * (unsigned int)v157, 8);
  if ( v168 )
  {
    v123 = v167;
    v168 = 0;
    if ( v167 )
    {
      v124 = (_QWORD *)v165;
      v125 = (_QWORD *)(v165 + 16LL * v167);
      do
      {
        if ( *v124 != -8192 && *v124 != -4096 )
        {
          v126 = v124[1];
          if ( v126 )
            sub_B91220((__int64)(v124 + 1), v126);
        }
        v124 += 2;
      }
      while ( v125 != v124 );
      v123 = v167;
    }
    sub_C7D6A0(v165, 16 * v123, 8);
  }
  v108 = v163;
  if ( v163 )
  {
    v109 = v161;
    v150.m128i_i64[1] = 2;
    v151 = 0;
    v110 = -4096;
    v111 = &v161[8 * (unsigned __int64)v163];
    v152 = -4096;
    v150.m128i_i64[0] = (__int64)&unk_49DD7B0;
    v153 = 0;
    v155 = 2;
    v156 = 0;
    v157 = -8192;
    v154 = (char *)&unk_49DD7B0;
    k = 0;
    while ( 1 )
    {
      v112 = v109[3];
      if ( v110 != v112 )
      {
        v110 = v157;
        if ( v112 != v157 )
        {
          v113 = v109[7];
          if ( v113 != -4096 && v113 != 0 && v113 != -8192 )
          {
            sub_BD60C0(v109 + 5);
            v112 = v109[3];
          }
          v110 = v112;
        }
      }
      *v109 = &unk_49DB368;
      if ( v110 != 0 && v110 != -4096 && v110 != -8192 )
        sub_BD60C0(v109 + 1);
      v109 += 8;
      if ( v111 == v109 )
        break;
      v110 = v152;
    }
    v154 = (char *)&unk_49DB368;
    if ( v157 != 0 && v157 != -4096 && v157 != -8192 )
      sub_BD60C0(&v155);
    v150.m128i_i64[0] = (__int64)&unk_49DB368;
    if ( v152 != -4096 && v152 != 0 && v152 != -8192 )
      sub_BD60C0(&v150.m128i_i64[1]);
    v108 = v163;
  }
  sub_C7D6A0((__int64)v161, (unsigned __int64)v108 << 6, 8);
}
