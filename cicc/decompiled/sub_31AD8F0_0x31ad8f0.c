// Function: sub_31AD8F0
// Address: 0x31ad8f0
//
__int64 __fastcall sub_31AD8F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 *v7; // rax
  __int64 v8; // rbx
  __int64 v9; // r15
  __int64 v10; // r12
  __int64 v11; // r13
  unsigned __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rcx
  char v18; // al
  __int64 *v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r9
  __int64 v22; // rsi
  __int64 *v23; // rax
  __int64 v24; // rcx
  __int64 v25; // rbx
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rcx
  _QWORD *v29; // rax
  __int64 result; // rax
  __int64 v31; // rdx
  __int64 v32; // rsi
  __int64 v33; // r8
  __int64 v34; // r13
  __int64 *v35; // rax
  unsigned __int64 v36; // rdi
  __int64 *v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 *v40; // rsi
  unsigned int v41; // r14d
  int v42; // eax
  int v43; // edx
  int v44; // eax
  __int64 v45; // r15
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rbx
  int v49; // ebx
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rdx
  __int64 *v53; // rbx
  __int64 v54; // r15
  __int64 v55; // rax
  int v56; // edx
  __int64 *v57; // rbx
  __int64 *v58; // r14
  unsigned __int64 v59; // rdi
  unsigned __int64 v60; // rdi
  unsigned __int64 v61; // rdi
  __int64 v62; // rbx
  __int64 v63; // r14
  char v64; // al
  int v65; // eax
  __int64 *v66; // r14
  __int64 v67; // rsi
  unsigned __int64 v68; // rax
  bool v69; // zf
  __int64 v70; // rbx
  _QWORD *v71; // rax
  _QWORD *v72; // rdx
  __int64 v73; // r14
  __int64 v74; // r13
  __int64 v75; // rbx
  __int64 v76; // rsi
  _QWORD *v77; // rax
  _QWORD *v78; // rdx
  __int64 v79; // rax
  _QWORD *v80; // rax
  __int64 v81; // rsi
  unsigned __int64 v82; // rax
  __int64 v83; // rdx
  __int64 v84; // rax
  __int64 v85; // rbx
  __int64 v86; // r15
  __int64 *v87; // r13
  __int64 v88; // r12
  __int64 v89; // rdi
  unsigned __int8 v90; // dl
  __int64 v91; // rdx
  __int64 *v92; // rax
  __int64 v93; // rsi
  __int64 *v94; // rax
  char v95; // al
  __int64 i; // rax
  __int64 v97; // rax
  __int64 *v98; // rbx
  __int64 *v99; // r14
  unsigned __int64 v100; // rdi
  unsigned __int64 v101; // rdi
  unsigned __int64 v102; // rdi
  __int64 v103; // rdx
  __int64 v104; // rdi
  char *v105; // r14
  size_t v106; // rdx
  size_t v107; // rbx
  unsigned __int8 v108; // al
  char *v109; // rax
  size_t v110; // rdx
  __int64 v111; // rsi
  unsigned int v112; // eax
  size_t v113; // r12
  char *v114; // rbx
  __int64 v115; // r14
  __int64 v116; // rdx
  char v117; // al
  size_t v118; // r13
  __int64 v119; // r12
  __int64 v120; // rdx
  __int64 v121; // rdx
  __int64 v122; // rdx
  __int64 *v123; // [rsp-20h] [rbp-820h]
  __int64 *v124; // [rsp-20h] [rbp-820h]
  __int64 *v125; // [rsp-20h] [rbp-820h]
  __int64 *v126; // [rsp-20h] [rbp-820h]
  __int64 *v127; // [rsp-20h] [rbp-820h]
  __int64 v128; // [rsp-18h] [rbp-818h]
  __int64 v129; // [rsp-18h] [rbp-818h]
  __int64 v130; // [rsp-18h] [rbp-818h]
  __int64 v131; // [rsp-18h] [rbp-818h]
  __int64 v132; // [rsp-18h] [rbp-818h]
  __int64 v133; // [rsp-10h] [rbp-810h]
  __int64 v134; // [rsp-8h] [rbp-808h]
  __int64 v135; // [rsp+8h] [rbp-7F8h]
  __int64 v136; // [rsp+10h] [rbp-7F0h]
  __int64 v137; // [rsp+18h] [rbp-7E8h]
  __int64 *v138; // [rsp+20h] [rbp-7E0h]
  __int64 *v139; // [rsp+28h] [rbp-7D8h]
  __int64 *v140; // [rsp+30h] [rbp-7D0h]
  __int64 v141; // [rsp+38h] [rbp-7C8h]
  __int64 v142; // [rsp+40h] [rbp-7C0h]
  __int64 v143; // [rsp+48h] [rbp-7B8h]
  __int64 v144; // [rsp+50h] [rbp-7B0h]
  __int64 *v145; // [rsp+58h] [rbp-7A8h]
  unsigned int v146; // [rsp+60h] [rbp-7A0h] BYREF
  char v147; // [rsp+64h] [rbp-79Ch]
  __int64 v148; // [rsp+68h] [rbp-798h] BYREF
  _QWORD v149[2]; // [rsp+70h] [rbp-790h] BYREF
  __int64 v150; // [rsp+80h] [rbp-780h]
  int v151; // [rsp+88h] [rbp-778h]
  __int64 v152; // [rsp+90h] [rbp-770h]
  __int64 v153; // [rsp+98h] [rbp-768h]
  _BYTE *v154; // [rsp+A0h] [rbp-760h]
  __int64 v155; // [rsp+A8h] [rbp-758h]
  _BYTE v156[16]; // [rsp+B0h] [rbp-750h] BYREF
  __int64 *v157; // [rsp+C0h] [rbp-740h] BYREF
  __int64 v158; // [rsp+C8h] [rbp-738h] BYREF
  __int64 v159; // [rsp+D0h] [rbp-730h] BYREF
  __int64 v160; // [rsp+D8h] [rbp-728h]
  __int64 v161; // [rsp+E0h] [rbp-720h]
  __int64 v162; // [rsp+E8h] [rbp-718h]
  __int64 v163; // [rsp+F0h] [rbp-710h]
  __int64 v164; // [rsp+F8h] [rbp-708h]
  __int16 v165; // [rsp+100h] [rbp-700h]
  __int64 v166; // [rsp+108h] [rbp-6F8h] BYREF
  char *v167; // [rsp+110h] [rbp-6F0h]
  __int64 v168; // [rsp+118h] [rbp-6E8h]
  int v169; // [rsp+120h] [rbp-6E0h]
  char v170; // [rsp+124h] [rbp-6DCh]
  char v171; // [rsp+128h] [rbp-6D8h] BYREF
  int v172; // [rsp+168h] [rbp-698h]

  v6 = a1;
  v7 = *(__int64 **)(*(_QWORD *)a1 + 32LL);
  v8 = *v7;
  v138 = *(__int64 **)(*(_QWORD *)a1 + 40LL);
  v141 = v8;
  if ( v7 != v138 )
  {
    v143 = v8;
    v9 = a1;
    v139 = v7 + 1;
    while ( 1 )
    {
      v10 = *(_QWORD *)(v143 + 56);
      v142 = v143 + 48;
      if ( v143 + 48 != v10 )
        break;
LABEL_34:
      if ( v138 == v139 )
      {
        v6 = v9;
        goto LABEL_232;
      }
      v28 = *v139++;
      v143 = v28;
    }
    while ( 1 )
    {
      if ( !v10 )
        BUG();
      v11 = v10 - 24;
      if ( *(_BYTE *)(v10 - 24) == 84 )
      {
        v148 = v10 - 24;
        v12 = *(unsigned __int8 *)(*(_QWORD *)(v10 - 16) + 8LL);
        if ( (unsigned __int8)v12 > 0xCu || (v13 = 4143, !_bittest64(&v13, v12)) )
        {
          v13 = (unsigned int)v12 & 0xFFFFFFFD;
          if ( (v12 & 0xFD) != 4 && (_BYTE)v12 != 14 )
          {
            sub_2AB8760(
              (__int64)"Found a non-int non-pointer PHI",
              31,
              "loop control flow is not understood by vectorizer",
              0x31u,
              (__int64)"CFGNotUnderstood",
              16,
              *(__int64 **)(v9 + 64),
              *(_QWORD *)v9,
              0);
            return 0;
          }
        }
        v14 = v143;
        if ( v141 == v143 )
        {
          v15 = *(_QWORD *)v9;
          if ( (*(_DWORD *)(v10 - 20) & 0x7FFFFFF) != 2 )
          {
            sub_2AB8760(
              (__int64)"Found an invalid PHI",
              20,
              "loop control flow is not understood by vectorizer",
              0x31u,
              (__int64)"CFGNotUnderstood",
              16,
              *(__int64 **)(v9 + 64),
              v15,
              v10 - 24);
            return 0;
          }
          v157 = 0;
          v167 = &v171;
          v16 = *(_QWORD *)(v9 + 16);
          v165 = 0;
          v17 = *(_QWORD *)(v9 + 424);
          v158 = 6;
          v159 = 0;
          v160 = 0;
          v161 = 0;
          v162 = 0;
          v163 = 0;
          v164 = 0;
          v166 = 0;
          v168 = 8;
          v169 = 0;
          v170 = 1;
          v18 = sub_1026850(
                  v10 - 24,
                  v15,
                  (__int64)&v157,
                  v17,
                  *(_QWORD *)(v9 + 432),
                  *(_QWORD *)(v9 + 40),
                  *(_QWORD *)(v16 + 112));
          v21 = v134;
          if ( v18 )
          {
            if ( v163 )
            {
              v19 = *(__int64 **)(v9 + 408);
              if ( !*v19 )
                *v19 = v163;
            }
            v22 = v161;
            if ( *(_BYTE *)(v9 + 372) )
            {
              v23 = *(__int64 **)(v9 + 352);
              v20 = *(unsigned int *)(v9 + 364);
              v19 = &v23[v20];
              if ( v23 == v19 )
              {
LABEL_140:
                if ( (unsigned int)v20 >= *(_DWORD *)(v9 + 360) )
                  goto LABEL_141;
                *(_DWORD *)(v9 + 364) = v20 + 1;
                *v19 = v22;
                ++*(_QWORD *)(v9 + 344);
              }
              else
              {
                while ( v161 != *v23 )
                {
                  if ( v19 == ++v23 )
                    goto LABEL_140;
                }
              }
            }
            else
            {
LABEL_141:
              sub_C8CC70(v9 + 344, v161, (__int64)v19, v20, v133, v21);
            }
            v25 = sub_31AD3D0(v9 + 80, &v148);
            v26 = *(_QWORD *)(v25 + 24);
            *(_QWORD *)v25 = v157;
            v27 = v160;
            if ( v26 != v160 )
            {
              if ( v26 != 0 && v26 != -4096 && v26 != -8192 )
              {
                sub_BD60C0((_QWORD *)(v25 + 8));
                v27 = v160;
              }
              *(_QWORD *)(v25 + 24) = v27;
              LOBYTE(v24) = v27 != 0;
              if ( v27 != -4096 && v27 != 0 && v27 != -8192 )
                sub_BD6050((unsigned __int64 *)(v25 + 8), v158 & 0xFFFFFFFFFFFFFFF8LL);
            }
            *(_QWORD *)(v25 + 32) = v161;
            *(_QWORD *)(v25 + 40) = v162;
            *(_QWORD *)(v25 + 48) = v163;
            *(_QWORD *)(v25 + 56) = v164;
            *(_WORD *)(v25 + 64) = v165;
            if ( (__int64 *)(v25 + 72) != &v166 )
              sub_C8CE00(v25 + 72, v25 + 104, (__int64)&v166, v24, a5, a6);
            *(_DWORD *)(v25 + 168) = v172;
LABEL_29:
            if ( v170 )
              goto LABEL_30;
            goto LABEL_59;
          }
          v31 = *(_QWORD *)(v9 + 16);
          v32 = *(_QWORD *)v9;
          v154 = v156;
          v149[0] = 6;
          v149[1] = 0;
          v150 = 0;
          v151 = 0;
          v152 = 0;
          v153 = 0;
          v155 = 0x200000000LL;
          if ( (unsigned __int8)sub_1023CA0(v148, v32, v31, (__int64)v149, 0)
            && (byte_5035508 || v151 != 2 || sub_1023590((__int64)v149)) )
          {
            sub_31AC420(v9, v148, (__int64)v149, v9 + 344, v33);
            if ( v151 == 3 )
            {
              v34 = v153;
              if ( v153 )
              {
                if ( !sub_B451B0(v153) )
                {
                  v35 = *(__int64 **)(v9 + 408);
                  if ( !*v35 )
                    *v35 = v34;
                }
              }
            }
          }
          else
          {
            if ( !(unsigned __int8)sub_1022990(v148, *(_QWORD *)v9, *(_QWORD *)(v9 + 40)) )
            {
              if ( !(unsigned __int8)sub_1023CA0(v148, *(_QWORD *)v9, *(_QWORD *)(v9 + 16), (__int64)v149, 1)
                || !byte_5035508 && v151 == 2 && !sub_1023590((__int64)v149) )
              {
                sub_2AB8760(
                  (__int64)"Found an unidentified PHI",
                  25,
                  "value that could not be identified as reduction is used outside the loop",
                  0x48u,
                  (__int64)"NonReductionValueUsedOutsideLoop",
                  32,
                  *(__int64 **)(v9 + 64),
                  *(_QWORD *)v9,
                  v148);
                if ( v154 != v156 )
                  _libc_free((unsigned __int64)v154);
                if ( v150 != 0 && v150 != -4096 && v150 != -8192 )
                  sub_BD60C0(v149);
                if ( !v170 )
                  _libc_free((unsigned __int64)v167);
                if ( v160 != -4096 && v160 != 0 && v160 != -8192 )
                {
                  sub_BD60C0(&v158);
                  return 0;
                }
                return 0;
              }
              sub_31AC420(v9, v148, (__int64)v149, v9 + 344, v39);
              v36 = (unsigned __int64)v154;
              if ( v154 == v156 )
              {
LABEL_56:
                if ( v150 == 0 || v150 == -4096 || v150 == -8192 )
                  goto LABEL_29;
                sub_BD60C0(v149);
                if ( v170 )
                {
LABEL_30:
                  if ( v160 != 0 && v160 != -4096 && v160 != -8192 )
                    sub_BD60C0(&v158);
                  goto LABEL_33;
                }
LABEL_59:
                _libc_free((unsigned __int64)v167);
                goto LABEL_30;
              }
LABEL_55:
              _libc_free(v36);
              goto LABEL_56;
            }
            a5 = v148;
            if ( !*(_BYTE *)(v9 + 372) )
              goto LABEL_199;
            v92 = *(__int64 **)(v9 + 352);
            v38 = *(unsigned int *)(v9 + 364);
            v37 = &v92[v38];
            if ( v92 == v37 )
            {
LABEL_201:
              if ( (unsigned int)v38 < *(_DWORD *)(v9 + 360) )
              {
                v38 = (unsigned int)(v38 + 1);
                *(_DWORD *)(v9 + 364) = v38;
                *v37 = a5;
                v93 = v148;
                ++*(_QWORD *)(v9 + 344);
                goto LABEL_167;
              }
LABEL_199:
              sub_C8CC70(v9 + 344, v148, (__int64)v37, v38, v148, a6);
              v93 = v148;
              goto LABEL_167;
            }
            while ( 1 )
            {
              v93 = *v92;
              if ( v148 == *v92 )
                break;
              if ( v37 == ++v92 )
                goto LABEL_201;
            }
LABEL_167:
            if ( !*(_BYTE *)(v9 + 268) )
              goto LABEL_198;
            v94 = *(__int64 **)(v9 + 248);
            v38 = *(unsigned int *)(v9 + 260);
            v37 = &v94[v38];
            if ( v94 != v37 )
            {
              while ( *v94 != v93 )
              {
                if ( v37 == ++v94 )
                  goto LABEL_203;
              }
              v36 = (unsigned __int64)v154;
              if ( v154 == v156 )
                goto LABEL_56;
              goto LABEL_55;
            }
LABEL_203:
            if ( (unsigned int)v38 < *(_DWORD *)(v9 + 256) )
            {
              *(_DWORD *)(v9 + 260) = v38 + 1;
              *v37 = v93;
              ++*(_QWORD *)(v9 + 240);
            }
            else
            {
LABEL_198:
              sub_C8CC70(v9 + 240, v93, (__int64)v37, v38, a5, a6);
            }
          }
          v36 = (unsigned __int64)v154;
          if ( v154 == v156 )
            goto LABEL_56;
          goto LABEL_55;
        }
        if ( !*(_BYTE *)(v9 + 372) )
          goto LABEL_133;
        v29 = *(_QWORD **)(v9 + 352);
        v14 = *(unsigned int *)(v9 + 364);
        v13 = (__int64)&v29[v14];
        if ( v29 != (_QWORD *)v13 )
        {
          while ( v11 != *v29 )
          {
            if ( (_QWORD *)v13 == ++v29 )
              goto LABEL_40;
          }
          goto LABEL_33;
        }
LABEL_40:
        if ( (unsigned int)v14 >= *(_DWORD *)(v9 + 360) )
        {
LABEL_133:
          sub_C8CC70(v9 + 344, v10 - 24, v13, v14, a5, a6);
          goto LABEL_33;
        }
LABEL_41:
        *(_DWORD *)(v9 + 364) = v14 + 1;
        *(_QWORD *)v13 = v11;
        ++*(_QWORD *)(v9 + 344);
        goto LABEL_33;
      }
      if ( *(_BYTE *)(v10 - 24) != 85 )
      {
        v62 = 0;
        goto LABEL_101;
      }
      if ( !(unsigned int)sub_9B78C0(v10 - 24, *(__int64 **)(v9 + 32)) )
      {
        v83 = *(_QWORD *)(v10 - 56);
        if ( *(_BYTE *)(v10 - 24) != 85 )
        {
          if ( !v83 || *(_BYTE *)v83 )
            goto LABEL_145;
LABEL_208:
          if ( *(_QWORD *)(v83 + 24) != *(_QWORD *)(v10 + 56) )
            goto LABEL_145;
          if ( !*(_QWORD *)(v9 + 32) )
          {
            v88 = v9;
            v85 = v11;
            goto LABEL_149;
          }
          v145 = &v159;
          v157 = &v159;
          v158 = 0x800000000LL;
          sub_D39570(v10 - 24, (unsigned int *)&v157);
          v97 = (unsigned int)v158;
          LOBYTE(v144) = 0;
          if ( !(_DWORD)v158 )
          {
            v104 = *(_QWORD *)(v10 - 56);
            v140 = *(__int64 **)(v9 + 32);
            if ( v104 )
            {
              if ( *(_BYTE *)v104 )
              {
                v104 = 0;
              }
              else if ( *(_QWORD *)(v104 + 24) != *(_QWORD *)(v10 + 56) )
              {
                v104 = 0;
              }
            }
            v105 = (char *)sub_BD5D20(v104);
            v107 = v106;
            LOBYTE(v137) = sub_97F890(*v140, v105, v106);
            if ( (_BYTE)v137 )
            {
              v146 = 0;
              v147 = 0;
              LODWORD(v148) = 0;
              BYTE4(v148) = 0;
              sub_980260(*v140, v105, v107, (__int64)&v146, (__int64)&v148);
              LODWORD(v149[0]) = 2;
              BYTE4(v149[0]) = 0;
              LOBYTE(v144) = v137;
              v112 = 2;
              v136 = v10;
              v113 = v107;
              v114 = v105;
              v135 = v11;
              while ( v112 <= v146 )
              {
                v115 = *v140;
                sub_97FA10(*v140, v114, v113, (__int64)v149, 0);
                if ( v116 )
                {
                  LOBYTE(v144) = 0;
                }
                else
                {
                  sub_97FA10(v115, v114, v113, (__int64)v149, 1);
                  LOBYTE(v144) = (v121 == 0) & v144;
                }
                v112 = 2 * LODWORD(v149[0]);
                LODWORD(v149[0]) *= 2;
                if ( BYTE4(v149[0]) )
                {
                  if ( !v147 )
                    break;
                }
              }
              BYTE4(v149[0]) = 1;
              LODWORD(v149[0]) = 1;
              v117 = v137;
              v137 = v135;
              v118 = v113;
              while ( (!v117 || BYTE4(v148)) && LODWORD(v149[0]) <= (unsigned int)v148 )
              {
                v119 = *v140;
                sub_97FA10(*v140, v114, v118, (__int64)v149, 0);
                if ( v120 )
                {
                  LOBYTE(v144) = 0;
                }
                else
                {
                  sub_97FA10(v119, v114, v118, (__int64)v149, 1);
                  LOBYTE(v144) = (v122 == 0) & v144;
                }
                v117 = BYTE4(v149[0]);
                LODWORD(v149[0]) *= 2;
              }
              v10 = v136;
              v11 = v137;
              LOBYTE(v144) = v144 ^ 1;
              v97 = (unsigned int)v158;
            }
            else
            {
              LOBYTE(v144) = 1;
              v97 = (unsigned int)v158;
            }
          }
          v98 = v157;
          v99 = &v157[28 * v97];
          if ( v157 != v99 )
          {
            do
            {
              v99 -= 28;
              v100 = v99[23];
              if ( (__int64 *)v100 != v99 + 25 )
                j_j___libc_free_0(v100);
              v101 = v99[19];
              if ( (__int64 *)v101 != v99 + 21 )
                j_j___libc_free_0(v101);
              v102 = v99[1];
              if ( (__int64 *)v102 != v99 + 3 )
                _libc_free(v102);
            }
            while ( v98 != v99 );
            v99 = v157;
          }
          if ( v99 != v145 )
            _libc_free((unsigned __int64)v99);
          if ( (_BYTE)v144 )
          {
LABEL_145:
            v84 = v9;
            v85 = v11;
            v86 = v10;
            v87 = *(__int64 **)(v84 + 32);
            v88 = v84;
            if ( v87 )
            {
              v89 = *(_QWORD *)(v86 - 56);
              if ( v89 )
              {
                if ( !*(_BYTE *)v89 && *(_QWORD *)(v89 + 24) == *(_QWORD *)(v86 + 56) )
                {
                  v108 = *(_BYTE *)(*(_QWORD *)(v86 - 16) + 8LL);
                  if ( v108 <= 3u || v108 == 5 || (v108 & 0xFD) == 4 )
                  {
                    v109 = (char *)sub_BD5D20(v89);
                    if ( (unsigned __int8)sub_980AF0(*v87, v109, v110, &v157) )
                    {
                      v111 = (unsigned int)v157;
                      if ( (unsigned __int8)sub_F50940(*(_QWORD **)(v88 + 32), (unsigned int)v157) )
                      {
                        v134 = v111;
                        sub_2AB8760(
                          (__int64)"Found a non-intrinsic callsite",
                          30,
                          "library call cannot be vectorized. Try compiling with -fno-math-errno, -ffast-math, or similar flags",
                          0x64u,
                          (__int64)"CantVectorizeLibcall",
                          20,
                          *(__int64 **)(v88 + 64),
                          *(_QWORD *)v88,
                          v85);
                        return 0;
                      }
                    }
                  }
                }
              }
            }
LABEL_149:
            sub_2AB8760(
              (__int64)"Found a non-intrinsic callsite",
              30,
              "call instruction cannot be vectorized",
              0x25u,
              (__int64)"CantVectorizeLibcall",
              20,
              *(__int64 **)(v88 + 64),
              *(_QWORD *)v88,
              v85);
            return 0;
          }
          goto LABEL_69;
        }
        if ( !v83 || *(_BYTE *)v83 )
          goto LABEL_145;
        if ( *(_QWORD *)(v83 + 24) != *(_QWORD *)(v10 + 56)
          || (*(_BYTE *)(v83 + 33) & 0x20) == 0
          || (unsigned int)(*(_DWORD *)(v83 + 36) - 68) > 3 )
        {
          goto LABEL_208;
        }
      }
LABEL_69:
      v40 = *(__int64 **)(v9 + 32);
      v41 = 0;
      v140 = *(__int64 **)(*(_QWORD *)(v9 + 16) + 112LL);
      v42 = sub_9B78C0(v11, v40);
      v43 = *(unsigned __int8 *)(v10 - 24);
      v145 = (__int64 *)v9;
      LODWORD(v144) = v42;
      v44 = v43 - 29;
      if ( v43 != 40 )
      {
LABEL_70:
        v45 = 0;
        if ( v44 != 56 )
        {
          if ( v44 != 5 )
            BUG();
          v45 = 64;
        }
        if ( *(char *)(v10 - 17) >= 0 )
          goto LABEL_84;
LABEL_74:
        v46 = sub_BD2BC0(v11);
        v48 = v46 + v47;
        if ( *(char *)(v10 - 17) >= 0 )
        {
          if ( (unsigned int)(v48 >> 4) )
LABEL_278:
            BUG();
        }
        else if ( (unsigned int)((v48 - sub_BD2BC0(v11)) >> 4) )
        {
          if ( *(char *)(v10 - 17) >= 0 )
            goto LABEL_278;
          v49 = *(_DWORD *)(sub_BD2BC0(v11) + 8);
          if ( *(char *)(v10 - 17) >= 0 )
            BUG();
          v50 = sub_BD2BC0(v11);
          v52 = 32LL * (unsigned int)(*(_DWORD *)(v50 + v51 - 4) - v49);
          goto LABEL_79;
        }
        goto LABEL_84;
      }
      while ( 1 )
      {
        v45 = 32LL * (unsigned int)sub_B491D0(v11);
        if ( *(char *)(v10 - 17) < 0 )
          goto LABEL_74;
LABEL_84:
        v52 = 0;
LABEL_79:
        if ( v41 >= (unsigned int)((32LL * (*(_DWORD *)(v10 - 20) & 0x7FFFFFF) - 32 - v45 - v52) >> 5) )
          break;
        v53 = v145;
        if ( sub_9B75A0((unsigned int)v144, v41, v145[3]) )
        {
          v54 = *v53;
          v55 = sub_DEEF40(
                  v53[2],
                  *(_QWORD *)(v11 + 32 * (v41 - (unsigned __int64)(*(_DWORD *)(v10 - 20) & 0x7FFFFFF))));
          if ( !sub_DADE90((__int64)v140, v55, v54) )
          {
            v130 = *v145;
            v125 = (__int64 *)v145[8];
            LOBYTE(v144) = 0;
            sub_2AB8760(
              (__int64)"Found unvectorizable intrinsic",
              30,
              "intrinsic instruction cannot be vectorized",
              0x2Au,
              (__int64)"CantVectorizeIntrinsic",
              22,
              v125,
              v130,
              v11);
            return (unsigned __int8)v144;
          }
        }
        v56 = *(unsigned __int8 *)(v10 - 24);
        ++v41;
        v44 = v56 - 29;
        if ( v56 != 40 )
          goto LABEL_70;
      }
      v9 = (__int64)v145;
      v145 = &v159;
      v157 = &v159;
      v158 = 0x800000000LL;
      sub_D39570(v11, (unsigned int *)&v157);
      v57 = v157;
      LODWORD(v144) = v158;
      a5 = 224LL * (unsigned int)v158;
      v58 = (__int64 *)((char *)v157 + a5);
      if ( v157 != (__int64 *)((char *)v157 + a5) )
      {
        do
        {
          v58 -= 28;
          v59 = v58[23];
          if ( (__int64 *)v59 != v58 + 25 )
            j_j___libc_free_0(v59);
          v60 = v58[19];
          if ( (__int64 *)v60 != v58 + 21 )
            j_j___libc_free_0(v60);
          v61 = v58[1];
          if ( (__int64 *)v61 != v58 + 3 )
            _libc_free(v61);
        }
        while ( v57 != v58 );
        v58 = v157;
      }
      if ( v58 != v145 )
        _libc_free((unsigned __int64)v58);
      if ( (_DWORD)v144 )
        *(_BYTE *)(v9 + 592) = 1;
      v62 = v11;
LABEL_101:
      v63 = *(_QWORD *)(v10 - 16);
      v64 = *(_BYTE *)(v63 + 8);
      if ( v64 == 15 )
      {
        if ( *(_BYTE *)(v10 - 24) != 85 || !sub_BCB420(*(_QWORD *)(v10 - 16)) )
          goto LABEL_131;
        v95 = *(_BYTE *)(v63 + 8);
        if ( v95 == 15 )
        {
          if ( !(unsigned __int8)sub_E45910(v63) )
            goto LABEL_131;
        }
        else if ( v95 != 7 && !(unsigned __int8)sub_BCBCB0(v63) )
        {
          goto LABEL_131;
        }
        for ( i = *(_QWORD *)(v10 - 8); i; i = *(_QWORD *)(i + 8) )
        {
          if ( **(_BYTE **)(i + 24) != 93 )
            goto LABEL_131;
        }
      }
      else if ( v64 != 7 && !(unsigned __int8)sub_BCBCB0(*(_QWORD *)(v10 - 16)) )
      {
        goto LABEL_131;
      }
      if ( (unsigned int)*(unsigned __int8 *)(v10 - 24) - 67 <= 0xC
        && ((*(_BYTE *)(v10 - 17) & 0x40) == 0
          ? (v91 = v11 - 32LL * (*(_DWORD *)(v10 - 20) & 0x7FFFFFF))
          : (v91 = *(_QWORD *)(v10 - 32)),
            !(unsigned __int8)sub_BCBCB0(*(_QWORD *)(*(_QWORD *)v91 + 8LL)))
        || (v65 = *(unsigned __int8 *)(v10 - 24), (_BYTE)v65 == 90) )
      {
LABEL_131:
        sub_2AB8760(
          (__int64)"Found unvectorizable type",
          25,
          "instruction return type cannot be vectorized",
          0x2Cu,
          (__int64)"CantVectorizeInstructionReturnType",
          34,
          *(__int64 **)(v9 + 64),
          *(_QWORD *)v9,
          v11);
        return 0;
      }
      if ( (_BYTE)v65 == 62 )
      {
        v66 = *(__int64 **)(*(_QWORD *)(v10 - 88) + 8LL);
        if ( !(unsigned __int8)sub_BCBCB0((__int64)v66) )
        {
          v132 = *(_QWORD *)v9;
          v127 = *(__int64 **)(v9 + 64);
          LOBYTE(v145) = 0;
          sub_2AB8760(
            (__int64)"Store instruction cannot be vectorized",
            38,
            "Store instruction cannot be vectorized",
            0x26u,
            (__int64)"CantVectorizeStore",
            18,
            v127,
            v132,
            v11);
          return (unsigned __int8)v145;
        }
        if ( (*(_BYTE *)(v10 - 17) & 0x20) != 0 )
        {
          if ( sub_B91C10(v11, 9) )
          {
            v67 = sub_BCDA70(v66, 2);
            _BitScanReverse64(&v68, 1LL << (*(_WORD *)(v10 - 22) >> 1));
            if ( !(unsigned __int8)sub_DFA3C0(*(__int64 ***)(v9 + 24), v67, 63 - ((unsigned int)v68 ^ 0x3F)) )
            {
              v131 = *(_QWORD *)v9;
              v126 = *(__int64 **)(v9 + 64);
              LOBYTE(v145) = 0;
              sub_2AB8760(
                (__int64)"nontemporal store instruction cannot be vectorized",
                50,
                "nontemporal store instruction cannot be vectorized",
                0x32u,
                (__int64)"CantVectorizeNontemporalStore",
                29,
                v126,
                v131,
                v11);
              return (unsigned __int8)v145;
            }
          }
        }
      }
      else if ( (_BYTE)v65 == 61 )
      {
        if ( (*(_BYTE *)(v10 - 17) & 0x20) != 0 )
        {
          if ( sub_B91C10(v11, 9) )
          {
            v81 = sub_BCDA70(*(__int64 **)(v10 - 16), 2);
            _BitScanReverse64(&v82, 1LL << (*(_WORD *)(v10 - 22) >> 1));
            if ( !(unsigned __int8)sub_DFA450(*(__int64 ***)(v9 + 24), v81, 63 - ((unsigned int)v82 ^ 0x3F)) )
            {
              v128 = *(_QWORD *)v9;
              v123 = *(__int64 **)(v9 + 64);
              LOBYTE(v145) = 0;
              sub_2AB8760(
                (__int64)"nontemporal load instruction cannot be vectorized",
                49,
                "nontemporal load instruction cannot be vectorized",
                0x31u,
                (__int64)"CantVectorizeNontemporalLoad",
                28,
                v123,
                v128,
                v11);
              return (unsigned __int8)v145;
            }
          }
        }
      }
      else
      {
        v90 = *(_BYTE *)(*(_QWORD *)(v10 - 16) + 8LL);
        if ( (v90 <= 3u || v90 == 5 || (v90 & 0xFD) == 4)
          && (v62 || (unsigned int)(v65 - 42) <= 0x11)
          && !sub_B45190(v11) )
        {
          *(_BYTE *)(*(_QWORD *)(v9 + 416) + 96LL) = 1;
        }
      }
      v69 = *(_BYTE *)(v9 + 372) == 0;
      v70 = *(_QWORD *)v9;
      v144 = v9 + 344;
      if ( v69 )
      {
        if ( sub_C8CA60(v144, v11) )
          goto LABEL_33;
      }
      else
      {
        v71 = *(_QWORD **)(v9 + 352);
        v72 = &v71[*(unsigned int *)(v9 + 364)];
        if ( v71 != v72 )
        {
          while ( v11 != *v71 )
          {
            if ( v72 == ++v71 )
              goto LABEL_114;
          }
          goto LABEL_33;
        }
      }
LABEL_114:
      v73 = *(_QWORD *)(v10 - 8);
      if ( v73 )
      {
        v145 = (__int64 *)v11;
        v74 = v70;
        v75 = v70 + 56;
        v76 = *(_QWORD *)(*(_QWORD *)(v73 + 24) + 40LL);
        if ( *(_BYTE *)(v74 + 84) )
        {
LABEL_116:
          v77 = *(_QWORD **)(v74 + 64);
          v78 = &v77[*(unsigned int *)(v74 + 76)];
          if ( v77 != v78 )
          {
            while ( v76 != *v77 )
            {
              if ( v78 == ++v77 )
                goto LABEL_123;
            }
            goto LABEL_120;
          }
        }
        else
        {
          while ( sub_C8CA60(v75, v76) )
          {
LABEL_120:
            v73 = *(_QWORD *)(v73 + 8);
            if ( !v73 )
              goto LABEL_33;
            v76 = *(_QWORD *)(*(_QWORD *)(v73 + 24) + 40LL);
            if ( *(_BYTE *)(v74 + 84) )
              goto LABEL_116;
          }
        }
LABEL_123:
        v11 = (__int64)v145;
        v79 = sub_D9B120(*(_QWORD *)(v9 + 16));
        if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64))(*(_QWORD *)v79 + 8LL))(v79, v76) )
        {
          v129 = *(_QWORD *)v9;
          v124 = *(__int64 **)(v9 + 64);
          LOBYTE(v145) = 0;
          sub_2AB8760(
            (__int64)"Value cannot be used outside the loop",
            37,
            "Value cannot be used outside the loop",
            0x25u,
            (__int64)"ValueUsedOutsideLoop",
            20,
            v124,
            v129,
            v11);
          return (unsigned __int8)v145;
        }
        if ( !*(_BYTE *)(v9 + 372) )
          goto LABEL_129;
        v80 = *(_QWORD **)(v9 + 352);
        v14 = *(unsigned int *)(v9 + 364);
        v13 = (__int64)&v80[v14];
        if ( v80 != (_QWORD *)v13 )
        {
          while ( v11 != *v80 )
          {
            if ( (_QWORD *)v13 == ++v80 )
              goto LABEL_128;
          }
          goto LABEL_33;
        }
LABEL_128:
        if ( (unsigned int)v14 >= *(_DWORD *)(v9 + 360) )
        {
LABEL_129:
          sub_C8CC70(v144, v11, v13, v14, a5, a6);
          goto LABEL_33;
        }
        goto LABEL_41;
      }
LABEL_33:
      v10 = *(_QWORD *)(v10 + 8);
      if ( v142 == v10 )
        goto LABEL_34;
    }
  }
LABEL_232:
  v103 = *(_QWORD *)(v6 + 72);
  if ( !v103 )
  {
    if ( *(_DWORD *)(v6 + 168) )
    {
      result = 1;
      if ( *(_QWORD *)(v6 + 336) )
        return result;
      v134 = 1;
      sub_2AB8760(
        (__int64)"Did not find one integer induction var",
        38,
        "integer loop induction variable could not be identified",
        0x37u,
        (__int64)"NoIntegerInductionVariable",
        26,
        *(__int64 **)(v6 + 64),
        *(_QWORD *)v6,
        0);
    }
    else
    {
      sub_2AB8760(
        (__int64)"Did not find one integer induction var",
        38,
        "loop induction variable could not be identified",
        0x2Fu,
        (__int64)"NoInductionVariable",
        19,
        *(__int64 **)(v6 + 64),
        *(_QWORD *)v6,
        0);
    }
    return 0;
  }
  result = 1;
  if ( *(_QWORD *)(v6 + 336) != *(_QWORD *)(v103 + 8) )
    *(_QWORD *)(v6 + 72) = 0;
  return result;
}
