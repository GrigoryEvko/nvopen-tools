// Function: sub_C05FA0
// Address: 0xc05fa0
//
__int64 __fastcall sub_C05FA0(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rbx
  unsigned __int64 v5; // rax
  __int64 v6; // r13
  __int64 result; // rax
  __m128i *v8; // rdx
  __m128i si128; // xmm0
  const char *v10; // rax
  size_t v11; // rdx
  __m128i *v12; // rdi
  const char *v13; // rsi
  unsigned __int64 v14; // rax
  __m128i v15; // xmm0
  __int64 v16; // rdi
  _BYTE *v17; // rax
  __int64 v18; // r13
  __m128i v19; // xmm1
  void (__fastcall *v20)(_QWORD, _QWORD, _QWORD); // rax
  __int64 v21; // rdx
  __m128i v22; // xmm0
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // rbx
  __int64 v26; // rax
  __int64 v27; // rbx
  __int64 v28; // r14
  _BYTE *v29; // r12
  unsigned int v30; // r10d
  __int64 v31; // r11
  __int64 v32; // rax
  __int64 v33; // r13
  _BYTE *v34; // rax
  __int64 v35; // rdi
  _BYTE *v36; // rax
  int v37; // eax
  __int64 v38; // rdx
  _QWORD *v39; // rax
  _QWORD *i; // rdx
  __int64 v41; // rax
  __int64 v42; // rbx
  __int64 v43; // r13
  _BYTE *v44; // r12
  __int64 v45; // rax
  const char *v46; // rsi
  const char *v47; // rax
  __int64 v48; // r13
  _BYTE *v49; // rax
  _BYTE *v50; // rax
  __int64 v51; // rax
  __int64 v52; // rax
  unsigned int v53; // eax
  __int64 v54; // rdx
  __int64 v55; // rdi
  _BYTE *v56; // rax
  __int64 v57; // r13
  _BYTE *v58; // rax
  __int64 v59; // rdi
  _BYTE *v60; // rax
  _BYTE *v61; // rax
  __int64 v62; // rdi
  __int64 v63; // rax
  char *v64; // r13
  char v65; // al
  __int64 v66; // rdi
  __int64 v67; // rax
  int v68; // edx
  unsigned __int64 v69; // rcx
  __int64 v70; // rsi
  _BYTE *v71; // r13
  __int64 v72; // rax
  __int64 v73; // r14
  __int64 v74; // rax
  __int64 v75; // rdi
  __int64 v76; // rax
  __int64 v77; // rdi
  __int64 v78; // r13
  _BYTE *v79; // rax
  bool v80; // zf
  int v81; // eax
  __int64 v82; // rdi
  _BYTE *v83; // r14
  __int64 v84; // r13
  _BYTE *v85; // rax
  _BYTE *v86; // rax
  __int64 v87; // rdi
  __int64 v88; // rax
  char v89; // al
  __int64 v90; // rax
  __int64 v91; // r13
  _BYTE *v92; // rax
  _BYTE *v93; // rax
  __int64 v94; // rax
  __int64 v95; // r8
  __int64 v96; // r9
  __int64 v97; // r13
  _BYTE *v98; // rax
  __int64 v99; // rax
  unsigned int v100; // ecx
  unsigned int v101; // eax
  _QWORD *v102; // rdi
  int v103; // ebx
  _QWORD *v104; // rax
  __int64 v105; // r12
  _BYTE *v106; // rax
  __int64 v107; // r13
  _BYTE *v108; // rax
  __int64 v109; // rax
  int v110; // eax
  __int64 v111; // rcx
  char *v112; // r12
  __int64 v113; // rbx
  char *v114; // r13
  unsigned __int64 v115; // rax
  __int64 *v116; // rbx
  __int64 *v117; // rdi
  char *v118; // r14
  __int64 *v119; // r13
  __int64 *v120; // rbx
  _BYTE *v121; // rax
  _BYTE *v122; // rdx
  _BYTE *v123; // rsi
  __int64 v124; // rsi
  unsigned __int8 v125; // al
  __int64 v126; // r13
  _BYTE *v127; // rax
  __int64 v128; // rax
  _BYTE *v129; // rdx
  __int64 v130; // r13
  _BYTE *v131; // rax
  __int64 v132; // rax
  __int64 v133; // r13
  _BYTE *v134; // rax
  __int64 v135; // rax
  __int64 v136; // r13
  _BYTE *v137; // rax
  __int64 v138; // rax
  __int64 v139; // r13
  _BYTE *v140; // rax
  __int64 v141; // rax
  __int64 v142; // r13
  _BYTE *v143; // rax
  __int64 v144; // rax
  __int64 v145; // r13
  _BYTE *v146; // rax
  __int64 v147; // rax
  unsigned int v148; // eax
  __int64 v149; // r13
  _BYTE *v150; // rax
  __int64 v151; // rax
  __int64 v152; // r13
  _BYTE *v153; // rax
  __int64 v154; // rax
  __int64 v155; // r13
  _BYTE *v156; // rax
  __int64 v157; // rax
  __int64 v158; // r13
  _BYTE *v159; // rax
  __int64 v160; // rax
  __int64 v161; // r13
  _BYTE *v162; // rax
  __int64 v163; // rax
  __int64 v164; // r13
  _BYTE *v165; // rax
  __int64 v166; // rax
  __int64 v167; // r14
  _BYTE *v168; // rax
  __int64 v169; // rax
  __int64 v170; // r13
  _BYTE *v171; // rax
  __int64 v172; // rax
  __int64 v173; // r13
  _BYTE *v174; // rax
  __int64 v175; // rax
  unsigned int v176; // eax
  __int64 v177; // rcx
  __int64 *v178; // r14
  __int64 *v179; // rcx
  __int64 v180; // r12
  __int64 *v181; // rbx
  __int64 v182; // rax
  char *v183; // rax
  unsigned int v184; // eax
  __int64 v185; // rcx
  __int64 v186; // rdi
  __int64 v187; // rax
  int v188; // eax
  _BYTE *v189; // rax
  __int64 v190; // rax
  unsigned __int64 v191; // rdx
  unsigned __int64 v192; // rax
  _QWORD *v193; // rax
  __int64 v194; // rdx
  _QWORD *j; // rdx
  _BYTE *v196; // rax
  __int64 v197; // r13
  _BYTE *v198; // rax
  __int64 v199; // rax
  _BYTE *v200; // rax
  __int64 *v201; // r12
  __int64 v202; // rcx
  __int64 *v203; // r14
  char v204; // al
  __int64 v205; // r13
  _BYTE *v206; // rax
  __int64 v207; // r13
  _BYTE *v208; // rax
  _BYTE *v209; // [rsp+10h] [rbp-90h]
  _BYTE *v210; // [rsp+10h] [rbp-90h]
  __int64 v211; // [rsp+18h] [rbp-88h]
  __int64 v212; // [rsp+20h] [rbp-80h]
  __int64 v213; // [rsp+20h] [rbp-80h]
  _BYTE *v214; // [rsp+20h] [rbp-80h]
  __int64 v215; // [rsp+28h] [rbp-78h]
  size_t v216; // [rsp+28h] [rbp-78h]
  _QWORD *v217; // [rsp+30h] [rbp-70h] BYREF
  _BYTE *v218; // [rsp+38h] [rbp-68h] BYREF
  __m128i v219; // [rsp+40h] [rbp-60h] BYREF
  void (__fastcall *v220)(_QWORD, _QWORD, _QWORD); // [rsp+50h] [rbp-50h]
  __int64 v221; // [rsp+58h] [rbp-48h]
  char v222; // [rsp+60h] [rbp-40h]
  char v223; // [rsp+61h] [rbp-3Fh]

  v212 = a2 + 72;
  if ( a2 + 72 != (*(_QWORD *)(a2 + 72) & 0xFFFFFFFFFFFFFFF8LL) )
  {
    a1[33] = a2;
    v3 = (__int64)(a1 + 20);
    *(_DWORD *)(v3 + 120) = *(_DWORD *)(a2 + 92);
    sub_B1F440(v3);
  }
  v4 = *(_QWORD *)(a2 + 80);
  if ( v212 == v4 )
  {
LABEL_20:
    v18 = *a1;
    v217 = a1;
    sub_E34680(a1 + 259);
    v219.m128i_i64[1] = (__int64)&v217;
    v19 = _mm_loadu_si128((const __m128i *)a1 + 130);
    v219.m128i_i64[0] = (__int64)sub_BDA210;
    v20 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))a1[262];
    a1[262] = sub_BD8F60;
    v21 = a1[263];
    v22 = _mm_loadu_si128(&v219);
    a1[259] = v18;
    v220 = v20;
    v221 = v21;
    v219 = v19;
    a1[263] = sub_BD8DA0;
    *((__m128i *)a1 + 130) = v22;
    if ( v20 )
      v20(&v219, &v219, 3);
    a1[277] = a2;
    *((_BYTE *)a1 + 152) = 0;
    sub_BF0CF0(a1, a2);
    v211 = *(_QWORD *)(a2 + 80);
    if ( v212 != v211 )
    {
      while ( 1 )
      {
        v25 = v211;
        a2 = v211 - 24;
        v211 = *(_QWORD *)(v211 + 8);
        sub_BDE380((__int64)a1, a2);
        v26 = v25 + 24;
        v27 = *(_QWORD *)(v25 + 32);
        v215 = v26;
        if ( v26 != v27 )
          break;
LABEL_38:
        if ( v212 == v211 )
          goto LABEL_39;
      }
      while ( 1 )
      {
        v28 = v27;
        v27 = *(_QWORD *)(v27 + 8);
        v29 = (_BYTE *)(v28 - 24);
        a2 = v28 - 24;
        sub_BE60A0((__int64)a1, v28 - 24);
        v30 = *(_DWORD *)(v28 - 20) & 0x7FFFFFF;
        if ( v30 )
        {
          v31 = v30;
          v32 = 0;
          a2 = 32LL * v30;
          v24 = *(_BYTE *)(v28 - 17) & 0x40;
          while ( 1 )
          {
            v23 = (__int64)&v29[-a2];
            if ( (_BYTE)v24 )
              v23 = *(_QWORD *)(v28 - 32);
            if ( !*(_QWORD *)(v23 + v32) )
              break;
            v32 += 32;
            if ( a2 == v32 )
            {
              switch ( *(_BYTE *)(v28 - 24) )
              {
                case 0x1E:
                  goto LABEL_78;
                case 0x1F:
                  if ( v30 != 3 || sub_BCAC40(*(_QWORD *)(*(_QWORD *)(v28 - 120) + 8LL), 1) )
                    goto LABEL_403;
                  a2 = (__int64)&v219;
                  v64 = *(char **)(v28 - 120);
                  v223 = 1;
                  v219.m128i_i64[0] = (__int64)"Branch condition is not 'i1' type!";
                  v222 = 3;
                  sub_BDBF70(a1, (__int64)&v219);
                  if ( *a1 )
                  {
                    a2 = v28 - 24;
                    sub_BDBD80((__int64)a1, (_BYTE *)(v28 - 24));
                    if ( v64 )
                      goto LABEL_467;
                  }
                  goto LABEL_37;
                case 0x20:
                  goto LABEL_79;
                case 0x21:
                  goto LABEL_77;
                case 0x22:
                  goto LABEL_133;
                case 0x23:
                  goto LABEL_113;
                case 0x24:
                  goto LABEL_403;
                case 0x25:
                  goto LABEL_87;
                case 0x26:
                  goto LABEL_110;
                case 0x27:
                  goto LABEL_98;
                case 0x28:
                  goto LABEL_83;
                case 0x29:
                  goto LABEL_80;
                case 0x2A:
                case 0x2B:
                case 0x2C:
                case 0x2D:
                case 0x2E:
                case 0x2F:
                case 0x30:
                case 0x31:
                case 0x32:
                case 0x33:
                case 0x34:
                case 0x35:
                case 0x36:
                case 0x37:
                case 0x38:
                case 0x39:
                case 0x3A:
                case 0x3B:
                  goto LABEL_62;
                case 0x3C:
                  goto LABEL_161;
                case 0x3D:
                  goto LABEL_160;
                case 0x3E:
                  goto LABEL_162;
                case 0x3F:
                  goto LABEL_159;
                case 0x40:
                  goto LABEL_129;
                case 0x41:
                  goto LABEL_127;
                case 0x42:
                  goto LABEL_126;
                case 0x43:
                  goto LABEL_125;
                case 0x44:
                  goto LABEL_124;
                case 0x45:
                  goto LABEL_123;
                case 0x46:
                  goto LABEL_122;
                case 0x47:
                  goto LABEL_121;
                case 0x48:
                  goto LABEL_120;
                case 0x49:
                  goto LABEL_119;
                case 0x4A:
                  goto LABEL_118;
                case 0x4B:
                  goto LABEL_117;
                case 0x4C:
                  goto LABEL_189;
                case 0x4D:
                  goto LABEL_188;
                case 0x4E:
                  goto LABEL_186;
                case 0x4F:
                  goto LABEL_185;
                case 0x50:
                  goto LABEL_179;
                case 0x51:
                  goto LABEL_170;
                case 0x52:
                  goto LABEL_169;
                case 0x53:
                  goto LABEL_163;
                case 0x54:
                  goto LABEL_213;
                case 0x55:
                  goto LABEL_201;
                case 0x56:
                  goto LABEL_206;
                case 0x57:
                case 0x58:
                  goto LABEL_198;
                case 0x59:
                case 0x60:
                  goto LABEL_200;
                case 0x5A:
                  goto LABEL_199;
                case 0x5B:
                  goto LABEL_131;
                case 0x5C:
                  goto LABEL_157;
                case 0x5D:
                  goto LABEL_154;
                case 0x5E:
                  goto LABEL_190;
                case 0x5F:
                  goto LABEL_136;
                default:
                  goto LABEL_517;
              }
            }
          }
          v33 = *a1;
          v223 = 1;
          v219.m128i_i64[0] = (__int64)"Operand is null";
          v222 = 3;
          if ( v33 )
          {
            sub_CA0E80(&v219, v33);
            v34 = *(_BYTE **)(v33 + 32);
            if ( (unsigned __int64)v34 >= *(_QWORD *)(v33 + 24) )
            {
              sub_CB5D20(v33, 10);
            }
            else
            {
              v23 = (__int64)(v34 + 1);
              *(_QWORD *)(v33 + 32) = v34 + 1;
              *v34 = 10;
            }
            a2 = *a1;
            *((_BYTE *)a1 + 152) = 1;
            if ( !a2 )
              goto LABEL_37;
            if ( *(_BYTE *)(v28 - 24) <= 0x1Cu )
            {
              sub_A5C020((_BYTE *)(v28 - 24), a2, 1, (__int64)(a1 + 2));
              v35 = *a1;
              v50 = *(_BYTE **)(*a1 + 32LL);
              if ( (unsigned __int64)v50 >= *(_QWORD *)(*a1 + 24LL) )
              {
LABEL_495:
                a2 = 10;
                sub_CB5D20(v35, 10);
                goto LABEL_37;
              }
              v23 = (__int64)(v50 + 1);
              *(_QWORD *)(v35 + 32) = v50 + 1;
              *v50 = 10;
            }
            else
            {
              sub_A693B0(v28 - 24, (_BYTE *)a2, (__int64)(a1 + 2), 0);
              v35 = *a1;
              v36 = *(_BYTE **)(*a1 + 32LL);
              if ( (unsigned __int64)v36 >= *(_QWORD *)(*a1 + 24LL) )
                goto LABEL_495;
              v23 = (__int64)(v36 + 1);
              *(_QWORD *)(v35 + 32) = v36 + 1;
              *v36 = 10;
            }
          }
          else
          {
            *((_BYTE *)a1 + 152) = 1;
          }
          goto LABEL_37;
        }
        switch ( *(_BYTE *)(v28 - 24) )
        {
          case 0x1E:
LABEL_78:
            a2 = v28 - 24;
            sub_BF9240(a1, v28 - 24);
            goto LABEL_37;
          case 0x1F:
          case 0x24:
            goto LABEL_403;
          case 0x20:
LABEL_79:
            a2 = v28 - 24;
            sub_BF93C0(a1, v28 - 24);
            goto LABEL_37;
          case 0x21:
LABEL_77:
            a2 = v28 - 24;
            sub_BF9780(a1, v28 - 24);
            goto LABEL_37;
          case 0x22:
LABEL_133:
            a2 = v28 - 24;
            sub_C04D20((__int64)a1, (_BYTE *)(v28 - 24));
            goto LABEL_37;
          case 0x23:
LABEL_113:
            if ( (*(_BYTE *)(sub_B43CB0(v28 - 24) + 2) & 8) != 0 )
            {
              v23 = a1[102];
              v72 = *(_QWORD *)(*(_QWORD *)(v28 - 56) + 8LL);
              if ( !v23 )
              {
                a1[102] = v72;
                goto LABEL_403;
              }
              if ( v23 == v72 )
                goto LABEL_403;
              v164 = *a1;
              v223 = 1;
              v219.m128i_i64[0] = (__int64)"The resume instruction should have a consistent result type inside a function.";
              v222 = 3;
              if ( !v164 )
              {
                *((_BYTE *)a1 + 152) = 1;
                goto LABEL_37;
              }
              a2 = v164;
              sub_CA0E80(&v219, v164);
              v165 = *(_BYTE **)(v164 + 32);
              if ( (unsigned __int64)v165 >= *(_QWORD *)(v164 + 24) )
              {
                a2 = 10;
                sub_CB5D20(v164, 10);
              }
              else
              {
                v23 = (__int64)(v165 + 1);
                *(_QWORD *)(v164 + 32) = v165 + 1;
                *v165 = 10;
              }
              v166 = *a1;
              *((_BYTE *)a1 + 152) = 1;
              if ( !v166 )
                goto LABEL_37;
            }
            else
            {
              v170 = *a1;
              v223 = 1;
              v219.m128i_i64[0] = (__int64)"ResumeInst needs to be in a function with a personality.";
              v222 = 3;
              if ( !v170 )
              {
                *((_BYTE *)a1 + 152) = 1;
                goto LABEL_37;
              }
              a2 = v170;
              sub_CA0E80(&v219, v170);
              v171 = *(_BYTE **)(v170 + 32);
              if ( (unsigned __int64)v171 >= *(_QWORD *)(v170 + 24) )
              {
                a2 = 10;
                sub_CB5D20(v170, 10);
              }
              else
              {
                v23 = (__int64)(v171 + 1);
                *(_QWORD *)(v170 + 32) = v171 + 1;
                *v171 = 10;
              }
              v172 = *a1;
              *((_BYTE *)a1 + 152) = 1;
              if ( !v172 )
                goto LABEL_37;
            }
            goto LABEL_511;
          case 0x25:
            v31 = 0;
LABEL_87:
            v209 = *(_BYTE **)(v28 - 32 * v31 - 24);
            if ( *v209 == 80 )
            {
              if ( (*(_BYTE *)(v28 - 22) & 1) == 0 )
                goto LABEL_403;
              v186 = *(_QWORD *)&v29[32 * (1 - v31)];
              if ( !v186 )
                goto LABEL_403;
              v187 = sub_AA4FF0(v186);
              if ( !v187 )
                BUG();
              v188 = *(unsigned __int8 *)(v187 - 24);
              if ( (unsigned int)(v188 - 39) <= 0x38
                && ((1LL << ((unsigned __int8)v188 - 39)) & 0x100060000000001LL) != 0
                && (_BYTE)v188 != 95 )
              {
LABEL_403:
                a2 = v28 - 24;
                sub_BF90E0(a1, v28 - 24);
                goto LABEL_37;
              }
              a2 = (__int64)&v219;
              v223 = 1;
              v219.m128i_i64[0] = (__int64)"CleanupReturnInst must unwind to an EH block which is not a landingpad.";
              v222 = 3;
              sub_BDBF70(a1, (__int64)&v219);
              if ( !*a1 )
                goto LABEL_37;
              goto LABEL_511;
            }
            v57 = *a1;
            v223 = 1;
            v219.m128i_i64[0] = (__int64)"CleanupReturnInst needs to be provided a CleanupPad";
            v222 = 3;
            if ( !v57 )
            {
              *((_BYTE *)a1 + 152) = 1;
              goto LABEL_37;
            }
            sub_CA0E80(&v219, v57);
            v58 = *(_BYTE **)(v57 + 32);
            if ( (unsigned __int64)v58 >= *(_QWORD *)(v57 + 24) )
            {
              sub_CB5D20(v57, 10);
            }
            else
            {
              v23 = (__int64)(v58 + 1);
              *(_QWORD *)(v57 + 32) = v58 + 1;
              *v58 = 10;
            }
            a2 = *a1;
            *((_BYTE *)a1 + 152) = 1;
            if ( !a2 )
              goto LABEL_37;
            if ( *(_BYTE *)(v28 - 24) <= 0x1Cu )
            {
              sub_A5C020((_BYTE *)(v28 - 24), a2, 1, (__int64)(a1 + 2));
              v59 = *a1;
              v60 = *(_BYTE **)(*a1 + 32LL);
              if ( (unsigned __int64)v60 < *(_QWORD *)(*a1 + 24LL) )
              {
LABEL_94:
                *(_QWORD *)(v59 + 32) = v60 + 1;
                *v60 = 10;
                goto LABEL_95;
              }
            }
            else
            {
              sub_A693B0(v28 - 24, (_BYTE *)a2, (__int64)(a1 + 2), 0);
              v59 = *a1;
              v60 = *(_BYTE **)(*a1 + 32LL);
              if ( (unsigned __int64)v60 < *(_QWORD *)(*a1 + 24LL) )
                goto LABEL_94;
            }
            sub_CB5D20(v59, 10);
LABEL_95:
            a2 = *a1;
            if ( *v209 <= 0x1Cu )
            {
              sub_A5C020(v209, a2, 1, (__int64)(a1 + 2));
              v35 = *a1;
              v196 = *(_BYTE **)(*a1 + 32LL);
              if ( (unsigned __int64)v196 >= *(_QWORD *)(*a1 + 24LL) )
                goto LABEL_495;
              v23 = (__int64)(v196 + 1);
              *(_QWORD *)(v35 + 32) = v196 + 1;
              *v196 = 10;
            }
            else
            {
              sub_A693B0((__int64)v209, (_BYTE *)a2, (__int64)(a1 + 2), 0);
              v35 = *a1;
              v61 = *(_BYTE **)(*a1 + 32LL);
              if ( (unsigned __int64)v61 >= *(_QWORD *)(*a1 + 24LL) )
                goto LABEL_495;
              v23 = (__int64)(v61 + 1);
              *(_QWORD *)(v35 + 32) = v61 + 1;
              *v61 = 10;
            }
            goto LABEL_37;
          case 0x26:
LABEL_110:
            v71 = *(_BYTE **)(v28 - 88);
            if ( *v71 == 81 )
              goto LABEL_403;
            a2 = (__int64)&v219;
            v223 = 1;
            v219.m128i_i64[0] = (__int64)"CatchReturnInst needs to be provided a CatchPad";
            v222 = 3;
            sub_BDBF70(a1, (__int64)&v219);
            if ( *a1 )
            {
              sub_BDBD80((__int64)a1, (_BYTE *)(v28 - 24));
              a2 = (__int64)v71;
              sub_BDBD80((__int64)a1, v71);
            }
            goto LABEL_37;
          case 0x27:
LABEL_98:
            v62 = *(_QWORD *)(v28 + 16);
            if ( (*(_BYTE *)(*(_QWORD *)(v62 + 72) + 2LL) & 8) == 0 )
            {
              v142 = *a1;
              v223 = 1;
              v219.m128i_i64[0] = (__int64)"CatchSwitchInst needs to be in a function with a personality.";
              v222 = 3;
              if ( !v142 )
              {
                *((_BYTE *)a1 + 152) = 1;
                goto LABEL_37;
              }
              a2 = v142;
              sub_CA0E80(&v219, v142);
              v143 = *(_BYTE **)(v142 + 32);
              if ( (unsigned __int64)v143 >= *(_QWORD *)(v142 + 24) )
              {
                a2 = 10;
                sub_CB5D20(v142, 10);
              }
              else
              {
                v23 = (__int64)(v143 + 1);
                *(_QWORD *)(v142 + 32) = v143 + 1;
                *v143 = 10;
              }
              v144 = *a1;
              *((_BYTE *)a1 + 152) = 1;
              if ( !v144 )
                goto LABEL_37;
              goto LABEL_511;
            }
            v63 = sub_AA4FF0(v62);
            if ( !v63 || v29 != (_BYTE *)(v63 - 24) )
            {
              v145 = *a1;
              v223 = 1;
              v219.m128i_i64[0] = (__int64)"CatchSwitchInst not the first non-PHI instruction in the block.";
              v222 = 3;
              if ( !v145 )
              {
                *((_BYTE *)a1 + 152) = 1;
                goto LABEL_37;
              }
              a2 = v145;
              sub_CA0E80(&v219, v145);
              v146 = *(_BYTE **)(v145 + 32);
              if ( (unsigned __int64)v146 >= *(_QWORD *)(v145 + 24) )
              {
                a2 = 10;
                sub_CB5D20(v145, 10);
              }
              else
              {
                v23 = (__int64)(v146 + 1);
                *(_QWORD *)(v145 + 32) = v146 + 1;
                *v146 = 10;
              }
              v147 = *a1;
              *((_BYTE *)a1 + 152) = 1;
              if ( !v147 )
                goto LABEL_37;
              goto LABEL_511;
            }
            v23 = *(_QWORD *)(v28 - 32);
            v64 = *(char **)v23;
            v65 = **(_BYTE **)v23;
            if ( v65 != 21 && (unsigned __int8)(v65 - 80) > 1u )
            {
              v167 = *a1;
              v223 = 1;
              v219.m128i_i64[0] = (__int64)"CatchSwitchInst has an invalid parent.";
              v222 = 3;
              if ( !v167 )
              {
                *((_BYTE *)a1 + 152) = 1;
                goto LABEL_37;
              }
              a2 = v167;
              sub_CA0E80(&v219, v167);
              v168 = *(_BYTE **)(v167 + 32);
              if ( (unsigned __int64)v168 >= *(_QWORD *)(v167 + 24) )
              {
                a2 = 10;
                sub_CB5D20(v167, 10);
              }
              else
              {
                v23 = (__int64)(v168 + 1);
                *(_QWORD *)(v167 + 32) = v168 + 1;
                *v168 = 10;
              }
              v169 = *a1;
              *((_BYTE *)a1 + 152) = 1;
              if ( v169 )
              {
LABEL_467:
                a2 = (__int64)v64;
                sub_BDBD80((__int64)a1, v64);
                goto LABEL_37;
              }
              goto LABEL_37;
            }
            if ( (*(_BYTE *)(v28 - 22) & 1) == 0 )
              goto LABEL_377;
            v66 = *(_QWORD *)(v23 + 32);
            if ( !v66 )
              goto LABEL_392;
            v67 = sub_AA4FF0(v66);
            if ( !v67 )
              BUG();
            v68 = *(unsigned __int8 *)(v67 - 24);
            v69 = (unsigned int)(v68 - 39);
            if ( (unsigned int)v69 > 0x38 || (v70 = 0x100060000000001LL, !_bittest64(&v70, v69)) || (_BYTE)v68 == 95 )
            {
              a2 = (__int64)&v219;
              v223 = 1;
              v219.m128i_i64[0] = (__int64)"CatchSwitchInst must unwind to an EH block which is not a landingpad.";
              v222 = 3;
              sub_BDBF70(a1, (__int64)&v219);
              if ( !*a1 )
                goto LABEL_37;
LABEL_511:
              a2 = v28 - 24;
              sub_BDBD80((__int64)a1, (_BYTE *)(v28 - 24));
              goto LABEL_37;
            }
            if ( (unsigned __int8)(v68 - 80) <= 1u )
            {
              v183 = *(char **)(v67 - 56);
LABEL_389:
              if ( v64 == v183 )
              {
                v219.m128i_i64[0] = v28 - 24;
                *(_QWORD *)sub_C04EB0((__int64)(a1 + 108), v219.m128i_i64) = v29;
              }
              goto LABEL_391;
            }
            v183 = **(char ***)(v67 - 32);
            if ( v183 )
              goto LABEL_389;
LABEL_391:
            if ( (*(_BYTE *)(v28 - 22) & 1) != 0 )
            {
LABEL_392:
              v184 = *(_DWORD *)(v28 - 20) & 0x7FFFFFF;
              if ( v184 != 2 )
              {
                v185 = *(_QWORD *)(v28 - 32);
                v178 = (__int64 *)(v185 + 32LL * v184);
                v179 = (__int64 *)(v185 + 64);
LABEL_379:
                if ( v178 == v179 )
                {
LABEL_385:
                  sub_BE1230(a1, (__int64)v29);
                  a2 = (__int64)v29;
                  sub_BF90E0(a1, (__int64)v29);
                  goto LABEL_37;
                }
                v210 = v29;
                v180 = v27;
                v181 = v179;
                while ( 1 )
                {
                  v64 = (char *)*v181;
                  v182 = sub_AA4FF0(*v181);
                  if ( !v182 )
                    BUG();
                  if ( *(_BYTE *)(v182 - 24) != 81 )
                    break;
                  v181 += 4;
                  if ( v178 == v181 )
                  {
                    v27 = v180;
                    v29 = v210;
                    goto LABEL_385;
                  }
                }
                a2 = (__int64)&v219;
                v223 = 1;
                v27 = v180;
                v222 = 3;
                v219.m128i_i64[0] = (__int64)"CatchSwitchInst handlers must be catchpads";
                sub_BDBF70(a1, (__int64)&v219);
                if ( *a1 )
                {
                  a2 = (__int64)v210;
                  sub_BDBD80((__int64)a1, v210);
                  if ( v64 )
                    goto LABEL_467;
                }
                goto LABEL_37;
              }
            }
            else
            {
LABEL_377:
              v176 = *(_DWORD *)(v28 - 20) & 0x7FFFFFF;
              if ( v176 != 1 )
              {
                v177 = *(_QWORD *)(v28 - 32);
                v178 = (__int64 *)(v177 + 32LL * v176);
                v179 = (__int64 *)(v177 + 32);
                goto LABEL_379;
              }
            }
            a2 = (__int64)&v219;
            v223 = 1;
            v219.m128i_i64[0] = (__int64)"CatchSwitchInst cannot have empty handler list";
            v222 = 3;
            sub_BDBF70(a1, (__int64)&v219);
            if ( *a1 )
              goto LABEL_511;
LABEL_37:
            if ( v215 == v27 )
              goto LABEL_38;
            break;
          case 0x28:
LABEL_83:
            v56 = *(_BYTE **)(v28 - 56);
            if ( *v56 == 25 )
            {
              if ( v56[104] )
              {
                a2 = (__int64)&v219;
                v223 = 1;
                v219.m128i_i64[0] = (__int64)"Unwinding from Callbr is not allowed";
                v222 = 3;
                sub_BDBF70(a1, (__int64)&v219);
              }
              else
              {
                sub_BDD0F0(a1, v28 - 24);
                a2 = v28 - 24;
                sub_BF90E0(a1, v28 - 24);
              }
              goto LABEL_37;
            }
            v130 = *a1;
            v223 = 1;
            v219.m128i_i64[0] = (__int64)"Callbr is currently only used for asm-goto!";
            v222 = 3;
            if ( !v130 )
            {
              *((_BYTE *)a1 + 152) = 1;
              goto LABEL_37;
            }
            a2 = v130;
            sub_CA0E80(&v219, v130);
            v131 = *(_BYTE **)(v130 + 32);
            if ( (unsigned __int64)v131 >= *(_QWORD *)(v130 + 24) )
            {
              a2 = 10;
              sub_CB5D20(v130, 10);
            }
            else
            {
              v23 = (__int64)(v131 + 1);
              *(_QWORD *)(v130 + 32) = v131 + 1;
              *v131 = 10;
            }
            v132 = *a1;
            *((_BYTE *)a1 + 152) = 1;
            if ( v132 )
              goto LABEL_511;
            goto LABEL_37;
          case 0x29:
LABEL_80:
            v55 = *(_QWORD *)(v28 - 16);
            if ( v55 == *(_QWORD *)(*(_QWORD *)(v28 - 56) + 8LL) )
            {
              if ( (unsigned __int8)sub_BDB700(v55) )
                goto LABEL_200;
              a2 = (__int64)&v219;
              v223 = 1;
              v219.m128i_i64[0] = (__int64)"FNeg operator only works with float types!";
              v222 = 3;
              sub_BDBF70(a1, (__int64)&v219);
              if ( !*a1 )
                goto LABEL_37;
            }
            else
            {
              a2 = (__int64)&v219;
              v223 = 1;
              v219.m128i_i64[0] = (__int64)"Unary operators must have same type foroperands and result!";
              v222 = 3;
              sub_BDBF70(a1, (__int64)&v219);
              if ( !*a1 )
                goto LABEL_37;
            }
            goto LABEL_511;
          case 0x2A:
          case 0x2B:
          case 0x2C:
          case 0x2D:
          case 0x2E:
          case 0x2F:
          case 0x30:
          case 0x31:
          case 0x32:
          case 0x33:
          case 0x34:
          case 0x35:
          case 0x36:
          case 0x37:
          case 0x38:
          case 0x39:
          case 0x3A:
          case 0x3B:
LABEL_62:
            a2 = v28 - 24;
            sub_BF98B0(a1, (unsigned __int8 *)(v28 - 24));
            goto LABEL_37;
          case 0x3C:
LABEL_161:
            a2 = v28 - 24;
            sub_BF9C00((__int64)a1, v28 - 24);
            goto LABEL_37;
          case 0x3D:
LABEL_160:
            a2 = v28 - 24;
            sub_BF9E70((__int64)a1, v28 - 24);
            goto LABEL_37;
          case 0x3E:
LABEL_162:
            a2 = v28 - 24;
            sub_BFA0F0((__int64)a1, v28 - 24);
            goto LABEL_37;
          case 0x3F:
LABEL_159:
            a2 = v28 - 24;
            sub_BFA370(a1, v28 - 24);
            goto LABEL_37;
          case 0x40:
LABEL_129:
            if ( (*(_WORD *)(v28 - 22) & 7u) - 4 <= 3 )
              goto LABEL_200;
            a2 = (__int64)&v219;
            v223 = 1;
            v219.m128i_i64[0] = (__int64)"fence instructions may only have acquire, release, acq_rel, or seq_cst ordering.";
            v222 = 3;
            sub_BDBF70(a1, (__int64)&v219);
            if ( *a1 )
              goto LABEL_511;
            goto LABEL_37;
          case 0x41:
LABEL_127:
            v73 = *(_QWORD *)(*(_QWORD *)(v28 - 88) + 8LL);
            if ( (*(_BYTE *)(v73 + 8) & 0xFD) == 0xC )
            {
              sub_BDBDF0((__int64)a1, v73, v29);
              a2 = (__int64)v29;
              sub_BF6FE0((__int64)a1, (__int64)v29);
            }
            else
            {
              v126 = *a1;
              v223 = 1;
              v219.m128i_i64[0] = (__int64)"cmpxchg operand must have integer or pointer type";
              v222 = 3;
              if ( v126 )
              {
                a2 = v126;
                sub_CA0E80(&v219, v126);
                v127 = *(_BYTE **)(v126 + 32);
                if ( (unsigned __int64)v127 >= *(_QWORD *)(v126 + 24) )
                {
                  a2 = 10;
                  sub_CB5D20(v126, 10);
                }
                else
                {
                  v23 = (__int64)(v127 + 1);
                  *(_QWORD *)(v126 + 32) = v127 + 1;
                  *v127 = 10;
                }
                v128 = *a1;
                *((_BYTE *)a1 + 152) = 1;
                if ( v128 )
                {
                  v129 = *(_BYTE **)(v128 + 32);
                  if ( (unsigned __int64)v129 >= *(_QWORD *)(v128 + 24) )
                  {
                    v128 = sub_CB5D20(v128, 32);
                  }
                  else
                  {
                    *(_QWORD *)(v128 + 32) = v129 + 1;
                    *v129 = 32;
                  }
                  sub_A587F0(v73, v128, 0, 0);
                  a2 = (__int64)v29;
                  sub_BDBD80((__int64)a1, v29);
                }
              }
              else
              {
                *((_BYTE *)a1 + 152) = 1;
              }
            }
            goto LABEL_37;
          case 0x42:
LABEL_126:
            a2 = v28 - 24;
            sub_BFABA0(a1, v28 - 24);
            goto LABEL_37;
          case 0x43:
LABEL_125:
            a2 = v28 - 24;
            sub_BFAE80((__int64)a1, v28 - 24);
            goto LABEL_37;
          case 0x44:
LABEL_124:
            a2 = v28 - 24;
            sub_BFB020((__int64)a1, v28 - 24);
            goto LABEL_37;
          case 0x45:
LABEL_123:
            a2 = v28 - 24;
            sub_BFB1C0((__int64)a1, v28 - 24);
            goto LABEL_37;
          case 0x46:
LABEL_122:
            a2 = v28 - 24;
            sub_BFB360((__int64)a1, v28 - 24);
            goto LABEL_37;
          case 0x47:
LABEL_121:
            a2 = v28 - 24;
            sub_BFB520((__int64)a1, v28 - 24);
            goto LABEL_37;
          case 0x48:
LABEL_120:
            a2 = v28 - 24;
            sub_BFB6E0((__int64)a1, v28 - 24);
            goto LABEL_37;
          case 0x49:
LABEL_119:
            a2 = v28 - 24;
            sub_BFB8B0((__int64)a1, v28 - 24);
            goto LABEL_37;
          case 0x4A:
LABEL_118:
            a2 = v28 - 24;
            sub_BFBA80((__int64)a1, v28 - 24);
            goto LABEL_37;
          case 0x4B:
LABEL_117:
            a2 = v28 - 24;
            sub_BFBC40((__int64)a1, v28 - 24);
            goto LABEL_37;
          case 0x4C:
LABEL_189:
            a2 = v28 - 24;
            sub_BFBE00((__int64)a1, v28 - 24);
            goto LABEL_37;
          case 0x4D:
LABEL_188:
            a2 = v28 - 24;
            sub_BFBF90((__int64)a1, v28 - 24);
            goto LABEL_37;
          case 0x4E:
LABEL_186:
            if ( (unsigned __int8)sub_B50F30(49, *(_QWORD *)(*(_QWORD *)(v28 - 56) + 8LL), *(_QWORD *)(v28 - 16)) )
              goto LABEL_200;
            a2 = (__int64)&v219;
            v223 = 1;
            v219.m128i_i64[0] = (__int64)"Invalid bitcast";
            v222 = 3;
            sub_BDBF70(a1, (__int64)&v219);
            if ( *a1 )
              goto LABEL_511;
            goto LABEL_37;
          case 0x4F:
LABEL_185:
            a2 = v28 - 24;
            sub_BFC120((__int64)a1, v28 - 24);
            goto LABEL_37;
          case 0x50:
LABEL_179:
            v87 = *(_QWORD *)(v28 + 16);
            if ( (*(_BYTE *)(*(_QWORD *)(v87 + 72) + 2LL) & 8) != 0 )
            {
              v88 = sub_AA4FF0(v87);
              if ( v88 && v29 == (_BYTE *)(v88 - 24) )
              {
                v89 = **(_BYTE **)(v28 - 56);
                if ( v89 == 21 || (unsigned __int8)(v89 - 80) <= 1u )
                {
                  sub_BE1230(a1, v28 - 24);
                  a2 = v28 - 24;
                  sub_C05160((__int64)a1, (_BYTE *)(v28 - 24));
                  goto LABEL_37;
                }
                v149 = *a1;
                v223 = 1;
                v219.m128i_i64[0] = (__int64)"CleanupPadInst has an invalid parent.";
                v222 = 3;
                if ( !v149 )
                {
                  *((_BYTE *)a1 + 152) = 1;
                  goto LABEL_37;
                }
                a2 = v149;
                sub_CA0E80(&v219, v149);
                v150 = *(_BYTE **)(v149 + 32);
                if ( (unsigned __int64)v150 >= *(_QWORD *)(v149 + 24) )
                {
                  a2 = 10;
                  sub_CB5D20(v149, 10);
                }
                else
                {
                  v23 = (__int64)(v150 + 1);
                  *(_QWORD *)(v149 + 32) = v150 + 1;
                  *v150 = 10;
                }
                v151 = *a1;
                *((_BYTE *)a1 + 152) = 1;
                if ( !v151 )
                  goto LABEL_37;
              }
              else
              {
                v152 = *a1;
                v223 = 1;
                v219.m128i_i64[0] = (__int64)"CleanupPadInst not the first non-PHI instruction in the block.";
                v222 = 3;
                if ( !v152 )
                {
                  *((_BYTE *)a1 + 152) = 1;
                  goto LABEL_37;
                }
                a2 = v152;
                sub_CA0E80(&v219, v152);
                v153 = *(_BYTE **)(v152 + 32);
                if ( (unsigned __int64)v153 >= *(_QWORD *)(v152 + 24) )
                {
                  a2 = 10;
                  sub_CB5D20(v152, 10);
                }
                else
                {
                  v23 = (__int64)(v153 + 1);
                  *(_QWORD *)(v152 + 32) = v153 + 1;
                  *v153 = 10;
                }
                v154 = *a1;
                *((_BYTE *)a1 + 152) = 1;
                if ( !v154 )
                  goto LABEL_37;
              }
            }
            else
            {
              v136 = *a1;
              v223 = 1;
              v219.m128i_i64[0] = (__int64)"CleanupPadInst needs to be in a function with a personality.";
              v222 = 3;
              if ( !v136 )
              {
                *((_BYTE *)a1 + 152) = 1;
                goto LABEL_37;
              }
              a2 = v136;
              sub_CA0E80(&v219, v136);
              v137 = *(_BYTE **)(v136 + 32);
              if ( (unsigned __int64)v137 >= *(_QWORD *)(v136 + 24) )
              {
                a2 = 10;
                sub_CB5D20(v136, 10);
              }
              else
              {
                v23 = (__int64)(v137 + 1);
                *(_QWORD *)(v136 + 32) = v137 + 1;
                *v137 = 10;
              }
              v138 = *a1;
              *((_BYTE *)a1 + 152) = 1;
              if ( !v138 )
                goto LABEL_37;
            }
            goto LABEL_511;
          case 0x51:
LABEL_170:
            v82 = *(_QWORD *)(v28 + 16);
            if ( (*(_BYTE *)(*(_QWORD *)(v82 + 72) + 2LL) & 8) != 0 )
            {
              v83 = *(_BYTE **)(v28 - 56);
              if ( *v83 == 39 )
              {
                v190 = sub_AA4FF0(v82);
                if ( v190 && v29 == (_BYTE *)(v190 - 24) )
                {
                  sub_BE1230(a1, (__int64)v29);
                  a2 = (__int64)v29;
                  sub_C05160((__int64)a1, v29);
                }
                else
                {
                  a2 = (__int64)&v219;
                  v223 = 1;
                  v219.m128i_i64[0] = (__int64)"CatchPadInst not the first non-PHI instruction in the block.";
                  v222 = 3;
                  sub_BDBF70(a1, (__int64)&v219);
                  if ( *a1 )
                  {
                    a2 = (__int64)v29;
                    sub_BDBD80((__int64)a1, v29);
                  }
                }
              }
              else
              {
                v84 = *a1;
                v223 = 1;
                v219.m128i_i64[0] = (__int64)"CatchPadInst needs to be directly nested in a CatchSwitchInst.";
                v222 = 3;
                if ( !v84 )
                {
                  *((_BYTE *)a1 + 152) = 1;
                  goto LABEL_37;
                }
                sub_CA0E80(&v219, v84);
                v85 = *(_BYTE **)(v84 + 32);
                if ( (unsigned __int64)v85 >= *(_QWORD *)(v84 + 24) )
                {
                  sub_CB5D20(v84, 10);
                }
                else
                {
                  v23 = (__int64)(v85 + 1);
                  *(_QWORD *)(v84 + 32) = v85 + 1;
                  *v85 = 10;
                }
                a2 = *a1;
                *((_BYTE *)a1 + 152) = 1;
                if ( a2 )
                {
                  if ( *v83 <= 0x1Cu )
                  {
                    sub_A5C020(v83, a2, 1, (__int64)(a1 + 2));
                    v35 = *a1;
                    v189 = *(_BYTE **)(*a1 + 32LL);
                    if ( (unsigned __int64)v189 >= *(_QWORD *)(*a1 + 24LL) )
                      goto LABEL_495;
                    v23 = (__int64)(v189 + 1);
                    *(_QWORD *)(v35 + 32) = v189 + 1;
                    *v189 = 10;
                  }
                  else
                  {
                    sub_A693B0((__int64)v83, (_BYTE *)a2, (__int64)(a1 + 2), 0);
                    v35 = *a1;
                    v86 = *(_BYTE **)(*a1 + 32LL);
                    if ( (unsigned __int64)v86 >= *(_QWORD *)(*a1 + 24LL) )
                      goto LABEL_495;
                    v23 = (__int64)(v86 + 1);
                    *(_QWORD *)(v35 + 32) = v86 + 1;
                    *v86 = 10;
                  }
                }
              }
              goto LABEL_37;
            }
            v155 = *a1;
            v223 = 1;
            v219.m128i_i64[0] = (__int64)"CatchPadInst needs to be in a function with a personality.";
            v222 = 3;
            if ( !v155 )
            {
              *((_BYTE *)a1 + 152) = 1;
              goto LABEL_37;
            }
            a2 = v155;
            sub_CA0E80(&v219, v155);
            v156 = *(_BYTE **)(v155 + 32);
            if ( (unsigned __int64)v156 >= *(_QWORD *)(v155 + 24) )
            {
              a2 = 10;
              sub_CB5D20(v155, 10);
            }
            else
            {
              v23 = (__int64)(v156 + 1);
              *(_QWORD *)(v155 + 32) = v156 + 1;
              *v156 = 10;
            }
            v157 = *a1;
            *((_BYTE *)a1 + 152) = 1;
            if ( !v157 )
              goto LABEL_37;
            goto LABEL_511;
          case 0x52:
LABEL_169:
            a2 = v28 - 24;
            sub_BFC2F0((__int64)a1, v28 - 24);
            goto LABEL_37;
          case 0x53:
LABEL_163:
            v23 = *(_QWORD *)(*(_QWORD *)(v28 - 88) + 8LL);
            if ( v23 == *(_QWORD *)(*(_QWORD *)(v28 - 56) + 8LL) )
            {
              v81 = *(unsigned __int8 *)(v23 + 8);
              v24 = (unsigned int)(v81 - 17);
              if ( (unsigned int)v24 <= 1 )
                LOBYTE(v81) = *(_BYTE *)(**(_QWORD **)(v23 + 16) + 8LL);
              if ( (unsigned __int8)v81 <= 3u || (_BYTE)v81 == 5 || (v81 & 0xFD) == 4 )
              {
                if ( (*(_BYTE *)(v28 - 22) & 0x30) == 0 )
                  goto LABEL_200;
                v173 = *a1;
                v223 = 1;
                v219.m128i_i64[0] = (__int64)"Invalid predicate in FCmp instruction!";
                v222 = 3;
                if ( !v173 )
                {
                  *((_BYTE *)a1 + 152) = 1;
                  goto LABEL_37;
                }
                a2 = v173;
                sub_CA0E80(&v219, v173);
                v174 = *(_BYTE **)(v173 + 32);
                if ( (unsigned __int64)v174 >= *(_QWORD *)(v173 + 24) )
                {
                  a2 = 10;
                  sub_CB5D20(v173, 10);
                }
                else
                {
                  v23 = (__int64)(v174 + 1);
                  *(_QWORD *)(v173 + 32) = v174 + 1;
                  *v174 = 10;
                }
                v175 = *a1;
                *((_BYTE *)a1 + 152) = 1;
                if ( !v175 )
                  goto LABEL_37;
              }
              else
              {
                v107 = *a1;
                v223 = 1;
                v219.m128i_i64[0] = (__int64)"Invalid operand types for FCmp instruction";
                v222 = 3;
                if ( !v107 )
                {
                  *((_BYTE *)a1 + 152) = 1;
                  goto LABEL_37;
                }
                a2 = v107;
                sub_CA0E80(&v219, v107);
                v108 = *(_BYTE **)(v107 + 32);
                if ( (unsigned __int64)v108 >= *(_QWORD *)(v107 + 24) )
                {
                  a2 = 10;
                  sub_CB5D20(v107, 10);
                }
                else
                {
                  v23 = (__int64)(v108 + 1);
                  *(_QWORD *)(v107 + 32) = v108 + 1;
                  *v108 = 10;
                }
                v109 = *a1;
                *((_BYTE *)a1 + 152) = 1;
                if ( !v109 )
                  goto LABEL_37;
              }
            }
            else
            {
              v133 = *a1;
              v223 = 1;
              v219.m128i_i64[0] = (__int64)"Both operands to FCmp instruction are not of the same type!";
              v222 = 3;
              if ( !v133 )
              {
                *((_BYTE *)a1 + 152) = 1;
                goto LABEL_37;
              }
              a2 = v133;
              sub_CA0E80(&v219, v133);
              v134 = *(_BYTE **)(v133 + 32);
              if ( (unsigned __int64)v134 >= *(_QWORD *)(v133 + 24) )
              {
                a2 = 10;
                sub_CB5D20(v133, 10);
              }
              else
              {
                v23 = (__int64)(v134 + 1);
                *(_QWORD *)(v133 + 32) = v134 + 1;
                *v134 = 10;
              }
              v135 = *a1;
              *((_BYTE *)a1 + 152) = 1;
              if ( !v135 )
                goto LABEL_37;
            }
            goto LABEL_511;
          case 0x54:
LABEL_213:
            a2 = v28 - 24;
            sub_BFC450((__int64)a1, v28 - 24);
            goto LABEL_37;
          case 0x55:
LABEL_201:
            v94 = *(_QWORD *)(v28 - 56);
            if ( v94
              && !*(_BYTE *)v94
              && *(_QWORD *)(v94 + 24) == *(_QWORD *)(v28 + 56)
              && ((v148 = *(_DWORD *)(v94 + 36), v148 >= 0x46) || v148) )
            {
              a2 = v28 - 24;
              sub_C04E70(a1, v28 - 24);
            }
            else
            {
              a2 = v28 - 24;
              sub_BFC6A0(a1, v28 - 24);
              if ( (*(_WORD *)(v28 - 22) & 3) == 2 )
              {
                a2 = v28 - 24;
                sub_BEEBD0(a1, v28 - 24, v23, v24, v95, v96);
              }
            }
            goto LABEL_37;
          case 0x56:
LABEL_206:
            a2 = *(_QWORD *)(v28 - 88);
            if ( sub_B489D0(*(_QWORD *)(v28 - 120), a2, *(_QWORD *)(v28 - 56)) )
            {
              v139 = *a1;
              v223 = 1;
              v219.m128i_i64[0] = (__int64)"Invalid operands for select instruction!";
              v222 = 3;
              if ( !v139 )
              {
                *((_BYTE *)a1 + 152) = 1;
                goto LABEL_37;
              }
              a2 = v139;
              sub_CA0E80(&v219, v139);
              v140 = *(_BYTE **)(v139 + 32);
              if ( (unsigned __int64)v140 >= *(_QWORD *)(v139 + 24) )
              {
                a2 = 10;
                sub_CB5D20(v139, 10);
              }
              else
              {
                v23 = (__int64)(v140 + 1);
                *(_QWORD *)(v139 + 32) = v140 + 1;
                *v140 = 10;
              }
              v141 = *a1;
              *((_BYTE *)a1 + 152) = 1;
              if ( !v141 )
                goto LABEL_37;
            }
            else
            {
              v24 = *(_QWORD *)(v28 - 16);
              if ( *(_QWORD *)(*(_QWORD *)(v28 - 88) + 8LL) == v24 )
                goto LABEL_200;
              v97 = *a1;
              v223 = 1;
              v219.m128i_i64[0] = (__int64)"Select values must have same type as select instruction!";
              v222 = 3;
              if ( !v97 )
              {
                *((_BYTE *)a1 + 152) = 1;
                goto LABEL_37;
              }
              a2 = v97;
              sub_CA0E80(&v219, v97);
              v98 = *(_BYTE **)(v97 + 32);
              if ( (unsigned __int64)v98 >= *(_QWORD *)(v97 + 24) )
              {
                a2 = 10;
                sub_CB5D20(v97, 10);
              }
              else
              {
                v23 = (__int64)(v98 + 1);
                *(_QWORD *)(v97 + 32) = v98 + 1;
                *v98 = 10;
              }
              v99 = *a1;
              *((_BYTE *)a1 + 152) = 1;
              if ( !v99 )
                goto LABEL_37;
            }
            goto LABEL_511;
          case 0x57:
          case 0x58:
LABEL_198:
            a2 = (__int64)&v219;
            v218 = (_BYTE *)(v28 - 24);
            v223 = 1;
            v219.m128i_i64[0] = (__int64)"User-defined operators should not live outside of a pass!";
            v222 = 3;
            sub_BE0C10(a1, (__int64)&v219, &v218);
            goto LABEL_37;
          case 0x59:
          case 0x60:
            goto LABEL_200;
          case 0x5A:
LABEL_199:
            if ( (unsigned __int8)sub_B4DF70(*(_QWORD *)(v28 - 88), *(_QWORD *)(v28 - 56)) )
              goto LABEL_200;
            a2 = (__int64)&v219;
            v223 = 1;
            v219.m128i_i64[0] = (__int64)"Invalid extractelement operands!";
            v222 = 3;
            sub_BDBF70(a1, (__int64)&v219);
            if ( *a1 )
              goto LABEL_511;
            goto LABEL_37;
          case 0x5B:
LABEL_131:
            if ( (unsigned __int8)sub_B4E100(*(_QWORD *)(v28 - 120), *(_QWORD *)(v28 - 88), *(_QWORD *)(v28 - 56)) )
              goto LABEL_200;
            a2 = (__int64)&v219;
            v223 = 1;
            v219.m128i_i64[0] = (__int64)"Invalid insertelement operands!";
            v222 = 3;
            sub_BDBF70(a1, (__int64)&v219);
            if ( *a1 )
              goto LABEL_511;
            goto LABEL_37;
          case 0x5C:
LABEL_157:
            if ( (unsigned __int8)sub_B4E140(
                                    *(_QWORD *)(v28 - 88),
                                    *(_QWORD *)(v28 - 56),
                                    *(_DWORD **)(v28 + 48),
                                    *(unsigned int *)(v28 + 56)) )
              goto LABEL_200;
            a2 = (__int64)&v219;
            v223 = 1;
            v219.m128i_i64[0] = (__int64)"Invalid shufflevector operands!";
            v222 = 3;
            sub_BDBF70(a1, (__int64)&v219);
            if ( *a1 )
              goto LABEL_511;
            goto LABEL_37;
          case 0x5D:
LABEL_154:
            if ( *(_QWORD *)(v28 - 16) == sub_B501B0(
                                            *(_QWORD *)(*(_QWORD *)(v28 - 56) + 8LL),
                                            *(unsigned int **)(v28 + 48),
                                            *(unsigned int *)(v28 + 56)) )
              goto LABEL_200;
            a2 = (__int64)&v219;
            v223 = 1;
            v219.m128i_i64[0] = (__int64)"Invalid ExtractValueInst operands!";
            v222 = 3;
            sub_BDBF70(a1, (__int64)&v219);
            if ( *a1 )
              goto LABEL_511;
            goto LABEL_37;
          case 0x5E:
LABEL_190:
            a2 = *(_QWORD *)(v28 + 48);
            v90 = sub_B501B0(*(_QWORD *)(*(_QWORD *)(v28 - 88) + 8LL), (unsigned int *)a2, *(unsigned int *)(v28 + 56));
            v23 = *(_QWORD *)(v28 - 56);
            if ( *(_QWORD *)(v23 + 8) == v90 )
              goto LABEL_200;
            v91 = *a1;
            v223 = 1;
            v219.m128i_i64[0] = (__int64)"Invalid InsertValueInst operands!";
            v222 = 3;
            if ( !v91 )
            {
              *((_BYTE *)a1 + 152) = 1;
              goto LABEL_37;
            }
            sub_CA0E80(&v219, v91);
            v92 = *(_BYTE **)(v91 + 32);
            if ( (unsigned __int64)v92 >= *(_QWORD *)(v91 + 24) )
            {
              sub_CB5D20(v91, 10);
            }
            else
            {
              v23 = (__int64)(v92 + 1);
              *(_QWORD *)(v91 + 32) = v92 + 1;
              *v92 = 10;
            }
            a2 = *a1;
            *((_BYTE *)a1 + 152) = 1;
            if ( !a2 )
              goto LABEL_37;
            if ( *(_BYTE *)(v28 - 24) <= 0x1Cu )
            {
              sub_A5C020((_BYTE *)(v28 - 24), a2, 1, (__int64)(a1 + 2));
              v35 = *a1;
              v200 = *(_BYTE **)(*a1 + 32LL);
              if ( (unsigned __int64)v200 >= *(_QWORD *)(*a1 + 24LL) )
                goto LABEL_495;
              v23 = (__int64)(v200 + 1);
              *(_QWORD *)(v35 + 32) = v200 + 1;
              *v200 = 10;
            }
            else
            {
              sub_A693B0(v28 - 24, (_BYTE *)a2, (__int64)(a1 + 2), 0);
              v35 = *a1;
              v93 = *(_BYTE **)(*a1 + 32LL);
              if ( (unsigned __int64)v93 >= *(_QWORD *)(*a1 + 24LL) )
                goto LABEL_495;
              v23 = (__int64)(v93 + 1);
              *(_QWORD *)(v35 + 32) = v93 + 1;
              *v93 = 10;
            }
            goto LABEL_37;
          case 0x5F:
            if ( (*(_BYTE *)(v28 - 22) & 1) == 0 )
            {
              a2 = (__int64)&v219;
              v223 = 1;
              v219.m128i_i64[0] = (__int64)"LandingPadInst needs at least one clause or to be a cleanup.";
              v222 = 3;
              sub_BDBF70(a1, (__int64)&v219);
              if ( !*a1 )
                goto LABEL_37;
              goto LABEL_511;
            }
LABEL_136:
            a2 = v28 - 24;
            sub_BE1230(a1, v28 - 24);
            v74 = a1[102];
            if ( v74 )
            {
              if ( v74 != *(_QWORD *)(v28 - 16) )
              {
                v161 = *a1;
                v223 = 1;
                v219.m128i_i64[0] = (__int64)"The landingpad instruction should have a consistent result type inside a function.";
                v222 = 3;
                if ( !v161 )
                {
                  *((_BYTE *)a1 + 152) = 1;
                  goto LABEL_37;
                }
                a2 = v161;
                sub_CA0E80(&v219, v161);
                v162 = *(_BYTE **)(v161 + 32);
                if ( (unsigned __int64)v162 >= *(_QWORD *)(v161 + 24) )
                {
                  a2 = 10;
                  sub_CB5D20(v161, 10);
                }
                else
                {
                  v23 = (__int64)(v162 + 1);
                  *(_QWORD *)(v161 + 32) = v162 + 1;
                  *v162 = 10;
                }
                v163 = *a1;
                *((_BYTE *)a1 + 152) = 1;
                if ( !v163 )
                  goto LABEL_37;
                goto LABEL_511;
              }
            }
            else
            {
              a1[102] = *(_QWORD *)(v28 - 16);
            }
            v75 = *(_QWORD *)(v28 + 16);
            if ( (*(_BYTE *)(*(_QWORD *)(v75 + 72) + 2LL) & 8) != 0 )
            {
              if ( v29 == (_BYTE *)sub_AA5EB0(v75) )
              {
                if ( (*(_DWORD *)(v28 - 20) & 0x7FFFFFF) == 0 )
                {
LABEL_200:
                  a2 = v28 - 24;
                  sub_BF6FE0((__int64)a1, v28 - 24);
                  goto LABEL_37;
                }
                v76 = 0;
                v77 = 32LL * (*(_DWORD *)(v28 - 20) & 0x7FFFFFF);
                v23 = *(_BYTE *)(v28 - 17) & 0x40;
                while ( 2 )
                {
                  if ( (_BYTE)v23 )
                  {
                    v24 = *(_QWORD *)(*(_QWORD *)(v28 - 32) + v76);
                    a2 = *(unsigned __int8 *)(*(_QWORD *)(v24 + 8) + 8LL);
                    if ( (_BYTE)a2 == 16 )
                      goto LABEL_147;
                  }
                  else
                  {
                    v24 = *(_QWORD *)(v28 - v77 + v76 - 24);
                    a2 = *(unsigned __int8 *)(*(_QWORD *)(v24 + 8) + 8LL);
                    if ( (_BYTE)a2 == 16 )
                    {
LABEL_147:
                      v24 = *(unsigned __int8 *)v24;
                      if ( (_BYTE)v24 != 14 && (_BYTE)v24 != 9 )
                      {
                        v78 = *a1;
                        v223 = 1;
                        v219.m128i_i64[0] = (__int64)"Filter operand is not an array of constants!";
                        v222 = 3;
                        if ( v78 )
                        {
                          a2 = v78;
                          sub_CA0E80(&v219, v78);
                          v79 = *(_BYTE **)(v78 + 32);
                          if ( (unsigned __int64)v79 >= *(_QWORD *)(v78 + 24) )
                          {
                            a2 = 10;
                            sub_CB5D20(v78, 10);
                          }
                          else
                          {
                            v23 = (__int64)(v79 + 1);
                            *(_QWORD *)(v78 + 32) = v79 + 1;
                            *v79 = 10;
                          }
                        }
                        v80 = *a1 == 0;
                        *((_BYTE *)a1 + 152) = 1;
                        if ( v80 )
                          goto LABEL_37;
                        goto LABEL_511;
                      }
LABEL_144:
                      v76 += 32;
                      if ( v77 == v76 )
                        goto LABEL_200;
                      continue;
                    }
                  }
                  break;
                }
                if ( (_BYTE)a2 != 14 )
                {
                  v207 = *a1;
                  v223 = 1;
                  v219.m128i_i64[0] = (__int64)"Catch operand does not have pointer type!";
                  v222 = 3;
                  if ( v207 )
                  {
                    a2 = v207;
                    sub_CA0E80(&v219, v207);
                    v208 = *(_BYTE **)(v207 + 32);
                    if ( (unsigned __int64)v208 >= *(_QWORD *)(v207 + 24) )
                    {
                      a2 = 10;
                      sub_CB5D20(v207, 10);
                    }
                    else
                    {
                      v23 = (__int64)(v208 + 1);
                      *(_QWORD *)(v207 + 32) = v208 + 1;
                      *v208 = 10;
                    }
                  }
                  v80 = *a1 == 0;
                  *((_BYTE *)a1 + 152) = 1;
                  if ( v80 )
                    goto LABEL_37;
                  goto LABEL_511;
                }
                goto LABEL_144;
              }
              v197 = *a1;
              v223 = 1;
              v219.m128i_i64[0] = (__int64)"LandingPadInst not the first non-PHI instruction in the block.";
              v222 = 3;
              if ( !v197 )
              {
                *((_BYTE *)a1 + 152) = 1;
                goto LABEL_37;
              }
              a2 = v197;
              sub_CA0E80(&v219, v197);
              v198 = *(_BYTE **)(v197 + 32);
              if ( (unsigned __int64)v198 >= *(_QWORD *)(v197 + 24) )
              {
                a2 = 10;
                sub_CB5D20(v197, 10);
              }
              else
              {
                v23 = (__int64)(v198 + 1);
                *(_QWORD *)(v197 + 32) = v198 + 1;
                *v198 = 10;
              }
              v199 = *a1;
              *((_BYTE *)a1 + 152) = 1;
              if ( !v199 )
                goto LABEL_37;
            }
            else
            {
              v158 = *a1;
              v223 = 1;
              v219.m128i_i64[0] = (__int64)"LandingPadInst needs to be in a function with a personality.";
              v222 = 3;
              if ( !v158 )
              {
                *((_BYTE *)a1 + 152) = 1;
                goto LABEL_37;
              }
              a2 = v158;
              sub_CA0E80(&v219, v158);
              v159 = *(_BYTE **)(v158 + 32);
              if ( (unsigned __int64)v159 >= *(_QWORD *)(v158 + 24) )
              {
                a2 = 10;
                sub_CB5D20(v158, 10);
              }
              else
              {
                v23 = (__int64)(v159 + 1);
                *(_QWORD *)(v158 + 32) = v159 + 1;
                *v159 = 10;
              }
              v160 = *a1;
              *((_BYTE *)a1 + 152) = 1;
              if ( !v160 )
                goto LABEL_37;
            }
            goto LABEL_511;
          default:
LABEL_517:
            BUG();
        }
      }
    }
LABEL_39:
    sub_C058E0((__int64)a1, a2, v23, v24);
    if ( !*((_DWORD *)a1 + 556) )
      sub_E36D60(a1 + 259, a1 + 20);
    ++a1[36];
    if ( !*((_BYTE *)a1 + 316) )
    {
      v53 = 4 * (*((_DWORD *)a1 + 77) - *((_DWORD *)a1 + 78));
      v54 = *((unsigned int *)a1 + 76);
      if ( v53 < 0x20 )
        v53 = 32;
      if ( (unsigned int)v54 > v53 )
      {
        sub_C8C990(a1 + 36);
LABEL_43:
        v37 = *((_DWORD *)a1 + 220);
        ++a1[108];
        *((_DWORD *)a1 + 466) = 0;
        a1[102] = 0;
        *((_BYTE *)a1 + 824) = 0;
        if ( !v37 )
        {
          if ( *((_DWORD *)a1 + 221) )
          {
            v38 = *((unsigned int *)a1 + 222);
            if ( (unsigned int)v38 <= 0x40 )
            {
LABEL_46:
              v39 = (_QWORD *)a1[109];
              for ( i = &v39[2 * v38]; i != v39; v39 += 2 )
                *v39 = -4096;
              a1[110] = 0;
              goto LABEL_49;
            }
            sub_C7D6A0(a1[109], 16LL * *((unsigned int *)a1 + 222), 8);
            a1[109] = 0;
            a1[110] = 0;
            *((_DWORD *)a1 + 222) = 0;
          }
LABEL_49:
          v41 = *((unsigned int *)a1 + 570);
          *((_DWORD *)a1 + 226) = 0;
          if ( (_DWORD)v41 )
          {
            v42 = a1[284];
            v43 = v42 + 8 * v41;
            while ( 1 )
            {
              v44 = *(_BYTE **)v42;
              v45 = *(_QWORD *)(*(_QWORD *)v42 - 32LL * (*(_DWORD *)(*(_QWORD *)v42 + 4LL) & 0x7FFFFFF));
              if ( *(_BYTE *)v45 != 24 )
                break;
              v46 = *(const char **)(v45 + 24);
              if ( (unsigned __int8)(*v46 - 5) > 0x1Fu )
              {
                v223 = 1;
                v47 = "!id.scope.list must point to an MDNode";
                goto LABEL_54;
              }
              if ( (*(v46 - 16) & 2) != 0 )
                v110 = *((_DWORD *)v46 - 6);
              else
                v110 = (*((_WORD *)v46 - 8) >> 6) & 0xF;
              if ( v110 != 1 )
              {
                v205 = *a1;
                v223 = 1;
                v219.m128i_i64[0] = (__int64)"!id.scope.list must point to a list with a single scope";
                v222 = 3;
                if ( v205 )
                {
                  sub_CA0E80(&v219, v205);
                  v206 = *(_BYTE **)(v205 + 32);
                  if ( (unsigned __int64)v206 >= *(_QWORD *)(v205 + 24) )
                  {
                    sub_CB5D20(v205, 10);
                  }
                  else
                  {
                    *(_QWORD *)(v205 + 32) = v206 + 1;
                    *v206 = 10;
                  }
                }
                v80 = *a1 == 0;
                *((_BYTE *)a1 + 152) = 1;
                if ( !v80 )
                  sub_BDBD80((__int64)a1, v44);
                goto LABEL_234;
              }
              v42 += 8;
              sub_BECBF0(a1, v46);
              if ( v43 == v42 )
              {
                if ( (_BYTE)qword_4F83768 )
                {
                  v111 = *((unsigned int *)a1 + 570);
                  v112 = (char *)a1[284];
                  v113 = 8 * v111;
                  v114 = &v112[8 * v111];
                  if ( v112 != v114 )
                  {
                    _BitScanReverse64(&v115, v113 >> 3);
                    sub_BDA2E0((char *)a1[284], (__int64 *)&v112[8 * v111], 2LL * (int)(63 - (v115 ^ 0x3F)));
                    if ( (unsigned __int64)v113 <= 0x80 )
                    {
                      sub_BD9E50(v112, v114);
                    }
                    else
                    {
                      v116 = (__int64 *)(v112 + 128);
                      sub_BD9E50(v112, v112 + 128);
                      if ( v114 != v112 + 128 )
                      {
                        do
                        {
                          v117 = v116++;
                          sub_BD9DB0(v117);
                        }
                        while ( v114 != (char *)v116 );
                      }
                    }
                    v112 = (char *)a1[284];
                    v111 = *((unsigned int *)a1 + 570);
                  }
                  v118 = &v112[8 * v111];
                  v119 = (__int64 *)v112;
                  if ( v118 != v112 )
                  {
                    do
                    {
                      v213 = v111;
                      v120 = v119;
                      v121 = sub_A17150((_BYTE *)(*(_QWORD *)(*(_QWORD *)(*v119
                                                                        - 32LL * (*(_DWORD *)(*v119 + 4) & 0x7FFFFFF))
                                                            + 24LL)
                                                - 16LL));
                      v111 = v213;
                      v122 = v121;
                      do
                      {
                        if ( v118 == (char *)++v120 )
                          break;
                        v124 = *(_QWORD *)(*(_QWORD *)(*v120 - 32LL * (*(_DWORD *)(*v120 + 4) & 0x7FFFFFF)) + 24LL);
                        v125 = *(_BYTE *)(v124 - 16);
                        v123 = (v125 & 2) != 0
                             ? *(_BYTE **)(v124 - 32)
                             : (_BYTE *)(-16 - 8LL * ((v125 >> 2) & 0xF) + v124);
                      }
                      while ( v123 == v122 );
                      if ( (char *)v120 - (char *)v119 <= 248 )
                      {
                        if ( v120 != v119 )
                        {
                          v201 = v119;
                          while ( 2 )
                          {
                            v202 = *v201;
                            v203 = v119;
                            do
                            {
                              if ( v202 != *v203 )
                              {
                                v214 = (_BYTE *)v202;
                                v204 = sub_B19DB0((__int64)(a1 + 20), v202, *v203);
                                v202 = (__int64)v214;
                                if ( v204 )
                                {
                                  v105 = *a1;
                                  v223 = 1;
                                  v219.m128i_i64[0] = (__int64)"llvm.experimental.noalias.scope.decl dominates another on"
                                                               "e with the same scope";
                                  v222 = 3;
                                  if ( v105 )
                                  {
                                    sub_CA0E80(&v219, v105);
                                    v106 = *(_BYTE **)(v105 + 32);
                                    if ( (unsigned __int64)v106 >= *(_QWORD *)(v105 + 24) )
                                    {
                                      sub_CB5D20(v105, 10);
                                    }
                                    else
                                    {
                                      *(_QWORD *)(v105 + 32) = v106 + 1;
                                      *v106 = 10;
                                    }
                                  }
                                  v80 = *a1 == 0;
                                  *((_BYTE *)a1 + 152) = 1;
                                  if ( !v80 && v214 )
                                    sub_BDBD80((__int64)a1, v214);
                                  goto LABEL_234;
                                }
                              }
                              ++v203;
                            }
                            while ( v120 != v203 );
                            if ( v120 != ++v201 )
                              continue;
                            break;
                          }
                          v112 = (char *)a1[284];
                          v111 = *((unsigned int *)a1 + 570);
                          v119 = v120;
                        }
                      }
                      else
                      {
                        v119 = v120;
                      }
                      v118 = &v112[8 * v111];
                    }
                    while ( v119 != (__int64 *)v118 );
                  }
                }
                goto LABEL_234;
              }
            }
            v223 = 1;
            v47 = "llvm.experimental.noalias.scope.decl must have a MetadataAsValue argument";
LABEL_54:
            v48 = *a1;
            v219.m128i_i64[0] = (__int64)v47;
            v222 = 3;
            if ( v48 )
            {
              sub_CA0E80(&v219, v48);
              v49 = *(_BYTE **)(v48 + 32);
              if ( (unsigned __int64)v49 >= *(_QWORD *)(v48 + 24) )
              {
                sub_CB5D20(v48, 10);
              }
              else
              {
                *(_QWORD *)(v48 + 32) = v49 + 1;
                *v49 = 10;
              }
              result = *a1;
              *((_BYTE *)a1 + 152) = 1;
              if ( result )
              {
                sub_BDBD80((__int64)a1, v44);
                result = *((unsigned __int8 *)a1 + 152) ^ 1u;
              }
            }
            else
            {
              *((_BYTE *)a1 + 152) = 1;
              result = 0;
            }
            goto LABEL_59;
          }
LABEL_234:
          result = *((unsigned __int8 *)a1 + 152) ^ 1u;
LABEL_59:
          *((_DWORD *)a1 + 570) = 0;
          return result;
        }
        v100 = 4 * v37;
        v38 = *((unsigned int *)a1 + 222);
        if ( (unsigned int)(4 * v37) < 0x40 )
          v100 = 64;
        if ( v100 < (unsigned int)v38 )
        {
          v101 = v37 - 1;
          if ( v101 )
          {
            _BitScanReverse(&v101, v101);
            v102 = (_QWORD *)a1[109];
            v103 = 1 << (33 - (v101 ^ 0x1F));
            if ( v103 < 64 )
              v103 = 64;
            if ( v103 == (_DWORD)v38 )
            {
              a1[110] = 0;
              v104 = &v102[2 * (unsigned int)v103];
              do
              {
                if ( v102 )
                  *v102 = -4096;
                v102 += 2;
              }
              while ( v104 != v102 );
              goto LABEL_49;
            }
          }
          else
          {
            v102 = (_QWORD *)a1[109];
            v103 = 64;
          }
          sub_C7D6A0(v102, 16LL * *((unsigned int *)a1 + 222), 8);
          v191 = ((((((((4 * v103 / 3u + 1) | ((unsigned __int64)(4 * v103 / 3u + 1) >> 1)) >> 2)
                    | (4 * v103 / 3u + 1)
                    | ((unsigned __int64)(4 * v103 / 3u + 1) >> 1)) >> 4)
                  | (((4 * v103 / 3u + 1) | ((unsigned __int64)(4 * v103 / 3u + 1) >> 1)) >> 2)
                  | (4 * v103 / 3u + 1)
                  | ((unsigned __int64)(4 * v103 / 3u + 1) >> 1)) >> 8)
                | (((((4 * v103 / 3u + 1) | ((unsigned __int64)(4 * v103 / 3u + 1) >> 1)) >> 2)
                  | (4 * v103 / 3u + 1)
                  | ((unsigned __int64)(4 * v103 / 3u + 1) >> 1)) >> 4)
                | (((4 * v103 / 3u + 1) | ((unsigned __int64)(4 * v103 / 3u + 1) >> 1)) >> 2)
                | (4 * v103 / 3u + 1)
                | ((unsigned __int64)(4 * v103 / 3u + 1) >> 1)) >> 16;
          v192 = (v191
                | (((((((4 * v103 / 3u + 1) | ((unsigned __int64)(4 * v103 / 3u + 1) >> 1)) >> 2)
                    | (4 * v103 / 3u + 1)
                    | ((unsigned __int64)(4 * v103 / 3u + 1) >> 1)) >> 4)
                  | (((4 * v103 / 3u + 1) | ((unsigned __int64)(4 * v103 / 3u + 1) >> 1)) >> 2)
                  | (4 * v103 / 3u + 1)
                  | ((unsigned __int64)(4 * v103 / 3u + 1) >> 1)) >> 8)
                | (((((4 * v103 / 3u + 1) | ((unsigned __int64)(4 * v103 / 3u + 1) >> 1)) >> 2)
                  | (4 * v103 / 3u + 1)
                  | ((unsigned __int64)(4 * v103 / 3u + 1) >> 1)) >> 4)
                | (((4 * v103 / 3u + 1) | ((unsigned __int64)(4 * v103 / 3u + 1) >> 1)) >> 2)
                | (4 * v103 / 3u + 1)
                | ((unsigned __int64)(4 * v103 / 3u + 1) >> 1))
               + 1;
          *((_DWORD *)a1 + 222) = v192;
          v193 = (_QWORD *)sub_C7D670(16 * v192, 8);
          v194 = *((unsigned int *)a1 + 222);
          a1[110] = 0;
          a1[109] = v193;
          for ( j = &v193[2 * v194]; j != v193; v193 += 2 )
          {
            if ( v193 )
              *v193 = -4096;
          }
          goto LABEL_49;
        }
        goto LABEL_46;
      }
      memset((void *)a1[37], -1, 8 * v54);
    }
    *(_QWORD *)((char *)a1 + 308) = 0;
    goto LABEL_43;
  }
  while ( 1 )
  {
    if ( !v4 )
      BUG();
    v5 = *(_QWORD *)(v4 + 24) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v5 == v4 + 24 )
      break;
    if ( !v5 )
      BUG();
    if ( (unsigned int)*(unsigned __int8 *)(v5 - 24) - 30 > 0xA )
      break;
    v4 = *(_QWORD *)(v4 + 8);
    if ( v212 == v4 )
      goto LABEL_20;
  }
  v6 = *a1;
  result = 0;
  if ( *a1 )
  {
    v8 = *(__m128i **)(v6 + 32);
    if ( *(_QWORD *)(v6 + 24) - (_QWORD)v8 <= 0x18u )
    {
      v6 = sub_CB6200(*a1, "Basic Block in function '", 25);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F64280);
      v8[1].m128i_i8[8] = 39;
      v8[1].m128i_i64[0] = 0x206E6F6974636E75LL;
      *v8 = si128;
      *(_QWORD *)(v6 + 32) += 25LL;
    }
    v10 = sub_BD5D20(a2);
    v12 = *(__m128i **)(v6 + 32);
    v13 = v10;
    v14 = *(_QWORD *)(v6 + 24) - (_QWORD)v12;
    if ( v14 < v11 )
    {
      v51 = sub_CB6200(v6, v13, v11);
      v12 = *(__m128i **)(v51 + 32);
      v6 = v51;
      v14 = *(_QWORD *)(v51 + 24) - (_QWORD)v12;
    }
    else if ( v11 )
    {
      v216 = v11;
      memcpy(v12, v13, v11);
      v52 = *(_QWORD *)(v6 + 24);
      v12 = (__m128i *)(v216 + *(_QWORD *)(v6 + 32));
      *(_QWORD *)(v6 + 32) = v12;
      v14 = v52 - (_QWORD)v12;
    }
    if ( v14 <= 0x1B )
    {
      sub_CB6200(v6, "' does not have terminator!\n", 28);
    }
    else
    {
      v15 = _mm_load_si128((const __m128i *)&xmmword_3F64290);
      qmemcpy(&v12[1], "terminator!\n", 12);
      *v12 = v15;
      *(_QWORD *)(v6 + 32) += 28LL;
    }
    sub_A5C020((_BYTE *)(v4 - 24), *a1, 1, (__int64)(a1 + 2));
    v16 = *a1;
    v17 = *(_BYTE **)(*a1 + 32LL);
    if ( *(_BYTE **)(*a1 + 24LL) == v17 )
    {
      sub_CB6200(v16, "\n", 1);
      return 0;
    }
    else
    {
      *v17 = 10;
      result = 0;
      ++*(_QWORD *)(v16 + 32);
    }
  }
  return result;
}
