// Function: sub_166A310
// Address: 0x166a310
//
__int64 __fastcall sub_166A310(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rbx
  unsigned __int64 v4; // rax
  __int64 v5; // r13
  __int64 result; // rax
  __m128i *v7; // rdx
  __m128i si128; // xmm0
  const char *v9; // rax
  size_t v10; // rdx
  __m128i *v11; // rdi
  const char *v12; // rsi
  unsigned __int64 v13; // rax
  __m128i v14; // xmm0
  __int64 v15; // rdi
  _BYTE *v16; // rax
  __int64 v17; // rbx
  __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rbx
  __int64 v21; // r12
  __int64 v22; // r13
  unsigned int v23; // r9d
  __int64 v24; // r10
  __int64 v25; // rax
  __int64 v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // r14
  _BYTE *v29; // rax
  __int64 v30; // rsi
  __int64 v31; // rdi
  _BYTE *v32; // rax
  void *v33; // rdi
  unsigned int v34; // eax
  __int64 v35; // rdx
  int v36; // eax
  __int64 v37; // rdx
  _QWORD *v38; // rax
  _QWORD *j; // rdx
  __int64 v40; // rax
  _BYTE *v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rdi
  __int64 v45; // r14
  __int64 v46; // r12
  _BYTE *v47; // rax
  __int64 v48; // rsi
  _BYTE *v49; // rax
  __int64 v50; // rdi
  char v51; // al
  __int64 v52; // rax
  __int64 v53; // r14
  unsigned __int64 v54; // rax
  __int64 v55; // rdx
  const char *v56; // rax
  __int64 v57; // r12
  _BYTE *v58; // rax
  __int64 v59; // rax
  unsigned __int64 v60; // rdi
  __int64 v61; // rdi
  __int64 *v62; // rdx
  __int64 v63; // r14
  char v64; // al
  __int64 v65; // rdi
  __int64 v66; // rdi
  int v67; // edx
  unsigned __int64 v68; // rax
  __int64 v69; // rcx
  __int64 v70; // rax
  __int64 v71; // r12
  _BYTE *v72; // rax
  __int64 v73; // rax
  __int64 v74; // rax
  _BYTE *v75; // rdx
  unsigned __int16 v76; // ax
  int v77; // edx
  int v78; // eax
  __int64 v79; // rax
  __int64 v80; // r14
  __int64 v81; // rdx
  __int64 v82; // rax
  __int64 v83; // r14
  __int64 v84; // r12
  __int64 *v85; // rax
  __int64 v86; // rax
  __int64 v87; // rdi
  unsigned int v88; // eax
  __int64 v89; // rdx
  __int64 v90; // rax
  _BYTE *v91; // rsi
  char v92; // di
  unsigned int v93; // ecx
  _QWORD *v94; // rdi
  unsigned int v95; // eax
  int v96; // ebx
  unsigned __int64 v97; // rdx
  unsigned __int64 v98; // rax
  _QWORD *v99; // rax
  __int64 v100; // rdx
  _QWORD *i; // rdx
  __int64 v102; // rdi
  unsigned __int64 v103; // rax
  const char *v104; // rax
  __int64 v105; // rdx
  __int16 v106; // cx
  __int64 v107; // r12
  _BYTE *v108; // rax
  __int64 v109; // rax
  __int64 v110; // r12
  _BYTE *v111; // rax
  __int64 v112; // rax
  __int64 v113; // r12
  _BYTE *v114; // rax
  __int64 v115; // rax
  __int64 v116; // r12
  _BYTE *v117; // rax
  __int64 v118; // rax
  __int64 v119; // r12
  _BYTE *v120; // rax
  __int64 v121; // rax
  __int64 v122; // r12
  _BYTE *v123; // rax
  __int64 v124; // rax
  __int64 v125; // r12
  _BYTE *v126; // rax
  __int64 v127; // rax
  __int64 v128; // r12
  _BYTE *v129; // rax
  __int64 v130; // rax
  __int64 v131; // r12
  _BYTE *v132; // rax
  __int64 v133; // rax
  __int64 v134; // rdx
  __int64 v135; // r12
  _BYTE *v136; // rax
  __int64 v137; // rax
  __int64 v138; // r12
  _BYTE *v139; // rax
  __int64 v140; // rax
  __int64 v141; // r12
  _BYTE *v142; // rax
  __int64 v143; // rax
  __int64 v144; // r12
  _BYTE *v145; // rax
  __int64 v146; // rax
  __int64 v147; // r12
  _BYTE *v148; // rax
  __int64 v149; // rax
  __int64 v150; // r12
  _BYTE *v151; // rax
  __int64 v152; // rax
  _BYTE *v153; // rdx
  __int64 v154; // r12
  _BYTE *v155; // rax
  __int64 v156; // rax
  __int64 v157; // rdi
  int v158; // eax
  int v159; // eax
  int v160; // edx
  __int64 v161; // rdx
  __int64 v162; // rax
  __int64 *v163; // rdx
  __int64 *v164; // rcx
  __int64 *v165; // rax
  __int64 *v166; // r14
  __int64 v167; // r13
  __int64 *v168; // rbx
  __int64 v169; // r12
  _BYTE *v170; // rax
  __int64 v171; // rax
  __int64 v172; // rax
  _BYTE *v173; // rdx
  _BYTE *v174; // rax
  _BYTE *v175; // rdx
  __int64 v176; // r12
  _BYTE *v177; // rax
  bool v178; // zf
  _QWORD *v179; // rax
  char v180; // [rsp+0h] [rbp-C0h]
  __int64 v181; // [rsp+0h] [rbp-C0h]
  __int64 v182; // [rsp+8h] [rbp-B8h]
  __int64 v183; // [rsp+10h] [rbp-B0h]
  __int64 v184; // [rsp+18h] [rbp-A8h]
  size_t v185; // [rsp+18h] [rbp-A8h]
  __int64 v186[2]; // [rsp+20h] [rbp-A0h] BYREF
  char v187; // [rsp+30h] [rbp-90h]
  char v188; // [rsp+31h] [rbp-8Fh]
  const char *v189; // [rsp+40h] [rbp-80h] BYREF
  _BYTE *v190; // [rsp+48h] [rbp-78h]
  _BYTE *v191; // [rsp+50h] [rbp-70h]
  __int64 v192; // [rsp+58h] [rbp-68h]
  int v193; // [rsp+60h] [rbp-60h]
  _BYTE v194[88]; // [rsp+68h] [rbp-58h] BYREF

  v183 = a2 + 72;
  if ( a2 + 72 != (*(_QWORD *)(a2 + 72) & 0xFFFFFFFFFFFFFFF8LL) )
  {
    a1[18] = a2;
    sub_15D3930((__int64)(a1 + 10));
  }
  v3 = *(_QWORD *)(a2 + 80);
  if ( v183 == v3 )
  {
LABEL_20:
    *((_BYTE *)a1 + 72) = 0;
    sub_165E480(a1, a2);
    v182 = *(_QWORD *)(a2 + 80);
    if ( v183 != v182 )
    {
      while ( 1 )
      {
        v17 = v182;
        v18 = v182 - 24;
        v182 = *(_QWORD *)(v182 + 8);
        sub_1651200(a1, v18);
        v19 = v17 + 16;
        v20 = *(_QWORD *)(v17 + 24);
        v184 = v19;
        if ( v19 != v20 )
          break;
LABEL_36:
        if ( v183 == v182 )
          goto LABEL_37;
      }
      while ( 1 )
      {
        v21 = v20;
        v20 = *(_QWORD *)(v20 + 8);
        v22 = v21 - 24;
        v23 = *(_DWORD *)(v21 - 4) & 0xFFFFFFF;
        if ( !v23 )
          break;
        v24 = v23;
        v25 = 0;
        v26 = 24LL * v23;
        while ( 1 )
        {
          v27 = v22 - v26;
          if ( (*(_BYTE *)(v21 - 1) & 0x40) != 0 )
            v27 = *(_QWORD *)(v21 - 32);
          if ( !*(_QWORD *)(v27 + v25) )
            break;
          v25 += 24;
          if ( v26 == v25 )
          {
            switch ( *(_BYTE *)(v21 - 8) )
            {
              case 0x18:
              case 0x58:
                if ( !v23 )
                  goto LABEL_177;
                goto LABEL_178;
              case 0x19:
                goto LABEL_173;
              case 0x1A:
                if ( v23 != 3 || sub_1642F90(**(_QWORD **)(v21 - 96), 1) )
                  goto LABEL_201;
                v84 = *(_QWORD *)(v21 - 96);
                v189 = "Branch condition is not 'i1' type!";
                LOWORD(v191) = 259;
                sub_164FF40(a1, (__int64)&v189);
                if ( *a1 )
                  goto LABEL_382;
                goto LABEL_35;
              case 0x1B:
                goto LABEL_199;
              case 0x1C:
                goto LABEL_198;
              case 0x1D:
                goto LABEL_164;
              case 0x1E:
                goto LABEL_160;
              case 0x1F:
                goto LABEL_201;
              case 0x20:
                goto LABEL_166;
              case 0x21:
                goto LABEL_169;
              case 0x22:
                goto LABEL_121;
              case 0x23:
              case 0x24:
              case 0x25:
              case 0x26:
              case 0x27:
              case 0x28:
              case 0x29:
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
                goto LABEL_54;
              case 0x35:
                goto LABEL_112;
              case 0x36:
                goto LABEL_111;
              case 0x37:
                goto LABEL_102;
              case 0x38:
                goto LABEL_101;
              case 0x39:
                goto LABEL_192;
              case 0x3A:
                goto LABEL_148;
              case 0x3B:
                goto LABEL_138;
              case 0x3C:
                goto LABEL_137;
              case 0x3D:
                goto LABEL_136;
              case 0x3E:
                goto LABEL_135;
              case 0x3F:
                goto LABEL_134;
              case 0x40:
                goto LABEL_133;
              case 0x41:
                goto LABEL_196;
              case 0x42:
                goto LABEL_195;
              case 0x43:
                goto LABEL_197;
              case 0x44:
                goto LABEL_172;
              case 0x45:
                goto LABEL_194;
              case 0x46:
                goto LABEL_100;
              case 0x47:
                goto LABEL_98;
              case 0x48:
                goto LABEL_97;
              case 0x49:
                goto LABEL_92;
              case 0x4A:
                goto LABEL_83;
              case 0x4B:
                goto LABEL_82;
              case 0x4C:
                goto LABEL_81;
              case 0x4D:
                goto LABEL_80;
              case 0x4E:
                goto LABEL_79;
              case 0x4F:
                goto LABEL_78;
              case 0x50:
              case 0x51:
                goto LABEL_77;
              case 0x52:
                goto LABEL_305;
              case 0x53:
                goto LABEL_75;
              case 0x54:
                goto LABEL_73;
              case 0x55:
                goto LABEL_71;
              case 0x56:
                goto LABEL_68;
              case 0x57:
                goto LABEL_65;
            }
          }
        }
        v28 = *a1;
        v189 = "Operand is null";
        LOWORD(v191) = 259;
        if ( v28 )
        {
          sub_16E2CE0(&v189, v28);
          v29 = *(_BYTE **)(v28 + 24);
          if ( (unsigned __int64)v29 >= *(_QWORD *)(v28 + 16) )
          {
            sub_16E7DE0(v28, 10);
          }
          else
          {
            *(_QWORD *)(v28 + 24) = v29 + 1;
            *v29 = 10;
          }
          v30 = *a1;
          *((_BYTE *)a1 + 72) = 1;
          if ( !v30 )
            goto LABEL_35;
          if ( *(_BYTE *)(v21 - 8) <= 0x17u )
          {
            sub_1553920((__int64 *)(v21 - 24), v30, 1, (__int64)(a1 + 2));
            v31 = *a1;
            v41 = *(_BYTE **)(*a1 + 24LL);
            if ( (unsigned __int64)v41 >= *(_QWORD *)(*a1 + 16LL) )
            {
LABEL_469:
              sub_16E7DE0(v31, 10);
              goto LABEL_35;
            }
            *(_QWORD *)(v31 + 24) = v41 + 1;
            *v41 = 10;
          }
          else
          {
            sub_155BD40(v21 - 24, v30, (__int64)(a1 + 2), 0);
            v31 = *a1;
            v32 = *(_BYTE **)(*a1 + 24LL);
            if ( (unsigned __int64)v32 >= *(_QWORD *)(*a1 + 16LL) )
              goto LABEL_469;
            *(_QWORD *)(v31 + 24) = v32 + 1;
            *v32 = 10;
          }
        }
        else
        {
          *((_BYTE *)a1 + 72) = 1;
        }
LABEL_35:
        if ( v184 == v20 )
          goto LABEL_36;
      }
      switch ( *(_BYTE *)(v21 - 8) )
      {
        case 0x19:
LABEL_173:
          v85 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v21 + 16) + 56LL) + 24LL) + 16LL);
          v80 = *v85;
          if ( *(_BYTE *)(*v85 + 8) )
          {
            if ( v23 == 1 && v80 == **(_QWORD **)(v21 - 48) )
              goto LABEL_201;
            v189 = "Function return type does not match operand type of return inst!";
            LOWORD(v191) = 259;
            sub_164FF40(a1, (__int64)&v189);
            if ( !*a1 )
              goto LABEL_35;
          }
          else
          {
            if ( !v23 )
              goto LABEL_201;
            v189 = "Found return instr that returns non-void in Function of void return type!";
            LOWORD(v191) = 259;
            sub_164FF40(a1, (__int64)&v189);
            if ( !*a1 )
              goto LABEL_35;
          }
          goto LABEL_445;
        case 0x1A:
        case 0x1F:
          goto LABEL_201;
        case 0x1B:
LABEL_199:
          sub_16658F0(a1, v21 - 24);
          goto LABEL_35;
        case 0x1C:
LABEL_198:
          sub_1665C90(a1, v21 - 24);
          goto LABEL_35;
        case 0x1D:
LABEL_164:
          sub_1668E00((__int64)a1, v21 - 24);
          goto LABEL_35;
        case 0x1E:
LABEL_160:
          if ( (*(_BYTE *)(sub_15F2060(v21 - 24) + 18) & 8) != 0 )
          {
            v81 = a1[89];
            v82 = **(_QWORD **)(v21 - 48);
            if ( !v81 )
            {
              a1[89] = v82;
LABEL_201:
              sub_1665790(a1, v21 - 24);
              goto LABEL_35;
            }
            if ( v81 == v82 )
              goto LABEL_201;
            v147 = *a1;
            v189 = "The resume instruction should have a consistent result type inside a function.";
            LOWORD(v191) = 259;
            if ( v147 )
            {
              sub_16E2CE0(&v189, v147);
              v148 = *(_BYTE **)(v147 + 24);
              if ( (unsigned __int64)v148 >= *(_QWORD *)(v147 + 16) )
              {
                sub_16E7DE0(v147, 10);
              }
              else
              {
                *(_QWORD *)(v147 + 24) = v148 + 1;
                *v148 = 10;
              }
              v149 = *a1;
              *((_BYTE *)a1 + 72) = 1;
              if ( !v149 )
                goto LABEL_35;
              goto LABEL_465;
            }
            *((_BYTE *)a1 + 72) = 1;
          }
          else
          {
            v154 = *a1;
            v189 = "ResumeInst needs to be in a function with a personality.";
            LOWORD(v191) = 259;
            if ( v154 )
            {
              sub_16E2CE0(&v189, v154);
              v155 = *(_BYTE **)(v154 + 24);
              if ( (unsigned __int64)v155 >= *(_QWORD *)(v154 + 16) )
              {
                sub_16E7DE0(v154, 10);
              }
              else
              {
                *(_QWORD *)(v154 + 24) = v155 + 1;
                *v155 = 10;
              }
              v156 = *a1;
              *((_BYTE *)a1 + 72) = 1;
              if ( !v156 )
                goto LABEL_35;
              goto LABEL_465;
            }
            *((_BYTE *)a1 + 72) = 1;
          }
          goto LABEL_35;
        case 0x20:
          v24 = 0;
LABEL_166:
          v83 = *(_QWORD *)(v22 - 24 * v24);
          if ( *(_BYTE *)(v83 + 16) != 73 )
          {
            v189 = "CleanupReturnInst needs to be provided a CleanupPad";
            LOWORD(v191) = 259;
            sub_164FF40(a1, (__int64)&v189);
            if ( *a1 )
            {
              sub_164FA80(a1, v21 - 24);
              sub_164FA80(a1, v83);
            }
            goto LABEL_35;
          }
          if ( (*(_BYTE *)(v21 - 6) & 1) == 0
            || (v157 = *(_QWORD *)(v22 + 24 * (1 - v24))) == 0
            || (v158 = *(unsigned __int8 *)(sub_157ED20(v157) + 16), (unsigned int)(v158 - 34) <= 0x36)
            && ((1LL << ((unsigned __int8)v158 - 34)) & 0x40018000000001LL) != 0
            && (_BYTE)v158 != 88 )
          {
LABEL_358:
            sub_1665790(a1, v22);
            goto LABEL_35;
          }
          v189 = "CleanupReturnInst must unwind to an EH block which is not a landingpad.";
          LOWORD(v191) = 259;
          sub_164FF40(a1, (__int64)&v189);
          if ( !*a1 )
            goto LABEL_35;
          goto LABEL_465;
        case 0x21:
LABEL_169:
          v84 = *(_QWORD *)(v21 - 72);
          if ( *(_BYTE *)(v84 + 16) == 74 )
            goto LABEL_358;
          v189 = "CatchReturnInst needs to be provided a CatchPad";
          LOWORD(v191) = 259;
          sub_164FF40(a1, (__int64)&v189);
          if ( *a1 )
            goto LABEL_382;
          goto LABEL_35;
        case 0x22:
LABEL_121:
          v61 = *(_QWORD *)(v21 + 16);
          if ( (*(_BYTE *)(*(_QWORD *)(v61 + 56) + 18LL) & 8) == 0 )
          {
            v116 = *a1;
            v189 = "CatchSwitchInst needs to be in a function with a personality.";
            LOWORD(v191) = 259;
            if ( !v116 )
            {
              *((_BYTE *)a1 + 72) = 1;
              goto LABEL_35;
            }
            sub_16E2CE0(&v189, v116);
            v117 = *(_BYTE **)(v116 + 24);
            if ( (unsigned __int64)v117 >= *(_QWORD *)(v116 + 16) )
            {
              sub_16E7DE0(v116, 10);
            }
            else
            {
              *(_QWORD *)(v116 + 24) = v117 + 1;
              *v117 = 10;
            }
            v118 = *a1;
            *((_BYTE *)a1 + 72) = 1;
            if ( !v118 )
              goto LABEL_35;
            goto LABEL_465;
          }
          if ( v22 != sub_157ED20(v61) )
          {
            v125 = *a1;
            v189 = "CatchSwitchInst not the first non-PHI instruction in the block.";
            LOWORD(v191) = 259;
            if ( !v125 )
            {
              *((_BYTE *)a1 + 72) = 1;
              goto LABEL_35;
            }
            sub_16E2CE0(&v189, v125);
            v126 = *(_BYTE **)(v125 + 24);
            if ( (unsigned __int64)v126 >= *(_QWORD *)(v125 + 16) )
            {
              sub_16E7DE0(v125, 10);
            }
            else
            {
              *(_QWORD *)(v125 + 24) = v126 + 1;
              *v126 = 10;
            }
            v127 = *a1;
            *((_BYTE *)a1 + 72) = 1;
            if ( !v127 )
              goto LABEL_35;
            goto LABEL_465;
          }
          if ( (*(_BYTE *)(v21 - 1) & 0x40) != 0 )
            v62 = *(__int64 **)(v21 - 32);
          else
            v62 = (__int64 *)(v22 - 24LL * (*(_DWORD *)(v21 - 4) & 0xFFFFFFF));
          v63 = *v62;
          v64 = *(_BYTE *)(*v62 + 16);
          if ( v64 != 16 && (unsigned __int8)(v64 - 73) > 1u )
          {
            v119 = *a1;
            v189 = "CatchSwitchInst has an invalid parent.";
            LOWORD(v191) = 259;
            if ( v119 )
            {
              sub_16E2CE0(&v189, v119);
              v120 = *(_BYTE **)(v119 + 24);
              if ( (unsigned __int64)v120 >= *(_QWORD *)(v119 + 16) )
              {
                sub_16E7DE0(v119, 10);
              }
              else
              {
                *(_QWORD *)(v119 + 24) = v120 + 1;
                *v120 = 10;
              }
              v121 = *a1;
              *((_BYTE *)a1 + 72) = 1;
              if ( v121 )
                sub_164FA80(a1, v63);
            }
            else
            {
              *((_BYTE *)a1 + 72) = 1;
            }
            goto LABEL_35;
          }
          v180 = *(_BYTE *)(v21 - 6) & 1;
          if ( !v180 )
            goto LABEL_383;
          v65 = v62[3];
          if ( !v65 )
            goto LABEL_371;
          v66 = sub_157ED20(v65);
          v67 = *(unsigned __int8 *)(v66 + 16);
          v68 = (unsigned int)(v67 - 34);
          if ( (unsigned int)v68 > 0x36 || (v69 = 0x40018000000001LL, !_bittest64(&v69, v68)) || (_BYTE)v67 == 88 )
          {
            v189 = "CatchSwitchInst must unwind to an EH block which is not a landingpad.";
            LOWORD(v191) = 259;
            sub_164FF40(a1, (__int64)&v189);
            if ( !*a1 )
              goto LABEL_35;
            goto LABEL_473;
          }
          if ( v63 == sub_164ED90(v66) )
          {
            v189 = (const char *)(v21 - 24);
            *(_QWORD *)sub_1668F90((__int64)(a1 + 95), (unsigned __int64 *)&v189) = v22;
          }
          if ( (*(_BYTE *)(v21 - 6) & 1) != 0 )
          {
LABEL_371:
            v159 = *(_DWORD *)(v21 - 4);
            v160 = (v159 & 0xFFFFFFF) - 2;
          }
          else
          {
LABEL_383:
            v159 = *(_DWORD *)(v21 - 4);
            v180 = 0;
            v160 = (v159 & 0xFFFFFFF) - 1;
          }
          if ( v160 )
          {
            v161 = 24LL * (v159 & 0xFFFFFFF);
            if ( (*(_BYTE *)(v21 - 1) & 0x40) != 0 )
            {
              v162 = *(_QWORD *)(v21 - 32);
              v163 = (__int64 *)(v162 + v161);
            }
            else
            {
              v162 = v22 - v161;
              v163 = (__int64 *)(v21 - 24);
            }
            v164 = (__int64 *)(v162 + 24);
            v165 = (__int64 *)(v162 + 48);
            if ( v180 )
              v164 = v165;
            v166 = v164;
            if ( v163 == v164 )
              goto LABEL_451;
            v181 = v21 - 24;
            v167 = v20;
            v168 = v163;
            while ( 1 )
            {
              v84 = sub_15A5110(*v166);
              if ( *(_BYTE *)(sub_157ED20(v84) + 16) != 74 )
                break;
              v166 += 3;
              if ( v168 == v166 )
              {
                v20 = v167;
                v22 = v181;
LABEL_451:
                sub_1654C20(a1, v22);
                sub_1665790(a1, v22);
                goto LABEL_35;
              }
            }
            v20 = v167;
            LOWORD(v191) = 259;
            v22 = v181;
            v189 = "CatchSwitchInst handlers must be catchpads";
            sub_164FF40(a1, (__int64)&v189);
            if ( *a1 )
            {
LABEL_382:
              sub_164FA80(a1, v22);
              sub_164FA80(a1, v84);
            }
            goto LABEL_35;
          }
          v189 = "CatchSwitchInst cannot have empty handler list";
          LOWORD(v191) = 259;
          sub_164FF40(a1, (__int64)&v189);
          if ( !*a1 )
            goto LABEL_35;
          goto LABEL_473;
        case 0x23:
        case 0x24:
        case 0x25:
        case 0x26:
        case 0x27:
        case 0x28:
        case 0x29:
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
LABEL_54:
          sub_1665DD0((__int64)a1, v21 - 24);
          goto LABEL_35;
        case 0x35:
LABEL_112:
          v189 = 0;
          v55 = a1[7];
          v190 = v194;
          v191 = v194;
          v192 = 4;
          v193 = 0;
          if ( *(_DWORD *)(*(_QWORD *)(v21 - 24) + 8LL) >> 8 != *(_DWORD *)(v55 + 4) )
          {
            v188 = 1;
            v56 = "Allocation instruction pointer not in the stack address space!";
            goto LABEL_114;
          }
          v102 = *(_QWORD *)(v21 + 32);
          v103 = *(unsigned __int8 *)(v102 + 8);
          if ( (unsigned __int8)v103 > 0xFu || (v105 = 35454, !_bittest64(&v105, v103)) )
          {
            if ( (unsigned int)(v103 - 13) > 1 && (_DWORD)v103 != 16 || !sub_16435F0(v102, (__int64)&v189) )
            {
              v188 = 1;
              v104 = "Cannot allocate unsized type";
              goto LABEL_224;
            }
          }
          if ( *(_BYTE *)(**(_QWORD **)(v21 - 48) + 8LL) != 11 )
          {
            v188 = 1;
            v56 = "Alloca array size must have integer type";
LABEL_114:
            v57 = *a1;
            v186[0] = (__int64)v56;
            v187 = 3;
            if ( v57 )
            {
              sub_16E2CE0(v186, v57);
              v58 = *(_BYTE **)(v57 + 24);
              if ( (unsigned __int64)v58 >= *(_QWORD *)(v57 + 16) )
              {
                sub_16E7DE0(v57, 10);
              }
              else
              {
                *(_QWORD *)(v57 + 24) = v58 + 1;
                *v58 = 10;
              }
              v59 = *a1;
              *((_BYTE *)a1 + 72) = 1;
              if ( v59 )
                goto LABEL_118;
            }
            else
            {
              *((_BYTE *)a1 + 72) = 1;
            }
LABEL_119:
            v60 = (unsigned __int64)v191;
            if ( v191 == v190 )
              goto LABEL_35;
LABEL_120:
            _libc_free(v60);
            goto LABEL_35;
          }
          v106 = *(_WORD *)(v21 - 6);
          if ( (unsigned int)(1 << v106) > 0x40000001 )
          {
            v188 = 1;
            v104 = "huge alignment values are unsupported";
LABEL_224:
            v186[0] = (__int64)v104;
            v187 = 3;
            sub_164FF40(a1, (__int64)v186);
            if ( *a1 )
LABEL_118:
              sub_164FA80(a1, v22);
            goto LABEL_119;
          }
          if ( (v106 & 0x40) != 0 )
            sub_165B9A0((__int64)a1, v21 - 24);
          sub_1663F80((__int64)a1, v21 - 24);
          v60 = (unsigned __int64)v191;
          if ( v191 != v190 )
            goto LABEL_120;
          goto LABEL_35;
        case 0x36:
LABEL_111:
          sub_1666000((__int64)a1, v21 - 24);
          goto LABEL_35;
        case 0x37:
LABEL_102:
          v52 = **(_QWORD **)(v21 - 48);
          if ( *(_BYTE *)(v52 + 8) != 15 )
          {
            v107 = *a1;
            v189 = "Store operand must be a pointer.";
            LOWORD(v191) = 259;
            if ( !v107 )
            {
              *((_BYTE *)a1 + 72) = 1;
              goto LABEL_35;
            }
            sub_16E2CE0(&v189, v107);
            v108 = *(_BYTE **)(v107 + 24);
            if ( (unsigned __int64)v108 >= *(_QWORD *)(v107 + 16) )
            {
              sub_16E7DE0(v107, 10);
            }
            else
            {
              *(_QWORD *)(v107 + 24) = v108 + 1;
              *v108 = 10;
            }
            v109 = *a1;
            *((_BYTE *)a1 + 72) = 1;
            if ( !v109 )
              goto LABEL_35;
            goto LABEL_465;
          }
          v53 = *(_QWORD *)(v52 + 24);
          if ( v53 != **(_QWORD **)(v21 - 72) )
          {
            v150 = *a1;
            v189 = "Stored value type does not match pointer operand type!";
            LOWORD(v191) = 259;
            if ( !v150 )
            {
              *((_BYTE *)a1 + 72) = 1;
              goto LABEL_35;
            }
            sub_16E2CE0(&v189, v150);
            v151 = *(_BYTE **)(v150 + 24);
            if ( (unsigned __int64)v151 >= *(_QWORD *)(v150 + 16) )
            {
              sub_16E7DE0(v150, 10);
            }
            else
            {
              *(_QWORD *)(v150 + 24) = v151 + 1;
              *v151 = 10;
            }
            v152 = *a1;
            *((_BYTE *)a1 + 72) = 1;
            if ( v152 )
            {
              sub_164FA80(a1, v22);
              if ( v53 )
              {
                v74 = *a1;
                v153 = *(_BYTE **)(*a1 + 24LL);
                if ( (unsigned __int64)v153 >= *(_QWORD *)(*a1 + 16LL) )
                  goto LABEL_454;
                *(_QWORD *)(v74 + 24) = v153 + 1;
                *v153 = 32;
                goto LABEL_427;
              }
            }
            goto LABEL_35;
          }
          if ( (unsigned int)(1 << (*(unsigned __int16 *)(v21 - 6) >> 1)) > 0x40000001 )
          {
            v189 = "huge alignment values are unsupported";
            LOWORD(v191) = 259;
            sub_164FF40(a1, (__int64)&v189);
            if ( !*a1 )
              goto LABEL_35;
            goto LABEL_473;
          }
          v54 = *(unsigned __int8 *)(v53 + 8);
          if ( (unsigned __int8)v54 > 0xFu || (v134 = 35454, !_bittest64(&v134, v54)) )
          {
            if ( (unsigned int)(v54 - 13) > 1 && (_DWORD)v54 != 16 || !sub_16435F0(v53, 0) )
            {
              v189 = "storing unsized types is not allowed";
              LOWORD(v191) = 259;
              sub_164FF40(a1, (__int64)&v189);
              if ( !*a1 )
                goto LABEL_35;
              goto LABEL_473;
            }
          }
          if ( !sub_15F32D0(v21 - 24) )
          {
            if ( *(_BYTE *)(v21 + 32) == 1 )
              goto LABEL_305;
            v189 = "Non-atomic store cannot have SynchronizationScope specified";
            LOWORD(v191) = 259;
            sub_164FF40(a1, (__int64)&v189);
            if ( !*a1 )
              goto LABEL_35;
            goto LABEL_473;
          }
          if ( ((*(unsigned __int16 *)(v21 - 6) >> 7) & 5) == 4 )
          {
            v189 = "Store cannot have Acquire ordering";
            LOWORD(v191) = 259;
            sub_164FF40(a1, (__int64)&v189);
            if ( !*a1 )
              goto LABEL_35;
            goto LABEL_473;
          }
          if ( (unsigned int)(1 << (*(unsigned __int16 *)(v21 - 6) >> 1)) >> 1 )
          {
            if ( (*(_BYTE *)(v53 + 8) & 0xFB) == 0xB || (unsigned __int8)(*(_BYTE *)(v53 + 8) - 1) <= 5u )
            {
              sub_164FB00((__int64)a1, v53, v21 - 24);
              goto LABEL_305;
            }
            v189 = "atomic store operand must have integer, pointer, or floating point type!";
            LOWORD(v191) = 259;
            sub_164FF40(a1, (__int64)&v189);
            if ( *a1 )
            {
              sub_164ECF0(*a1, v53);
              sub_164FA80(a1, v21 - 24);
            }
            goto LABEL_35;
          }
          v176 = *a1;
          v189 = "Atomic store must specify explicit alignment";
          LOWORD(v191) = 259;
          if ( v176 )
          {
            sub_16E2CE0(&v189, v176);
            v177 = *(_BYTE **)(v176 + 24);
            if ( (unsigned __int64)v177 >= *(_QWORD *)(v176 + 16) )
            {
              sub_16E7DE0(v176, 10);
            }
            else
            {
              *(_QWORD *)(v176 + 24) = v177 + 1;
              *v177 = 10;
            }
          }
          v178 = *a1 == 0;
          *((_BYTE *)a1 + 72) = 1;
          if ( v178 )
            goto LABEL_35;
          goto LABEL_465;
        case 0x38:
LABEL_101:
          sub_1666270(a1, v21 - 24);
          goto LABEL_35;
        case 0x39:
LABEL_192:
          if ( ((*(unsigned __int16 *)(v21 - 6) >> 1) & 0x7FFFBFFFu) - 4 <= 3 )
            goto LABEL_305;
          v189 = "fence instructions may only have acquire, release, acq_rel, or seq_cst ordering.";
          LOWORD(v191) = 259;
          sub_164FF40(a1, (__int64)&v189);
          if ( *a1 )
            goto LABEL_473;
          goto LABEL_35;
        case 0x3A:
LABEL_148:
          v76 = *(_WORD *)(v21 - 6);
          v77 = (v76 >> 2) & 7;
          if ( !v77 )
          {
            v141 = *a1;
            v189 = "cmpxchg instructions must be atomic.";
            LOWORD(v191) = 259;
            if ( !v141 )
            {
              *((_BYTE *)a1 + 72) = 1;
              goto LABEL_35;
            }
            sub_16E2CE0(&v189, v141);
            v142 = *(_BYTE **)(v141 + 24);
            if ( (unsigned __int64)v142 >= *(_QWORD *)(v141 + 16) )
            {
              sub_16E7DE0(v141, 10);
            }
            else
            {
              *(_QWORD *)(v141 + 24) = v142 + 1;
              *v142 = 10;
            }
            v143 = *a1;
            *((_BYTE *)a1 + 72) = 1;
            if ( !v143 )
              goto LABEL_35;
            goto LABEL_465;
          }
          v78 = (unsigned __int8)v76 >> 5;
          if ( !v78 )
          {
            v189 = "cmpxchg instructions must be atomic.";
            LOWORD(v191) = 259;
            sub_164FF40(a1, (__int64)&v189);
            if ( !*a1 )
              goto LABEL_35;
LABEL_473:
            sub_164FA80(a1, v21 - 24);
            goto LABEL_35;
          }
          if ( v77 == 1 )
          {
            v189 = "cmpxchg instructions cannot be unordered.";
            LOWORD(v191) = 259;
            sub_164FF40(a1, (__int64)&v189);
            if ( !*a1 )
              goto LABEL_35;
            goto LABEL_473;
          }
          if ( v78 == 1 )
          {
            v189 = "cmpxchg instructions cannot be unordered.";
            LOWORD(v191) = 259;
            sub_164FF40(a1, (__int64)&v189);
            if ( !*a1 )
              goto LABEL_35;
            goto LABEL_473;
          }
          if ( byte_42880A0[8 * v78 + v77] )
          {
            v189 = "cmpxchg instructions failure argument shall be no stronger than the success argument";
            LOWORD(v191) = 259;
            sub_164FF40(a1, (__int64)&v189);
            if ( !*a1 )
              goto LABEL_35;
            goto LABEL_473;
          }
          if ( (unsigned int)(v78 - 5) <= 1 )
          {
            v189 = "cmpxchg failure ordering cannot include release semantics";
            LOWORD(v191) = 259;
            sub_164FF40(a1, (__int64)&v189);
            if ( !*a1 )
              goto LABEL_35;
            goto LABEL_473;
          }
          v79 = **(_QWORD **)(v21 - 96);
          if ( *(_BYTE *)(v79 + 8) != 15 )
          {
            v189 = "First cmpxchg operand must be a pointer.";
            LOWORD(v191) = 259;
            sub_164FF40(a1, (__int64)&v189);
            if ( !*a1 )
              goto LABEL_35;
            goto LABEL_473;
          }
          v80 = *(_QWORD *)(v79 + 24);
          if ( (*(_BYTE *)(v80 + 8) & 0xFB) == 0xB )
          {
            sub_164FB00((__int64)a1, v80, v21 - 24);
            if ( v80 != **(_QWORD **)(v21 - 72) )
            {
              v189 = "Expected value type does not match pointer operand type!";
              LOWORD(v191) = 259;
              sub_164FF40(a1, (__int64)&v189);
              if ( !*a1 )
                goto LABEL_35;
LABEL_445:
              sub_164FA80(a1, v21 - 24);
              sub_164ECF0(*a1, v80);
              goto LABEL_35;
            }
            if ( v80 == **(_QWORD **)(v21 - 48) )
              goto LABEL_305;
            v189 = "Stored value type does not match pointer operand type!";
            LOWORD(v191) = 259;
            sub_164FF40(a1, (__int64)&v189);
            if ( *a1 )
              goto LABEL_445;
          }
          else
          {
            v189 = "cmpxchg operand must have integer or pointer type";
            LOWORD(v191) = 259;
            sub_164FF40(a1, (__int64)&v189);
            v172 = *a1;
            if ( *a1 )
            {
              v173 = *(_BYTE **)(v172 + 24);
              if ( (unsigned __int64)v173 >= *(_QWORD *)(v172 + 16) )
              {
                v172 = sub_16E7DE0(*a1, 32);
              }
              else
              {
                *(_QWORD *)(v172 + 24) = v173 + 1;
                *v173 = 32;
              }
              sub_154E060(v80, v172, 0, 0);
              sub_164FA80(a1, v21 - 24);
            }
          }
          goto LABEL_35;
        case 0x3B:
LABEL_138:
          if ( ((*(unsigned __int16 *)(v21 - 6) >> 2) & 7) != 0 )
          {
            if ( ((*(unsigned __int16 *)(v21 - 6) >> 2) & 7) == 1 )
            {
              v189 = "atomicrmw instructions cannot be unordered.";
              LOWORD(v191) = 259;
              sub_164FF40(a1, (__int64)&v189);
              if ( !*a1 )
                goto LABEL_35;
              goto LABEL_473;
            }
            v70 = **(_QWORD **)(v21 - 72);
            if ( *(_BYTE *)(v70 + 8) != 15 )
            {
              v113 = *a1;
              v189 = "First atomicrmw operand must be a pointer.";
              LOWORD(v191) = 259;
              if ( !v113 )
              {
                *((_BYTE *)a1 + 72) = 1;
                goto LABEL_35;
              }
              sub_16E2CE0(&v189, v113);
              v114 = *(_BYTE **)(v113 + 24);
              if ( (unsigned __int64)v114 >= *(_QWORD *)(v113 + 16) )
              {
                sub_16E7DE0(v113, 10);
              }
              else
              {
                *(_QWORD *)(v113 + 24) = v114 + 1;
                *v114 = 10;
              }
              v115 = *a1;
              *((_BYTE *)a1 + 72) = 1;
              if ( !v115 )
                goto LABEL_35;
              goto LABEL_465;
            }
            v53 = *(_QWORD *)(v70 + 24);
            if ( *(_BYTE *)(v53 + 8) == 11 )
            {
              sub_164FB00((__int64)a1, *(_QWORD *)(v70 + 24), v21 - 24);
              if ( v53 != **(_QWORD **)(v21 - 48) )
              {
                v189 = "Argument value type does not match pointer operand type!";
                LOWORD(v191) = 259;
                sub_164FF40(a1, (__int64)&v189);
                if ( !*a1 )
                  goto LABEL_35;
                sub_164FA80(a1, v21 - 24);
                v74 = *a1;
                v175 = *(_BYTE **)(*a1 + 24LL);
                if ( (unsigned __int64)v175 < *(_QWORD *)(*a1 + 16LL) )
                {
                  *(_QWORD *)(v74 + 24) = v175 + 1;
                  *v175 = 32;
                  goto LABEL_427;
                }
LABEL_454:
                v74 = sub_16E7DE0(v74, 32);
                goto LABEL_427;
              }
              if ( ((*(unsigned __int16 *)(v21 - 6) >> 5) & 0x7FFFBFFu) <= 0xA )
                goto LABEL_305;
              v189 = "Invalid binary operation!";
              LOWORD(v191) = 259;
              sub_164FF40(a1, (__int64)&v189);
              if ( *a1 )
                goto LABEL_473;
            }
            else
            {
              v71 = *a1;
              v189 = "atomicrmw operand must have integer type!";
              LOWORD(v191) = 259;
              if ( v71 )
              {
                sub_16E2CE0(&v189, v71);
                v72 = *(_BYTE **)(v71 + 24);
                if ( (unsigned __int64)v72 >= *(_QWORD *)(v71 + 16) )
                {
                  sub_16E7DE0(v71, 10);
                }
                else
                {
                  *(_QWORD *)(v71 + 24) = v72 + 1;
                  *v72 = 10;
                }
                v73 = *a1;
                *((_BYTE *)a1 + 72) = 1;
                if ( v73 )
                {
                  sub_164FA80(a1, v22);
                  v74 = *a1;
                  v75 = *(_BYTE **)(*a1 + 24LL);
                  if ( (unsigned __int64)v75 < *(_QWORD *)(*a1 + 16LL) )
                  {
                    *(_QWORD *)(v74 + 24) = v75 + 1;
                    *v75 = 32;
LABEL_427:
                    sub_154E060(v53, v74, 0, 0);
                    goto LABEL_35;
                  }
                  goto LABEL_454;
                }
              }
              else
              {
                *((_BYTE *)a1 + 72) = 1;
              }
            }
          }
          else
          {
            v144 = *a1;
            v189 = "atomicrmw instructions must be atomic.";
            LOWORD(v191) = 259;
            if ( v144 )
            {
              sub_16E2CE0(&v189, v144);
              v145 = *(_BYTE **)(v144 + 24);
              if ( (unsigned __int64)v145 >= *(_QWORD *)(v144 + 16) )
              {
                sub_16E7DE0(v144, 10);
              }
              else
              {
                *(_QWORD *)(v144 + 24) = v145 + 1;
                *v145 = 10;
              }
              v146 = *a1;
              *((_BYTE *)a1 + 72) = 1;
              if ( !v146 )
                goto LABEL_35;
              goto LABEL_465;
            }
            *((_BYTE *)a1 + 72) = 1;
          }
          goto LABEL_35;
        case 0x3C:
LABEL_137:
          sub_16668F0((__int64)a1, v21 - 24);
          goto LABEL_35;
        case 0x3D:
LABEL_136:
          sub_1666A60((__int64)a1, v21 - 24);
          goto LABEL_35;
        case 0x3E:
LABEL_135:
          sub_1666BD0((__int64)a1, v21 - 24);
          goto LABEL_35;
        case 0x3F:
LABEL_134:
          sub_1666D40((__int64)a1, (__int64 *)(v21 - 24));
          goto LABEL_35;
        case 0x40:
LABEL_133:
          sub_1666EE0((__int64)a1, (__int64 *)(v21 - 24));
          goto LABEL_35;
        case 0x41:
LABEL_196:
          sub_1667080((__int64)a1, (__int64 *)(v21 - 24));
          goto LABEL_35;
        case 0x42:
LABEL_195:
          sub_1667220((__int64)a1, (__int64 *)(v21 - 24));
          goto LABEL_35;
        case 0x43:
LABEL_197:
          sub_16673C0((__int64)a1, v21 - 24);
          goto LABEL_35;
        case 0x44:
LABEL_172:
          sub_1667540((__int64)a1, v21 - 24);
          goto LABEL_35;
        case 0x45:
LABEL_194:
          sub_16676C0((__int64)a1, v21 - 24);
          goto LABEL_35;
        case 0x46:
LABEL_100:
          sub_16678F0((__int64)a1, v21 - 24);
          goto LABEL_35;
        case 0x47:
LABEL_98:
          if ( (unsigned __int8)sub_15FC090(47, *(_QWORD **)(v21 - 48), *(_QWORD *)(v21 - 24)) )
            goto LABEL_305;
          v189 = "Invalid bitcast";
          LOWORD(v191) = 259;
          sub_164FF40(a1, (__int64)&v189);
          if ( *a1 )
            goto LABEL_473;
          goto LABEL_35;
        case 0x48:
LABEL_97:
          sub_1667B10((__int64)a1, v21 - 24);
          goto LABEL_35;
        case 0x49:
LABEL_92:
          v50 = *(_QWORD *)(v21 + 16);
          if ( (*(_BYTE *)(*(_QWORD *)(v50 + 56) + 18LL) & 8) != 0 )
          {
            if ( v22 == sub_157ED20(v50) )
            {
              v51 = *(_BYTE *)(*(_QWORD *)(v21 - 48) + 16LL);
              if ( v51 == 16 || (unsigned __int8)(v51 - 73) <= 1u )
                goto LABEL_96;
              v131 = *a1;
              v189 = "CleanupPadInst has an invalid parent.";
              LOWORD(v191) = 259;
              if ( !v131 )
              {
                *((_BYTE *)a1 + 72) = 1;
                goto LABEL_35;
              }
              sub_16E2CE0(&v189, v131);
              v132 = *(_BYTE **)(v131 + 24);
              if ( (unsigned __int64)v132 >= *(_QWORD *)(v131 + 16) )
              {
                sub_16E7DE0(v131, 10);
              }
              else
              {
                *(_QWORD *)(v131 + 24) = v132 + 1;
                *v132 = 10;
              }
              v133 = *a1;
              *((_BYTE *)a1 + 72) = 1;
              if ( !v133 )
                goto LABEL_35;
            }
            else
            {
              v128 = *a1;
              v189 = "CleanupPadInst not the first non-PHI instruction in the block.";
              LOWORD(v191) = 259;
              if ( !v128 )
              {
                *((_BYTE *)a1 + 72) = 1;
                goto LABEL_35;
              }
              sub_16E2CE0(&v189, v128);
              v129 = *(_BYTE **)(v128 + 24);
              if ( (unsigned __int64)v129 >= *(_QWORD *)(v128 + 16) )
              {
                sub_16E7DE0(v128, 10);
              }
              else
              {
                *(_QWORD *)(v128 + 24) = v129 + 1;
                *v129 = 10;
              }
              v130 = *a1;
              *((_BYTE *)a1 + 72) = 1;
              if ( !v130 )
                goto LABEL_35;
            }
          }
          else
          {
            v110 = *a1;
            v189 = "CleanupPadInst needs to be in a function with a personality.";
            LOWORD(v191) = 259;
            if ( !v110 )
            {
              *((_BYTE *)a1 + 72) = 1;
              goto LABEL_35;
            }
            sub_16E2CE0(&v189, v110);
            v111 = *(_BYTE **)(v110 + 24);
            if ( (unsigned __int64)v111 >= *(_QWORD *)(v110 + 16) )
            {
              sub_16E7DE0(v110, 10);
            }
            else
            {
              *(_QWORD *)(v110 + 24) = v111 + 1;
              *v111 = 10;
            }
            v112 = *a1;
            *((_BYTE *)a1 + 72) = 1;
            if ( !v112 )
              goto LABEL_35;
          }
          goto LABEL_465;
        case 0x4A:
LABEL_83:
          v44 = *(_QWORD *)(v21 + 16);
          if ( (*(_BYTE *)(*(_QWORD *)(v44 + 56) + 18LL) & 8) == 0 )
          {
            v122 = *a1;
            v189 = "CatchPadInst needs to be in a function with a personality.";
            LOWORD(v191) = 259;
            if ( !v122 )
            {
              *((_BYTE *)a1 + 72) = 1;
              goto LABEL_35;
            }
            sub_16E2CE0(&v189, v122);
            v123 = *(_BYTE **)(v122 + 24);
            if ( (unsigned __int64)v123 >= *(_QWORD *)(v122 + 16) )
            {
              sub_16E7DE0(v122, 10);
            }
            else
            {
              *(_QWORD *)(v122 + 24) = v123 + 1;
              *v123 = 10;
            }
            v124 = *a1;
            *((_BYTE *)a1 + 72) = 1;
            if ( !v124 )
              goto LABEL_35;
            goto LABEL_465;
          }
          v45 = *(_QWORD *)(v21 - 48);
          if ( *(_BYTE *)(v45 + 16) != 34 )
          {
            v46 = *a1;
            v189 = "CatchPadInst needs to be directly nested in a CatchSwitchInst.";
            LOWORD(v191) = 259;
            if ( !v46 )
            {
              *((_BYTE *)a1 + 72) = 1;
              goto LABEL_35;
            }
            sub_16E2CE0(&v189, v46);
            v47 = *(_BYTE **)(v46 + 24);
            if ( (unsigned __int64)v47 >= *(_QWORD *)(v46 + 16) )
            {
              sub_16E7DE0(v46, 10);
            }
            else
            {
              *(_QWORD *)(v46 + 24) = v47 + 1;
              *v47 = 10;
            }
            v48 = *a1;
            *((_BYTE *)a1 + 72) = 1;
            if ( v48 )
            {
              if ( *(_BYTE *)(v45 + 16) <= 0x17u )
              {
                sub_1553920((__int64 *)v45, v48, 1, (__int64)(a1 + 2));
                v31 = *a1;
                v174 = *(_BYTE **)(*a1 + 24LL);
                if ( (unsigned __int64)v174 >= *(_QWORD *)(*a1 + 16LL) )
                  goto LABEL_469;
                *(_QWORD *)(v31 + 24) = v174 + 1;
                *v174 = 10;
              }
              else
              {
                sub_155BD40(v45, v48, (__int64)(a1 + 2), 0);
                v31 = *a1;
                v49 = *(_BYTE **)(*a1 + 24LL);
                if ( (unsigned __int64)v49 >= *(_QWORD *)(*a1 + 16LL) )
                  goto LABEL_469;
                *(_QWORD *)(v31 + 24) = v49 + 1;
                *v49 = 10;
              }
            }
            goto LABEL_35;
          }
          if ( v22 == sub_157ED20(v44) )
          {
LABEL_96:
            sub_1654C20(a1, v21 - 24);
            sub_1669240((__int64)a1, v21 - 24);
            goto LABEL_35;
          }
          v189 = "CatchPadInst not the first non-PHI instruction in the block.";
          LOWORD(v191) = 259;
          sub_164FF40(a1, (__int64)&v189);
          if ( !*a1 )
            goto LABEL_35;
          goto LABEL_473;
        case 0x4B:
LABEL_82:
          sub_1667CA0((__int64)a1, v21 - 24);
          goto LABEL_35;
        case 0x4C:
LABEL_81:
          sub_1667DE0((__int64)a1, v21 - 24);
          goto LABEL_35;
        case 0x4D:
LABEL_80:
          sub_1667F10((__int64)a1, v21 - 24);
          goto LABEL_35;
        case 0x4E:
LABEL_79:
          sub_1668F50(a1, v21 - 24);
          goto LABEL_35;
        case 0x4F:
LABEL_78:
          sub_1668150((__int64)a1, (_QWORD *)(v21 - 24));
          goto LABEL_35;
        case 0x50:
        case 0x51:
LABEL_77:
          v186[0] = v21 - 24;
          v189 = "User-defined operators should not live outside of a pass!";
          LOWORD(v191) = 259;
          sub_1654980(a1, (__int64)&v189, v186);
          goto LABEL_35;
        case 0x52:
          goto LABEL_305;
        case 0x53:
LABEL_75:
          if ( sub_15FA460(*(_QWORD *)(v21 - 72), *(_QWORD *)(v21 - 48)) )
            goto LABEL_305;
          v189 = "Invalid extractelement operands!";
          LOWORD(v191) = 259;
          sub_164FF40(a1, (__int64)&v189);
          if ( *a1 )
            goto LABEL_473;
          goto LABEL_35;
        case 0x54:
LABEL_73:
          if ( (unsigned __int8)sub_15FA630(*(_QWORD *)(v21 - 96), *(_QWORD **)(v21 - 72), *(_QWORD *)(v21 - 48)) )
            goto LABEL_305;
          v189 = "Invalid insertelement operands!";
          LOWORD(v191) = 259;
          sub_164FF40(a1, (__int64)&v189);
          if ( *a1 )
            goto LABEL_473;
          goto LABEL_35;
        case 0x55:
LABEL_71:
          if ( (unsigned __int8)sub_15FA830(*(_QWORD *)(v21 - 96), *(_QWORD **)(v21 - 72), *(_QWORD *)(v21 - 48)) )
            goto LABEL_305;
          v189 = "Invalid shufflevector operands!";
          LOWORD(v191) = 259;
          sub_164FF40(a1, (__int64)&v189);
          if ( *a1 )
            goto LABEL_473;
          goto LABEL_35;
        case 0x56:
LABEL_68:
          if ( *(_QWORD *)(v21 - 24) == sub_15FB2A0(
                                          **(_QWORD **)(v21 - 48),
                                          *(unsigned int **)(v21 + 32),
                                          *(unsigned int *)(v21 + 40)) )
            goto LABEL_305;
          v189 = "Invalid ExtractValueInst operands!";
          LOWORD(v191) = 259;
          sub_164FF40(a1, (__int64)&v189);
          if ( *a1 )
            goto LABEL_473;
          goto LABEL_35;
        case 0x57:
LABEL_65:
          if ( **(_QWORD **)(v21 - 48) == sub_15FB2A0(
                                            **(_QWORD **)(v21 - 72),
                                            *(unsigned int **)(v21 + 32),
                                            *(unsigned int *)(v21 + 40)) )
            goto LABEL_305;
          v189 = "Invalid InsertValueInst operands!";
          LOWORD(v191) = 259;
          sub_164FF40(a1, (__int64)&v189);
          if ( *a1 )
            goto LABEL_473;
          goto LABEL_35;
        case 0x58:
LABEL_177:
          if ( (*(_BYTE *)(v21 - 6) & 1) == 0 )
          {
            v189 = "LandingPadInst needs at least one clause or to be a cleanup.";
            LOWORD(v191) = 259;
            sub_164FF40(a1, (__int64)&v189);
            if ( !*a1 )
              goto LABEL_35;
            goto LABEL_473;
          }
LABEL_178:
          sub_1654C20(a1, v21 - 24);
          v86 = a1[89];
          if ( v86 )
          {
            if ( v86 != *(_QWORD *)(v21 - 24) )
            {
              v138 = *a1;
              v189 = "The landingpad instruction should have a consistent result type inside a function.";
              LOWORD(v191) = 259;
              if ( !v138 )
              {
                *((_BYTE *)a1 + 72) = 1;
                goto LABEL_35;
              }
              sub_16E2CE0(&v189, v138);
              v139 = *(_BYTE **)(v138 + 24);
              if ( (unsigned __int64)v139 >= *(_QWORD *)(v138 + 16) )
              {
                sub_16E7DE0(v138, 10);
              }
              else
              {
                *(_QWORD *)(v138 + 24) = v139 + 1;
                *v139 = 10;
              }
              v140 = *a1;
              *((_BYTE *)a1 + 72) = 1;
              if ( !v140 )
                goto LABEL_35;
LABEL_465:
              sub_164FA80(a1, v22);
              goto LABEL_35;
            }
          }
          else
          {
            a1[89] = *(_QWORD *)(v21 - 24);
          }
          v87 = *(_QWORD *)(v21 + 16);
          if ( (*(_BYTE *)(*(_QWORD *)(v87 + 56) + 18LL) & 8) != 0 )
          {
            if ( v22 == sub_157F7B0(v87) )
            {
              v88 = *(_DWORD *)(v21 - 4) & 0xFFFFFFF;
              if ( !v88 )
              {
LABEL_305:
                sub_1663F80((__int64)a1, v21 - 24);
                goto LABEL_35;
              }
              v89 = 24LL * v88;
              v90 = 0;
              while ( 2 )
              {
                if ( (*(_BYTE *)(v21 - 1) & 0x40) != 0 )
                {
                  v91 = *(_BYTE **)(*(_QWORD *)(v21 - 32) + v90);
                  v92 = *(_BYTE *)(*(_QWORD *)v91 + 8LL);
                  if ( v92 != 14 )
                    goto LABEL_185;
LABEL_189:
                  if ( ((v91[16] - 6) & 0xFB) != 0 )
                  {
                    v189 = "Filter operand is not an array of constants!";
                    LOWORD(v191) = 259;
                    sub_164FF40(a1, (__int64)&v189);
                    if ( !*a1 )
                      goto LABEL_35;
                    goto LABEL_473;
                  }
                }
                else
                {
                  v91 = *(_BYTE **)(v21 - v89 + v90 - 24);
                  v92 = *(_BYTE *)(*(_QWORD *)v91 + 8LL);
                  if ( v92 == 14 )
                    goto LABEL_189;
LABEL_185:
                  if ( v92 != 15 )
                  {
                    v189 = "Catch operand does not have pointer type!";
                    LOWORD(v191) = 259;
                    sub_164FF40(a1, (__int64)&v189);
                    if ( !*a1 )
                      goto LABEL_35;
                    goto LABEL_473;
                  }
                }
                v90 += 24;
                if ( v89 == v90 )
                  goto LABEL_305;
                continue;
              }
            }
            v169 = *a1;
            v189 = "LandingPadInst not the first non-PHI instruction in the block.";
            LOWORD(v191) = 259;
            if ( !v169 )
            {
              *((_BYTE *)a1 + 72) = 1;
              goto LABEL_35;
            }
            sub_16E2CE0(&v189, v169);
            v170 = *(_BYTE **)(v169 + 24);
            if ( (unsigned __int64)v170 >= *(_QWORD *)(v169 + 16) )
            {
              sub_16E7DE0(v169, 10);
            }
            else
            {
              *(_QWORD *)(v169 + 24) = v170 + 1;
              *v170 = 10;
            }
            v171 = *a1;
            *((_BYTE *)a1 + 72) = 1;
            if ( !v171 )
              goto LABEL_35;
          }
          else
          {
            v135 = *a1;
            v189 = "LandingPadInst needs to be in a function with a personality.";
            LOWORD(v191) = 259;
            if ( !v135 )
            {
              *((_BYTE *)a1 + 72) = 1;
              goto LABEL_35;
            }
            sub_16E2CE0(&v189, v135);
            v136 = *(_BYTE **)(v135 + 24);
            if ( (unsigned __int64)v136 >= *(_QWORD *)(v135 + 16) )
            {
              sub_16E7DE0(v135, 10);
            }
            else
            {
              *(_QWORD *)(v135 + 24) = v136 + 1;
              *v136 = 10;
            }
            v137 = *a1;
            *((_BYTE *)a1 + 72) = 1;
            if ( !v137 )
              goto LABEL_35;
          }
          goto LABEL_465;
        default:
          ++*(_DWORD *)v21;
          BUG();
      }
    }
LABEL_37:
    sub_1669A40(a1);
    ++a1[20];
    v33 = (void *)a1[22];
    if ( v33 != (void *)a1[21] )
    {
      v34 = 4 * (*((_DWORD *)a1 + 47) - *((_DWORD *)a1 + 48));
      v35 = *((unsigned int *)a1 + 46);
      if ( v34 < 0x20 )
        v34 = 32;
      if ( (unsigned int)v35 > v34 )
      {
        sub_16CC920(a1 + 20);
LABEL_43:
        v36 = *((_DWORD *)a1 + 194);
        ++a1[95];
        *((_DWORD *)a1 + 366) = 0;
        a1[89] = 0;
        *((_BYTE *)a1 + 720) = 0;
        if ( v36 )
        {
          v93 = 4 * v36;
          v37 = *((unsigned int *)a1 + 196);
          if ( (unsigned int)(4 * v36) < 0x40 )
            v93 = 64;
          if ( (unsigned int)v37 <= v93 )
            goto LABEL_46;
          v94 = (_QWORD *)a1[96];
          v95 = v36 - 1;
          if ( v95 )
          {
            _BitScanReverse(&v95, v95);
            v96 = 1 << (33 - (v95 ^ 0x1F));
            if ( v96 < 64 )
              v96 = 64;
            if ( (_DWORD)v37 == v96 )
            {
              a1[97] = 0;
              v179 = &v94[2 * (unsigned int)v37];
              do
              {
                if ( v94 )
                  *v94 = -8;
                v94 += 2;
              }
              while ( v179 != v94 );
              goto LABEL_49;
            }
          }
          else
          {
            v96 = 64;
          }
          j___libc_free_0(v94);
          v97 = ((((((((4 * v96 / 3u + 1) | ((unsigned __int64)(4 * v96 / 3u + 1) >> 1)) >> 2)
                   | (4 * v96 / 3u + 1)
                   | ((unsigned __int64)(4 * v96 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v96 / 3u + 1) | ((unsigned __int64)(4 * v96 / 3u + 1) >> 1)) >> 2)
                 | (4 * v96 / 3u + 1)
                 | ((unsigned __int64)(4 * v96 / 3u + 1) >> 1)) >> 8)
               | (((((4 * v96 / 3u + 1) | ((unsigned __int64)(4 * v96 / 3u + 1) >> 1)) >> 2)
                 | (4 * v96 / 3u + 1)
                 | ((unsigned __int64)(4 * v96 / 3u + 1) >> 1)) >> 4)
               | (((4 * v96 / 3u + 1) | ((unsigned __int64)(4 * v96 / 3u + 1) >> 1)) >> 2)
               | (4 * v96 / 3u + 1)
               | ((unsigned __int64)(4 * v96 / 3u + 1) >> 1)) >> 16;
          v98 = (v97
               | (((((((4 * v96 / 3u + 1) | ((unsigned __int64)(4 * v96 / 3u + 1) >> 1)) >> 2)
                   | (4 * v96 / 3u + 1)
                   | ((unsigned __int64)(4 * v96 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v96 / 3u + 1) | ((unsigned __int64)(4 * v96 / 3u + 1) >> 1)) >> 2)
                 | (4 * v96 / 3u + 1)
                 | ((unsigned __int64)(4 * v96 / 3u + 1) >> 1)) >> 8)
               | (((((4 * v96 / 3u + 1) | ((unsigned __int64)(4 * v96 / 3u + 1) >> 1)) >> 2)
                 | (4 * v96 / 3u + 1)
                 | ((unsigned __int64)(4 * v96 / 3u + 1) >> 1)) >> 4)
               | (((4 * v96 / 3u + 1) | ((unsigned __int64)(4 * v96 / 3u + 1) >> 1)) >> 2)
               | (4 * v96 / 3u + 1)
               | ((unsigned __int64)(4 * v96 / 3u + 1) >> 1))
              + 1;
          *((_DWORD *)a1 + 196) = v98;
          v99 = (_QWORD *)sub_22077B0(16 * v98);
          v100 = *((unsigned int *)a1 + 196);
          a1[97] = 0;
          a1[96] = v99;
          for ( i = &v99[2 * v100]; i != v99; v99 += 2 )
          {
            if ( v99 )
              *v99 = -8;
          }
        }
        else if ( *((_DWORD *)a1 + 195) )
        {
          v37 = *((unsigned int *)a1 + 196);
          if ( (unsigned int)v37 <= 0x40 )
          {
LABEL_46:
            v38 = (_QWORD *)a1[96];
            for ( j = &v38[2 * v37]; j != v38; v38 += 2 )
              *v38 = -8;
            a1[97] = 0;
            goto LABEL_49;
          }
          j___libc_free_0(a1[96]);
          a1[96] = 0;
          a1[97] = 0;
          *((_DWORD *)a1 + 196) = 0;
        }
LABEL_49:
        v40 = a1[99];
        if ( v40 != a1[100] )
          a1[100] = v40;
        return *((unsigned __int8 *)a1 + 72) ^ 1u;
      }
      memset(v33, -1, 8 * v35);
    }
    *(_QWORD *)((char *)a1 + 188) = 0;
    goto LABEL_43;
  }
  while ( 1 )
  {
    if ( !v3 )
      BUG();
    v4 = *(_QWORD *)(v3 + 16) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v4 == v3 + 16 )
      break;
    if ( !v4 )
      BUG();
    if ( (unsigned int)*(unsigned __int8 *)(v4 - 8) - 25 > 9 )
      break;
    v3 = *(_QWORD *)(v3 + 8);
    if ( v183 == v3 )
      goto LABEL_20;
  }
  v5 = *a1;
  result = 0;
  if ( *a1 )
  {
    v7 = *(__m128i **)(v5 + 24);
    if ( *(_QWORD *)(v5 + 16) - (_QWORD)v7 <= 0x18u )
    {
      v5 = sub_16E7EE0(*a1, "Basic Block in function '", 25);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F64280);
      v7[1].m128i_i8[8] = 39;
      v7[1].m128i_i64[0] = 0x206E6F6974636E75LL;
      *v7 = si128;
      *(_QWORD *)(v5 + 24) += 25LL;
    }
    v9 = sub_1649960(a2);
    v11 = *(__m128i **)(v5 + 24);
    v12 = v9;
    v13 = *(_QWORD *)(v5 + 16) - (_QWORD)v11;
    if ( v13 < v10 )
    {
      v42 = sub_16E7EE0(v5, v12);
      v11 = *(__m128i **)(v42 + 24);
      v5 = v42;
      v13 = *(_QWORD *)(v42 + 16) - (_QWORD)v11;
    }
    else if ( v10 )
    {
      v185 = v10;
      memcpy(v11, v12, v10);
      v43 = *(_QWORD *)(v5 + 16);
      v11 = (__m128i *)(v185 + *(_QWORD *)(v5 + 24));
      *(_QWORD *)(v5 + 24) = v11;
      v13 = v43 - (_QWORD)v11;
    }
    if ( v13 <= 0x1B )
    {
      sub_16E7EE0(v5, "' does not have terminator!\n", 28);
    }
    else
    {
      v14 = _mm_load_si128((const __m128i *)&xmmword_3F64290);
      qmemcpy(&v11[1], "terminator!\n", 12);
      *v11 = v14;
      *(_QWORD *)(v5 + 24) += 28LL;
    }
    sub_1553920((__int64 *)(v3 - 24), *a1, 1, (__int64)(a1 + 2));
    v15 = *a1;
    v16 = *(_BYTE **)(*a1 + 24LL);
    if ( *(_BYTE **)(*a1 + 16LL) == v16 )
    {
      sub_16E7EE0(v15, "\n", 1);
      return 0;
    }
    else
    {
      *v16 = 10;
      result = 0;
      ++*(_QWORD *)(v15 + 24);
    }
  }
  return result;
}
