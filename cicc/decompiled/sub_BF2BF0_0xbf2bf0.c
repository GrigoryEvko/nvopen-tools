// Function: sub_BF2BF0
// Address: 0xbf2bf0
//
__int64 __fastcall sub_BF2BF0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 result; // rax
  __int64 v4; // rdi
  unsigned int v5; // r15d
  __int64 v6; // rbx
  unsigned __int8 v7; // al
  __int64 v8; // r12
  _BYTE *v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rdx
  const char *v12; // rdi
  __int64 v13; // rdi
  _BYTE *v14; // rax
  unsigned __int8 v15; // al
  __int64 v16; // r13
  __int64 v17; // rdx
  _BYTE *v18; // r12
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  unsigned __int8 v23; // al
  __int64 v24; // r13
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // r12
  _BYTE *v28; // rax
  __int64 *v29; // r9
  __int64 *v30; // rbx
  __int64 *v31; // r15
  _BYTE *v32; // r13
  __int64 v33; // rdx
  unsigned int v34; // ecx
  const char **v35; // rax
  const char *v36; // r10
  const char *v37; // rax
  unsigned __int8 v38; // cl
  __int64 v39; // rdx
  unsigned __int8 v40; // al
  const char *v41; // r14
  _BYTE *v42; // r12
  __int64 *v43; // rax
  __int64 v44; // rax
  const char **v45; // rax
  __int64 v46; // r12
  const char **v47; // r13
  _BYTE *v48; // rax
  const char *v49; // r13
  int v50; // esi
  int v51; // r10d
  const char **v52; // r9
  unsigned int v53; // edx
  const char **v54; // rax
  const char *v55; // r8
  __int64 v56; // rax
  __int64 v57; // rsi
  unsigned int v58; // edi
  __int64 v59; // rax
  _BYTE *v60; // rax
  const char **v61; // rdx
  const char *v62; // rax
  _BYTE *v63; // rax
  __int64 v64; // rcx
  const char *v65; // rax
  int v66; // eax
  __int64 v67; // r8
  _BYTE *v68; // rax
  __int64 v69; // rdi
  _BYTE *v70; // rax
  const char **v71; // rax
  unsigned __int8 v72; // al
  __int64 v73; // r13
  __int64 v74; // rax
  __int64 v75; // rax
  _BYTE *v76; // r8
  int v77; // eax
  _BYTE *v78; // rax
  _BYTE *v79; // r8
  __int64 v80; // rdx
  const void *v81; // rax
  size_t v82; // rdx
  __int64 v83; // rax
  const void *v84; // rax
  size_t v85; // rdx
  const void *v86; // rax
  size_t v87; // rdx
  __int64 v88; // rax
  const void *v89; // rax
  size_t v90; // rdx
  _BYTE *v91; // rax
  __int64 v92; // rdx
  const char **v93; // r9
  const char **v94; // r13
  __int64 v95; // r15
  const char **v96; // r14
  _BYTE *v97; // rax
  __int64 v98; // rdi
  _BYTE *v99; // rax
  const char *v100; // r12
  __int64 v101; // r12
  __int64 v102; // rax
  _BYTE *v103; // rax
  _BYTE *v104; // rax
  _BYTE *v105; // rax
  const char *v106; // rcx
  int v107; // edx
  int v108; // r8d
  const char **v109; // rax
  char *v110; // r12
  __int64 *v111; // rdx
  unsigned __int8 **v112; // rax
  unsigned __int8 *v113; // rax
  _BYTE *v114; // rdx
  unsigned __int8 **v115; // rax
  unsigned __int8 *v116; // rax
  __int64 v117; // rax
  const char **v118; // rax
  const char *v119; // rax
  __int64 *v120; // [rsp+0h] [rbp-160h]
  _BYTE *v121; // [rsp+0h] [rbp-160h]
  _BYTE *v122; // [rsp+0h] [rbp-160h]
  __int64 *v123; // [rsp+0h] [rbp-160h]
  _BYTE *v124; // [rsp+8h] [rbp-158h]
  unsigned int v125; // [rsp+8h] [rbp-158h]
  _BYTE *v126; // [rsp+10h] [rbp-150h]
  __int64 v127; // [rsp+20h] [rbp-140h]
  __int64 v128; // [rsp+28h] [rbp-138h]
  int v129; // [rsp+30h] [rbp-130h]
  __int64 v130; // [rsp+30h] [rbp-130h]
  __int64 v131; // [rsp+38h] [rbp-128h]
  unsigned int v132; // [rsp+44h] [rbp-11Ch] BYREF
  const char **v133; // [rsp+48h] [rbp-118h] BYREF
  __int64 v134; // [rsp+50h] [rbp-110h] BYREF
  __int64 v135; // [rsp+58h] [rbp-108h]
  __int64 v136; // [rsp+60h] [rbp-100h]
  unsigned int v137; // [rsp+68h] [rbp-F8h]
  const char *v138; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v139; // [rsp+78h] [rbp-E8h]
  char v140; // [rsp+90h] [rbp-D0h]
  char v141; // [rsp+91h] [rbp-CFh]
  __int64 *v142; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v143; // [rsp+A8h] [rbp-B8h]
  _BYTE v144[176]; // [rsp+B0h] [rbp-B0h] BYREF

  v2 = a1;
  result = *(_QWORD *)(a1 + 8);
  v4 = *(_QWORD *)(result + 864);
  v131 = v4;
  if ( !v4 )
    return result;
  v5 = 0;
  v134 = 0;
  v142 = (__int64 *)v144;
  v135 = 0;
  v136 = 0;
  v137 = 0;
  v143 = 0x1000000000LL;
  v127 = -1;
  v129 = sub_B91A00(v4);
  v128 = -1;
  if ( !v129 )
    goto LABEL_38;
  do
  {
    while ( 1 )
    {
      a2 = v5;
      v6 = sub_B91A10(v131, v5);
      v7 = *(_BYTE *)(v6 - 16);
      if ( (v7 & 2) != 0 )
      {
        if ( *(_DWORD *)(v6 - 24) != 3 )
          goto LABEL_8;
        v43 = *(__int64 **)(v6 - 32);
        v42 = (_BYTE *)(v6 - 16);
      }
      else
      {
        if ( ((*(_WORD *)(v6 - 16) >> 6) & 0xF) != 3 )
        {
LABEL_8:
          v8 = *(_QWORD *)v2;
          v141 = 1;
          v138 = "incorrect number of operands in module flag";
          v140 = 3;
          if ( !v8 )
            goto LABEL_61;
          sub_CA0E80(&v138, v8);
          v9 = *(_BYTE **)(v8 + 32);
          if ( (unsigned __int64)v9 >= *(_QWORD *)(v8 + 24) )
          {
            sub_CB5D20(v8, 10);
          }
          else
          {
            *(_QWORD *)(v8 + 32) = v9 + 1;
            *v9 = 10;
          }
          a2 = *(_QWORD *)v2;
          *(_BYTE *)(v2 + 152) = 1;
          if ( a2 )
          {
            v10 = *(_QWORD *)(v2 + 8);
            v11 = v2 + 16;
            v12 = (const char *)v6;
            goto LABEL_13;
          }
          goto LABEL_15;
        }
        v42 = (_BYTE *)(v6 - 16);
        v43 = (__int64 *)(v6 - 8LL * ((v7 >> 2) & 0xF) - 16);
      }
      a2 = (__int64)&v132;
      if ( (unsigned __int8)sub_BA9150(*v43, &v132) )
      {
        v49 = (const char *)*((_QWORD *)sub_A17150(v42) + 1);
        if ( !v49 || *v49 )
        {
          v105 = sub_A17150(v42);
          v141 = 1;
          v61 = (const char **)(v105 + 8);
          v62 = "invalid ID operand in module flag (expected metadata string)";
          goto LABEL_82;
        }
        if ( v132 == 7 )
        {
          v102 = *((_QWORD *)sub_A17150(v42) + 2);
          if ( !v102 || *(_BYTE *)v102 != 1 || **(_BYTE **)(v102 + 136) != 17 )
          {
            v103 = sub_A17150(v42);
            v141 = 1;
            v61 = (const char **)(v103 + 16);
            v62 = "invalid value for 'max' module flag (expected constant integer)";
            goto LABEL_82;
          }
        }
        else if ( v132 > 7 )
        {
          if ( v132 == 8 )
          {
            v56 = *((_QWORD *)sub_A17150(v42) + 2);
            if ( !v56 )
              goto LABEL_81;
            if ( *(_BYTE *)v56 != 1 )
              goto LABEL_81;
            v57 = *(_QWORD *)(v56 + 136);
            if ( *(_BYTE *)v57 != 17 )
              goto LABEL_81;
            v58 = *(_DWORD *)(v57 + 32);
            v59 = *(_QWORD *)(v57 + 24);
            if ( v58 > 0x40 )
              v59 = *(_QWORD *)(v59 + 8LL * ((v58 - 1) >> 6));
            if ( (v59 & (1LL << ((unsigned __int8)v58 - 1))) != 0 )
            {
LABEL_81:
              v60 = sub_A17150(v42);
              v141 = 1;
              v61 = (const char **)(v60 + 16);
              v62 = "invalid value for 'min' module flag (expected constant non-negative integer)";
LABEL_82:
              a2 = (__int64)&v138;
              v138 = v62;
              v140 = 3;
              sub_BE7680((_BYTE *)v2, (__int64)&v138, v61);
              goto LABEL_15;
            }
          }
        }
        else if ( v132 == 3 )
        {
          v76 = (_BYTE *)*((_QWORD *)sub_A17150(v42) + 2);
          if ( (unsigned __int8)(*v76 - 5) > 0x1Fu
            || ((*(v76 - 16) & 2) == 0 ? (v77 = (*((_WORD *)v76 - 8) >> 6) & 0xF) : (v77 = *((_DWORD *)v76 - 6)),
                v77 != 2) )
          {
            v104 = sub_A17150(v42);
            v141 = 1;
            v61 = (const char **)(v104 + 16);
            v62 = "invalid value for 'require' module flag (expected metadata pair)";
            goto LABEL_82;
          }
          v124 = v76;
          v126 = v76 - 16;
          v78 = sub_A17150(v76 - 16);
          v79 = v124;
          if ( **(_BYTE **)v78 )
          {
            v109 = (const char **)sub_A17150(v126);
            v141 = 1;
            v61 = v109;
            v62 = "invalid value for 'require' module flag (first value operand should be a string)";
            goto LABEL_82;
          }
          v80 = (unsigned int)v143;
          if ( (unsigned __int64)(unsigned int)v143 + 1 > HIDWORD(v143) )
          {
            sub_C8D5F0(&v142, v144, (unsigned int)v143 + 1LL, 8);
            v80 = (unsigned int)v143;
            v79 = v124;
          }
          v142[v80] = (__int64)v79;
          LODWORD(v143) = v143 + 1;
          if ( v132 == 3 )
            goto LABEL_126;
        }
        else if ( v132 - 5 <= 1 && (unsigned __int8)(**((_BYTE **)sub_A17150(v42) + 2) - 5) > 0x1Fu )
        {
          v63 = sub_A17150(v42);
          v141 = 1;
          v61 = (const char **)(v63 + 16);
          v62 = "invalid value for 'append'-type module flag (expected a metadata node)";
          goto LABEL_82;
        }
        v50 = v137;
        v138 = v49;
        v139 = v6;
        if ( !v137 )
        {
          ++v134;
          v133 = 0;
          goto LABEL_162;
        }
        v51 = 1;
        v52 = 0;
        v53 = (v137 - 1) & (((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4));
        v54 = (const char **)(v135 + 16LL * v53);
        v55 = *v54;
        if ( v49 != *v54 )
        {
          while ( v55 != (const char *)-4096LL )
          {
            if ( v55 != (const char *)-8192LL || v52 )
              v54 = v52;
            v53 = (v137 - 1) & (v51 + v53);
            v55 = *(const char **)(v135 + 16LL * v53);
            if ( v49 == v55 )
              goto LABEL_72;
            ++v51;
            v52 = v54;
            v54 = (const char **)(v135 + 16LL * v53);
          }
          if ( !v52 )
            v52 = v54;
          ++v134;
          v107 = v136 + 1;
          v133 = v52;
          if ( 4 * ((int)v136 + 1) >= 3 * v137 )
          {
LABEL_162:
            v50 = 2 * v137;
            goto LABEL_163;
          }
          v106 = v49;
          if ( v137 - HIDWORD(v136) - v107 <= v137 >> 3 )
          {
LABEL_163:
            sub_BF2A10((__int64)&v134, v50);
            sub_BF0430((__int64)&v134, (__int64 *)&v138, &v133);
            v106 = v138;
            v52 = v133;
            v107 = v136 + 1;
          }
          LODWORD(v136) = v107;
          if ( *v52 != (const char *)-4096LL )
            --HIDWORD(v136);
          *v52 = v106;
          v52[1] = (const char *)v139;
LABEL_126:
          v81 = (const void *)sub_B91420((__int64)v49);
          if ( !sub_9691B0(v81, v82, "wchar_size", 10)
            || (v83 = *((_QWORD *)sub_A17150(v42) + 2)) != 0 && *(_BYTE *)v83 == 1 && **(_BYTE **)(v83 + 136) == 17 )
          {
            v84 = (const void *)sub_B91420((__int64)v49);
            if ( sub_9691B0(v84, v85, "Linker Options", 14)
              && !sub_BA8DC0(*(_QWORD *)(v2 + 8), (__int64)"llvm.linker.options", 19) )
            {
              v141 = 1;
              v119 = "'Linker Options' named metadata no longer supported";
            }
            else
            {
              v86 = (const void *)sub_B91420((__int64)v49);
              if ( !sub_9691B0(v86, v87, "SemanticInterposition", 21)
                || (v88 = *((_QWORD *)sub_A17150(v42) + 2)) != 0 && *(_BYTE *)v88 == 1 && **(_BYTE **)(v88 + 136) == 17 )
              {
                v89 = (const void *)sub_B91420((__int64)v49);
                a2 = v90;
                if ( !sub_9691B0(v89, v90, "CG Profile", 10) )
                  goto LABEL_15;
                v91 = sub_A17150(v42);
                v93 = (const char **)sub_A17150((_BYTE *)(*((_QWORD *)v91 + 2) - 16LL));
                if ( v93 == &v93[v92] )
                  goto LABEL_15;
                a2 = (__int64)&v138;
                v94 = &v93[v92];
                v125 = v5;
                v95 = v2;
                v96 = v93;
                while ( 1 )
                {
                  v100 = *v96;
                  if ( *v96 && (unsigned __int8)(*v100 - 5) <= 0x1Fu )
                  {
                    if ( (*(v100 - 16) & 2) != 0 )
                    {
                      if ( *((_DWORD *)v100 - 6) == 3 )
                        goto LABEL_183;
                    }
                    else if ( ((*((_WORD *)v100 - 8) >> 6) & 0xF) == 3 )
                    {
LABEL_183:
                      v110 = (char *)(v100 - 16);
                      v111 = (__int64 *)sub_A17150(v110);
                      v112 = (unsigned __int8 **)*v111;
                      if ( *v111 )
                      {
                        if ( (unsigned int)*(unsigned __int8 *)v112 - 1 > 1
                          || (v120 = v111, v113 = sub_BD3990(v112[17], a2), v111 = v120, *v113) )
                        {
                          a2 = (__int64)&v138;
                          v123 = v111;
                          v141 = 1;
                          v138 = "expected a Function or null";
                          v140 = 3;
                          sub_BDBF70((__int64 *)v95, (__int64)&v138);
                          if ( *(_QWORD *)v95 )
                          {
                            a2 = *v123;
                            if ( *v123 )
                              sub_BD9900((__int64 *)v95, (const char *)a2);
                          }
                        }
                      }
                      v114 = sub_A17150(v110);
                      v115 = (unsigned __int8 **)*((_QWORD *)v114 + 1);
                      if ( v115 )
                      {
                        if ( (unsigned int)*(unsigned __int8 *)v115 - 1 > 1
                          || (v121 = v114, v116 = sub_BD3990(v115[17], a2), v114 = v121, *v116) )
                        {
                          a2 = (__int64)&v138;
                          v122 = v114;
                          v141 = 1;
                          v138 = "expected a Function or null";
                          v140 = 3;
                          sub_BDBF70((__int64 *)v95, (__int64)&v138);
                          if ( *(_QWORD *)v95 )
                          {
                            a2 = *((_QWORD *)v122 + 1);
                            if ( a2 )
                              sub_BD9900((__int64 *)v95, (const char *)a2);
                          }
                        }
                      }
                      v117 = *((_QWORD *)sub_A17150(v110) + 2);
                      if ( !v117
                        || *(_BYTE *)v117 != 1
                        || *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v117 + 136) + 8LL) + 8LL) != 12 )
                      {
                        v118 = (const char **)sub_A17150(v110);
                        a2 = (__int64)&v138;
                        v141 = 1;
                        v140 = 3;
                        v138 = "expected an integer constant";
                        sub_BE7680((_BYTE *)v95, (__int64)&v138, v118 + 2);
                      }
LABEL_145:
                      if ( v94 == ++v96 )
                        goto LABEL_152;
                      continue;
                    }
                  }
                  v101 = *(_QWORD *)v95;
                  v141 = 1;
                  v138 = "expected a MDNode triple";
                  v140 = 3;
                  if ( v101 )
                  {
                    sub_CA0E80(&v138, v101);
                    v97 = *(_BYTE **)(v101 + 32);
                    if ( (unsigned __int64)v97 >= *(_QWORD *)(v101 + 24) )
                    {
                      sub_CB5D20(v101, 10);
                    }
                    else
                    {
                      *(_QWORD *)(v101 + 32) = v97 + 1;
                      *v97 = 10;
                    }
                    a2 = *(_QWORD *)v95;
                    *(_BYTE *)(v95 + 152) = 1;
                    if ( a2 && *v96 )
                    {
                      sub_A62C00(*v96, a2, v95 + 16, *(_QWORD *)(v95 + 8));
                      v98 = *(_QWORD *)v95;
                      v99 = *(_BYTE **)(*(_QWORD *)v95 + 32LL);
                      if ( (unsigned __int64)v99 >= *(_QWORD *)(*(_QWORD *)v95 + 24LL) )
                      {
                        a2 = 10;
                        sub_CB5D20(v98, 10);
                      }
                      else
                      {
                        *(_QWORD *)(v98 + 32) = v99 + 1;
                        *v99 = 10;
                      }
                    }
                    goto LABEL_145;
                  }
                  ++v96;
                  *(_BYTE *)(v95 + 152) = 1;
                  if ( v94 == v96 )
                  {
LABEL_152:
                    v2 = v95;
                    v5 = v125;
                    goto LABEL_15;
                  }
                }
              }
              v141 = 1;
              v119 = "SemanticInterposition metadata requires constant integer argument";
            }
          }
          else
          {
            v141 = 1;
            v119 = "wchar_size metadata requires constant integer argument";
          }
          a2 = (__int64)&v138;
          v138 = v119;
          v140 = 3;
          sub_BDBF70((__int64 *)v2, (__int64)&v138);
          goto LABEL_15;
        }
LABEL_72:
        a2 = (__int64)&v138;
        v141 = 1;
        v138 = "module flag identifiers must be unique (or of 'require' type)";
        v140 = 3;
        sub_BDBF70((__int64 *)v2, (__int64)&v138);
        if ( *(_QWORD *)v2 )
        {
          a2 = (__int64)v49;
          sub_BD9900((__int64 *)v2, v49);
        }
      }
      else
      {
        v44 = *(_QWORD *)sub_A17150(v42);
        if ( !v44 || *(_BYTE *)v44 != 1 || **(_BYTE **)(v44 + 136) != 17 )
        {
          v71 = (const char **)sub_A17150(v42);
          v141 = 1;
          v61 = v71;
          v62 = "invalid behavior operand in module flag (expected constant integer)";
          goto LABEL_82;
        }
        v45 = (const char **)sub_A17150(v42);
        v46 = *(_QWORD *)v2;
        v141 = 1;
        v47 = v45;
        v140 = 3;
        v138 = "invalid behavior operand in module flag (unexpected constant)";
        if ( !v46 )
        {
LABEL_61:
          *(_BYTE *)(v2 + 152) = 1;
          goto LABEL_15;
        }
        sub_CA0E80(&v138, v46);
        v48 = *(_BYTE **)(v46 + 32);
        if ( (unsigned __int64)v48 >= *(_QWORD *)(v46 + 24) )
        {
          sub_CB5D20(v46, 10);
        }
        else
        {
          *(_QWORD *)(v46 + 32) = v48 + 1;
          *v48 = 10;
        }
        a2 = *(_QWORD *)v2;
        *(_BYTE *)(v2 + 152) = 1;
        if ( a2 )
        {
          v12 = *v47;
          if ( *v47 )
          {
            v10 = *(_QWORD *)(v2 + 8);
            v11 = v2 + 16;
LABEL_13:
            sub_A62C00(v12, a2, v11, v10);
            v13 = *(_QWORD *)v2;
            v14 = *(_BYTE **)(*(_QWORD *)v2 + 32LL);
            if ( (unsigned __int64)v14 >= *(_QWORD *)(*(_QWORD *)v2 + 24LL) )
            {
              a2 = 10;
              sub_CB5D20(v13, 10);
            }
            else
            {
              *(_QWORD *)(v13 + 32) = v14 + 1;
              *v14 = 10;
            }
          }
        }
      }
LABEL_15:
      v15 = *(_BYTE *)(v6 - 16);
      if ( (v15 & 2) != 0 )
      {
        if ( *(_DWORD *)(v6 - 24) != 3 )
          goto LABEL_5;
        v17 = *(_QWORD *)(v6 - 32);
        v16 = v6 - 16;
      }
      else
      {
        if ( ((*(_WORD *)(v6 - 16) >> 6) & 0xF) != 3 )
          goto LABEL_5;
        v16 = v6 - 16;
        v17 = v6 - 16 - 8LL * ((v15 >> 2) & 0xF);
      }
      v18 = *(_BYTE **)(v17 + 8);
      if ( v18 && !*v18 )
      {
        v19 = sub_B91420(*(_QWORD *)(v17 + 8));
        if ( v20 == 29
          && !(*(_QWORD *)v19 ^ 0x2D34366863726161LL | *(_QWORD *)(v19 + 8) ^ 0x747561702D666C65LL)
          && *(_QWORD *)(v19 + 16) == 0x616C702D69626168LL
          && *(_DWORD *)(v19 + 24) == 1919903348
          && *(_BYTE *)(v19 + 28) == 109 )
        {
          v72 = *(_BYTE *)(v6 - 16);
          if ( (v72 & 2) != 0 )
            v73 = *(_QWORD *)(v6 - 32);
          else
            v73 = v16 - 8LL * ((v72 >> 2) & 0xF);
          v74 = *(_QWORD *)(v73 + 16);
          if ( v74 )
          {
            if ( *(_BYTE *)v74 == 1 )
            {
              v75 = *(_QWORD *)(v74 + 136);
              if ( *(_BYTE *)v75 == 17 )
              {
                if ( *(_DWORD *)(v75 + 32) <= 0x40u )
                  v128 = *(_QWORD *)(v75 + 24);
                else
                  v128 = **(_QWORD **)(v75 + 24);
              }
            }
          }
          goto LABEL_5;
        }
        v21 = sub_B91420((__int64)v18);
        if ( v22 == 28
          && !(*(_QWORD *)v21 ^ 0x2D34366863726161LL | *(_QWORD *)(v21 + 8) ^ 0x747561702D666C65LL)
          && *(_QWORD *)(v21 + 16) == 0x7265762D69626168LL
          && *(_DWORD *)(v21 + 24) == 1852795251 )
        {
          v23 = *(_BYTE *)(v6 - 16);
          v24 = (v23 & 2) != 0 ? *(_QWORD *)(v6 - 32) : v16 - 8LL * ((v23 >> 2) & 0xF);
          v25 = *(_QWORD *)(v24 + 16);
          if ( v25 )
          {
            if ( *(_BYTE *)v25 == 1 )
            {
              v26 = *(_QWORD *)(v25 + 136);
              if ( *(_BYTE *)v26 == 17 )
                break;
            }
          }
        }
      }
LABEL_5:
      if ( v129 == ++v5 )
        goto LABEL_33;
    }
    if ( *(_DWORD *)(v26 + 32) <= 0x40u )
    {
      v127 = *(_QWORD *)(v26 + 24);
      goto LABEL_5;
    }
    ++v5;
    v127 = **(_QWORD **)(v26 + 24);
  }
  while ( v129 != v5 );
LABEL_33:
  if ( (v128 == -1) != (v127 == -1) )
  {
    v27 = *(_QWORD *)v2;
    v141 = 1;
    v138 = "either both or no 'aarch64-elf-pauthabi-platform' and 'aarch64-elf-pauthabi-version' module flags must be present";
    v140 = 3;
    if ( v27 )
    {
      a2 = v27;
      sub_CA0E80(&v138, v27);
      v28 = *(_BYTE **)(v27 + 32);
      if ( (unsigned __int64)v28 >= *(_QWORD *)(v27 + 24) )
      {
        a2 = 10;
        sub_CB5D20(v27, 10);
      }
      else
      {
        *(_QWORD *)(v27 + 32) = v28 + 1;
        *v28 = 10;
      }
    }
    *(_BYTE *)(v2 + 152) = 1;
  }
LABEL_38:
  v29 = v142;
  v30 = &v142[(unsigned int)v143];
  if ( v30 == v142 )
    goto LABEL_86;
  v31 = v142;
  v32 = (_BYTE *)v2;
  do
  {
    v39 = *v31;
    v40 = *(_BYTE *)(*v31 - 16);
    if ( (v40 & 2) != 0 )
      v33 = *(_QWORD *)(v39 - 32);
    else
      v33 = -16 - 8LL * ((v40 >> 2) & 0xF) + v39;
    v41 = *(const char **)v33;
    a2 = v137;
    if ( !v137 )
      goto LABEL_94;
    a2 = v137 - 1;
    v34 = a2 & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
    v35 = (const char **)(v135 + 16LL * v34);
    v36 = *v35;
    if ( *v35 != v41 )
    {
      v66 = 1;
      while ( v36 != (const char *)-4096LL )
      {
        v108 = v66 + 1;
        v34 = a2 & (v66 + v34);
        v35 = (const char **)(v135 + 16LL * v34);
        v36 = *v35;
        if ( *v35 == v41 )
          goto LABEL_43;
        v66 = v108;
      }
LABEL_94:
      v141 = 1;
      v65 = "invalid requirement on flag, flag is not present in module";
      goto LABEL_95;
    }
LABEL_43:
    v37 = v35[1];
    if ( !v37 )
      goto LABEL_94;
    v38 = *(v37 - 16);
    if ( (v38 & 2) != 0 )
    {
      if ( *(_QWORD *)(v33 + 8) == *(_QWORD *)(*((_QWORD *)v37 - 4) + 16LL) )
        goto LABEL_46;
    }
    else
    {
      v64 = 8LL * ((v38 >> 2) & 0xF);
      a2 = -16 - v64;
      if ( *(_QWORD *)(v33 + 8) == *(_QWORD *)&v37[-v64] )
        goto LABEL_46;
    }
    v141 = 1;
    v65 = "invalid requirement on flag, flag does not have the required value";
LABEL_95:
    v67 = *(_QWORD *)v32;
    v138 = v65;
    v140 = 3;
    if ( v67 )
    {
      v130 = v67;
      sub_CA0E80(&v138, v67);
      v68 = *(_BYTE **)(v130 + 32);
      if ( (unsigned __int64)v68 >= *(_QWORD *)(v130 + 24) )
      {
        sub_CB5D20(v130, 10);
      }
      else
      {
        *(_QWORD *)(v130 + 32) = v68 + 1;
        *v68 = 10;
      }
      a2 = *(_QWORD *)v32;
      v32[152] = 1;
      if ( v41 && a2 )
      {
        sub_A62C00(v41, a2, (__int64)(v32 + 16), *((_QWORD *)v32 + 1));
        v69 = *(_QWORD *)v32;
        v70 = *(_BYTE **)(*(_QWORD *)v32 + 32LL);
        if ( (unsigned __int64)v70 >= *(_QWORD *)(*(_QWORD *)v32 + 24LL) )
        {
          a2 = 10;
          sub_CB5D20(v69, 10);
        }
        else
        {
          *(_QWORD *)(v69 + 32) = v70 + 1;
          *v70 = 10;
        }
      }
    }
    else
    {
      v32[152] = 1;
    }
LABEL_46:
    ++v31;
  }
  while ( v30 != v31 );
  v29 = v142;
LABEL_86:
  if ( v29 != (__int64 *)v144 )
    _libc_free(v29, a2);
  return sub_C7D6A0(v135, 16LL * v137, 8);
}
