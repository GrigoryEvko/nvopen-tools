// Function: sub_26BF480
// Address: 0x26bf480
//
__int64 __fastcall sub_26BF480(char *a1, char *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // r14
  char *v5; // r12
  _BOOL8 v6; // rax
  __int64 v7; // r13
  unsigned __int64 v8; // rbx
  unsigned __int64 v9; // rax
  bool v10; // al
  __int64 v11; // r13
  _QWORD *v12; // rax
  _QWORD *v13; // r14
  _QWORD *v14; // r13
  _QWORD *v15; // r15
  char v16; // r14
  unsigned __int64 v17; // r12
  unsigned __int64 v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  unsigned int v22; // esi
  _QWORD *v23; // r12
  __int64 v24; // r13
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // r13
  unsigned int v28; // eax
  _QWORD *v29; // r9
  _QWORD *v30; // r12
  __int64 v31; // rbx
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // r15
  unsigned int v35; // eax
  __int64 v36; // r14
  __int64 v37; // r15
  int *v38; // r14
  size_t v39; // r12
  int *v40; // r14
  size_t v41; // r13
  _QWORD *v42; // r12
  _QWORD *v43; // r13
  unsigned __int64 v44; // r15
  unsigned __int64 v45; // rbx
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rax
  unsigned int v49; // ecx
  _QWORD *v50; // r15
  __int64 v51; // r12
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // r12
  unsigned int v55; // eax
  _QWORD *v56; // r13
  _QWORD *v57; // r15
  __int64 v58; // rbx
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // rbx
  unsigned int v62; // eax
  _QWORD *v63; // r9
  _QWORD *v64; // r15
  __int64 v65; // r12
  __int64 v66; // rax
  __int64 v67; // rdx
  __int64 v68; // r13
  unsigned int v69; // eax
  __int64 v70; // r14
  __int64 v71; // r13
  bool v72; // cf
  bool v73; // al
  __int64 v74; // rax
  __int64 v75; // rdx
  __int64 v76; // r15
  unsigned int v77; // eax
  _QWORD *v78; // r13
  __int64 v79; // r12
  __int64 v80; // rax
  __int64 v81; // rdx
  __int64 v82; // r12
  unsigned int v83; // eax
  _QWORD *v84; // rbx
  _QWORD *v85; // r13
  __int64 v86; // r15
  __int64 v87; // rax
  __int64 v88; // rdx
  __int64 v89; // r15
  unsigned int v90; // eax
  _QWORD *v91; // r9
  _QWORD *v92; // r13
  __int64 v93; // r12
  __int64 v94; // rax
  __int64 v95; // rdx
  __int64 v96; // rbx
  unsigned int v97; // eax
  __int64 v98; // r14
  __int64 v99; // rbx
  unsigned __int64 v100; // rbx
  int *v101; // r14
  size_t v102; // r12
  int *v103; // r14
  size_t v104; // r13
  __int64 v105; // rax
  __int64 v106; // rdx
  __int64 v107; // rax
  unsigned int v108; // esi
  _QWORD *v109; // r13
  __int64 v110; // r15
  __int64 v111; // rax
  __int64 v112; // rdx
  __int64 v113; // r15
  unsigned int v114; // eax
  _QWORD *v115; // r9
  _QWORD *v116; // r12
  __int64 v117; // r13
  __int64 v118; // rax
  __int64 v119; // rdx
  __int64 v120; // rbx
  unsigned int v121; // eax
  __int64 v122; // r14
  __int64 v123; // rbx
  __int64 v124; // rax
  __int64 v125; // r14
  unsigned __int64 v126; // rbx
  unsigned __int64 v127; // rax
  bool v128; // al
  __int64 v129; // r12
  __int64 i; // rbx
  char *v131; // rbx
  _QWORD *v132; // rcx
  __int64 v133; // r13
  size_t v134; // rbx
  _QWORD *v135; // rax
  _QWORD *v136; // rax
  _QWORD *v137; // rax
  size_t v138; // rbx
  __int64 v139; // [rsp+8h] [rbp-178h]
  char *v140; // [rsp+10h] [rbp-170h]
  __int64 v141; // [rsp+18h] [rbp-168h]
  _QWORD *v142; // [rsp+20h] [rbp-160h]
  __int64 v143; // [rsp+20h] [rbp-160h]
  _QWORD *v144; // [rsp+30h] [rbp-150h]
  _QWORD *v145; // [rsp+30h] [rbp-150h]
  unsigned __int64 v146; // [rsp+38h] [rbp-148h]
  _QWORD *v147; // [rsp+38h] [rbp-148h]
  unsigned __int64 v148; // [rsp+40h] [rbp-140h]
  _QWORD *v149; // [rsp+40h] [rbp-140h]
  _QWORD *v150; // [rsp+40h] [rbp-140h]
  _QWORD *v151; // [rsp+48h] [rbp-138h]
  _QWORD *v152; // [rsp+48h] [rbp-138h]
  _QWORD *v153; // [rsp+48h] [rbp-138h]
  unsigned __int64 v154; // [rsp+48h] [rbp-138h]
  unsigned __int64 v155; // [rsp+50h] [rbp-130h]
  _QWORD *v156; // [rsp+50h] [rbp-130h]
  _QWORD *v157; // [rsp+50h] [rbp-130h]
  _QWORD *v158; // [rsp+50h] [rbp-130h]
  _QWORD *v159; // [rsp+58h] [rbp-128h]
  char v160; // [rsp+58h] [rbp-128h]
  char v161; // [rsp+58h] [rbp-128h]
  _QWORD *v162; // [rsp+58h] [rbp-128h]
  _QWORD *v163; // [rsp+60h] [rbp-120h]
  _QWORD *v164; // [rsp+60h] [rbp-120h]
  _QWORD *v165; // [rsp+60h] [rbp-120h]
  _QWORD *v166; // [rsp+60h] [rbp-120h]
  char *v167; // [rsp+68h] [rbp-118h]
  _QWORD *v168; // [rsp+70h] [rbp-110h]
  _QWORD *v169; // [rsp+70h] [rbp-110h]
  _QWORD *v170; // [rsp+70h] [rbp-110h]
  _QWORD *v171; // [rsp+70h] [rbp-110h]
  char *v172; // [rsp+80h] [rbp-100h]
  char v173; // [rsp+88h] [rbp-F8h]
  unsigned __int64 v174; // [rsp+88h] [rbp-F8h]
  unsigned __int64 v175; // [rsp+88h] [rbp-F8h]
  char v176; // [rsp+88h] [rbp-F8h]
  char *v177; // [rsp+90h] [rbp-F0h]
  _QWORD *v178; // [rsp+98h] [rbp-E8h]
  char *v179; // [rsp+98h] [rbp-E8h]
  _QWORD *v180; // [rsp+98h] [rbp-E8h]
  _QWORD v181[2]; // [rsp+A0h] [rbp-E0h] BYREF
  int v182[52]; // [rsp+B0h] [rbp-D0h] BYREF

  result = a2 - a1;
  v139 = a3;
  v172 = a2;
  if ( a2 - a1 > 128 )
  {
    if ( a3 )
    {
      while ( 1 )
      {
        --v139;
        v4 = *((_QWORD *)a1 + 1);
        v5 = &a1[8 * ((__int64)(((v172 - a1) >> 3) + ((unsigned __int64)(v172 - a1) >> 63)) >> 1)];
        v6 = sub_EF9210((_QWORD *)v4);
        v7 = *(_QWORD *)v5;
        v8 = v6;
        v9 = sub_EF9210(*(_QWORD **)v5);
        if ( v8 == v9 )
        {
          v134 = sub_26BA4C0(*(int **)(v4 + 16), *(_QWORD *)(v4 + 24));
          v10 = v134 < sub_26BA4C0(*(int **)(v7 + 16), *(_QWORD *)(v7 + 24));
        }
        else
        {
          v10 = v8 > v9;
        }
        v11 = *((_QWORD *)v172 - 1);
        if ( !v10 )
          break;
        if ( !sub_26BE300(*(_QWORD **)v5, *((_QWORD **)v172 - 1)) )
        {
          if ( sub_26BE300(*((_QWORD **)a1 + 1), *((_QWORD **)v172 - 1)) )
          {
            v135 = *(_QWORD **)a1;
            *(_QWORD *)a1 = *((_QWORD *)v172 - 1);
            *((_QWORD *)v172 - 1) = v135;
            v13 = *(_QWORD **)a1;
            v14 = (_QWORD *)*((_QWORD *)a1 + 1);
            goto LABEL_8;
          }
          goto LABEL_257;
        }
        v12 = *(_QWORD **)a1;
        *(_QWORD *)a1 = *(_QWORD *)v5;
        *(_QWORD *)v5 = v12;
        v13 = *(_QWORD **)a1;
        v14 = (_QWORD *)*((_QWORD *)a1 + 1);
LABEL_8:
        v167 = a1 + 8;
        v177 = v172;
        v15 = v13;
        v16 = unk_4F838D3;
        while ( 1 )
        {
          v140 = v167;
          if ( v16 )
          {
            v17 = v14[8];
            if ( v17 )
              goto LABEL_11;
          }
          v105 = v14[20];
          if ( !v14[14] )
          {
            if ( !v105 )
              goto LABEL_224;
            v107 = v14[18];
LABEL_187:
            v180 = (_QWORD *)(v107 + 48);
            if ( *(_QWORD *)(v107 + 64) == v107 + 48 )
            {
LABEL_224:
              v17 = v14[7] != 0;
              goto LABEL_213;
            }
            v166 = v15;
            v17 = 0;
            v162 = v14;
            v109 = *(_QWORD **)(v107 + 64);
            while ( 2 )
            {
              if ( v16 )
              {
                v110 = v109[14];
                if ( v110 )
                {
LABEL_210:
                  v17 += v110;
                  v109 = (_QWORD *)sub_220EF30((__int64)v109);
                  if ( v180 == v109 )
                  {
                    v15 = v166;
                    v14 = v162;
                    goto LABEL_212;
                  }
                  continue;
                }
              }
              break;
            }
            v111 = v109[26];
            if ( v109[20] )
            {
              v112 = v109[18];
              if ( !v111
                || (v113 = v109[24], v114 = *(_DWORD *)(v113 + 32), *(_DWORD *)(v112 + 32) < v114)
                || *(_DWORD *)(v112 + 32) == v114 && *(_DWORD *)(v112 + 36) < *(_DWORD *)(v113 + 36) )
              {
                v110 = *(_QWORD *)(v112 + 40);
LABEL_209:
                if ( v110 )
                  goto LABEL_210;
LABEL_228:
                v110 = v109[13] != 0;
                goto LABEL_210;
              }
            }
            else
            {
              if ( !v111 )
                goto LABEL_228;
              v113 = v109[24];
            }
            v115 = *(_QWORD **)(v113 + 64);
            v171 = (_QWORD *)(v113 + 48);
            if ( v115 == (_QWORD *)(v113 + 48) )
              goto LABEL_228;
            v176 = v16;
            v110 = 0;
            v158 = v109;
            v154 = v17;
            v116 = v115;
            while ( 2 )
            {
              if ( v176 )
              {
                v117 = v116[14];
                if ( v117 )
                  goto LABEL_207;
              }
              v118 = v116[26];
              if ( v116[20] )
              {
                v119 = v116[18];
                if ( !v118
                  || (v120 = v116[24], v121 = *(_DWORD *)(v120 + 32), *(_DWORD *)(v119 + 32) < v121)
                  || *(_DWORD *)(v119 + 32) == v121 && *(_DWORD *)(v119 + 36) < *(_DWORD *)(v120 + 36) )
                {
                  v117 = *(_QWORD *)(v119 + 40);
                  goto LABEL_206;
                }
LABEL_203:
                v122 = *(_QWORD *)(v120 + 64);
                v123 = v120 + 48;
                if ( v122 != v123 )
                {
                  v117 = 0;
                  do
                  {
                    v117 += sub_EF9210((_QWORD *)(v122 + 48));
                    v122 = sub_220EF30(v122);
                  }
                  while ( v123 != v122 );
LABEL_206:
                  if ( v117 )
                  {
LABEL_207:
                    v110 += v117;
                    v116 = (_QWORD *)sub_220EF30((__int64)v116);
                    if ( v171 == v116 )
                    {
                      v16 = v176;
                      v109 = v158;
                      v17 = v154;
                      goto LABEL_209;
                    }
                    continue;
                  }
                }
              }
              else if ( v118 )
              {
                v120 = v116[24];
                goto LABEL_203;
              }
              break;
            }
            v117 = v116[13] != 0;
            goto LABEL_207;
          }
          v106 = v14[12];
          if ( v105 )
          {
            v107 = v14[18];
            v108 = *(_DWORD *)(v107 + 32);
            if ( *(_DWORD *)(v106 + 32) >= v108
              && (*(_DWORD *)(v106 + 32) != v108 || *(_DWORD *)(v106 + 36) >= *(_DWORD *)(v107 + 36)) )
            {
              goto LABEL_187;
            }
          }
          v17 = *(_QWORD *)(v106 + 40);
LABEL_212:
          if ( !v17 )
            goto LABEL_224;
LABEL_213:
          if ( !v16 )
          {
            v19 = v15[20];
            if ( !v15[14] )
              goto LABEL_215;
            goto LABEL_13;
          }
LABEL_11:
          v18 = v15[8];
          if ( v18 )
            goto LABEL_42;
          v19 = v15[20];
          if ( !v15[14] )
          {
LABEL_215:
            if ( !v19 )
              goto LABEL_216;
            v21 = v15[18];
LABEL_16:
            v178 = (_QWORD *)(v21 + 48);
            if ( *(_QWORD *)(v21 + 64) == v21 + 48 )
            {
LABEL_216:
              v18 = v15[7] != 0;
              goto LABEL_42;
            }
            v163 = v15;
            v18 = 0;
            v159 = v14;
            v155 = v17;
            v23 = *(_QWORD **)(v21 + 64);
            while ( 2 )
            {
              if ( v16 )
              {
                v24 = v23[14];
                if ( v24 )
                {
LABEL_39:
                  v18 += v24;
                  v23 = (_QWORD *)sub_220EF30((__int64)v23);
                  if ( v178 == v23 )
                  {
                    v15 = v163;
                    v14 = v159;
                    v17 = v155;
                    goto LABEL_41;
                  }
                  continue;
                }
              }
              break;
            }
            v25 = v23[26];
            if ( v23[20] )
            {
              v26 = v23[18];
              if ( !v25
                || (v27 = v23[24], v28 = *(_DWORD *)(v27 + 32), *(_DWORD *)(v26 + 32) < v28)
                || *(_DWORD *)(v26 + 32) == v28 && *(_DWORD *)(v26 + 36) < *(_DWORD *)(v27 + 36) )
              {
                v24 = *(_QWORD *)(v26 + 40);
LABEL_38:
                if ( v24 )
                  goto LABEL_39;
LABEL_226:
                v24 = v23[13] != 0;
                goto LABEL_39;
              }
            }
            else
            {
              if ( !v25 )
                goto LABEL_226;
              v27 = v23[24];
            }
            v29 = *(_QWORD **)(v27 + 64);
            v168 = (_QWORD *)(v27 + 48);
            if ( v29 == (_QWORD *)(v27 + 48) )
              goto LABEL_226;
            v173 = v16;
            v24 = 0;
            v148 = v18;
            v151 = v23;
            v30 = v29;
            while ( 2 )
            {
              if ( v173 )
              {
                v31 = v30[14];
                if ( v31 )
                  goto LABEL_36;
              }
              v32 = v30[26];
              if ( v30[20] )
              {
                v33 = v30[18];
                if ( !v32
                  || (v34 = v30[24], v35 = *(_DWORD *)(v34 + 32), *(_DWORD *)(v33 + 32) < v35)
                  || *(_DWORD *)(v33 + 32) == v35 && *(_DWORD *)(v33 + 36) < *(_DWORD *)(v34 + 36) )
                {
                  v31 = *(_QWORD *)(v33 + 40);
                  goto LABEL_35;
                }
LABEL_32:
                v36 = *(_QWORD *)(v34 + 64);
                v37 = v34 + 48;
                if ( v36 != v37 )
                {
                  v31 = 0;
                  do
                  {
                    v31 += sub_EF9210((_QWORD *)(v36 + 48));
                    v36 = sub_220EF30(v36);
                  }
                  while ( v37 != v36 );
LABEL_35:
                  if ( v31 )
                  {
LABEL_36:
                    v24 += v31;
                    v30 = (_QWORD *)sub_220EF30((__int64)v30);
                    if ( v168 == v30 )
                    {
                      v16 = v173;
                      v23 = v151;
                      v18 = v148;
                      goto LABEL_38;
                    }
                    continue;
                  }
                }
              }
              else if ( v32 )
              {
                v34 = v30[24];
                goto LABEL_32;
              }
              break;
            }
            v31 = v30[13] != 0;
            goto LABEL_36;
          }
LABEL_13:
          v20 = v15[12];
          if ( v19 )
          {
            v21 = v15[18];
            v22 = *(_DWORD *)(v21 + 32);
            if ( *(_DWORD *)(v20 + 32) >= v22
              && (*(_DWORD *)(v20 + 32) != v22 || *(_DWORD *)(v20 + 36) >= *(_DWORD *)(v21 + 36)) )
            {
              goto LABEL_16;
            }
          }
          v18 = *(_QWORD *)(v20 + 40);
LABEL_41:
          if ( !v18 )
            goto LABEL_216;
LABEL_42:
          if ( v18 != v17 )
            break;
          v38 = (int *)v14[2];
          v39 = v14[3];
          if ( v38 )
          {
            sub_C7D030(v182);
            sub_C7D280(v182, v38, v39);
            sub_C7D290(v182, v181);
            v39 = v181[0];
          }
          v40 = (int *)v15[2];
          v41 = v15[3];
          if ( v40 )
          {
            sub_C7D030(v182);
            sub_C7D280(v182, v40, v41);
            sub_C7D290(v182, v181);
            v41 = v181[0];
          }
          v15 = *(_QWORD **)a1;
          v16 = unk_4F838D3;
          if ( v41 <= v39 )
            goto LABEL_50;
LABEL_44:
          v14 = (_QWORD *)*((_QWORD *)v167 + 1);
          v167 += 8;
        }
        if ( v18 < v17 )
          goto LABEL_44;
LABEL_50:
        v42 = v15;
        v179 = v177 - 8;
        while ( 1 )
        {
          v177 = v179;
          v43 = *(_QWORD **)v179;
          if ( v16 )
          {
            v44 = v42[8];
            if ( v44 )
              goto LABEL_53;
          }
          v74 = v42[20];
          if ( !v42[14] )
          {
            if ( !v74 )
              goto LABEL_150;
            v76 = v42[18];
LABEL_103:
            v170 = (_QWORD *)(v76 + 48);
            if ( *(_QWORD *)(v76 + 64) == v76 + 48 )
            {
LABEL_150:
              v44 = v42[7] != 0;
              goto LABEL_140;
            }
            v150 = v42;
            v175 = 0;
            v153 = *(_QWORD **)v179;
            v78 = *(_QWORD **)(v76 + 64);
            while ( 2 )
            {
              if ( v16 )
              {
                v79 = v78[14];
                if ( v79 )
                {
LABEL_137:
                  v175 += v79;
                  v78 = (_QWORD *)sub_220EF30((__int64)v78);
                  if ( v170 == v78 )
                  {
                    v43 = v153;
                    v42 = v150;
                    v44 = v175;
                    goto LABEL_139;
                  }
                  continue;
                }
              }
              break;
            }
            v80 = v78[26];
            if ( v78[20] )
            {
              v81 = v78[18];
              if ( !v80
                || (v82 = v78[24], v83 = *(_DWORD *)(v82 + 32), *(_DWORD *)(v81 + 32) < v83)
                || *(_DWORD *)(v81 + 32) == v83 && *(_DWORD *)(v81 + 36) < *(_DWORD *)(v82 + 36) )
              {
                v79 = *(_QWORD *)(v81 + 40);
LABEL_136:
                if ( v79 )
                  goto LABEL_137;
LABEL_152:
                v79 = v78[13] != 0;
                goto LABEL_137;
              }
            }
            else
            {
              if ( !v80 )
                goto LABEL_152;
              v82 = v78[24];
            }
            v84 = *(_QWORD **)(v82 + 64);
            v165 = (_QWORD *)(v82 + 48);
            if ( v84 == (_QWORD *)(v82 + 48) )
              goto LABEL_152;
            v147 = v78;
            v79 = 0;
            v85 = v84;
            while ( 2 )
            {
              if ( v16 )
              {
                v86 = v85[14];
                if ( v86 )
                {
LABEL_134:
                  v79 += v86;
                  v85 = (_QWORD *)sub_220EF30((__int64)v85);
                  if ( v165 == v85 )
                  {
                    v78 = v147;
                    goto LABEL_136;
                  }
                  continue;
                }
              }
              break;
            }
            v87 = v85[26];
            if ( v85[20] )
            {
              v88 = v85[18];
              if ( !v87
                || (v89 = v85[24], v90 = *(_DWORD *)(v89 + 32), *(_DWORD *)(v88 + 32) < v90)
                || *(_DWORD *)(v88 + 32) == v90 && *(_DWORD *)(v88 + 36) < *(_DWORD *)(v89 + 36) )
              {
                v86 = *(_QWORD *)(v88 + 40);
LABEL_133:
                if ( v86 )
                  goto LABEL_134;
LABEL_158:
                v86 = v85[13] != 0;
                goto LABEL_134;
              }
            }
            else
            {
              if ( !v87 )
                goto LABEL_158;
              v89 = v85[24];
            }
            v91 = *(_QWORD **)(v89 + 64);
            v157 = (_QWORD *)(v89 + 48);
            if ( v91 == (_QWORD *)(v89 + 48) )
              goto LABEL_158;
            v161 = v16;
            v86 = 0;
            v143 = v79;
            v145 = v85;
            v92 = v91;
            while ( 2 )
            {
              if ( v161 )
              {
                v93 = v92[14];
                if ( v93 )
                  goto LABEL_131;
              }
              v94 = v92[26];
              if ( v92[20] )
              {
                v95 = v92[18];
                if ( !v94
                  || (v96 = v92[24], v97 = *(_DWORD *)(v96 + 32), *(_DWORD *)(v95 + 32) < v97)
                  || *(_DWORD *)(v95 + 32) == v97 && *(_DWORD *)(v95 + 36) < *(_DWORD *)(v96 + 36) )
                {
                  v93 = *(_QWORD *)(v95 + 40);
                  goto LABEL_130;
                }
LABEL_127:
                v98 = *(_QWORD *)(v96 + 64);
                v99 = v96 + 48;
                if ( v98 != v99 )
                {
                  v93 = 0;
                  do
                  {
                    v93 += sub_EF9210((_QWORD *)(v98 + 48));
                    v98 = sub_220EF30(v98);
                  }
                  while ( v99 != v98 );
LABEL_130:
                  if ( v93 )
                  {
LABEL_131:
                    v86 += v93;
                    v92 = (_QWORD *)sub_220EF30((__int64)v92);
                    if ( v157 == v92 )
                    {
                      v16 = v161;
                      v85 = v145;
                      v79 = v143;
                      goto LABEL_133;
                    }
                    continue;
                  }
                }
              }
              else if ( v94 )
              {
                v96 = v92[24];
                goto LABEL_127;
              }
              break;
            }
            v93 = v92[13] != 0;
            goto LABEL_131;
          }
          v75 = v42[12];
          if ( v74 )
          {
            v76 = v42[18];
            v77 = *(_DWORD *)(v76 + 32);
            if ( *(_DWORD *)(v75 + 32) >= v77
              && (*(_DWORD *)(v75 + 32) != v77 || *(_DWORD *)(v75 + 36) >= *(_DWORD *)(v76 + 36)) )
            {
              goto LABEL_103;
            }
          }
          v44 = *(_QWORD *)(v75 + 40);
LABEL_139:
          if ( !v44 )
            goto LABEL_150;
LABEL_140:
          if ( !v16 )
          {
            v46 = v43[20];
            if ( !v43[14] )
              goto LABEL_142;
            goto LABEL_55;
          }
LABEL_53:
          v45 = v43[8];
          if ( v45 )
            goto LABEL_95;
          v46 = v43[20];
          if ( !v43[14] )
          {
LABEL_142:
            if ( !v46 )
              goto LABEL_143;
            v48 = v43[18];
LABEL_58:
            v169 = (_QWORD *)(v48 + 48);
            if ( *(_QWORD *)(v48 + 64) == v48 + 48 )
              goto LABEL_143;
            v152 = v43;
            v149 = v42;
            v174 = 0;
            v146 = v44;
            v50 = *(_QWORD **)(v48 + 64);
            while ( 2 )
            {
              if ( v16 )
              {
                v51 = v50[14];
                if ( v51 )
                {
LABEL_92:
                  v174 += v51;
                  v50 = (_QWORD *)sub_220EF30((__int64)v50);
                  if ( v169 == v50 )
                  {
                    v43 = v152;
                    v42 = v149;
                    v45 = v174;
                    v44 = v146;
                    goto LABEL_94;
                  }
                  continue;
                }
              }
              break;
            }
            v52 = v50[26];
            if ( v50[20] )
            {
              v53 = v50[18];
              if ( !v52
                || (v54 = v50[24], v55 = *(_DWORD *)(v54 + 32), *(_DWORD *)(v53 + 32) < v55)
                || *(_DWORD *)(v53 + 32) == v55 && *(_DWORD *)(v53 + 36) < *(_DWORD *)(v54 + 36) )
              {
                v51 = *(_QWORD *)(v53 + 40);
LABEL_91:
                if ( v51 )
                  goto LABEL_92;
LABEL_154:
                v51 = v50[13] != 0;
                goto LABEL_92;
              }
            }
            else
            {
              if ( !v52 )
                goto LABEL_154;
              v54 = v50[24];
            }
            v56 = *(_QWORD **)(v54 + 64);
            v164 = (_QWORD *)(v54 + 48);
            if ( v56 == (_QWORD *)(v54 + 48) )
              goto LABEL_154;
            v144 = v50;
            v51 = 0;
            v57 = v56;
            while ( 2 )
            {
              if ( v16 )
              {
                v58 = v57[14];
                if ( v58 )
                {
LABEL_89:
                  v51 += v58;
                  v57 = (_QWORD *)sub_220EF30((__int64)v57);
                  if ( v164 == v57 )
                  {
                    v50 = v144;
                    goto LABEL_91;
                  }
                  continue;
                }
              }
              break;
            }
            v59 = v57[26];
            if ( v57[20] )
            {
              v60 = v57[18];
              if ( !v59
                || (v61 = v57[24], v62 = *(_DWORD *)(v61 + 32), *(_DWORD *)(v60 + 32) < v62)
                || *(_DWORD *)(v60 + 32) == v62 && *(_DWORD *)(v60 + 36) < *(_DWORD *)(v61 + 36) )
              {
                v58 = *(_QWORD *)(v60 + 40);
LABEL_88:
                if ( v58 )
                  goto LABEL_89;
LABEL_160:
                v58 = v57[13] != 0;
                goto LABEL_89;
              }
            }
            else
            {
              if ( !v59 )
                goto LABEL_160;
              v61 = v57[24];
            }
            v63 = *(_QWORD **)(v61 + 64);
            v156 = (_QWORD *)(v61 + 48);
            if ( v63 == (_QWORD *)(v61 + 48) )
              goto LABEL_160;
            v160 = v16;
            v58 = 0;
            v141 = v51;
            v142 = v57;
            v64 = v63;
            while ( 2 )
            {
              if ( v160 )
              {
                v65 = v64[14];
                if ( v65 )
                  goto LABEL_86;
              }
              v66 = v64[26];
              if ( v64[20] )
              {
                v67 = v64[18];
                if ( !v66
                  || (v68 = v64[24], v69 = *(_DWORD *)(v68 + 32), *(_DWORD *)(v67 + 32) < v69)
                  || *(_DWORD *)(v67 + 32) == v69 && *(_DWORD *)(v67 + 36) < *(_DWORD *)(v68 + 36) )
                {
                  v65 = *(_QWORD *)(v67 + 40);
                  goto LABEL_85;
                }
LABEL_82:
                v70 = *(_QWORD *)(v68 + 64);
                v71 = v68 + 48;
                if ( v70 != v71 )
                {
                  v65 = 0;
                  do
                  {
                    v65 += sub_EF9210((_QWORD *)(v70 + 48));
                    v70 = sub_220EF30(v70);
                  }
                  while ( v71 != v70 );
LABEL_85:
                  if ( v65 )
                  {
LABEL_86:
                    v58 += v65;
                    v64 = (_QWORD *)sub_220EF30((__int64)v64);
                    if ( v156 == v64 )
                    {
                      v16 = v160;
                      v57 = v142;
                      v51 = v141;
                      goto LABEL_88;
                    }
                    continue;
                  }
                }
              }
              else if ( v66 )
              {
                v68 = v64[24];
                goto LABEL_82;
              }
              break;
            }
            v65 = v64[13] != 0;
            goto LABEL_86;
          }
LABEL_55:
          v47 = v43[12];
          if ( v46 )
          {
            v48 = v43[18];
            v49 = *(_DWORD *)(v48 + 32);
            if ( *(_DWORD *)(v47 + 32) >= v49
              && (*(_DWORD *)(v47 + 32) != v49 || *(_DWORD *)(v47 + 36) >= *(_DWORD *)(v48 + 36)) )
            {
              goto LABEL_58;
            }
          }
          v45 = *(_QWORD *)(v47 + 40);
LABEL_94:
          if ( v45 )
          {
LABEL_95:
            v72 = v45 < v44;
            if ( v45 != v44 )
              goto LABEL_96;
            goto LABEL_144;
          }
LABEL_143:
          v100 = v43[7] != 0;
          v72 = v100 < v44;
          if ( v100 != v44 )
          {
LABEL_96:
            v73 = v72;
            goto LABEL_97;
          }
LABEL_144:
          v101 = (int *)v42[2];
          v102 = v42[3];
          if ( v101 )
          {
            sub_C7D030(v182);
            sub_C7D280(v182, v101, v102);
            sub_C7D290(v182, v181);
            v102 = v181[0];
          }
          v103 = (int *)v43[2];
          v104 = v43[3];
          if ( v103 )
          {
            sub_C7D030(v182);
            sub_C7D280(v182, v103, v104);
            sub_C7D290(v182, v181);
            v104 = v181[0];
          }
          v73 = v104 > v102;
LABEL_97:
          v179 -= 8;
          if ( !v73 )
            break;
          v42 = *(_QWORD **)a1;
          v16 = unk_4F838D3;
        }
        if ( v167 < v177 )
        {
          v124 = *(_QWORD *)v167;
          *(_QWORD *)v167 = *(_QWORD *)v177;
          *(_QWORD *)v177 = v124;
          v15 = *(_QWORD **)a1;
          v16 = unk_4F838D3;
          goto LABEL_44;
        }
        sub_26BF480(v167, v172, v139);
        result = v167 - a1;
        if ( v167 - a1 <= 128 )
          return result;
        if ( !v139 )
          goto LABEL_259;
        v172 = v167;
      }
      v125 = *((_QWORD *)a1 + 1);
      v126 = sub_EF9210((_QWORD *)v125);
      v127 = sub_EF9210((_QWORD *)v11);
      if ( v126 == v127 )
      {
        v138 = sub_26BA4C0(*(int **)(v125 + 16), *(_QWORD *)(v125 + 24));
        v128 = v138 < sub_26BA4C0(*(int **)(v11 + 16), *(_QWORD *)(v11 + 24));
      }
      else
      {
        v128 = v126 > v127;
      }
      if ( !v128 )
      {
        if ( sub_26BE300(*(_QWORD **)v5, *((_QWORD **)v172 - 1)) )
        {
          v136 = *(_QWORD **)a1;
          *(_QWORD *)a1 = *((_QWORD *)v172 - 1);
          *((_QWORD *)v172 - 1) = v136;
        }
        else
        {
          v137 = *(_QWORD **)a1;
          *(_QWORD *)a1 = *(_QWORD *)v5;
          *(_QWORD *)v5 = v137;
        }
        v13 = *(_QWORD **)a1;
        v14 = (_QWORD *)*((_QWORD *)a1 + 1);
        goto LABEL_8;
      }
LABEL_257:
      v14 = *(_QWORD **)a1;
      v13 = (_QWORD *)*((_QWORD *)a1 + 1);
      *((_QWORD *)a1 + 1) = *(_QWORD *)a1;
      *(_QWORD *)a1 = v13;
      goto LABEL_8;
    }
    v140 = a2;
LABEL_259:
    v129 = result >> 3;
    for ( i = ((result >> 3) - 2) >> 1; ; --i )
    {
      sub_26BEF90((__int64)a1, i, v129, *(_QWORD **)&a1[8 * i]);
      if ( !i )
        break;
    }
    v131 = v140 - 8;
    do
    {
      v132 = *(_QWORD **)v131;
      v133 = v131 - a1;
      v131 -= 8;
      *((_QWORD *)v131 + 1) = *(_QWORD *)a1;
      result = (__int64)sub_26BEF90((__int64)a1, 0, v133 >> 3, v132);
    }
    while ( v133 > 8 );
  }
  return result;
}
