// Function: sub_18A5D70
// Address: 0x18a5d70
//
__int64 __fastcall sub_18A5D70(__int64 *a1, char *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 *v4; // r14
  __int64 v5; // r12
  unsigned __int64 v6; // rbx
  unsigned __int64 v7; // rax
  __int64 v8; // r13
  unsigned __int64 v9; // rbx
  _QWORD *v10; // r15
  unsigned __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned int v14; // ecx
  _QWORD *v15; // r13
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rbx
  unsigned int v19; // eax
  _QWORD *v20; // r9
  __int64 v21; // rbx
  _QWORD *v22; // r14
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // r12
  unsigned int v26; // eax
  __int64 v27; // r13
  __int64 v28; // r12
  __int64 v29; // r15
  __int64 v30; // rax
  __int64 v31; // rdx
  unsigned int v32; // ebx
  _QWORD *v33; // r12
  unsigned __int64 v34; // r13
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rbx
  unsigned int v38; // eax
  _QWORD *v39; // r15
  _QWORD *v40; // rbx
  __int64 v41; // r12
  _QWORD *v42; // r13
  _QWORD *v43; // r15
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // r14
  unsigned int v47; // eax
  _QWORD *v48; // r9
  _QWORD *v49; // r14
  __int64 v50; // rbx
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // rax
  unsigned int v54; // ecx
  __int64 v55; // rax
  __int64 v56; // rax
  char *v57; // r14
  __int64 v58; // rax
  __int64 v59; // rdx
  unsigned int v60; // esi
  unsigned __int64 v61; // r13
  _QWORD *v62; // r15
  unsigned __int64 v63; // r12
  __int64 v64; // rax
  __int64 v65; // rdx
  __int64 v66; // rax
  unsigned int v67; // esi
  _QWORD *v68; // r13
  __int64 v69; // rax
  __int64 v70; // rdx
  __int64 v71; // rax
  unsigned int v72; // esi
  _QWORD *v73; // r15
  __int64 v74; // r14
  __int64 v75; // rax
  __int64 v76; // rdx
  __int64 v77; // rbx
  unsigned int v78; // eax
  _QWORD *v79; // r9
  __int64 v80; // rbx
  _QWORD *v81; // r15
  __int64 v82; // rax
  __int64 v83; // rdx
  __int64 v84; // r12
  unsigned int v85; // eax
  __int64 v86; // r14
  __int64 v87; // r12
  __int64 v88; // r13
  unsigned __int64 v89; // r12
  __int64 v90; // rax
  __int64 v91; // rdx
  unsigned int v92; // esi
  _QWORD *v93; // r15
  unsigned __int64 v94; // rbx
  __int64 v95; // rax
  __int64 v96; // rdx
  __int64 v97; // rax
  unsigned int v98; // esi
  _QWORD *v99; // r13
  __int64 v100; // rax
  __int64 v101; // rdx
  __int64 v102; // rax
  unsigned int v103; // esi
  _QWORD *v104; // r15
  __int64 v105; // r14
  __int64 v106; // rax
  __int64 v107; // rdx
  __int64 v108; // rbx
  unsigned int v109; // eax
  _QWORD *v110; // r9
  __int64 v111; // rbx
  _QWORD *v112; // r14
  __int64 v113; // rax
  __int64 v114; // rdx
  __int64 v115; // r12
  unsigned int v116; // eax
  __int64 v117; // r15
  __int64 v118; // r12
  __int64 v119; // r13
  _QWORD *v120; // rcx
  unsigned __int64 v121; // rbx
  __int64 v122; // r12
  __int64 i; // rbx
  char *v124; // rbx
  _QWORD *v125; // rcx
  __int64 v126; // r13
  unsigned __int64 v127; // rbx
  bool v128; // cc
  __int64 *v129; // rax
  unsigned __int64 v130; // rbx
  __int64 v131; // [rsp+8h] [rbp-E8h]
  char *v132; // [rsp+10h] [rbp-E0h]
  _QWORD *v133; // [rsp+18h] [rbp-D8h]
  _QWORD *v134; // [rsp+20h] [rbp-D0h]
  __int64 v135; // [rsp+20h] [rbp-D0h]
  __int64 v136; // [rsp+30h] [rbp-C0h]
  _QWORD *v137; // [rsp+30h] [rbp-C0h]
  _QWORD *v138; // [rsp+38h] [rbp-B8h]
  unsigned __int64 v139; // [rsp+38h] [rbp-B8h]
  _QWORD *v140; // [rsp+40h] [rbp-B0h]
  _QWORD *v141; // [rsp+40h] [rbp-B0h]
  unsigned __int64 v142; // [rsp+48h] [rbp-A8h]
  char *v143; // [rsp+48h] [rbp-A8h]
  char *v144; // [rsp+50h] [rbp-A0h]
  _QWORD *v145; // [rsp+50h] [rbp-A0h]
  _QWORD *v146; // [rsp+58h] [rbp-98h]
  _QWORD *v147; // [rsp+58h] [rbp-98h]
  unsigned __int64 v148; // [rsp+58h] [rbp-98h]
  char *v149; // [rsp+60h] [rbp-90h]
  _QWORD *v150; // [rsp+68h] [rbp-88h]
  unsigned __int64 v151; // [rsp+68h] [rbp-88h]
  _QWORD *v152; // [rsp+68h] [rbp-88h]
  _QWORD *v153; // [rsp+68h] [rbp-88h]
  unsigned __int64 v154; // [rsp+70h] [rbp-80h]
  _QWORD *v155; // [rsp+70h] [rbp-80h]
  _QWORD *v156; // [rsp+70h] [rbp-80h]
  _QWORD *v157; // [rsp+70h] [rbp-80h]
  _QWORD *v158; // [rsp+78h] [rbp-78h]
  _QWORD *v159; // [rsp+78h] [rbp-78h]
  __int64 v160; // [rsp+78h] [rbp-78h]
  __int64 v161; // [rsp+78h] [rbp-78h]
  _QWORD *v162; // [rsp+80h] [rbp-70h]
  _QWORD *v163; // [rsp+88h] [rbp-68h]
  unsigned __int64 v164; // [rsp+88h] [rbp-68h]
  _QWORD *v165; // [rsp+88h] [rbp-68h]
  _QWORD *v166; // [rsp+88h] [rbp-68h]
  _QWORD *v167; // [rsp+90h] [rbp-60h]
  _QWORD *v168; // [rsp+98h] [rbp-58h]
  _QWORD *v169; // [rsp+A0h] [rbp-50h]
  _QWORD *v170; // [rsp+A0h] [rbp-50h]
  _QWORD *v171; // [rsp+A0h] [rbp-50h]
  _QWORD *v172; // [rsp+A0h] [rbp-50h]
  __int64 v173; // [rsp+A8h] [rbp-48h]
  char *v174; // [rsp+B0h] [rbp-40h]
  __int64 v175; // [rsp+B8h] [rbp-38h]

  result = a2 - (char *)a1;
  v131 = a3;
  v132 = a2;
  if ( a2 - (char *)a1 <= 128 )
    return result;
  if ( !a3 )
  {
    v149 = a2;
    goto LABEL_220;
  }
  while ( 2 )
  {
    --v131;
    v4 = &a1[(__int64)(((v132 - (char *)a1) >> 3) + ((unsigned __int64)(v132 - (char *)a1) >> 63)) >> 1];
    v5 = *v4;
    v168 = (_QWORD *)a1[1];
    v6 = sub_18A58D0((__int64)v168);
    v7 = sub_18A58D0(v5);
    v8 = *((_QWORD *)v132 - 1);
    v162 = (_QWORD *)*a1;
    if ( v6 <= v7 )
    {
      v121 = sub_18A58D0((__int64)v168);
      if ( v121 > sub_18A58D0(v8) )
      {
        *a1 = (__int64)v168;
        a1[1] = (__int64)v162;
        v10 = (_QWORD *)*((_QWORD *)v132 - 1);
        goto LABEL_6;
      }
      v130 = sub_18A58D0(v5);
      v128 = v130 <= sub_18A58D0(v8);
      v129 = a1;
      if ( v128 )
      {
        *a1 = v5;
        *v4 = (__int64)v162;
        v168 = (_QWORD *)*a1;
        v162 = (_QWORD *)a1[1];
        v10 = (_QWORD *)*((_QWORD *)v132 - 1);
        goto LABEL_6;
      }
LABEL_227:
      v10 = v162;
      *v129 = v8;
      *((_QWORD *)v132 - 1) = v162;
      v168 = (_QWORD *)*v129;
      v162 = (_QWORD *)v129[1];
      goto LABEL_6;
    }
    v9 = sub_18A58D0(v5);
    if ( v9 <= sub_18A58D0(v8) )
    {
      v127 = sub_18A58D0((__int64)v168);
      v128 = v127 <= sub_18A58D0(v8);
      v129 = a1;
      if ( v128 )
      {
        *a1 = (__int64)v168;
        a1[1] = (__int64)v162;
        v10 = (_QWORD *)*((_QWORD *)v132 - 1);
        goto LABEL_6;
      }
      goto LABEL_227;
    }
    *a1 = v5;
    *v4 = (__int64)v162;
    v168 = (_QWORD *)*a1;
    v10 = (_QWORD *)*((_QWORD *)v132 - 1);
    v162 = (_QWORD *)a1[1];
LABEL_6:
    v173 = v168[15];
    v175 = v168[9];
    v167 = a1 + 2;
    v174 = v132;
    while ( 2 )
    {
      v11 = v162[15];
      v149 = (char *)(v167 - 1);
      if ( v162[9] )
      {
        v12 = v162[7];
        if ( !v11
          || (v13 = v162[13], v14 = *(_DWORD *)(v13 + 32), *(_DWORD *)(v12 + 32) < v14)
          || *(_DWORD *)(v12 + 32) == v14 && *(_DWORD *)(v12 + 36) < *(_DWORD *)(v13 + 36) )
        {
          v11 = *(_QWORD *)(v12 + 40);
          goto LABEL_31;
        }
LABEL_11:
        v15 = *(_QWORD **)(v13 + 64);
        v11 = 0;
        v169 = (_QWORD *)(v13 + 48);
        if ( v15 != (_QWORD *)(v13 + 48) )
        {
          v158 = v10;
          do
          {
            v16 = v15[23];
            if ( v15[17] )
            {
              v17 = v15[15];
              if ( v16 )
              {
                v18 = v15[21];
                v19 = *(_DWORD *)(v18 + 32);
                if ( *(_DWORD *)(v17 + 32) >= v19
                  && (*(_DWORD *)(v17 + 32) != v19 || *(_DWORD *)(v17 + 36) >= *(_DWORD *)(v18 + 36)) )
                {
                  goto LABEL_17;
                }
              }
              v11 += *(_QWORD *)(v17 + 40);
            }
            else if ( v16 )
            {
              v18 = v15[21];
LABEL_17:
              v20 = *(_QWORD **)(v18 + 64);
              v163 = (_QWORD *)(v18 + 48);
              if ( v20 == (_QWORD *)(v18 + 48) )
                goto LABEL_29;
              v150 = v15;
              v21 = 0;
              v154 = v11;
              v22 = v20;
              while ( 2 )
              {
                v23 = v22[23];
                if ( v22[17] )
                {
                  v24 = v22[15];
                  if ( !v23
                    || (v25 = v22[21], v26 = *(_DWORD *)(v25 + 32), *(_DWORD *)(v24 + 32) < v26)
                    || *(_DWORD *)(v24 + 32) == v26 && *(_DWORD *)(v24 + 36) < *(_DWORD *)(v25 + 36) )
                  {
                    v21 += *(_QWORD *)(v24 + 40);
                    goto LABEL_27;
                  }
                }
                else
                {
                  if ( !v23 )
                    goto LABEL_27;
                  v25 = v22[21];
                }
                v27 = *(_QWORD *)(v25 + 64);
                v28 = v25 + 48;
                if ( v27 != v28 )
                {
                  v29 = 0;
                  do
                  {
                    v29 += sub_18A58D0(v27 + 64);
                    v27 = sub_220EF30(v27);
                  }
                  while ( v28 != v27 );
                  v21 += v29;
                }
LABEL_27:
                v22 = (_QWORD *)sub_220EF30(v22);
                if ( v163 == v22 )
                {
                  v15 = v150;
                  v11 = v21 + v154;
                  break;
                }
                continue;
              }
            }
LABEL_29:
            v15 = (_QWORD *)sub_220EF30(v15);
          }
          while ( v169 != v15 );
          v10 = v158;
        }
        goto LABEL_31;
      }
      if ( v11 )
      {
        v13 = v162[13];
        goto LABEL_11;
      }
LABEL_31:
      if ( v175 )
      {
        v30 = v168[7];
        if ( v173 )
        {
          v31 = v168[13];
          v32 = *(_DWORD *)(v31 + 32);
          if ( *(_DWORD *)(v30 + 32) >= v32
            && (*(_DWORD *)(v30 + 32) != v32 || *(_DWORD *)(v30 + 36) >= *(_DWORD *)(v31 + 36)) )
          {
            goto LABEL_35;
          }
        }
        v34 = *(_QWORD *)(v30 + 40);
      }
      else
      {
        v34 = 0;
        if ( v173 )
        {
          v31 = v168[13];
LABEL_35:
          v33 = *(_QWORD **)(v31 + 64);
          v34 = 0;
          v170 = (_QWORD *)(v31 + 48);
          if ( v33 == (_QWORD *)(v31 + 48) )
            goto LABEL_60;
          v164 = v11;
          v159 = v10;
          while ( 2 )
          {
            v35 = v33[23];
            if ( v33[17] )
            {
              v36 = v33[15];
              if ( !v35
                || (v37 = v33[21], v38 = *(_DWORD *)(v37 + 32), *(_DWORD *)(v36 + 32) < v38)
                || *(_DWORD *)(v36 + 32) == v38 && *(_DWORD *)(v36 + 36) < *(_DWORD *)(v37 + 36) )
              {
                v34 += *(_QWORD *)(v36 + 40);
                goto LABEL_58;
              }
            }
            else
            {
              if ( !v35 )
                goto LABEL_58;
              v37 = v33[21];
            }
            v39 = *(_QWORD **)(v37 + 64);
            v40 = (_QWORD *)(v37 + 48);
            if ( v39 != v40 )
            {
              v155 = v33;
              v151 = v34;
              v41 = 0;
              v42 = v39;
              v43 = v40;
              do
              {
                v44 = v42[23];
                if ( v42[17] )
                {
                  v45 = v42[15];
                  if ( v44 )
                  {
                    v46 = v42[21];
                    v47 = *(_DWORD *)(v46 + 32);
                    if ( *(_DWORD *)(v45 + 32) >= v47
                      && (*(_DWORD *)(v45 + 32) != v47 || *(_DWORD *)(v45 + 36) >= *(_DWORD *)(v46 + 36)) )
                    {
                      goto LABEL_47;
                    }
                  }
                  v41 += *(_QWORD *)(v45 + 40);
                }
                else if ( v44 )
                {
                  v46 = v42[21];
LABEL_47:
                  v48 = *(_QWORD **)(v46 + 64);
                  v49 = (_QWORD *)(v46 + 48);
                  if ( v48 == v49 )
                    goto LABEL_56;
                  v50 = 0;
                  while ( 2 )
                  {
                    v51 = v48[23];
                    if ( v48[17] )
                    {
                      v52 = v48[15];
                      if ( !v51
                        || (v53 = v48[21], v54 = *(_DWORD *)(v53 + 32), *(_DWORD *)(v52 + 32) < v54)
                        || *(_DWORD *)(v52 + 32) == v54 && *(_DWORD *)(v52 + 36) < *(_DWORD *)(v53 + 36) )
                      {
                        v50 += *(_QWORD *)(v52 + 40);
LABEL_54:
                        v48 = (_QWORD *)sub_220EF30(v48);
                        if ( v49 == v48 )
                        {
                          v41 += v50;
                          goto LABEL_56;
                        }
                        continue;
                      }
                    }
                    else if ( !v51 )
                    {
                      goto LABEL_54;
                    }
                    break;
                  }
                  v146 = v48;
                  v55 = sub_18A5060((__int64)(v48 + 8));
                  v48 = v146;
                  v50 += v55;
                  goto LABEL_54;
                }
LABEL_56:
                v42 = (_QWORD *)sub_220EF30(v42);
              }
              while ( v43 != v42 );
              v56 = v41;
              v33 = v155;
              v34 = v56 + v151;
            }
LABEL_58:
            v33 = (_QWORD *)sub_220EF30(v33);
            if ( v170 == v33 )
            {
              v11 = v164;
              v10 = v159;
              break;
            }
            continue;
          }
        }
      }
LABEL_60:
      if ( v34 < v11 )
        goto LABEL_151;
      v57 = v174 - 8;
      v174 -= 8;
      if ( v175 )
      {
LABEL_62:
        v58 = v168[7];
        if ( !v173
          || (v59 = v168[13], v60 = *(_DWORD *)(v59 + 32), *(_DWORD *)(v58 + 32) < v60)
          || *(_DWORD *)(v58 + 32) == v60 && *(_DWORD *)(v58 + 36) < *(_DWORD *)(v59 + 36) )
        {
          v61 = *(_QWORD *)(v58 + 40);
          goto LABEL_101;
        }
        goto LABEL_65;
      }
      while ( 1 )
      {
        v61 = 0;
        if ( v173 )
        {
          v59 = v168[13];
LABEL_65:
          v61 = 0;
          v171 = (_QWORD *)(v59 + 48);
          if ( *(_QWORD *)(v59 + 64) != v59 + 48 )
          {
            v144 = v57;
            v147 = v10;
            v62 = *(_QWORD **)(v59 + 64);
            v63 = 0;
            do
            {
              v64 = v62[23];
              if ( v62[17] )
              {
                v65 = v62[15];
                if ( v64 )
                {
                  v66 = v62[21];
                  v67 = *(_DWORD *)(v66 + 32);
                  if ( *(_DWORD *)(v65 + 32) >= v67
                    && (*(_DWORD *)(v65 + 32) != v67 || *(_DWORD *)(v65 + 36) >= *(_DWORD *)(v66 + 36)) )
                  {
                    goto LABEL_71;
                  }
                }
                v63 += *(_QWORD *)(v65 + 40);
              }
              else if ( v64 )
              {
                v66 = v62[21];
LABEL_71:
                v68 = *(_QWORD **)(v66 + 64);
                v165 = (_QWORD *)(v66 + 48);
                if ( v68 == (_QWORD *)(v66 + 48) )
                  goto LABEL_99;
                v160 = 0;
                v142 = v63;
                v140 = v62;
                while ( 2 )
                {
                  v69 = v68[23];
                  if ( v68[17] )
                  {
                    v70 = v68[15];
                    if ( !v69
                      || (v71 = v68[21], v72 = *(_DWORD *)(v71 + 32), *(_DWORD *)(v70 + 32) < v72)
                      || *(_DWORD *)(v70 + 32) == v72 && *(_DWORD *)(v70 + 36) < *(_DWORD *)(v71 + 36) )
                    {
                      v160 += *(_QWORD *)(v70 + 40);
                      goto LABEL_97;
                    }
                  }
                  else
                  {
                    if ( !v69 )
                      goto LABEL_97;
                    v71 = v68[21];
                  }
                  v73 = *(_QWORD **)(v71 + 64);
                  v156 = (_QWORD *)(v71 + 48);
                  if ( v73 != (_QWORD *)(v71 + 48) )
                  {
                    v138 = v68;
                    v74 = 0;
                    do
                    {
                      v75 = v73[23];
                      if ( v73[17] )
                      {
                        v76 = v73[15];
                        if ( v75 )
                        {
                          v77 = v73[21];
                          v78 = *(_DWORD *)(v77 + 32);
                          if ( *(_DWORD *)(v76 + 32) >= v78
                            && (*(_DWORD *)(v76 + 32) != v78 || *(_DWORD *)(v76 + 36) >= *(_DWORD *)(v77 + 36)) )
                          {
                            goto LABEL_83;
                          }
                        }
                        v74 += *(_QWORD *)(v76 + 40);
                      }
                      else if ( v75 )
                      {
                        v77 = v73[21];
LABEL_83:
                        v79 = *(_QWORD **)(v77 + 64);
                        v152 = (_QWORD *)(v77 + 48);
                        if ( v79 == (_QWORD *)(v77 + 48) )
                          goto LABEL_95;
                        v136 = v74;
                        v80 = 0;
                        v134 = v73;
                        v81 = v79;
                        while ( 2 )
                        {
                          v82 = v81[23];
                          if ( v81[17] )
                          {
                            v83 = v81[15];
                            if ( !v82
                              || (v84 = v81[21], v85 = *(_DWORD *)(v84 + 32), *(_DWORD *)(v83 + 32) < v85)
                              || *(_DWORD *)(v83 + 32) == v85 && *(_DWORD *)(v83 + 36) < *(_DWORD *)(v84 + 36) )
                            {
                              v80 += *(_QWORD *)(v83 + 40);
                              goto LABEL_93;
                            }
                          }
                          else
                          {
                            if ( !v82 )
                              goto LABEL_93;
                            v84 = v81[21];
                          }
                          v86 = *(_QWORD *)(v84 + 64);
                          v87 = v84 + 48;
                          if ( v86 != v87 )
                          {
                            v88 = 0;
                            do
                            {
                              v88 += sub_18A58D0(v86 + 64);
                              v86 = sub_220EF30(v86);
                            }
                            while ( v87 != v86 );
                            v80 += v88;
                          }
LABEL_93:
                          v81 = (_QWORD *)sub_220EF30(v81);
                          if ( v152 == v81 )
                          {
                            v73 = v134;
                            v74 = v80 + v136;
                            break;
                          }
                          continue;
                        }
                      }
LABEL_95:
                      v73 = (_QWORD *)sub_220EF30(v73);
                    }
                    while ( v156 != v73 );
                    v160 += v74;
                    v68 = v138;
                  }
LABEL_97:
                  v68 = (_QWORD *)sub_220EF30(v68);
                  if ( v165 == v68 )
                  {
                    v62 = v140;
                    v63 = v160 + v142;
                    break;
                  }
                  continue;
                }
              }
LABEL_99:
              v62 = (_QWORD *)sub_220EF30(v62);
            }
            while ( v171 != v62 );
            v10 = v147;
            v57 = v144;
            v61 = v63;
          }
        }
LABEL_101:
        v89 = v10[15];
        if ( !v10[9] )
        {
          if ( v89 )
          {
            v91 = v10[13];
LABEL_105:
            v89 = 0;
            v172 = (_QWORD *)(v91 + 48);
            if ( *(_QWORD *)(v91 + 64) != v91 + 48 )
            {
              v148 = v61;
              v143 = v57;
              v145 = v10;
              v93 = *(_QWORD **)(v91 + 64);
              v94 = 0;
              do
              {
                v95 = v93[23];
                if ( v93[17] )
                {
                  v96 = v93[15];
                  if ( v95 )
                  {
                    v97 = v93[21];
                    v98 = *(_DWORD *)(v97 + 32);
                    if ( *(_DWORD *)(v96 + 32) >= v98
                      && (*(_DWORD *)(v96 + 32) != v98 || *(_DWORD *)(v96 + 36) >= *(_DWORD *)(v97 + 36)) )
                    {
                      goto LABEL_111;
                    }
                  }
                  v94 += *(_QWORD *)(v96 + 40);
                }
                else if ( v95 )
                {
                  v97 = v93[21];
LABEL_111:
                  v99 = *(_QWORD **)(v97 + 64);
                  v166 = (_QWORD *)(v97 + 48);
                  if ( v99 == (_QWORD *)(v97 + 48) )
                    goto LABEL_139;
                  v161 = 0;
                  v141 = v93;
                  v139 = v94;
                  while ( 2 )
                  {
                    v100 = v99[23];
                    if ( v99[17] )
                    {
                      v101 = v99[15];
                      if ( !v100
                        || (v102 = v99[21], v103 = *(_DWORD *)(v102 + 32), *(_DWORD *)(v101 + 32) < v103)
                        || *(_DWORD *)(v101 + 32) == v103 && *(_DWORD *)(v101 + 36) < *(_DWORD *)(v102 + 36) )
                      {
                        v161 += *(_QWORD *)(v101 + 40);
                        goto LABEL_137;
                      }
                    }
                    else
                    {
                      if ( !v100 )
                        goto LABEL_137;
                      v102 = v99[21];
                    }
                    v104 = *(_QWORD **)(v102 + 64);
                    v157 = (_QWORD *)(v102 + 48);
                    if ( v104 != (_QWORD *)(v102 + 48) )
                    {
                      v137 = v99;
                      v105 = 0;
                      do
                      {
                        v106 = v104[23];
                        if ( v104[17] )
                        {
                          v107 = v104[15];
                          if ( v106 )
                          {
                            v108 = v104[21];
                            v109 = *(_DWORD *)(v108 + 32);
                            if ( *(_DWORD *)(v107 + 32) >= v109
                              && (*(_DWORD *)(v107 + 32) != v109 || *(_DWORD *)(v107 + 36) >= *(_DWORD *)(v108 + 36)) )
                            {
                              goto LABEL_123;
                            }
                          }
                          v105 += *(_QWORD *)(v107 + 40);
                        }
                        else if ( v106 )
                        {
                          v108 = v104[21];
LABEL_123:
                          v110 = *(_QWORD **)(v108 + 64);
                          v153 = (_QWORD *)(v108 + 48);
                          if ( v110 == (_QWORD *)(v108 + 48) )
                            goto LABEL_135;
                          v133 = v104;
                          v111 = 0;
                          v135 = v105;
                          v112 = v110;
                          while ( 2 )
                          {
                            v113 = v112[23];
                            if ( v112[17] )
                            {
                              v114 = v112[15];
                              if ( !v113
                                || (v115 = v112[21], v116 = *(_DWORD *)(v115 + 32), *(_DWORD *)(v114 + 32) < v116)
                                || *(_DWORD *)(v114 + 32) == v116 && *(_DWORD *)(v114 + 36) < *(_DWORD *)(v115 + 36) )
                              {
                                v111 += *(_QWORD *)(v114 + 40);
                                goto LABEL_133;
                              }
                            }
                            else
                            {
                              if ( !v113 )
                                goto LABEL_133;
                              v115 = v112[21];
                            }
                            v117 = *(_QWORD *)(v115 + 64);
                            v118 = v115 + 48;
                            if ( v117 != v118 )
                            {
                              v119 = 0;
                              do
                              {
                                v119 += sub_18A58D0(v117 + 64);
                                v117 = sub_220EF30(v117);
                              }
                              while ( v118 != v117 );
                              v111 += v119;
                            }
LABEL_133:
                            v112 = (_QWORD *)sub_220EF30(v112);
                            if ( v153 == v112 )
                            {
                              v104 = v133;
                              v105 = v111 + v135;
                              break;
                            }
                            continue;
                          }
                        }
LABEL_135:
                        v104 = (_QWORD *)sub_220EF30(v104);
                      }
                      while ( v157 != v104 );
                      v161 += v105;
                      v99 = v137;
                    }
LABEL_137:
                    v99 = (_QWORD *)sub_220EF30(v99);
                    if ( v166 == v99 )
                    {
                      v93 = v141;
                      v94 = v161 + v139;
                      break;
                    }
                    continue;
                  }
                }
LABEL_139:
                v93 = (_QWORD *)sub_220EF30(v93);
              }
              while ( v172 != v93 );
              v61 = v148;
              v10 = v145;
              v89 = v94;
              v57 = v143;
            }
          }
          v57 -= 8;
          if ( v89 >= v61 )
            break;
          goto LABEL_142;
        }
        v90 = v10[7];
        if ( v89 )
        {
          v91 = v10[13];
          v92 = *(_DWORD *)(v91 + 32);
          if ( *(_DWORD *)(v90 + 32) >= v92
            && (*(_DWORD *)(v90 + 32) != v92 || *(_DWORD *)(v90 + 36) >= *(_DWORD *)(v91 + 36)) )
          {
            goto LABEL_105;
          }
        }
        v57 -= 8;
        if ( *(_QWORD *)(v90 + 40) >= v61 )
          break;
LABEL_142:
        v10 = *(_QWORD **)v57;
        v174 = v57;
        if ( v175 )
          goto LABEL_62;
      }
      if ( v174 > v149 )
      {
        *(v167 - 1) = v10;
        v10 = (_QWORD *)*((_QWORD *)v174 - 1);
        *(_QWORD *)v174 = v162;
        v168 = (_QWORD *)*a1;
        v175 = *(_QWORD *)(*a1 + 72);
        v173 = *(_QWORD *)(*a1 + 120);
LABEL_151:
        v120 = (_QWORD *)*v167++;
        v162 = v120;
        continue;
      }
      break;
    }
    sub_18A5D70(v149, v132, v131);
    result = v149 - (char *)a1;
    if ( v149 - (char *)a1 > 128 )
    {
      if ( v131 )
      {
        v132 = (char *)(v167 - 1);
        continue;
      }
LABEL_220:
      v122 = result >> 3;
      for ( i = ((result >> 3) - 2) >> 1; ; --i )
      {
        sub_18A5930((__int64)a1, i, v122, (_QWORD *)a1[i]);
        if ( !i )
          break;
      }
      v124 = v149 - 8;
      do
      {
        v125 = *(_QWORD **)v124;
        v126 = v124 - (char *)a1;
        v124 -= 8;
        *((_QWORD *)v124 + 1) = *a1;
        result = (__int64)sub_18A5930((__int64)a1, 0, v126 >> 3, v125);
      }
      while ( v126 > 8 );
    }
    return result;
  }
}
