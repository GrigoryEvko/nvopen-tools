// Function: sub_1CBD150
// Address: 0x1cbd150
//
__int64 __fastcall sub_1CBD150(_QWORD *a1, unsigned __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // rbx
  char v5; // al
  __int64 result; // rax
  _QWORD *v7; // rax
  _QWORD *v8; // rdx
  _QWORD *v9; // r13
  _BOOL4 v10; // r15d
  __int64 v11; // rax
  unsigned __int64 v12; // r13
  unsigned __int64 v13; // rdi
  _QWORD *v14; // r8
  unsigned __int64 v15; // r14
  _QWORD *v16; // rax
  _QWORD *v17; // rsi
  _QWORD *v18; // rax
  _QWORD *v19; // rax
  _QWORD *v20; // rdx
  _QWORD *v21; // r13
  _BOOL4 v22; // r15d
  __int64 v23; // rax
  unsigned __int64 v24; // r13
  unsigned __int64 v25; // rdi
  _QWORD *v26; // r8
  unsigned __int64 v27; // r14
  _QWORD *v28; // rax
  _QWORD *v29; // rsi
  _QWORD *v30; // rax
  _QWORD *v31; // rax
  _QWORD *v32; // rdx
  _QWORD *v33; // r13
  _BOOL4 v34; // r15d
  __int64 v35; // rax
  unsigned __int64 v36; // rdi
  _QWORD *v37; // rax
  _QWORD *v38; // rsi
  _QWORD *v39; // rax
  _QWORD *v40; // rdx
  _QWORD *v41; // r13
  _BOOL4 v42; // r15d
  __int64 v43; // rax
  unsigned __int64 v44; // rdi
  _QWORD *v45; // rax
  _QWORD *v46; // rsi
  _QWORD *v47; // rax
  _QWORD *v48; // rdx
  _QWORD *v49; // r13
  _BOOL4 v50; // r15d
  __int64 v51; // rax
  unsigned __int64 v52; // rdi
  _QWORD *v53; // rax
  _QWORD *v54; // rsi
  _QWORD *v55; // rax
  _QWORD *v56; // rdx
  _QWORD *v57; // r13
  _BOOL4 v58; // r15d
  __int64 v59; // rax
  _QWORD *v60; // rax
  _QWORD *v61; // rdx
  _QWORD *v62; // r13
  _BOOL4 v63; // r15d
  __int64 v64; // rax
  unsigned __int64 v65; // rdi
  _QWORD *v66; // r8
  unsigned __int64 v67; // r14
  _QWORD *v68; // rax
  _QWORD *v69; // rsi
  unsigned __int64 v70; // rdi
  _QWORD *v71; // rax
  _QWORD *v72; // rsi
  _QWORD *v73; // rax
  _QWORD *v74; // rdx
  _QWORD *v75; // r13
  _BOOL4 v76; // r8d
  __int64 v77; // rax
  _QWORD *v78; // rax
  _QWORD *v79; // rdx
  _QWORD *v80; // r13
  _BOOL4 v81; // r15d
  __int64 v82; // rax
  unsigned __int64 v83; // rdi
  _QWORD *v84; // r8
  unsigned __int64 v85; // r14
  _QWORD *v86; // rax
  _QWORD *v87; // rsi
  unsigned __int64 v88; // rdi
  _QWORD *v89; // rax
  _QWORD *v90; // rsi
  __int64 v91; // rax
  int v92; // eax
  unsigned int v93; // edx
  _QWORD *v94; // rax
  _QWORD *v95; // rdx
  _QWORD *v96; // r13
  _BOOL4 v97; // r8d
  __int64 v98; // rax
  _QWORD *v99; // rax
  _QWORD *v100; // rdx
  _QWORD *v101; // r13
  _BOOL4 v102; // r15d
  __int64 v103; // rax
  unsigned __int64 v104; // rdi
  _QWORD *v105; // r8
  unsigned __int64 v106; // r14
  _QWORD *v107; // rax
  _QWORD *v108; // rsi
  unsigned __int64 v109; // rdi
  _QWORD *v110; // rax
  _QWORD *v111; // rsi
  _QWORD *v112; // rax
  _QWORD *v113; // rax
  _QWORD *v114; // rdx
  _QWORD *v115; // r13
  _BOOL4 v116; // r15d
  __int64 v117; // rax
  unsigned __int64 v118; // rdi
  _QWORD *v119; // rax
  _QWORD *v120; // rsi
  _QWORD *v121; // rax
  _QWORD *v122; // rax
  _QWORD *v123; // rax
  _QWORD *v124; // rdx
  _QWORD *v125; // r13
  _BOOL4 v126; // r14d
  __int64 v127; // rax
  unsigned __int64 v128; // rdi
  _QWORD *v129; // rax
  _QWORD *v130; // rsi
  _QWORD *v131; // rax
  _QWORD *v132; // rdx
  _QWORD *v133; // r13
  _BOOL4 v134; // r14d
  __int64 v135; // rax
  unsigned __int64 v136; // rdi
  _QWORD *v137; // rax
  _QWORD *v138; // rsi
  _BOOL4 v139; // [rsp+0h] [rbp-40h]
  _BOOL4 v140; // [rsp+0h] [rbp-40h]
  unsigned __int64 v141[7]; // [rsp+8h] [rbp-38h] BYREF

  v4 = a2;
  v5 = *(_BYTE *)(a2 + 16);
  v141[0] = a2;
  if ( v5 == 55 )
  {
    result = 0;
    if ( a3 != *(_QWORD *)(a2 - 24) )
      return result;
    v113 = sub_1C444D0((__int64)(a1 + 1), v141);
    v115 = v114;
    if ( v114 )
    {
      v116 = 1;
      if ( !v113 && v114 != a1 + 2 )
        v116 = a2 < v114[4];
      v117 = sub_22077B0(40);
      *(_QWORD *)(v117 + 32) = v141[0];
      sub_220F040(v116, v117, v115, a1 + 2);
      ++a1[6];
      v4 = v141[0];
    }
    v24 = *(_QWORD *)(v4 + 40);
    v118 = a1[14];
    v26 = *(_QWORD **)(a1[13] + 8 * (v24 % v118));
    v27 = v24 % v118;
    if ( !v26 )
      goto LABEL_30;
    v119 = (_QWORD *)*v26;
    if ( v24 != *(_QWORD *)(*v26 + 8LL) )
    {
      while ( 1 )
      {
        v120 = (_QWORD *)*v119;
        if ( !*v119 )
          goto LABEL_30;
        v26 = v119;
        if ( v27 != v120[1] % v118 )
          goto LABEL_30;
        v119 = (_QWORD *)*v119;
        if ( v24 == v120[1] )
          goto LABEL_29;
      }
    }
    goto LABEL_29;
  }
  if ( v5 != 54 )
  {
    switch ( v5 )
    {
      case 'T':
        result = 0;
        if ( a3 != *(_QWORD *)(a2 - 72) )
          return result;
        v19 = sub_1C444D0((__int64)(a1 + 1), v141);
        v21 = v20;
        if ( v20 )
        {
          v22 = 1;
          if ( !v19 && v20 != a1 + 2 )
            v22 = a2 < v20[4];
          v23 = sub_22077B0(40);
          *(_QWORD *)(v23 + 32) = v141[0];
          sub_220F040(v22, v23, v21, a1 + 2);
          ++a1[6];
          v4 = v141[0];
        }
        v24 = *(_QWORD *)(v4 + 40);
        v25 = a1[14];
        v26 = *(_QWORD **)(a1[13] + 8 * (v24 % v25));
        v27 = v24 % v25;
        if ( !v26 )
          goto LABEL_30;
        v28 = (_QWORD *)*v26;
        if ( v24 != *(_QWORD *)(*v26 + 8LL) )
        {
          while ( 1 )
          {
            v29 = (_QWORD *)*v28;
            if ( !*v28 )
              goto LABEL_30;
            v26 = v28;
            if ( v27 != v29[1] % v25 )
              goto LABEL_30;
            v28 = (_QWORD *)*v28;
            if ( v24 == v29[1] )
              goto LABEL_29;
          }
        }
        goto LABEL_29;
      case 'S':
        result = 0;
        if ( a3 != *(_QWORD *)(a2 - 48) )
          return result;
        v31 = sub_1C444D0((__int64)(a1 + 7), v141);
        v33 = v32;
        if ( v32 )
        {
          v34 = 1;
          if ( !v31 && v32 != a1 + 8 )
            v34 = a2 < v32[4];
          v35 = sub_22077B0(40);
          *(_QWORD *)(v35 + 32) = v141[0];
          sub_220F040(v34, v35, v33, a1 + 8);
          ++a1[12];
          v4 = v141[0];
        }
        v12 = *(_QWORD *)(v4 + 40);
        v36 = a1[21];
        v14 = *(_QWORD **)(a1[20] + 8 * (v12 % v36));
        v15 = v12 % v36;
        if ( !v14 )
          goto LABEL_15;
        v37 = (_QWORD *)*v14;
        if ( v12 != *(_QWORD *)(*v14 + 8LL) )
        {
          while ( 1 )
          {
            v38 = (_QWORD *)*v37;
            if ( !*v37 )
              goto LABEL_15;
            v14 = v37;
            if ( v15 != v38[1] % v36 )
              goto LABEL_15;
            v37 = (_QWORD *)*v37;
            if ( v12 == v38[1] )
              goto LABEL_14;
          }
        }
        goto LABEL_14;
      case 'W':
        result = 0;
        if ( a3 != *(_QWORD *)(a2 - 48) )
          return result;
        v39 = sub_1C444D0((__int64)(a1 + 1), v141);
        v41 = v40;
        if ( v40 )
        {
          v42 = 1;
          if ( !v39 && v40 != a1 + 2 )
            v42 = a2 < v40[4];
          v43 = sub_22077B0(40);
          *(_QWORD *)(v43 + 32) = v141[0];
          sub_220F040(v42, v43, v41, a1 + 2);
          ++a1[6];
          v4 = v141[0];
        }
        v24 = *(_QWORD *)(v4 + 40);
        v44 = a1[14];
        v26 = *(_QWORD **)(a1[13] + 8 * (v24 % v44));
        v27 = v24 % v44;
        if ( !v26 )
          goto LABEL_30;
        v45 = (_QWORD *)*v26;
        if ( v24 != *(_QWORD *)(*v26 + 8LL) )
        {
          do
          {
            v46 = (_QWORD *)*v45;
            if ( !*v45 )
              goto LABEL_30;
            v26 = v45;
            if ( v27 != v46[1] % v44 )
              goto LABEL_30;
            v45 = (_QWORD *)*v45;
          }
          while ( v24 != v46[1] );
        }
LABEL_29:
        if ( !*v26 )
        {
LABEL_30:
          v30 = (_QWORD *)sub_22077B0(16);
          if ( v30 )
            *v30 = 0;
          v30[1] = v24;
          sub_1CBA730(a1 + 13, v27, v24, (__int64)v30, 1);
        }
        return 1;
      case 'V':
        result = 0;
        if ( *(_QWORD *)(a2 - 24) != a3 )
          return result;
        v47 = sub_1C444D0((__int64)(a1 + 7), v141);
        v49 = v48;
        if ( v48 )
        {
          v50 = 1;
          if ( !v47 && v48 != a1 + 8 )
            v50 = a2 < v48[4];
          v51 = sub_22077B0(40);
          *(_QWORD *)(v51 + 32) = v141[0];
          sub_220F040(v50, v51, v49, a1 + 8);
          ++a1[12];
          v4 = v141[0];
        }
        v12 = *(_QWORD *)(v4 + 40);
        v52 = a1[21];
        v14 = *(_QWORD **)(a1[20] + 8 * (v12 % v52));
        v15 = v12 % v52;
        if ( !v14 )
          goto LABEL_15;
        v53 = (_QWORD *)*v14;
        if ( v12 != *(_QWORD *)(*v14 + 8LL) )
        {
          do
          {
            v54 = (_QWORD *)*v53;
            if ( !*v53 )
              goto LABEL_15;
            v14 = v53;
            if ( v15 != v54[1] % v52 )
              goto LABEL_15;
            v53 = (_QWORD *)*v53;
          }
          while ( v12 != v54[1] );
        }
LABEL_14:
        if ( *v14 )
          return 1;
LABEL_15:
        v18 = (_QWORD *)sub_22077B0(16);
        if ( v18 )
          *v18 = 0;
        v18[1] = v12;
        sub_1CBA730(a1 + 20, v15, v12, (__int64)v18, 1);
        return 1;
      case ':':
        result = 0;
        if ( a3 != *(_QWORD *)(a2 - 72) )
          return result;
        v55 = sub_1C444D0((__int64)(a1 + 1), v141);
        v57 = v56;
        if ( v56 )
        {
          v58 = 1;
          if ( !v55 && v56 != a1 + 2 )
            v58 = a2 < v56[4];
          v59 = sub_22077B0(40);
          *(_QWORD *)(v59 + 32) = v141[0];
          sub_220F040(v58, v59, v57, a1 + 2);
          ++a1[6];
        }
        v60 = sub_1C444D0((__int64)(a1 + 7), v141);
        v62 = v61;
        if ( v61 )
        {
          v63 = 1;
          if ( !v60 && v61 != a1 + 8 )
            v63 = v141[0] < v61[4];
          v64 = sub_22077B0(40);
          *(_QWORD *)(v64 + 32) = v141[0];
          sub_220F040(v63, v64, v62, a1 + 8);
          ++a1[12];
        }
        v65 = a1[14];
        v12 = *(_QWORD *)(v141[0] + 40);
        v66 = *(_QWORD **)(a1[13] + 8 * (v12 % v65));
        v67 = v12 % v65;
        if ( v66 )
        {
          v68 = (_QWORD *)*v66;
          if ( v12 == *(_QWORD *)(*v66 + 8LL) )
          {
LABEL_79:
            if ( *v66 )
            {
LABEL_80:
              v70 = a1[21];
              v14 = *(_QWORD **)(a1[20] + 8 * (v12 % v70));
              v15 = v12 % v70;
              if ( !v14 )
                goto LABEL_15;
              v71 = (_QWORD *)*v14;
              if ( *(_QWORD *)(*v14 + 8LL) != v12 )
              {
                while ( 1 )
                {
                  v72 = (_QWORD *)*v71;
                  if ( !*v71 )
                    goto LABEL_15;
                  v14 = v71;
                  if ( v15 != v72[1] % v70 )
                    goto LABEL_15;
                  v71 = (_QWORD *)*v71;
                  if ( v72[1] == v12 )
                    goto LABEL_14;
                }
              }
              goto LABEL_14;
            }
          }
          else
          {
            while ( 1 )
            {
              v69 = (_QWORD *)*v68;
              if ( !*v68 )
                break;
              v66 = v68;
              if ( v67 != v69[1] % v65 )
                break;
              v68 = (_QWORD *)*v68;
              if ( v12 == v69[1] )
                goto LABEL_79;
            }
          }
        }
        v121 = (_QWORD *)sub_22077B0(16);
        if ( v121 )
          *v121 = 0;
        v121[1] = v12;
        sub_1CBA730(a1 + 13, v67, v12, (__int64)v121, 1);
        v12 = *(_QWORD *)(v141[0] + 40);
        goto LABEL_80;
      case ';':
        result = 0;
        if ( a3 != *(_QWORD *)(a2 - 48) )
          return result;
        v73 = sub_1C444D0((__int64)(a1 + 1), v141);
        v75 = v74;
        if ( v74 )
        {
          v76 = 1;
          if ( !v73 && v74 != a1 + 2 )
            v76 = a2 < v74[4];
          v139 = v76;
          v77 = sub_22077B0(40);
          *(_QWORD *)(v77 + 32) = v141[0];
          sub_220F040(v139, v77, v75, a1 + 2);
          ++a1[6];
        }
        v78 = sub_1C444D0((__int64)(a1 + 7), v141);
        v80 = v79;
        if ( v79 )
        {
          v81 = 1;
          if ( !v78 && v79 != a1 + 8 )
            v81 = v141[0] < v79[4];
          v82 = sub_22077B0(40);
          *(_QWORD *)(v82 + 32) = v141[0];
          sub_220F040(v81, v82, v80, a1 + 8);
          ++a1[12];
        }
        v83 = a1[14];
        v12 = *(_QWORD *)(v141[0] + 40);
        v84 = *(_QWORD **)(a1[13] + 8 * (v12 % v83));
        v85 = v12 % v83;
        if ( v84 )
        {
          v86 = (_QWORD *)*v84;
          if ( v12 == *(_QWORD *)(*v84 + 8LL) )
          {
LABEL_99:
            if ( *v84 )
            {
LABEL_100:
              v88 = a1[21];
              v14 = *(_QWORD **)(a1[20] + 8 * (v12 % v88));
              v15 = v12 % v88;
              if ( !v14 )
                goto LABEL_15;
              v89 = (_QWORD *)*v14;
              if ( *(_QWORD *)(*v14 + 8LL) != v12 )
              {
                while ( 1 )
                {
                  v90 = (_QWORD *)*v89;
                  if ( !*v89 )
                    goto LABEL_15;
                  v14 = v89;
                  if ( v15 != v90[1] % v88 )
                    goto LABEL_15;
                  v89 = (_QWORD *)*v89;
                  if ( v90[1] == v12 )
                    goto LABEL_14;
                }
              }
              goto LABEL_14;
            }
          }
          else
          {
            while ( 1 )
            {
              v87 = (_QWORD *)*v86;
              if ( !*v86 )
                break;
              v84 = v86;
              if ( v85 != v87[1] % v83 )
                break;
              v86 = (_QWORD *)*v86;
              if ( v12 == v87[1] )
                goto LABEL_99;
            }
          }
        }
        v122 = (_QWORD *)sub_22077B0(16);
        if ( v122 )
          *v122 = 0;
        v122[1] = v12;
        sub_1CBA730(a1 + 13, v85, v12, (__int64)v122, 1);
        v12 = *(_QWORD *)(v141[0] + 40);
        goto LABEL_100;
    }
    if ( v5 != 78 )
      return 0;
    v91 = *(_QWORD *)(a2 - 24);
    if ( !*(_BYTE *)(v91 + 16) && (*(_BYTE *)(v91 + 33) & 0x20) != 0 )
    {
      v92 = *(_DWORD *)(v91 + 36);
      if ( (v92 & 0xFFFFFFFD) == 0x85 || v92 == 137 )
      {
        if ( a3 == *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)) )
        {
          v123 = sub_1C444D0((__int64)(a1 + 1), v141);
          v125 = v124;
          if ( v124 )
          {
            v126 = 1;
            if ( !v123 && a1 + 2 != v124 )
              v126 = a2 < v124[4];
            v127 = sub_22077B0(40);
            *(_QWORD *)(v127 + 32) = v141[0];
            sub_220F040(v126, v127, v125, a1 + 2);
            ++a1[6];
            v4 = v141[0];
          }
          v24 = *(_QWORD *)(v4 + 40);
          v128 = a1[14];
          v26 = *(_QWORD **)(a1[13] + 8 * (v24 % v128));
          v27 = v24 % v128;
          if ( !v26 )
            goto LABEL_30;
          v129 = (_QWORD *)*v26;
          if ( v24 != *(_QWORD *)(*v26 + 8LL) )
          {
            while ( 1 )
            {
              v130 = (_QWORD *)*v129;
              if ( !*v129 )
                goto LABEL_30;
              v26 = v129;
              if ( v27 != v130[1] % v128 )
                goto LABEL_30;
              v129 = (_QWORD *)*v129;
              if ( v24 == v130[1] )
                goto LABEL_29;
            }
          }
          goto LABEL_29;
        }
        if ( (v92 & 0xFFFFFFFD) == 0x85 && a3 == *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))) )
        {
          v131 = sub_1C444D0((__int64)(a1 + 7), v141);
          v133 = v132;
          if ( v132 )
          {
            v134 = 1;
            if ( !v131 && v132 != a1 + 8 )
              v134 = a2 < v132[4];
            v135 = sub_22077B0(40);
            *(_QWORD *)(v135 + 32) = v141[0];
            sub_220F040(v134, v135, v133, a1 + 8);
            ++a1[12];
            v4 = v141[0];
          }
          v12 = *(_QWORD *)(v4 + 40);
          v136 = a1[21];
          v14 = *(_QWORD **)(a1[20] + 8 * (v12 % v136));
          v15 = v12 % v136;
          if ( !v14 )
            goto LABEL_15;
          v137 = (_QWORD *)*v14;
          if ( v12 != *(_QWORD *)(*v14 + 8LL) )
          {
            while ( 1 )
            {
              v138 = (_QWORD *)*v137;
              if ( !*v137 )
                goto LABEL_15;
              v14 = v137;
              if ( v15 != v138[1] % v136 )
                goto LABEL_15;
              v137 = (_QWORD *)*v137;
              if ( v12 == v138[1] )
                goto LABEL_14;
            }
          }
          goto LABEL_14;
        }
      }
      v93 = v92 - 116;
      result = 0;
      if ( v93 <= 1 )
        return result;
    }
    v94 = sub_1C444D0((__int64)(a1 + 7), v141);
    v96 = v95;
    if ( v95 )
    {
      v97 = 1;
      if ( !v94 && v95 != a1 + 8 )
        v97 = a2 < v95[4];
      v140 = v97;
      v98 = sub_22077B0(40);
      *(_QWORD *)(v98 + 32) = v141[0];
      sub_220F040(v140, v98, v96, a1 + 8);
      ++a1[12];
    }
    v99 = sub_1C444D0((__int64)(a1 + 1), v141);
    v101 = v100;
    if ( v100 )
    {
      v102 = 1;
      if ( !v99 && v100 != a1 + 2 )
        v102 = v141[0] < v100[4];
      v103 = sub_22077B0(40);
      *(_QWORD *)(v103 + 32) = v141[0];
      sub_220F040(v102, v103, v101, a1 + 2);
      ++a1[6];
    }
    v104 = a1[14];
    v12 = *(_QWORD *)(v141[0] + 40);
    v105 = *(_QWORD **)(a1[13] + 8 * (v12 % v104));
    v106 = v12 % v104;
    if ( v105 )
    {
      v107 = (_QWORD *)*v105;
      if ( v12 == *(_QWORD *)(*v105 + 8LL) )
      {
LABEL_126:
        if ( *v105 )
        {
LABEL_127:
          v109 = a1[21];
          v14 = *(_QWORD **)(a1[20] + 8 * (v12 % v109));
          v15 = v12 % v109;
          if ( !v14 )
            goto LABEL_15;
          v110 = (_QWORD *)*v14;
          if ( *(_QWORD *)(*v14 + 8LL) != v12 )
          {
            while ( 1 )
            {
              v111 = (_QWORD *)*v110;
              if ( !*v110 )
                goto LABEL_15;
              v14 = v110;
              if ( v15 != v111[1] % v109 )
                goto LABEL_15;
              v110 = (_QWORD *)*v110;
              if ( v111[1] == v12 )
                goto LABEL_14;
            }
          }
          goto LABEL_14;
        }
      }
      else
      {
        while ( 1 )
        {
          v108 = (_QWORD *)*v107;
          if ( !*v107 )
            break;
          v105 = v107;
          if ( v106 != v108[1] % v104 )
            break;
          v107 = (_QWORD *)*v107;
          if ( v12 == v108[1] )
            goto LABEL_126;
        }
      }
    }
    v112 = (_QWORD *)sub_22077B0(16);
    if ( v112 )
      *v112 = 0;
    v112[1] = v12;
    sub_1CBA730(a1 + 13, v106, v12, (__int64)v112, 1);
    v12 = *(_QWORD *)(v141[0] + 40);
    goto LABEL_127;
  }
  result = 0;
  if ( a3 == *(_QWORD *)(a2 - 24) )
  {
    v7 = sub_1C444D0((__int64)(a1 + 7), v141);
    v9 = v8;
    if ( v8 )
    {
      v10 = 1;
      if ( !v7 && v8 != a1 + 8 )
        v10 = a2 < v8[4];
      v11 = sub_22077B0(40);
      *(_QWORD *)(v11 + 32) = v141[0];
      sub_220F040(v10, v11, v9, a1 + 8);
      ++a1[12];
      v4 = v141[0];
    }
    v12 = *(_QWORD *)(v4 + 40);
    v13 = a1[21];
    v14 = *(_QWORD **)(a1[20] + 8 * (v12 % v13));
    v15 = v12 % v13;
    if ( !v14 )
      goto LABEL_15;
    v16 = (_QWORD *)*v14;
    if ( v12 != *(_QWORD *)(*v14 + 8LL) )
    {
      while ( 1 )
      {
        v17 = (_QWORD *)*v16;
        if ( !*v16 )
          goto LABEL_15;
        v14 = v16;
        if ( v15 != v17[1] % v13 )
          goto LABEL_15;
        v16 = (_QWORD *)*v16;
        if ( v12 == v17[1] )
          goto LABEL_14;
      }
    }
    goto LABEL_14;
  }
  return result;
}
