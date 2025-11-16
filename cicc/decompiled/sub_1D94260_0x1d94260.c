// Function: sub_1D94260
// Address: 0x1d94260
//
__int64 __fastcall sub_1D94260(
        __int64 *a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        char *a6,
        __int64 a7,
        __int64 a8)
{
  char *v8; // r13
  __int64 *v9; // r12
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 *v12; // rax
  __int64 v13; // r10
  __int64 v14; // r11
  __int64 *v15; // r8
  __int64 v16; // r15
  __int64 v17; // r11
  int v18; // ecx
  __int64 result; // rax
  __int64 v20; // rbx
  __int64 *v21; // r15
  __int64 *v22; // r14
  __int64 v23; // rdx
  __int64 v24; // rdi
  __int64 v25; // rsi
  __int64 *v26; // r14
  __int64 v27; // rbx
  __int64 *i; // r15
  __int64 v29; // rdi
  __int64 v30; // rdx
  unsigned int v31; // r8d
  int v32; // esi
  unsigned int v33; // edi
  int v34; // ecx
  __int64 v35; // rdi
  __int64 v36; // rbx
  __int64 *v37; // r14
  __int64 v38; // r12
  __int64 v39; // rdi
  __int64 *v40; // r15
  __int64 *v41; // r14
  __int64 v42; // rbx
  __int64 v43; // rdx
  __int64 v44; // rdi
  __int64 v45; // rcx
  char *v46; // rbx
  __int64 *v47; // r14
  __int64 v48; // rdi
  __int64 v49; // rdx
  unsigned int v50; // r8d
  int v51; // esi
  unsigned int v52; // edi
  int v53; // ecx
  __int64 v54; // rdi
  unsigned __int8 v55; // cl
  __int64 *v56; // r12
  __int64 *v57; // rbx
  __int64 v58; // r13
  __int64 v59; // rsi
  __int64 v60; // rdi
  __int64 v61; // rax
  __int64 v62; // r13
  __int64 *v63; // r12
  __int64 *v64; // rbx
  __int64 v65; // rcx
  __int64 v66; // rdi
  char *v67; // rcx
  __int64 v68; // r12
  __int64 *v69; // rbx
  char *v70; // r13
  __int64 v71; // rsi
  __int64 v72; // rdi
  __int64 v73; // r11
  __int64 *v74; // rax
  __int64 v75; // rax
  char *v76; // r12
  __int64 *v77; // rbx
  __int64 v78; // r13
  __int64 v79; // rax
  __int64 v80; // rdi
  __int64 v81; // rdx
  __int64 v82; // rdx
  __int64 v83; // r13
  char *v84; // r12
  __int64 v85; // rbx
  __int64 v86; // rcx
  __int64 v87; // rdi
  __int64 *v88; // rbx
  __int64 *v89; // r12
  __int64 v90; // r13
  __int64 v91; // rsi
  __int64 v92; // rdi
  __int64 v93; // r11
  unsigned __int8 v94; // cl
  __int64 v95; // r12
  __int64 v96; // rbx
  __int64 *v97; // r13
  __int64 v98; // rdi
  __int64 v99; // r12
  __int64 v100; // rbx
  __int64 v101; // r13
  __int64 v102; // rdi
  char *v103; // [rsp+0h] [rbp-A0h]
  char *v104; // [rsp+8h] [rbp-98h]
  char *v105; // [rsp+8h] [rbp-98h]
  __int64 v106; // [rsp+8h] [rbp-98h]
  __int64 v107; // [rsp+10h] [rbp-90h]
  __int64 v108; // [rsp+10h] [rbp-90h]
  __int64 v109; // [rsp+10h] [rbp-90h]
  int v110; // [rsp+18h] [rbp-88h]
  char *v111; // [rsp+18h] [rbp-88h]
  char *v112; // [rsp+18h] [rbp-88h]
  int v113; // [rsp+18h] [rbp-88h]
  char *v114; // [rsp+18h] [rbp-88h]
  __int64 v115; // [rsp+20h] [rbp-80h]
  __int64 v116; // [rsp+20h] [rbp-80h]
  __int64 v117; // [rsp+20h] [rbp-80h]
  __int64 v118; // [rsp+20h] [rbp-80h]
  char *v119; // [rsp+20h] [rbp-80h]
  __int64 *v120; // [rsp+28h] [rbp-78h]
  __int64 v121; // [rsp+28h] [rbp-78h]
  int v122; // [rsp+28h] [rbp-78h]
  __int64 v123; // [rsp+28h] [rbp-78h]
  __int64 v124; // [rsp+28h] [rbp-78h]
  __int64 v125; // [rsp+28h] [rbp-78h]
  __int64 v126; // [rsp+30h] [rbp-70h]
  int v127; // [rsp+30h] [rbp-70h]
  __int64 v128; // [rsp+30h] [rbp-70h]
  __int64 v129; // [rsp+30h] [rbp-70h]
  int v130; // [rsp+30h] [rbp-70h]
  int v131; // [rsp+30h] [rbp-70h]
  __int64 v132; // [rsp+38h] [rbp-68h]
  __int64 v133; // [rsp+38h] [rbp-68h]
  __int64 v134; // [rsp+38h] [rbp-68h]
  __int64 v135; // [rsp+38h] [rbp-68h]
  __int64 v136; // [rsp+38h] [rbp-68h]
  __int64 v137; // [rsp+40h] [rbp-60h]
  __int64 v138; // [rsp+48h] [rbp-58h]
  __int64 *v139; // [rsp+48h] [rbp-58h]
  __int64 *v140; // [rsp+50h] [rbp-50h]
  __int64 *v141; // [rsp+58h] [rbp-48h]
  __int64 *v142; // [rsp+58h] [rbp-48h]
  __int64 v143; // [rsp+60h] [rbp-40h]
  __int64 v144; // [rsp+60h] [rbp-40h]
  __int64 v145; // [rsp+68h] [rbp-38h]

  while ( 1 )
  {
    v8 = a6;
    v9 = a1;
    v10 = a5;
    v145 = a3;
    v11 = a7;
    if ( a5 <= a7 )
      v11 = a5;
    if ( v11 >= a4 )
      break;
    if ( a5 <= a7 )
    {
      result = a3 - (_QWORD)a2;
      v143 = a3 - (_QWORD)a2;
      v20 = (a3 - (__int64)a2) >> 3;
      if ( a3 - (__int64)a2 <= 0 )
        return result;
      v142 = a2;
      v21 = a2;
      v22 = (__int64 *)a6;
      do
      {
        v23 = *v21;
        *v21 = 0;
        v24 = *v22;
        *v22 = v23;
        if ( v24 )
          j_j___libc_free_0(v24, 24);
        ++v21;
        ++v22;
        --v20;
      }
      while ( v20 );
      v25 = v143;
      if ( v143 <= 0 )
        v25 = 8;
      result = (__int64)&v8[v25];
      if ( v142 != v9 )
      {
        if ( v8 == (char *)result )
          return result;
        v26 = v142 - 1;
        v27 = result - 8;
        for ( i = (__int64 *)(v145 - 8); ; --i )
        {
          v30 = *(_QWORD *)v27;
          v31 = *(_DWORD *)(*(_QWORD *)v27 + 8LL);
          v32 = *(_DWORD *)(*(_QWORD *)v27 + 12LL);
          if ( v31 == 7 )
            v32 = -(*(_DWORD *)(v30 + 16) + v32);
          result = *v26;
          v33 = *(_DWORD *)(*v26 + 8);
          v34 = *(_DWORD *)(*v26 + 12);
          if ( v33 == 7 )
            v34 = -(*(_DWORD *)(result + 16) + v34);
          if ( v32 > v34
            || v32 == v34
            && ((v94 = *(_BYTE *)(v30 + 20), (v94 & 1) == 0) && (*(_BYTE *)(result + 20) & 1) != 0
             || ((*(_BYTE *)(result + 20) ^ v94) & 1) == 0
             && (v31 < v33
              || v31 == v33
              && *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v30 + 16LL) + 48LL) < *(_DWORD *)(*(_QWORD *)(*(_QWORD *)result
                                                                                                + 16LL)
                                                                                    + 48LL))) )
          {
            *v26 = 0;
            v35 = *i;
            *i = result;
            if ( v35 )
              j_j___libc_free_0(v35, 24);
            if ( v26 == v9 )
            {
              result = v27 + 8 - (_QWORD)v8;
              v95 = result >> 3;
              if ( result > 0 )
              {
                v96 = -8 * v95 + v27;
                v97 = &i[-v95];
                do
                {
                  result = *(_QWORD *)(v96 + 8 * v95);
                  *(_QWORD *)(v96 + 8 * v95) = 0;
                  v98 = v97[v95 - 1];
                  v97[v95 - 1] = result;
                  if ( v98 )
                    result = j_j___libc_free_0(v98, 24);
                  --v95;
                }
                while ( v95 );
              }
              return result;
            }
            --v26;
          }
          else
          {
            *(_QWORD *)v27 = 0;
            v29 = *i;
            *i = v30;
            if ( v29 )
              result = j_j___libc_free_0(v29, 24);
            if ( v8 == (char *)v27 )
              return result;
            v27 -= 8;
          }
        }
      }
      v99 = v25 >> 3;
      if ( v25 > 0 )
      {
        v100 = result - 8 * v99;
        v101 = -8 * v99 + v145;
        do
        {
          result = *(_QWORD *)(v100 + 8 * v99 - 8);
          *(_QWORD *)(v100 + 8 * v99 - 8) = 0;
          v102 = *(_QWORD *)(v101 + 8 * v99 - 8);
          *(_QWORD *)(v101 + 8 * v99 - 8) = result;
          if ( v102 )
            result = j_j___libc_free_0(v102, 24);
          --v99;
        }
        while ( v99 );
      }
      return result;
    }
    if ( a5 >= a4 )
    {
      v128 = a4;
      v16 = a5 / 2;
      v140 = &a2[a5 / 2];
      v74 = sub_1D92C20(a1, (__int64)a2, v140);
      v14 = v128;
      v15 = a2;
      v141 = v74;
      v13 = (__int64)a2;
      v138 = v74 - a1;
    }
    else
    {
      v132 = a4;
      v138 = a4 / 2;
      v141 = &a1[a4 / 2];
      v12 = sub_1D92CF0(a2, a3, v141);
      v13 = (__int64)a2;
      v14 = v132;
      v140 = v12;
      v15 = a2;
      v16 = v12 - a2;
    }
    v137 = v14 - v138;
    if ( v14 - v138 > v16 && a7 >= v16 )
    {
      v17 = (__int64)v141;
      if ( !v16 )
        goto LABEL_10;
      v129 = v13 - (_QWORD)v141;
      v75 = (v13 - (__int64)v141) >> 3;
      v135 = (__int64)v140 - v13;
      v117 = v75;
      if ( (__int64)v140 - v13 <= 0 )
      {
        if ( v129 <= 0 )
          goto LABEL_10;
        v136 = 0;
        v124 = 0;
LABEL_103:
        v82 = v117;
        v130 = (int)v9;
        v118 = v10;
        v114 = v8;
        v83 = v75;
        v82 *= -8;
        v84 = (char *)v140 + v82;
        v85 = v82 + v13;
        do
        {
          v86 = *(_QWORD *)(v85 + 8 * v83 - 8);
          *(_QWORD *)(v85 + 8 * v83 - 8) = 0;
          v87 = *(_QWORD *)&v84[8 * v83 - 8];
          *(_QWORD *)&v84[8 * v83 - 8] = v86;
          if ( v87 )
            j_j___libc_free_0(v87, 24);
          --v83;
        }
        while ( v83 );
        LODWORD(v9) = v130;
        v10 = v118;
        v8 = v114;
      }
      else
      {
        v123 = (v13 - (__int64)v141) >> 3;
        v109 = v13;
        v113 = (int)a1;
        v76 = v8;
        v106 = v10;
        v77 = (__int64 *)v13;
        v103 = v8;
        v78 = ((__int64)v140 - v13) >> 3;
        do
        {
          v79 = *v77;
          *v77 = 0;
          v80 = *(_QWORD *)v76;
          *(_QWORD *)v76 = v79;
          if ( v80 )
            j_j___libc_free_0(v80, 24);
          ++v77;
          v76 += 8;
          --v78;
        }
        while ( v78 );
        v81 = 8;
        v75 = v123;
        LODWORD(v9) = v113;
        v13 = v109;
        v10 = v106;
        v8 = v103;
        if ( v135 > 0 )
          v81 = v135;
        v124 = v81;
        v136 = v81 >> 3;
        if ( v129 > 0 )
          goto LABEL_103;
      }
      if ( v124 <= 0 )
      {
        v17 = (__int64)v141;
      }
      else
      {
        v131 = (int)v9;
        v125 = v10;
        v88 = (__int64 *)v8;
        v119 = v8;
        v89 = v141;
        v90 = v136;
        do
        {
          v91 = *v88;
          *v88 = 0;
          v92 = *v89;
          *v89 = v91;
          if ( v92 )
            j_j___libc_free_0(v92, 24);
          ++v88;
          ++v89;
          --v90;
        }
        while ( v90 );
        LODWORD(v9) = v131;
        v10 = v125;
        v8 = v119;
        v93 = v136;
        if ( v136 <= 0 )
          v93 = 1;
        v17 = (__int64)&v141[v93];
      }
      goto LABEL_10;
    }
    if ( a7 < v137 )
    {
      v17 = sub_1D919E0((__int64)v141, v13, (__int64)v140);
      goto LABEL_10;
    }
    v17 = (__int64)v140;
    if ( !v137 )
      goto LABEL_10;
    v126 = (__int64)v140 - v13;
    v133 = v13 - (_QWORD)v141;
    v115 = ((__int64)v140 - v13) >> 3;
    if ( v13 - (__int64)v141 <= 0 )
    {
      if ( v126 <= 0 )
        goto LABEL_10;
      v111 = v8;
      v134 = 0;
      v121 = 0;
    }
    else
    {
      v120 = v15;
      v110 = (int)a1;
      v56 = (__int64 *)v8;
      v107 = v10;
      v57 = v141;
      v104 = v8;
      v58 = (v13 - (__int64)v141) >> 3;
      do
      {
        v59 = *v57;
        *v57 = 0;
        v60 = *v56;
        *v56 = v59;
        if ( v60 )
          j_j___libc_free_0(v60, 24);
        ++v57;
        ++v56;
        --v58;
      }
      while ( v58 );
      v61 = 8;
      v15 = v120;
      v8 = v104;
      LODWORD(v9) = v110;
      v10 = v107;
      if ( v133 > 0 )
        v61 = v133;
      v121 = v61;
      v111 = &v104[v61];
      v134 = v61 >> 3;
      if ( v126 <= 0 )
        goto LABEL_85;
    }
    v105 = v8;
    v62 = v115;
    v127 = (int)v9;
    v108 = v10;
    v63 = v141;
    v64 = v15;
    do
    {
      v65 = *v64;
      *v64 = 0;
      v66 = *v63;
      *v63 = v65;
      if ( v66 )
        j_j___libc_free_0(v66, 24);
      ++v64;
      ++v63;
      --v62;
    }
    while ( v62 );
    LODWORD(v9) = v127;
    v10 = v108;
    v8 = v105;
LABEL_85:
    if ( v121 <= 0 )
    {
      v17 = (__int64)v140;
    }
    else
    {
      v67 = v111;
      v122 = (int)v9;
      v116 = v10;
      v112 = v8;
      v68 = v134;
      v69 = &v140[-v134];
      v70 = &v67[-8 * v134];
      do
      {
        v71 = *(_QWORD *)&v70[8 * v68 - 8];
        *(_QWORD *)&v70[8 * v68 - 8] = 0;
        v72 = v69[v68 - 1];
        v69[v68 - 1] = v71;
        if ( v72 )
          j_j___libc_free_0(v72, 24);
        --v68;
      }
      while ( v68 );
      v73 = 0x1FFFFFFFFFFFFFFFLL;
      LODWORD(v9) = v122;
      if ( v134 > 0 )
        v73 = -1 * v134;
      v10 = v116;
      v8 = v112;
      v17 = (__int64)&v140[v73];
    }
LABEL_10:
    v18 = v138;
    v139 = (__int64 *)v17;
    sub_1D94260((_DWORD)v9, (_DWORD)v141, v17, v18, v16, (_DWORD)v8, a7, a8);
    a4 = v137;
    a6 = v8;
    a5 = v10 - v16;
    a2 = v140;
    a3 = v145;
    a1 = v139;
  }
  v40 = a1;
  v41 = (__int64 *)a6;
  result = (char *)a2 - (char *)a1;
  v144 = (char *)a2 - (char *)a1;
  v42 = a2 - a1;
  if ( (char *)a2 - (char *)a1 > 0 )
  {
    do
    {
      v43 = *v40;
      *v40 = 0;
      v44 = *v41;
      *v41 = v43;
      if ( v44 )
        j_j___libc_free_0(v44, 24);
      ++v40;
      ++v41;
      --v42;
    }
    while ( v42 );
    v45 = v144;
    result = 8;
    if ( v144 <= 0 )
      v45 = 8;
    v46 = &v8[v45];
    if ( v8 != &v8[v45] )
    {
      v47 = a2;
      while ( v47 != (__int64 *)v145 )
      {
        v49 = *v47;
        v50 = *(_DWORD *)(*v47 + 8);
        v51 = *(_DWORD *)(*v47 + 12);
        if ( v50 == 7 )
          v51 = -(*(_DWORD *)(v49 + 16) + v51);
        result = *(_QWORD *)v8;
        v52 = *(_DWORD *)(*(_QWORD *)v8 + 8LL);
        v53 = *(_DWORD *)(*(_QWORD *)v8 + 12LL);
        if ( v52 == 7 )
          v53 = -(*(_DWORD *)(result + 16) + v53);
        if ( v51 > v53
          || v51 == v53
          && ((v55 = *(_BYTE *)(v49 + 20), (v55 & 1) == 0) && (*(_BYTE *)(result + 20) & 1) != 0
           || ((*(_BYTE *)(result + 20) ^ v55) & 1) == 0
           && (v50 < v52
            || v50 == v52
            && *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v49 + 16LL) + 48LL) < *(_DWORD *)(*(_QWORD *)(*(_QWORD *)result + 16LL)
                                                                                  + 48LL))) )
        {
          *v47 = 0;
          v54 = *v9;
          *v9 = v49;
          if ( v54 )
            result = j_j___libc_free_0(v54, 24);
          ++v47;
        }
        else
        {
          *(_QWORD *)v8 = 0;
          v48 = *v9;
          *v9 = result;
          if ( v48 )
            result = j_j___libc_free_0(v48, 24);
          v8 += 8;
        }
        ++v9;
        if ( v8 == v46 )
          return result;
      }
      v36 = v46 - v8;
      v37 = v9;
      v38 = v36 >> 3;
      if ( v36 > 0 )
      {
        do
        {
          result = *(_QWORD *)v8;
          *(_QWORD *)v8 = 0;
          v39 = *v37;
          *v37 = result;
          if ( v39 )
            result = j_j___libc_free_0(v39, 24);
          v8 += 8;
          ++v37;
          --v38;
        }
        while ( v38 );
      }
    }
  }
  return result;
}
