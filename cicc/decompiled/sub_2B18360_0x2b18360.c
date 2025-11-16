// Function: sub_2B18360
// Address: 0x2b18360
//
void __fastcall sub_2B18360(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v7; // rax
  __int64 v8; // r15
  __int64 v10; // r14
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rbx
  __int64 v14; // rdx
  __int64 v15; // rax
  int v16; // ecx
  __int64 v17; // r10
  __int64 v18; // rcx
  unsigned __int64 v19; // rbx
  __int64 v20; // r14
  __int64 v21; // r8
  char **v22; // r12
  __int64 v23; // rdx
  char **v24; // rsi
  __int64 v25; // rdi
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // r12
  __int64 v30; // r8
  __int64 v31; // rbx
  __int64 v32; // r13
  __int64 v33; // rdi
  __int64 v34; // rax
  char **v35; // rsi
  __int64 v36; // rax
  char **v37; // rsi
  __int64 v38; // rbx
  unsigned __int64 v39; // r13
  char **v40; // r15
  __int64 v41; // r12
  __int64 v42; // rax
  char **v43; // rsi
  __int64 v44; // rdi
  __int64 v45; // rcx
  unsigned __int64 v46; // rbx
  __int64 v47; // r14
  char **v48; // r13
  __int64 v49; // rax
  char **v50; // rsi
  __int64 v51; // rdi
  __int64 v52; // rcx
  __int64 v53; // rdx
  __int64 v54; // rax
  char **v55; // rsi
  __int64 v56; // r13
  __int64 v57; // rbx
  __int64 v58; // r13
  __int64 i; // r12
  __int64 v60; // rdx
  __int64 v61; // rdi
  __int64 v62; // rdx
  __int64 v63; // rcx
  unsigned __int64 v64; // r13
  __int64 v65; // r15
  char **v66; // rbx
  __int64 v67; // rdx
  char **v68; // rsi
  __int64 v69; // rdi
  __int64 v70; // rdx
  __int64 v71; // r15
  __int64 v72; // r13
  __int64 v73; // r15
  char **v74; // rbx
  unsigned __int64 v75; // r12
  __int64 v76; // rdx
  char **v77; // rsi
  __int64 v78; // rdi
  signed __int64 v79; // r12
  __int64 v80; // r15
  char **v81; // r13
  __int64 v82; // rax
  char **v83; // rsi
  __int64 v84; // rdi
  __int64 v85; // rax
  __int64 v86; // rcx
  char **v87; // rbx
  unsigned __int64 v88; // r13
  __int64 v89; // r15
  __int64 v90; // rdx
  char **v91; // rsi
  __int64 v92; // rdi
  __int64 v93; // rax
  unsigned __int64 v94; // r13
  char **v95; // r12
  __int64 v96; // r15
  __int64 v97; // rax
  char **v98; // rsi
  __int64 v99; // rdi
  __int64 v100; // r15
  signed __int64 v101; // r12
  char **v102; // r13
  __int64 v103; // rax
  char **v104; // rsi
  __int64 v105; // rdi
  __int64 v106; // rax
  __int64 v107; // rdi
  __int64 v108; // rdx
  __int64 v109; // rcx
  unsigned __int64 v110; // r12
  char **v111; // r13
  __int64 v112; // rbx
  __int64 v113; // rax
  char **v114; // rsi
  __int64 v115; // rdi
  unsigned __int64 v116; // r14
  __int64 v117; // rbx
  __int64 v118; // r12
  __int64 v119; // rdi
  char **v120; // r13
  __int64 v121; // [rsp+0h] [rbp-90h]
  __int64 v122; // [rsp+10h] [rbp-80h]
  __int64 v123; // [rsp+10h] [rbp-80h]
  __int64 v124; // [rsp+10h] [rbp-80h]
  __int64 v125; // [rsp+10h] [rbp-80h]
  __int64 v126; // [rsp+18h] [rbp-78h]
  __int64 v127; // [rsp+18h] [rbp-78h]
  __int64 v128; // [rsp+18h] [rbp-78h]
  signed __int64 v129; // [rsp+20h] [rbp-70h]
  signed __int64 v130; // [rsp+20h] [rbp-70h]
  __int64 v132; // [rsp+30h] [rbp-60h]
  __int64 v133; // [rsp+30h] [rbp-60h]
  __int64 v134; // [rsp+38h] [rbp-58h]
  __int64 v135; // [rsp+40h] [rbp-50h]
  __int64 v136; // [rsp+40h] [rbp-50h]
  __int64 v137; // [rsp+48h] [rbp-48h]
  __int64 v138; // [rsp+50h] [rbp-40h]
  __int64 v139; // [rsp+50h] [rbp-40h]
  __int64 v140; // [rsp+50h] [rbp-40h]

  v7 = a5;
  v8 = a6;
  if ( a7 <= a5 )
    v7 = a7;
  v137 = a3;
  v138 = a4;
  if ( v7 < a4 )
  {
    v10 = a5;
    if ( a7 >= a5 )
    {
LABEL_37:
      v45 = v137 - a2;
      v140 = v137 - a2;
      v46 = 0x8E38E38E38E38E39LL * ((v137 - a2) >> 3);
      if ( v137 - a2 <= 0 )
        return;
      v47 = v8 + 8;
      v48 = (char **)(a2 + 8);
      do
      {
        v49 = (__int64)*(v48 - 1);
        v50 = v48;
        v51 = v47;
        v48 += 9;
        v47 += 72;
        *(_QWORD *)(v47 - 80) = v49;
        sub_2B0D090(v51, v50, a3, v45, a5, a6);
        --v46;
      }
      while ( v46 );
      v52 = v137 - a2;
      v53 = 8;
      v54 = v140 - 64;
      if ( v140 <= 0 )
        v54 = 8;
      v55 = (char **)(v8 + v54);
      if ( v140 <= 0 )
        v52 = 72;
      v56 = v8 + v52;
      if ( a2 != a1 )
      {
        if ( v8 == v56 )
          return;
        v57 = a2 - 72;
        v58 = v56 - 72;
        for ( i = v137; ; i -= 72 )
        {
          v61 = i - 64;
          if ( *(_DWORD *)(v58 + 16) > *(_DWORD *)(v57 + 16) )
          {
            v60 = *(_QWORD *)v57;
            *(_QWORD *)(i - 72) = *(_QWORD *)v57;
            sub_2B0D090(v61, (char **)(v57 + 8), v60, v52, a5, a6);
            if ( v57 == a1 )
            {
              v107 = i;
              v108 = 0x8E38E38E38E38E39LL;
              v109 = (v58 + 72 - v8) >> 3;
              v110 = 0x8E38E38E38E38E39LL * v109;
              if ( v58 + 72 - v8 > 0 )
              {
                v111 = (char **)(v58 + 8);
                v112 = v107 - 136;
                do
                {
                  v113 = (__int64)*(v111 - 1);
                  v114 = v111;
                  v115 = v112;
                  v111 -= 9;
                  v112 -= 72;
                  *(_QWORD *)(v112 + 64) = v113;
                  sub_2B0D090(v115, v114, v108, v109, a5, a6);
                  --v110;
                }
                while ( v110 );
              }
              return;
            }
            v57 -= 72;
          }
          else
          {
            v62 = *(_QWORD *)v58;
            *(_QWORD *)(i - 72) = *(_QWORD *)v58;
            sub_2B0D090(v61, (char **)(v58 + 8), v62, v52, a5, a6);
            if ( v8 == v58 )
              return;
            v58 -= 72;
          }
        }
      }
      v116 = 0x8E38E38E38E38E39LL * (v52 >> 3);
      if ( v52 > 0 )
      {
        v117 = v56 - 136;
        v118 = v137 - 64;
        while ( 1 )
        {
          v119 = v118;
          v120 = (char **)v117;
          v118 -= 72;
          *(_QWORD *)(v118 + 64) = *(_QWORD *)(v117 + 64);
          sub_2B0D090(v119, v55, v53, v52, a5, a6);
          if ( !--v116 )
            break;
          v117 -= 72;
          v55 = v120;
        }
      }
      return;
    }
    if ( a5 >= a4 )
      goto LABEL_15;
LABEL_6:
    v132 = a4 / 2;
    v135 = a1 + 72 * (a4 / 2);
    v134 = sub_2B0EA80(a2, v137, v135);
    v13 = 0x8E38E38E38E38E39LL * ((v134 - a2) >> 3);
    while ( 1 )
    {
      v138 -= v132;
      if ( v138 > v13 && a7 >= v13 )
      {
        v14 = v135;
        v15 = v135;
        if ( !v13 )
          goto LABEL_10;
        v128 = a2 - v135;
        v86 = v134 - a2;
        if ( v134 - a2 <= 0 )
        {
          if ( v128 <= 0 )
            goto LABEL_10;
          v130 = 0;
          v125 = 0;
LABEL_76:
          v94 = 0x8E38E38E38E38E39LL * ((a2 - v135) >> 3);
          v95 = (char **)(a2 - 64);
          v96 = v134 - 64;
          do
          {
            v97 = (__int64)*(v95 - 1);
            v98 = v95;
            v99 = v96;
            v95 -= 9;
            v96 -= 72;
            *(_QWORD *)(v96 + 64) = v97;
            sub_2B0D090(v99, v98, v14, v86, v11, v12);
            --v94;
          }
          while ( v94 );
        }
        else
        {
          v124 = v13;
          v87 = (char **)(a2 + 8);
          v88 = 0x8E38E38E38E38E39LL * ((v134 - a2) >> 3);
          v89 = a6 + 8;
          do
          {
            v90 = (__int64)*(v87 - 1);
            v91 = v87;
            v92 = v89;
            v87 += 9;
            v89 += 72;
            *(_QWORD *)(v89 - 80) = v90;
            sub_2B0D090(v92, v91, v90, v86, v11, v12);
            --v88;
          }
          while ( v88 );
          v14 = v134 - a2;
          v93 = 72;
          v13 = v124;
          if ( v134 - a2 > 0 )
            v93 = v134 - a2;
          v125 = v93;
          v130 = 0x8E38E38E38E38E39LL * (v93 >> 3);
          if ( v128 > 0 )
            goto LABEL_76;
        }
        v15 = v135;
        if ( v125 > 0 )
        {
          v100 = v135 + 8;
          v101 = v130;
          v102 = (char **)(a6 + 8);
          do
          {
            v103 = (__int64)*(v102 - 1);
            v104 = v102;
            v105 = v100;
            v102 += 9;
            v100 += 72;
            *(_QWORD *)(v100 - 80) = v103;
            sub_2B0D090(v105, v104, v14, v86, v11, v12);
            --v101;
          }
          while ( v101 );
          v106 = 72 * v130;
          if ( v130 <= 0 )
            v106 = 72;
          v15 = v135 + v106;
        }
        goto LABEL_10;
      }
      if ( a7 < v138 )
      {
        v15 = sub_2B11FA0(v135, a2, v134, v138, v11, v12);
        goto LABEL_10;
      }
      v15 = v134;
      if ( !v138 )
        goto LABEL_10;
      v126 = v134 - a2;
      v63 = a2 - v135;
      if ( a2 - v135 <= 0 )
      {
        if ( v126 <= 0 )
          goto LABEL_10;
        v129 = 0;
        v72 = 0;
        v123 = a6;
        v121 = v135 + 8;
      }
      else
      {
        v122 = v13;
        v64 = 0x8E38E38E38E38E39LL * ((a2 - v135) >> 3);
        v65 = a6 + 8;
        v121 = v135 + 8;
        v66 = (char **)(v135 + 8);
        do
        {
          v67 = (__int64)*(v66 - 1);
          v68 = v66;
          v69 = v65;
          v66 += 9;
          v65 += 72;
          *(_QWORD *)(v65 - 80) = v67;
          sub_2B0D090(v69, v68, v67, v63, v11, v12);
          --v64;
        }
        while ( v64 );
        v71 = 72;
        v13 = v122;
        v63 = 0x8E38E38E38E38E39LL;
        if ( a2 - v135 > 0 )
          v71 = a2 - v135;
        v72 = v71;
        v123 = v71 + a6;
        v129 = 0x8E38E38E38E38E39LL * (v71 >> 3);
        if ( v126 <= 0 )
          goto LABEL_64;
      }
      v127 = v13;
      v73 = v121;
      v74 = (char **)(a2 + 8);
      v75 = 0x8E38E38E38E38E39LL * ((v134 - a2) >> 3);
      do
      {
        v76 = (__int64)*(v74 - 1);
        v77 = v74;
        v78 = v73;
        v74 += 9;
        v73 += 72;
        *(_QWORD *)(v73 - 80) = v76;
        sub_2B0D090(v78, v77, v76, v63, v11, v12);
        --v75;
      }
      while ( v75 );
      v13 = v127;
LABEL_64:
      v15 = v134;
      if ( v72 > 0 )
      {
        v79 = v129;
        v80 = v134 - 64;
        v81 = (char **)(v123 - 64);
        do
        {
          v82 = (__int64)*(v81 - 1);
          v83 = v81;
          v84 = v80;
          v81 -= 9;
          v80 -= 72;
          *(_QWORD *)(v80 + 64) = v82;
          sub_2B0D090(v84, v83, v70, v63, v11, v12);
          --v79;
        }
        while ( v79 );
        v85 = -72 * v129;
        if ( v129 <= 0 )
          v85 = -72;
        v15 = v134 + v85;
      }
LABEL_10:
      v16 = v132;
      v10 -= v13;
      v133 = v15;
      sub_2B18360(a1, v135, v15, v16, v13, a6, a7);
      a3 = v10;
      if ( a7 <= v10 )
        a3 = a7;
      if ( a3 >= v138 )
      {
        v8 = a6;
        a2 = v134;
        a1 = v133;
        break;
      }
      if ( a7 >= v10 )
      {
        v8 = a6;
        a2 = v134;
        a1 = v133;
        goto LABEL_37;
      }
      a4 = v138;
      a2 = v134;
      a1 = v133;
      if ( v10 < v138 )
        goto LABEL_6;
LABEL_15:
      v13 = v10 / 2;
      v134 = a2 + 72 * (v10 / 2);
      v135 = sub_2B0EAE0(a1, a2, v134);
      v132 = 0x8E38E38E38E38E39LL * ((v135 - v17) >> 3);
    }
  }
  v18 = a2 - a1;
  v139 = a2 - a1;
  v19 = 0x8E38E38E38E38E39LL * ((a2 - a1) >> 3);
  if ( a2 - a1 <= 0 )
    return;
  v136 = a2;
  v20 = v8 + 8;
  v21 = a1 + 8;
  v22 = (char **)(a1 + 8);
  do
  {
    v23 = (__int64)*(v22 - 1);
    v24 = v22;
    v25 = v20;
    v22 += 9;
    v20 += 72;
    *(_QWORD *)(v20 - 80) = v23;
    sub_2B0D090(v25, v24, v23, v18, v21, a6);
    --v19;
  }
  while ( v19 );
  v27 = v139;
  v28 = 72;
  v29 = v136;
  v30 = a1 + 8;
  if ( v139 > 0 )
    v28 = v139;
  v31 = v8 + v28;
  if ( v8 == v8 + v28 )
    return;
  if ( v137 == v136 )
    goto LABEL_31;
  v32 = a1;
  v33 = a1 + 8;
  while ( *(_DWORD *)(v29 + 16) > *(_DWORD *)(v8 + 16) )
  {
    v34 = *(_QWORD *)v29;
    v35 = (char **)(v29 + 8);
    v32 += 72;
    v29 += 72;
    *(_QWORD *)(v32 - 72) = v34;
    sub_2B0D090(v33, v35, v26, v27, v30, a6);
    if ( v31 == v8 )
      goto LABEL_30;
LABEL_26:
    if ( v137 == v29 )
      goto LABEL_30;
    v33 = v32 + 8;
  }
  v36 = *(_QWORD *)v8;
  v37 = (char **)(v8 + 8);
  v8 += 72;
  v32 += 72;
  *(_QWORD *)(v32 - 72) = v36;
  sub_2B0D090(v33, v37, v26, v27, v30, a6);
  if ( v31 != v8 )
    goto LABEL_26;
LABEL_30:
  a1 = v32;
LABEL_31:
  if ( v8 != v31 )
  {
    v38 = v31 - v8;
    v39 = 0x8E38E38E38E38E39LL * (v38 >> 3);
    if ( v38 > 0 )
    {
      v40 = (char **)(v8 + 8);
      v41 = a1 + 8;
      do
      {
        v42 = (__int64)*(v40 - 1);
        v43 = v40;
        v44 = v41;
        v40 += 9;
        v41 += 72;
        *(_QWORD *)(v41 - 80) = v42;
        sub_2B0D090(v44, v43, v26, v27, v30, a6);
        --v39;
      }
      while ( v39 );
    }
  }
}
