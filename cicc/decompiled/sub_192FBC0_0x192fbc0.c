// Function: sub_192FBC0
// Address: 0x192fbc0
//
void __fastcall sub_192FBC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, _DWORD *a6, __int64 a7)
{
  __int64 v7; // rax
  char *v8; // r15
  __int64 v10; // r14
  int v11; // r8d
  int v12; // r9d
  __int64 v13; // rbx
  __int64 v14; // rax
  int v15; // ecx
  __int64 v16; // r10
  __int64 v17; // rcx
  unsigned __int64 v18; // rbx
  _DWORD *v19; // r14
  int v20; // r8d
  char **v21; // r12
  int v22; // edx
  char **v23; // rsi
  __int64 v24; // rdi
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // rax
  _DWORD *v29; // r12
  int v30; // r8d
  char *v31; // rbx
  __int64 v32; // r13
  __int64 i; // rdi
  int v34; // eax
  char **v35; // rsi
  int v36; // eax
  char **v37; // rsi
  __int64 v38; // rcx
  unsigned __int64 v39; // rbx
  _DWORD *v40; // r14
  char **v41; // r13
  int v42; // eax
  char **v43; // rsi
  __int64 v44; // rdi
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rcx
  char *v48; // rbx
  char **v49; // rsi
  __int64 v50; // r14
  __int64 v51; // r13
  char *v52; // rbx
  __int64 v53; // rdx
  _DWORD *v54; // r12
  __int64 v55; // rdi
  __int64 v56; // rdx
  __int64 v57; // rcx
  unsigned __int64 v58; // r13
  __int64 v59; // r15
  char **v60; // rbx
  int v61; // edx
  char **v62; // rsi
  __int64 v63; // rdi
  __int64 v64; // rdx
  __int64 v65; // rdx
  __int64 v66; // r15
  __int64 v67; // r13
  __int64 v68; // r15
  char **v69; // rbx
  unsigned __int64 v70; // r12
  int v71; // edx
  char **v72; // rsi
  __int64 v73; // rdi
  __int64 v74; // rdx
  signed __int64 v75; // r12
  _DWORD *v76; // r15
  char **v77; // r13
  int v78; // eax
  char **v79; // rsi
  __int64 v80; // rdi
  __int64 v81; // rax
  __int64 v82; // rbx
  unsigned __int64 v83; // r13
  char **v84; // r15
  __int64 v85; // r12
  int v86; // eax
  char **v87; // rsi
  __int64 v88; // rdi
  __int64 v89; // rcx
  char **v90; // rbx
  unsigned __int64 v91; // r13
  __int64 v92; // r15
  int v93; // edx
  char **v94; // rsi
  __int64 v95; // rdi
  __int64 v96; // rdx
  __int64 v97; // rdx
  __int64 v98; // rax
  __int64 v99; // r13
  char **v100; // rbx
  unsigned __int64 v101; // r12
  _DWORD *v102; // r15
  int v103; // edx
  char **v104; // rsi
  __int64 v105; // rdi
  __int64 v106; // rdx
  __int64 v107; // r15
  signed __int64 v108; // r12
  char **v109; // r13
  int v110; // eax
  char **v111; // rsi
  __int64 v112; // rdi
  __int64 v113; // rax
  __int64 v114; // rdx
  unsigned __int64 v115; // r13
  char **v116; // rbx
  _DWORD *v117; // r12
  int v118; // eax
  char **v119; // rsi
  __int64 v120; // rdi
  __int64 v121; // rdx
  unsigned __int64 v122; // r13
  char **v123; // rbx
  _DWORD *v124; // r12
  __int64 v125; // rdi
  char **v126; // r14
  __int64 v127; // [rsp+0h] [rbp-90h]
  __int64 v128; // [rsp+8h] [rbp-88h]
  __int64 v129; // [rsp+10h] [rbp-80h]
  char *v130; // [rsp+10h] [rbp-80h]
  __int64 v131; // [rsp+18h] [rbp-78h]
  __int64 v132; // [rsp+18h] [rbp-78h]
  __int64 v133; // [rsp+18h] [rbp-78h]
  __int64 v134; // [rsp+18h] [rbp-78h]
  signed __int64 v135; // [rsp+20h] [rbp-70h]
  signed __int64 v136; // [rsp+20h] [rbp-70h]
  __int64 v138; // [rsp+30h] [rbp-60h]
  __int64 v139; // [rsp+30h] [rbp-60h]
  __int64 v140; // [rsp+38h] [rbp-58h]
  __int64 v141; // [rsp+40h] [rbp-50h]
  _DWORD *v142; // [rsp+40h] [rbp-50h]
  __int64 v143; // [rsp+48h] [rbp-48h]
  __int64 v144; // [rsp+50h] [rbp-40h]
  __int64 v145; // [rsp+50h] [rbp-40h]
  __int64 v146; // [rsp+50h] [rbp-40h]

  v7 = a5;
  v8 = (char *)a6;
  if ( a7 <= a5 )
    v7 = a7;
  v143 = a3;
  v144 = a4;
  if ( v7 < a4 )
  {
    v10 = a5;
    if ( a7 >= a5 )
    {
LABEL_32:
      v38 = v143 - a2;
      v146 = v143 - a2;
      v39 = 0x8E38E38E38E38E39LL * ((v143 - a2) >> 3);
      if ( v143 - a2 <= 0 )
        return;
      v40 = v8 + 24;
      v41 = (char **)(a2 + 24);
      do
      {
        v42 = *((_DWORD *)v41 - 6);
        v43 = v41;
        v44 = (__int64)v40;
        v41 += 9;
        v40 += 18;
        *(v40 - 24) = v42;
        *(v40 - 23) = *((_DWORD *)v41 - 23);
        *(v40 - 22) = *((_DWORD *)v41 - 22);
        *(v40 - 21) = *((_DWORD *)v41 - 21);
        *(v40 - 20) = *((_DWORD *)v41 - 20);
        sub_192DBD0(v44, v43, a3, v38, a5, (int)a6);
        --v39;
      }
      while ( v39 );
      v45 = 72;
      v46 = v146 - 48;
      if ( v146 > 0 )
        v45 = v143 - a2;
      v47 = 24;
      if ( v146 <= 0 )
        v46 = 24;
      v48 = &v8[v45];
      v49 = (char **)&v8[v46];
      if ( a2 == a1 )
      {
        v121 = 0x8E38E38E38E38E39LL;
        v122 = 0x8E38E38E38E38E39LL * (v45 >> 3);
        v123 = (char **)(v48 - 120);
        v124 = (_DWORD *)(v143 - 48);
        while ( 1 )
        {
          v125 = (__int64)v124;
          v126 = v123;
          v124 -= 18;
          v124[12] = *((_DWORD *)v123 + 12);
          v124[13] = *((_DWORD *)v123 + 13);
          v124[14] = *((_DWORD *)v123 + 14);
          v124[15] = *((_DWORD *)v123 + 15);
          v124[16] = *((_DWORD *)v123 + 16);
          sub_192DBD0(v125, v49, v121, v47, a5, (int)a6);
          if ( !--v122 )
            break;
          v123 -= 9;
          v49 = v126;
        }
        return;
      }
      if ( v8 == v48 )
        return;
      v50 = v143;
      v51 = a2 - 72;
      v52 = v48 - 72;
      while ( 1 )
      {
        v54 = (_DWORD *)(v50 - 72);
        v55 = v50 - 48;
        if ( *((_DWORD *)v52 + 4) > *(_DWORD *)(v51 + 16) )
        {
          *(_DWORD *)(v50 - 72) = *(_DWORD *)v51;
          v54[1] = *(_DWORD *)(v51 + 4);
          v54[2] = *(_DWORD *)(v51 + 8);
          v54[3] = *(_DWORD *)(v51 + 12);
          v53 = *(unsigned int *)(v51 + 16);
          v54[4] = v53;
          sub_192DBD0(v55, (char **)(v51 + 24), v53, v47, a5, (int)a6);
          if ( v51 == a1 )
          {
            v114 = 0x8E38E38E38E38E39LL;
            v115 = 0x8E38E38E38E38E39LL * ((v52 + 72 - v8) >> 3);
            if ( v52 + 72 - v8 > 0 )
            {
              v116 = (char **)(v52 + 24);
              v117 = (_DWORD *)(v50 - 120);
              do
              {
                v118 = *((_DWORD *)v116 - 6);
                v119 = v116;
                v120 = (__int64)v117;
                v116 -= 9;
                v117 -= 18;
                v117[12] = v118;
                v117[13] = *((_DWORD *)v116 + 13);
                v117[14] = *((_DWORD *)v116 + 14);
                v117[15] = *((_DWORD *)v116 + 15);
                v117[16] = *((_DWORD *)v116 + 16);
                sub_192DBD0(v120, v119, v114, v47, a5, (int)a6);
                --v115;
              }
              while ( v115 );
            }
            return;
          }
          v51 -= 72;
        }
        else
        {
          v56 = *(unsigned int *)v52;
          *(_DWORD *)(v50 - 72) = v56;
          v54[1] = *((_DWORD *)v52 + 1);
          v54[2] = *((_DWORD *)v52 + 2);
          v54[3] = *((_DWORD *)v52 + 3);
          v54[4] = *((_DWORD *)v52 + 4);
          sub_192DBD0(v55, (char **)v52 + 3, v56, v47, a5, (int)a6);
          if ( v8 == v52 )
            return;
          v52 -= 72;
        }
        v50 -= 72;
      }
    }
    if ( a5 >= a4 )
      goto LABEL_15;
LABEL_6:
    v138 = a4 / 2;
    v141 = a1 + 72 * (a4 / 2);
    v140 = sub_192E3D0(a2, v143, v141);
    v13 = 0x8E38E38E38E38E39LL * ((v140 - a2) >> 3);
    while ( 1 )
    {
      v144 -= v138;
      if ( v144 > v13 && a7 >= v13 )
      {
        v14 = v141;
        if ( !v13 )
          goto LABEL_10;
        v133 = a2 - v141;
        v89 = v140 - a2;
        if ( v140 - a2 <= 0 )
        {
          if ( v133 <= 0 )
            goto LABEL_10;
          v136 = 0;
          v99 = 0;
LABEL_77:
          v134 = v13;
          v100 = (char **)(a2 - 48);
          v101 = 0x8E38E38E38E38E39LL * ((a2 - v141) >> 3);
          v102 = (_DWORD *)(v140 - 48);
          do
          {
            v103 = *((_DWORD *)v100 - 6);
            v104 = v100;
            v105 = (__int64)v102;
            v100 -= 9;
            v102 -= 18;
            v102[12] = v103;
            v102[13] = *((_DWORD *)v100 + 13);
            v102[14] = *((_DWORD *)v100 + 14);
            v102[15] = *((_DWORD *)v100 + 15);
            v106 = *((unsigned int *)v100 + 16);
            v102[16] = v106;
            sub_192DBD0(v105, v104, v106, v89, v11, v12);
            --v101;
          }
          while ( v101 );
          v13 = v134;
        }
        else
        {
          v128 = v13;
          v90 = (char **)(a2 + 24);
          v91 = 0x8E38E38E38E38E39LL * ((v140 - a2) >> 3);
          v92 = (__int64)(a6 + 6);
          do
          {
            v93 = *((_DWORD *)v90 - 6);
            v94 = v90;
            v95 = v92;
            v90 += 9;
            v92 += 72;
            *(_DWORD *)(v92 - 96) = v93;
            *(_DWORD *)(v92 - 92) = *((_DWORD *)v90 - 23);
            *(_DWORD *)(v92 - 88) = *((_DWORD *)v90 - 22);
            *(_DWORD *)(v92 - 84) = *((_DWORD *)v90 - 21);
            v96 = *((unsigned int *)v90 - 20);
            *(_DWORD *)(v92 - 80) = v96;
            sub_192DBD0(v95, v94, v96, v89, v11, v12);
            --v91;
          }
          while ( v91 );
          v98 = v140 - a2;
          v89 = 0x8E38E38E38E38E39LL;
          v13 = v128;
          if ( v140 - a2 <= 0 )
            v98 = 72;
          v99 = v98;
          v136 = 0x8E38E38E38E38E39LL * (v98 >> 3);
          if ( v133 > 0 )
            goto LABEL_77;
        }
        v14 = v141;
        if ( v99 > 0 )
        {
          v107 = v141 + 24;
          v108 = v136;
          v109 = (char **)(a6 + 6);
          do
          {
            v110 = *((_DWORD *)v109 - 6);
            v111 = v109;
            v112 = v107;
            v109 += 9;
            v107 += 72;
            *(_DWORD *)(v107 - 96) = v110;
            *(_DWORD *)(v107 - 92) = *((_DWORD *)v109 - 23);
            *(_DWORD *)(v107 - 88) = *((_DWORD *)v109 - 22);
            *(_DWORD *)(v107 - 84) = *((_DWORD *)v109 - 21);
            *(_DWORD *)(v107 - 80) = *((_DWORD *)v109 - 20);
            sub_192DBD0(v112, v111, v97, v89, v11, v12);
            --v108;
          }
          while ( v108 );
          v113 = 72 * v136;
          if ( v136 <= 0 )
            v113 = 72;
          v14 = v141 + v113;
        }
        goto LABEL_10;
      }
      if ( a7 < v144 )
      {
        v14 = sub_192EB50(v141, a2, v140, v144, v11, v12);
        goto LABEL_10;
      }
      v14 = v140;
      if ( !v144 )
        goto LABEL_10;
      v131 = v140 - a2;
      v57 = a2 - v141;
      if ( a2 - v141 <= 0 )
      {
        if ( v131 <= 0 )
          goto LABEL_10;
        v135 = 0;
        v67 = 0;
        v130 = (char *)a6;
        v127 = v141 + 24;
      }
      else
      {
        v129 = v13;
        v58 = 0x8E38E38E38E38E39LL * ((a2 - v141) >> 3);
        v59 = (__int64)(a6 + 6);
        v127 = v141 + 24;
        v60 = (char **)(v141 + 24);
        do
        {
          v61 = *((_DWORD *)v60 - 6);
          v62 = v60;
          v63 = v59;
          v60 += 9;
          v59 += 72;
          *(_DWORD *)(v59 - 96) = v61;
          *(_DWORD *)(v59 - 92) = *((_DWORD *)v60 - 23);
          *(_DWORD *)(v59 - 88) = *((_DWORD *)v60 - 22);
          *(_DWORD *)(v59 - 84) = *((_DWORD *)v60 - 21);
          v64 = *((unsigned int *)v60 - 20);
          *(_DWORD *)(v59 - 80) = v64;
          sub_192DBD0(v63, v62, v64, v57, v11, v12);
          --v58;
        }
        while ( v58 );
        v66 = 72;
        v13 = v129;
        v57 = 0x8E38E38E38E38E39LL;
        if ( a2 - v141 > 0 )
          v66 = a2 - v141;
        v67 = v66;
        v130 = (char *)a6 + v66;
        v135 = 0x8E38E38E38E38E39LL * (v66 >> 3);
        if ( v131 <= 0 )
          goto LABEL_59;
      }
      v132 = v13;
      v68 = v127;
      v69 = (char **)(a2 + 24);
      v70 = 0x8E38E38E38E38E39LL * ((v140 - a2) >> 3);
      do
      {
        v71 = *((_DWORD *)v69 - 6);
        v72 = v69;
        v73 = v68;
        v69 += 9;
        v68 += 72;
        *(_DWORD *)(v68 - 96) = v71;
        *(_DWORD *)(v68 - 92) = *((_DWORD *)v69 - 23);
        *(_DWORD *)(v68 - 88) = *((_DWORD *)v69 - 22);
        *(_DWORD *)(v68 - 84) = *((_DWORD *)v69 - 21);
        v74 = *((unsigned int *)v69 - 20);
        *(_DWORD *)(v68 - 80) = v74;
        sub_192DBD0(v73, v72, v74, v57, v11, v12);
        --v70;
      }
      while ( v70 );
      v13 = v132;
LABEL_59:
      v14 = v140;
      if ( v67 > 0 )
      {
        v75 = v135;
        v76 = (_DWORD *)(v140 - 48);
        v77 = (char **)(v130 - 48);
        do
        {
          v78 = *((_DWORD *)v77 - 6);
          v79 = v77;
          v80 = (__int64)v76;
          v77 -= 9;
          v76 -= 18;
          v76[12] = v78;
          v76[13] = *((_DWORD *)v77 + 13);
          v76[14] = *((_DWORD *)v77 + 14);
          v76[15] = *((_DWORD *)v77 + 15);
          v76[16] = *((_DWORD *)v77 + 16);
          sub_192DBD0(v80, v79, v65, v57, v11, v12);
          --v75;
        }
        while ( v75 );
        v81 = -72 * v135;
        if ( v135 <= 0 )
          v81 = -72;
        v14 = v140 + v81;
      }
LABEL_10:
      v15 = v138;
      v10 -= v13;
      v139 = v14;
      sub_192FBC0(a1, v141, v14, v15, v13, (_DWORD)a6, a7);
      a3 = v10;
      if ( a7 <= v10 )
        a3 = a7;
      if ( a3 >= v144 )
      {
        v8 = (char *)a6;
        a2 = v140;
        a1 = v139;
        break;
      }
      if ( a7 >= v10 )
      {
        v8 = (char *)a6;
        a2 = v140;
        a1 = v139;
        goto LABEL_32;
      }
      a4 = v144;
      a2 = v140;
      a1 = v139;
      if ( v10 < v144 )
        goto LABEL_6;
LABEL_15:
      v13 = v10 / 2;
      v140 = a2 + 72 * (v10 / 2);
      v141 = sub_192E430(a1, a2, v140);
      v138 = 0x8E38E38E38E38E39LL * ((v141 - v16) >> 3);
    }
  }
  v17 = a2 - a1;
  v145 = a2 - a1;
  v18 = 0x8E38E38E38E38E39LL * ((a2 - a1) >> 3);
  if ( a2 - a1 > 0 )
  {
    v142 = (_DWORD *)a2;
    v19 = v8 + 24;
    v20 = a1 + 24;
    v21 = (char **)(a1 + 24);
    do
    {
      v22 = *((_DWORD *)v21 - 6);
      v23 = v21;
      v24 = (__int64)v19;
      v21 += 9;
      v19 += 18;
      *(v19 - 24) = v22;
      *(v19 - 23) = *((_DWORD *)v21 - 23);
      *(v19 - 22) = *((_DWORD *)v21 - 22);
      *(v19 - 21) = *((_DWORD *)v21 - 21);
      v25 = *((unsigned int *)v21 - 20);
      *(v19 - 20) = v25;
      sub_192DBD0(v24, v23, v25, v17, v20, (int)a6);
      --v18;
    }
    while ( v18 );
    v27 = v145;
    v28 = 72;
    v29 = v142;
    v30 = a1 + 24;
    if ( v145 > 0 )
      v28 = v145;
    v31 = &v8[v28];
    if ( v8 != &v8[v28] )
    {
      if ( (_DWORD *)v143 != v142 )
      {
        v32 = a1;
        for ( i = a1 + 24; ; i = v32 + 24 )
        {
          if ( v29[4] > *((_DWORD *)v8 + 4) )
          {
            v34 = *v29;
            v35 = (char **)(v29 + 6);
            v32 += 72;
            v29 += 18;
            *(_DWORD *)(v32 - 72) = v34;
            *(_DWORD *)(v32 - 68) = *(v29 - 17);
            *(_DWORD *)(v32 - 64) = *(v29 - 16);
            *(_DWORD *)(v32 - 60) = *(v29 - 15);
            *(_DWORD *)(v32 - 56) = *(v29 - 14);
            sub_192DBD0(i, v35, v26, v27, v30, (int)a6);
            if ( v8 == v31 )
              return;
          }
          else
          {
            v36 = *(_DWORD *)v8;
            v37 = (char **)(v8 + 24);
            v8 += 72;
            v32 += 72;
            *(_DWORD *)(v32 - 72) = v36;
            *(_DWORD *)(v32 - 68) = *((_DWORD *)v8 - 17);
            *(_DWORD *)(v32 - 64) = *((_DWORD *)v8 - 16);
            *(_DWORD *)(v32 - 60) = *((_DWORD *)v8 - 15);
            *(_DWORD *)(v32 - 56) = *((_DWORD *)v8 - 14);
            sub_192DBD0(i, v37, v26, v27, v30, (int)a6);
            if ( v8 == v31 )
              return;
          }
          if ( (_DWORD *)v143 == v29 )
            break;
        }
        a1 = v32;
      }
      if ( v8 != v31 )
      {
        v82 = v31 - v8;
        v83 = 0x8E38E38E38E38E39LL * (v82 >> 3);
        if ( v82 > 0 )
        {
          v84 = (char **)(v8 + 24);
          v85 = a1 + 24;
          do
          {
            v86 = *((_DWORD *)v84 - 6);
            v87 = v84;
            v88 = v85;
            v84 += 9;
            v85 += 72;
            *(_DWORD *)(v85 - 96) = v86;
            *(_DWORD *)(v85 - 92) = *((_DWORD *)v84 - 23);
            *(_DWORD *)(v85 - 88) = *((_DWORD *)v84 - 22);
            *(_DWORD *)(v85 - 84) = *((_DWORD *)v84 - 21);
            *(_DWORD *)(v85 - 80) = *((_DWORD *)v84 - 20);
            sub_192DBD0(v88, v87, v26, v27, v30, (int)a6);
            --v83;
          }
          while ( v83 );
        }
      }
    }
  }
}
