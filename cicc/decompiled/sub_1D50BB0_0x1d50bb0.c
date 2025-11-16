// Function: sub_1D50BB0
// Address: 0x1d50bb0
//
unsigned __int64 __fastcall sub_1D50BB0(_QWORD *a1)
{
  _QWORD *v1; // r15
  _QWORD *v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // rbx
  __int64 v6; // r12
  __int64 v7; // r13
  int v8; // eax
  __int64 v9; // rax
  _QWORD *v10; // r13
  __int64 v11; // r12
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 *v15; // r13
  __int64 *v16; // rax
  __int64 *v17; // rbx
  __int64 v18; // r8
  unsigned __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // rdi
  __int64 v22; // rsi
  __int64 v23; // r12
  int v24; // edx
  __int64 v25; // rbx
  int v26; // r13d
  _QWORD *v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r14
  int v30; // eax
  __int64 v31; // r13
  int v32; // ebx
  unsigned int v33; // r12d
  _QWORD *v34; // rax
  __int64 v35; // r15
  _QWORD *v36; // r14
  __int64 v37; // rsi
  __int64 v38; // rax
  unsigned int v39; // ecx
  __int64 v40; // rax
  unsigned int v41; // esi
  bool v42; // cf
  __int64 v43; // rax
  __int64 v44; // r10
  __int64 v45; // rax
  int v46; // edx
  _QWORD *v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v50; // rdx
  _DWORD *v51; // rax
  __int64 v52; // r12
  __int64 v53; // rbx
  __int64 v54; // r15
  __int64 v55; // rax
  __int64 v56; // r13
  int v57; // eax
  int v58; // eax
  __int64 v59; // rax
  int v60; // eax
  __int64 v61; // rax
  __int64 v62; // r14
  __int64 v63; // r12
  __int64 v64; // rbx
  unsigned __int64 v65; // rdi
  __int64 v66; // rdi
  __int64 v67; // r14
  __int64 v68; // r12
  unsigned __int64 v69; // rax
  __int64 v70; // rbx
  __int64 v71; // rax
  __int64 v72; // rdi
  __int64 v73; // rsi
  __int64 v74; // r13
  int v75; // edx
  __int64 v76; // r12
  _QWORD *v77; // rdx
  __int64 v78; // rax
  __int64 v79; // rcx
  __int64 i; // r13
  _DWORD *v81; // rax
  __int64 v82; // r12
  __int64 v83; // r14
  __int64 v84; // r10
  int v85; // eax
  __int64 v86; // rax
  __int64 v87; // rbx
  __int64 v88; // rdi
  __int64 v89; // r14
  __int64 v90; // r12
  unsigned __int64 result; // rax
  __int64 v92; // rdi
  unsigned __int64 v93; // rcx
  __int64 v94; // rdx
  unsigned __int64 v95; // rsi
  __int64 v96; // r12
  int v97; // edx
  __int64 v98; // rbx
  int v99; // r13d
  __int64 v100; // rax
  __int64 v101; // r12
  __int64 v102; // r14
  _QWORD *v103; // r13
  __int64 v104; // rbx
  __int64 v105; // r15
  __int64 v106; // rbx
  _QWORD *v107; // r14
  __int64 v108; // r13
  __int64 v109; // rbx
  __int64 v110; // rsi
  int v111; // eax
  __int64 v112; // rsi
  __int64 v113; // rax
  __int64 v114; // rax
  __int64 v115; // r15
  __int64 v116; // rdx
  _QWORD *v117; // rcx
  _QWORD *v118; // rax
  int v119; // eax
  __int64 v120; // rsi
  __int64 v121; // rdi
  __int64 v122; // rdx
  __int64 v123; // rsi
  __int64 v124; // r13
  int v125; // edx
  __int64 v126; // r12
  __int64 v127; // rsi
  __int64 v128; // r12
  int v129; // edx
  __int64 v130; // rbx
  int v131; // r13d
  _QWORD *v132; // rax
  __int64 v133; // rdi
  __int64 v134; // rbx
  __int64 v135; // rdi
  __int64 v136; // rsi
  __int64 v137; // r12
  int v138; // edx
  __int64 v139; // rbx
  int v140; // r13d
  __int64 v141; // rdi
  int v142; // edx
  __int64 v143; // [rsp-8h] [rbp-1C8h]
  __int64 v144; // [rsp+8h] [rbp-1B8h]
  __int64 v145; // [rsp+8h] [rbp-1B8h]
  int v146; // [rsp+10h] [rbp-1B0h]
  __int64 v147; // [rsp+10h] [rbp-1B0h]
  int v148; // [rsp+18h] [rbp-1A8h]
  _QWORD *v149; // [rsp+18h] [rbp-1A8h]
  __int64 v150; // [rsp+20h] [rbp-1A0h]
  unsigned __int64 v151; // [rsp+20h] [rbp-1A0h]
  __int64 v152; // [rsp+28h] [rbp-198h]
  __int64 v153; // [rsp+28h] [rbp-198h]
  __int64 v154; // [rsp+28h] [rbp-198h]
  __int64 v155; // [rsp+28h] [rbp-198h]
  int v156; // [rsp+30h] [rbp-190h]
  __int64 v157; // [rsp+30h] [rbp-190h]
  __int64 v158; // [rsp+38h] [rbp-188h]
  __int64 v159; // [rsp+38h] [rbp-188h]
  __int64 v160; // [rsp+38h] [rbp-188h]
  __int64 v161; // [rsp+38h] [rbp-188h]
  int v162; // [rsp+38h] [rbp-188h]
  __int64 v163; // [rsp+38h] [rbp-188h]
  __int64 v164; // [rsp+38h] [rbp-188h]
  int v165; // [rsp+38h] [rbp-188h]
  int v166; // [rsp+148h] [rbp-78h]
  _QWORD v167[2]; // [rsp+150h] [rbp-70h]
  __int64 v168; // [rsp+160h] [rbp-60h] BYREF
  int v169; // [rsp+168h] [rbp-58h]
  __int64 v170; // [rsp+170h] [rbp-50h]
  __int64 v171; // [rsp+178h] [rbp-48h]
  __int64 v172; // [rsp+180h] [rbp-40h]

  v1 = a1;
  v2 = (_QWORD *)a1[31];
  v3 = v2[113];
  v4 = (v2[114] - v3) >> 4;
  if ( (_DWORD)v4 )
  {
    v5 = 0;
    v158 = 16LL * (unsigned int)(v4 - 1);
    while ( 1 )
    {
      v6 = *(_QWORD *)(v3 + v5);
      v7 = a1[32];
      if ( (unsigned __int8)sub_1DD6970(v2[98], *(_QWORD *)(v6 + 24)) )
      {
        v8 = *(_DWORD *)(*(_QWORD *)(a1[31] + 904LL) + v5 + 8);
        v168 = 0;
        v170 = 0;
        v169 = v8;
        v171 = 0;
        v172 = 0;
        sub_1E1A9C0(v6, v7, &v168);
        v9 = *(_QWORD *)(a1[31] + 784LL);
        LOBYTE(v168) = 4;
        v170 = 0;
        LODWORD(v168) = v168 & 0xFFF000FF;
        v171 = v9;
        sub_1E1A9C0(v6, v7, &v168);
      }
      if ( v158 == v5 )
        break;
      v2 = (_QWORD *)a1[31];
      v5 += 16;
      v3 = v2[113];
    }
    v1 = a1;
  }
  v10 = (_QWORD *)v1[35];
  v11 = v10[82];
  if ( v11 )
  {
    v12 = v10[83];
    v13 = v10[84];
    if ( v12 )
    {
      if ( !v13 )
        goto LABEL_22;
      v14 = v10[82];
      v15 = (__int64 *)(v11 + 24);
      v16 = sub_1D471D0(v14);
      v17 = v16;
      if ( v16 != (__int64 *)(v11 + 24) )
      {
        v18 = v12 + 24;
        if ( v15 != (__int64 *)(v12 + 24) )
        {
          if ( v12 != v11 )
          {
            sub_1DD5C00(v12 + 16, v11 + 16, v16, v11 + 24);
            v18 = v12 + 24;
          }
          if ( v15 != v17 )
          {
            v19 = *(_QWORD *)(v11 + 24) & 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)((*v17 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v15;
            *(_QWORD *)(v11 + 24) = *(_QWORD *)(v11 + 24) & 7LL | *v17 & 0xFFFFFFFFFFFFFFF8LL;
            v20 = *(_QWORD *)(v12 + 24);
            *(_QWORD *)(v19 + 8) = v18;
            v20 &= 0xFFFFFFFFFFFFFFF8LL;
            *v17 = v20 | *v17 & 7;
            *(_QWORD *)(v20 + 8) = v17;
            *(_QWORD *)(v12 + 24) = v19 | *(_QWORD *)(v12 + 24) & 7LL;
          }
        }
      }
      *(_QWORD *)(v1[31] + 784LL) = v11;
      *(_QWORD *)(v1[31] + 792LL) = v15;
      v21 = v1[35];
      v22 = v21 + 656;
      sub_206AE40(v21, v21 + 656, v11);
      v23 = v1[34];
      v25 = sub_2051C20(v1[35]);
      v26 = v24;
      if ( v25 )
      {
        nullsub_686();
        v22 = 0;
        *(_QWORD *)(v23 + 176) = v25;
        *(_DWORD *)(v23 + 184) = v26;
        sub_1D23870();
      }
      else
      {
        *(_QWORD *)(v23 + 176) = 0;
        *(_DWORD *)(v23 + 184) = v24;
      }
      sub_20515F0(v1[35], v22);
      sub_1D50350((__int64)v1);
      v27 = (_QWORD *)v1[35];
      v28 = v27[84];
      if ( v28 + 24 != (*(_QWORD *)(v28 + 24) & 0xFFFFFFFFFFFFFFF8LL) )
        goto LABEL_21;
      *(_QWORD *)(v1[31] + 784LL) = v28;
      *(_QWORD *)(v1[31] + 792LL) = v28 + 24;
      v141 = v1[35];
      v136 = v141 + 656;
      sub_2053930(v141, v141 + 656);
      v137 = v1[34];
      v139 = sub_2051C20(v1[35]);
      v140 = v142;
      if ( !v139 )
      {
        *(_QWORD *)(v137 + 176) = 0;
        *(_DWORD *)(v137 + 184) = v142;
        goto LABEL_141;
      }
      goto LABEL_140;
    }
    if ( !v13 )
    {
      v133 = v10[82];
      *(_QWORD *)(v1[31] + 784LL) = v11;
      v134 = v1[31];
      *(_QWORD *)(v134 + 792) = sub_1D471D0(v133);
      v135 = v1[35];
      v136 = v135 + 656;
      sub_206AE40(v135, v135 + 656, v11);
      v137 = v1[34];
      v139 = sub_2051C20(v1[35]);
      v140 = v138;
      if ( !v139 )
      {
        *(_QWORD *)(v137 + 176) = 0;
        *(_DWORD *)(v137 + 184) = v138;
        goto LABEL_141;
      }
LABEL_140:
      nullsub_686();
      v136 = 0;
      *(_QWORD *)(v137 + 176) = v139;
      *(_DWORD *)(v137 + 184) = v140;
      sub_1D23870();
LABEL_141:
      sub_20515F0(v1[35], v136);
      sub_1D50350((__int64)v1);
      v27 = (_QWORD *)v1[35];
LABEL_21:
      v27[82] = 0;
      v27[83] = 0;
      v10 = (_QWORD *)v1[35];
    }
  }
LABEL_22:
  v29 = v10[79];
  v144 = v10[80];
  if ( v144 != v29 )
  {
    do
    {
      if ( !*(_BYTE *)(v29 + 45) )
      {
        v127 = v29;
        *(_QWORD *)(v1[31] + 784LL) = *(_QWORD *)(v29 + 48);
        *(_QWORD *)(v1[31] + 792LL) = *(_QWORD *)(v1[31] + 784LL) + 24LL;
        sub_206BB00(v1[35], v29, *(_QWORD *)(v1[31] + 784LL));
        v128 = v1[34];
        v130 = sub_2051C20(v1[35]);
        v131 = v129;
        if ( v130 )
        {
          nullsub_686();
          v127 = 0;
          *(_QWORD *)(v128 + 176) = v130;
          *(_DWORD *)(v128 + 184) = v131;
          sub_1D23870();
        }
        else
        {
          *(_QWORD *)(v128 + 176) = 0;
          *(_DWORD *)(v128 + 184) = v129;
        }
        sub_20515F0(v1[35], v127);
        sub_1D50350((__int64)v1);
      }
      v30 = *(_DWORD *)(v29 + 72);
      v31 = 32;
      v32 = 2;
      v33 = *(_DWORD *)(v29 + 176);
      v156 = v30;
      v146 = v30 + 2;
      if ( v30 )
      {
        v34 = v1;
        v35 = v29;
        v36 = v34;
        while ( 1 )
        {
          v38 = v31 - 32 + *(_QWORD *)(v35 + 64);
          v39 = *(_DWORD *)(v38 + 24);
          v40 = *(_QWORD *)(v38 + 8);
          v41 = v33 - v39;
          v42 = v33 < v39;
          v33 = 0;
          if ( !v42 )
            v33 = v41;
          *(_QWORD *)(v36[31] + 784LL) = v40;
          *(_QWORD *)(v36[31] + 792LL) = *(_QWORD *)(v36[31] + 784LL) + 24LL;
          v43 = *(_QWORD *)(v35 + 64);
          if ( *(_BYTE *)(v35 + 46) && v156 == v32 )
            v44 = *(_QWORD *)(v43 + v31 + 16);
          else
            v44 = v156 == v32 - 1 ? *(_QWORD *)(v35 + 56) : *(_QWORD *)(v43 + v31 + 8);
          sub_20548A0(v36[35], v35, v44, v33, *(_DWORD *)(v35 + 40), v43 + v31 - 32, *(_QWORD *)(v36[31] + 784LL));
          v160 = v36[34];
          v45 = sub_2051C20(v36[35]);
          v37 = v143;
          if ( v45 )
          {
            v148 = v46;
            v152 = v160;
            v159 = v45;
            nullsub_686();
            v37 = 0;
            *(_QWORD *)(v152 + 176) = v159;
            *(_DWORD *)(v152 + 184) = v148;
            sub_1D23870();
          }
          else
          {
            *(_QWORD *)(v160 + 176) = 0;
            *(_DWORD *)(v160 + 184) = v46;
          }
          sub_20515F0(v36[35], v37);
          sub_1D50350((__int64)v36);
          if ( *(_BYTE *)(v35 + 46) )
          {
            if ( v156 == v32 )
              break;
          }
          ++v32;
          v31 += 32;
          if ( v146 == v32 )
          {
            v47 = v36;
            v29 = v35;
            v1 = v47;
            goto LABEL_41;
          }
        }
        v132 = v36;
        v29 = v35;
        --*(_DWORD *)(v35 + 72);
        v1 = v132;
      }
LABEL_41:
      v48 = v1[31];
      v49 = *(_QWORD *)(v48 + 904);
      v50 = (*(_QWORD *)(v48 + 912) - v49) >> 4;
      if ( (_DWORD)v50 )
      {
        v149 = v1;
        v153 = 0;
        v147 = 16LL * (unsigned int)(v50 - 1);
        while ( 1 )
        {
          v51 = (_DWORD *)(v153 + v49);
          v52 = *(_QWORD *)v51;
          v157 = v149[32];
          v53 = *(_QWORD *)(*(_QWORD *)v51 + 24LL);
          if ( *(_QWORD *)(v29 + 56) == v53 )
          {
            v58 = v51[2];
            v168 = 0;
            v170 = 0;
            v169 = v58;
            v171 = 0;
            v172 = 0;
            sub_1E1A9C0(v52, v157, &v168);
            v59 = *(_QWORD *)(v29 + 48);
            LOBYTE(v168) = 4;
            v170 = 0;
            LODWORD(v168) = v168 & 0xFFF000FF;
            v171 = v59;
            sub_1E1A9C0(v52, v157, &v168);
            if ( !*(_BYTE *)(v29 + 46) )
            {
              v60 = *(_DWORD *)(*(_QWORD *)(v149[31] + 904LL) + v153 + 8);
              v168 = 0;
              v170 = 0;
              v169 = v60;
              v171 = 0;
              v172 = 0;
              sub_1E1A9C0(v52, v157, &v168);
              v61 = *(_QWORD *)(*(_QWORD *)(v29 + 64) + 32LL * *(unsigned int *)(v29 + 72) - 24);
              LOBYTE(v168) = 4;
              v170 = 0;
              LODWORD(v168) = v168 & 0xFFF000FF;
              v171 = v61;
              sub_1E1A9C0(v52, v157, &v168);
            }
          }
          v54 = 0;
          v55 = *(unsigned int *)(v29 + 72);
          v161 = 32 * v55;
          if ( (_DWORD)v55 )
          {
            do
            {
              while ( 1 )
              {
                v56 = *(_QWORD *)(*(_QWORD *)(v29 + 64) + v54 + 8);
                if ( (unsigned __int8)sub_1DD6970(v56, v53) )
                  break;
                v54 += 32;
                if ( v161 == v54 )
                  goto LABEL_49;
              }
              v54 += 32;
              v57 = *(_DWORD *)(*(_QWORD *)(v149[31] + 904LL) + v153 + 8);
              v168 = 0;
              v170 = 0;
              v169 = v57;
              v171 = 0;
              v172 = 0;
              sub_1E1A9C0(v52, v157, &v168);
              LOBYTE(v168) = 4;
              LODWORD(v168) = v168 & 0xFFF000FF;
              v170 = 0;
              v171 = v56;
              sub_1E1A9C0(v52, v157, &v168);
            }
            while ( v161 != v54 );
          }
LABEL_49:
          if ( v147 == v153 )
            break;
          v153 += 16;
          v49 = *(_QWORD *)(v149[31] + 904LL);
        }
        v1 = v149;
      }
      v29 += 184;
    }
    while ( v144 != v29 );
    v10 = (_QWORD *)v1[35];
    v62 = v10[79];
    v63 = v10[80];
    if ( v62 != v63 )
    {
      v64 = v10[79];
      do
      {
        v65 = *(_QWORD *)(v64 + 64);
        if ( v65 != v64 + 80 )
          _libc_free(v65);
        if ( *(_DWORD *)(v64 + 24) > 0x40u )
        {
          v66 = *(_QWORD *)(v64 + 16);
          if ( v66 )
            j_j___libc_free_0_0(v66);
        }
        if ( *(_DWORD *)(v64 + 8) > 0x40u && *(_QWORD *)v64 )
          j_j___libc_free_0_0(*(_QWORD *)v64);
        v64 += 184;
      }
      while ( v64 != v63 );
      v10[80] = v62;
      v10 = (_QWORD *)v1[35];
    }
  }
  v67 = v10[77];
  v68 = v10[76];
  v69 = 0xCCCCCCCCCCCCCCCDLL * ((v67 - v68) >> 4);
  if ( (_DWORD)v69 )
  {
    v70 = 0;
    v150 = 80LL * (unsigned int)v69;
    do
    {
      v71 = v68 + v70;
      if ( !*(_BYTE *)(v68 + v70 + 48) )
      {
        *(_QWORD *)(v1[31] + 784LL) = *(_QWORD *)(v71 + 40);
        *(_QWORD *)(v1[31] + 792LL) = *(_QWORD *)(v1[31] + 784LL) + 24LL;
        v121 = v1[35];
        v122 = v70 + *(_QWORD *)(v121 + 608);
        v123 = v122 + 56;
        sub_206A770(v121, v122 + 56, v122, *(_QWORD *)(v1[31] + 784LL));
        v124 = v1[34];
        v126 = sub_2051C20(v1[35]);
        if ( v126 )
        {
          v165 = v125;
          nullsub_686();
          v123 = 0;
          *(_QWORD *)(v124 + 176) = v126;
          *(_DWORD *)(v124 + 184) = v165;
          sub_1D23870();
        }
        else
        {
          *(_QWORD *)(v124 + 176) = 0;
          *(_DWORD *)(v124 + 184) = v125;
        }
        sub_20515F0(v1[35], v123);
        sub_1D50350((__int64)v1);
        v71 = v70 + *(_QWORD *)(v1[35] + 608LL);
      }
      *(_QWORD *)(v1[31] + 784LL) = *(_QWORD *)(v71 + 64);
      *(_QWORD *)(v1[31] + 792LL) = *(_QWORD *)(v1[31] + 784LL) + 24LL;
      v72 = v1[35];
      v73 = v70 + *(_QWORD *)(v72 + 608) + 56;
      sub_2053040(v72, v73);
      v74 = v1[34];
      v76 = sub_2051C20(v1[35]);
      if ( v76 )
      {
        v162 = v75;
        nullsub_686();
        v73 = 0;
        *(_QWORD *)(v74 + 176) = v76;
        *(_DWORD *)(v74 + 184) = v162;
        sub_1D23870();
      }
      else
      {
        *(_QWORD *)(v74 + 176) = 0;
        *(_DWORD *)(v74 + 184) = v75;
      }
      sub_20515F0(v1[35], v73);
      sub_1D50350((__int64)v1);
      v77 = (_QWORD *)v1[31];
      v78 = v77[113];
      v79 = (v77[114] - v78) >> 4;
      if ( (_DWORD)v79 )
      {
        v163 = 16LL * (unsigned int)(v79 - 1);
        for ( i = 0; ; i += 16 )
        {
          v81 = (_DWORD *)(i + v78);
          v82 = v1[32];
          v83 = *(_QWORD *)v81;
          v84 = *(_QWORD *)(*(_QWORD *)v81 + 24LL);
          if ( *(_QWORD *)(*(_QWORD *)(v1[35] + 608LL) + v70 + 72) == v84 )
          {
            v111 = v81[2];
            v112 = v1[32];
            v155 = v84;
            v168 = 0;
            v169 = v111;
            v170 = 0;
            v171 = 0;
            v172 = 0;
            sub_1E1A9C0(v83, v112, &v168);
            v113 = *(_QWORD *)(*(_QWORD *)(v1[35] + 608LL) + v70 + 40);
            LOBYTE(v168) = 4;
            v170 = 0;
            LODWORD(v168) = v168 & 0xFFF000FF;
            v171 = v113;
            sub_1E1A9C0(v83, v82, &v168);
            v77 = (_QWORD *)v1[31];
            v84 = v155;
          }
          if ( (unsigned __int8)sub_1DD6970(v77[98], v84) )
          {
            v85 = *(_DWORD *)(*(_QWORD *)(v1[31] + 904LL) + i + 8);
            v168 = 0;
            v170 = 0;
            v169 = v85;
            v171 = 0;
            v172 = 0;
            sub_1E1A9C0(v83, v82, &v168);
            v86 = *(_QWORD *)(v1[31] + 784LL);
            LOBYTE(v168) = 4;
            v170 = 0;
            LODWORD(v168) = v168 & 0xFFF000FF;
            v171 = v86;
            sub_1E1A9C0(v83, v82, &v168);
            if ( v163 == i )
              break;
          }
          else if ( v163 == i )
          {
            break;
          }
          v77 = (_QWORD *)v1[31];
          v78 = v77[113];
        }
      }
      v10 = (_QWORD *)v1[35];
      v70 += 80;
      v68 = v10[76];
    }
    while ( v150 != v70 );
    v67 = v10[77];
  }
  if ( v67 != v68 )
  {
    v87 = v68;
    do
    {
      if ( *(_DWORD *)(v87 + 24) > 0x40u )
      {
        v88 = *(_QWORD *)(v87 + 16);
        if ( v88 )
          j_j___libc_free_0_0(v88);
      }
      if ( *(_DWORD *)(v87 + 8) > 0x40u && *(_QWORD *)v87 )
        j_j___libc_free_0_0(*(_QWORD *)v87);
      v87 += 80;
    }
    while ( v67 != v87 );
    v10[77] = v68;
    v10 = (_QWORD *)v1[35];
  }
  v89 = v10[74];
  v90 = v10[73];
  result = 0xCCCCCCCCCCCCCCCDLL * ((v89 - v90) >> 4);
  if ( (_DWORD)result )
  {
    v151 = 0;
    v145 = 80LL * (unsigned int)result;
    while ( 1 )
    {
      v166 = 1;
      *(_QWORD *)(v1[31] + 784LL) = *(_QWORD *)(v90 + v151 + 48);
      *(_QWORD *)(v1[31] + 792LL) = *(_QWORD *)(v1[31] + 784LL) + 24LL;
      v92 = v1[35];
      v93 = *(_QWORD *)(v92 + 584) + v151;
      v94 = *(_QWORD *)(v93 + 40);
      v95 = v93;
      v167[0] = *(_QWORD *)(v93 + 32);
      if ( v94 != v167[0] )
      {
        v167[1] = v94;
        v166 = 2;
      }
      sub_2069F40(v92, v93, *(_QWORD *)(v1[31] + 784LL));
      v96 = v1[34];
      v98 = sub_2051C20(v1[35]);
      v99 = v97;
      if ( v98 )
      {
        nullsub_686();
        v95 = 0;
        *(_QWORD *)(v96 + 176) = v98;
        *(_DWORD *)(v96 + 184) = v99;
        sub_1D23870();
      }
      else
      {
        *(_QWORD *)(v96 + 176) = 0;
        *(_DWORD *)(v96 + 184) = v97;
      }
      sub_20515F0(v1[35], v95);
      sub_1D50350((__int64)v1);
      v100 = v1[31];
      v101 = *(_QWORD *)(v100 + 784);
      v102 = 0;
      v164 = 8LL * (unsigned int)(v166 - 1);
      v103 = v1;
      while ( 1 )
      {
        *(_QWORD *)(v100 + 784) = *(_QWORD *)((char *)v167 + v102);
        *(_QWORD *)(v103[31] + 792LL) = *(_QWORD *)(v103[31] + 784LL) + 24LL;
        if ( (unsigned __int8)sub_1DD6970(v101, *(_QWORD *)(v103[31] + 784LL)) )
        {
          v104 = *(_QWORD *)(v103[31] + 784LL);
          v105 = *(_QWORD *)(v104 + 32);
          v106 = v104 + 24;
          if ( v106 != v105 )
            break;
        }
        if ( v102 == v164 )
          goto LABEL_107;
LABEL_100:
        v100 = v103[31];
        v102 += 8;
      }
      v154 = v102;
      v107 = v103;
      v108 = v105;
      do
      {
        if ( **(_WORD **)(v108 + 16) && **(_WORD **)(v108 + 16) != 45 )
          break;
        v114 = v107[31];
        v115 = v107[32];
        LODWORD(v116) = 0;
        v117 = *(_QWORD **)(v114 + 904);
        if ( *v117 == v108 )
        {
          v118 = *(_QWORD **)(v114 + 904);
        }
        else
        {
          do
          {
            v116 = (unsigned int)(v116 + 1);
            v118 = &v117[2 * v116];
          }
          while ( *v118 != v108 );
        }
        v119 = *((_DWORD *)v118 + 2);
        v120 = v107[32];
        v168 = 0;
        v170 = 0;
        v169 = v119;
        v171 = 0;
        v172 = 0;
        sub_1E1A9C0(v108, v120, &v168);
        LOBYTE(v168) = 4;
        v170 = 0;
        LODWORD(v168) = v168 & 0xFFF000FF;
        v171 = v101;
        sub_1E1A9C0(v108, v115, &v168);
        if ( (*(_BYTE *)v108 & 4) == 0 )
        {
          while ( (*(_BYTE *)(v108 + 46) & 8) != 0 )
            v108 = *(_QWORD *)(v108 + 8);
        }
        v108 = *(_QWORD *)(v108 + 8);
      }
      while ( v106 != v108 );
      v103 = v107;
      v102 = v154;
      if ( v154 != v164 )
        goto LABEL_100;
LABEL_107:
      v1 = v103;
      v10 = (_QWORD *)v103[35];
      v151 += 80LL;
      result = v151;
      v90 = v10[73];
      if ( v145 == v151 )
      {
        v89 = v10[74];
        break;
      }
    }
  }
  if ( v90 != v89 )
  {
    v109 = v90;
    do
    {
      v110 = *(_QWORD *)(v109 + 56);
      if ( v110 )
        result = sub_161E7C0(v109 + 56, v110);
      v109 += 80;
    }
    while ( v109 != v89 );
    v10[74] = v90;
  }
  return result;
}
