// Function: sub_2880090
// Address: 0x2880090
//
__int64 __fastcall sub_2880090(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 *a5,
        __int64 a6,
        __int64 a7,
        int a8,
        __int64 a9,
        __int64 a10,
        unsigned __int16 a11,
        unsigned __int16 a12,
        unsigned __int16 a13,
        __int64 a14)
{
  __int64 *v17; // rax
  __int64 v18; // rax
  _BYTE *v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // rax
  int v22; // eax
  int v23; // edx
  int v24; // eax
  int v25; // eax
  int v26; // eax
  __int64 v27; // r13
  _QWORD *v28; // r13
  _QWORD *v29; // rbx
  unsigned __int64 v30; // rsi
  _QWORD *v31; // rax
  _QWORD *v32; // rdi
  __int64 v33; // rcx
  __int64 v34; // rdx
  __int64 v35; // rax
  _QWORD *v36; // rdi
  __int64 v37; // rcx
  __int64 v38; // rdx
  _QWORD *v39; // r13
  _QWORD *v40; // rbx
  unsigned __int64 v41; // rsi
  _QWORD *v42; // rax
  _QWORD *v43; // rdi
  __int64 v44; // rcx
  __int64 v45; // rdx
  __int64 v46; // rax
  _QWORD *v47; // rdi
  __int64 v48; // rcx
  __int64 v49; // rdx
  _QWORD *v50; // r13
  _QWORD *v51; // rbx
  unsigned __int64 v52; // rsi
  _QWORD *v53; // rax
  _QWORD *v54; // rdi
  __int64 v55; // rcx
  __int64 v56; // rdx
  __int64 v57; // rax
  _QWORD *v58; // rdi
  __int64 v59; // rcx
  __int64 v60; // rdx
  _QWORD *v61; // r13
  _QWORD *v62; // rbx
  unsigned __int64 v63; // rsi
  _QWORD *v64; // rax
  _QWORD *v65; // rdi
  __int64 v66; // rcx
  __int64 v67; // rdx
  __int64 v68; // rax
  _QWORD *v69; // rdi
  __int64 v70; // rcx
  __int64 v71; // rdx
  _QWORD *v72; // r13
  _QWORD *v73; // rbx
  unsigned __int64 v74; // rsi
  _QWORD *v75; // rax
  _QWORD *v76; // rdi
  __int64 v77; // rcx
  __int64 v78; // rdx
  __int64 v79; // rax
  _QWORD *v80; // rdi
  __int64 v81; // rcx
  __int64 v82; // rdx
  _QWORD *v83; // r13
  _QWORD *v84; // rbx
  unsigned __int64 v85; // rsi
  _QWORD *v86; // rax
  _QWORD *v87; // rdi
  __int64 v88; // rcx
  __int64 v89; // rdx
  __int64 v90; // rax
  _QWORD *v91; // rdi
  __int64 v92; // rcx
  __int64 v93; // rdx
  _QWORD *v94; // r13
  _QWORD *v95; // rbx
  unsigned __int64 v96; // rsi
  _QWORD *v97; // rax
  _QWORD *v98; // rdi
  __int64 v99; // rcx
  __int64 v100; // rdx
  __int64 v101; // rax
  _QWORD *v102; // rdi
  __int64 v103; // rcx
  __int64 v104; // rdx
  _QWORD *v105; // r13
  _QWORD *v106; // rbx
  unsigned __int64 v107; // rsi
  _QWORD *v108; // rax
  _QWORD *v109; // rdi
  __int64 v110; // rcx
  __int64 v111; // rdx
  __int64 v112; // rax
  _QWORD *v113; // rdi
  __int64 v114; // rcx
  __int64 v115; // rdx
  _QWORD *v116; // r13
  _QWORD *v117; // rbx
  unsigned __int64 v118; // rsi
  _QWORD *v119; // rax
  _QWORD *v120; // rdi
  __int64 v121; // rcx
  __int64 v122; // rdx
  __int64 v123; // rax
  _QWORD *v124; // rdi
  __int64 v125; // rcx
  __int64 v126; // rdx
  _QWORD *v127; // r13
  _QWORD *v128; // rbx
  unsigned __int64 v129; // rsi
  _QWORD *v130; // rax
  _QWORD *v131; // rdi
  __int64 v132; // rcx
  __int64 v133; // rdx
  __int64 v134; // rax
  _QWORD *v135; // rdi
  __int64 v136; // rcx
  __int64 v137; // rdx
  _QWORD *v138; // r13
  _QWORD *v139; // rbx
  unsigned __int64 v140; // rsi
  _QWORD *v141; // rax
  _QWORD *v142; // rdi
  __int64 v143; // rcx
  __int64 v144; // rdx
  __int64 v145; // rax
  _QWORD *v146; // rdi
  __int64 v147; // rcx
  __int64 v148; // rdx
  int v150; // eax
  int v151; // edx
  __int64 v152; // rdx
  __int64 v153; // rcx
  __int64 v154; // r8
  __int64 v155; // r9
  _QWORD *v158; // [rsp+18h] [rbp-78h]
  _QWORD *v159; // [rsp+20h] [rbp-70h] BYREF
  _QWORD v160[2]; // [rsp+30h] [rbp-60h] BYREF
  __int64 v161; // [rsp+40h] [rbp-50h]
  __int64 v162; // [rsp+48h] [rbp-48h]
  __int64 v163; // [rsp+50h] [rbp-40h]

  *(_WORD *)(a1 + 68) = 0;
  v17 = *(__int64 **)(a2 + 32);
  *(_BYTE *)(a1 + 60) = 0;
  *(_DWORD *)(a1 + 64) = 1;
  v18 = sub_AA4B30(*v17);
  v19 = *(_BYTE **)(v18 + 232);
  v20 = *(_QWORD *)(v18 + 240);
  v159 = v160;
  v158 = (_QWORD *)v18;
  sub_287EDE0((__int64 *)&v159, v19, (__int64)&v19[v20]);
  v161 = v158[33];
  v21 = v158[35];
  v162 = v158[34];
  v163 = v21;
  if ( (unsigned int)(v161 - 42) > 1 )
  {
    if ( v159 != v160 )
      j_j___libc_free_0((unsigned __int64)v159);
    v150 = dword_50035C8;
    v151 = qword_50026E8;
    if ( a8 <= 2 )
      v151 = dword_5002608;
    *(_DWORD *)(a1 + 8) = dword_50035C8;
    *(_DWORD *)(a1 + 16) = v150;
    v24 = 8;
    *(_DWORD *)a1 = v151;
    *(_DWORD *)(a1 + 4) = 400;
    *(_DWORD *)(a1 + 12) = 150;
    *(_DWORD *)(a1 + 20) = 0;
  }
  else
  {
    if ( v159 != v160 )
      j_j___libc_free_0((unsigned __int64)v159);
    v22 = dword_50035C8;
    v23 = qword_50026E8;
    if ( a8 <= 1 )
      v23 = dword_5002608;
    *(_DWORD *)a1 = v23;
    *(_DWORD *)(a1 + 4) = 400;
    *(_DWORD *)(a1 + 12) = 150;
    *(_DWORD *)(a1 + 20) = 0;
    *(_DWORD *)(a1 + 8) = v22;
    *(_DWORD *)(a1 + 16) = v22;
    v24 = 4;
  }
  *(_DWORD *)(a1 + 24) = v24;
  v25 = qword_5002D08;
  *(_BYTE *)(a1 + 76) = 0;
  *(_QWORD *)(a1 + 44) = 0x10000;
  *(_DWORD *)(a1 + 32) = v25;
  *(_QWORD *)(a1 + 36) = 0x2FFFFFFFFLL;
  v26 = qword_5003328;
  *(_DWORD *)(a1 + 28) = -1;
  *(_DWORD *)(a1 + 56) = v26;
  *(_DWORD *)(a1 + 52) = 60;
  *(_DWORD *)(a1 + 72) = qword_4F8C268[8];
  sub_DFA030(a4);
  v27 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 72LL);
  if ( (unsigned __int8)sub_B2D610(v27, 47)
    || (unsigned __int8)sub_B2D610(v27, 18)
    || (unsigned int)sub_F6E5D0(a2, 18, v152, v153, v154, v155) != 5 && sub_11F3070(**(_QWORD **)(a2 + 32), a6, a5) )
  {
    *(_DWORD *)(a1 + 4) = 100;
    *(_DWORD *)a1 = *(_DWORD *)(a1 + 8);
    *(_DWORD *)(a1 + 12) = *(_DWORD *)(a1 + 16);
  }
  v28 = sub_C52410();
  v29 = v28 + 1;
  v30 = sub_C959E0();
  v31 = (_QWORD *)v28[2];
  if ( v31 )
  {
    v32 = v28 + 1;
    do
    {
      while ( 1 )
      {
        v33 = v31[2];
        v34 = v31[3];
        if ( v30 <= v31[4] )
          break;
        v31 = (_QWORD *)v31[3];
        if ( !v34 )
          goto LABEL_14;
      }
      v32 = v31;
      v31 = (_QWORD *)v31[2];
    }
    while ( v33 );
LABEL_14:
    if ( v32 != v29 && v30 >= v32[4] )
      v29 = v32;
  }
  if ( v29 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v35 = v29[7];
    if ( v35 )
    {
      v36 = v29 + 6;
      do
      {
        while ( 1 )
        {
          v37 = *(_QWORD *)(v35 + 16);
          v38 = *(_QWORD *)(v35 + 24);
          if ( *(_DWORD *)(v35 + 32) >= dword_5003628 )
            break;
          v35 = *(_QWORD *)(v35 + 24);
          if ( !v38 )
            goto LABEL_23;
        }
        v36 = (_QWORD *)v35;
        v35 = *(_QWORD *)(v35 + 16);
      }
      while ( v37 );
LABEL_23:
      if ( v36 != v29 + 6 && dword_5003628 >= *((_DWORD *)v36 + 8) && *((int *)v36 + 9) > 0 )
        *(_DWORD *)a1 = dword_50036A8;
    }
  }
  v39 = sub_C52410();
  v40 = v39 + 1;
  v41 = sub_C959E0();
  v42 = (_QWORD *)v39[2];
  if ( v42 )
  {
    v43 = v39 + 1;
    do
    {
      while ( 1 )
      {
        v44 = v42[2];
        v45 = v42[3];
        if ( v41 <= v42[4] )
          break;
        v42 = (_QWORD *)v42[3];
        if ( !v45 )
          goto LABEL_32;
      }
      v43 = v42;
      v42 = (_QWORD *)v42[2];
    }
    while ( v44 );
LABEL_32:
    if ( v43 != v40 && v41 >= v43[4] )
      v40 = v43;
  }
  if ( v40 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v46 = v40[7];
    if ( v46 )
    {
      v47 = v40 + 6;
      do
      {
        while ( 1 )
        {
          v48 = *(_QWORD *)(v46 + 16);
          v49 = *(_QWORD *)(v46 + 24);
          if ( *(_DWORD *)(v46 + 32) >= dword_5003468 )
            break;
          v46 = *(_QWORD *)(v46 + 24);
          if ( !v49 )
            goto LABEL_41;
        }
        v47 = (_QWORD *)v46;
        v46 = *(_QWORD *)(v46 + 16);
      }
      while ( v48 );
LABEL_41:
      if ( v47 != v40 + 6 && dword_5003468 >= *((_DWORD *)v47 + 8) && *((int *)v47 + 9) > 0 )
        *(_DWORD *)(a1 + 12) = qword_50034E8;
    }
  }
  v50 = sub_C52410();
  v51 = v50 + 1;
  v52 = sub_C959E0();
  v53 = (_QWORD *)v50[2];
  if ( v53 )
  {
    v54 = v50 + 1;
    do
    {
      while ( 1 )
      {
        v55 = v53[2];
        v56 = v53[3];
        if ( v52 <= v53[4] )
          break;
        v53 = (_QWORD *)v53[3];
        if ( !v56 )
          goto LABEL_50;
      }
      v54 = v53;
      v53 = (_QWORD *)v53[2];
    }
    while ( v55 );
LABEL_50:
    if ( v54 != v51 && v52 >= v54[4] )
      v51 = v54;
  }
  if ( v51 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v57 = v51[7];
    if ( v57 )
    {
      v58 = v51 + 6;
      do
      {
        while ( 1 )
        {
          v59 = *(_QWORD *)(v57 + 16);
          v60 = *(_QWORD *)(v57 + 24);
          if ( *(_DWORD *)(v57 + 32) >= dword_5003388 )
            break;
          v57 = *(_QWORD *)(v57 + 24);
          if ( !v60 )
            goto LABEL_59;
        }
        v58 = (_QWORD *)v57;
        v57 = *(_QWORD *)(v57 + 16);
      }
      while ( v59 );
LABEL_59:
      if ( v58 != v51 + 6 && dword_5003388 >= *((_DWORD *)v58 + 8) && *((int *)v58 + 9) > 0 )
        *(_DWORD *)(a1 + 4) = qword_5003408;
    }
  }
  v61 = sub_C52410();
  v62 = v61 + 1;
  v63 = sub_C959E0();
  v64 = (_QWORD *)v61[2];
  if ( v64 )
  {
    v65 = v61 + 1;
    do
    {
      while ( 1 )
      {
        v66 = v64[2];
        v67 = v64[3];
        if ( v63 <= v64[4] )
          break;
        v64 = (_QWORD *)v64[3];
        if ( !v67 )
          goto LABEL_68;
      }
      v65 = v64;
      v64 = (_QWORD *)v64[2];
    }
    while ( v66 );
LABEL_68:
    if ( v65 != v62 && v63 >= v65[4] )
      v62 = v65;
  }
  if ( v62 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v68 = v62[7];
    if ( v68 )
    {
      v69 = v62 + 6;
      do
      {
        while ( 1 )
        {
          v70 = *(_QWORD *)(v68 + 16);
          v71 = *(_QWORD *)(v68 + 24);
          if ( *(_DWORD *)(v68 + 32) >= dword_50030E8 )
            break;
          v68 = *(_QWORD *)(v68 + 24);
          if ( !v71 )
            goto LABEL_77;
        }
        v69 = (_QWORD *)v68;
        v68 = *(_QWORD *)(v68 + 16);
      }
      while ( v70 );
LABEL_77:
      if ( v69 != v62 + 6 && dword_50030E8 >= *((_DWORD *)v69 + 8) && *((int *)v69 + 9) > 0 )
        *(_DWORD *)(a1 + 28) = dword_5003168;
    }
  }
  v72 = sub_C52410();
  v73 = v72 + 1;
  v74 = sub_C959E0();
  v75 = (_QWORD *)v72[2];
  if ( v75 )
  {
    v76 = v72 + 1;
    do
    {
      while ( 1 )
      {
        v77 = v75[2];
        v78 = v75[3];
        if ( v74 <= v75[4] )
          break;
        v75 = (_QWORD *)v75[3];
        if ( !v78 )
          goto LABEL_86;
      }
      v76 = v75;
      v75 = (_QWORD *)v75[2];
    }
    while ( v77 );
LABEL_86:
    if ( v76 != v73 && v74 >= v76[4] )
      v73 = v76;
  }
  if ( v73 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v79 = v73[7];
    if ( v79 )
    {
      v80 = v73 + 6;
      do
      {
        while ( 1 )
        {
          v81 = *(_QWORD *)(v79 + 16);
          v82 = *(_QWORD *)(v79 + 24);
          if ( *(_DWORD *)(v79 + 32) >= dword_5002C88 )
            break;
          v79 = *(_QWORD *)(v79 + 24);
          if ( !v82 )
            goto LABEL_95;
        }
        v80 = (_QWORD *)v79;
        v79 = *(_QWORD *)(v79 + 16);
      }
      while ( v81 );
LABEL_95:
      if ( v80 != v73 + 6 && dword_5002C88 >= *((_DWORD *)v80 + 8) && *((int *)v80 + 9) > 0 )
        *(_DWORD *)(a1 + 32) = qword_5002D08;
    }
  }
  v83 = sub_C52410();
  v84 = v83 + 1;
  v85 = sub_C959E0();
  v86 = (_QWORD *)v83[2];
  if ( v86 )
  {
    v87 = v83 + 1;
    do
    {
      while ( 1 )
      {
        v88 = v86[2];
        v89 = v86[3];
        if ( v85 <= v86[4] )
          break;
        v86 = (_QWORD *)v86[3];
        if ( !v89 )
          goto LABEL_104;
      }
      v87 = v86;
      v86 = (_QWORD *)v86[2];
    }
    while ( v88 );
LABEL_104:
    if ( v87 != v84 && v85 >= v87[4] )
      v84 = v87;
  }
  if ( v84 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v90 = v84[7];
    if ( v90 )
    {
      v91 = v84 + 6;
      do
      {
        while ( 1 )
        {
          v92 = *(_QWORD *)(v90 + 16);
          v93 = *(_QWORD *)(v90 + 24);
          if ( *(_DWORD *)(v90 + 32) >= dword_5003008 )
            break;
          v90 = *(_QWORD *)(v90 + 24);
          if ( !v93 )
            goto LABEL_113;
        }
        v91 = (_QWORD *)v90;
        v90 = *(_QWORD *)(v90 + 16);
      }
      while ( v92 );
LABEL_113:
      if ( v91 != v84 + 6 && dword_5003008 >= *((_DWORD *)v91 + 8) && *((int *)v91 + 9) > 0 )
        *(_DWORD *)(a1 + 36) = qword_5003088;
    }
  }
  v94 = sub_C52410();
  v95 = v94 + 1;
  v96 = sub_C959E0();
  v97 = (_QWORD *)v94[2];
  if ( v97 )
  {
    v98 = v94 + 1;
    do
    {
      while ( 1 )
      {
        v99 = v97[2];
        v100 = v97[3];
        if ( v96 <= v97[4] )
          break;
        v97 = (_QWORD *)v97[3];
        if ( !v100 )
          goto LABEL_122;
      }
      v98 = v97;
      v97 = (_QWORD *)v97[2];
    }
    while ( v99 );
LABEL_122:
    if ( v98 != v95 && v96 >= v98[4] )
      v95 = v98;
  }
  if ( v95 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v101 = v95[7];
    if ( v101 )
    {
      v102 = v95 + 6;
      do
      {
        while ( 1 )
        {
          v103 = *(_QWORD *)(v101 + 16);
          v104 = *(_QWORD *)(v101 + 24);
          if ( *(_DWORD *)(v101 + 32) >= dword_5002F28 )
            break;
          v101 = *(_QWORD *)(v101 + 24);
          if ( !v104 )
            goto LABEL_131;
        }
        v102 = (_QWORD *)v101;
        v101 = *(_QWORD *)(v101 + 16);
      }
      while ( v103 );
LABEL_131:
      if ( v102 != v95 + 6 && dword_5002F28 >= *((_DWORD *)v102 + 8) && *((int *)v102 + 9) > 0 )
        *(_BYTE *)(a1 + 44) = qword_5002FA8;
    }
  }
  v105 = sub_C52410();
  v106 = v105 + 1;
  v107 = sub_C959E0();
  v108 = (_QWORD *)v105[2];
  if ( v108 )
  {
    v109 = v105 + 1;
    do
    {
      while ( 1 )
      {
        v110 = v108[2];
        v111 = v108[3];
        if ( v107 <= v108[4] )
          break;
        v108 = (_QWORD *)v108[3];
        if ( !v111 )
          goto LABEL_140;
      }
      v109 = v108;
      v108 = (_QWORD *)v108[2];
    }
    while ( v110 );
LABEL_140:
    if ( v109 != v106 && v107 >= v109[4] )
      v106 = v109;
  }
  if ( v106 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v112 = v106[7];
    if ( v112 )
    {
      v113 = v106 + 6;
      do
      {
        while ( 1 )
        {
          v114 = *(_QWORD *)(v112 + 16);
          v115 = *(_QWORD *)(v112 + 24);
          if ( *(_DWORD *)(v112 + 32) >= dword_5002E48 )
            break;
          v112 = *(_QWORD *)(v112 + 24);
          if ( !v115 )
            goto LABEL_149;
        }
        v113 = (_QWORD *)v112;
        v112 = *(_QWORD *)(v112 + 16);
      }
      while ( v114 );
LABEL_149:
      if ( v113 != v106 + 6 && dword_5002E48 >= *((_DWORD *)v113 + 8) && *((int *)v113 + 9) > 0 )
        *(_BYTE *)(a1 + 46) = qword_5002EC8;
    }
  }
  v116 = sub_C52410();
  v117 = v116 + 1;
  v118 = sub_C959E0();
  v119 = (_QWORD *)v116[2];
  if ( v119 )
  {
    v120 = v116 + 1;
    do
    {
      while ( 1 )
      {
        v121 = v119[2];
        v122 = v119[3];
        if ( v118 <= v119[4] )
          break;
        v119 = (_QWORD *)v119[3];
        if ( !v122 )
          goto LABEL_158;
      }
      v120 = v119;
      v119 = (_QWORD *)v119[2];
    }
    while ( v121 );
LABEL_158:
    if ( v120 != v117 && v118 >= v120[4] )
      v117 = v120;
  }
  if ( v117 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v123 = v117[7];
    if ( v123 )
    {
      v124 = v117 + 6;
      do
      {
        while ( 1 )
        {
          v125 = *(_QWORD *)(v123 + 16);
          v126 = *(_QWORD *)(v123 + 24);
          if ( *(_DWORD *)(v123 + 32) >= dword_5002D68 )
            break;
          v123 = *(_QWORD *)(v123 + 24);
          if ( !v126 )
            goto LABEL_167;
        }
        v124 = (_QWORD *)v123;
        v123 = *(_QWORD *)(v123 + 16);
      }
      while ( v125 );
LABEL_167:
      if ( v124 != v117 + 6 && dword_5002D68 >= *((_DWORD *)v124 + 8) && *((int *)v124 + 9) > 0 )
        *(_BYTE *)(a1 + 45) = qword_5002DE8;
    }
  }
  if ( !(_DWORD)qword_5002D08 )
    *(_BYTE *)(a1 + 49) = 0;
  v127 = sub_C52410();
  v128 = v127 + 1;
  v129 = sub_C959E0();
  v130 = (_QWORD *)v127[2];
  if ( v130 )
  {
    v131 = v127 + 1;
    do
    {
      while ( 1 )
      {
        v132 = v130[2];
        v133 = v130[3];
        if ( v129 <= v130[4] )
          break;
        v130 = (_QWORD *)v130[3];
        if ( !v133 )
          goto LABEL_178;
      }
      v131 = v130;
      v130 = (_QWORD *)v130[2];
    }
    while ( v132 );
LABEL_178:
    if ( v131 != v128 && v129 >= v131[4] )
      v128 = v131;
  }
  if ( v128 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v134 = v128[7];
    if ( v134 )
    {
      v135 = v128 + 6;
      do
      {
        while ( 1 )
        {
          v136 = *(_QWORD *)(v134 + 16);
          v137 = *(_QWORD *)(v134 + 24);
          if ( *(_DWORD *)(v134 + 32) >= dword_5002908 )
            break;
          v134 = *(_QWORD *)(v134 + 24);
          if ( !v137 )
            goto LABEL_187;
        }
        v135 = (_QWORD *)v134;
        v134 = *(_QWORD *)(v134 + 16);
      }
      while ( v136 );
LABEL_187:
      if ( v135 != v128 + 6 && dword_5002908 >= *((_DWORD *)v135 + 8) && *((int *)v135 + 9) > 0 )
        *(_BYTE *)(a1 + 50) = qword_5002988;
    }
  }
  v138 = sub_C52410();
  v139 = v138 + 1;
  v140 = sub_C959E0();
  v141 = (_QWORD *)v138[2];
  if ( v141 )
  {
    v142 = v138 + 1;
    do
    {
      while ( 1 )
      {
        v143 = v141[2];
        v144 = v141[3];
        if ( v140 <= v141[4] )
          break;
        v141 = (_QWORD *)v141[3];
        if ( !v144 )
          goto LABEL_196;
      }
      v142 = v141;
      v141 = (_QWORD *)v141[2];
    }
    while ( v143 );
LABEL_196:
    if ( v142 != v139 && v140 >= v142[4] )
      v139 = v142;
  }
  if ( v139 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v145 = v139[7];
    if ( v145 )
    {
      v146 = v139 + 6;
      do
      {
        while ( 1 )
        {
          v147 = *(_QWORD *)(v145 + 16);
          v148 = *(_QWORD *)(v145 + 24);
          if ( *(_DWORD *)(v145 + 32) >= dword_50032A8 )
            break;
          v145 = *(_QWORD *)(v145 + 24);
          if ( !v148 )
            goto LABEL_205;
        }
        v146 = (_QWORD *)v145;
        v145 = *(_QWORD *)(v145 + 16);
      }
      while ( v147 );
LABEL_205:
      if ( v146 != v139 + 6 && dword_50032A8 >= *((_DWORD *)v146 + 8) && *((int *)v146 + 9) > 0 )
        *(_DWORD *)(a1 + 56) = qword_5003328;
    }
  }
  if ( BYTE4(a9) )
  {
    *(_DWORD *)a1 = a9;
    *(_DWORD *)(a1 + 12) = a9;
  }
  if ( BYTE4(a10) )
    *(_DWORD *)(a1 + 20) = a10;
  if ( HIBYTE(a11) )
    *(_BYTE *)(a1 + 44) = a11;
  if ( HIBYTE(a12) )
    *(_BYTE *)(a1 + 45) = a12;
  if ( HIBYTE(a13) )
    *(_BYTE *)(a1 + 49) = a13;
  if ( BYTE4(a14) )
    *(_DWORD *)(a1 + 36) = a14;
  return a1;
}
