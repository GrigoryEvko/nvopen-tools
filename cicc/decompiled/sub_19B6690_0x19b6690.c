// Function: sub_19B6690
// Address: 0x19b6690
//
__int64 __fastcall sub_19B6690(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        int a5,
        int *a6,
        __int64 a7,
        _BYTE *a8,
        _BYTE *a9,
        _BYTE *a10,
        _BYTE *a11)
{
  int v11; // eax
  __int64 v14; // r13
  unsigned __int64 v15; // rdi
  _QWORD *v16; // rax
  _DWORD *v17; // r8
  __int64 v18; // rsi
  __int64 v19; // rcx
  unsigned __int64 v20; // rdi
  _QWORD *v21; // rax
  _DWORD *v22; // r8
  __int64 v23; // rsi
  __int64 v24; // rcx
  unsigned __int64 v25; // rdi
  _QWORD *v26; // rax
  _DWORD *v27; // r8
  __int64 v28; // rsi
  __int64 v29; // rcx
  unsigned __int64 v30; // rdi
  _QWORD *v31; // rax
  _DWORD *v32; // r8
  __int64 v33; // rsi
  __int64 v34; // rcx
  unsigned __int64 v35; // rdi
  _QWORD *v36; // rax
  _DWORD *v37; // r8
  __int64 v38; // rsi
  __int64 v39; // rcx
  unsigned __int64 v40; // rdi
  _QWORD *v41; // rax
  _DWORD *v42; // r8
  __int64 v43; // rsi
  __int64 v44; // rcx
  unsigned __int64 v45; // rdi
  _QWORD *v46; // rax
  _DWORD *v47; // r8
  __int64 v48; // rsi
  __int64 v49; // rcx
  unsigned __int64 v50; // rdi
  _QWORD *v51; // rax
  _DWORD *v52; // r8
  __int64 v53; // rsi
  __int64 v54; // rcx
  unsigned __int64 v55; // rdi
  _QWORD *v56; // rax
  _DWORD *v57; // r8
  __int64 v58; // rsi
  __int64 v59; // rcx
  unsigned __int64 v60; // rdi
  _QWORD *v61; // rax
  _DWORD *v62; // r8
  __int64 v63; // rsi
  __int64 v64; // rcx
  unsigned __int64 v65; // rdi
  _QWORD *v66; // rax
  _DWORD *v67; // r8
  __int64 v68; // rsi
  __int64 v69; // rcx
  int v70; // eax
  __int64 v72; // rax
  _DWORD *v73; // rdi
  __int64 v74; // rcx
  __int64 v75; // rdx
  __int64 v76; // rax
  _DWORD *v77; // r9
  _DWORD *v78; // r8
  __int64 v79; // rsi
  __int64 v80; // rcx
  __int64 v81; // rax
  _DWORD *v82; // r9
  _DWORD *v83; // r8
  __int64 v84; // rsi
  __int64 v85; // rcx
  __int64 v86; // rax
  _DWORD *v87; // r9
  _DWORD *v88; // r8
  __int64 v89; // rsi
  __int64 v90; // rcx
  __int64 v91; // rax
  _DWORD *v92; // r9
  _DWORD *v93; // r8
  __int64 v94; // rsi
  __int64 v95; // rcx
  __int64 v96; // rax
  _DWORD *v97; // r9
  _DWORD *v98; // r8
  __int64 v99; // rsi
  __int64 v100; // rcx
  __int64 v101; // rax
  _DWORD *v102; // r9
  _DWORD *v103; // r8
  __int64 v104; // rsi
  __int64 v105; // rcx
  __int64 v106; // rax
  _DWORD *v107; // r9
  _DWORD *v108; // r8
  __int64 v109; // rsi
  __int64 v110; // rcx
  __int64 v111; // rax
  _DWORD *v112; // r9
  _DWORD *v113; // r8
  __int64 v114; // rsi
  __int64 v115; // rcx
  __int64 v116; // rax
  _DWORD *v117; // r9
  _DWORD *v118; // r8
  __int64 v119; // rsi
  __int64 v120; // rcx
  __int64 v121; // rax
  _DWORD *v122; // r9
  _DWORD *v123; // r8
  __int64 v124; // rsi
  __int64 v125; // rcx

  v11 = 405;
  if ( a5 <= 1 )
    v11 = 150;
  *(_QWORD *)(a1 + 4) = 400;
  *(_QWORD *)(a1 + 12) = 150;
  *(_DWORD *)a1 = v11;
  *(_QWORD *)(a1 + 28) = 0xFFFFFFFF00000004LL;
  *(_QWORD *)(a1 + 36) = 0x2FFFFFFFFLL;
  *(_QWORD *)(a1 + 44) = 0x1000000010000LL;
  *(_QWORD *)(a1 + 20) = 0;
  *(_BYTE *)(a1 + 52) = 0;
  *(_DWORD *)(a1 + 56) = 60;
  sub_14A2A00(a4);
  v14 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 56LL) + 112LL;
  if ( (unsigned __int8)sub_1560180(v14, 34) || (unsigned __int8)sub_1560180(v14, 17) )
  {
    *(_DWORD *)a1 = *(_DWORD *)(a1 + 8);
    *(_DWORD *)(a1 + 12) = *(_DWORD *)(a1 + 16);
  }
  v15 = sub_16D5D50();
  v16 = *(_QWORD **)&dword_4FA0208[2];
  if ( *(_QWORD *)&dword_4FA0208[2] )
  {
    v17 = dword_4FA0208;
    do
    {
      while ( 1 )
      {
        v18 = v16[2];
        v19 = v16[3];
        if ( v15 <= v16[4] )
          break;
        v16 = (_QWORD *)v16[3];
        if ( !v19 )
          goto LABEL_10;
      }
      v17 = v16;
      v16 = (_QWORD *)v16[2];
    }
    while ( v18 );
LABEL_10:
    if ( v17 != dword_4FA0208 && v15 >= *((_QWORD *)v17 + 4) )
    {
      v86 = *((_QWORD *)v17 + 7);
      v87 = v17 + 12;
      if ( v86 )
      {
        v88 = v17 + 12;
        do
        {
          while ( 1 )
          {
            v89 = *(_QWORD *)(v86 + 16);
            v90 = *(_QWORD *)(v86 + 24);
            if ( *(_DWORD *)(v86 + 32) >= dword_4FB3228 )
              break;
            v86 = *(_QWORD *)(v86 + 24);
            if ( !v90 )
              goto LABEL_129;
          }
          v88 = (_DWORD *)v86;
          v86 = *(_QWORD *)(v86 + 16);
        }
        while ( v89 );
LABEL_129:
        if ( v87 != v88 && dword_4FB3228 >= v88[8] && (int)v88[9] > 0 )
          *(_DWORD *)a1 = dword_4FB32C0;
      }
    }
  }
  v20 = sub_16D5D50();
  v21 = *(_QWORD **)&dword_4FA0208[2];
  if ( *(_QWORD *)&dword_4FA0208[2] )
  {
    v22 = dword_4FA0208;
    do
    {
      while ( 1 )
      {
        v23 = v21[2];
        v24 = v21[3];
        if ( v20 <= v21[4] )
          break;
        v21 = (_QWORD *)v21[3];
        if ( !v24 )
          goto LABEL_17;
      }
      v22 = v21;
      v21 = (_QWORD *)v21[2];
    }
    while ( v23 );
LABEL_17:
    if ( v22 != dword_4FA0208 && v20 >= *((_QWORD *)v22 + 4) )
    {
      v91 = *((_QWORD *)v22 + 7);
      v92 = v22 + 12;
      if ( v91 )
      {
        v93 = v22 + 12;
        do
        {
          while ( 1 )
          {
            v94 = *(_QWORD *)(v91 + 16);
            v95 = *(_QWORD *)(v91 + 24);
            if ( *(_DWORD *)(v91 + 32) >= dword_4FB3148 )
              break;
            v91 = *(_QWORD *)(v91 + 24);
            if ( !v95 )
              goto LABEL_138;
          }
          v93 = (_DWORD *)v91;
          v91 = *(_QWORD *)(v91 + 16);
        }
        while ( v94 );
LABEL_138:
        if ( v92 != v93 && dword_4FB3148 >= v93[8] && (int)v93[9] > 0 )
          *(_DWORD *)(a1 + 12) = dword_4FB31E0;
      }
    }
  }
  v25 = sub_16D5D50();
  v26 = *(_QWORD **)&dword_4FA0208[2];
  if ( *(_QWORD *)&dword_4FA0208[2] )
  {
    v27 = dword_4FA0208;
    do
    {
      while ( 1 )
      {
        v28 = v26[2];
        v29 = v26[3];
        if ( v25 <= v26[4] )
          break;
        v26 = (_QWORD *)v26[3];
        if ( !v29 )
          goto LABEL_24;
      }
      v27 = v26;
      v26 = (_QWORD *)v26[2];
    }
    while ( v28 );
LABEL_24:
    if ( v27 != dword_4FA0208 && v25 >= *((_QWORD *)v27 + 4) )
    {
      v96 = *((_QWORD *)v27 + 7);
      v97 = v27 + 12;
      if ( v96 )
      {
        v98 = v27 + 12;
        do
        {
          while ( 1 )
          {
            v99 = *(_QWORD *)(v96 + 16);
            v100 = *(_QWORD *)(v96 + 24);
            if ( *(_DWORD *)(v96 + 32) >= dword_4FB3068 )
              break;
            v96 = *(_QWORD *)(v96 + 24);
            if ( !v100 )
              goto LABEL_147;
          }
          v98 = (_DWORD *)v96;
          v96 = *(_QWORD *)(v96 + 16);
        }
        while ( v99 );
LABEL_147:
        if ( v97 != v98 && dword_4FB3068 >= v98[8] && (int)v98[9] > 0 )
          *(_DWORD *)(a1 + 4) = dword_4FB3100;
      }
    }
  }
  v30 = sub_16D5D50();
  v31 = *(_QWORD **)&dword_4FA0208[2];
  if ( *(_QWORD *)&dword_4FA0208[2] )
  {
    v32 = dword_4FA0208;
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
          goto LABEL_31;
      }
      v32 = v31;
      v31 = (_QWORD *)v31[2];
    }
    while ( v33 );
LABEL_31:
    if ( v32 != dword_4FA0208 && v30 >= *((_QWORD *)v32 + 4) )
    {
      v101 = *((_QWORD *)v32 + 7);
      v102 = v32 + 12;
      if ( v101 )
      {
        v103 = v32 + 12;
        do
        {
          while ( 1 )
          {
            v104 = *(_QWORD *)(v101 + 16);
            v105 = *(_QWORD *)(v101 + 24);
            if ( *(_DWORD *)(v101 + 32) >= dword_4FB2DC8 )
              break;
            v101 = *(_QWORD *)(v101 + 24);
            if ( !v105 )
              goto LABEL_156;
          }
          v103 = (_DWORD *)v101;
          v101 = *(_QWORD *)(v101 + 16);
        }
        while ( v104 );
LABEL_156:
        if ( v102 != v103 && dword_4FB2DC8 >= v103[8] && (int)v103[9] > 0 )
          *(_DWORD *)(a1 + 32) = dword_4FB2E60;
      }
    }
  }
  v35 = sub_16D5D50();
  v36 = *(_QWORD **)&dword_4FA0208[2];
  if ( *(_QWORD *)&dword_4FA0208[2] )
  {
    v37 = dword_4FA0208;
    do
    {
      while ( 1 )
      {
        v38 = v36[2];
        v39 = v36[3];
        if ( v35 <= v36[4] )
          break;
        v36 = (_QWORD *)v36[3];
        if ( !v39 )
          goto LABEL_38;
      }
      v37 = v36;
      v36 = (_QWORD *)v36[2];
    }
    while ( v38 );
LABEL_38:
    if ( v37 != dword_4FA0208 && v35 >= *((_QWORD *)v37 + 4) )
    {
      v106 = *((_QWORD *)v37 + 7);
      v107 = v37 + 12;
      if ( v106 )
      {
        v108 = v37 + 12;
        do
        {
          while ( 1 )
          {
            v109 = *(_QWORD *)(v106 + 16);
            v110 = *(_QWORD *)(v106 + 24);
            if ( *(_DWORD *)(v106 + 32) >= dword_4FB2CE8 )
              break;
            v106 = *(_QWORD *)(v106 + 24);
            if ( !v110 )
              goto LABEL_165;
          }
          v108 = (_DWORD *)v106;
          v106 = *(_QWORD *)(v106 + 16);
        }
        while ( v109 );
LABEL_165:
        if ( v107 != v108 && dword_4FB2CE8 >= v108[8] && (int)v108[9] > 0 )
          *(_DWORD *)(a1 + 36) = dword_4FB2D80;
      }
    }
  }
  v40 = sub_16D5D50();
  v41 = *(_QWORD **)&dword_4FA0208[2];
  if ( *(_QWORD *)&dword_4FA0208[2] )
  {
    v42 = dword_4FA0208;
    do
    {
      while ( 1 )
      {
        v43 = v41[2];
        v44 = v41[3];
        if ( v40 <= v41[4] )
          break;
        v41 = (_QWORD *)v41[3];
        if ( !v44 )
          goto LABEL_45;
      }
      v42 = v41;
      v41 = (_QWORD *)v41[2];
    }
    while ( v43 );
LABEL_45:
    if ( v42 != dword_4FA0208 && v40 >= *((_QWORD *)v42 + 4) )
    {
      v111 = *((_QWORD *)v42 + 7);
      v112 = v42 + 12;
      if ( v111 )
      {
        v113 = v42 + 12;
        do
        {
          while ( 1 )
          {
            v114 = *(_QWORD *)(v111 + 16);
            v115 = *(_QWORD *)(v111 + 24);
            if ( *(_DWORD *)(v111 + 32) >= dword_4FB2C08 )
              break;
            v111 = *(_QWORD *)(v111 + 24);
            if ( !v115 )
              goto LABEL_174;
          }
          v113 = (_DWORD *)v111;
          v111 = *(_QWORD *)(v111 + 16);
        }
        while ( v114 );
LABEL_174:
        if ( v112 != v113 && dword_4FB2C08 >= v113[8] && (int)v113[9] > 0 )
          *(_DWORD *)(a1 + 24) = dword_4FB2CA0;
      }
    }
  }
  v45 = sub_16D5D50();
  v46 = *(_QWORD **)&dword_4FA0208[2];
  if ( *(_QWORD *)&dword_4FA0208[2] )
  {
    v47 = dword_4FA0208;
    do
    {
      while ( 1 )
      {
        v48 = v46[2];
        v49 = v46[3];
        if ( v45 <= v46[4] )
          break;
        v46 = (_QWORD *)v46[3];
        if ( !v49 )
          goto LABEL_52;
      }
      v47 = v46;
      v46 = (_QWORD *)v46[2];
    }
    while ( v48 );
LABEL_52:
    if ( v47 != dword_4FA0208 && v45 >= *((_QWORD *)v47 + 4) )
    {
      v116 = *((_QWORD *)v47 + 7);
      v117 = v47 + 12;
      if ( v116 )
      {
        v118 = v47 + 12;
        do
        {
          while ( 1 )
          {
            v119 = *(_QWORD *)(v116 + 16);
            v120 = *(_QWORD *)(v116 + 24);
            if ( *(_DWORD *)(v116 + 32) >= dword_4FB2B28 )
              break;
            v116 = *(_QWORD *)(v116 + 24);
            if ( !v120 )
              goto LABEL_183;
          }
          v118 = (_DWORD *)v116;
          v116 = *(_QWORD *)(v116 + 16);
        }
        while ( v119 );
LABEL_183:
        if ( v117 != v118 && dword_4FB2B28 >= v118[8] && (int)v118[9] > 0 )
          *(_BYTE *)(a1 + 44) = byte_4FB2BC0;
      }
    }
  }
  v50 = sub_16D5D50();
  v51 = *(_QWORD **)&dword_4FA0208[2];
  if ( *(_QWORD *)&dword_4FA0208[2] )
  {
    v52 = dword_4FA0208;
    do
    {
      while ( 1 )
      {
        v53 = v51[2];
        v54 = v51[3];
        if ( v50 <= v51[4] )
          break;
        v51 = (_QWORD *)v51[3];
        if ( !v54 )
          goto LABEL_59;
      }
      v52 = v51;
      v51 = (_QWORD *)v51[2];
    }
    while ( v53 );
LABEL_59:
    if ( v52 != dword_4FA0208 && v50 >= *((_QWORD *)v52 + 4) )
    {
      v121 = *((_QWORD *)v52 + 7);
      v122 = v52 + 12;
      if ( v121 )
      {
        v123 = v52 + 12;
        do
        {
          while ( 1 )
          {
            v124 = *(_QWORD *)(v121 + 16);
            v125 = *(_QWORD *)(v121 + 24);
            if ( *(_DWORD *)(v121 + 32) >= dword_4FB2A48 )
              break;
            v121 = *(_QWORD *)(v121 + 24);
            if ( !v125 )
              goto LABEL_192;
          }
          v123 = (_DWORD *)v121;
          v121 = *(_QWORD *)(v121 + 16);
        }
        while ( v124 );
LABEL_192:
        if ( v122 != v123 && dword_4FB2A48 >= v123[8] && (int)v123[9] > 0 )
          *(_BYTE *)(a1 + 46) = byte_4FB2AE0;
      }
    }
  }
  v55 = sub_16D5D50();
  v56 = *(_QWORD **)&dword_4FA0208[2];
  if ( *(_QWORD *)&dword_4FA0208[2] )
  {
    v57 = dword_4FA0208;
    do
    {
      while ( 1 )
      {
        v58 = v56[2];
        v59 = v56[3];
        if ( v55 <= v56[4] )
          break;
        v56 = (_QWORD *)v56[3];
        if ( !v59 )
          goto LABEL_66;
      }
      v57 = v56;
      v56 = (_QWORD *)v56[2];
    }
    while ( v58 );
LABEL_66:
    if ( v57 != dword_4FA0208 && v55 >= *((_QWORD *)v57 + 4) )
    {
      v76 = *((_QWORD *)v57 + 7);
      v77 = v57 + 12;
      if ( v76 )
      {
        v78 = v57 + 12;
        do
        {
          while ( 1 )
          {
            v79 = *(_QWORD *)(v76 + 16);
            v80 = *(_QWORD *)(v76 + 24);
            if ( *(_DWORD *)(v76 + 32) >= dword_4FB2968 )
              break;
            v76 = *(_QWORD *)(v76 + 24);
            if ( !v80 )
              goto LABEL_111;
          }
          v78 = (_DWORD *)v76;
          v76 = *(_QWORD *)(v76 + 16);
        }
        while ( v79 );
LABEL_111:
        if ( v77 != v78 && dword_4FB2968 >= v78[8] && (int)v78[9] > 0 )
          *(_BYTE *)(a1 + 45) = byte_4FB2A00;
      }
    }
  }
  if ( !dword_4FB2920 )
    *(_BYTE *)(a1 + 49) = 0;
  v60 = sub_16D5D50();
  v61 = *(_QWORD **)&dword_4FA0208[2];
  if ( *(_QWORD *)&dword_4FA0208[2] )
  {
    v62 = dword_4FA0208;
    do
    {
      while ( 1 )
      {
        v63 = v61[2];
        v64 = v61[3];
        if ( v60 <= v61[4] )
          break;
        v61 = (_QWORD *)v61[3];
        if ( !v64 )
          goto LABEL_75;
      }
      v62 = v61;
      v61 = (_QWORD *)v61[2];
    }
    while ( v63 );
LABEL_75:
    if ( v62 != dword_4FA0208 && v60 >= *((_QWORD *)v62 + 4) )
    {
      v81 = *((_QWORD *)v62 + 7);
      v82 = v62 + 12;
      if ( v81 )
      {
        v83 = v62 + 12;
        do
        {
          while ( 1 )
          {
            v84 = *(_QWORD *)(v81 + 16);
            v85 = *(_QWORD *)(v81 + 24);
            if ( *(_DWORD *)(v81 + 32) >= dword_4FB2508 )
              break;
            v81 = *(_QWORD *)(v81 + 24);
            if ( !v85 )
              goto LABEL_120;
          }
          v83 = (_DWORD *)v81;
          v81 = *(_QWORD *)(v81 + 16);
        }
        while ( v84 );
LABEL_120:
        if ( v82 != v83 && dword_4FB2508 >= v83[8] && (int)v83[9] > 0 )
          *(_BYTE *)(a1 + 50) = byte_4FB25A0;
      }
    }
  }
  v65 = sub_16D5D50();
  v66 = *(_QWORD **)&dword_4FA0208[2];
  if ( *(_QWORD *)&dword_4FA0208[2] )
  {
    v67 = dword_4FA0208;
    do
    {
      while ( 1 )
      {
        v68 = v66[2];
        v69 = v66[3];
        if ( v65 <= v66[4] )
          break;
        v66 = (_QWORD *)v66[3];
        if ( !v69 )
          goto LABEL_82;
      }
      v67 = v66;
      v66 = (_QWORD *)v66[2];
    }
    while ( v68 );
LABEL_82:
    if ( v67 != dword_4FA0208 && v65 >= *((_QWORD *)v67 + 4) )
    {
      v72 = *((_QWORD *)v67 + 7);
      if ( v72 )
      {
        v73 = v67 + 12;
        do
        {
          while ( 1 )
          {
            v74 = *(_QWORD *)(v72 + 16);
            v75 = *(_QWORD *)(v72 + 24);
            if ( *(_DWORD *)(v72 + 32) >= dword_4FB2348 )
              break;
            v72 = *(_QWORD *)(v72 + 24);
            if ( !v75 )
              goto LABEL_102;
          }
          v73 = (_DWORD *)v72;
          v72 = *(_QWORD *)(v72 + 16);
        }
        while ( v74 );
LABEL_102:
        if ( v67 + 12 != v73 && dword_4FB2348 >= v73[8] && (int)v73[9] > 0 )
          *(_BYTE *)(a1 + 51) = byte_4FB23E0;
      }
    }
  }
  if ( *((_BYTE *)a6 + 4) )
  {
    v70 = *a6;
    *(_DWORD *)a1 = *a6;
    *(_DWORD *)(a1 + 12) = v70;
  }
  if ( *(_BYTE *)(a7 + 4) )
    *(_DWORD *)(a1 + 20) = *(_DWORD *)a7;
  if ( a8[1] )
    *(_BYTE *)(a1 + 44) = *a8;
  if ( a9[1] )
    *(_BYTE *)(a1 + 45) = *a9;
  if ( a10[1] )
    *(_BYTE *)(a1 + 49) = *a10;
  if ( a11[1] )
    *(_BYTE *)(a1 + 50) = *a11;
  return a1;
}
