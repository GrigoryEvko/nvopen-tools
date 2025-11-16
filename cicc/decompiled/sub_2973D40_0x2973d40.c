// Function: sub_2973D40
// Address: 0x2973d40
//
unsigned __int64 __fastcall sub_2973D40(__int64 a1)
{
  _QWORD *v2; // r13
  _QWORD *v3; // r12
  unsigned __int64 v4; // rsi
  _QWORD *v5; // rax
  _QWORD *v6; // rdi
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rax
  _QWORD *v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // rdx
  _QWORD *v13; // r13
  _QWORD *v14; // r12
  unsigned __int64 v15; // rsi
  _QWORD *v16; // rax
  _QWORD *v17; // rdi
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // rax
  _QWORD *v21; // rdi
  __int64 v22; // rcx
  __int64 v23; // rdx
  _QWORD *v24; // r13
  _QWORD *v25; // r12
  unsigned __int64 v26; // rsi
  _QWORD *v27; // rax
  _QWORD *v28; // rdi
  __int64 v29; // rcx
  __int64 v30; // rdx
  __int64 v31; // rax
  _QWORD *v32; // rdi
  __int64 v33; // rcx
  __int64 v34; // rdx
  _QWORD *v35; // r13
  _QWORD *v36; // r12
  unsigned __int64 v37; // rsi
  _QWORD *v38; // rax
  _QWORD *v39; // rdi
  __int64 v40; // rcx
  __int64 v41; // rdx
  __int64 v42; // rax
  _QWORD *v43; // rdi
  __int64 v44; // rcx
  __int64 v45; // rdx
  _QWORD *v46; // r13
  _QWORD *v47; // r12
  unsigned __int64 v48; // rsi
  _QWORD *v49; // rax
  _QWORD *v50; // rdi
  __int64 v51; // rcx
  __int64 v52; // rdx
  __int64 v53; // rax
  _QWORD *v54; // rdi
  __int64 v55; // rcx
  __int64 v56; // rdx
  _QWORD *v57; // r13
  _QWORD *v58; // r12
  unsigned __int64 v59; // rsi
  _QWORD *v60; // rax
  _QWORD *v61; // rdi
  __int64 v62; // rcx
  __int64 v63; // rdx
  __int64 v64; // rax
  _QWORD *v65; // rdi
  __int64 v66; // rcx
  __int64 v67; // rdx
  _QWORD *v68; // r13
  _QWORD *v69; // r12
  unsigned __int64 v70; // rsi
  _QWORD *v71; // rax
  _QWORD *v72; // rdi
  __int64 v73; // rcx
  __int64 v74; // rdx
  __int64 v75; // rax
  _QWORD *v76; // rdi
  __int64 v77; // rcx
  __int64 v78; // rdx
  _QWORD *v79; // r13
  _QWORD *v80; // r12
  unsigned __int64 v81; // rsi
  _QWORD *v82; // rax
  _QWORD *v83; // rdi
  __int64 v84; // rcx
  __int64 v85; // rdx
  __int64 v86; // rax
  _QWORD *v87; // rdi
  __int64 v88; // rcx
  __int64 v89; // rdx
  _QWORD *v90; // r13
  char *v91; // r12
  unsigned __int64 v92; // rsi
  char *v93; // rax
  char *v94; // rdi
  __int64 v95; // rcx
  __int64 v96; // rdx
  unsigned __int64 result; // rax
  char *v98; // rdi
  __int64 v99; // rcx
  __int64 v100; // rdx

  v2 = sub_C52410();
  v3 = v2 + 1;
  v4 = sub_C959E0();
  v5 = (_QWORD *)v2[2];
  if ( v5 )
  {
    v6 = v2 + 1;
    do
    {
      while ( 1 )
      {
        v7 = v5[2];
        v8 = v5[3];
        if ( v4 <= v5[4] )
          break;
        v5 = (_QWORD *)v5[3];
        if ( !v8 )
          goto LABEL_6;
      }
      v6 = v5;
      v5 = (_QWORD *)v5[2];
    }
    while ( v7 );
LABEL_6:
    if ( v3 != v6 && v4 >= v6[4] )
      v3 = v6;
  }
  if ( v3 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v9 = v3[7];
    if ( v9 )
    {
      v10 = v3 + 6;
      do
      {
        while ( 1 )
        {
          v11 = *(_QWORD *)(v9 + 16);
          v12 = *(_QWORD *)(v9 + 24);
          if ( *(_DWORD *)(v9 + 32) >= dword_5006D28 )
            break;
          v9 = *(_QWORD *)(v9 + 24);
          if ( !v12 )
            goto LABEL_15;
        }
        v10 = (_QWORD *)v9;
        v9 = *(_QWORD *)(v9 + 16);
      }
      while ( v11 );
LABEL_15:
      if ( v3 + 6 != v10 && dword_5006D28 >= *((_DWORD *)v10 + 8) && *((_DWORD *)v10 + 9) )
        *(_DWORD *)a1 = qword_5006DA8;
    }
  }
  v13 = sub_C52410();
  v14 = v13 + 1;
  v15 = sub_C959E0();
  v16 = (_QWORD *)v13[2];
  if ( v16 )
  {
    v17 = v13 + 1;
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
          goto LABEL_22;
      }
      v17 = v16;
      v16 = (_QWORD *)v16[2];
    }
    while ( v18 );
LABEL_22:
    if ( v14 != v17 && v15 >= v17[4] )
      v14 = v17;
  }
  if ( v14 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v20 = v14[7];
    if ( v20 )
    {
      v21 = v14 + 6;
      do
      {
        while ( 1 )
        {
          v22 = *(_QWORD *)(v20 + 16);
          v23 = *(_QWORD *)(v20 + 24);
          if ( *(_DWORD *)(v20 + 32) >= dword_50069A8 )
            break;
          v20 = *(_QWORD *)(v20 + 24);
          if ( !v23 )
            goto LABEL_31;
        }
        v21 = (_QWORD *)v20;
        v20 = *(_QWORD *)(v20 + 16);
      }
      while ( v22 );
LABEL_31:
      if ( v14 + 6 != v21 && dword_50069A8 >= *((_DWORD *)v21 + 8) && *((_DWORD *)v21 + 9) )
        *(_BYTE *)(a1 + 4) = qword_5006A28;
    }
  }
  v24 = sub_C52410();
  v25 = v24 + 1;
  v26 = sub_C959E0();
  v27 = (_QWORD *)v24[2];
  if ( v27 )
  {
    v28 = v24 + 1;
    do
    {
      while ( 1 )
      {
        v29 = v27[2];
        v30 = v27[3];
        if ( v26 <= v27[4] )
          break;
        v27 = (_QWORD *)v27[3];
        if ( !v30 )
          goto LABEL_38;
      }
      v28 = v27;
      v27 = (_QWORD *)v27[2];
    }
    while ( v29 );
LABEL_38:
    if ( v25 != v28 && v26 >= v28[4] )
      v25 = v28;
  }
  if ( v25 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v31 = v25[7];
    if ( v31 )
    {
      v32 = v25 + 6;
      do
      {
        while ( 1 )
        {
          v33 = *(_QWORD *)(v31 + 16);
          v34 = *(_QWORD *)(v31 + 24);
          if ( *(_DWORD *)(v31 + 32) >= dword_5006B68 )
            break;
          v31 = *(_QWORD *)(v31 + 24);
          if ( !v34 )
            goto LABEL_47;
        }
        v32 = (_QWORD *)v31;
        v31 = *(_QWORD *)(v31 + 16);
      }
      while ( v33 );
LABEL_47:
      if ( v25 + 6 != v32 && dword_5006B68 >= *((_DWORD *)v32 + 8) && *((_DWORD *)v32 + 9) )
        *(_BYTE *)(a1 + 5) = qword_5006BE8;
    }
  }
  v35 = sub_C52410();
  v36 = v35 + 1;
  v37 = sub_C959E0();
  v38 = (_QWORD *)v35[2];
  if ( v38 )
  {
    v39 = v35 + 1;
    do
    {
      while ( 1 )
      {
        v40 = v38[2];
        v41 = v38[3];
        if ( v37 <= v38[4] )
          break;
        v38 = (_QWORD *)v38[3];
        if ( !v41 )
          goto LABEL_54;
      }
      v39 = v38;
      v38 = (_QWORD *)v38[2];
    }
    while ( v40 );
LABEL_54:
    if ( v36 != v39 && v37 >= v39[4] )
      v36 = v39;
  }
  if ( v36 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v42 = v36[7];
    if ( v42 )
    {
      v43 = v36 + 6;
      do
      {
        while ( 1 )
        {
          v44 = *(_QWORD *)(v42 + 16);
          v45 = *(_QWORD *)(v42 + 24);
          if ( *(_DWORD *)(v42 + 32) >= dword_5006A88 )
            break;
          v42 = *(_QWORD *)(v42 + 24);
          if ( !v45 )
            goto LABEL_63;
        }
        v43 = (_QWORD *)v42;
        v42 = *(_QWORD *)(v42 + 16);
      }
      while ( v44 );
LABEL_63:
      if ( v36 + 6 != v43 && dword_5006A88 >= *((_DWORD *)v43 + 8) && *((_DWORD *)v43 + 9) )
        *(_BYTE *)(a1 + 6) = qword_5006B08;
    }
  }
  v46 = sub_C52410();
  v47 = v46 + 1;
  v48 = sub_C959E0();
  v49 = (_QWORD *)v46[2];
  if ( v49 )
  {
    v50 = v46 + 1;
    do
    {
      while ( 1 )
      {
        v51 = v49[2];
        v52 = v49[3];
        if ( v48 <= v49[4] )
          break;
        v49 = (_QWORD *)v49[3];
        if ( !v52 )
          goto LABEL_70;
      }
      v50 = v49;
      v49 = (_QWORD *)v49[2];
    }
    while ( v51 );
LABEL_70:
    if ( v47 != v50 && v48 >= v50[4] )
      v47 = v50;
  }
  if ( v47 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v53 = v47[7];
    if ( v53 )
    {
      v54 = v47 + 6;
      do
      {
        while ( 1 )
        {
          v55 = *(_QWORD *)(v53 + 16);
          v56 = *(_QWORD *)(v53 + 24);
          if ( *(_DWORD *)(v53 + 32) >= dword_5006C48 )
            break;
          v53 = *(_QWORD *)(v53 + 24);
          if ( !v56 )
            goto LABEL_79;
        }
        v54 = (_QWORD *)v53;
        v53 = *(_QWORD *)(v53 + 16);
      }
      while ( v55 );
LABEL_79:
      if ( v47 + 6 != v54 && dword_5006C48 >= *((_DWORD *)v54 + 8) && *((_DWORD *)v54 + 9) )
        *(_BYTE *)(a1 + 7) = qword_5006CC8;
    }
  }
  v57 = sub_C52410();
  v58 = v57 + 1;
  v59 = sub_C959E0();
  v60 = (_QWORD *)v57[2];
  if ( v60 )
  {
    v61 = v57 + 1;
    do
    {
      while ( 1 )
      {
        v62 = v60[2];
        v63 = v60[3];
        if ( v59 <= v60[4] )
          break;
        v60 = (_QWORD *)v60[3];
        if ( !v63 )
          goto LABEL_86;
      }
      v61 = v60;
      v60 = (_QWORD *)v60[2];
    }
    while ( v62 );
LABEL_86:
    if ( v58 != v61 && v59 >= v61[4] )
      v58 = v61;
  }
  if ( v58 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v64 = v58[7];
    if ( v64 )
    {
      v65 = v58 + 6;
      do
      {
        while ( 1 )
        {
          v66 = *(_QWORD *)(v64 + 16);
          v67 = *(_QWORD *)(v64 + 24);
          if ( *(_DWORD *)(v64 + 32) >= dword_50068C8 )
            break;
          v64 = *(_QWORD *)(v64 + 24);
          if ( !v67 )
            goto LABEL_95;
        }
        v65 = (_QWORD *)v64;
        v64 = *(_QWORD *)(v64 + 16);
      }
      while ( v66 );
LABEL_95:
      if ( v58 + 6 != v65 && dword_50068C8 >= *((_DWORD *)v65 + 8) && *((_DWORD *)v65 + 9) )
        *(_BYTE *)(a1 + 8) = qword_5006948;
    }
  }
  v68 = sub_C52410();
  v69 = v68 + 1;
  v70 = sub_C959E0();
  v71 = (_QWORD *)v68[2];
  if ( v71 )
  {
    v72 = v68 + 1;
    do
    {
      while ( 1 )
      {
        v73 = v71[2];
        v74 = v71[3];
        if ( v70 <= v71[4] )
          break;
        v71 = (_QWORD *)v71[3];
        if ( !v74 )
          goto LABEL_102;
      }
      v72 = v71;
      v71 = (_QWORD *)v71[2];
    }
    while ( v73 );
LABEL_102:
    if ( v69 != v72 && v70 >= v72[4] )
      v69 = v72;
  }
  if ( v69 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v75 = v69[7];
    if ( v75 )
    {
      v76 = v69 + 6;
      do
      {
        while ( 1 )
        {
          v77 = *(_QWORD *)(v75 + 16);
          v78 = *(_QWORD *)(v75 + 24);
          if ( *(_DWORD *)(v75 + 32) >= dword_50067E8 )
            break;
          v75 = *(_QWORD *)(v75 + 24);
          if ( !v78 )
            goto LABEL_111;
        }
        v76 = (_QWORD *)v75;
        v75 = *(_QWORD *)(v75 + 16);
      }
      while ( v77 );
LABEL_111:
      if ( v69 + 6 != v76 && dword_50067E8 >= *((_DWORD *)v76 + 8) && *((_DWORD *)v76 + 9) )
        *(_BYTE *)(a1 + 9) = qword_5006868;
    }
  }
  v79 = sub_C52410();
  v80 = v79 + 1;
  v81 = sub_C959E0();
  v82 = (_QWORD *)v79[2];
  if ( v82 )
  {
    v83 = v79 + 1;
    do
    {
      while ( 1 )
      {
        v84 = v82[2];
        v85 = v82[3];
        if ( v81 <= v82[4] )
          break;
        v82 = (_QWORD *)v82[3];
        if ( !v85 )
          goto LABEL_118;
      }
      v83 = v82;
      v82 = (_QWORD *)v82[2];
    }
    while ( v84 );
LABEL_118:
    if ( v80 != v83 && v81 >= v83[4] )
      v80 = v83;
  }
  if ( v80 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v86 = v80[7];
    if ( v86 )
    {
      v87 = v80 + 6;
      do
      {
        while ( 1 )
        {
          v88 = *(_QWORD *)(v86 + 16);
          v89 = *(_QWORD *)(v86 + 24);
          if ( *(_DWORD *)(v86 + 32) >= dword_5006708 )
            break;
          v86 = *(_QWORD *)(v86 + 24);
          if ( !v89 )
            goto LABEL_127;
        }
        v87 = (_QWORD *)v86;
        v86 = *(_QWORD *)(v86 + 16);
      }
      while ( v88 );
LABEL_127:
      if ( v80 + 6 != v87 && dword_5006708 >= *((_DWORD *)v87 + 8) && *((_DWORD *)v87 + 9) )
        *(_BYTE *)(a1 + 10) = qword_5006788;
    }
  }
  v90 = sub_C52410();
  v91 = (char *)(v90 + 1);
  v92 = sub_C959E0();
  v93 = (char *)v90[2];
  if ( v93 )
  {
    v94 = (char *)(v90 + 1);
    do
    {
      while ( 1 )
      {
        v95 = *((_QWORD *)v93 + 2);
        v96 = *((_QWORD *)v93 + 3);
        if ( v92 <= *((_QWORD *)v93 + 4) )
          break;
        v93 = (char *)*((_QWORD *)v93 + 3);
        if ( !v96 )
          goto LABEL_134;
      }
      v94 = v93;
      v93 = (char *)*((_QWORD *)v93 + 2);
    }
    while ( v95 );
LABEL_134:
    if ( v91 != v94 && v92 >= *((_QWORD *)v94 + 4) )
      v91 = v94;
  }
  result = (unsigned __int64)sub_C52410() + 8;
  if ( v91 != (char *)result )
  {
    result = *((_QWORD *)v91 + 7);
    if ( result )
    {
      v98 = v91 + 48;
      do
      {
        while ( 1 )
        {
          v99 = *(_QWORD *)(result + 16);
          v100 = *(_QWORD *)(result + 24);
          if ( *(_DWORD *)(result + 32) >= dword_5006628 )
            break;
          result = *(_QWORD *)(result + 24);
          if ( !v100 )
            goto LABEL_143;
        }
        v98 = (char *)result;
        result = *(_QWORD *)(result + 16);
      }
      while ( v99 );
LABEL_143:
      if ( v91 + 48 != v98 && dword_5006628 >= *((_DWORD *)v98 + 8) )
      {
        if ( *((_DWORD *)v98 + 9) )
        {
          result = (unsigned __int8)qword_50066A8;
          *(_BYTE *)(a1 + 15) = qword_50066A8;
        }
      }
    }
  }
  return result;
}
