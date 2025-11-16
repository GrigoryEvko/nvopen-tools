// Function: sub_272F660
// Address: 0x272f660
//
unsigned __int64 __fastcall sub_272F660(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  __int64 v7; // rax
  __int64 v8; // r15
  __int64 v9; // r13
  __int64 v10; // r12
  __int64 v11; // rbx
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rcx
  __int64 v17; // rcx
  __int64 v18; // r10
  int v19; // ecx
  unsigned __int64 result; // rax
  __int64 v21; // rcx
  unsigned __int64 v22; // rbx
  char **v23; // r12
  __int64 v24; // r13
  char **v25; // rsi
  __int64 v26; // rdi
  __int64 v27; // rcx
  __int64 v28; // r12
  __int64 v29; // rdx
  __int64 v30; // rbx
  __int64 v31; // rdx
  __int64 v32; // r14
  __int64 v33; // rcx
  __int64 v34; // rcx
  __int64 v35; // rsi
  __int64 v36; // r14
  __int64 v37; // rbx
  unsigned __int64 v38; // r12
  char **v39; // rsi
  __int64 v40; // rdi
  __int64 v41; // rcx
  char **v42; // rbx
  __int64 v43; // r12
  unsigned __int64 v44; // r14
  char **v45; // rsi
  __int64 v46; // rdi
  __int64 v47; // rdi
  __int64 v48; // r12
  __int64 v49; // r14
  char **v50; // rsi
  __int64 v51; // rsi
  __int64 v52; // rdi
  char **v53; // rsi
  unsigned __int64 v54; // rcx
  __int64 v55; // rdx
  unsigned __int64 v56; // r14
  __int64 v57; // r12
  char **v58; // rbx
  char **v59; // rsi
  __int64 v60; // rdi
  __int64 v61; // rdx
  __int64 v62; // rcx
  __int64 v63; // r14
  __int64 v64; // rdx
  char **v65; // rbx
  unsigned __int64 v66; // r12
  __int64 v67; // r13
  char **v68; // rsi
  __int64 v69; // rdi
  __int64 v70; // r12
  __int64 v71; // rbx
  __int64 v72; // r14
  __int64 v73; // r10
  __int64 v74; // rax
  __int64 v75; // rcx
  __int64 v76; // rdx
  char **v77; // r12
  unsigned __int64 v78; // r14
  __int64 v79; // rbx
  char **v80; // rsi
  __int64 v81; // rdi
  __int64 v82; // r14
  __int64 v83; // rbx
  __int64 v84; // r14
  char **v85; // r14
  __int64 v86; // rbx
  __int64 v87; // r12
  char **v88; // rsi
  __int64 v89; // rdi
  __int64 v90; // r10
  __int64 v91; // rbx
  __int64 v92; // rdx
  unsigned __int64 v93; // r13
  unsigned __int64 v94; // rbx
  __int64 v95; // r13
  __int64 v96; // [rsp+0h] [rbp-90h]
  unsigned __int64 v97; // [rsp+8h] [rbp-88h]
  __int64 v98; // [rsp+8h] [rbp-88h]
  __int64 v99; // [rsp+10h] [rbp-80h]
  __int64 v100; // [rsp+10h] [rbp-80h]
  __int64 v101; // [rsp+10h] [rbp-80h]
  __int64 v102; // [rsp+10h] [rbp-80h]
  __int64 v103; // [rsp+18h] [rbp-78h]
  int v104; // [rsp+18h] [rbp-78h]
  __int64 v105; // [rsp+18h] [rbp-78h]
  __int64 v106; // [rsp+18h] [rbp-78h]
  __int64 v107; // [rsp+18h] [rbp-78h]
  __int64 v108; // [rsp+18h] [rbp-78h]
  __int64 v109; // [rsp+20h] [rbp-70h]
  __int64 v110; // [rsp+20h] [rbp-70h]
  __int64 v111; // [rsp+20h] [rbp-70h]
  __int64 v112; // [rsp+20h] [rbp-70h]
  __int64 v113; // [rsp+28h] [rbp-68h]
  __int64 v114; // [rsp+28h] [rbp-68h]
  __int64 v115; // [rsp+30h] [rbp-60h]
  __int64 v116; // [rsp+38h] [rbp-58h]
  __int64 v117; // [rsp+38h] [rbp-58h]
  __int64 v118; // [rsp+40h] [rbp-50h]
  __int64 v119; // [rsp+48h] [rbp-48h]
  __int64 v120; // [rsp+48h] [rbp-48h]
  __int64 v121; // [rsp+48h] [rbp-48h]
  __int64 v122; // [rsp+50h] [rbp-40h]
  __int64 v123; // [rsp+58h] [rbp-38h]
  __int64 v124; // [rsp+58h] [rbp-38h]
  __int64 v125; // [rsp+58h] [rbp-38h]

  while ( 1 )
  {
    v7 = a5;
    v8 = a6;
    v9 = a1;
    v10 = a2;
    v11 = a7;
    v122 = a3;
    v123 = a5;
    if ( a5 > a7 )
      v7 = a7;
    if ( a4 <= v7 )
    {
      v41 = a2 - a1;
      v125 = a2 - a1;
      result = 0xCF3CF3CF3CF3CF3DLL * ((a2 - a1) >> 3);
      v42 = (char **)a1;
      if ( a2 - a1 <= 0 )
        return result;
      v121 = a2;
      v43 = a6;
      v44 = 0xCF3CF3CF3CF3CF3DLL * ((a2 - a1) >> 3);
      do
      {
        v45 = v42;
        v46 = v43;
        v42 += 21;
        v43 += 168;
        sub_272D8A0(v46, v45, a3, v41, a5, a6);
        *(_QWORD *)(v43 - 24) = *(v42 - 3);
        *(_QWORD *)(v43 - 16) = *(v42 - 2);
        a3 = *((unsigned int *)v42 - 2);
        *(_DWORD *)(v43 - 8) = a3;
        --v44;
      }
      while ( v44 );
      v47 = v125;
      result = 168;
      v48 = v121;
      if ( v125 <= 0 )
        v47 = 168;
      v49 = v8 + v47;
      if ( v8 == v8 + v47 )
        return result;
      while ( 1 )
      {
        if ( v48 == v122 )
        {
          result = 0xCF3CF3CF3CF3CF3DLL;
          v36 = v49 - v8;
          v37 = v9;
          v38 = 0xCF3CF3CF3CF3CF3DLL * (v36 >> 3);
          if ( v36 > 0 )
          {
            do
            {
              v39 = (char **)v8;
              v40 = v37;
              v8 += 168;
              v37 += 168;
              sub_272D8A0(v40, v39, a3, v41, a5, a6);
              *(_QWORD *)(v37 - 24) = *(_QWORD *)(v8 - 24);
              *(_QWORD *)(v37 - 16) = *(_QWORD *)(v8 - 16);
              result = *(unsigned int *)(v8 - 8);
              *(_DWORD *)(v37 - 8) = result;
              --v38;
            }
            while ( v38 );
          }
          return result;
        }
        v51 = *(_QWORD *)(v8 + 144);
        v52 = *(_QWORD *)(v48 + 144);
        if ( *(_QWORD *)(v52 + 8) == *(_QWORD *)(v51 + 8) )
        {
          if ( (int)sub_C49970(v52 + 24, (unsigned __int64 *)(v51 + 24)) >= 0 )
          {
LABEL_43:
            v53 = (char **)v8;
            v8 += 168;
            sub_272D8A0(v9, v53, a3, v41, a5, a6);
            *(_QWORD *)(v9 + 144) = *(_QWORD *)(v8 - 24);
            *(_QWORD *)(v9 + 152) = *(_QWORD *)(v8 - 16);
            result = *(unsigned int *)(v8 - 8);
            *(_DWORD *)(v9 + 160) = result;
            goto LABEL_39;
          }
        }
        else if ( *(_DWORD *)(v52 + 32) >= *(_DWORD *)(v51 + 32) )
        {
          goto LABEL_43;
        }
        v50 = (char **)v48;
        v48 += 168;
        sub_272D8A0(v9, v50, a3, v41, a5, a6);
        *(_QWORD *)(v9 + 144) = *(_QWORD *)(v48 - 24);
        *(_QWORD *)(v9 + 152) = *(_QWORD *)(v48 - 16);
        result = *(unsigned int *)(v48 - 8);
        *(_DWORD *)(v9 + 160) = result;
LABEL_39:
        v9 += 168;
        if ( v8 == v49 )
          return result;
      }
    }
    v12 = a2;
    if ( a5 <= a7 )
      break;
    v113 = a4;
    if ( a4 <= a5 )
    {
      v115 = a5 / 2;
      v118 = a2 + 168 * (a5 / 2);
      v74 = sub_272DAD0(a1, a2, v118);
      v16 = v113;
      v119 = v74;
      v116 = 0xCF3CF3CF3CF3CF3DLL * ((v74 - a1) >> 3);
    }
    else
    {
      v116 = a4 / 2;
      v119 = a1 + 168 * (a4 / 2);
      v13 = sub_272DA00(a2, a3, v119);
      v16 = v113;
      v118 = v13;
      v115 = 0xCF3CF3CF3CF3CF3DLL * ((v13 - a2) >> 3);
    }
    v17 = v16 - v116;
    v114 = v17;
    if ( v17 > v115 && v115 <= a7 )
    {
      v18 = v119;
      if ( !v115 )
        goto LABEL_10;
      v75 = a2 - v119;
      v106 = a2 - v119;
      v111 = v118 - a2;
      v76 = 0xCF3CF3CF3CF3CF3DLL * ((a2 - v119) >> 3);
      v98 = v76;
      if ( v118 - a2 <= 0 )
      {
        if ( v106 <= 0 )
          goto LABEL_10;
        v112 = 0;
        v102 = 0;
LABEL_70:
        v107 = v11;
        v83 = v118;
        v84 = v98;
        do
        {
          v10 -= 168;
          v83 -= 168;
          sub_272D8A0(v83, (char **)v10, v76, v75, v14, v15);
          *(_QWORD *)(v83 + 144) = *(_QWORD *)(v10 + 144);
          *(_QWORD *)(v83 + 152) = *(_QWORD *)(v10 + 152);
          v76 = *(unsigned int *)(v10 + 160);
          *(_DWORD *)(v83 + 160) = v76;
          --v84;
        }
        while ( v84 );
        v11 = v107;
      }
      else
      {
        v101 = a2;
        v77 = (char **)a2;
        v78 = 0xCF3CF3CF3CF3CF3DLL * ((v118 - a2) >> 3);
        v79 = v8;
        do
        {
          v80 = v77;
          v81 = v79;
          v77 += 21;
          v79 += 168;
          sub_272D8A0(v81, v80, v76, v75, v14, v15);
          *(_QWORD *)(v79 - 24) = *(v77 - 3);
          *(_QWORD *)(v79 - 16) = *(v77 - 2);
          v76 = *((unsigned int *)v77 - 2);
          *(_DWORD *)(v79 - 8) = v76;
          --v78;
        }
        while ( v78 );
        v82 = 168;
        v10 = v101;
        v11 = a7;
        if ( v111 > 0 )
          v82 = v111;
        v102 = v82;
        v76 = 0xCF3CF3CF3CF3CF3DLL * (v82 >> 3);
        v112 = v76;
        if ( v106 > 0 )
          goto LABEL_70;
      }
      if ( v102 <= 0 )
      {
        v18 = v119;
      }
      else
      {
        v108 = v11;
        v85 = (char **)v8;
        v86 = v119;
        v87 = v112;
        do
        {
          v88 = v85;
          v89 = v86;
          v85 += 21;
          v86 += 168;
          sub_272D8A0(v89, v88, v76, v75, v14, v15);
          *(_QWORD *)(v86 - 24) = *(v85 - 3);
          *(_QWORD *)(v86 - 16) = *(v85 - 2);
          v76 = *((unsigned int *)v85 - 2);
          *(_DWORD *)(v86 - 8) = v76;
          --v87;
        }
        while ( v87 );
        v11 = v108;
        v90 = 168 * v112;
        if ( v112 <= 0 )
          v90 = 168;
        v18 = v119 + v90;
      }
      goto LABEL_10;
    }
    if ( v17 > a7 )
    {
      v18 = sub_272E010(v119, a2, v118, v17, v14, v15);
      goto LABEL_10;
    }
    v18 = v118;
    if ( !v17 )
      goto LABEL_10;
    v103 = v118 - a2;
    v54 = (v118 - a2) >> 3;
    v109 = a2 - v119;
    v55 = (a2 - v119) >> 3;
    v97 = 0xCF3CF3CF3CF3CF3DLL * v54;
    if ( a2 - v119 <= 0 )
    {
      if ( v103 <= 0 )
        goto LABEL_10;
      v110 = 0;
      v63 = v8;
      v100 = 0;
    }
    else
    {
      v99 = a2;
      v56 = 0xCF3CF3CF3CF3CF3DLL * ((a2 - v119) >> 3);
      v57 = v8;
      v58 = (char **)v119;
      do
      {
        v59 = v58;
        v60 = v57;
        v58 += 21;
        v57 += 168;
        sub_272D8A0(v60, v59, v55, v54, v14, v15);
        *(_QWORD *)(v57 - 24) = *(v58 - 3);
        *(_QWORD *)(v57 - 16) = *(v58 - 2);
        v54 = *((unsigned int *)v58 - 2);
        *(_DWORD *)(v57 - 8) = v54;
        --v56;
      }
      while ( v56 );
      v61 = 168;
      v10 = v99;
      v11 = a7;
      if ( v109 > 0 )
        v61 = v109;
      v62 = v61;
      v100 = v61;
      v63 = v8 + v61;
      v64 = 0xCF3CF3CF3CF3CF3DLL;
      v54 = 0xCF3CF3CF3CF3CF3DLL * (v62 >> 3);
      v110 = v54;
      if ( v103 <= 0 )
        goto LABEL_57;
    }
    v64 = v119;
    v96 = v11;
    v65 = (char **)v10;
    v66 = v97;
    v104 = v9;
    v67 = v119;
    do
    {
      v68 = v65;
      v69 = v67;
      v65 += 21;
      v67 += 168;
      sub_272D8A0(v69, v68, v64, v54, v14, v15);
      *(_QWORD *)(v67 - 24) = *(v65 - 3);
      *(_QWORD *)(v67 - 16) = *(v65 - 2);
      v54 = *((unsigned int *)v65 - 2);
      *(_DWORD *)(v67 - 8) = v54;
      --v66;
    }
    while ( v66 );
    LODWORD(v9) = v104;
    v11 = v96;
LABEL_57:
    if ( v100 <= 0 )
    {
      v18 = v118;
    }
    else
    {
      v70 = v118;
      v105 = v11;
      v71 = v63;
      v72 = v110;
      do
      {
        v71 -= 168;
        v70 -= 168;
        sub_272D8A0(v70, (char **)v71, v64, v54, v14, v15);
        *(_QWORD *)(v70 + 144) = *(_QWORD *)(v71 + 144);
        *(_QWORD *)(v70 + 152) = *(_QWORD *)(v71 + 152);
        v64 = *(unsigned int *)(v71 + 160);
        *(_DWORD *)(v70 + 160) = v64;
        --v72;
      }
      while ( v72 );
      v11 = v105;
      v73 = -168 * v110;
      if ( v110 <= 0 )
        v73 = -168;
      v18 = v118 + v73;
    }
LABEL_10:
    v19 = v116;
    v117 = v18;
    sub_272F660(v9, v119, v18, v19, v115, v8, v11);
    a7 = v11;
    a4 = v114;
    a6 = v8;
    a2 = v118;
    a3 = v122;
    a5 = v123 - v115;
    a1 = v117;
  }
  result = 0xCF3CF3CF3CF3CF3DLL;
  v21 = a3 - a2;
  v124 = a3 - a2;
  v22 = 0xCF3CF3CF3CF3CF3DLL * ((a3 - a2) >> 3);
  if ( a3 - a2 <= 0 )
    return result;
  v120 = a1;
  v23 = (char **)a2;
  v24 = a6;
  do
  {
    v25 = v23;
    v26 = v24;
    v23 += 21;
    v24 += 168;
    sub_272D8A0(v26, v25, a3, v21, a5, a6);
    result = (unsigned __int64)*(v23 - 3);
    *(_QWORD *)(v24 - 24) = result;
    *(_QWORD *)(v24 - 16) = *(v23 - 2);
    *(_DWORD *)(v24 - 8) = *((_DWORD *)v23 - 2);
    --v22;
  }
  while ( v22 );
  v27 = v124;
  v28 = v122;
  v29 = v124 - 168;
  if ( v124 <= 0 )
    v29 = 0;
  v30 = v8 + v29;
  if ( v124 <= 0 )
    v27 = 168;
  v31 = v8 + v27;
  if ( v120 != v12 )
  {
    if ( v8 == v31 )
      return result;
    v32 = v12 - 168;
    while ( 1 )
    {
      v35 = *(_QWORD *)(v32 + 144);
      if ( *(_QWORD *)(result + 8) == *(_QWORD *)(v35 + 8) )
      {
        LODWORD(result) = (unsigned int)sub_C49970(result + 24, (unsigned __int64 *)(v35 + 24)) >> 31;
      }
      else
      {
        v33 = *(unsigned int *)(v35 + 32);
        LOBYTE(result) = *(_DWORD *)(result + 32) < (unsigned int)v33;
      }
      v28 -= 168;
      if ( (_BYTE)result )
      {
        sub_272D8A0(v28, (char **)v32, v31, v33, a5, a6);
        *(_QWORD *)(v28 + 144) = *(_QWORD *)(v32 + 144);
        *(_QWORD *)(v28 + 152) = *(_QWORD *)(v32 + 152);
        *(_DWORD *)(v28 + 160) = *(_DWORD *)(v32 + 160);
        if ( v32 == v120 )
        {
          v91 = v30 + 168;
          v92 = 0xCF3CF3CF3CF3CF3DLL;
          result = v91 - v8;
          v93 = 0xCF3CF3CF3CF3CF3DLL * ((v91 - v8) >> 3);
          if ( v91 - v8 > 0 )
          {
            do
            {
              v91 -= 168;
              v28 -= 168;
              sub_272D8A0(v28, (char **)v91, v92, v34, a5, a6);
              *(_QWORD *)(v28 + 144) = *(_QWORD *)(v91 + 144);
              *(_QWORD *)(v28 + 152) = *(_QWORD *)(v91 + 152);
              result = *(unsigned int *)(v91 + 160);
              *(_DWORD *)(v28 + 160) = result;
              --v93;
            }
            while ( v93 );
          }
          return result;
        }
        v32 -= 168;
      }
      else
      {
        sub_272D8A0(v28, (char **)v30, v31, v33, a5, a6);
        *(_QWORD *)(v28 + 144) = *(_QWORD *)(v30 + 144);
        *(_QWORD *)(v28 + 152) = *(_QWORD *)(v30 + 152);
        result = *(unsigned int *)(v30 + 160);
        *(_DWORD *)(v28 + 160) = result;
        if ( v8 == v30 )
          return result;
        v30 -= 168;
      }
      result = *(_QWORD *)(v30 + 144);
    }
  }
  result = 0xCF3CF3CF3CF3CF3DLL;
  v94 = 0xCF3CF3CF3CF3CF3DLL * (v27 >> 3);
  if ( v27 > 0 )
  {
    v95 = v8 + v27;
    do
    {
      v95 -= 168;
      v28 -= 168;
      sub_272D8A0(v28, (char **)v95, v31, v27, a5, a6);
      *(_QWORD *)(v28 + 144) = *(_QWORD *)(v95 + 144);
      *(_QWORD *)(v28 + 152) = *(_QWORD *)(v95 + 152);
      result = *(unsigned int *)(v95 + 160);
      *(_DWORD *)(v28 + 160) = result;
      --v94;
    }
    while ( v94 );
  }
  return result;
}
