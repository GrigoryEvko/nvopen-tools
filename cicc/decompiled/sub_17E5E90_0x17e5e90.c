// Function: sub_17E5E90
// Address: 0x17e5e90
//
__int64 __fastcall sub_17E5E90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, char *a6, __int64 a7)
{
  char *v7; // r13
  __int64 *v8; // r12
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // r10
  __int64 v13; // r11
  _QWORD *v14; // rcx
  __int64 v15; // r15
  __int64 v16; // r11
  int v17; // ecx
  __int64 result; // rax
  __int64 v19; // rbx
  __int64 *v20; // r15
  char *v21; // r14
  __int64 v22; // rdx
  __int64 v23; // rdi
  __int64 v24; // rsi
  __int64 *v25; // r14
  __int64 *v26; // rbx
  __int64 *i; // r15
  __int64 v28; // rdi
  __int64 v29; // rdx
  __int64 v30; // rdi
  __int64 v31; // rbx
  __int64 *v32; // r14
  __int64 v33; // r12
  __int64 v34; // rdi
  __int64 *v35; // r15
  char *v36; // r14
  __int64 v37; // rbx
  __int64 v38; // rdx
  __int64 v39; // rdi
  __int64 v40; // rsi
  char *v41; // rbx
  __int64 v42; // r15
  __int64 v43; // rdi
  __int64 v44; // rdx
  __int64 v45; // rdi
  char *v46; // r12
  __int64 *v47; // rbx
  __int64 v48; // r13
  __int64 v49; // rcx
  __int64 v50; // rdi
  __int64 v51; // rax
  __int64 v52; // r13
  __int64 *v53; // r12
  __int64 *v54; // rbx
  __int64 v55; // rsi
  __int64 v56; // rdi
  char *v57; // rcx
  __int64 v58; // r13
  char *v59; // r12
  __int64 v60; // rbx
  __int64 v61; // rsi
  __int64 v62; // rdi
  __int64 v63; // r11
  __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // r12
  char *v67; // rbx
  __int64 *v68; // r13
  __int64 v69; // rax
  __int64 v70; // rdi
  __int64 v71; // rdx
  __int64 v72; // rdx
  __int64 v73; // r12
  __int64 v74; // rbx
  __int64 v75; // r13
  __int64 v76; // rcx
  __int64 v77; // rdi
  char *v78; // rbx
  __int64 *v79; // r12
  __int64 v80; // r13
  __int64 v81; // rsi
  __int64 v82; // rdi
  __int64 v83; // r11
  __int64 v84; // r12
  __int64 *v85; // rbx
  __int64 *v86; // r13
  __int64 v87; // rdi
  __int64 v88; // r12
  __int64 v89; // rbx
  __int64 v90; // r13
  __int64 v91; // rdi
  char *v92; // [rsp+8h] [rbp-98h]
  char *v93; // [rsp+10h] [rbp-90h]
  char *v94; // [rsp+10h] [rbp-90h]
  __int64 v95; // [rsp+10h] [rbp-90h]
  __int64 v96; // [rsp+18h] [rbp-88h]
  __int64 v97; // [rsp+18h] [rbp-88h]
  __int64 v98; // [rsp+18h] [rbp-88h]
  int v99; // [rsp+20h] [rbp-80h]
  char *v100; // [rsp+20h] [rbp-80h]
  char *v101; // [rsp+20h] [rbp-80h]
  int v102; // [rsp+20h] [rbp-80h]
  char *v103; // [rsp+20h] [rbp-80h]
  _QWORD *v104; // [rsp+28h] [rbp-78h]
  __int64 v105; // [rsp+28h] [rbp-78h]
  __int64 v106; // [rsp+28h] [rbp-78h]
  __int64 v107; // [rsp+28h] [rbp-78h]
  __int64 v108; // [rsp+28h] [rbp-78h]
  char *v109; // [rsp+28h] [rbp-78h]
  __int64 v110; // [rsp+30h] [rbp-70h]
  int v111; // [rsp+30h] [rbp-70h]
  __int64 v112; // [rsp+30h] [rbp-70h]
  __int64 v113; // [rsp+30h] [rbp-70h]
  __int64 v114; // [rsp+30h] [rbp-70h]
  __int64 v115; // [rsp+38h] [rbp-68h]
  int v116; // [rsp+38h] [rbp-68h]
  __int64 v117; // [rsp+38h] [rbp-68h]
  int v118; // [rsp+38h] [rbp-68h]
  int v119; // [rsp+38h] [rbp-68h]
  __int64 v120; // [rsp+40h] [rbp-60h]
  __int64 v121; // [rsp+40h] [rbp-60h]
  __int64 v122; // [rsp+40h] [rbp-60h]
  __int64 v123; // [rsp+40h] [rbp-60h]
  __int64 v124; // [rsp+48h] [rbp-58h]
  __int64 v125; // [rsp+50h] [rbp-50h]
  __int64 v126; // [rsp+50h] [rbp-50h]
  __int64 v127; // [rsp+58h] [rbp-48h]
  __int64 *v128; // [rsp+58h] [rbp-48h]
  __int64 v129; // [rsp+58h] [rbp-48h]
  __int64 v130; // [rsp+60h] [rbp-40h]
  __int64 v131; // [rsp+60h] [rbp-40h]
  __int64 v132; // [rsp+60h] [rbp-40h]
  __int64 v133; // [rsp+68h] [rbp-38h]

  while ( 1 )
  {
    v7 = a6;
    v8 = (__int64 *)a1;
    v133 = a3;
    v9 = a7;
    if ( a5 <= a7 )
      v9 = a5;
    if ( v9 >= a4 )
      break;
    v10 = a5;
    if ( a5 <= a7 )
    {
      result = a3 - a2;
      v131 = a3 - a2;
      v19 = (a3 - a2) >> 3;
      if ( a3 - a2 <= 0 )
        return result;
      v128 = (__int64 *)a2;
      v20 = (__int64 *)a2;
      v21 = a6;
      do
      {
        v22 = *v20;
        *v20 = 0;
        v23 = *(_QWORD *)v21;
        *(_QWORD *)v21 = v22;
        if ( v23 )
          j_j___libc_free_0(v23, 32);
        ++v20;
        v21 += 8;
        --v19;
      }
      while ( v19 );
      v24 = v131;
      if ( v131 <= 0 )
        v24 = 8;
      result = (__int64)&v7[v24];
      if ( v128 != v8 )
      {
        if ( v7 == (char *)result )
          return result;
        v25 = v128 - 1;
        v26 = (__int64 *)(result - 8);
        for ( i = (__int64 *)(v133 - 8); ; --i )
        {
          v29 = *v25;
          result = *v26;
          if ( *(_QWORD *)(*v26 + 16) > *(_QWORD *)(*v25 + 16) )
          {
            *v25 = 0;
            v28 = *i;
            *i = v29;
            if ( v28 )
              j_j___libc_free_0(v28, 32);
            if ( v25 == v8 )
            {
              result = (char *)(v26 + 1) - v7;
              v84 = result >> 3;
              if ( result > 0 )
              {
                v85 = &v26[-v84];
                v86 = &i[-v84];
                do
                {
                  result = v85[v84];
                  v85[v84] = 0;
                  v87 = v86[v84 - 1];
                  v86[v84 - 1] = result;
                  if ( v87 )
                    result = j_j___libc_free_0(v87, 32);
                  --v84;
                }
                while ( v84 );
              }
              return result;
            }
            --v25;
          }
          else
          {
            *v26 = 0;
            v30 = *i;
            *i = result;
            if ( v30 )
              result = j_j___libc_free_0(v30, 32);
            if ( v7 == (char *)v26 )
              return result;
            --v26;
          }
        }
      }
      v88 = v24 >> 3;
      if ( v24 > 0 )
      {
        v89 = result - 8 * v88;
        v90 = -8 * v88 + v133;
        do
        {
          result = *(_QWORD *)(v89 + 8 * v88 - 8);
          *(_QWORD *)(v89 + 8 * v88 - 8) = 0;
          v91 = *(_QWORD *)(v90 + 8 * v88 - 8);
          *(_QWORD *)(v90 + 8 * v88 - 8) = result;
          if ( v91 )
            result = j_j___libc_free_0(v91, 32);
          --v88;
        }
        while ( v88 );
      }
      return result;
    }
    if ( a5 >= a4 )
    {
      v15 = a5 / 2;
      v127 = a2 + 8 * (a5 / 2);
      v64 = sub_17E24B0(a1, a2, v127);
      v14 = (_QWORD *)a2;
      v130 = v64;
      v125 = (v64 - a1) >> 3;
    }
    else
    {
      v125 = a4 / 2;
      v130 = a1 + 8 * (a4 / 2);
      v11 = sub_17E2460(a2, a3, v130);
      v14 = (_QWORD *)a2;
      v127 = v11;
      v15 = (v11 - v12) >> 3;
    }
    v124 = v13 - v125;
    if ( v13 - v125 > v15 && a7 >= v15 )
    {
      v16 = v130;
      if ( !v15 )
        goto LABEL_10;
      v117 = v12 - v130;
      v65 = (v12 - v130) >> 3;
      v122 = v127 - v12;
      v107 = v65;
      if ( v127 - v12 <= 0 )
      {
        if ( v117 <= 0 )
          goto LABEL_10;
        v123 = 0;
        v113 = 0;
LABEL_86:
        v72 = v107;
        v118 = (int)v8;
        v73 = v65;
        v108 = v10;
        v103 = v7;
        v72 *= -8;
        v74 = v127 + v72;
        v75 = v72 + v12;
        do
        {
          v76 = *(_QWORD *)(v75 + 8 * v73 - 8);
          *(_QWORD *)(v75 + 8 * v73 - 8) = 0;
          v77 = *(_QWORD *)(v74 + 8 * v73 - 8);
          *(_QWORD *)(v74 + 8 * v73 - 8) = v76;
          if ( v77 )
            j_j___libc_free_0(v77, 32);
          --v73;
        }
        while ( v73 );
        LODWORD(v8) = v118;
        v10 = v108;
        v7 = v103;
      }
      else
      {
        v112 = (v12 - v130) >> 3;
        v98 = v12;
        v102 = a1;
        v66 = (v127 - v12) >> 3;
        v95 = v10;
        v67 = v7;
        v92 = v7;
        v68 = (__int64 *)v12;
        do
        {
          v69 = *v68;
          *v68 = 0;
          v70 = *(_QWORD *)v67;
          *(_QWORD *)v67 = v69;
          if ( v70 )
            j_j___libc_free_0(v70, 32);
          ++v68;
          v67 += 8;
          --v66;
        }
        while ( v66 );
        v71 = 8;
        v65 = v112;
        LODWORD(v8) = v102;
        v12 = v98;
        v10 = v95;
        v7 = v92;
        if ( v122 > 0 )
          v71 = v122;
        v113 = v71;
        v123 = v71 >> 3;
        if ( v117 > 0 )
          goto LABEL_86;
      }
      if ( v113 <= 0 )
      {
        v16 = v130;
      }
      else
      {
        v119 = (int)v8;
        v114 = v10;
        v78 = v7;
        v109 = v7;
        v79 = (__int64 *)v130;
        v80 = v123;
        do
        {
          v81 = *(_QWORD *)v78;
          *(_QWORD *)v78 = 0;
          v82 = *v79;
          *v79 = v81;
          if ( v82 )
            j_j___libc_free_0(v82, 32);
          v78 += 8;
          ++v79;
          --v80;
        }
        while ( v80 );
        LODWORD(v8) = v119;
        v10 = v114;
        v7 = v109;
        v83 = 8 * v123;
        if ( v123 <= 0 )
          v83 = 8;
        v16 = v130 + v83;
      }
      goto LABEL_10;
    }
    if ( a7 < v124 )
    {
      v16 = sub_17E20D0(v130, v12, v127);
      goto LABEL_10;
    }
    v16 = v127;
    if ( !v124 )
      goto LABEL_10;
    v115 = v127 - v12;
    v120 = v12 - v130;
    v110 = (v127 - v12) >> 3;
    if ( v12 - v130 <= 0 )
    {
      if ( v115 <= 0 )
        goto LABEL_10;
      v100 = v7;
      v121 = 0;
      v105 = 0;
    }
    else
    {
      v104 = v14;
      v99 = a1;
      v46 = v7;
      v96 = v10;
      v47 = (__int64 *)v130;
      v93 = v7;
      v48 = (v12 - v130) >> 3;
      do
      {
        v49 = *v47;
        *v47 = 0;
        v50 = *(_QWORD *)v46;
        *(_QWORD *)v46 = v49;
        if ( v50 )
          j_j___libc_free_0(v50, 32);
        ++v47;
        v46 += 8;
        --v48;
      }
      while ( v48 );
      v51 = 8;
      v7 = v93;
      v14 = v104;
      LODWORD(v8) = v99;
      v10 = v96;
      if ( v120 > 0 )
        v51 = v120;
      v105 = v51;
      v100 = &v93[v51];
      v121 = v51 >> 3;
      if ( v115 <= 0 )
        goto LABEL_69;
    }
    v94 = v7;
    v52 = v110;
    v116 = (int)v8;
    v97 = v10;
    v53 = (__int64 *)v130;
    v54 = v14;
    do
    {
      v55 = *v54;
      *v54 = 0;
      v56 = *v53;
      *v53 = v55;
      if ( v56 )
        j_j___libc_free_0(v56, 32);
      ++v54;
      ++v53;
      --v52;
    }
    while ( v52 );
    LODWORD(v8) = v116;
    v10 = v97;
    v7 = v94;
LABEL_69:
    if ( v105 <= 0 )
    {
      v16 = v127;
    }
    else
    {
      v57 = v100;
      v111 = (int)v8;
      v106 = v10;
      v101 = v7;
      v58 = v121;
      v59 = &v57[-8 * v121];
      v60 = -8 * v121 + v127;
      do
      {
        v61 = *(_QWORD *)&v59[8 * v58 - 8];
        *(_QWORD *)&v59[8 * v58 - 8] = 0;
        v62 = *(_QWORD *)(v60 + 8 * v58 - 8);
        *(_QWORD *)(v60 + 8 * v58 - 8) = v61;
        if ( v62 )
          j_j___libc_free_0(v62, 32);
        --v58;
      }
      while ( v58 );
      v63 = -8;
      LODWORD(v8) = v111;
      if ( v121 > 0 )
        v63 = -8 * v121;
      v10 = v106;
      v7 = v101;
      v16 = v127 + v63;
    }
LABEL_10:
    v17 = v125;
    v126 = v16;
    sub_17E5E90((_DWORD)v8, v130, v16, v17, v15, (_DWORD)v7, a7);
    a6 = v7;
    a4 = v124;
    a2 = v127;
    a5 = v10 - v15;
    a1 = v126;
    a3 = v133;
  }
  v35 = (__int64 *)a1;
  v36 = a6;
  result = a2 - a1;
  v132 = a2 - a1;
  v37 = (a2 - a1) >> 3;
  if ( a2 - a1 > 0 )
  {
    v129 = a2;
    do
    {
      v38 = *v35;
      *v35 = 0;
      v39 = *(_QWORD *)v36;
      *(_QWORD *)v36 = v38;
      if ( v39 )
        j_j___libc_free_0(v39, 32);
      ++v35;
      v36 += 8;
      --v37;
    }
    while ( v37 );
    v40 = v132;
    result = 8;
    if ( v132 <= 0 )
      v40 = 8;
    v41 = &v7[v40];
    if ( v7 != &v7[v40] )
    {
      v42 = v129;
      while ( v42 != v133 )
      {
        result = *(_QWORD *)v7;
        v44 = *(_QWORD *)v42;
        if ( *(_QWORD *)(*(_QWORD *)v42 + 16LL) > *(_QWORD *)(*(_QWORD *)v7 + 16LL) )
        {
          *(_QWORD *)v42 = 0;
          v43 = *v8;
          *v8 = v44;
          if ( v43 )
            result = j_j___libc_free_0(v43, 32);
          v42 += 8;
        }
        else
        {
          *(_QWORD *)v7 = 0;
          v45 = *v8;
          *v8 = result;
          if ( v45 )
            result = j_j___libc_free_0(v45, 32);
          v7 += 8;
        }
        ++v8;
        if ( v7 == v41 )
          return result;
      }
      v31 = v41 - v7;
      v32 = v8;
      v33 = v31 >> 3;
      if ( v31 > 0 )
      {
        do
        {
          result = *(_QWORD *)v7;
          *(_QWORD *)v7 = 0;
          v34 = *v32;
          *v32 = result;
          if ( v34 )
            result = j_j___libc_free_0(v34, 32);
          v7 += 8;
          ++v32;
          --v33;
        }
        while ( v33 );
      }
    }
  }
  return result;
}
