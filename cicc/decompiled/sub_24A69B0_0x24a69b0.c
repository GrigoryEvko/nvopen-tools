// Function: sub_24A69B0
// Address: 0x24a69b0
//
void __fastcall sub_24A69B0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int64 *a6,
        __int64 a7)
{
  unsigned __int64 *v7; // r13
  unsigned __int64 *v8; // r12
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // r10
  __int64 v13; // r11
  unsigned __int64 *v14; // rcx
  __int64 v15; // r15
  __int64 v16; // r11
  int v17; // ecx
  __int64 v18; // rbx
  unsigned __int64 *v19; // r15
  unsigned __int64 *v20; // r14
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // rdi
  __int64 v23; // rsi
  unsigned __int64 *v24; // rax
  unsigned __int64 *v25; // r14
  unsigned __int64 *v26; // rbx
  unsigned __int64 *i; // r15
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rdx
  unsigned __int64 v30; // rax
  unsigned __int64 v31; // rdi
  __int64 v32; // rbx
  unsigned __int64 *v33; // r14
  __int64 v34; // r12
  unsigned __int64 v35; // rax
  unsigned __int64 v36; // rdi
  unsigned __int64 *v37; // r15
  unsigned __int64 *v38; // r14
  __int64 v39; // rbx
  unsigned __int64 v40; // rdx
  unsigned __int64 v41; // rdi
  __int64 v42; // rsi
  unsigned __int64 *v43; // rbx
  unsigned __int64 *v44; // r15
  unsigned __int64 v45; // rdi
  unsigned __int64 v46; // rax
  unsigned __int64 v47; // rdx
  unsigned __int64 v48; // rdi
  unsigned __int64 *v49; // r12
  unsigned __int64 *v50; // rbx
  __int64 v51; // r13
  unsigned __int64 v52; // rcx
  unsigned __int64 v53; // rdi
  __int64 v54; // rax
  __int64 v55; // r13
  unsigned __int64 *v56; // r12
  unsigned __int64 *v57; // rbx
  unsigned __int64 v58; // rsi
  unsigned __int64 v59; // rdi
  unsigned __int64 *v60; // rcx
  __int64 v61; // r13
  unsigned __int64 *v62; // r12
  __int64 v63; // rbx
  unsigned __int64 v64; // rsi
  unsigned __int64 v65; // rdi
  __int64 v66; // r11
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // r12
  unsigned __int64 *v70; // rbx
  unsigned __int64 *v71; // r13
  unsigned __int64 v72; // rax
  unsigned __int64 v73; // rdi
  __int64 v74; // rdx
  __int64 v75; // rdx
  __int64 v76; // r12
  __int64 v77; // rbx
  __int64 v78; // r13
  __int64 v79; // rcx
  unsigned __int64 v80; // rdi
  unsigned __int64 *v81; // rbx
  unsigned __int64 *v82; // r12
  __int64 v83; // r13
  unsigned __int64 v84; // rsi
  unsigned __int64 v85; // rdi
  __int64 v86; // r11
  __int64 v87; // r12
  unsigned __int64 *v88; // rbx
  unsigned __int64 *v89; // r13
  unsigned __int64 v90; // rax
  unsigned __int64 v91; // rdi
  __int64 v92; // r12
  unsigned __int64 *v93; // rbx
  __int64 v94; // r13
  unsigned __int64 v95; // rax
  unsigned __int64 v96; // rdi
  unsigned __int64 *v97; // [rsp+8h] [rbp-98h]
  unsigned __int64 *v98; // [rsp+10h] [rbp-90h]
  unsigned __int64 *v99; // [rsp+10h] [rbp-90h]
  __int64 v100; // [rsp+10h] [rbp-90h]
  __int64 v101; // [rsp+18h] [rbp-88h]
  __int64 v102; // [rsp+18h] [rbp-88h]
  __int64 v103; // [rsp+18h] [rbp-88h]
  int v104; // [rsp+20h] [rbp-80h]
  unsigned __int64 *v105; // [rsp+20h] [rbp-80h]
  unsigned __int64 *v106; // [rsp+20h] [rbp-80h]
  int v107; // [rsp+20h] [rbp-80h]
  unsigned __int64 *v108; // [rsp+20h] [rbp-80h]
  unsigned __int64 *v109; // [rsp+28h] [rbp-78h]
  __int64 v110; // [rsp+28h] [rbp-78h]
  __int64 v111; // [rsp+28h] [rbp-78h]
  __int64 v112; // [rsp+28h] [rbp-78h]
  __int64 v113; // [rsp+28h] [rbp-78h]
  unsigned __int64 *v114; // [rsp+28h] [rbp-78h]
  __int64 v115; // [rsp+30h] [rbp-70h]
  int v116; // [rsp+30h] [rbp-70h]
  __int64 v117; // [rsp+30h] [rbp-70h]
  __int64 v118; // [rsp+30h] [rbp-70h]
  __int64 v119; // [rsp+30h] [rbp-70h]
  __int64 v120; // [rsp+38h] [rbp-68h]
  int v121; // [rsp+38h] [rbp-68h]
  __int64 v122; // [rsp+38h] [rbp-68h]
  int v123; // [rsp+38h] [rbp-68h]
  int v124; // [rsp+38h] [rbp-68h]
  __int64 v125; // [rsp+40h] [rbp-60h]
  __int64 v126; // [rsp+40h] [rbp-60h]
  __int64 v127; // [rsp+40h] [rbp-60h]
  __int64 v128; // [rsp+40h] [rbp-60h]
  __int64 v129; // [rsp+48h] [rbp-58h]
  __int64 v130; // [rsp+50h] [rbp-50h]
  __int64 v131; // [rsp+50h] [rbp-50h]
  __int64 v132; // [rsp+58h] [rbp-48h]
  unsigned __int64 *v133; // [rsp+58h] [rbp-48h]
  __int64 v134; // [rsp+58h] [rbp-48h]
  __int64 v135; // [rsp+60h] [rbp-40h]
  __int64 v136; // [rsp+60h] [rbp-40h]
  __int64 v137; // [rsp+60h] [rbp-40h]
  __int64 v138; // [rsp+68h] [rbp-38h]

  while ( 1 )
  {
    v7 = a6;
    v8 = (unsigned __int64 *)a1;
    v138 = a3;
    v9 = a7;
    if ( a5 <= a7 )
      v9 = a5;
    if ( v9 >= a4 )
      break;
    v10 = a5;
    if ( a5 <= a7 )
    {
      v136 = a3 - a2;
      v18 = (a3 - a2) >> 3;
      if ( a3 - a2 <= 0 )
        return;
      v133 = (unsigned __int64 *)a2;
      v19 = (unsigned __int64 *)a2;
      v20 = a6;
      do
      {
        v21 = *v19;
        *v19 = 0;
        v22 = *v20;
        *v20 = v21;
        if ( v22 )
          j_j___libc_free_0(v22);
        ++v19;
        ++v20;
        --v18;
      }
      while ( v18 );
      v23 = v136;
      if ( v136 <= 0 )
        v23 = 8;
      v24 = (unsigned __int64 *)((char *)v7 + v23);
      if ( v133 != v8 )
      {
        if ( v7 == v24 )
          return;
        v25 = v133 - 1;
        v26 = v24 - 1;
        for ( i = (unsigned __int64 *)(v138 - 8); ; --i )
        {
          v29 = *v25;
          v30 = *v26;
          if ( *(_QWORD *)(*v26 + 16) > *(_QWORD *)(*v25 + 16) )
          {
            *v25 = 0;
            v28 = *i;
            *i = v29;
            if ( v28 )
              j_j___libc_free_0(v28);
            if ( v25 == v8 )
            {
              v87 = v26 + 1 - v7;
              if ( (char *)(v26 + 1) - (char *)v7 > 0 )
              {
                v88 = &v26[-v87];
                v89 = &i[-v87];
                do
                {
                  v90 = v88[v87];
                  v88[v87] = 0;
                  v91 = v89[v87 - 1];
                  v89[v87 - 1] = v90;
                  if ( v91 )
                    j_j___libc_free_0(v91);
                  --v87;
                }
                while ( v87 );
              }
              return;
            }
            --v25;
          }
          else
          {
            *v26 = 0;
            v31 = *i;
            *i = v30;
            if ( v31 )
              j_j___libc_free_0(v31);
            if ( v7 == v26 )
              return;
            --v26;
          }
        }
      }
      v92 = v23 >> 3;
      if ( v23 > 0 )
      {
        v93 = &v24[-v92];
        v94 = -8 * v92 + v138;
        do
        {
          v95 = v93[v92 - 1];
          v93[v92 - 1] = 0;
          v96 = *(_QWORD *)(v94 + 8 * v92 - 8);
          *(_QWORD *)(v94 + 8 * v92 - 8) = v95;
          if ( v96 )
            j_j___libc_free_0(v96);
          --v92;
        }
        while ( v92 );
      }
      return;
    }
    if ( a5 >= a4 )
    {
      v15 = a5 / 2;
      v132 = a2 + 8 * (a5 / 2);
      v67 = sub_24A31C0(a1, a2, v132);
      v14 = (unsigned __int64 *)a2;
      v135 = v67;
      v130 = (v67 - a1) >> 3;
    }
    else
    {
      v130 = a4 / 2;
      v135 = a1 + 8 * (a4 / 2);
      v11 = sub_24A3170(a2, a3, v135);
      v14 = (unsigned __int64 *)a2;
      v132 = v11;
      v15 = (v11 - v12) >> 3;
    }
    v129 = v13 - v130;
    if ( v13 - v130 > v15 && a7 >= v15 )
    {
      v16 = v135;
      if ( !v15 )
        goto LABEL_10;
      v122 = v12 - v135;
      v68 = (v12 - v135) >> 3;
      v127 = v132 - v12;
      v112 = v68;
      if ( v132 - v12 <= 0 )
      {
        if ( v122 <= 0 )
          goto LABEL_10;
        v128 = 0;
        v118 = 0;
LABEL_86:
        v75 = v112;
        v123 = (int)v8;
        v76 = v68;
        v113 = v10;
        v108 = v7;
        v75 *= -8;
        v77 = v132 + v75;
        v78 = v75 + v12;
        do
        {
          v79 = *(_QWORD *)(v78 + 8 * v76 - 8);
          *(_QWORD *)(v78 + 8 * v76 - 8) = 0;
          v80 = *(_QWORD *)(v77 + 8 * v76 - 8);
          *(_QWORD *)(v77 + 8 * v76 - 8) = v79;
          if ( v80 )
            j_j___libc_free_0(v80);
          --v76;
        }
        while ( v76 );
        LODWORD(v8) = v123;
        v10 = v113;
        v7 = v108;
      }
      else
      {
        v117 = (v12 - v135) >> 3;
        v103 = v12;
        v107 = a1;
        v69 = (v132 - v12) >> 3;
        v100 = v10;
        v70 = v7;
        v97 = v7;
        v71 = (unsigned __int64 *)v12;
        do
        {
          v72 = *v71;
          *v71 = 0;
          v73 = *v70;
          *v70 = v72;
          if ( v73 )
            j_j___libc_free_0(v73);
          ++v71;
          ++v70;
          --v69;
        }
        while ( v69 );
        v74 = 8;
        v68 = v117;
        LODWORD(v8) = v107;
        v12 = v103;
        v10 = v100;
        v7 = v97;
        if ( v127 > 0 )
          v74 = v127;
        v118 = v74;
        v128 = v74 >> 3;
        if ( v122 > 0 )
          goto LABEL_86;
      }
      if ( v118 <= 0 )
      {
        v16 = v135;
      }
      else
      {
        v124 = (int)v8;
        v119 = v10;
        v81 = v7;
        v114 = v7;
        v82 = (unsigned __int64 *)v135;
        v83 = v128;
        do
        {
          v84 = *v81;
          *v81 = 0;
          v85 = *v82;
          *v82 = v84;
          if ( v85 )
            j_j___libc_free_0(v85);
          ++v81;
          ++v82;
          --v83;
        }
        while ( v83 );
        LODWORD(v8) = v124;
        v10 = v119;
        v7 = v114;
        v86 = 8 * v128;
        if ( v128 <= 0 )
          v86 = 8;
        v16 = v135 + v86;
      }
      goto LABEL_10;
    }
    if ( a7 < v129 )
    {
      v16 = sub_24A2CF0(v135, v12, v132);
      goto LABEL_10;
    }
    v16 = v132;
    if ( !v129 )
      goto LABEL_10;
    v120 = v132 - v12;
    v125 = v12 - v135;
    v115 = (v132 - v12) >> 3;
    if ( v12 - v135 <= 0 )
    {
      if ( v120 <= 0 )
        goto LABEL_10;
      v105 = v7;
      v126 = 0;
      v110 = 0;
    }
    else
    {
      v109 = v14;
      v104 = a1;
      v49 = v7;
      v101 = v10;
      v50 = (unsigned __int64 *)v135;
      v98 = v7;
      v51 = (v12 - v135) >> 3;
      do
      {
        v52 = *v50;
        *v50 = 0;
        v53 = *v49;
        *v49 = v52;
        if ( v53 )
          j_j___libc_free_0(v53);
        ++v50;
        ++v49;
        --v51;
      }
      while ( v51 );
      v54 = 8;
      v7 = v98;
      v14 = v109;
      LODWORD(v8) = v104;
      v10 = v101;
      if ( v125 > 0 )
        v54 = v125;
      v110 = v54;
      v105 = (unsigned __int64 *)((char *)v98 + v54);
      v126 = v54 >> 3;
      if ( v120 <= 0 )
        goto LABEL_69;
    }
    v99 = v7;
    v55 = v115;
    v121 = (int)v8;
    v102 = v10;
    v56 = (unsigned __int64 *)v135;
    v57 = v14;
    do
    {
      v58 = *v57;
      *v57 = 0;
      v59 = *v56;
      *v56 = v58;
      if ( v59 )
        j_j___libc_free_0(v59);
      ++v57;
      ++v56;
      --v55;
    }
    while ( v55 );
    LODWORD(v8) = v121;
    v10 = v102;
    v7 = v99;
LABEL_69:
    if ( v110 <= 0 )
    {
      v16 = v132;
    }
    else
    {
      v60 = v105;
      v116 = (int)v8;
      v111 = v10;
      v106 = v7;
      v61 = v126;
      v62 = &v60[-v126];
      v63 = -8 * v126 + v132;
      do
      {
        v64 = v62[v61 - 1];
        v62[v61 - 1] = 0;
        v65 = *(_QWORD *)(v63 + 8 * v61 - 8);
        *(_QWORD *)(v63 + 8 * v61 - 8) = v64;
        if ( v65 )
          j_j___libc_free_0(v65);
        --v61;
      }
      while ( v61 );
      v66 = -8;
      LODWORD(v8) = v116;
      if ( v126 > 0 )
        v66 = -8 * v126;
      v10 = v111;
      v7 = v106;
      v16 = v132 + v66;
    }
LABEL_10:
    v17 = v130;
    v131 = v16;
    sub_24A69B0((_DWORD)v8, v135, v16, v17, v15, (_DWORD)v7, a7);
    a6 = v7;
    a4 = v129;
    a2 = v132;
    a5 = v10 - v15;
    a1 = v131;
    a3 = v138;
  }
  v37 = (unsigned __int64 *)a1;
  v38 = a6;
  v137 = a2 - a1;
  v39 = (a2 - a1) >> 3;
  if ( a2 - a1 > 0 )
  {
    v134 = a2;
    do
    {
      v40 = *v37;
      *v37 = 0;
      v41 = *v38;
      *v38 = v40;
      if ( v41 )
        j_j___libc_free_0(v41);
      ++v37;
      ++v38;
      --v39;
    }
    while ( v39 );
    v42 = v137;
    if ( v137 <= 0 )
      v42 = 8;
    v43 = (unsigned __int64 *)((char *)v7 + v42);
    if ( v7 != (unsigned __int64 *)((char *)v7 + v42) )
    {
      v44 = (unsigned __int64 *)v134;
      while ( v44 != (unsigned __int64 *)v138 )
      {
        v46 = *v7;
        v47 = *v44;
        if ( *(_QWORD *)(*v44 + 16) > *(_QWORD *)(*v7 + 16) )
        {
          *v44 = 0;
          v45 = *v8;
          *v8 = v47;
          if ( v45 )
            j_j___libc_free_0(v45);
          ++v44;
        }
        else
        {
          *v7 = 0;
          v48 = *v8;
          *v8 = v46;
          if ( v48 )
            j_j___libc_free_0(v48);
          ++v7;
        }
        ++v8;
        if ( v7 == v43 )
          return;
      }
      v32 = (char *)v43 - (char *)v7;
      v33 = v8;
      v34 = v32 >> 3;
      if ( v32 > 0 )
      {
        do
        {
          v35 = *v7;
          *v7 = 0;
          v36 = *v33;
          *v33 = v35;
          if ( v36 )
            j_j___libc_free_0(v36);
          ++v7;
          ++v33;
          --v34;
        }
        while ( v34 );
      }
    }
  }
}
