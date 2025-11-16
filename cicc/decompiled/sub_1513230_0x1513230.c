// Function: sub_1513230
// Address: 0x1513230
//
__int64 __fastcall sub_1513230(__int64 a1)
{
  _QWORD *v2; // rax
  unsigned int v3; // edi
  __int64 v4; // r14
  __int64 v5; // r9
  unsigned __int64 v6; // r8
  unsigned __int64 v7; // rax
  unsigned int v8; // ebx
  unsigned __int64 *v9; // r10
  unsigned __int64 v10; // rsi
  unsigned int v11; // r12d
  unsigned __int64 v12; // rsi
  char v13; // bl
  int v14; // r12d
  __int64 v15; // r13
  unsigned __int64 v16; // r11
  unsigned __int64 v17; // r8
  unsigned int v18; // r14d
  unsigned __int64 *v19; // r10
  unsigned __int64 v20; // rsi
  unsigned int v21; // r8d
  unsigned __int64 v22; // r13
  unsigned int v23; // edi
  unsigned __int64 v24; // rax
  unsigned int v25; // eax
  __int64 v26; // r11
  unsigned __int64 v27; // r8
  __int64 v28; // rax
  __int64 v29; // rdx
  char v30; // cl
  unsigned int v31; // r8d
  __int64 v32; // rax
  __int64 v33; // rdx
  char v34; // cl
  unsigned __int64 v35; // r9
  int v36; // eax
  unsigned __int64 v37; // r10
  unsigned __int64 v38; // rax
  unsigned __int64 *v39; // r9
  unsigned __int64 v40; // rsi
  unsigned int v41; // eax
  __int64 v42; // r12
  unsigned __int64 v43; // r11
  unsigned __int64 v44; // r8
  unsigned int v45; // ebx
  unsigned __int64 *v46; // r10
  unsigned __int64 v47; // rdi
  unsigned int v48; // r8d
  unsigned __int64 v49; // r14
  __int64 v50; // rbx
  __int64 v51; // rax
  _QWORD *v52; // rax
  __int64 result; // rax
  __int64 v54; // rsi
  __int64 v55; // rax
  volatile signed __int32 *v56; // r12
  unsigned __int64 v57; // rcx
  __int64 v58; // r14
  __int64 v59; // r13
  unsigned __int64 v60; // r11
  unsigned __int64 v61; // r8
  unsigned int v62; // r12d
  unsigned __int64 *v63; // r10
  unsigned __int64 v64; // rdi
  unsigned int v65; // r8d
  unsigned __int64 v66; // rbx
  __int64 v67; // rax
  unsigned __int64 *v68; // rax
  unsigned int v69; // edi
  __int64 v70; // r12
  unsigned __int64 v71; // r11
  unsigned __int64 v72; // r8
  unsigned int v73; // ebx
  unsigned __int64 *v74; // r10
  unsigned __int64 v75; // rsi
  unsigned int v76; // r8d
  unsigned __int64 v77; // rbx
  __int64 v78; // r12
  __int64 v79; // rax
  unsigned __int64 *v80; // rax
  unsigned int v81; // eax
  __int64 v82; // rdx
  __int64 v83; // rdi
  char v84; // cl
  unsigned __int64 v85; // r8
  int v86; // eax
  unsigned __int64 v87; // rdx
  unsigned int v88; // r8d
  __int64 v89; // rsi
  __int64 v90; // rdx
  char v91; // cl
  unsigned __int64 v92; // r9
  char v93; // r12
  __int64 v94; // r13
  unsigned __int64 v95; // r11
  unsigned __int64 v96; // r8
  unsigned int v97; // r14d
  unsigned __int64 *v98; // r10
  unsigned __int64 v99; // rsi
  unsigned int v100; // r8d
  unsigned __int64 v101; // r13
  unsigned int v102; // edi
  unsigned __int64 v103; // rax
  unsigned int v104; // r8d
  __int64 v105; // rax
  __int64 v106; // rdx
  char v107; // cl
  unsigned __int64 v108; // r9
  char v109; // r12
  __int64 v110; // r13
  unsigned __int64 v111; // r11
  unsigned __int64 v112; // r8
  unsigned int v113; // r14d
  unsigned __int64 *v114; // r10
  unsigned __int64 v115; // rsi
  unsigned int v116; // r8d
  unsigned __int64 v117; // r13
  unsigned int v118; // edi
  unsigned __int64 v119; // rax
  unsigned int v120; // r8d
  __int64 v121; // rax
  __int64 v122; // rdx
  char v123; // cl
  unsigned __int64 v124; // r9
  __int64 v125; // rbx
  __int64 v126; // rax
  _QWORD *v127; // rax
  unsigned __int64 v128; // rax
  unsigned __int64 v129; // rdx
  unsigned int v130; // r8d
  __int64 v131; // rax
  __int64 v132; // rdx
  char v133; // cl
  unsigned __int64 v134; // r9
  unsigned int v135; // r8d
  __int64 v136; // rdx
  __int64 v137; // rsi
  char v138; // cl
  unsigned __int64 v139; // r9
  unsigned __int64 v140; // rax
  char v141; // [rsp+8h] [rbp-58h]
  __int64 v142; // [rsp+8h] [rbp-58h]
  int v143; // [rsp+14h] [rbp-4Ch]
  __int64 v144; // [rsp+18h] [rbp-48h]
  int i; // [rsp+18h] [rbp-48h]
  _QWORD *v146; // [rsp+20h] [rbp-40h] BYREF
  volatile signed __int32 *v147; // [rsp+28h] [rbp-38h]

  v146 = 0;
  v2 = (_QWORD *)sub_22077B0(544);
  if ( v2 )
  {
    v2[1] = 0x100000001LL;
    v2[3] = 0x2000000000LL;
    *v2 = &unk_49ECD20;
    v2[2] = v2 + 4;
  }
  v3 = *(_DWORD *)(a1 + 32);
  v4 = (__int64)(v2 + 2);
  v147 = (volatile signed __int32 *)v2;
  v146 = v2 + 2;
  if ( v3 > 4 )
  {
    v140 = *(_QWORD *)(a1 + 24);
    *(_DWORD *)(a1 + 32) = v3 - 5;
    *(_QWORD *)(a1 + 24) = v140 >> 5;
    LODWORD(v12) = v140 & 0x1F;
    goto LABEL_10;
  }
  v5 = 0;
  if ( v3 )
    v5 = *(_QWORD *)(a1 + 24);
  v6 = *(_QWORD *)(a1 + 16);
  v7 = *(_QWORD *)(a1 + 8);
  v8 = 5 - v3;
  if ( v6 >= v7 )
    goto LABEL_25;
  v9 = (unsigned __int64 *)(v6 + *(_QWORD *)a1);
  if ( v7 < v6 + 8 )
  {
    v25 = v7 - v6;
    *(_QWORD *)(a1 + 24) = 0;
    v26 = v25;
    v11 = 8 * v25;
    v27 = v25 + v6;
    if ( v25 )
    {
      v28 = 0;
      v10 = 0;
      do
      {
        v29 = *((unsigned __int8 *)v9 + v28);
        v30 = 8 * v28++;
        v10 |= v29 << v30;
        *(_QWORD *)(a1 + 24) = v10;
      }
      while ( v26 != v28 );
      *(_QWORD *)(a1 + 16) = v27;
      *(_DWORD *)(a1 + 32) = v11;
      if ( v8 <= v11 )
        goto LABEL_9;
    }
    else
    {
      *(_QWORD *)(a1 + 16) = v27;
LABEL_157:
      *(_DWORD *)(a1 + 32) = 0;
    }
LABEL_25:
    sub_16BD130("Unexpected end of file", 1);
  }
  v10 = *v9;
  *(_QWORD *)(a1 + 16) = v6 + 8;
  v11 = 64;
LABEL_9:
  *(_QWORD *)(a1 + 24) = v10 >> v8;
  *(_DWORD *)(a1 + 32) = v3 + v11 - 5;
  v12 = v5 | (((0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v3 + 59)) & v10) << v3);
LABEL_10:
  v143 = v12;
  if ( (v12 & 0x10) != 0 )
  {
    v144 = v4;
    v13 = 0;
    v14 = v12 & 0xF;
    do
    {
      v23 = *(_DWORD *)(a1 + 32);
      v13 += 4;
      if ( v23 <= 4 )
      {
        v15 = 0;
        if ( v23 )
          v15 = *(_QWORD *)(a1 + 24);
        v16 = *(_QWORD *)(a1 + 16);
        v17 = *(_QWORD *)(a1 + 8);
        v18 = 5 - v23;
        if ( v16 >= v17 )
          goto LABEL_25;
        v19 = (unsigned __int64 *)(v16 + *(_QWORD *)a1);
        if ( v17 < v16 + 8 )
        {
          *(_QWORD *)(a1 + 24) = 0;
          v31 = v17 - v16;
          if ( !v31 )
            goto LABEL_157;
          v32 = 0;
          v20 = 0;
          do
          {
            v33 = *((unsigned __int8 *)v19 + v32);
            v34 = 8 * v32++;
            v20 |= v33 << v34;
            *(_QWORD *)(a1 + 24) = v20;
          }
          while ( v31 != v32 );
          v35 = v16 + v31;
          v21 = 8 * v31;
          *(_QWORD *)(a1 + 16) = v35;
          *(_DWORD *)(a1 + 32) = v21;
          if ( v18 > v21 )
            goto LABEL_25;
        }
        else
        {
          v20 = *v19;
          *(_QWORD *)(a1 + 16) = v16 + 8;
          v21 = 64;
        }
        *(_QWORD *)(a1 + 24) = v20 >> v18;
        *(_DWORD *)(a1 + 32) = v23 + v21 - 5;
        v22 = (((0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v23 + 59)) & v20) << v23) | v15;
      }
      else
      {
        v24 = *(_QWORD *)(a1 + 24);
        *(_DWORD *)(a1 + 32) = v23 - 5;
        *(_QWORD *)(a1 + 24) = v24 >> 5;
        LOBYTE(v22) = v24 & 0x1F;
      }
      v14 |= (v22 & 0xF) << v13;
    }
    while ( (v22 & 0x10) != 0 );
    v143 = v14;
    v4 = v144;
  }
  if ( !v143 )
    goto LABEL_55;
  for ( i = 0; i != v143; ++i )
  {
    v36 = *(_DWORD *)(a1 + 32);
    if ( v36 )
    {
      v57 = *(_QWORD *)(a1 + 24);
      v41 = v36 - 1;
      *(_DWORD *)(a1 + 32) = v41;
      *(_QWORD *)(a1 + 24) = v57 >> 1;
      if ( (v57 & 1) != 0 )
        goto LABEL_66;
    }
    else
    {
      v37 = *(_QWORD *)(a1 + 16);
      v38 = *(_QWORD *)(a1 + 8);
      if ( v37 >= v38 )
        goto LABEL_25;
      v39 = (unsigned __int64 *)(v37 + *(_QWORD *)a1);
      if ( v38 < v37 + 8 )
      {
        *(_QWORD *)(a1 + 24) = 0;
        v81 = v38 - v37;
        if ( !v81 )
          goto LABEL_25;
        v82 = 0;
        v40 = 0;
        do
        {
          v83 = *((unsigned __int8 *)v39 + v82);
          v84 = 8 * v82++;
          v40 |= v83 << v84;
          *(_QWORD *)(a1 + 24) = v40;
        }
        while ( v81 != v82 );
        v85 = v37 + v81;
        v86 = 8 * v81;
        *(_QWORD *)(a1 + 16) = v85;
        *(_DWORD *)(a1 + 32) = v86;
        if ( !v86 )
          goto LABEL_25;
        v41 = v86 - 1;
      }
      else
      {
        v40 = *v39;
        *(_QWORD *)(a1 + 16) = v37 + 8;
        v41 = 63;
      }
      *(_DWORD *)(a1 + 32) = v41;
      *(_QWORD *)(a1 + 24) = v40 >> 1;
      if ( (v40 & 1) != 0 )
      {
LABEL_66:
        v58 = (__int64)v146;
        if ( v41 > 7 )
        {
          v129 = *(_QWORD *)(a1 + 24);
          *(_DWORD *)(a1 + 32) = v41 - 8;
          LODWORD(v66) = (unsigned __int8)v129;
          *(_QWORD *)(a1 + 24) = v129 >> 8;
        }
        else
        {
          v59 = 0;
          if ( v41 )
            v59 = *(_QWORD *)(a1 + 24);
          v60 = *(_QWORD *)(a1 + 16);
          v61 = *(_QWORD *)(a1 + 8);
          v62 = 8 - v41;
          if ( v60 >= v61 )
            goto LABEL_25;
          v63 = (unsigned __int64 *)(v60 + *(_QWORD *)a1);
          if ( v61 < v60 + 8 )
          {
            *(_QWORD *)(a1 + 24) = 0;
            v135 = v61 - v60;
            if ( !v135 )
              goto LABEL_157;
            v136 = 0;
            v64 = 0;
            do
            {
              v137 = *((unsigned __int8 *)v63 + v136);
              v138 = 8 * v136++;
              v64 |= v137 << v138;
              *(_QWORD *)(a1 + 24) = v64;
            }
            while ( v135 != v136 );
            v139 = v60 + v135;
            v65 = 8 * v135;
            *(_QWORD *)(a1 + 16) = v139;
            *(_DWORD *)(a1 + 32) = v65;
            if ( v62 > v65 )
              goto LABEL_25;
          }
          else
          {
            v64 = *v63;
            *(_QWORD *)(a1 + 16) = v60 + 8;
            v65 = 64;
          }
          *(_QWORD *)(a1 + 24) = v64 >> v62;
          *(_DWORD *)(a1 + 32) = v65 + v41 - 8;
          v66 = v59 | (((0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v41 + 56)) & v64) << v41);
        }
        if ( (v66 & 0x80u) != 0LL )
        {
          v142 = v58;
          v66 &= 0x7Fu;
          v109 = 0;
          do
          {
            v118 = *(_DWORD *)(a1 + 32);
            v109 += 7;
            if ( v118 <= 7 )
            {
              v110 = 0;
              if ( v118 )
                v110 = *(_QWORD *)(a1 + 24);
              v111 = *(_QWORD *)(a1 + 16);
              v112 = *(_QWORD *)(a1 + 8);
              v113 = 8 - v118;
              if ( v111 >= v112 )
                goto LABEL_25;
              v114 = (unsigned __int64 *)(v111 + *(_QWORD *)a1);
              if ( v112 < v111 + 8 )
              {
                *(_QWORD *)(a1 + 24) = 0;
                v120 = v112 - v111;
                if ( !v120 )
                  goto LABEL_157;
                v121 = 0;
                v115 = 0;
                do
                {
                  v122 = *((unsigned __int8 *)v114 + v121);
                  v123 = 8 * v121++;
                  v115 |= v122 << v123;
                  *(_QWORD *)(a1 + 24) = v115;
                }
                while ( v120 != v121 );
                v124 = v111 + v120;
                v116 = 8 * v120;
                *(_QWORD *)(a1 + 16) = v124;
                *(_DWORD *)(a1 + 32) = v116;
                if ( v113 > v116 )
                  goto LABEL_25;
              }
              else
              {
                v115 = *v114;
                *(_QWORD *)(a1 + 16) = v111 + 8;
                v116 = 64;
              }
              *(_QWORD *)(a1 + 24) = v115 >> v113;
              *(_DWORD *)(a1 + 32) = v118 + v116 - 8;
              v117 = (((0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v118 + 56)) & v115) << v118) | v110;
            }
            else
            {
              v119 = *(_QWORD *)(a1 + 24);
              *(_DWORD *)(a1 + 32) = v118 - 8;
              LOBYTE(v117) = v119;
              *(_QWORD *)(a1 + 24) = v119 >> 8;
            }
            v66 |= (v117 & 0x7F) << v109;
          }
          while ( (v117 & 0x80) != 0 );
          v58 = v142;
          v67 = *(unsigned int *)(v142 + 8);
          if ( (unsigned int)v67 < *(_DWORD *)(v142 + 12) )
            goto LABEL_75;
        }
        else
        {
          v67 = *(unsigned int *)(v58 + 8);
          v66 = (unsigned int)v66;
          if ( (unsigned int)v67 < *(_DWORD *)(v58 + 12) )
          {
LABEL_75:
            v68 = (unsigned __int64 *)(*(_QWORD *)v58 + 16 * v67);
            *v68 = v66;
            v68[1] = 1;
            ++*(_DWORD *)(v58 + 8);
            continue;
          }
        }
        sub_16CD150(v58, v58 + 16, 0, 16);
        v67 = *(unsigned int *)(v58 + 8);
        goto LABEL_75;
      }
    }
    if ( v41 > 2 )
    {
      v87 = *(_QWORD *)(a1 + 24);
      *(_DWORD *)(a1 + 32) = v41 - 3;
      *(_QWORD *)(a1 + 24) = v87 >> 3;
      LODWORD(v49) = v87 & 7;
    }
    else
    {
      v42 = 0;
      if ( v41 )
        v42 = *(_QWORD *)(a1 + 24);
      v43 = *(_QWORD *)(a1 + 16);
      v44 = *(_QWORD *)(a1 + 8);
      v45 = 3 - v41;
      if ( v43 >= v44 )
        goto LABEL_25;
      v46 = (unsigned __int64 *)(v43 + *(_QWORD *)a1);
      if ( v44 < v43 + 8 )
      {
        *(_QWORD *)(a1 + 24) = 0;
        v88 = v44 - v43;
        if ( !v88 )
          goto LABEL_157;
        v89 = 0;
        v47 = 0;
        do
        {
          v90 = *((unsigned __int8 *)v46 + v89);
          v91 = 8 * v89++;
          v47 |= v90 << v91;
          *(_QWORD *)(a1 + 24) = v47;
        }
        while ( v88 != v89 );
        v92 = v43 + v88;
        v48 = 8 * v88;
        *(_QWORD *)(a1 + 16) = v92;
        *(_DWORD *)(a1 + 32) = v48;
        if ( v45 > v48 )
          goto LABEL_25;
      }
      else
      {
        v47 = *v46;
        *(_QWORD *)(a1 + 16) = v43 + 8;
        v48 = 64;
      }
      *(_QWORD *)(a1 + 24) = v47 >> v45;
      *(_DWORD *)(a1 + 32) = v48 + v41 - 3;
      v49 = v42 | (((0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v41 + 61)) & v47) << v41);
    }
    if ( (unsigned int)v49 > 2 )
    {
      if ( (unsigned int)(v49 - 3) > 2 )
LABEL_48:
        sub_16BD130("Invalid encoding", 1);
      v50 = (__int64)v146;
      v51 = *((unsigned int *)v146 + 2);
      if ( (unsigned int)v51 >= *((_DWORD *)v146 + 3) )
      {
        sub_16CD150(v146, v146 + 2, 0, 16);
        v51 = *((unsigned int *)v146 + 2);
      }
      v52 = (_QWORD *)(*v146 + 16 * v51);
      *v52 = 0;
      v52[1] = 2 * (v49 & 7);
      ++*(_DWORD *)(v50 + 8);
    }
    else
    {
      if ( !(_DWORD)v49 )
        goto LABEL_48;
      v69 = *(_DWORD *)(a1 + 32);
      if ( v69 > 4 )
      {
        v128 = *(_QWORD *)(a1 + 24);
        *(_DWORD *)(a1 + 32) = v69 - 5;
        *(_QWORD *)(a1 + 24) = v128 >> 5;
        LODWORD(v77) = v128 & 0x1F;
      }
      else
      {
        v70 = 0;
        if ( v69 )
          v70 = *(_QWORD *)(a1 + 24);
        v71 = *(_QWORD *)(a1 + 16);
        v72 = *(_QWORD *)(a1 + 8);
        v73 = 5 - v69;
        if ( v71 >= v72 )
          goto LABEL_25;
        v74 = (unsigned __int64 *)(v71 + *(_QWORD *)a1);
        if ( v72 < v71 + 8 )
        {
          *(_QWORD *)(a1 + 24) = 0;
          v130 = v72 - v71;
          if ( !v130 )
            goto LABEL_157;
          v131 = 0;
          v75 = 0;
          do
          {
            v132 = *((unsigned __int8 *)v74 + v131);
            v133 = 8 * v131++;
            v75 |= v132 << v133;
            *(_QWORD *)(a1 + 24) = v75;
          }
          while ( v130 != v131 );
          v134 = v71 + v130;
          v76 = 8 * v130;
          *(_QWORD *)(a1 + 16) = v134;
          *(_DWORD *)(a1 + 32) = v76;
          if ( v73 > v76 )
            goto LABEL_25;
        }
        else
        {
          v75 = *v74;
          *(_QWORD *)(a1 + 16) = v71 + 8;
          v76 = 64;
        }
        *(_QWORD *)(a1 + 24) = v75 >> v73;
        *(_DWORD *)(a1 + 32) = v69 + v76 - 5;
        v77 = v70 | (((0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v69 + 59)) & v75) << v69);
      }
      if ( (v77 & 0x10) != 0 )
      {
        v141 = v49;
        v77 &= 0xFu;
        v93 = 0;
        do
        {
          v102 = *(_DWORD *)(a1 + 32);
          v93 += 4;
          if ( v102 <= 4 )
          {
            v94 = 0;
            if ( v102 )
              v94 = *(_QWORD *)(a1 + 24);
            v95 = *(_QWORD *)(a1 + 16);
            v96 = *(_QWORD *)(a1 + 8);
            v97 = 5 - v102;
            if ( v95 >= v96 )
              goto LABEL_25;
            v98 = (unsigned __int64 *)(v95 + *(_QWORD *)a1);
            if ( v96 < v95 + 8 )
            {
              *(_QWORD *)(a1 + 24) = 0;
              v104 = v96 - v95;
              if ( !v104 )
                goto LABEL_157;
              v105 = 0;
              v99 = 0;
              do
              {
                v106 = *((unsigned __int8 *)v98 + v105);
                v107 = 8 * v105++;
                v99 |= v106 << v107;
                *(_QWORD *)(a1 + 24) = v99;
              }
              while ( v104 != v105 );
              v108 = v95 + v104;
              v100 = 8 * v104;
              *(_QWORD *)(a1 + 16) = v108;
              *(_DWORD *)(a1 + 32) = v100;
              if ( v97 > v100 )
                goto LABEL_25;
            }
            else
            {
              v99 = *v98;
              *(_QWORD *)(a1 + 16) = v95 + 8;
              v100 = 64;
            }
            *(_QWORD *)(a1 + 24) = v99 >> v97;
            *(_DWORD *)(a1 + 32) = v102 + v100 - 5;
            v101 = (((0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v102 + 59)) & v99) << v102) | v94;
          }
          else
          {
            v103 = *(_QWORD *)(a1 + 24);
            *(_DWORD *)(a1 + 32) = v102 - 5;
            *(_QWORD *)(a1 + 24) = v103 >> 5;
            LOBYTE(v101) = v103 & 0x1F;
          }
          v77 |= (v101 & 0xF) << v93;
        }
        while ( (v101 & 0x10) != 0 );
        LOBYTE(v49) = v141;
        if ( v77 )
        {
LABEL_85:
          if ( v77 > 0x40 )
            sub_16BD130("Fixed or VBR abbrev record with size > MaxChunkData", 1);
          v78 = (__int64)v146;
          v79 = *((unsigned int *)v146 + 2);
          if ( (unsigned int)v79 >= *((_DWORD *)v146 + 3) )
          {
            sub_16CD150(v146, v146 + 2, 0, 16);
            v79 = *((unsigned int *)v146 + 2);
          }
          v80 = (unsigned __int64 *)(*v146 + 16 * v79);
          *v80 = v77;
          v80[1] = (unsigned __int8)(2 * (v49 & 7));
          ++*(_DWORD *)(v78 + 8);
          continue;
        }
      }
      else
      {
        v77 = (unsigned int)v77;
        if ( (_DWORD)v77 )
          goto LABEL_85;
      }
      v125 = (__int64)v146;
      v126 = *((unsigned int *)v146 + 2);
      if ( (unsigned int)v126 >= *((_DWORD *)v146 + 3) )
      {
        sub_16CD150(v146, v146 + 2, 0, 16);
        v126 = *((unsigned int *)v146 + 2);
      }
      v127 = (_QWORD *)(*v146 + 16 * v126);
      *v127 = 0;
      v127[1] = 1;
      ++*(_DWORD *)(v125 + 8);
    }
  }
  v4 = (__int64)v146;
LABEL_55:
  result = *(unsigned int *)(v4 + 8);
  if ( !(_DWORD)result )
    sub_16BD130("Abbrev record with no operands", 1);
  v54 = *(_QWORD *)(a1 + 48);
  if ( v54 == *(_QWORD *)(a1 + 56) )
  {
    result = sub_1512F90((char **)(a1 + 40), (char *)v54, (__int64 *)&v146);
  }
  else
  {
    if ( v54 )
    {
      v55 = (__int64)v146;
      *(_QWORD *)(v54 + 8) = 0;
      *(_QWORD *)v54 = v55;
      result = (__int64)v147;
      v146 = 0;
      v147 = 0;
      *(_QWORD *)(v54 + 8) = result;
      v54 = *(_QWORD *)(a1 + 48);
    }
    *(_QWORD *)(a1 + 48) = v54 + 16;
  }
  v56 = v147;
  if ( v147 )
  {
    if ( &_pthread_key_create )
    {
      result = (unsigned int)_InterlockedExchangeAdd(v147 + 2, 0xFFFFFFFF);
    }
    else
    {
      result = *((unsigned int *)v147 + 2);
      *((_DWORD *)v147 + 2) = result - 1;
    }
    if ( (_DWORD)result == 1 )
    {
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v56 + 16LL))(v56);
      if ( &_pthread_key_create )
      {
        result = (unsigned int)_InterlockedExchangeAdd(v56 + 3, 0xFFFFFFFF);
      }
      else
      {
        result = *((unsigned int *)v56 + 3);
        *((_DWORD *)v56 + 3) = result - 1;
      }
      if ( (_DWORD)result == 1 )
        return (*(__int64 (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v56 + 24LL))(v56);
    }
  }
  return result;
}
