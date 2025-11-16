// Function: sub_21084A0
// Address: 0x21084a0
//
__int64 __fastcall sub_21084A0(__int64 a1, _QWORD *a2)
{
  _QWORD *v2; // r14
  __int64 *v3; // rsi
  __int64 result; // rax
  __int64 v6; // r15
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 v10; // rbx
  __int64 *v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rsi
  __int64 *v15; // rdx
  __int64 v16; // rdx
  int v17; // r13d
  __int64 v18; // rbx
  unsigned int v19; // esi
  __int64 v20; // r8
  unsigned int v21; // edx
  __int64 *v22; // rax
  __int64 v23; // rdi
  __int64 *v24; // rbx
  __int64 v25; // r15
  __int64 v26; // rax
  int v27; // r8d
  int v28; // r9d
  __int64 v29; // r12
  unsigned int v30; // r15d
  __int64 *v31; // rcx
  __int64 v32; // rdx
  _QWORD *v33; // rax
  __int64 v34; // rbx
  __int64 v35; // r14
  __int64 v36; // rbx
  __int64 v37; // rdx
  _QWORD *v38; // rbx
  __int64 v39; // rcx
  __int64 v40; // r9
  unsigned int v41; // edi
  _QWORD *v42; // rax
  __int64 v43; // r8
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // r14
  __int64 v47; // r13
  unsigned int v48; // esi
  int v49; // r15d
  __int64 v50; // r9
  unsigned int v51; // edi
  __int64 *v52; // rax
  __int64 v53; // r8
  unsigned int v54; // esi
  int v55; // r14d
  int v56; // r14d
  __int64 v57; // r11
  unsigned int v58; // esi
  int v59; // eax
  _QWORD *v60; // rdx
  __int64 v61; // r9
  __int64 v62; // rdx
  unsigned int v63; // esi
  __int64 v64; // r9
  unsigned int v65; // ecx
  __int64 *v66; // rax
  __int64 v67; // r8
  __int64 *v68; // r11
  int v69; // edi
  int v70; // edi
  int v71; // r11d
  int v72; // eax
  int v73; // r11d
  int v74; // r11d
  __int64 v75; // r10
  _QWORD *v76; // rsi
  unsigned int v77; // r14d
  int v78; // edi
  __int64 v79; // r8
  int v80; // eax
  int v81; // r9d
  __int64 v82; // r11
  unsigned int v83; // edx
  __int64 v84; // r8
  int v85; // r10d
  __int64 *v86; // rsi
  int v87; // eax
  int v88; // r9d
  __int64 v89; // r11
  int v90; // r10d
  unsigned int v91; // edx
  __int64 v92; // r8
  int v93; // r11d
  __int64 *v94; // r12
  int v95; // edi
  int v96; // ecx
  __int64 v97; // rdx
  int v98; // eax
  int v99; // r10d
  __int64 v100; // r12
  int v101; // edi
  __int64 *v102; // rsi
  unsigned int v103; // r11d
  __int64 v104; // r8
  int v105; // eax
  int v106; // r10d
  __int64 v107; // r12
  unsigned int v108; // r11d
  __int64 v109; // r8
  int v110; // edi
  int v111; // r10d
  __int64 *v112; // r11
  int v113; // edi
  int v114; // edx
  __int64 v115; // rdx
  int v116; // eax
  int v117; // r9d
  int v118; // r10d
  __int64 v119; // r11
  __int64 *v120; // rsi
  unsigned int v121; // ecx
  __int64 v122; // rdi
  int v123; // eax
  int v124; // r9d
  __int64 v125; // r11
  unsigned int v126; // ecx
  __int64 v127; // rdi
  int v128; // r10d
  int v129; // r8d
  _QWORD *v130; // rdi
  __int64 v131; // [rsp+8h] [rbp-98h]
  __int64 v132; // [rsp+10h] [rbp-90h]
  int v133; // [rsp+10h] [rbp-90h]
  __int64 v134; // [rsp+10h] [rbp-90h]
  __int64 v135; // [rsp+10h] [rbp-90h]
  __int64 v136; // [rsp+18h] [rbp-88h]
  _QWORD *v137; // [rsp+20h] [rbp-80h]
  _QWORD *v138; // [rsp+20h] [rbp-80h]
  _QWORD *v139; // [rsp+28h] [rbp-78h]
  __int64 v140; // [rsp+28h] [rbp-78h]
  __int64 *v141; // [rsp+30h] [rbp-70h]
  __int64 *v142; // [rsp+30h] [rbp-70h]
  __int64 v143; // [rsp+30h] [rbp-70h]
  __int64 v144; // [rsp+30h] [rbp-70h]
  __int64 *v145; // [rsp+38h] [rbp-68h]
  __int64 v146; // [rsp+38h] [rbp-68h]
  __m128i v147; // [rsp+40h] [rbp-60h] BYREF
  __int64 v148; // [rsp+50h] [rbp-50h]
  __int64 v149; // [rsp+58h] [rbp-48h]
  __int64 v150; // [rsp+60h] [rbp-40h]

  v2 = a2;
  v3 = (__int64 *)*a2;
  v136 = a1 + 24;
  result = (__int64)&v3[*((unsigned int *)v2 + 2)];
  v145 = v3;
  v141 = (__int64 *)result;
  if ( (__int64 *)result == v3 )
    return result;
  do
  {
    while ( 1 )
    {
      v6 = *v145;
      if ( *(_QWORD *)(v6 + 16) == v6 )
        break;
LABEL_3:
      if ( ++v145 == v141 )
        goto LABEL_18;
    }
    v7 = *(_QWORD *)v6;
    v8 = sub_1DD5D10(*(_QWORD *)v6);
    v9 = *(_QWORD *)(v7 + 32);
    v10 = v8;
    if ( v9 == v8 )
      goto LABEL_12;
    do
    {
      if ( (unsigned __int8)sub_2107D70(a1, v9) )
      {
        v38 = (_QWORD *)*v2;
        v39 = *v2 + 8LL * *((unsigned int *)v2 + 2);
        if ( *v2 == v39 )
          break;
        v140 = v6;
        v138 = v2;
        while ( 1 )
        {
          v45 = *(_QWORD *)(*v38 + 56LL);
          if ( !v45 )
            goto LABEL_42;
          v46 = *(_QWORD *)(a1 + 8);
          v47 = *(_QWORD *)(v45 + 24);
          v48 = *(_DWORD *)(v46 + 24);
          v49 = *(_DWORD *)(*(_QWORD *)(v45 + 32) + 8LL);
          if ( v48 )
          {
            v50 = *(_QWORD *)(v46 + 8);
            v51 = (v48 - 1) & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
            v52 = (__int64 *)(v50 + 16LL * v51);
            v53 = *v52;
            if ( v47 == *v52 )
              goto LABEL_46;
            v133 = 1;
            v68 = 0;
            while ( v53 != -8 )
            {
              if ( !v68 && v53 == -16 )
                v68 = v52;
              v51 = (v48 - 1) & (v133 + v51);
              v52 = (__int64 *)(v50 + 16LL * v51);
              v53 = *v52;
              if ( v47 == *v52 )
                goto LABEL_46;
              ++v133;
            }
            v69 = *(_DWORD *)(v46 + 16);
            if ( v68 )
              v52 = v68;
            ++*(_QWORD *)v46;
            v70 = v69 + 1;
            if ( 4 * v70 < 3 * v48 )
            {
              if ( v48 - *(_DWORD *)(v46 + 20) - v70 > v48 >> 3 )
                goto LABEL_63;
              v131 = v39;
              sub_1DA35E0(v46, v48);
              v87 = *(_DWORD *)(v46 + 24);
              if ( !v87 )
              {
LABEL_198:
                ++*(_DWORD *)(v46 + 16);
                BUG();
              }
              v88 = v87 - 1;
              v89 = *(_QWORD *)(v46 + 8);
              v90 = 1;
              v39 = v131;
              v91 = (v87 - 1) & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
              v70 = *(_DWORD *)(v46 + 16) + 1;
              v86 = 0;
              v52 = (__int64 *)(v89 + 16LL * v91);
              v92 = *v52;
              if ( v47 == *v52 )
                goto LABEL_63;
              while ( v92 != -8 )
              {
                if ( v92 == -16 && !v86 )
                  v86 = v52;
                v91 = v88 & (v90 + v91);
                v52 = (__int64 *)(v89 + 16LL * v91);
                v92 = *v52;
                if ( v47 == *v52 )
                  goto LABEL_63;
                ++v90;
              }
              goto LABEL_91;
            }
          }
          else
          {
            ++*(_QWORD *)v46;
          }
          v135 = v39;
          sub_1DA35E0(v46, 2 * v48);
          v80 = *(_DWORD *)(v46 + 24);
          if ( !v80 )
            goto LABEL_198;
          v81 = v80 - 1;
          v82 = *(_QWORD *)(v46 + 8);
          v39 = v135;
          v83 = (v80 - 1) & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
          v70 = *(_DWORD *)(v46 + 16) + 1;
          v52 = (__int64 *)(v82 + 16LL * v83);
          v84 = *v52;
          if ( v47 == *v52 )
            goto LABEL_63;
          v85 = 1;
          v86 = 0;
          while ( v84 != -8 )
          {
            if ( !v86 && v84 == -16 )
              v86 = v52;
            v83 = v81 & (v85 + v83);
            v52 = (__int64 *)(v82 + 16LL * v83);
            v84 = *v52;
            if ( v47 == *v52 )
              goto LABEL_63;
            ++v85;
          }
LABEL_91:
          if ( v86 )
            v52 = v86;
LABEL_63:
          *(_DWORD *)(v46 + 16) = v70;
          if ( *v52 != -8 )
            --*(_DWORD *)(v46 + 20);
          *v52 = v47;
          *((_DWORD *)v52 + 2) = 0;
LABEL_46:
          *((_DWORD *)v52 + 2) = v49;
          v54 = *(_DWORD *)(a1 + 48);
          if ( !v54 )
          {
            ++*(_QWORD *)(a1 + 24);
            goto LABEL_48;
          }
          v40 = *(_QWORD *)(a1 + 32);
          v41 = (v54 - 1) & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
          v42 = (_QWORD *)(v40 + 16LL * v41);
          v43 = *v42;
          if ( v47 != *v42 )
          {
            v71 = 1;
            v60 = 0;
            while ( v43 != -8 )
            {
              if ( !v60 && v43 == -16 )
                v60 = v42;
              v41 = (v54 - 1) & (v71 + v41);
              v42 = (_QWORD *)(v40 + 16LL * v41);
              v43 = *v42;
              if ( v47 == *v42 )
                goto LABEL_40;
              ++v71;
            }
            if ( !v60 )
              v60 = v42;
            v72 = *(_DWORD *)(a1 + 40);
            ++*(_QWORD *)(a1 + 24);
            v59 = v72 + 1;
            if ( 4 * v59 >= 3 * v54 )
            {
LABEL_48:
              v132 = v39;
              sub_2107BB0(v136, 2 * v54);
              v55 = *(_DWORD *)(a1 + 48);
              if ( !v55 )
                goto LABEL_197;
              v56 = v55 - 1;
              v57 = *(_QWORD *)(a1 + 32);
              v39 = v132;
              v58 = v56 & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
              v59 = *(_DWORD *)(a1 + 40) + 1;
              v60 = (_QWORD *)(v57 + 16LL * v58);
              v61 = *v60;
              if ( v47 != *v60 )
              {
                v129 = 1;
                v130 = 0;
                while ( v61 != -8 )
                {
                  if ( !v130 && v61 == -16 )
                    v130 = v60;
                  v58 = v56 & (v129 + v58);
                  v60 = (_QWORD *)(v57 + 16LL * v58);
                  v61 = *v60;
                  if ( v47 == *v60 )
                    goto LABEL_50;
                  ++v129;
                }
                if ( v130 )
                  v60 = v130;
              }
            }
            else if ( v54 - *(_DWORD *)(a1 + 44) - v59 <= v54 >> 3 )
            {
              v134 = v39;
              sub_2107BB0(v136, v54);
              v73 = *(_DWORD *)(a1 + 48);
              if ( !v73 )
              {
LABEL_197:
                ++*(_DWORD *)(a1 + 40);
                BUG();
              }
              v74 = v73 - 1;
              v75 = *(_QWORD *)(a1 + 32);
              v76 = 0;
              v77 = v74 & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
              v39 = v134;
              v78 = 1;
              v59 = *(_DWORD *)(a1 + 40) + 1;
              v60 = (_QWORD *)(v75 + 16LL * v77);
              v79 = *v60;
              if ( v47 != *v60 )
              {
                while ( v79 != -8 )
                {
                  if ( v79 == -16 && !v76 )
                    v76 = v60;
                  v77 = v74 & (v78 + v77);
                  v60 = (_QWORD *)(v75 + 16LL * v77);
                  v79 = *v60;
                  if ( v47 == *v60 )
                    goto LABEL_50;
                  ++v78;
                }
                if ( v76 )
                  v60 = v76;
              }
            }
LABEL_50:
            *(_DWORD *)(a1 + 40) = v59;
            if ( *v60 != -8 )
              --*(_DWORD *)(a1 + 44);
            *v60 = v47;
            v44 = 0;
            v60[1] = 0;
            goto LABEL_41;
          }
LABEL_40:
          v44 = v42[1];
LABEL_41:
          *(_DWORD *)(v44 + 8) = v49;
LABEL_42:
          if ( (_QWORD *)v39 == ++v38 )
          {
            v6 = v140;
            v2 = v138;
            goto LABEL_12;
          }
        }
      }
      v11 = (__int64 *)*v2;
      v12 = *v2 + 8LL * *((unsigned int *)v2 + 2);
      if ( *v2 != v12 )
      {
        do
        {
          v13 = *v11++;
          *(_QWORD *)(v13 + 56) = 0;
        }
        while ( (__int64 *)v12 != v11 );
      }
      if ( !v9 )
        BUG();
      if ( (*(_BYTE *)v9 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v9 + 46) & 8) != 0 )
          v9 = *(_QWORD *)(v9 + 8);
      }
      v9 = *(_QWORD *)(v9 + 8);
    }
    while ( v10 != v9 );
LABEL_12:
    if ( *(_DWORD *)(v6 + 8) )
      goto LABEL_3;
    v14 = *(_QWORD *)v6;
    v15 = (__int64 *)(*(_QWORD *)(*(_QWORD *)v6 + 24LL) & 0xFFFFFFFFFFFFFFF8LL);
    if ( v15 != (__int64 *)(*(_QWORD *)v6 + 24LL) )
      v15 = *(__int64 **)(v14 + 32);
    sub_21072E0(
      0,
      v14,
      v15,
      *(_QWORD *)(*(_QWORD *)a1 + 16LL),
      *(_QWORD *)(*(_QWORD *)a1 + 40LL),
      *(_QWORD *)(*(_QWORD *)a1 + 32LL));
    v17 = *(_DWORD *)(*(_QWORD *)(v16 + 32) + 8LL);
    *(_DWORD *)(v6 + 8) = v17;
    v18 = *(_QWORD *)(a1 + 8);
    v19 = *(_DWORD *)(v18 + 24);
    if ( !v19 )
    {
      ++*(_QWORD *)v18;
LABEL_137:
      sub_1DA35E0(v18, 2 * v19);
      v123 = *(_DWORD *)(v18 + 24);
      if ( v123 )
      {
        v124 = v123 - 1;
        v125 = *(_QWORD *)(v18 + 8);
        v114 = *(_DWORD *)(v18 + 16) + 1;
        v126 = (v123 - 1) & (((unsigned int)*(_QWORD *)v6 >> 9) ^ ((unsigned int)*(_QWORD *)v6 >> 4));
        v22 = (__int64 *)(v125 + 16LL * v126);
        v127 = *v22;
        if ( *(_QWORD *)v6 == *v22 )
          goto LABEL_125;
        v128 = 1;
        v120 = 0;
        while ( v127 != -8 )
        {
          if ( v127 == -16 && !v120 )
            v120 = v22;
          v126 = v124 & (v128 + v126);
          v22 = (__int64 *)(v125 + 16LL * v126);
          v127 = *v22;
          if ( *(_QWORD *)v6 == *v22 )
            goto LABEL_125;
          ++v128;
        }
        goto LABEL_141;
      }
LABEL_196:
      ++*(_DWORD *)(v18 + 16);
      BUG();
    }
    v20 = *(_QWORD *)(v18 + 8);
    v21 = (v19 - 1) & (((unsigned int)*(_QWORD *)v6 >> 9) ^ ((unsigned int)*(_QWORD *)v6 >> 4));
    v22 = (__int64 *)(v20 + 16LL * v21);
    v23 = *v22;
    if ( *v22 == *(_QWORD *)v6 )
      goto LABEL_17;
    v111 = 1;
    v112 = 0;
    while ( v23 != -8 )
    {
      if ( !v112 && v23 == -16 )
        v112 = v22;
      v21 = (v19 - 1) & (v111 + v21);
      v22 = (__int64 *)(v20 + 16LL * v21);
      v23 = *v22;
      if ( *(_QWORD *)v6 == *v22 )
        goto LABEL_17;
      ++v111;
    }
    v113 = *(_DWORD *)(v18 + 16);
    if ( v112 )
      v22 = v112;
    ++*(_QWORD *)v18;
    v114 = v113 + 1;
    if ( 4 * (v113 + 1) >= 3 * v19 )
      goto LABEL_137;
    if ( v19 - *(_DWORD *)(v18 + 20) - v114 <= v19 >> 3 )
    {
      sub_1DA35E0(v18, v19);
      v116 = *(_DWORD *)(v18 + 24);
      if ( v116 )
      {
        v117 = v116 - 1;
        v118 = 1;
        v119 = *(_QWORD *)(v18 + 8);
        v114 = *(_DWORD *)(v18 + 16) + 1;
        v120 = 0;
        v121 = (v116 - 1) & (((unsigned int)*(_QWORD *)v6 >> 9) ^ ((unsigned int)*(_QWORD *)v6 >> 4));
        v22 = (__int64 *)(v119 + 16LL * v121);
        v122 = *v22;
        if ( *(_QWORD *)v6 == *v22 )
          goto LABEL_125;
        while ( v122 != -8 )
        {
          if ( v122 == -16 && !v120 )
            v120 = v22;
          v121 = v117 & (v118 + v121);
          v22 = (__int64 *)(v119 + 16LL * v121);
          v122 = *v22;
          if ( *(_QWORD *)v6 == *v22 )
            goto LABEL_125;
          ++v118;
        }
LABEL_141:
        if ( v120 )
          v22 = v120;
        goto LABEL_125;
      }
      goto LABEL_196;
    }
LABEL_125:
    *(_DWORD *)(v18 + 16) = v114;
    if ( *v22 != -8 )
      --*(_DWORD *)(v18 + 20);
    v115 = *(_QWORD *)v6;
    *((_DWORD *)v22 + 2) = 0;
    *v22 = v115;
LABEL_17:
    ++v145;
    *((_DWORD *)v22 + 2) = v17;
  }
  while ( v145 != v141 );
LABEL_18:
  result = *v2 + 8LL * *((unsigned int *)v2 + 2);
  v139 = (_QWORD *)*v2;
  v146 = result;
  if ( *v2 != result )
  {
    v137 = (_QWORD *)a1;
    while ( 1 )
    {
      while ( 1 )
      {
        v24 = *(__int64 **)(v146 - 8);
        v25 = v24[2];
        if ( (__int64 *)v25 != v24 )
          break;
        v26 = sub_1E69D00(*(_QWORD *)(*v137 + 40LL), *((_DWORD *)v24 + 2));
        v29 = v26;
        if ( !v26 || **(_WORD **)(v26 + 16) != 45 && **(_WORD **)(v26 + 16) || *(_DWORD *)(v26 + 40) > 1u )
          goto LABEL_21;
        v30 = 0;
        v31 = v24;
        if ( *((_DWORD *)v24 + 10) )
        {
          do
          {
            v32 = v30;
            v142 = v31;
            ++v30;
            v33 = *(_QWORD **)(v31[6] + 8 * v32);
            v34 = *v33;
            v35 = *(_QWORD *)(*v33 + 56LL);
            LODWORD(v33) = *(_DWORD *)(v33[2] + 8LL);
            v147.m128i_i64[0] = 0;
            v148 = 0;
            v147.m128i_i32[2] = (int)v33;
            v149 = 0;
            v150 = 0;
            sub_1E1A9C0(v29, v35, &v147);
            v147.m128i_i8[0] = 4;
            v148 = 0;
            v147.m128i_i32[0] &= 0xFFF000FF;
            v149 = v34;
            sub_1E1A9C0(v29, v35, &v147);
            v31 = v142;
          }
          while ( *((_DWORD *)v142 + 10) != v30 );
        }
        v36 = v137[2];
        if ( !v36 )
          goto LABEL_21;
        v37 = *(unsigned int *)(v36 + 8);
        if ( (unsigned int)v37 >= *(_DWORD *)(v36 + 12) )
        {
          sub_16CD150(v137[2], (const void *)(v36 + 16), 0, 8, v27, v28);
          v37 = *(unsigned int *)(v36 + 8);
        }
        v146 -= 8;
        result = v146;
        *(_QWORD *)(*(_QWORD *)v36 + 8 * v37) = v29;
        ++*(_DWORD *)(v36 + 8);
        if ( v139 == (_QWORD *)v146 )
          return result;
      }
      if ( *((_DWORD *)v24 + 10) > 1u )
        break;
LABEL_21:
      v146 -= 8;
      result = v146;
      if ( v139 == (_QWORD *)v146 )
        return result;
    }
    v62 = v137[1];
    v63 = *(_DWORD *)(v62 + 24);
    if ( v63 )
    {
      v64 = *(_QWORD *)(v62 + 8);
      v65 = (v63 - 1) & (((unsigned int)*v24 >> 9) ^ ((unsigned int)*v24 >> 4));
      v66 = (__int64 *)(v64 + 16LL * v65);
      v67 = *v66;
      if ( *v24 == *v66 )
      {
LABEL_56:
        *((_DWORD *)v66 + 2) = *(_DWORD *)(v25 + 8);
        goto LABEL_21;
      }
      v93 = 1;
      v94 = 0;
      while ( v67 != -8 )
      {
        if ( !v94 && v67 == -16 )
          v94 = v66;
        v65 = (v63 - 1) & (v93 + v65);
        v66 = (__int64 *)(v64 + 16LL * v65);
        v67 = *v66;
        if ( *v24 == *v66 )
          goto LABEL_56;
        ++v93;
      }
      v95 = *(_DWORD *)(v62 + 16);
      if ( v94 )
        v66 = v94;
      ++*(_QWORD *)v62;
      v96 = v95 + 1;
      if ( 4 * (v95 + 1) < 3 * v63 )
      {
        if ( v63 - *(_DWORD *)(v62 + 20) - v96 <= v63 >> 3 )
        {
          v143 = v62;
          sub_1DA35E0(v62, v63);
          v62 = v143;
          v98 = *(_DWORD *)(v143 + 24);
          if ( !v98 )
            goto LABEL_194;
          v99 = v98 - 1;
          v100 = *(_QWORD *)(v143 + 8);
          v101 = 1;
          v96 = *(_DWORD *)(v143 + 16) + 1;
          v102 = 0;
          v103 = (v98 - 1) & (((unsigned int)*v24 >> 9) ^ ((unsigned int)*v24 >> 4));
          v66 = (__int64 *)(v100 + 16LL * v103);
          v104 = *v66;
          if ( *v24 != *v66 )
          {
            while ( v104 != -8 )
            {
              if ( v104 == -16 && !v102 )
                v102 = v66;
              v103 = v99 & (v101 + v103);
              v66 = (__int64 *)(v100 + 16LL * v103);
              v104 = *v66;
              if ( *v24 == *v66 )
                goto LABEL_100;
              ++v101;
            }
            goto LABEL_116;
          }
        }
        goto LABEL_100;
      }
    }
    else
    {
      ++*(_QWORD *)v62;
    }
    v144 = v62;
    sub_1DA35E0(v62, 2 * v63);
    v62 = v144;
    v105 = *(_DWORD *)(v144 + 24);
    if ( !v105 )
    {
LABEL_194:
      ++*(_DWORD *)(v62 + 16);
      BUG();
    }
    v106 = v105 - 1;
    v107 = *(_QWORD *)(v144 + 8);
    v96 = *(_DWORD *)(v144 + 16) + 1;
    v108 = (v105 - 1) & (((unsigned int)*v24 >> 9) ^ ((unsigned int)*v24 >> 4));
    v66 = (__int64 *)(v107 + 16LL * v108);
    v109 = *v66;
    if ( *v66 != *v24 )
    {
      v110 = 1;
      v102 = 0;
      while ( v109 != -8 )
      {
        if ( !v102 && v109 == -16 )
          v102 = v66;
        v108 = v106 & (v110 + v108);
        v66 = (__int64 *)(v107 + 16LL * v108);
        v109 = *v66;
        if ( *v24 == *v66 )
          goto LABEL_100;
        ++v110;
      }
LABEL_116:
      if ( v102 )
        v66 = v102;
    }
LABEL_100:
    *(_DWORD *)(v62 + 16) = v96;
    if ( *v66 != -8 )
      --*(_DWORD *)(v62 + 20);
    v97 = *v24;
    *((_DWORD *)v66 + 2) = 0;
    *v66 = v97;
    goto LABEL_56;
  }
  return result;
}
