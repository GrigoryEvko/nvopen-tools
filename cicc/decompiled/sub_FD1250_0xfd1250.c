// Function: sub_FD1250
// Address: 0xfd1250
//
__int64 __fastcall sub_FD1250(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v5; // r12
  char v6; // r8
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rsi
  unsigned __int64 v10; // rax
  _QWORD *v11; // r8
  unsigned int i; // eax
  _QWORD *v13; // rdx
  __int64 v14; // rbx
  unsigned int v15; // r9d
  _QWORD *v16; // rdi
  __int64 v17; // rcx
  _QWORD *v18; // r13
  int v19; // eax
  __int64 v20; // rcx
  __int64 v21; // rsi
  unsigned int v22; // edx
  __int64 *v23; // rax
  __int64 v24; // r9
  __int64 v25; // r12
  __int64 v26; // r11
  __int64 v27; // rdx
  int v28; // r8d
  int v29; // r9d
  unsigned int v30; // ecx
  unsigned int v31; // r10d
  __int64 *v32; // rax
  __int64 v33; // rdi
  __int64 v34; // r13
  unsigned int v35; // eax
  unsigned __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rax
  int v39; // eax
  int v40; // eax
  __int64 v41; // rdx
  int v42; // eax
  unsigned int v43; // ecx
  __int64 v44; // rsi
  _DWORD *v45; // rax
  unsigned __int64 v46; // rdx
  __int64 v47; // rax
  _QWORD *v48; // rax
  __int64 v49; // rbx
  __int64 v50; // rdx
  __int64 v51; // r13
  int v52; // edi
  unsigned int v53; // edx
  __int64 *v54; // rax
  __int64 v55; // r10
  __int64 v56; // r8
  __int64 v57; // r11
  __int64 v58; // rdx
  int v59; // ecx
  unsigned int v60; // r9d
  __int64 *v61; // rax
  __int64 v62; // rdi
  unsigned int v63; // eax
  unsigned __int64 v64; // rdx
  __int64 v65; // rax
  __int64 v66; // rax
  int v67; // esi
  unsigned int v68; // esi
  __int64 v69; // rax
  _DWORD *v70; // rax
  __int64 v71; // r9
  unsigned __int64 v72; // rdx
  __int64 v73; // rax
  _QWORD *v74; // rax
  __int64 v75; // rax
  unsigned __int64 v76; // rdx
  __int64 v77; // rcx
  __int64 v78; // r12
  __int64 v79; // rsi
  __int64 v80; // rax
  unsigned int v81; // r11d
  int j; // r8d
  int v83; // eax
  int v84; // r10d
  __int64 v85; // rdi
  __int64 v86; // rdx
  int v87; // r10d
  __int64 *v88; // r9
  int v89; // eax
  int v90; // eax
  unsigned int v91; // edx
  __int64 v92; // rsi
  int v93; // r8d
  _QWORD *v94; // rdi
  int v95; // edi
  _QWORD *v96; // rsi
  unsigned int v97; // r12d
  __int64 v98; // rcx
  unsigned int v99; // ecx
  __int64 v100; // rsi
  unsigned int v101; // edx
  __int64 *v102; // rax
  __int64 v103; // r9
  __int64 v104; // r12
  _QWORD *v105; // rax
  int v106; // edi
  _QWORD *v107; // rax
  int v108; // r8d
  int v109; // edi
  int v110; // eax
  int v111; // r8d
  __int64 v112; // r10
  unsigned int v113; // edi
  int v114; // ecx
  __int64 *v115; // rdx
  int v116; // r9d
  int v117; // eax
  __int64 v118; // r10
  __int64 v119; // rax
  __int64 v120; // r10
  int v121; // eax
  int v122; // eax
  int v123; // eax
  __int64 v124; // rdi
  unsigned int v125; // edx
  int v126; // r9d
  __int64 v127; // rcx
  int v128; // eax
  __int64 v129; // rdi
  int v130; // r9d
  unsigned int v131; // edx
  int v132; // eax
  int v133; // r8d
  __int64 v134; // r10
  int v135; // ecx
  unsigned int v136; // edi
  int v137; // r9d
  __int64 *v138; // r8
  int v139; // eax
  int v140; // r8d
  __int64 v141; // [rsp+8h] [rbp-F8h]
  __int64 v142; // [rsp+8h] [rbp-F8h]
  __int64 v143; // [rsp+10h] [rbp-F0h]
  __int64 v144; // [rsp+10h] [rbp-F0h]
  int v145; // [rsp+18h] [rbp-E8h]
  int v146; // [rsp+18h] [rbp-E8h]
  int v147; // [rsp+18h] [rbp-E8h]
  int v148; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v149; // [rsp+20h] [rbp-E0h]
  int v150; // [rsp+30h] [rbp-D0h]
  __int64 v151; // [rsp+30h] [rbp-D0h]
  __int64 v152; // [rsp+30h] [rbp-D0h]
  __int64 v153; // [rsp+30h] [rbp-D0h]
  __int64 v154; // [rsp+30h] [rbp-D0h]
  __int64 v155; // [rsp+30h] [rbp-D0h]
  __int64 v156; // [rsp+30h] [rbp-D0h]
  __int64 v157; // [rsp+38h] [rbp-C8h]
  __int64 v158; // [rsp+40h] [rbp-C0h]
  int v159; // [rsp+48h] [rbp-B8h]
  unsigned int v160; // [rsp+4Ch] [rbp-B4h]
  __int64 v161; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v162; // [rsp+58h] [rbp-A8h] BYREF
  __int64 v163; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v164; // [rsp+68h] [rbp-98h]
  __int64 v165; // [rsp+70h] [rbp-90h]
  __int64 v166; // [rsp+78h] [rbp-88h]
  _QWORD *v167; // [rsp+80h] [rbp-80h] BYREF
  __int64 v168; // [rsp+88h] [rbp-78h]
  _QWORD v169[14]; // [rsp+90h] [rbp-70h] BYREF

  result = (unsigned int)a2 >> 9;
  v160 = result ^ ((unsigned int)a2 >> 4);
  v158 = *(_QWORD *)(a2 + 16);
  if ( !v158 )
    return result;
  while ( 2 )
  {
    v5 = *(_QWORD *)(v158 + 24);
    if ( *(_BYTE *)v5 <= 0x1Cu || !sub_FCD520(*(_QWORD *)(v158 + 24)) )
      goto LABEL_3;
    if ( v6 == 84 )
      v7 = *(_QWORD *)(*(_QWORD *)(v5 - 8)
                     + 32LL * *(unsigned int *)(v5 + 72)
                     + 8LL * (unsigned int)((v158 - *(_QWORD *)(v5 - 8)) >> 5));
    else
      v7 = *(_QWORD *)(v5 + 40);
    v8 = *(_QWORD *)a1;
    if ( *(_BYTE *)a2 <= 0x1Cu )
    {
      v85 = *(_QWORD *)(v8 + 80);
      v86 = v85 - 24;
      if ( !v85 )
        v86 = 0;
      v157 = v86;
    }
    else
    {
      v157 = *(_QWORD *)(a2 + 40);
    }
    v9 = *(_QWORD *)(v8 + 40) + 312LL;
    v10 = sub_FCD870(a2, v9);
    v159 = v10;
    v149 = HIDWORD(v10);
    if ( *(_BYTE *)v5 != 84 )
      goto LABEL_11;
    v99 = *(_DWORD *)(a1 + 136);
    v100 = *(_QWORD *)(a1 + 120);
    if ( v99 )
    {
      v101 = (v99 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v102 = (__int64 *)(v100 + 16LL * v101);
      v103 = *v102;
      if ( v7 == *v102 )
        goto LABEL_111;
      v139 = 1;
      while ( v103 != -4096 )
      {
        v140 = v139 + 1;
        v101 = (v99 - 1) & (v139 + v101);
        v102 = (__int64 *)(v100 + 16LL * v101);
        v103 = *v102;
        if ( v7 == *v102 )
          goto LABEL_111;
        v139 = v140;
      }
    }
    v102 = (__int64 *)(v100 + 16LL * v99);
LABEL_111:
    v104 = v102[1];
    v9 = a2;
    v167 = (_QWORD *)a2;
    if ( !(unsigned __int8)sub_FD0CF0(a1, a2, v104) )
    {
      v105 = sub_FCF1D0(a1 + 56, (__int64 *)&v167);
      v9 = *(_QWORD *)(v104 + 96);
      *(_QWORD *)(v9 + 8LL * (*(_DWORD *)v105 >> 6)) |= 1LL << *(_DWORD *)v105;
    }
LABEL_11:
    v11 = v169;
    v169[0] = v7;
    v163 = 0;
    v164 = 0;
    v165 = 0;
    v166 = 0;
    v167 = v169;
    v168 = 0x800000001LL;
    for ( i = 1; ; i = v168 )
    {
      v13 = &v11[i];
      if ( !i )
        break;
      while ( 1 )
      {
        v9 = (unsigned int)v166;
        --i;
        v14 = *(v13 - 1);
        LODWORD(v168) = i;
        if ( !(_DWORD)v166 )
        {
          ++v163;
          goto LABEL_96;
        }
        v15 = (v166 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
        v16 = (_QWORD *)(v164 + 8LL * v15);
        v17 = *v16;
        if ( v14 != *v16 )
          break;
LABEL_15:
        --v13;
        if ( !i )
          goto LABEL_16;
      }
      v150 = 1;
      v18 = 0;
      while ( v17 != -4096 )
      {
        if ( v17 != -8192 || v18 )
          v16 = v18;
        v15 = (v166 - 1) & (v150 + v15);
        v17 = *(_QWORD *)(v164 + 8LL * v15);
        if ( v14 == v17 )
          goto LABEL_15;
        ++v150;
        v18 = v16;
        v16 = (_QWORD *)(v164 + 8LL * v15);
      }
      if ( !v18 )
        v18 = v16;
      ++v163;
      v19 = v165 + 1;
      if ( 4 * ((int)v165 + 1) < (unsigned int)(3 * v166) )
      {
        if ( (int)v166 - (v19 + HIDWORD(v165)) > (unsigned int)v166 >> 3 )
          goto LABEL_25;
        sub_CF28B0((__int64)&v163, v166);
        if ( (_DWORD)v166 )
        {
          v95 = 1;
          v96 = 0;
          v97 = (v166 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
          v18 = (_QWORD *)(v164 + 8LL * v97);
          v98 = *v18;
          v19 = v165 + 1;
          if ( v14 != *v18 )
          {
            while ( v98 != -4096 )
            {
              if ( !v96 && v98 == -8192 )
                v96 = v18;
              v97 = (v166 - 1) & (v95 + v97);
              v18 = (_QWORD *)(v164 + 8LL * v97);
              v98 = *v18;
              if ( v14 == *v18 )
                goto LABEL_25;
              ++v95;
            }
            if ( v96 )
              v18 = v96;
          }
          goto LABEL_25;
        }
LABEL_213:
        LODWORD(v165) = v165 + 1;
        BUG();
      }
LABEL_96:
      sub_CF28B0((__int64)&v163, 2 * v166);
      if ( !(_DWORD)v166 )
        goto LABEL_213;
      v91 = (v166 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v18 = (_QWORD *)(v164 + 8LL * v91);
      v92 = *v18;
      v19 = v165 + 1;
      if ( v14 != *v18 )
      {
        v93 = 1;
        v94 = 0;
        while ( v92 != -4096 )
        {
          if ( v92 == -8192 && !v94 )
            v94 = v18;
          v91 = (v166 - 1) & (v93 + v91);
          v18 = (_QWORD *)(v164 + 8LL * v91);
          v92 = *v18;
          if ( v14 == *v18 )
            goto LABEL_25;
          ++v93;
        }
        if ( v94 )
          v18 = v94;
      }
LABEL_25:
      LODWORD(v165) = v19;
      if ( *v18 != -4096 )
        --HIDWORD(v165);
      *v18 = v14;
      v20 = *(unsigned int *)(a1 + 136);
      v21 = *(_QWORD *)(a1 + 120);
      if ( (_DWORD)v20 )
      {
        v22 = (v20 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
        v23 = (__int64 *)(v21 + 16LL * v22);
        v24 = *v23;
        if ( v14 == *v23 )
          goto LABEL_29;
        v39 = 1;
        while ( v24 != -4096 )
        {
          v108 = v39 + 1;
          v22 = (v20 - 1) & (v39 + v22);
          v23 = (__int64 *)(v21 + 16LL * v22);
          v24 = *v23;
          if ( v14 == *v23 )
            goto LABEL_29;
          v39 = v108;
        }
      }
      v23 = (__int64 *)(v21 + 16 * v20);
LABEL_29:
      v9 = *(unsigned int *)(a1 + 80);
      v25 = v23[1];
      v26 = a1 + 56;
      v27 = *(_QWORD *)(a1 + 64);
      if ( !(_DWORD)v9 )
        goto LABEL_33;
      v28 = v9 - 1;
      v29 = 1;
      v30 = (v9 - 1) & v160;
      v31 = v30;
      v32 = (__int64 *)(v27 + 16LL * v30);
      v33 = *v32;
      v34 = *v32;
      if ( *v32 == a2 )
      {
LABEL_31:
        v35 = *((_DWORD *)v32 + 2);
        v36 = v35 & 0x3F;
        v37 = 8LL * (v35 >> 6);
        goto LABEL_32;
      }
      while ( 1 )
      {
        if ( v34 == -4096 )
          goto LABEL_33;
        v31 = v28 & (v29 + v31);
        v34 = *(_QWORD *)(v27 + 16LL * v31);
        if ( a2 == v34 )
          break;
        ++v29;
      }
      v87 = 1;
      v88 = 0;
      while ( v33 != -4096 )
      {
        if ( v33 == -8192 && !v88 )
          v88 = v32;
        v30 = v28 & (v87 + v30);
        v32 = (__int64 *)(v27 + 16LL * v30);
        v33 = *v32;
        if ( v34 == *v32 )
          goto LABEL_31;
        ++v87;
      }
      if ( !v88 )
        v88 = v32;
      v89 = *(_DWORD *)(a1 + 72);
      ++*(_QWORD *)(a1 + 56);
      v90 = v89 + 1;
      if ( 4 * v90 >= (unsigned int)(3 * v9) )
      {
        sub_CE2410(a1 + 56, 2 * v9);
        v110 = *(_DWORD *)(a1 + 80);
        if ( !v110 )
          goto LABEL_212;
        v111 = v110 - 1;
        v112 = *(_QWORD *)(a1 + 64);
        v26 = a1 + 56;
        v113 = v111 & v160;
        v90 = *(_DWORD *)(a1 + 72) + 1;
        v88 = (__int64 *)(v112 + 16LL * (v111 & v160));
        v9 = *v88;
        if ( *v88 == v34 )
          goto LABEL_92;
        v114 = 1;
        v115 = 0;
        while ( v9 != -4096 )
        {
          if ( v9 == -8192 && !v115 )
            v115 = v88;
          v113 = v111 & (v114 + v113);
          v88 = (__int64 *)(v112 + 16LL * v113);
          v9 = *v88;
          if ( v34 == *v88 )
            goto LABEL_92;
          ++v114;
        }
      }
      else
      {
        if ( (int)v9 - *(_DWORD *)(a1 + 76) - v90 > (unsigned int)v9 >> 3 )
          goto LABEL_92;
        sub_CE2410(a1 + 56, v9);
        v132 = *(_DWORD *)(a1 + 80);
        if ( !v132 )
        {
LABEL_212:
          ++*(_DWORD *)(a1 + 72);
          BUG();
        }
        v133 = v132 - 1;
        v134 = *(_QWORD *)(a1 + 64);
        v115 = 0;
        v26 = a1 + 56;
        v135 = 1;
        v136 = v133 & v160;
        v90 = *(_DWORD *)(a1 + 72) + 1;
        v88 = (__int64 *)(v134 + 16LL * (v133 & v160));
        v9 = *v88;
        if ( *v88 == v34 )
          goto LABEL_92;
        while ( v9 != -4096 )
        {
          if ( v9 == -8192 && !v115 )
            v115 = v88;
          v136 = v133 & (v135 + v136);
          v88 = (__int64 *)(v134 + 16LL * v136);
          v9 = *v88;
          if ( v34 == *v88 )
            goto LABEL_92;
          ++v135;
        }
      }
      if ( v115 )
        v88 = v115;
LABEL_92:
      *(_DWORD *)(a1 + 72) = v90;
      if ( *v88 != -4096 )
        --*(_DWORD *)(a1 + 76);
      *v88 = v34;
      v36 = 0;
      v37 = 0;
      *((_DWORD *)v88 + 2) = 0;
LABEL_32:
      v38 = *(_QWORD *)(*(_QWORD *)(v25 + 24) + v37);
      if ( _bittest64(&v38, v36) )
        goto LABEL_35;
LABEL_33:
      if ( *(_BYTE *)a2 == 84 )
      {
        if ( v157 == v14 )
        {
          v9 = a2;
          v156 = v26;
          v162 = a2;
          if ( !(unsigned __int8)sub_FD0FA0(a1, a2, v25) )
          {
            v107 = sub_FCF1D0(v156, &v162);
            v9 = *(_QWORD *)(v25 + 24);
            *(_QWORD *)(v9 + 8LL * (*(_DWORD *)v107 >> 6)) |= 1LL << *(_DWORD *)v107;
            *(_DWORD *)(v25 + 12) += v149;
            *(_DWORD *)(v25 + 8) += v159;
            *(_DWORD *)v25 += v159;
            *(_DWORD *)(v25 + 4) += v149;
          }
          goto LABEL_35;
        }
      }
      else if ( v157 == v14 )
      {
        goto LABEL_35;
      }
      v40 = *(_DWORD *)(a1 + 80);
      v41 = *(_QWORD *)(a1 + 64);
      v161 = a2;
      v162 = a2;
      if ( v40 )
      {
        v42 = v40 - 1;
        v43 = v42 & v160;
        v44 = *(_QWORD *)(v41 + 16LL * (v42 & v160));
        if ( a2 == v44 )
        {
LABEL_42:
          v9 = (__int64)&v162;
          v151 = v26;
          v45 = sub_FCF1D0(v26, &v162);
          v26 = v151;
          v46 = (unsigned int)*v45;
          v47 = *(_QWORD *)(*(_QWORD *)(v25 + 24) + 8LL * (*v45 >> 6));
          if ( _bittest64(&v47, v46) )
            goto LABEL_44;
        }
        else
        {
          v109 = 1;
          while ( v44 != -4096 )
          {
            v43 = v42 & (v109 + v43);
            v44 = *(_QWORD *)(v41 + 16LL * v43);
            if ( a2 == v44 )
              goto LABEL_42;
            ++v109;
          }
        }
      }
      v152 = v26;
      v48 = sub_FCF1D0(v26, &v161);
      v9 = *(_QWORD *)(v25 + 24);
      v26 = v152;
      *(_QWORD *)(v9 + 8LL * (*(_DWORD *)v48 >> 6)) |= 1LL << *(_DWORD *)v48;
      *(_DWORD *)(v25 + 12) += v149;
      *(_DWORD *)(v25 + 8) += v159;
      *(_DWORD *)v25 += v159;
      *(_DWORD *)(v25 + 4) += v149;
LABEL_44:
      v49 = *(_QWORD *)(v14 + 16);
      if ( v49 )
      {
        while ( 1 )
        {
          v50 = *(_QWORD *)(v49 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v50 - 30) <= 0xAu )
            break;
          v49 = *(_QWORD *)(v49 + 8);
          if ( !v49 )
            goto LABEL_35;
        }
        v51 = v26;
LABEL_64:
        v77 = *(unsigned int *)(a1 + 136);
        v78 = *(_QWORD *)(v50 + 40);
        v79 = *(_QWORD *)(a1 + 120);
        if ( (_DWORD)v77 )
        {
          v52 = v77 - 1;
          v53 = (v77 - 1) & (((unsigned int)v78 >> 9) ^ ((unsigned int)v78 >> 4));
          v54 = (__int64 *)(v79 + 16LL * v53);
          v55 = *v54;
          if ( v78 == *v54 )
          {
            v56 = v54[1];
LABEL_49:
            v57 = v54[1];
          }
          else
          {
            v80 = *v54;
            v81 = (v77 - 1) & (((unsigned int)v78 >> 9) ^ ((unsigned int)v78 >> 4));
            for ( j = 1; ; j = v137 )
            {
              if ( v80 == -4096 )
              {
                v56 = *(_QWORD *)(v79 + 16LL * (unsigned int)v77 + 8);
                goto LABEL_72;
              }
              v137 = j + 1;
              v81 = v52 & (j + v81);
              v138 = (__int64 *)(v79 + 16LL * v81);
              v80 = *v138;
              if ( v78 == *v138 )
                break;
            }
            v56 = v138[1];
LABEL_72:
            v83 = 1;
            while ( v55 != -4096 )
            {
              v116 = v83 + 1;
              v53 = v52 & (v83 + v53);
              v54 = (__int64 *)(v79 + 16LL * v53);
              v55 = *v54;
              if ( v78 == *v54 )
                goto LABEL_49;
              v83 = v116;
            }
            v57 = *(_QWORD *)(v79 + 16 * v77 + 8);
          }
          v9 = *(unsigned int *)(a1 + 80);
          v58 = *(_QWORD *)(a1 + 64);
          if ( !(_DWORD)v9 )
          {
LABEL_66:
            v161 = a2;
            goto LABEL_58;
          }
        }
        else
        {
          v56 = *(_QWORD *)(v79 + 8);
          v9 = *(unsigned int *)(a1 + 80);
          v58 = *(_QWORD *)(a1 + 64);
          v57 = v56;
          if ( !(_DWORD)v9 )
            goto LABEL_66;
        }
        v59 = v9 - 1;
        v60 = (v9 - 1) & v160;
        v61 = (__int64 *)(v58 + 16LL * v60);
        v62 = *v61;
        if ( a2 == *v61 )
          goto LABEL_52;
        v155 = *v61;
        v84 = 1;
        v145 = (v9 - 1) & v160;
        while ( v155 != -4096 )
        {
          v117 = v84 + 1;
          v118 = v59 & (unsigned int)(v145 + v84);
          v145 = v118;
          v155 = *(_QWORD *)(v58 + 16 * v118);
          if ( a2 == v155 )
          {
            v146 = 1;
            v119 = v58 + 16LL * (v59 & v160);
            v120 = 0;
            while ( v62 != -4096 )
            {
              if ( v120 || v62 != -8192 )
                v119 = v120;
              v60 = v59 & (v146 + v60);
              v62 = *(_QWORD *)(v58 + 16LL * v60);
              if ( v155 == v62 )
              {
                v61 = (__int64 *)(v58 + 16LL * v60);
LABEL_52:
                v63 = *((_DWORD *)v61 + 2);
                v64 = v63 & 0x3F;
                v65 = 8LL * (v63 >> 6);
LABEL_53:
                v66 = *(_QWORD *)(*(_QWORD *)(v57 + 96) + v65);
                if ( _bittest64(&v66, v64) )
                  goto LABEL_62;
                v67 = *(_DWORD *)(a1 + 80);
                v58 = *(_QWORD *)(a1 + 64);
                v161 = a2;
                v162 = a2;
                if ( v67 )
                {
                  v59 = v67 - 1;
                  goto LABEL_56;
                }
LABEL_58:
                v154 = v56;
                v74 = sub_FCF1D0(v51, &v161);
                v56 = v154;
                v9 = *(_QWORD *)(v154 + 96);
                *(_QWORD *)(v9 + 8LL * (*(_DWORD *)v74 >> 6)) |= 1LL << *(_DWORD *)v74;
                goto LABEL_59;
              }
              ++v146;
              v120 = v119;
              v119 = v58 + 16LL * v60;
            }
            if ( !v120 )
              v120 = v119;
            v121 = *(_DWORD *)(a1 + 72);
            ++*(_QWORD *)(a1 + 56);
            v122 = v121 + 1;
            if ( 4 * v122 >= (unsigned int)(3 * v9) )
            {
              v141 = v56;
              v143 = v57;
              sub_CE2410(v51, 2 * v9);
              v123 = *(_DWORD *)(a1 + 80);
              if ( v123 )
              {
                v124 = *(_QWORD *)(a1 + 64);
                v147 = v123 - 1;
                v57 = v143;
                v125 = (v123 - 1) & v160;
                v56 = v141;
                v122 = *(_DWORD *)(a1 + 72) + 1;
                v120 = v124 + 16LL * v125;
                v9 = *(_QWORD *)v120;
                if ( *(_QWORD *)v120 == v155 )
                  goto LABEL_142;
                v126 = 1;
                v127 = 0;
                while ( v9 != -4096 )
                {
                  if ( v9 == -8192 && !v127 )
                    v127 = v120;
                  v125 = v147 & (v126 + v125);
                  v120 = v124 + 16LL * v125;
                  v9 = *(_QWORD *)v120;
                  if ( v155 == *(_QWORD *)v120 )
                    goto LABEL_142;
                  ++v126;
                }
LABEL_157:
                if ( v127 )
                  v120 = v127;
                goto LABEL_142;
              }
            }
            else
            {
              if ( (int)v9 - *(_DWORD *)(a1 + 76) - v122 > (unsigned int)v9 >> 3 )
              {
LABEL_142:
                *(_DWORD *)(a1 + 72) = v122;
                if ( *(_QWORD *)v120 != -4096 )
                  --*(_DWORD *)(a1 + 76);
                *(_DWORD *)(v120 + 8) = 0;
                v64 = 0;
                *(_QWORD *)v120 = v155;
                v65 = 0;
                goto LABEL_53;
              }
              v142 = v56;
              v144 = v57;
              sub_CE2410(v51, v9);
              v128 = *(_DWORD *)(a1 + 80);
              if ( v128 )
              {
                v129 = *(_QWORD *)(a1 + 64);
                v127 = 0;
                v148 = v128 - 1;
                v57 = v144;
                v130 = 1;
                v131 = (v128 - 1) & v160;
                v56 = v142;
                v122 = *(_DWORD *)(a1 + 72) + 1;
                v120 = v129 + 16LL * v131;
                v9 = *(_QWORD *)v120;
                if ( *(_QWORD *)v120 == v155 )
                  goto LABEL_142;
                while ( v9 != -4096 )
                {
                  if ( !v127 && v9 == -8192 )
                    v127 = v120;
                  v131 = v148 & (v130 + v131);
                  v120 = v129 + 16LL * v131;
                  v9 = *(_QWORD *)v120;
                  if ( v155 == *(_QWORD *)v120 )
                    goto LABEL_142;
                  ++v130;
                }
                goto LABEL_157;
              }
            }
            ++*(_DWORD *)(a1 + 72);
            BUG();
          }
          v84 = v117;
        }
        v161 = a2;
        v162 = a2;
LABEL_56:
        v68 = v59 & v160;
        v69 = *(_QWORD *)(v58 + 16LL * (v59 & v160));
        if ( a2 != v69 )
        {
          v106 = 1;
          while ( v69 != -4096 )
          {
            v68 = v59 & (v106 + v68);
            v69 = *(_QWORD *)(v58 + 16LL * v68);
            if ( a2 == v69 )
              goto LABEL_57;
            ++v106;
          }
          goto LABEL_58;
        }
LABEL_57:
        v9 = (__int64)&v162;
        v153 = v56;
        v70 = sub_FCF1D0(v51, &v162);
        v56 = v153;
        v72 = (unsigned int)*v70;
        v73 = *(_QWORD *)(*(_QWORD *)(v153 + 96) + 8LL * (*v70 >> 6));
        if ( !_bittest64(&v73, v72) )
          goto LABEL_58;
LABEL_59:
        v75 = (unsigned int)v168;
        v76 = (unsigned int)v168 + 1LL;
        if ( v76 > HIDWORD(v168) )
        {
          v9 = (__int64)v169;
          sub_C8D5F0((__int64)&v167, v169, v76, 8u, v56, v71);
          v75 = (unsigned int)v168;
        }
        v167[v75] = v78;
        LODWORD(v168) = v168 + 1;
LABEL_62:
        while ( 1 )
        {
          v49 = *(_QWORD *)(v49 + 8);
          if ( !v49 )
            break;
          v50 = *(_QWORD *)(v49 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v50 - 30) <= 0xAu )
            goto LABEL_64;
        }
      }
LABEL_35:
      v11 = v167;
    }
LABEL_16:
    if ( v11 != v169 )
      _libc_free(v11, v9);
    sub_C7D6A0(v164, 8LL * (unsigned int)v166, 8);
LABEL_3:
    result = *(_QWORD *)(v158 + 8);
    v158 = result;
    if ( result )
      continue;
    return result;
  }
}
