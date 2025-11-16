// Function: sub_1C2B230
// Address: 0x1c2b230
//
void __fastcall sub_1C2B230(__int64 a1, __int64 a2)
{
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // r12
  __int64 v7; // r15
  unsigned __int64 v8; // r14
  void *v9; // r8
  __int64 v10; // rax
  unsigned int v11; // esi
  __int64 v12; // r9
  unsigned int v13; // r8d
  __int64 *v14; // rax
  __int64 v15; // rdi
  __int64 v16; // r12
  unsigned int v17; // esi
  __int64 v18; // r8
  unsigned int v19; // edi
  __int64 *v20; // rax
  __int64 v21; // rcx
  __int64 v22; // r15
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rax
  unsigned int v25; // r14d
  __int64 v26; // r15
  unsigned int v27; // esi
  __int64 *v28; // rcx
  __int64 v29; // rax
  __int64 *v30; // rax
  __int64 v31; // r9
  int v32; // edi
  __int64 v33; // r11
  int v34; // r10d
  __int64 *v35; // rcx
  unsigned int v36; // r9d
  unsigned int v37; // edx
  unsigned int v38; // esi
  __int64 *v39; // rcx
  __int64 v40; // r10
  __int64 v41; // rcx
  unsigned int v42; // edx
  __int64 *v43; // rax
  __int64 v44; // rsi
  unsigned int v45; // esi
  __int64 *v46; // r9
  __int64 v47; // r8
  __int64 v48; // rdx
  unsigned int v49; // edi
  __int64 *v50; // rax
  __int64 v51; // rcx
  __int64 v52; // rsi
  __int64 v53; // r12
  __int64 v54; // rax
  unsigned int v55; // edx
  __int64 v56; // r8
  unsigned __int64 v57; // rsi
  unsigned __int64 v58; // rdx
  unsigned int v59; // r12d
  __int64 v60; // r14
  __int64 v61; // r12
  _QWORD *v62; // r13
  _QWORD *i; // r12
  _QWORD *v64; // rsi
  __int64 v65; // rax
  size_t v66; // rdx
  void *v67; // r12
  __int64 v68; // rax
  void *v69; // rdi
  int v70; // ecx
  unsigned int v71; // esi
  int v72; // ecx
  int v73; // ecx
  int v74; // r9d
  int v75; // eax
  int v76; // edi
  int v77; // ecx
  int v78; // ecx
  __int64 v79; // rdx
  int v80; // r10d
  int v81; // eax
  int v82; // eax
  int v83; // r9d
  __int64 v84; // rdi
  __int64 v85; // rsi
  int v86; // r11d
  __int64 *v87; // r10
  int v88; // r8d
  __int64 v89; // rsi
  int v90; // r10d
  unsigned int v91; // r11d
  int v92; // r10d
  __int64 *v93; // rdx
  int v94; // eax
  int v95; // ecx
  int v96; // r11d
  __int64 *v97; // rdx
  int v98; // eax
  int v99; // edi
  int v100; // eax
  int v101; // esi
  __int64 v102; // r8
  unsigned int v103; // eax
  __int64 v104; // rdi
  int v105; // r10d
  __int64 *v106; // r9
  int v107; // eax
  int v108; // esi
  __int64 v109; // r8
  unsigned int v110; // eax
  __int64 v111; // r9
  int v112; // r11d
  __int64 *v113; // r10
  int v114; // eax
  int v115; // eax
  __int64 v116; // rdi
  __int64 *v117; // r8
  unsigned int v118; // r12d
  int v119; // r9d
  __int64 v120; // rsi
  int v121; // eax
  int v122; // eax
  __int64 v123; // r8
  __int64 *v124; // r9
  unsigned int v125; // r12d
  int v126; // r10d
  __int64 v127; // rsi
  __int64 *v128; // r10
  __int64 *v129; // r11
  __int64 v130; // rax
  __int64 v131; // rax
  __int64 v132; // [rsp+10h] [rbp-60h]
  int v133; // [rsp+1Ch] [rbp-54h]
  __int64 v134; // [rsp+20h] [rbp-50h]
  unsigned int v135; // [rsp+20h] [rbp-50h]
  int v136; // [rsp+20h] [rbp-50h]
  void *v137; // [rsp+28h] [rbp-48h]
  unsigned int v138; // [rsp+30h] [rbp-40h]
  __int64 v139; // [rsp+30h] [rbp-40h]
  int n; // [rsp+38h] [rbp-38h]
  size_t na; // [rsp+38h] [rbp-38h]

  v4 = *(_QWORD *)(a1 + 96) - *(_QWORD *)(a1 + 88);
  v5 = sub_22077B0(72);
  v6 = v4 >> 3;
  v7 = v5;
  if ( v5 )
  {
    *(_QWORD *)v5 = 0;
    *(_QWORD *)(v5 + 8) = 0;
    *(_QWORD *)(v5 + 16) = 0;
    *(_QWORD *)(v5 + 24) = 0;
    *(_QWORD *)(v5 + 32) = 0;
    *(_DWORD *)(v5 + 40) = v6;
    v138 = (unsigned int)(v6 + 63) >> 6;
    v8 = 8LL * v138;
    v9 = (void *)malloc(v8);
    if ( !v9 )
    {
      if ( v8 || (v130 = malloc(1u), v9 = 0, !v130) )
      {
        v137 = v9;
        sub_16BD1C0("Allocation failed", 1u);
        v9 = v137;
      }
      else
      {
        v9 = (void *)v130;
      }
    }
    *(_QWORD *)(v7 + 24) = v9;
    *(_QWORD *)(v7 + 32) = v138;
    if ( v138 )
    {
      memset(v9, 0, v8);
      *(_QWORD *)(v7 + 48) = 0;
      *(_QWORD *)(v7 + 56) = 0;
      *(_DWORD *)(v7 + 64) = v6;
      v68 = malloc(v8);
      v69 = (void *)v68;
      if ( v68 )
      {
        *(_QWORD *)(v7 + 48) = v68;
        *(_QWORD *)(v7 + 56) = v138;
LABEL_55:
        memset(v69, 0, v8);
        goto LABEL_6;
      }
    }
    else
    {
      *(_QWORD *)(v7 + 48) = 0;
      *(_QWORD *)(v7 + 56) = 0;
      *(_DWORD *)(v7 + 64) = v6;
      v10 = malloc(v8);
      if ( v10 )
      {
        *(_QWORD *)(v7 + 48) = v10;
        *(_QWORD *)(v7 + 56) = 0;
        goto LABEL_6;
      }
    }
    if ( v8 || (v69 = (void *)malloc(1u)) == 0 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v69 = 0;
    }
    *(_QWORD *)(v7 + 48) = v69;
    *(_QWORD *)(v7 + 56) = v138;
    if ( v138 )
      goto LABEL_55;
  }
LABEL_6:
  v11 = *(_DWORD *)(a1 + 136);
  v132 = a1 + 112;
  if ( !v11 )
  {
    ++*(_QWORD *)(a1 + 112);
LABEL_128:
    sub_1C29D90(v132, 2 * v11);
    v107 = *(_DWORD *)(a1 + 136);
    if ( !v107 )
      goto LABEL_210;
    v108 = v107 - 1;
    v109 = *(_QWORD *)(a1 + 120);
    v110 = (v107 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v99 = *(_DWORD *)(a1 + 128) + 1;
    v97 = (__int64 *)(v109 + 16LL * v110);
    v111 = *v97;
    if ( *v97 != a2 )
    {
      v112 = 1;
      v113 = 0;
      while ( v111 != -8 )
      {
        if ( !v113 && v111 == -16 )
          v113 = v97;
        v110 = v108 & (v112 + v110);
        v97 = (__int64 *)(v109 + 16LL * v110);
        v111 = *v97;
        if ( *v97 == a2 )
          goto LABEL_114;
        ++v112;
      }
      if ( v113 )
        v97 = v113;
    }
    goto LABEL_114;
  }
  v12 = *(_QWORD *)(a1 + 120);
  v13 = (v11 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v14 = (__int64 *)(v12 + 16LL * v13);
  v15 = *v14;
  if ( *v14 == a2 )
  {
LABEL_8:
    v16 = v14[1];
    v14[1] = v7;
    if ( v16 )
    {
      _libc_free(*(_QWORD *)(v16 + 48));
      _libc_free(*(_QWORD *)(v16 + 24));
      j_j___libc_free_0(v16, 72);
    }
    v17 = *(_DWORD *)(a1 + 136);
    if ( v17 )
      goto LABEL_11;
LABEL_117:
    ++*(_QWORD *)(a1 + 112);
    goto LABEL_118;
  }
  v96 = 1;
  v97 = 0;
  while ( v15 != -8 )
  {
    if ( v15 == -16 && !v97 )
      v97 = v14;
    v13 = (v11 - 1) & (v96 + v13);
    v14 = (__int64 *)(v12 + 16LL * v13);
    v15 = *v14;
    if ( *v14 == a2 )
      goto LABEL_8;
    ++v96;
  }
  if ( !v97 )
    v97 = v14;
  v98 = *(_DWORD *)(a1 + 128);
  ++*(_QWORD *)(a1 + 112);
  v99 = v98 + 1;
  if ( 4 * (v98 + 1) >= 3 * v11 )
    goto LABEL_128;
  if ( v11 - *(_DWORD *)(a1 + 132) - v99 <= v11 >> 3 )
  {
    sub_1C29D90(v132, v11);
    v121 = *(_DWORD *)(a1 + 136);
    if ( !v121 )
      goto LABEL_210;
    v122 = v121 - 1;
    v123 = *(_QWORD *)(a1 + 120);
    v124 = 0;
    v125 = v122 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v126 = 1;
    v99 = *(_DWORD *)(a1 + 128) + 1;
    v97 = (__int64 *)(v123 + 16LL * v125);
    v127 = *v97;
    if ( *v97 != a2 )
    {
      while ( v127 != -8 )
      {
        if ( !v124 && v127 == -16 )
          v124 = v97;
        v125 = v122 & (v126 + v125);
        v97 = (__int64 *)(v123 + 16LL * v125);
        v127 = *v97;
        if ( *v97 == a2 )
          goto LABEL_114;
        ++v126;
      }
      if ( v124 )
        v97 = v124;
    }
  }
LABEL_114:
  *(_DWORD *)(a1 + 128) = v99;
  if ( *v97 != -8 )
    --*(_DWORD *)(a1 + 132);
  *v97 = a2;
  v97[1] = v7;
  v17 = *(_DWORD *)(a1 + 136);
  if ( !v17 )
    goto LABEL_117;
LABEL_11:
  v18 = *(_QWORD *)(a1 + 120);
  v19 = (v17 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v20 = (__int64 *)(v18 + 16LL * v19);
  v21 = *v20;
  if ( *v20 == a2 )
  {
    v22 = v20[1];
    goto LABEL_13;
  }
  v92 = 1;
  v93 = 0;
  while ( v21 != -8 )
  {
    if ( v21 != -16 || v93 )
      v20 = v93;
    v19 = (v17 - 1) & (v92 + v19);
    v129 = (__int64 *)(v18 + 16LL * v19);
    v21 = *v129;
    if ( *v129 == a2 )
    {
      v22 = v129[1];
      goto LABEL_13;
    }
    ++v92;
    v93 = v20;
    v20 = (__int64 *)(v18 + 16LL * v19);
  }
  if ( !v93 )
    v93 = v20;
  v94 = *(_DWORD *)(a1 + 128);
  ++*(_QWORD *)(a1 + 112);
  v95 = v94 + 1;
  if ( 4 * (v94 + 1) >= 3 * v17 )
  {
LABEL_118:
    sub_1C29D90(v132, 2 * v17);
    v100 = *(_DWORD *)(a1 + 136);
    if ( v100 )
    {
      v101 = v100 - 1;
      v102 = *(_QWORD *)(a1 + 120);
      v103 = (v100 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v95 = *(_DWORD *)(a1 + 128) + 1;
      v93 = (__int64 *)(v102 + 16LL * v103);
      v104 = *v93;
      if ( *v93 != a2 )
      {
        v105 = 1;
        v106 = 0;
        while ( v104 != -8 )
        {
          if ( !v106 && v104 == -16 )
            v106 = v93;
          v103 = v101 & (v105 + v103);
          v93 = (__int64 *)(v102 + 16LL * v103);
          v104 = *v93;
          if ( *v93 == a2 )
            goto LABEL_105;
          ++v105;
        }
        if ( v106 )
          v93 = v106;
      }
      goto LABEL_105;
    }
    goto LABEL_210;
  }
  if ( v17 - *(_DWORD *)(a1 + 132) - v95 <= v17 >> 3 )
  {
    sub_1C29D90(v132, v17);
    v114 = *(_DWORD *)(a1 + 136);
    if ( v114 )
    {
      v115 = v114 - 1;
      v116 = *(_QWORD *)(a1 + 120);
      v117 = 0;
      v118 = v115 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v119 = 1;
      v95 = *(_DWORD *)(a1 + 128) + 1;
      v93 = (__int64 *)(v116 + 16LL * v118);
      v120 = *v93;
      if ( *v93 != a2 )
      {
        while ( v120 != -8 )
        {
          if ( !v117 && v120 == -16 )
            v117 = v93;
          v118 = v115 & (v119 + v118);
          v93 = (__int64 *)(v116 + 16LL * v118);
          v120 = *v93;
          if ( *v93 == a2 )
            goto LABEL_105;
          ++v119;
        }
        if ( v117 )
          v93 = v117;
      }
      goto LABEL_105;
    }
LABEL_210:
    ++*(_DWORD *)(a1 + 128);
    BUG();
  }
LABEL_105:
  *(_DWORD *)(a1 + 128) = v95;
  if ( *v93 != -8 )
    --*(_DWORD *)(a1 + 132);
  *v93 = a2;
  v22 = 0;
  v93[1] = 0;
LABEL_13:
  v23 = sub_157EBA0(a2);
  if ( v23 )
  {
    n = sub_15F4D60(v23);
    v24 = sub_157EBA0(a2);
    if ( n )
    {
      v139 = v22;
      v25 = 0;
      v26 = v24;
      while ( 1 )
      {
        v53 = sub_15F4DF0(v26, v25);
        sub_3953E20(a1, a2, v53);
        if ( v53 == a2 )
          goto LABEL_34;
        v54 = *(_QWORD *)(a1 + 16);
        v55 = *(_DWORD *)(v54 + 48);
        v56 = *(_QWORD *)(v54 + 32);
        if ( !v55 )
          goto LABEL_37;
        v27 = (v55 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v28 = (__int64 *)(v56 + 16LL * v27);
        v29 = *v28;
        if ( *v28 != a2 )
          break;
LABEL_17:
        v30 = (__int64 *)(v56 + 16LL * v55);
        if ( v30 != v28 )
        {
          v31 = v28[1];
          goto LABEL_19;
        }
LABEL_38:
        v31 = 0;
LABEL_19:
        v32 = *(_DWORD *)(a1 + 168);
        v33 = *(_QWORD *)(a1 + 152);
        if ( v32 )
        {
          v34 = v32 - 1;
          v35 = (__int64 *)(v33 + 16LL * ((v32 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4))));
          v134 = *v35;
          if ( *v35 == v31 )
          {
LABEL_21:
            v36 = *((_DWORD *)v35 + 2);
          }
          else
          {
            v71 = (v32 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
            v72 = 1;
            while ( v134 != -8 )
            {
              v71 = v34 & (v72 + v71);
              v133 = v72 + 1;
              v35 = (__int64 *)(v33 + 16LL * v71);
              v134 = *v35;
              if ( v31 == *v35 )
                goto LABEL_21;
              v72 = v133;
            }
            v36 = 0;
          }
          if ( !v55 )
          {
            v41 = 0;
            goto LABEL_28;
          }
        }
        else
        {
          v36 = 0;
          if ( !v55 )
            goto LABEL_34;
        }
        v37 = v55 - 1;
        v38 = v37 & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
        v39 = (__int64 *)(v56 + 16LL * v38);
        v40 = *v39;
        if ( v53 != *v39 )
        {
          v70 = 1;
          while ( v40 != -8 )
          {
            v38 = v37 & (v70 + v38);
            v136 = v70 + 1;
            v39 = (__int64 *)(v56 + 16LL * v38);
            v40 = *v39;
            if ( v53 == *v39 )
              goto LABEL_24;
            v70 = v136;
          }
LABEL_58:
          v41 = 0;
          goto LABEL_26;
        }
LABEL_24:
        if ( v39 == v30 )
          goto LABEL_58;
        v41 = v39[1];
LABEL_26:
        if ( !v32 )
          goto LABEL_34;
        v34 = v32 - 1;
LABEL_28:
        v42 = v34 & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
        v43 = (__int64 *)(v33 + 16LL * v42);
        v44 = *v43;
        if ( *v43 != v41 )
        {
          v75 = 1;
          while ( v44 != -8 )
          {
            v76 = v75 + 1;
            v42 = v34 & (v75 + v42);
            v43 = (__int64 *)(v33 + 16LL * v42);
            v44 = *v43;
            if ( *v43 == v41 )
              goto LABEL_29;
            v75 = v76;
          }
          goto LABEL_34;
        }
LABEL_29:
        if ( *((_DWORD *)v43 + 2) > v36 )
        {
          v45 = *(_DWORD *)(a1 + 136);
          if ( v45 )
          {
            LODWORD(v46) = v45 - 1;
            v47 = *(_QWORD *)(a1 + 120);
            v48 = ((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4);
            v49 = (v45 - 1) & v48;
            v50 = (__int64 *)(v47 + 16LL * v49);
            v51 = *v50;
            if ( v53 == *v50 )
            {
              v52 = v50[1];
              goto LABEL_33;
            }
            v80 = 1;
            v48 = 0;
            while ( v51 != -8 )
            {
              if ( v51 != -16 || v48 )
                v50 = (__int64 *)v48;
              v48 = (unsigned int)(v80 + 1);
              v49 = (unsigned int)v46 & (v80 + v49);
              v128 = (__int64 *)(v47 + 16LL * v49);
              v51 = *v128;
              if ( v53 == *v128 )
              {
                v52 = v128[1];
                goto LABEL_33;
              }
              v80 = v48;
              v48 = (__int64)v50;
              v50 = (__int64 *)(v47 + 16LL * v49);
            }
            if ( !v48 )
              v48 = (__int64)v50;
            v81 = *(_DWORD *)(a1 + 128);
            ++*(_QWORD *)(a1 + 112);
            v82 = v81 + 1;
            if ( 4 * v82 < 3 * v45 )
            {
              v51 = v45 - *(_DWORD *)(a1 + 132) - v82;
              if ( (unsigned int)v51 <= v45 >> 3 )
              {
                v135 = ((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4);
                sub_1C29D90(v132, v45);
                v88 = *(_DWORD *)(a1 + 136);
                if ( !v88 )
                  goto LABEL_210;
                LODWORD(v47) = v88 - 1;
                v89 = *(_QWORD *)(a1 + 120);
                v46 = 0;
                v90 = 1;
                v91 = v47 & v135;
                v82 = *(_DWORD *)(a1 + 128) + 1;
                v48 = v89 + 16LL * ((unsigned int)v47 & v135);
                v51 = *(_QWORD *)v48;
                if ( v53 != *(_QWORD *)v48 )
                {
                  while ( v51 != -8 )
                  {
                    if ( !v46 && v51 == -16 )
                      v46 = (__int64 *)v48;
                    v91 = v47 & (v90 + v91);
                    v48 = v89 + 16LL * v91;
                    v51 = *(_QWORD *)v48;
                    if ( v53 == *(_QWORD *)v48 )
                      goto LABEL_82;
                    ++v90;
                  }
                  if ( v46 )
                    v48 = (__int64)v46;
                }
              }
              goto LABEL_82;
            }
          }
          else
          {
            ++*(_QWORD *)(a1 + 112);
          }
          sub_1C29D90(v132, 2 * v45);
          v83 = *(_DWORD *)(a1 + 136);
          if ( !v83 )
            goto LABEL_210;
          LODWORD(v46) = v83 - 1;
          v84 = *(_QWORD *)(a1 + 120);
          v51 = (unsigned int)v46 & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
          v82 = *(_DWORD *)(a1 + 128) + 1;
          v48 = v84 + 16 * v51;
          v85 = *(_QWORD *)v48;
          if ( v53 != *(_QWORD *)v48 )
          {
            v86 = 1;
            v87 = 0;
            while ( v85 != -8 )
            {
              if ( v85 == -16 && !v87 )
                v87 = (__int64 *)v48;
              LODWORD(v47) = v86 + 1;
              v51 = (unsigned int)v46 & (v86 + (_DWORD)v51);
              v48 = v84 + 16LL * (unsigned int)v51;
              v85 = *(_QWORD *)v48;
              if ( v53 == *(_QWORD *)v48 )
                goto LABEL_82;
              ++v86;
            }
            if ( v87 )
              v48 = (__int64)v87;
          }
LABEL_82:
          *(_DWORD *)(a1 + 128) = v82;
          if ( *(_QWORD *)v48 != -8 )
            --*(_DWORD *)(a1 + 132);
          *(_QWORD *)v48 = v53;
          v52 = 0;
          *(_QWORD *)(v48 + 8) = 0;
LABEL_33:
          sub_1C28EA0(v139 + 48, v52 + 24, v48, v51, v47, (int)v46);
          sub_3953990(a1, a2, v53);
        }
LABEL_34:
        if ( n == ++v25 )
        {
          v22 = v139;
          goto LABEL_42;
        }
      }
      v73 = 1;
      while ( v29 != -8 )
      {
        v74 = v73 + 1;
        v27 = (v55 - 1) & (v73 + v27);
        v28 = (__int64 *)(v56 + 16LL * v27);
        v29 = *v28;
        if ( *v28 == a2 )
          goto LABEL_17;
        v73 = v74;
      }
LABEL_37:
      v30 = (__int64 *)(v56 + 16LL * v55);
      goto LABEL_38;
    }
  }
LABEL_42:
  v57 = *(unsigned int *)(v22 + 64);
  v58 = *(_QWORD *)(v22 + 32);
  *(_DWORD *)(v22 + 40) = v57;
  v59 = (unsigned int)(v57 + 63) >> 6;
  v60 = v59;
  if ( v57 <= v58 << 6 )
  {
    if ( (_DWORD)v57 )
    {
      memcpy(*(void **)(v22 + 24), *(const void **)(v22 + 48), 8LL * v59);
      v77 = *(_DWORD *)(v22 + 40);
      v58 = *(_QWORD *)(v22 + 32);
      v59 = (unsigned int)(v77 + 63) >> 6;
      v60 = v59;
      if ( v58 <= v59 )
      {
LABEL_71:
        v78 = v77 & 0x3F;
        if ( v78 )
          *(_QWORD *)(*(_QWORD *)(v22 + 24) + 8LL * (v59 - 1)) &= ~(-1LL << v78);
        goto LABEL_45;
      }
    }
    else if ( v58 <= v59 )
    {
      goto LABEL_45;
    }
    v79 = v58 - v60;
    if ( v79 )
      memset((void *)(*(_QWORD *)(v22 + 24) + 8 * v60), 0, 8 * v79);
    v77 = *(_DWORD *)(v22 + 40);
    goto LABEL_71;
  }
  v65 = malloc(8LL * v59);
  v66 = 8LL * v59;
  v67 = (void *)v65;
  if ( !v65 )
  {
    if ( 8 * v60 || (v131 = malloc(1u), v66 = 0, !v131) )
    {
      na = v66;
      sub_16BD1C0("Allocation failed", 1u);
      v66 = na;
    }
    else
    {
      v67 = (void *)v131;
    }
  }
  memcpy(v67, *(const void **)(v22 + 48), v66);
  _libc_free(*(_QWORD *)(v22 + 24));
  *(_QWORD *)(v22 + 24) = v67;
  *(_QWORD *)(v22 + 32) = v60;
LABEL_45:
  v61 = *(_QWORD *)(a2 + 40);
  v62 = (_QWORD *)(a2 + 40);
  for ( i = (_QWORD *)(v61 & 0xFFFFFFFFFFFFFFF8LL); v62 != i; i = (_QWORD *)(*i & 0xFFFFFFFFFFFFFFF8LL) )
  {
    v64 = i - 3;
    if ( !i )
      v64 = 0;
    sub_3954210(a1, v64, v22);
  }
}
