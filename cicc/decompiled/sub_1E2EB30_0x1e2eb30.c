// Function: sub_1E2EB30
// Address: 0x1e2eb30
//
__int64 __fastcall sub_1E2EB30(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r9
  unsigned int v7; // esi
  __int64 v8; // rdi
  unsigned int v9; // ecx
  __int64 *v10; // rax
  __int64 v11; // rdx
  __int64 *v12; // r8
  __int64 v13; // r14
  __int64 v14; // rdx
  unsigned int v15; // esi
  __int64 v16; // rax
  unsigned int v17; // edx
  int v18; // r10d
  unsigned int v19; // ecx
  __int64 *v20; // r15
  __int64 v21; // rdi
  __int64 v22; // rax
  unsigned __int64 *v23; // rax
  unsigned __int64 *v24; // rbx
  unsigned __int64 v25; // rdi
  unsigned int v26; // edi
  __int64 v27; // rbx
  __int64 v28; // rcx
  __int64 result; // rax
  __int64 v30; // rax
  _QWORD *v31; // r12
  __int64 v32; // rax
  __int64 v33; // rcx
  bool v34; // zf
  bool v35; // dl
  bool v36; // al
  __int64 v37; // rcx
  bool v38; // al
  unsigned __int64 v39; // r14
  __int64 *v40; // r8
  __int64 *p_src; // r13
  unsigned __int64 v42; // r12
  __int64 v43; // r15
  int v44; // r11d
  __int64 v45; // r10
  int v46; // eax
  int v47; // eax
  __int64 v48; // r12
  __int64 v49; // rdx
  __int64 v50; // rdx
  unsigned __int64 *v51; // r12
  unsigned __int64 v52; // rax
  __int64 v53; // r13
  unsigned __int64 v54; // rdi
  __int64 v55; // rax
  __int64 v56; // r12
  __int64 v57; // rcx
  __int64 v58; // r10
  __int64 v59; // rdx
  __int64 v60; // rdi
  __int64 v61; // rsi
  char *v62; // r9
  char *v63; // r15
  unsigned __int64 v64; // rcx
  signed __int64 v65; // r11
  __int64 v66; // r10
  unsigned __int64 v67; // rbx
  size_t v68; // rcx
  __int64 v69; // rbx
  __int64 v70; // rax
  int v71; // r9d
  __int64 v72; // rdx
  unsigned __int64 v73; // rax
  __int64 v74; // rdx
  int v75; // r15d
  __int64 *v76; // r11
  int v77; // edi
  int v78; // ecx
  void *v79; // rdi
  __int64 v80; // r8
  char *v81; // rbx
  size_t v82; // r10
  int v83; // eax
  int v84; // ecx
  __int64 v85; // rsi
  __int64 v86; // rdx
  __int64 v87; // rdi
  int v88; // r10d
  int v89; // eax
  int v90; // esi
  __int64 v91; // rdi
  __int64 v92; // rdx
  __int64 v93; // r8
  int v94; // r11d
  __int64 *v95; // r10
  int v96; // eax
  int v97; // edx
  __int64 v98; // rsi
  __int64 v99; // rdi
  unsigned int v100; // r15d
  __int64 v101; // rcx
  int v102; // eax
  int v103; // edx
  __int64 v104; // rdi
  __int64 *v105; // r8
  __int64 v106; // r14
  int v107; // r10d
  __int64 v108; // rsi
  int v109; // edx
  __int64 *v110; // r10
  int v111; // r10d
  size_t v112; // [rsp+8h] [rbp-B8h]
  signed __int64 v113; // [rsp+10h] [rbp-B0h]
  signed __int64 v114; // [rsp+10h] [rbp-B0h]
  __int64 v115; // [rsp+18h] [rbp-A8h]
  __int64 v116; // [rsp+18h] [rbp-A8h]
  __int64 v117; // [rsp+18h] [rbp-A8h]
  __int64 *v118; // [rsp+20h] [rbp-A0h]
  signed __int64 v119; // [rsp+20h] [rbp-A0h]
  char *v120; // [rsp+20h] [rbp-A0h]
  char *v121; // [rsp+20h] [rbp-A0h]
  char *v122; // [rsp+28h] [rbp-98h]
  __int64 *v123; // [rsp+28h] [rbp-98h]
  __int64 v124; // [rsp+28h] [rbp-98h]
  signed __int64 v125; // [rsp+28h] [rbp-98h]
  __int64 *v126; // [rsp+28h] [rbp-98h]
  __int64 *v127; // [rsp+30h] [rbp-90h]
  __int64 v128; // [rsp+38h] [rbp-88h]
  __int64 src; // [rsp+40h] [rbp-80h] BYREF
  __int64 *v130; // [rsp+48h] [rbp-78h] BYREF
  int v131; // [rsp+50h] [rbp-70h]
  void *v132; // [rsp+60h] [rbp-60h]
  _QWORD v133[2]; // [rsp+68h] [rbp-58h] BYREF
  __int64 v134; // [rsp+78h] [rbp-48h]
  __int64 v135; // [rsp+80h] [rbp-40h]

  v3 = a1 + 8;
  v7 = *(_DWORD *)(a1 + 32);
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 8);
LABEL_138:
    v128 = v3;
    sub_1E2E3C0(v3, 2 * v7);
    v89 = *(_DWORD *)(a1 + 32);
    if ( !v89 )
      goto LABEL_192;
    v90 = v89 - 1;
    v91 = *(_QWORD *)(a1 + 16);
    v3 = v128;
    LODWORD(v92) = (v89 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v78 = *(_DWORD *)(a1 + 24) + 1;
    v10 = (__int64 *)(v91 + 32LL * (unsigned int)v92);
    v93 = *v10;
    if ( a2 != *v10 )
    {
      v94 = 1;
      v95 = 0;
      while ( v93 != -8 )
      {
        if ( !v95 && v93 == -16 )
          v95 = v10;
        v92 = v90 & (unsigned int)(v92 + v94);
        v10 = (__int64 *)(v91 + 32 * v92);
        v93 = *v10;
        if ( a2 == *v10 )
          goto LABEL_118;
        ++v94;
      }
      if ( v95 )
        v10 = v95;
    }
    goto LABEL_118;
  }
  v8 = *(_QWORD *)(a1 + 16);
  v9 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v8 + 32LL * v9);
  v11 = *v10;
  if ( a2 == *v10 )
  {
    v12 = (__int64 *)v10[2];
    v13 = *((unsigned int *)v10 + 6);
    goto LABEL_4;
  }
  v75 = 1;
  v76 = 0;
  while ( v11 != -8 )
  {
    if ( v76 || v11 != -16 )
      v10 = v76;
    v9 = (v7 - 1) & (v75 + v9);
    v110 = (__int64 *)(v8 + 32LL * v9);
    v11 = *v110;
    if ( a2 == *v110 )
    {
      v12 = (__int64 *)v110[2];
      v13 = *((unsigned int *)v110 + 6);
      v10 = (__int64 *)(v8 + 32LL * v9);
      goto LABEL_4;
    }
    ++v75;
    v76 = v10;
    v10 = (__int64 *)(v8 + 32LL * v9);
  }
  v77 = *(_DWORD *)(a1 + 24);
  if ( v76 )
    v10 = v76;
  ++*(_QWORD *)(a1 + 8);
  v78 = v77 + 1;
  if ( 4 * (v77 + 1) >= 3 * v7 )
    goto LABEL_138;
  if ( v7 - *(_DWORD *)(a1 + 28) - v78 <= v7 >> 3 )
  {
    v128 = v3;
    sub_1E2E3C0(v3, v7);
    v102 = *(_DWORD *)(a1 + 32);
    if ( !v102 )
      goto LABEL_192;
    v103 = v102 - 1;
    v104 = *(_QWORD *)(a1 + 16);
    v105 = 0;
    LODWORD(v106) = (v102 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v3 = v128;
    v107 = 1;
    v78 = *(_DWORD *)(a1 + 24) + 1;
    v10 = (__int64 *)(v104 + 32LL * (unsigned int)v106);
    v108 = *v10;
    if ( a2 != *v10 )
    {
      while ( v108 != -8 )
      {
        if ( !v105 && v108 == -16 )
          v105 = v10;
        v106 = v103 & (unsigned int)(v106 + v107);
        v10 = (__int64 *)(v104 + 32 * v106);
        v108 = *v10;
        if ( a2 == *v10 )
          goto LABEL_118;
        ++v107;
      }
      if ( v105 )
        v10 = v105;
    }
  }
LABEL_118:
  *(_DWORD *)(a1 + 24) = v78;
  if ( *v10 != -8 )
    --*(_DWORD *)(a1 + 28);
  *v10 = a2;
  v13 = 0;
  v12 = 0;
  v10[1] = 0;
  v10[2] = 0;
  *((_DWORD *)v10 + 6) = 0;
LABEL_4:
  v14 = v10[1];
  v10[1] = 0;
  v15 = *(_DWORD *)(a1 + 32);
  v16 = *(_QWORD *)(a1 + 16);
  v130 = v12;
  src = v14;
  v131 = v13;
  if ( !v15 )
    goto LABEL_129;
  v17 = v15 - 1;
  v18 = 1;
  v19 = (v15 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v20 = (__int64 *)(v16 + 32LL * v19);
  v21 = *v20;
  if ( a2 == *v20 )
  {
LABEL_6:
    v22 = v20[1];
    if ( (v22 & 4) != 0 )
    {
      v23 = (unsigned __int64 *)(v22 & 0xFFFFFFFFFFFFFFF8LL);
      v24 = v23;
      if ( v23 )
      {
        v25 = *v23;
        if ( (unsigned __int64 *)*v23 != v23 + 2 )
        {
          v127 = v12;
          v128 = v3;
          _libc_free(v25);
          v12 = v127;
          v3 = v128;
        }
        v127 = v12;
        v128 = v3;
        j_j___libc_free_0(v24, 48);
        v12 = v127;
        v3 = v128;
      }
    }
    *v20 = -16;
    v15 = *(_DWORD *)(a1 + 32);
    --*(_DWORD *)(a1 + 24);
    v16 = *(_QWORD *)(a1 + 16);
    ++*(_DWORD *)(a1 + 28);
    if ( v15 )
    {
      v17 = v15 - 1;
      goto LABEL_13;
    }
LABEL_129:
    ++*(_QWORD *)(a1 + 8);
    v15 = 0;
    goto LABEL_130;
  }
  while ( v21 != -8 )
  {
    v19 = v17 & (v18 + v19);
    v20 = (__int64 *)(v16 + 32LL * v19);
    v21 = *v20;
    if ( a2 == *v20 )
      goto LABEL_6;
    ++v18;
  }
LABEL_13:
  v26 = v17 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v27 = v16 + 32LL * v26;
  v28 = *(_QWORD *)v27;
  if ( a3 != *(_QWORD *)v27 )
  {
    v44 = 1;
    v45 = 0;
    while ( v28 != -8 )
    {
      if ( v45 || v28 != -16 )
        v27 = v45;
      v26 = v17 & (v44 + v26);
      v28 = *(_QWORD *)(v16 + 32LL * v26);
      v128 = v16 + 32LL * v26;
      if ( a3 == v28 )
      {
        v27 = v16 + 32LL * v26;
        goto LABEL_14;
      }
      ++v44;
      v45 = v27;
      v27 = v128;
    }
    v46 = *(_DWORD *)(a1 + 24);
    if ( v45 )
      v27 = v45;
    ++*(_QWORD *)(a1 + 8);
    v47 = v46 + 1;
    if ( 4 * v47 < 3 * v15 )
    {
      if ( v15 - (v47 + *(_DWORD *)(a1 + 28)) > v15 >> 3 )
      {
LABEL_41:
        *(_DWORD *)(a1 + 24) = v47;
        if ( *(_QWORD *)v27 != -8 )
          --*(_DWORD *)(a1 + 28);
        *(_QWORD *)v27 = a3;
        result = 0;
        *(_QWORD *)(v27 + 8) = 0;
        *(_QWORD *)(v27 + 16) = 0;
        *(_DWORD *)(v27 + 24) = 0;
        goto LABEL_44;
      }
      v128 = (__int64)v12;
      sub_1E2E3C0(v3, v15);
      v96 = *(_DWORD *)(a1 + 32);
      if ( v96 )
      {
        v97 = v96 - 1;
        v98 = *(_QWORD *)(a1 + 16);
        v99 = 0;
        v100 = (v96 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
        v12 = (__int64 *)v128;
        LODWORD(v3) = 1;
        v47 = *(_DWORD *)(a1 + 24) + 1;
        v27 = v98 + 32LL * v100;
        v101 = *(_QWORD *)v27;
        if ( a3 != *(_QWORD *)v27 )
        {
          while ( v101 != -8 )
          {
            if ( !v99 && v101 == -16 )
              v99 = v27;
            v111 = v3 + 1;
            LODWORD(v3) = v97 & (v100 + v3);
            v100 = v3;
            v27 = v98 + 32LL * (unsigned int)v3;
            v101 = *(_QWORD *)v27;
            if ( a3 == *(_QWORD *)v27 )
              goto LABEL_41;
            LODWORD(v3) = v111;
          }
          if ( v99 )
            v27 = v99;
        }
        goto LABEL_41;
      }
LABEL_192:
      ++*(_DWORD *)(a1 + 24);
      BUG();
    }
LABEL_130:
    v128 = (__int64)v12;
    sub_1E2E3C0(v3, 2 * v15);
    v83 = *(_DWORD *)(a1 + 32);
    if ( v83 )
    {
      v84 = v83 - 1;
      v85 = *(_QWORD *)(a1 + 16);
      v12 = (__int64 *)v128;
      LODWORD(v86) = (v83 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v47 = *(_DWORD *)(a1 + 24) + 1;
      v27 = v85 + 32LL * (unsigned int)v86;
      v87 = *(_QWORD *)v27;
      if ( a3 != *(_QWORD *)v27 )
      {
        v88 = 1;
        v3 = 0;
        while ( v87 != -8 )
        {
          if ( v87 == -16 && !v3 )
            v3 = v27;
          v86 = v84 & (unsigned int)(v86 + v88);
          v27 = v85 + 32 * v86;
          v87 = *(_QWORD *)v27;
          if ( a3 == *(_QWORD *)v27 )
            goto LABEL_41;
          ++v88;
        }
        if ( v3 )
          v27 = v3;
      }
      goto LABEL_41;
    }
    goto LABEL_192;
  }
LABEL_14:
  result = *(_QWORD *)(v27 + 8);
  if ( (result & 0xFFFFFFFFFFFFFFF8LL) != 0 && ((result & 4) == 0 || *(_DWORD *)((result & 0xFFFFFFFFFFFFFFF8LL) + 8)) )
  {
    v134 = 0;
    v135 = 0;
    v133[0] = 2;
    v132 = &unk_49FBDD8;
    v30 = *(_QWORD *)(a1 + 40);
    v133[1] = 0;
    v31 = (_QWORD *)(v30 + 40 * v13);
    v32 = v31[3];
    if ( v32 )
    {
      if ( v32 == -16 || v32 == -8 )
      {
        v31[3] = 0;
        v37 = 0;
        v38 = v134 != -8 && v134 != 0 && v134 != -16;
      }
      else
      {
        sub_1649B30(v31 + 1);
        v33 = v134;
        v34 = v134 == -8;
        v31[3] = v134;
        v35 = v33 != 0;
        v36 = v33 != -16;
        if ( !v34 && v33 != 0 )
        {
          sub_1649AC0(v31 + 1, v133[0] & 0xFFFFFFFFFFFFFFF8LL);
          v37 = v135;
          v38 = v134 != 0 && v134 != -8 && v134 != -16;
        }
        else
        {
          v37 = v135;
          v38 = !v34 && v35 && v36;
        }
      }
      v31[4] = v37;
      v132 = &unk_49EE2B0;
      if ( v38 )
        sub_1649B30(v133);
    }
    else
    {
      v31[4] = 0;
    }
    v39 = src & 0xFFFFFFFFFFFFFFF8LL;
    LOBYTE(v128) = (src >> 2) & 1;
    if ( ((src >> 2) & 1) != 0 )
    {
      p_src = *(__int64 **)v39;
      v40 = (__int64 *)(*(_QWORD *)v39 + 8LL * *(unsigned int *)(v39 + 8));
      result = *(_QWORD *)(v27 + 8);
      v42 = result & 0xFFFFFFFFFFFFFFF8LL;
      if ( (result & 4) == 0 )
      {
LABEL_28:
        if ( v42 )
        {
          if ( p_src == v40 )
            goto LABEL_30;
          v127 = v40;
          v70 = sub_22077B0(48);
          v40 = v127;
          if ( v70 )
          {
            *(_QWORD *)v70 = v70 + 16;
            *(_QWORD *)(v70 + 8) = 0x400000000LL;
          }
          v72 = v70;
          v73 = v70 & 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v27 + 8) = v72 | 4;
          v74 = *(unsigned int *)(v73 + 8);
          if ( (unsigned int)v74 >= *(_DWORD *)(v73 + 12) )
          {
            v126 = v40;
            v127 = (__int64 *)v73;
            sub_16CD150(v73, (const void *)(v73 + 16), 0, 8, (int)v40, v71);
            v73 = (unsigned __int64)v127;
            v40 = v126;
            v74 = *((unsigned int *)v127 + 2);
          }
          *(_QWORD *)(*(_QWORD *)v73 + 8 * v74) = v42;
          v57 = 8;
          ++*(_DWORD *)(v73 + 8);
          v58 = *(_QWORD *)(v27 + 8);
          v56 = v58;
          goto LABEL_85;
        }
        v43 = 0;
        if ( p_src == v40 )
        {
LABEL_30:
          if ( (_BYTE)v128 && v39 )
          {
            if ( *(_QWORD *)v39 != v39 + 16 )
              _libc_free(*(_QWORD *)v39);
            return j_j___libc_free_0(v39, 48);
          }
          return result;
        }
LABEL_81:
        if ( v40 == p_src + 1 )
        {
          result = *p_src;
          *(_QWORD *)(v27 + 8) = *p_src;
          goto LABEL_30;
        }
        v127 = v40;
        v55 = sub_22077B0(48);
        v40 = v127;
        if ( v55 )
        {
          *(_QWORD *)v55 = v55 + 16;
          *(_QWORD *)(v55 + 8) = 0x400000000LL;
        }
        v56 = v55 | 4;
        v57 = v43;
        *(_QWORD *)(v27 + 8) = v55 | 4;
        LOBYTE(v58) = v55 | 4;
LABEL_85:
        v42 = v56 & 0xFFFFFFFFFFFFFFF8LL;
        result = *(unsigned int *)(v42 + 8);
        v59 = *(_QWORD *)v42;
        v60 = v42;
        v61 = 8 * result;
        v62 = (char *)(*(_QWORD *)v42 + 8 * result);
        if ( (v58 & 4) != 0 )
        {
          v127 = (__int64 *)v57;
          v63 = (char *)(v59 + v57);
        }
        else
        {
          v63 = (char *)(v27 + v57 + 8);
          v127 = (__int64 *)&v63[-v59];
        }
        v64 = *(unsigned int *)(v42 + 12);
        v65 = (char *)v40 - (char *)p_src;
        v66 = v40 - p_src;
        v67 = v66;
        if ( v63 != v62 )
        {
          if ( v66 + result > v64 )
          {
            v115 = v40 - p_src;
            v119 = (char *)v40 - (char *)p_src;
            v123 = v40;
            sub_16CD150(v42, (const void *)(v42 + 16), v66 + result, 8, (int)v40, (int)v62);
            v59 = *(_QWORD *)v42;
            v66 = v115;
            v65 = v119;
            v63 = (char *)v127 + *(_QWORD *)v42;
            result = *(unsigned int *)(v42 + 8);
            v40 = v123;
            v61 = 8 * result;
            v62 = (char *)(*(_QWORD *)v42 + 8 * result);
          }
          v68 = v61 - (_QWORD)v127;
          v69 = (v61 - (__int64)v127) >> 3;
          if ( v61 - (__int64)v127 >= (unsigned __int64)v65 )
          {
            v79 = v62;
            v80 = v61 - v65;
            v81 = (char *)(v59 + v61 - v65);
            v82 = v65;
            v124 = v65 >> 3;
            if ( v65 >> 3 > (unsigned __int64)*(unsigned int *)(v42 + 12) - result )
            {
              v112 = v65;
              v114 = v65;
              v117 = v61 - v65;
              v121 = v62;
              sub_16CD150(v42, (const void *)(v42 + 16), result + (v65 >> 3), 8, v80, (int)v62);
              result = *(unsigned int *)(v42 + 8);
              v82 = v112;
              v65 = v114;
              v80 = v117;
              v62 = v121;
              v79 = (void *)(*(_QWORD *)v42 + 8 * result);
            }
            if ( v81 != v62 )
            {
              v113 = v65;
              v116 = v80;
              v120 = v62;
              memmove(v79, v81, v82);
              LODWORD(result) = *(_DWORD *)(v42 + 8);
              v65 = v113;
              v80 = v116;
              v62 = v120;
            }
            *(_DWORD *)(v42 + 8) = v124 + result;
            if ( v81 != v63 )
            {
              v125 = v65;
              memmove(&v62[-(v80 - (_QWORD)v127)], v63, v80 - (_QWORD)v127);
              v65 = v125;
            }
            result = (__int64)memmove(v63, p_src, v65);
          }
          else
          {
            result += v66;
            *(_DWORD *)(v42 + 8) = result;
            if ( v63 != v62 )
            {
              v118 = v40;
              v122 = v62;
              v127 = (__int64 *)v68;
              result = (__int64)memcpy((void *)(v59 + 8LL * (unsigned int)result - v68), v63, v68);
              v40 = v118;
              v62 = v122;
              v68 = (size_t)v127;
            }
            if ( v69 )
            {
              for ( result = 0; result != v69; ++result )
                *(_QWORD *)&v63[8 * result] = p_src[result];
              p_src = (__int64 *)((char *)p_src + v68);
            }
            if ( v40 != p_src )
              result = (__int64)memcpy(v62, p_src, (char *)v40 - (char *)p_src);
          }
          goto LABEL_30;
        }
LABEL_109:
        if ( v67 > v64 - result )
        {
          v127 = (__int64 *)v65;
          sub_16CD150(v60, (const void *)(v42 + 16), v67 + result, 8, (int)v40, (int)v62);
          v65 = (signed __int64)v127;
          v62 = (char *)(*(_QWORD *)v42 + 8LL * *(unsigned int *)(v42 + 8));
        }
        result = (__int64)memcpy(v62, p_src, v65);
        *(_DWORD *)(v42 + 8) += v67;
        goto LABEL_30;
      }
    }
    else
    {
      if ( v39 )
      {
        v40 = (__int64 *)&v130;
        p_src = &src;
      }
      else
      {
        p_src = &src;
        v40 = &src;
      }
      result = *(_QWORD *)(v27 + 8);
      v42 = result & 0xFFFFFFFFFFFFFFF8LL;
      if ( (result & 4) == 0 )
        goto LABEL_28;
    }
    result = *(unsigned int *)(v42 + 8);
    if ( p_src == v40 )
      goto LABEL_30;
    v43 = 8 * result;
    if ( v42 )
    {
      v64 = *(unsigned int *)(v42 + 12);
      v62 = (char *)(*(_QWORD *)v42 + 8 * result);
      v60 = v42;
      v65 = (char *)v40 - (char *)p_src;
      v67 = v40 - p_src;
      goto LABEL_109;
    }
    goto LABEL_81;
  }
LABEL_44:
  v48 = *(_QWORD *)(a1 + 40) + 40LL * (unsigned int)v13;
  v49 = *(_QWORD *)(v48 + 24);
  if ( a3 != v49 )
  {
    if ( v49 != -8 && v49 != 0 && v49 != -16 )
    {
      v128 = (__int64)v12;
      sub_1649B30((_QWORD *)(v48 + 8));
      v12 = (__int64 *)v128;
    }
    *(_QWORD *)(v48 + 24) = a3;
    if ( a3 == -8 || a3 == 0 || a3 == -16 )
    {
      result = *(_QWORD *)(v27 + 8);
    }
    else
    {
      v128 = (__int64)v12;
      sub_164C220(v48 + 8);
      result = *(_QWORD *)(v27 + 8);
      v12 = (__int64 *)v128;
    }
  }
  v50 = (src >> 2) & 1;
  v51 = (unsigned __int64 *)(src & 0xFFFFFFFFFFFFFFF8LL);
  if ( (src & 0xFFFFFFFFFFFFFFF8LL) != 0 && (!(_BYTE)v50 || *(_DWORD *)((src & 0xFFFFFFFFFFFFFFF8LL) + 8)) )
  {
    if ( (result & 4) == 0 || (v52 = result & 0xFFFFFFFFFFFFFFF8LL, (v53 = v52) == 0) )
    {
LABEL_61:
      result = src;
      *(_QWORD *)(v27 + 16) = v12;
      *(_DWORD *)(v27 + 24) = v13;
      *(_QWORD *)(v27 + 8) = result;
      return result;
    }
    if ( (_BYTE)v50 )
    {
      v54 = *(_QWORD *)v52;
      if ( *(_QWORD *)v52 != v52 + 16 )
      {
        v128 = (__int64)v12;
        _libc_free(v54);
        v12 = (__int64 *)v128;
      }
      v128 = (__int64)v12;
      j_j___libc_free_0(v53, 48);
      v12 = (__int64 *)v128;
      goto LABEL_61;
    }
    *(_DWORD *)(v52 + 8) = 0;
    v109 = *(_DWORD *)(v52 + 12);
    result = 0;
    if ( !v109 )
    {
      v128 = (__int64)v12;
      sub_16CD150(v53, (const void *)(v53 + 16), 0, 8, (int)v12, v3);
      v12 = (__int64 *)v128;
      result = 8LL * *(unsigned int *)(v53 + 8);
    }
    *(_QWORD *)(*(_QWORD *)v53 + result) = v51;
    ++*(_DWORD *)(v53 + 8);
    *(_QWORD *)(v27 + 16) = v12;
    *(_DWORD *)(v27 + 24) = v13;
  }
  else
  {
    if ( (result & 4) != 0 )
    {
      result &= 0xFFFFFFFFFFFFFFF8LL;
      if ( result )
        *(_DWORD *)(result + 8) = 0;
    }
    else
    {
      *(_QWORD *)(v27 + 8) = 0;
    }
    *(_QWORD *)(v27 + 16) = v12;
    *(_DWORD *)(v27 + 24) = v13;
    if ( (_BYTE)v50 && v51 )
    {
      if ( (unsigned __int64 *)*v51 != v51 + 2 )
        _libc_free(*v51);
      return j_j___libc_free_0(v51, 48);
    }
  }
  return result;
}
