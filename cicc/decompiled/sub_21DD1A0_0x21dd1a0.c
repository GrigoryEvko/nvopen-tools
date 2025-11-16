// Function: sub_21DD1A0
// Address: 0x21dd1a0
//
__int64 __fastcall sub_21DD1A0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r11
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // rdi
  unsigned __int16 *v11; // rdx
  __int64 v12; // rax
  unsigned int v13; // esi
  __int64 v14; // r13
  __int64 v15; // rdi
  unsigned int v16; // ecx
  __int64 *v17; // rax
  __int64 v18; // rdx
  unsigned int v19; // esi
  __int64 v20; // r9
  unsigned int v21; // ebx
  __int64 v22; // rdx
  _QWORD *v23; // rax
  __int64 v24; // rdi
  int v26; // r9d
  int v27; // r9d
  __int64 v28; // r10
  int v29; // edx
  unsigned int v30; // ecx
  __int64 v31; // rdi
  int v32; // r8d
  _QWORD *v33; // rsi
  unsigned int v34; // esi
  __int64 v35; // r13
  __int64 v36; // r8
  unsigned int v37; // eax
  __int64 *v38; // rdx
  __int64 v39; // rcx
  int v40; // r10d
  __int64 *v41; // rdi
  int v42; // eax
  int v43; // ecx
  unsigned int v44; // esi
  __int64 v45; // r13
  __int64 v46; // r8
  unsigned int v47; // edx
  _QWORD *v48; // rcx
  __int64 v49; // rdi
  int v50; // r10d
  _QWORD *v51; // rax
  int v52; // ecx
  int v53; // edx
  __int64 v54; // rax
  int v55; // eax
  int v56; // r10d
  __int64 *v57; // r9
  int v58; // eax
  int v59; // ecx
  int v60; // esi
  int v61; // esi
  __int64 v62; // r8
  int v63; // ecx
  unsigned int v64; // eax
  __int64 *v65; // rdx
  __int64 v66; // rdi
  int v67; // ecx
  int v68; // ecx
  __int64 v69; // rdi
  unsigned int v70; // r15d
  __int64 v71; // rsi
  int v72; // edx
  __int64 *v73; // rax
  int v74; // esi
  int v75; // esi
  __int64 v76; // r9
  int v77; // ecx
  unsigned int v78; // eax
  __int64 *v79; // rdx
  __int64 v80; // r8
  int v81; // esi
  int v82; // esi
  __int64 v83; // r8
  unsigned int v84; // r15d
  __int64 v85; // rcx
  int v86; // edx
  __int64 *v87; // rax
  int v88; // edi
  int v89; // edi
  __int64 v90; // r9
  int v91; // esi
  __int64 v92; // r15
  _QWORD *v93; // rcx
  __int64 v94; // r8
  int v95; // r9d
  int v96; // r9d
  __int64 v97; // r10
  __int64 v98; // rcx
  __int64 v99; // r8
  int v100; // edi
  _QWORD *v101; // rsi
  int v102; // r8d
  _QWORD *v103; // rcx
  int v104; // ecx
  int v105; // r8d
  int v106; // r8d
  __int64 v107; // r10
  unsigned int v108; // ecx
  int v109; // r9d
  __int64 v110; // rdi
  __int64 v113; // [rsp+18h] [rbp-78h]
  __int64 v114; // [rsp+18h] [rbp-78h]
  __int64 v115; // [rsp+18h] [rbp-78h]
  __int64 v116; // [rsp+18h] [rbp-78h]
  __int64 v117; // [rsp+18h] [rbp-78h]
  __int64 v118; // [rsp+18h] [rbp-78h]
  __int64 v119; // [rsp+18h] [rbp-78h]
  __int64 v120; // [rsp+18h] [rbp-78h]
  __int64 v121; // [rsp+18h] [rbp-78h]
  __int64 v122; // [rsp+18h] [rbp-78h]
  __int64 v123; // [rsp+28h] [rbp-68h] BYREF
  char v124[96]; // [rsp+30h] [rbp-60h] BYREF

  v5 = a4;
  v8 = *(unsigned int *)(*(_QWORD *)(a1 + 32) + 8LL);
  if ( (int)v8 < 0 )
    v9 = *(_QWORD *)(*(_QWORD *)(a5 + 24) + 16 * (v8 & 0x7FFFFFFF) + 8);
  else
    v9 = *(_QWORD *)(*(_QWORD *)(a5 + 272) + 8 * v8);
  if ( !v9 )
    goto LABEL_13;
  if ( (*(_BYTE *)(v9 + 3) & 0x10) != 0 )
  {
    while ( 1 )
    {
      v9 = *(_QWORD *)(v9 + 32);
      if ( !v9 )
        break;
      if ( (*(_BYTE *)(v9 + 3) & 0x10) == 0 )
        goto LABEL_5;
    }
    v19 = *(_DWORD *)(a4 + 24);
    if ( v19 )
      goto LABEL_14;
    goto LABEL_19;
  }
LABEL_5:
  v10 = *(_QWORD *)(v9 + 16);
  v11 = *(unsigned __int16 **)(v10 + 16);
  v12 = *((_QWORD *)v11 + 2);
  if ( (v12 & 0x80u) != 0LL )
  {
    if ( a2 != 1 )
    {
      if ( a2 != 3 )
        sub_16BD130("Invalid image type in .tex", 1u);
      v13 = *(_DWORD *)(a3 + 24);
      v14 = *(_QWORD *)(v10 + 32) + 200LL;
      if ( v13 )
      {
        v15 = *(_QWORD *)(a3 + 8);
        v16 = (v13 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
        v17 = (__int64 *)(v15 + 8LL * v16);
        v18 = *v17;
        if ( v14 == *v17 )
          goto LABEL_12;
        v56 = 1;
        v57 = 0;
        while ( v18 != -8 )
        {
          if ( v18 != -16 || v57 )
            v17 = v57;
          v16 = (v13 - 1) & (v56 + v16);
          v18 = *(_QWORD *)(v15 + 8LL * v16);
          if ( v14 == v18 )
            goto LABEL_12;
          ++v56;
          v57 = v17;
          v17 = (__int64 *)(v15 + 8LL * v16);
        }
        if ( !v57 )
          v57 = v17;
        v58 = *(_DWORD *)(a3 + 16);
        ++*(_QWORD *)a3;
        v59 = v58 + 1;
        if ( 4 * (v58 + 1) < 3 * v13 )
        {
          if ( v13 - *(_DWORD *)(a3 + 20) - v59 > v13 >> 3 )
            goto LABEL_69;
          v117 = v5;
          sub_21DCD80(a3, v13);
          v67 = *(_DWORD *)(a3 + 24);
          if ( v67 )
          {
            v68 = v67 - 1;
            v69 = *(_QWORD *)(a3 + 8);
            v5 = v117;
            v70 = v68 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
            v57 = (__int64 *)(v69 + 8LL * v70);
            v71 = *v57;
            if ( v14 != *v57 )
            {
              v72 = 1;
              v73 = 0;
              while ( v71 != -8 )
              {
                if ( v71 == -16 && !v73 )
                  v73 = v57;
                v70 = v68 & (v72 + v70);
                v57 = (__int64 *)(v69 + 8LL * v70);
                v71 = *v57;
                if ( v14 == *v57 )
                  goto LABEL_76;
                ++v72;
              }
              if ( v73 )
              {
                v57 = v73;
                v59 = *(_DWORD *)(a3 + 16) + 1;
                goto LABEL_69;
              }
            }
            goto LABEL_76;
          }
          goto LABEL_189;
        }
      }
      else
      {
        ++*(_QWORD *)a3;
      }
      v116 = v5;
      sub_21DCD80(a3, 2 * v13);
      v60 = *(_DWORD *)(a3 + 24);
      if ( v60 )
      {
        v61 = v60 - 1;
        v62 = *(_QWORD *)(a3 + 8);
        v5 = v116;
        v63 = 1;
        v64 = v61 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
        v57 = (__int64 *)(v62 + 8LL * v64);
        v65 = 0;
        v66 = *v57;
        if ( v14 != *v57 )
        {
          while ( v66 != -8 )
          {
            if ( v66 == -16 && !v65 )
              v65 = v57;
            v64 = v61 & (v63 + v64);
            v57 = (__int64 *)(v62 + 8LL * v64);
            v66 = *v57;
            if ( v14 == *v57 )
              goto LABEL_76;
            ++v63;
          }
          v59 = *(_DWORD *)(a3 + 16) + 1;
          if ( v65 )
            v57 = v65;
LABEL_69:
          *(_DWORD *)(a3 + 16) = v59;
          if ( *v57 != -8 )
            --*(_DWORD *)(a3 + 20);
          *v57 = v14;
          goto LABEL_12;
        }
LABEL_76:
        v59 = *(_DWORD *)(a3 + 16) + 1;
        goto LABEL_69;
      }
LABEL_189:
      ++*(_DWORD *)(a3 + 16);
      BUG();
    }
    v114 = v5;
    v54 = *(_QWORD *)(v10 + 32) + 160LL;
LABEL_58:
    v123 = v54;
    sub_21DCF30((__int64)v124, a3, &v123);
    v5 = v114;
    goto LABEL_12;
  }
  if ( (v12 & 0x300) != 0 )
  {
    if ( a2 != 2 )
      sub_16BD130("Invalid image type in .suld", 1u);
    v34 = *(_DWORD *)(a3 + 24);
    v35 = *(_QWORD *)(v10 + 32) + 40LL * (unsigned int)(1 << ((BYTE1(v12) & 3) - 1));
    if ( v34 )
    {
      v36 = *(_QWORD *)(a3 + 8);
      v37 = (v34 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
      v38 = (__int64 *)(v36 + 8LL * v37);
      v39 = *v38;
      if ( v35 == *v38 )
        goto LABEL_12;
      v40 = 1;
      v41 = 0;
      while ( v39 != -8 )
      {
        if ( v39 == -16 && !v41 )
          v41 = v38;
        v37 = (v34 - 1) & (v40 + v37);
        v38 = (__int64 *)(v36 + 8LL * v37);
        v39 = *v38;
        if ( v35 == *v38 )
          goto LABEL_12;
        ++v40;
      }
      v42 = *(_DWORD *)(a3 + 16);
      if ( !v41 )
        v41 = v38;
      ++*(_QWORD *)a3;
      v43 = v42 + 1;
      if ( 4 * (v42 + 1) < 3 * v34 )
      {
        if ( v34 - *(_DWORD *)(a3 + 20) - v43 > v34 >> 3 )
          goto LABEL_39;
        v119 = v5;
        sub_21DCD80(a3, v34);
        v81 = *(_DWORD *)(a3 + 24);
        if ( !v81 )
          goto LABEL_189;
        v82 = v81 - 1;
        v83 = *(_QWORD *)(a3 + 8);
        v5 = v119;
        v84 = v82 & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
        v41 = (__int64 *)(v83 + 8LL * v84);
        v85 = *v41;
        if ( v35 != *v41 )
        {
          v86 = 1;
          v87 = 0;
          while ( v85 != -8 )
          {
            if ( v85 == -16 && !v87 )
              v87 = v41;
            v84 = v82 & (v86 + v84);
            v41 = (__int64 *)(v83 + 8LL * v84);
            v85 = *v41;
            if ( v35 == *v41 )
              goto LABEL_86;
            ++v86;
          }
          if ( v87 )
          {
            v41 = v87;
            v43 = *(_DWORD *)(a3 + 16) + 1;
            goto LABEL_39;
          }
        }
        goto LABEL_86;
      }
    }
    else
    {
      ++*(_QWORD *)a3;
    }
    v118 = v5;
    sub_21DCD80(a3, 2 * v34);
    v74 = *(_DWORD *)(a3 + 24);
    if ( !v74 )
      goto LABEL_189;
    v75 = v74 - 1;
    v76 = *(_QWORD *)(a3 + 8);
    v5 = v118;
    v77 = 1;
    v78 = v75 & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
    v41 = (__int64 *)(v76 + 8LL * v78);
    v79 = 0;
    v80 = *v41;
    if ( v35 != *v41 )
    {
      while ( v80 != -8 )
      {
        if ( v80 == -16 && !v79 )
          v79 = v41;
        v78 = v75 & (v77 + v78);
        v41 = (__int64 *)(v76 + 8LL * v78);
        v80 = *v41;
        if ( v35 == *v41 )
          goto LABEL_86;
        ++v77;
      }
      v43 = *(_DWORD *)(a3 + 16) + 1;
      if ( v79 )
        v41 = v79;
LABEL_39:
      *(_DWORD *)(a3 + 16) = v43;
      if ( *v41 != -8 )
        --*(_DWORD *)(a3 + 20);
      *v41 = v35;
      goto LABEL_12;
    }
LABEL_86:
    v43 = *(_DWORD *)(a3 + 16) + 1;
    goto LABEL_39;
  }
  if ( (v12 & 0x400) == 0 )
  {
    if ( (v12 & 0x800) == 0 )
    {
      v55 = *v11;
      if ( v55 == 15 || v55 == 4878 )
      {
        v115 = v5;
        if ( (unsigned __int8)sub_21DD1A0(v10, a2, a3, v5, a5) )
        {
          v5 = v115;
          goto LABEL_12;
        }
      }
      return 0;
    }
    if ( a2 - 1 > 1 )
      sub_16BD130("Invalid image type in suq.", 1u);
    v114 = v5;
    v54 = *(_QWORD *)(v10 + 32) + 40LL;
    goto LABEL_58;
  }
  if ( a2 != 2 )
    sub_16BD130("Invalid image type in .sust", 1u);
  v44 = *(_DWORD *)(a3 + 24);
  v45 = *(_QWORD *)(v10 + 32);
  if ( !v44 )
  {
    ++*(_QWORD *)a3;
    goto LABEL_100;
  }
  v46 = *(_QWORD *)(a3 + 8);
  v47 = (v44 - 1) & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
  v48 = (_QWORD *)(v46 + 8LL * v47);
  v49 = *v48;
  if ( v45 == *v48 )
    goto LABEL_12;
  v50 = 1;
  v51 = 0;
  while ( 1 )
  {
    if ( v49 == -8 )
    {
      if ( !v51 )
        v51 = v48;
      v52 = *(_DWORD *)(a3 + 16);
      ++*(_QWORD *)a3;
      v53 = v52 + 1;
      if ( 4 * (v52 + 1) < 3 * v44 )
      {
        if ( v44 - *(_DWORD *)(a3 + 20) - v53 <= v44 >> 3 )
        {
          v120 = v5;
          sub_21DCD80(a3, v44);
          v88 = *(_DWORD *)(a3 + 24);
          if ( !v88 )
            goto LABEL_189;
          v89 = v88 - 1;
          v90 = *(_QWORD *)(a3 + 8);
          v91 = 1;
          LODWORD(v92) = v89 & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
          v5 = v120;
          v53 = *(_DWORD *)(a3 + 16) + 1;
          v93 = 0;
          v51 = (_QWORD *)(v90 + 8LL * (unsigned int)v92);
          v94 = *v51;
          if ( v45 != *v51 )
          {
            while ( v94 != -8 )
            {
              if ( !v93 && v94 == -16 )
                v93 = v51;
              v92 = v89 & (unsigned int)(v92 + v91);
              v51 = (_QWORD *)(v90 + 8 * v92);
              v94 = *v51;
              if ( v45 == *v51 )
                goto LABEL_52;
              ++v91;
            }
            if ( v93 )
              v51 = v93;
          }
        }
        goto LABEL_52;
      }
LABEL_100:
      v121 = v5;
      sub_21DCD80(a3, 2 * v44);
      v95 = *(_DWORD *)(a3 + 24);
      if ( !v95 )
        goto LABEL_189;
      v96 = v95 - 1;
      v97 = *(_QWORD *)(a3 + 8);
      v5 = v121;
      LODWORD(v98) = v96 & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
      v53 = *(_DWORD *)(a3 + 16) + 1;
      v51 = (_QWORD *)(v97 + 8LL * (unsigned int)v98);
      v99 = *v51;
      if ( v45 != *v51 )
      {
        v100 = 1;
        v101 = 0;
        while ( v99 != -8 )
        {
          if ( !v101 && v99 == -16 )
            v101 = v51;
          v98 = v96 & (unsigned int)(v98 + v100);
          v51 = (_QWORD *)(v97 + 8 * v98);
          v99 = *v51;
          if ( v45 == *v51 )
            goto LABEL_52;
          ++v100;
        }
        if ( v101 )
          v51 = v101;
      }
LABEL_52:
      *(_DWORD *)(a3 + 16) = v53;
      if ( *v51 != -8 )
        --*(_DWORD *)(a3 + 20);
      *v51 = v45;
      break;
    }
    if ( !v51 && v49 == -16 )
      v51 = v48;
    v47 = (v44 - 1) & (v50 + v47);
    v48 = (_QWORD *)(v46 + 8LL * v47);
    v49 = *v48;
    if ( v45 == *v48 )
      break;
    ++v50;
  }
LABEL_12:
  while ( 1 )
  {
    v9 = *(_QWORD *)(v9 + 32);
    if ( !v9 )
      break;
    if ( (*(_BYTE *)(v9 + 3) & 0x10) == 0 )
      goto LABEL_5;
  }
LABEL_13:
  v19 = *(_DWORD *)(v5 + 24);
  if ( !v19 )
  {
LABEL_19:
    ++*(_QWORD *)v5;
    goto LABEL_20;
  }
LABEL_14:
  v20 = *(_QWORD *)(v5 + 8);
  v21 = ((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4);
  v22 = (v19 - 1) & v21;
  v23 = (_QWORD *)(v20 + 8 * v22);
  v24 = *v23;
  if ( *v23 == a1 )
    return 1;
  v102 = 1;
  v103 = 0;
  while ( v24 != -8 )
  {
    if ( v24 == -16 && !v103 )
      v103 = v23;
    LODWORD(v22) = (v19 - 1) & (v102 + v22);
    v23 = (_QWORD *)(v20 + 8LL * (unsigned int)v22);
    v24 = *v23;
    if ( *v23 == a1 )
      return 1;
    ++v102;
  }
  if ( v103 )
    v23 = v103;
  v104 = *(_DWORD *)(v5 + 16);
  ++*(_QWORD *)v5;
  v29 = v104 + 1;
  if ( 4 * (v104 + 1) < 3 * v19 )
  {
    if ( v19 - *(_DWORD *)(v5 + 20) - v29 > v19 >> 3 )
      goto LABEL_118;
    v122 = v5;
    sub_1E22DE0(v5, v19);
    v5 = v122;
    v105 = *(_DWORD *)(v122 + 24);
    if ( v105 )
    {
      v106 = v105 - 1;
      v107 = *(_QWORD *)(v122 + 8);
      v108 = v106 & v21;
      v109 = 1;
      v29 = *(_DWORD *)(v122 + 16) + 1;
      v33 = 0;
      v23 = (_QWORD *)(v107 + 8LL * (v106 & v21));
      v110 = *v23;
      if ( *v23 == a1 )
        goto LABEL_118;
      while ( v110 != -8 )
      {
        if ( !v33 && v110 == -16 )
          v33 = v23;
        v108 = v106 & (v109 + v108);
        v23 = (_QWORD *)(v107 + 8LL * v108);
        v110 = *v23;
        if ( *v23 == a1 )
          goto LABEL_118;
        ++v109;
      }
LABEL_166:
      if ( v33 )
        v23 = v33;
      goto LABEL_118;
    }
LABEL_188:
    ++*(_DWORD *)(v5 + 16);
    BUG();
  }
LABEL_20:
  v113 = v5;
  sub_1E22DE0(v5, 2 * v19);
  v5 = v113;
  v26 = *(_DWORD *)(v113 + 24);
  if ( !v26 )
    goto LABEL_188;
  v27 = v26 - 1;
  v28 = *(_QWORD *)(v113 + 8);
  v29 = *(_DWORD *)(v113 + 16) + 1;
  v30 = v27 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v23 = (_QWORD *)(v28 + 8LL * v30);
  v31 = *v23;
  if ( *v23 != a1 )
  {
    v32 = 1;
    v33 = 0;
    while ( v31 != -8 )
    {
      if ( !v33 && v31 == -16 )
        v33 = v23;
      v30 = v27 & (v32 + v30);
      v23 = (_QWORD *)(v28 + 8LL * v30);
      v31 = *v23;
      if ( *v23 == a1 )
        goto LABEL_118;
      ++v32;
    }
    goto LABEL_166;
  }
LABEL_118:
  *(_DWORD *)(v5 + 16) = v29;
  if ( *v23 != -8 )
    --*(_DWORD *)(v5 + 20);
  *v23 = a1;
  return 1;
}
