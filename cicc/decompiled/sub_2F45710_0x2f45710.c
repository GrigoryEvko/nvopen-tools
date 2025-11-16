// Function: sub_2F45710
// Address: 0x2f45710
//
__int64 __fastcall sub_2F45710(__int64 a1, unsigned __int64 a2, _QWORD *a3)
{
  unsigned int v5; // esi
  __int64 v6; // rdi
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // r8
  __int64 *v10; // r15
  int v12; // eax
  __int64 v13; // rdx
  unsigned __int64 v14; // rbx
  unsigned __int64 v15; // rax
  __int64 v16; // rcx
  unsigned int v17; // r14d
  __int64 v18; // r13
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // r11
  __int64 v21; // r8
  int v22; // edx
  __int64 v23; // rax
  unsigned int v24; // edx
  __int64 v25; // r9
  __int64 v26; // r11
  unsigned int v27; // eax
  __int64 v28; // r14
  __int64 *v29; // r8
  __int64 v30; // rdx
  unsigned __int64 v31; // rax
  __int64 j; // rax
  unsigned int v33; // r14d
  unsigned int v34; // r8d
  __int64 *v35; // rdx
  __int64 v36; // r9
  __int64 v37; // r13
  bool v38; // al
  __int64 v39; // rdx
  __int64 v40; // r15
  __int64 v41; // r9
  int v42; // r8d
  unsigned __int64 *v43; // rdx
  unsigned int v44; // edi
  unsigned __int64 *v45; // rax
  unsigned __int64 v46; // rcx
  _QWORD *v47; // rax
  unsigned int v48; // edx
  __int64 *v49; // rax
  __int64 v50; // r8
  int v51; // r9d
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rsi
  unsigned int v56; // ecx
  __int64 *v57; // rdx
  __int64 v58; // r8
  int v59; // eax
  int v60; // eax
  int v61; // r14d
  int v62; // r14d
  __int64 v63; // r9
  unsigned int v64; // ecx
  unsigned __int64 v65; // rsi
  int v66; // r10d
  unsigned __int64 *v67; // rdi
  int v68; // r10d
  int v69; // r10d
  __int64 v70; // rdi
  unsigned __int64 *v71; // rsi
  unsigned int v72; // r14d
  int v73; // r9d
  unsigned __int64 v74; // rcx
  int v75; // r9d
  int v76; // edx
  __int64 i; // rbx
  int v78; // r8d
  int v79; // edx
  __int64 v80; // r8
  __int64 v81; // rdx
  int v82; // r8d
  int v83; // eax
  int v84; // r9d
  int v85; // edx
  unsigned __int64 v86; // rax
  int v87; // r9d
  int v88; // eax
  int v89; // r13d
  __int64 v90; // [rsp+0h] [rbp-60h]
  __int64 v91; // [rsp+8h] [rbp-58h]
  __int64 v92; // [rsp+8h] [rbp-58h]
  int v93; // [rsp+8h] [rbp-58h]
  unsigned int v94; // [rsp+8h] [rbp-58h]
  char v95; // [rsp+10h] [rbp-50h]
  unsigned int v96; // [rsp+10h] [rbp-50h]
  unsigned __int64 v97; // [rsp+10h] [rbp-50h]
  __int64 v98; // [rsp+10h] [rbp-50h]
  int v99; // [rsp+18h] [rbp-48h]
  unsigned __int64 v100; // [rsp+18h] [rbp-48h]
  __int64 v101; // [rsp+20h] [rbp-40h]
  __int64 v102; // [rsp+20h] [rbp-40h]

  v5 = *(_DWORD *)(a1 + 40);
  v6 = *(_QWORD *)(a1 + 24);
  if ( !v5 )
  {
LABEL_8:
    v10 = (__int64 *)(v6 + 16LL * v5);
    goto LABEL_9;
  }
  v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v6 + 16LL * v7);
  v9 = *v8;
  if ( a2 != *v8 )
  {
    v12 = 1;
    while ( v9 != -4096 )
    {
      v75 = v12 + 1;
      v7 = (v5 - 1) & (v12 + v7);
      v8 = (__int64 *)(v6 + 16LL * v7);
      v9 = *v8;
      if ( a2 == *v8 )
        goto LABEL_3;
      v12 = v75;
    }
    goto LABEL_8;
  }
LABEL_3:
  v10 = (__int64 *)(v6 + 16LL * v5);
  if ( v8 != v10 )
  {
    LODWORD(v10) = 0;
    *a3 = v8[1];
    return (unsigned int)v10;
  }
LABEL_9:
  v13 = *(_QWORD *)a2;
  v14 = a2;
  v15 = a2;
  if ( (*(_QWORD *)a2 & 4) == 0 && (*(_BYTE *)(a2 + 44) & 8) != 0 )
  {
    do
      v15 = *(_QWORD *)(v15 + 8);
    while ( (*(_BYTE *)(v15 + 44) & 8) != 0 );
  }
  v16 = *(_QWORD *)(v15 + 8);
  v99 = 1;
  v17 = v5 - 1;
  v18 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 56LL);
  v101 = *(_QWORD *)(a1 + 8);
  if ( a2 == v18 )
  {
LABEL_122:
    v14 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 56LL);
    v26 = v101 + 48;
    if ( v16 != v101 + 48 )
      goto LABEL_20;
    goto LABEL_60;
  }
  while ( 1 )
  {
    v19 = v13 & 0xFFFFFFFFFFFFFFF8LL;
    v20 = v19;
    if ( !v19 )
      BUG();
    v21 = *(_QWORD *)v19;
    v95 = (*(__int64 *)v19 >> 2) & 1;
    if ( ((*(__int64 *)v19 >> 2) & 1) != 0 )
    {
      if ( !v5 )
        goto LABEL_121;
      v86 = v19;
    }
    else
    {
      v22 = *(_DWORD *)(v19 + 44);
      v23 = v21;
      if ( (v22 & 4) != 0 )
      {
        while ( 1 )
        {
          v86 = v23 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_BYTE *)(v86 + 44) & 4) == 0 )
            break;
          v23 = *(_QWORD *)v86;
        }
      }
      else
      {
        v86 = v20;
      }
      if ( !v5 )
        goto LABEL_105;
    }
    v24 = v17 & (((unsigned int)v86 >> 9) ^ ((unsigned int)v86 >> 4));
    v25 = *(_QWORD *)(v6 + 16LL * v24);
    if ( v86 == v25 )
      break;
    v93 = 1;
    while ( v25 != -4096 )
    {
      v24 = v17 & (v93 + v24);
      v25 = *(_QWORD *)(v6 + 16LL * v24);
      if ( v86 == v25 )
        goto LABEL_19;
      ++v93;
    }
    v14 = v20;
    if ( v95 )
      goto LABEL_109;
    v22 = *(_DWORD *)(v20 + 44);
LABEL_105:
    if ( (v22 & 4) == 0 )
    {
LABEL_121:
      v14 = v20;
      goto LABEL_109;
    }
    for ( i = v21; ; i = *(_QWORD *)v14 )
    {
      v14 = i & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_BYTE *)(v14 + 44) & 4) == 0 )
        break;
    }
LABEL_109:
    ++v99;
    if ( v14 == v18 )
      goto LABEL_122;
    v13 = *(_QWORD *)v14;
  }
LABEL_19:
  v26 = v101 + 48;
  if ( v16 == v101 + 48 )
  {
LABEL_75:
    v16 = v26;
    if ( v14 == v18 )
      goto LABEL_60;
    goto LABEL_24;
  }
LABEL_20:
  v27 = v5 - 1;
  while ( !v5 )
  {
LABEL_72:
    if ( !v16 )
      BUG();
    if ( (*(_BYTE *)v16 & 4) == 0 )
    {
      while ( (*(_BYTE *)(v16 + 44) & 8) != 0 )
        v16 = *(_QWORD *)(v16 + 8);
    }
    v16 = *(_QWORD *)(v16 + 8);
    ++v99;
    if ( v16 == v26 )
      goto LABEL_75;
  }
  LODWORD(v28) = v27 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
  v29 = (__int64 *)(v6 + 16LL * (unsigned int)v28);
  v30 = *v29;
  if ( v16 != *v29 )
  {
    v98 = *v29;
    v78 = 1;
    v94 = v27 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
    v90 = v30;
    while ( 1 )
    {
      if ( v98 == -4096 )
        goto LABEL_72;
      v79 = v78 + 1;
      v80 = v27 & (v94 + v78);
      v94 = v80;
      v98 = *(_QWORD *)(v6 + 16 * v80);
      if ( v98 == v16 )
        break;
      v78 = v79;
    }
    v96 = v5 - 1;
    v81 = v90;
    if ( v14 == v18 )
    {
      v37 = 0;
      v51 = 0;
LABEL_116:
      v82 = 1;
      while ( v81 != -4096 )
      {
        LODWORD(v28) = v96 & (v82 + v28);
        v88 = v82 + 1;
        v29 = (__int64 *)(v6 + 16LL * (unsigned int)v28);
        v81 = *v29;
        if ( *v29 == v16 )
          goto LABEL_58;
        v82 = v88;
      }
      v52 = v10[1];
      goto LABEL_59;
    }
LABEL_24:
    v31 = *(_QWORD *)v14 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v31 )
      BUG();
    if ( (*(_QWORD *)v31 & 4) == 0 && (*(_BYTE *)(v31 + 44) & 4) != 0 )
    {
      for ( j = *(_QWORD *)v31; ; j = *(_QWORD *)v31 )
      {
        v31 = j & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)(v31 + 44) & 4) == 0 )
          break;
      }
    }
    if ( !v5 )
    {
      v37 = v10[1];
      if ( v16 != v26 )
        goto LABEL_60;
      goto LABEL_33;
    }
    v33 = v5 - 1;
    v96 = v5 - 1;
    v34 = (v5 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
    v35 = (__int64 *)(v6 + 16LL * v34);
    v36 = *v35;
    if ( v31 == *v35 )
    {
LABEL_32:
      v37 = v35[1];
      if ( v16 == v26 )
      {
LABEL_33:
        v38 = 1;
        v39 = 1024;
        goto LABEL_34;
      }
      v51 = v35[1];
      LODWORD(v28) = (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4)) & v33;
      v29 = (__int64 *)(v6 + 16LL * (unsigned int)v28);
      v81 = *v29;
    }
    else
    {
      v85 = 1;
      while ( v36 != -4096 )
      {
        v89 = v85 + 1;
        v34 = v33 & (v85 + v34);
        v35 = (__int64 *)(v6 + 16LL * v34);
        v36 = *v35;
        if ( v31 == *v35 )
          goto LABEL_32;
        v85 = v89;
      }
      v37 = v10[1];
      if ( v16 == v26 )
        goto LABEL_33;
      v96 = v5 - 1;
      v51 = v10[1];
      v28 = v33 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v29 = (__int64 *)(v6 + 16 * v28);
      v81 = *v29;
    }
    if ( v16 == v81 )
      goto LABEL_58;
    goto LABEL_116;
  }
  if ( v14 != v18 )
    goto LABEL_24;
  v37 = 0;
  v51 = 0;
LABEL_58:
  v52 = v29[1];
LABEL_59:
  v53 = ((int)v52 - v51) / (unsigned int)(v99 + 1);
  v39 = v53;
  if ( !v53 )
  {
LABEL_60:
    v100 = a2;
    sub_2F45270(a1, v101);
    v54 = *(unsigned int *)(a1 + 40);
    v55 = *(_QWORD *)(a1 + 24);
    if ( (_DWORD)v54 )
    {
      v56 = (v54 - 1) & (((unsigned int)v100 >> 9) ^ ((unsigned int)v100 >> 4));
      v57 = (__int64 *)(v55 + 16LL * v56);
      v58 = *v57;
      if ( v100 == *v57 )
      {
LABEL_62:
        LODWORD(v10) = 1;
        *a3 = v57[1];
        return (unsigned int)v10;
      }
      v76 = 1;
      while ( v58 != -4096 )
      {
        v87 = v76 + 1;
        v56 = (v54 - 1) & (v76 + v56);
        v57 = (__int64 *)(v55 + 16LL * v56);
        v58 = *v57;
        if ( v100 == *v57 )
          goto LABEL_62;
        v76 = v87;
      }
    }
    v57 = (__int64 *)(v55 + 16 * v54);
    goto LABEL_62;
  }
  v26 = v16;
  v38 = v53 == 1024;
LABEL_34:
  LOBYTE(v10) = v38 && v37 == 0;
  if ( (_BYTE)v10 )
    goto LABEL_60;
  v102 = a1 + 16;
  if ( v14 == v26 )
    goto LABEL_46;
  v40 = v39;
  v97 = a2;
  while ( 2 )
  {
    v37 += v40;
    if ( !v5 )
    {
      ++*(_QWORD *)(a1 + 16);
      goto LABEL_81;
    }
    v41 = *(_QWORD *)(a1 + 24);
    v42 = 1;
    v43 = 0;
    v44 = (v5 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
    v45 = (unsigned __int64 *)(v41 + 16LL * v44);
    v46 = *v45;
    if ( *v45 == v14 )
      goto LABEL_41;
    while ( 2 )
    {
      if ( v46 == -4096 )
      {
        if ( !v43 )
          v43 = v45;
        v59 = *(_DWORD *)(a1 + 32);
        ++*(_QWORD *)(a1 + 16);
        v60 = v59 + 1;
        if ( 4 * v60 < 3 * v5 )
        {
          if ( v5 - *(_DWORD *)(a1 + 36) - v60 > v5 >> 3 )
          {
LABEL_69:
            *(_DWORD *)(a1 + 32) = v60;
            if ( *v43 != -4096 )
              --*(_DWORD *)(a1 + 36);
            *v43 = v14;
            v47 = v43 + 1;
            v43[1] = 0;
            goto LABEL_42;
          }
          v92 = v26;
          sub_2F45090(v102, v5);
          v68 = *(_DWORD *)(a1 + 40);
          if ( v68 )
          {
            v69 = v68 - 1;
            v70 = *(_QWORD *)(a1 + 24);
            v71 = 0;
            v72 = v69 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
            v26 = v92;
            v73 = 1;
            v60 = *(_DWORD *)(a1 + 32) + 1;
            v43 = (unsigned __int64 *)(v70 + 16LL * v72);
            v74 = *v43;
            if ( *v43 != v14 )
            {
              while ( v74 != -4096 )
              {
                if ( !v71 && v74 == -8192 )
                  v71 = v43;
                v72 = v69 & (v73 + v72);
                v43 = (unsigned __int64 *)(v70 + 16LL * v72);
                v74 = *v43;
                if ( *v43 == v14 )
                  goto LABEL_69;
                ++v73;
              }
              if ( v71 )
                v43 = v71;
            }
            goto LABEL_69;
          }
LABEL_163:
          ++*(_DWORD *)(a1 + 32);
          BUG();
        }
LABEL_81:
        v91 = v26;
        sub_2F45090(v102, 2 * v5);
        v61 = *(_DWORD *)(a1 + 40);
        if ( v61 )
        {
          v62 = v61 - 1;
          v63 = *(_QWORD *)(a1 + 24);
          v26 = v91;
          v64 = v62 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
          v60 = *(_DWORD *)(a1 + 32) + 1;
          v43 = (unsigned __int64 *)(v63 + 16LL * v64);
          v65 = *v43;
          if ( v14 != *v43 )
          {
            v66 = 1;
            v67 = 0;
            while ( v65 != -4096 )
            {
              if ( !v67 && v65 == -8192 )
                v67 = v43;
              v64 = v62 & (v66 + v64);
              v43 = (unsigned __int64 *)(v63 + 16LL * v64);
              v65 = *v43;
              if ( *v43 == v14 )
                goto LABEL_69;
              ++v66;
            }
            if ( v67 )
              v43 = v67;
          }
          goto LABEL_69;
        }
        goto LABEL_163;
      }
      if ( v43 || v46 != -8192 )
        v45 = v43;
      v44 = (v5 - 1) & (v42 + v44);
      v46 = *(_QWORD *)(v41 + 16LL * v44);
      if ( v46 != v14 )
      {
        ++v42;
        v43 = v45;
        v45 = (unsigned __int64 *)(v41 + 16LL * v44);
        continue;
      }
      break;
    }
    v45 = (unsigned __int64 *)(v41 + 16LL * v44);
LABEL_41:
    v47 = v45 + 1;
LABEL_42:
    *v47 = v37;
    if ( !v14 )
      BUG();
    if ( (*(_BYTE *)v14 & 4) != 0 )
    {
      v14 = *(_QWORD *)(v14 + 8);
      if ( v14 == v26 )
        goto LABEL_45;
LABEL_38:
      v5 = *(_DWORD *)(a1 + 40);
      continue;
    }
    break;
  }
  while ( (*(_BYTE *)(v14 + 44) & 8) != 0 )
    v14 = *(_QWORD *)(v14 + 8);
  v14 = *(_QWORD *)(v14 + 8);
  if ( v14 != v26 )
    goto LABEL_38;
LABEL_45:
  LODWORD(v10) = 0;
  a2 = v97;
  v6 = *(_QWORD *)(a1 + 24);
  v5 = *(_DWORD *)(a1 + 40);
LABEL_46:
  if ( v5 )
  {
    v48 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v49 = (__int64 *)(v6 + 16LL * v48);
    v50 = *v49;
    if ( a2 == *v49 )
      goto LABEL_48;
    v83 = 1;
    while ( v50 != -4096 )
    {
      v84 = v83 + 1;
      v48 = (v5 - 1) & (v83 + v48);
      v49 = (__int64 *)(v6 + 16LL * v48);
      v50 = *v49;
      if ( a2 == *v49 )
        goto LABEL_48;
      v83 = v84;
    }
  }
  v49 = (__int64 *)(v6 + 16LL * v5);
LABEL_48:
  *a3 = v49[1];
  return (unsigned int)v10;
}
