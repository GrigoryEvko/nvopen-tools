// Function: sub_3513A60
// Address: 0x3513a60
//
__int64 __fastcall sub_3513A60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 *v5; // r15
  __int64 v6; // rax
  __int64 *v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 *v10; // rdx
  __int64 *v11; // r11
  int v12; // r8d
  __int64 v13; // rsi
  __int64 *v14; // r10
  __int64 *v15; // rdi
  __int64 *v16; // rbx
  int v17; // r9d
  __int64 v18; // r13
  unsigned int v19; // ecx
  __int64 *v20; // rax
  __int64 v21; // r14
  __int64 v22; // r13
  unsigned int v23; // ecx
  __int64 *v24; // rax
  __int64 v25; // r14
  unsigned int v26; // ecx
  __int64 *v27; // rax
  __int64 v28; // r14
  __int64 v29; // r13
  unsigned int v30; // ecx
  __int64 *v31; // rax
  __int64 v32; // r14
  int v33; // eax
  int v34; // eax
  int v35; // eax
  int v36; // eax
  __int64 *v37; // rcx
  __int64 v38; // r9
  int v39; // r8d
  unsigned int v40; // edx
  __int64 *v41; // rax
  __int64 v42; // r11
  size_t v43; // rdx
  char *v44; // r13
  __int64 v45; // rax
  __int64 v46; // r13
  __int64 *v47; // r14
  __int64 v48; // r12
  int v49; // r8d
  __int64 v50; // r9
  __int64 *v51; // rdx
  unsigned int v52; // edi
  __int64 *v53; // rax
  __int64 v54; // rcx
  unsigned int v55; // esi
  int v56; // esi
  int v57; // esi
  __int64 v58; // r9
  unsigned int v59; // ecx
  int v60; // eax
  __int64 v61; // rdi
  unsigned __int64 v62; // rax
  int v65; // eax
  int v66; // ecx
  int v67; // ecx
  __int64 v68; // rdi
  __int64 *v69; // r11
  unsigned int v70; // r15d
  int v71; // r10d
  __int64 v72; // rsi
  int v73; // eax
  int v74; // r13d
  __int64 v75; // rcx
  unsigned int v76; // edx
  __int64 *v77; // rax
  __int64 v78; // r11
  __int64 v79; // rcx
  unsigned int v80; // edx
  __int64 *v81; // rax
  __int64 v82; // r11
  __int64 v83; // rcx
  unsigned int v84; // edx
  __int64 *v85; // rax
  __int64 v86; // r11
  int v87; // r15d
  __int64 *v88; // r10
  int v89; // eax
  int v90; // r10d
  __int64 v91; // rax
  int v92; // eax
  int v93; // r10d
  __int64 v94; // rax
  int v95; // eax
  int v96; // r10d
  __int64 v97; // rax
  __int64 v98; // [rsp+8h] [rbp-58h]
  int v99; // [rsp+14h] [rbp-4Ch]
  int v100; // [rsp+14h] [rbp-4Ch]
  int v101; // [rsp+14h] [rbp-4Ch]
  int v102; // [rsp+14h] [rbp-4Ch]
  char v103; // [rsp+14h] [rbp-4Ch]
  __int64 v105; // [rsp+18h] [rbp-48h]
  __int64 *v107; // [rsp+20h] [rbp-40h]
  __int64 *src; // [rsp+28h] [rbp-38h]
  void *srca; // [rsp+28h] [rbp-38h]

  v4 = a1;
  v5 = *(__int64 **)a3;
  v6 = 8LL * *(unsigned int *)(a3 + 8);
  v7 = (__int64 *)(*(_QWORD *)a3 + v6);
  v8 = v6 >> 3;
  v9 = v6 >> 5;
  src = v7;
  if ( !v9 )
  {
    v15 = v5;
LABEL_83:
    switch ( v8 )
    {
      case 2LL:
        v13 = *(_QWORD *)(v4 + 896);
        v12 = *(_DWORD *)(v4 + 912);
        break;
      case 3LL:
        v12 = *(_DWORD *)(v4 + 912);
        v79 = *v15;
        v13 = *(_QWORD *)(v4 + 896);
        if ( v12 )
        {
          v80 = (v12 - 1) & (((unsigned int)v79 >> 9) ^ ((unsigned int)v79 >> 4));
          v81 = (__int64 *)(v13 + 16LL * v80);
          v82 = *v81;
          if ( v79 == *v81 )
          {
LABEL_99:
            if ( a2 == v81[1] )
              goto LABEL_30;
          }
          else
          {
            v95 = 1;
            while ( v82 != -4096 )
            {
              v96 = v95 + 1;
              v97 = (v12 - 1) & (v80 + v95);
              v80 = v97;
              v81 = (__int64 *)(v13 + 16 * v97);
              v82 = *v81;
              if ( v79 == *v81 )
                goto LABEL_99;
              v95 = v96;
            }
          }
        }
        ++v15;
        break;
      case 1LL:
        v13 = *(_QWORD *)(v4 + 896);
        v12 = *(_DWORD *)(v4 + 912);
LABEL_93:
        v75 = *v15;
        if ( v12 )
        {
          v76 = (v12 - 1) & (((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4));
          v77 = (__int64 *)(v13 + 16LL * v76);
          v78 = *v77;
          if ( v75 == *v77 )
          {
LABEL_95:
            if ( a2 == v77[1] )
              goto LABEL_30;
          }
          else
          {
            v89 = 1;
            while ( v78 != -4096 )
            {
              v90 = v89 + 1;
              v91 = (v12 - 1) & (v76 + v89);
              v76 = v91;
              v77 = (__int64 *)(v13 + 16 * v91);
              v78 = *v77;
              if ( v75 == *v77 )
                goto LABEL_95;
              v89 = v90;
            }
          }
        }
        goto LABEL_86;
      default:
LABEL_86:
        v15 = src;
        goto LABEL_87;
    }
    v83 = *v15;
    if ( v12 )
    {
      v84 = (v12 - 1) & (((unsigned int)v83 >> 9) ^ ((unsigned int)v83 >> 4));
      v85 = (__int64 *)(v13 + 16LL * v84);
      v86 = *v85;
      if ( v83 == *v85 )
      {
LABEL_104:
        if ( a2 == v85[1] )
          goto LABEL_30;
      }
      else
      {
        v92 = 1;
        while ( v86 != -4096 )
        {
          v93 = v92 + 1;
          v94 = (v12 - 1) & (v84 + v92);
          v84 = v94;
          v85 = (__int64 *)(v13 + 16 * v94);
          v86 = *v85;
          if ( v83 == *v85 )
            goto LABEL_104;
          v92 = v93;
        }
      }
    }
    ++v15;
    goto LABEL_93;
  }
  v10 = v5 + 3;
  v11 = v5 + 2;
  v12 = *(_DWORD *)(a1 + 912);
  v13 = *(_QWORD *)(a1 + 896);
  v14 = v5 + 1;
  v15 = v5;
  v16 = &v5[4 * v9];
  v17 = v12 - 1;
  while ( 1 )
  {
    v29 = *(v10 - 3);
    if ( v12 )
    {
      v30 = v17 & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
      v31 = (__int64 *)(v13 + 16LL * v30);
      v32 = *v31;
      if ( v29 == *v31 )
      {
LABEL_3:
        if ( a2 == v31[1] )
        {
          v4 = a1;
          goto LABEL_30;
        }
      }
      else
      {
        v33 = 1;
        while ( v32 != -4096 )
        {
          v30 = v17 & (v33 + v30);
          v99 = v33 + 1;
          v31 = (__int64 *)(v13 + 16LL * v30);
          v32 = *v31;
          if ( v29 == *v31 )
            goto LABEL_3;
          v33 = v99;
        }
      }
      v18 = *(v10 - 2);
      v19 = v17 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v20 = (__int64 *)(v13 + 16LL * v19);
      v21 = *v20;
      if ( *v20 == v18 )
      {
LABEL_5:
        if ( a2 == v20[1] )
        {
          v4 = a1;
          v15 = v14;
          goto LABEL_30;
        }
      }
      else
      {
        v34 = 1;
        while ( v21 != -4096 )
        {
          v19 = v17 & (v34 + v19);
          v100 = v34 + 1;
          v20 = (__int64 *)(v13 + 16LL * v19);
          v21 = *v20;
          if ( *v20 == v18 )
            goto LABEL_5;
          v34 = v100;
        }
      }
      v22 = *(v10 - 1);
      v23 = v17 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v24 = (__int64 *)(v13 + 16LL * v23);
      v25 = *v24;
      if ( *v24 == v22 )
      {
LABEL_7:
        if ( a2 == v24[1] )
        {
          v4 = a1;
          v15 = v11;
          goto LABEL_30;
        }
      }
      else
      {
        v35 = 1;
        while ( v25 != -4096 )
        {
          v23 = v17 & (v35 + v23);
          v101 = v35 + 1;
          v24 = (__int64 *)(v13 + 16LL * v23);
          v25 = *v24;
          if ( *v24 == v22 )
            goto LABEL_7;
          v35 = v101;
        }
      }
      v26 = v17 & (((unsigned int)*v10 >> 9) ^ ((unsigned int)*v10 >> 4));
      v27 = (__int64 *)(v13 + 16LL * v26);
      v28 = *v27;
      if ( *v27 != *v10 )
      {
        v36 = 1;
        while ( v28 != -4096 )
        {
          v26 = v17 & (v36 + v26);
          v102 = v36 + 1;
          v27 = (__int64 *)(v13 + 16LL * v26);
          v28 = *v27;
          if ( *v27 == *v10 )
            goto LABEL_9;
          v36 = v102;
        }
        goto LABEL_10;
      }
LABEL_9:
      if ( a2 == v27[1] )
        break;
    }
LABEL_10:
    v15 += 4;
    v10 += 4;
    v11 += 4;
    v14 += 4;
    if ( v16 == v15 )
    {
      v4 = a1;
      v8 = src - v15;
      goto LABEL_83;
    }
  }
  v4 = a1;
  v15 = v10;
LABEL_30:
  if ( src == v15 || (v37 = v15 + 1, src == v15 + 1) )
  {
LABEL_87:
    v44 = (char *)v15;
    goto LABEL_41;
  }
  while ( 2 )
  {
    v38 = *v37;
    if ( !v12 )
      goto LABEL_38;
    v39 = v12 - 1;
    v40 = v39 & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
    v41 = (__int64 *)(v13 + 16LL * v40);
    v42 = *v41;
    if ( v38 == *v41 )
    {
LABEL_37:
      if ( a2 != v41[1] )
        goto LABEL_38;
      if ( src == ++v37 )
        break;
      goto LABEL_34;
    }
    v73 = 1;
    while ( v42 != -4096 )
    {
      v74 = v73 + 1;
      v40 = v39 & (v73 + v40);
      v41 = (__int64 *)(v13 + 16LL * v40);
      v42 = *v41;
      if ( v38 == *v41 )
        goto LABEL_37;
      v73 = v74;
    }
LABEL_38:
    ++v37;
    *v15++ = v38;
    if ( src != v37 )
    {
LABEL_34:
      v13 = *(_QWORD *)(v4 + 896);
      v12 = *(_DWORD *)(v4 + 912);
      continue;
    }
    break;
  }
  v5 = *(__int64 **)a3;
  v43 = *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8) - (_QWORD)src;
  v44 = (char *)v15 + v43;
  if ( src != (__int64 *)(*(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8)) )
  {
    memmove(v15, src, v43);
    v5 = *(__int64 **)a3;
  }
LABEL_41:
  v45 = (v44 - (char *)v5) >> 3;
  *(_DWORD *)(a3 + 8) = v45;
  if ( !(_DWORD)v45 )
    return 0;
  v46 = *v5;
  v105 = a2;
  srca = 0;
  v47 = v5 + 1;
  v48 = 0;
  v107 = &v5[(unsigned int)v45];
  v98 = v4 + 888;
  v103 = *(_BYTE *)(*v5 + 216);
  while ( 2 )
  {
    v55 = *(_DWORD *)(v4 + 912);
    if ( !v55 )
    {
      ++*(_QWORD *)(v4 + 888);
      goto LABEL_49;
    }
    v49 = 1;
    v50 = *(_QWORD *)(v4 + 896);
    v51 = 0;
    v52 = (v55 - 1) & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
    v53 = (__int64 *)(v50 + 16LL * v52);
    v54 = *v53;
    if ( v46 == *v53 )
    {
LABEL_44:
      if ( v105 == v53[1] )
      {
LABEL_45:
        if ( v107 == v47 )
          return v48;
        goto LABEL_46;
      }
    }
    else
    {
      while ( v54 != -4096 )
      {
        if ( v54 == -8192 && !v51 )
          v51 = v53;
        v52 = (v55 - 1) & (v49 + v52);
        v53 = (__int64 *)(v50 + 16LL * v52);
        v54 = *v53;
        if ( v46 == *v53 )
          goto LABEL_44;
        ++v49;
      }
      if ( !v51 )
        v51 = v53;
      v65 = *(_DWORD *)(v4 + 904);
      ++*(_QWORD *)(v4 + 888);
      v60 = v65 + 1;
      if ( 4 * v60 >= 3 * v55 )
      {
LABEL_49:
        sub_3512300(v98, 2 * v55);
        v56 = *(_DWORD *)(v4 + 912);
        if ( !v56 )
          goto LABEL_135;
        v57 = v56 - 1;
        v58 = *(_QWORD *)(v4 + 896);
        v59 = v57 & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
        v60 = *(_DWORD *)(v4 + 904) + 1;
        v51 = (__int64 *)(v58 + 16LL * v59);
        v61 = *v51;
        if ( *v51 != v46 )
        {
          v87 = 1;
          v88 = 0;
          while ( v61 != -4096 )
          {
            if ( v61 != -8192 || v88 )
              v51 = v88;
            v59 = v57 & (v87 + v59);
            v61 = *(_QWORD *)(v58 + 16LL * v59);
            if ( v46 == v61 )
            {
              v51 = (__int64 *)(v58 + 16LL * v59);
              goto LABEL_51;
            }
            v88 = v51;
            ++v87;
            v51 = (__int64 *)(v58 + 16LL * v59);
          }
          if ( v88 )
            v51 = v88;
        }
      }
      else if ( v55 - *(_DWORD *)(v4 + 908) - v60 <= v55 >> 3 )
      {
        sub_3512300(v98, v55);
        v66 = *(_DWORD *)(v4 + 912);
        if ( !v66 )
        {
LABEL_135:
          ++*(_DWORD *)(v4 + 904);
          BUG();
        }
        v67 = v66 - 1;
        v68 = *(_QWORD *)(v4 + 896);
        v69 = 0;
        v70 = v67 & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
        v71 = 1;
        v60 = *(_DWORD *)(v4 + 904) + 1;
        v51 = (__int64 *)(v68 + 16LL * v70);
        v72 = *v51;
        if ( v46 != *v51 )
        {
          while ( v72 != -4096 )
          {
            if ( !v69 && v72 == -8192 )
              v69 = v51;
            v70 = v67 & (v71 + v70);
            v51 = (__int64 *)(v68 + 16LL * v70);
            v72 = *v51;
            if ( v46 == *v51 )
              goto LABEL_51;
            ++v71;
          }
          if ( v69 )
            v51 = v69;
        }
      }
LABEL_51:
      *(_DWORD *)(v4 + 904) = v60;
      if ( *v51 != -4096 )
        --*(_DWORD *)(v4 + 908);
      *v51 = v46;
      v51[1] = 0;
    }
    v62 = sub_2F06CB0(*(_QWORD *)(v4 + 536), v46);
    if ( !v48 )
    {
      srca = (void *)v62;
      v48 = v46;
      goto LABEL_45;
    }
    if ( v103 == v62 <= (unsigned __int64)srca )
      v48 = v46;
    else
      v62 = (unsigned __int64)srca;
    srca = (void *)v62;
    if ( v107 != v47 )
    {
LABEL_46:
      v46 = *v47++;
      continue;
    }
    return v48;
  }
}
