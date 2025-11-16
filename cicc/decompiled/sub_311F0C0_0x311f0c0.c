// Function: sub_311F0C0
// Address: 0x311f0c0
//
void __fastcall sub_311F0C0(__int64 *a1, char **a2)
{
  int v2; // ebx
  char *v3; // r12
  int v5; // r13d
  size_t v6; // rdx
  __int64 v7; // rax
  char *v8; // rax
  int v9; // ecx
  char *v10; // rax
  __int64 v11; // rcx
  char *v12; // rdx
  char *v13; // rax
  __int64 v14; // rax
  unsigned __int64 v15; // r12
  int i; // r11d
  char *v17; // rax
  unsigned int v18; // esi
  int v19; // r13d
  char *v20; // rdx
  int v21; // r14d
  __int64 v22; // r10
  __int64 v23; // r8
  int *v24; // rdx
  int v25; // ebx
  unsigned int k; // edi
  int *v27; // rax
  int v28; // ecx
  unsigned __int64 v29; // r13
  __int64 v30; // rbx
  unsigned int v31; // esi
  __int64 v32; // r9
  __int64 v33; // r8
  _QWORD *v34; // rdi
  int v35; // r10d
  unsigned int v36; // eax
  __int64 *v37; // r12
  __int64 v38; // rcx
  __int64 v39; // rdx
  _QWORD *v40; // r14
  int v41; // eax
  unsigned __int64 *v42; // rdx
  int v43; // ecx
  int v44; // ecx
  __int64 v45; // r9
  __int64 v46; // rdx
  __int64 v47; // rsi
  int v48; // eax
  int v49; // r12d
  _QWORD *v50; // r11
  unsigned __int64 v51; // r12
  int v52; // edx
  __int64 v53; // rcx
  int v54; // r8d
  int v55; // esi
  int *v56; // rdi
  unsigned int j; // eax
  int v58; // r9d
  unsigned int v59; // eax
  unsigned int v60; // edi
  _QWORD *v61; // rax
  unsigned __int64 *v62; // rdx
  unsigned __int64 v63; // rdi
  int v64; // r14d
  _QWORD *v65; // rax
  int v66; // eax
  int v67; // eax
  int v68; // ecx
  int v69; // eax
  int v70; // eax
  __int64 v71; // rcx
  int *v72; // rsi
  unsigned int v73; // ebx
  int m; // edi
  int v75; // r8d
  unsigned int v76; // ebx
  int v77; // ecx
  int v78; // ecx
  __int64 v79; // r9
  int v80; // r12d
  __int64 v81; // rdx
  __int64 v82; // rsi
  int v83; // [rsp+8h] [rbp-78h]
  int v84; // [rsp+8h] [rbp-78h]
  int v85; // [rsp+Ch] [rbp-74h]
  __int64 v87; // [rsp+18h] [rbp-68h]
  int v88; // [rsp+20h] [rbp-60h]
  int v89; // [rsp+24h] [rbp-5Ch]
  int v90; // [rsp+28h] [rbp-58h]
  int v91; // [rsp+2Ch] [rbp-54h]
  int v92; // [rsp+30h] [rbp-50h]
  __int64 v93; // [rsp+30h] [rbp-50h]
  __int64 v94; // [rsp+30h] [rbp-50h]
  int v95; // [rsp+38h] [rbp-48h]
  _QWORD *v96; // [rsp+38h] [rbp-48h]
  unsigned __int64 v97[7]; // [rsp+48h] [rbp-38h] BYREF

  v2 = *(_DWORD *)*a2;
  v3 = *a2 + 4;
  *a2 = v3;
  if ( !v2 )
    return;
  v5 = 0;
  while ( 1 )
  {
    if ( v3 )
    {
      v6 = strlen(v3);
      v7 = v6 + 1;
    }
    else
    {
      v7 = 1;
      v6 = 0;
    }
    ++v5;
    *a2 = &v3[v7];
    sub_3118180(*a1, v3, v6);
    if ( v5 == v2 )
      break;
    v3 = *a2;
  }
  v8 = (char *)((unsigned __int64)(*a2 + 3) & 0xFFFFFFFFFFFFFFFCLL);
  *a2 = v8;
  v9 = *(_DWORD *)v8;
  v10 = v8 + 4;
  *a2 = v10;
  v85 = v9;
  if ( !v9 )
    return;
  v91 = 0;
  while ( 2 )
  {
    v11 = *(_QWORD *)v10;
    *a2 = v10 + 8;
    v87 = v11;
    LODWORD(v11) = *((_DWORD *)v10 + 2);
    *a2 = v10 + 12;
    v88 = v11;
    LODWORD(v11) = *((_DWORD *)v10 + 3);
    *a2 = v10 + 16;
    v12 = v10 + 20;
    v13 = v10 + 24;
    v89 = v11;
    LODWORD(v11) = *((_DWORD *)v13 - 2);
    *a2 = v12;
    v90 = v11;
    LODWORD(v11) = *((_DWORD *)v13 - 1);
    *a2 = v13;
    v95 = v11;
    v14 = sub_22077B0(0x20u);
    v15 = v14;
    if ( v14 )
    {
      *(_QWORD *)v14 = 0;
      *(_QWORD *)(v14 + 8) = 0;
      *(_QWORD *)(v14 + 16) = 0;
      *(_DWORD *)(v14 + 24) = 0;
    }
    if ( !v95 )
    {
      v29 = sub_22077B0(0x20u);
      if ( v29 )
      {
LABEL_27:
        *(_QWORD *)(v29 + 24) = v15;
        *(_QWORD *)v29 = v87;
        *(_DWORD *)(v29 + 8) = v88;
        *(_DWORD *)(v29 + 12) = v89;
        *(_DWORD *)(v29 + 16) = v90;
      }
      else if ( v15 )
      {
        goto LABEL_37;
      }
      v30 = *a1;
      v31 = *(_DWORD *)(*a1 + 24);
      if ( v31 )
        goto LABEL_29;
LABEL_38:
      ++*(_QWORD *)v30;
      goto LABEL_39;
    }
    for ( i = 0; i != v95; ++i )
    {
LABEL_15:
      v17 = *a2;
      v18 = *(_DWORD *)(v15 + 24);
      v19 = *(_DWORD *)*a2;
      *a2 += 4;
      v20 = v17 + 8;
      v21 = *((_DWORD *)v17 + 1);
      v17 += 16;
      *a2 = v20;
      v22 = *((_QWORD *)v17 - 1);
      *a2 = v17;
      if ( !v18 )
      {
        ++*(_QWORD *)v15;
LABEL_52:
        v83 = i;
        v93 = v22;
        sub_311B460(v15, 2 * v18);
        v52 = *(_DWORD *)(v15 + 24);
        if ( v52 )
        {
          i = v83;
          v22 = v93;
          v54 = 1;
          v55 = v52 - 1;
          v56 = 0;
          for ( j = (v52 - 1)
                  & (((0xBF58476D1CE4E5B9LL
                     * ((unsigned int)(37 * v21) | ((unsigned __int64)(unsigned int)(37 * v19) << 32))) >> 31)
                   ^ (756364221 * v21)); ; j = v55 & v59 )
          {
            v53 = *(_QWORD *)(v15 + 8);
            v24 = (int *)(v53 + 16LL * j);
            v58 = *v24;
            if ( v19 == *v24 && v21 == v24[1] )
              break;
            if ( v58 == -1 )
            {
              if ( v24[1] == -1 )
              {
                if ( v56 )
                  v24 = v56;
                v68 = *(_DWORD *)(v15 + 16) + 1;
                goto LABEL_85;
              }
            }
            else if ( v58 == -2 && v24[1] == -2 && !v56 )
            {
              v56 = (int *)(v53 + 16LL * j);
            }
            v59 = v54 + j;
            ++v54;
          }
          goto LABEL_101;
        }
LABEL_125:
        ++*(_DWORD *)(v15 + 16);
        BUG();
      }
      v23 = *(_QWORD *)(v15 + 8);
      v24 = 0;
      v92 = 1;
      v25 = ((0xBF58476D1CE4E5B9LL * ((unsigned int)(37 * v21) | ((unsigned __int64)(unsigned int)(37 * v19) << 32))) >> 31)
          ^ (756364221 * v21);
      for ( k = v25 & (v18 - 1); ; k = (v18 - 1) & v60 )
      {
        v27 = (int *)(v23 + 16LL * k);
        v28 = *v27;
        if ( v19 == *v27 && v21 == v27[1] )
        {
          if ( ++i == v95 )
            goto LABEL_26;
          goto LABEL_15;
        }
        if ( v28 == -1 )
          break;
        if ( v28 == -2 && v27[1] == -2 && !v24 )
          v24 = (int *)(v23 + 16LL * k);
LABEL_62:
        v60 = v92 + k;
        ++v92;
      }
      if ( v27[1] != -1 )
        goto LABEL_62;
      if ( !v24 )
        v24 = (int *)(v23 + 16LL * k);
      v67 = *(_DWORD *)(v15 + 16);
      ++*(_QWORD *)v15;
      v68 = v67 + 1;
      if ( 4 * (v67 + 1) >= 3 * v18 )
        goto LABEL_52;
      if ( v18 - *(_DWORD *)(v15 + 20) - v68 <= v18 >> 3 )
      {
        v84 = i;
        v94 = v22;
        sub_311B460(v15, v18);
        v69 = *(_DWORD *)(v15 + 24);
        if ( v69 )
        {
          v70 = v69 - 1;
          i = v84;
          v72 = 0;
          v22 = v94;
          v73 = v70 & v25;
          for ( m = 1; ; ++m )
          {
            v71 = *(_QWORD *)(v15 + 8);
            v24 = (int *)(v71 + 16LL * v73);
            v75 = *v24;
            if ( v19 == *v24 && v21 == v24[1] )
              break;
            if ( v75 == -1 )
            {
              if ( v24[1] == -1 )
              {
                if ( v72 )
                  v24 = v72;
                v68 = *(_DWORD *)(v15 + 16) + 1;
                goto LABEL_85;
              }
            }
            else if ( v75 == -2 && v24[1] == -2 && !v72 )
            {
              v72 = (int *)(v71 + 16LL * v73);
            }
            v76 = m + v73;
            v73 = v70 & v76;
          }
LABEL_101:
          v68 = *(_DWORD *)(v15 + 16) + 1;
          goto LABEL_85;
        }
        goto LABEL_125;
      }
LABEL_85:
      *(_DWORD *)(v15 + 16) = v68;
      if ( *v24 != -1 || v24[1] != -1 )
        --*(_DWORD *)(v15 + 20);
      *v24 = v19;
      v24[1] = v21;
      *((_QWORD *)v24 + 1) = v22;
    }
LABEL_26:
    v29 = sub_22077B0(0x20u);
    if ( v29 )
      goto LABEL_27;
LABEL_37:
    v29 = 0;
    sub_C7D6A0(*(_QWORD *)(v15 + 8), 16LL * *(unsigned int *)(v15 + 24), 8);
    j_j___libc_free_0(v15);
    v30 = *a1;
    v31 = *(_DWORD *)(*a1 + 24);
    if ( !v31 )
      goto LABEL_38;
LABEL_29:
    v32 = v31 - 1;
    v33 = *(_QWORD *)(v30 + 8);
    v34 = 0;
    v35 = 1;
    v36 = v32 & (((0xBF58476D1CE4E5B9LL * *(_QWORD *)v29) >> 31) ^ (484763065 * *(_DWORD *)v29));
    v37 = (__int64 *)(v33 + 72LL * v36);
    v38 = *v37;
    if ( *v37 == *(_QWORD *)v29 )
    {
LABEL_30:
      v39 = *((unsigned int *)v37 + 4);
      v40 = v37 + 1;
      v41 = v39;
      if ( (unsigned int)v39 < *((_DWORD *)v37 + 5) )
        goto LABEL_31;
      v61 = (_QWORD *)sub_C8D7D0((__int64)(v37 + 1), (__int64)(v37 + 3), 0, 8u, v97, v32);
      v62 = &v61[*((unsigned int *)v37 + 4)];
      if ( v62 )
      {
        *v62 = v29;
        v29 = 0;
      }
      v96 = v61;
      sub_311AF30((__int64)(v37 + 1), v61);
      v63 = v37[1];
      v64 = v97[0];
      v65 = v96;
      if ( v37 + 3 != (__int64 *)v63 )
      {
        _libc_free(v63);
        v65 = v96;
      }
      ++*((_DWORD *)v37 + 4);
      v37[1] = (__int64)v65;
      *((_DWORD *)v37 + 5) = v64;
LABEL_47:
      if ( v29 )
      {
        v51 = *(_QWORD *)(v29 + 24);
        if ( v51 )
        {
          sub_C7D6A0(*(_QWORD *)(v51 + 8), 16LL * *(unsigned int *)(v51 + 24), 8);
          j_j___libc_free_0(v51);
        }
        j_j___libc_free_0(v29);
      }
      goto LABEL_33;
    }
    while ( v38 != -1 )
    {
      if ( v38 == -2 && !v34 )
        v34 = v37;
      v36 = v32 & (v35 + v36);
      v37 = (__int64 *)(v33 + 72LL * v36);
      v38 = *v37;
      if ( *(_QWORD *)v29 == *v37 )
        goto LABEL_30;
      ++v35;
    }
    v66 = *(_DWORD *)(v30 + 16);
    if ( !v34 )
      v34 = v37;
    ++*(_QWORD *)v30;
    v48 = v66 + 1;
    if ( 4 * v48 < 3 * v31 )
    {
      if ( v31 - *(_DWORD *)(v30 + 20) - v48 > v31 >> 3 )
        goto LABEL_78;
      sub_311AFF0(v30, v31);
      v77 = *(_DWORD *)(v30 + 24);
      if ( v77 )
      {
        v78 = v77 - 1;
        v79 = *(_QWORD *)(v30 + 8);
        v50 = 0;
        v80 = 1;
        LODWORD(v81) = v78 & (((0xBF58476D1CE4E5B9LL * *(_QWORD *)v29) >> 31) ^ (484763065 * *(_DWORD *)v29));
        v34 = (_QWORD *)(v79 + 72LL * (unsigned int)v81);
        v82 = *v34;
        v48 = *(_DWORD *)(v30 + 16) + 1;
        if ( *v34 != *(_QWORD *)v29 )
        {
          while ( v82 != -1 )
          {
            if ( !v50 && v82 == -2 )
              v50 = v34;
            v81 = v78 & (unsigned int)(v81 + v80);
            v34 = (_QWORD *)(v79 + 72 * v81);
            v82 = *v34;
            if ( *(_QWORD *)v29 == *v34 )
              goto LABEL_78;
            ++v80;
          }
          goto LABEL_43;
        }
        goto LABEL_78;
      }
LABEL_126:
      ++*(_DWORD *)(v30 + 16);
      BUG();
    }
LABEL_39:
    sub_311AFF0(v30, 2 * v31);
    v43 = *(_DWORD *)(v30 + 24);
    if ( !v43 )
      goto LABEL_126;
    v44 = v43 - 1;
    v45 = *(_QWORD *)(v30 + 8);
    LODWORD(v46) = v44 & (((0xBF58476D1CE4E5B9LL * *(_QWORD *)v29) >> 31) ^ (484763065 * *(_DWORD *)v29));
    v34 = (_QWORD *)(v45 + 72LL * (unsigned int)v46);
    v47 = *v34;
    v48 = *(_DWORD *)(v30 + 16) + 1;
    if ( *(_QWORD *)v29 == *v34 )
      goto LABEL_78;
    v49 = 1;
    v50 = 0;
    while ( v47 != -1 )
    {
      if ( v47 == -2 && !v50 )
        v50 = v34;
      v46 = v44 & (unsigned int)(v46 + v49);
      v34 = (_QWORD *)(v45 + 72 * v46);
      v47 = *v34;
      if ( *(_QWORD *)v29 == *v34 )
        goto LABEL_78;
      ++v49;
    }
LABEL_43:
    if ( v50 )
      v34 = v50;
LABEL_78:
    *(_DWORD *)(v30 + 16) = v48;
    if ( *v34 != -1 )
      --*(_DWORD *)(v30 + 20);
    v40 = v34 + 1;
    v39 = 0;
    *v34 = *(_QWORD *)v29;
    v34[1] = v34 + 3;
    v34[2] = 0x600000000LL;
    v41 = 0;
LABEL_31:
    v42 = (unsigned __int64 *)(*v40 + 8 * v39);
    if ( !v42 )
    {
      *((_DWORD *)v40 + 2) = v41 + 1;
      goto LABEL_47;
    }
    *v42 = v29;
    ++*((_DWORD *)v40 + 2);
LABEL_33:
    if ( ++v91 != v85 )
    {
      v10 = *a2;
      continue;
    }
    break;
  }
}
