// Function: sub_311B700
// Address: 0x311b700
//
void __fastcall sub_311B700(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  unsigned __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // r15
  unsigned int v7; // esi
  __int64 v8; // r13
  int v9; // r8d
  int v10; // r14d
  __int64 v11; // r10
  _DWORD *v12; // rdx
  unsigned int j; // eax
  _DWORD *v14; // rcx
  int v15; // r9d
  _QWORD *v16; // rdx
  __int64 v17; // r15
  int v18; // ebx
  unsigned __int64 v19; // r13
  __int64 v20; // rax
  unsigned int v21; // esi
  __int64 v22; // rdx
  __int64 v23; // rdi
  _QWORD *v24; // r12
  __int64 v25; // r9
  unsigned int v26; // eax
  __int64 *v27; // rbx
  __int64 v28; // rcx
  __int64 v29; // rdx
  _QWORD *v30; // r12
  int v31; // eax
  unsigned __int64 *v32; // rdx
  int v33; // edx
  int v34; // esi
  __int64 v35; // r8
  unsigned int v36; // eax
  int v37; // ecx
  int v38; // r10d
  _QWORD *v39; // r9
  int v40; // r8d
  int v41; // esi
  int v42; // r8d
  int v43; // r10d
  __int64 v44; // rdi
  _DWORD *v45; // r9
  unsigned int i; // eax
  int v47; // r11d
  unsigned int v48; // eax
  unsigned __int64 v49; // r12
  unsigned int v50; // eax
  _QWORD *v51; // r14
  unsigned __int64 *v52; // rax
  unsigned __int64 v53; // rdi
  int v54; // r12d
  _QWORD *v55; // rax
  int v56; // eax
  int v57; // ecx
  int v58; // eax
  int v59; // r8d
  int v60; // esi
  int v61; // r8d
  int v62; // r10d
  __int64 v63; // rdi
  unsigned int k; // eax
  int v65; // r11d
  unsigned int v66; // eax
  int v67; // ecx
  int v68; // esi
  __int64 v69; // r8
  _QWORD *v70; // r9
  int v71; // r10d
  unsigned int v72; // eax
  __int64 v73; // rdi
  int v75; // [rsp+10h] [rbp-50h]
  int v76; // [rsp+14h] [rbp-4Ch]
  unsigned __int64 v78[7]; // [rsp+28h] [rbp-38h] BYREF

  v76 = sub_3118180(a1, (__int8 *)a2[1], a2[2]);
  v75 = sub_3118180(a1, (__int8 *)a2[5], a2[6]);
  v2 = sub_22077B0(0x20u);
  v3 = v2;
  if ( v2 )
  {
    *(_QWORD *)v2 = 0;
    *(_QWORD *)(v2 + 8) = 0;
    *(_QWORD *)(v2 + 16) = 0;
    *(_DWORD *)(v2 + 24) = 0;
  }
  v4 = a2[10];
  v5 = 16LL * *((unsigned int *)a2 + 22);
  v6 = v4 + v5;
  if ( v4 == v4 + v5 )
  {
    v17 = *a2;
    v18 = *((_DWORD *)a2 + 18);
    v19 = sub_22077B0(0x20u);
    if ( v19 )
    {
LABEL_17:
      *(_QWORD *)v19 = v17;
      *(_DWORD *)(v19 + 16) = v18;
      *(_DWORD *)(v19 + 8) = v76;
      *(_QWORD *)(v19 + 24) = v3;
      *(_DWORD *)(v19 + 12) = v75;
    }
    else if ( v3 )
    {
      goto LABEL_26;
    }
    v20 = a1;
    v21 = *(_DWORD *)(a1 + 24);
    if ( v21 )
      goto LABEL_19;
LABEL_27:
    ++*(_QWORD *)a1;
    goto LABEL_28;
  }
  do
  {
    v7 = *(_DWORD *)(v3 + 24);
    v8 = *(_QWORD *)(v4 + 8);
    if ( !v7 )
    {
      ++*(_QWORD *)v3;
LABEL_36:
      sub_311B460(v3, 2 * v7);
      v40 = *(_DWORD *)(v3 + 24);
      if ( v40 )
      {
        v41 = *(_DWORD *)(v4 + 4);
        v42 = v40 - 1;
        v43 = 1;
        v45 = 0;
        for ( i = v42
                & (((0xBF58476D1CE4E5B9LL
                   * ((unsigned int)(37 * v41) | ((unsigned __int64)(unsigned int)(37 * *(_DWORD *)v4) << 32))) >> 31)
                 ^ (756364221 * v41)); ; i = v42 & v48 )
        {
          v44 = *(_QWORD *)(v3 + 8);
          v12 = (_DWORD *)(v44 + 16LL * i);
          v47 = *v12;
          if ( *(_DWORD *)v4 == *v12 && v41 == v12[1] )
            break;
          if ( v47 == -1 )
          {
            if ( v12[1] == -1 )
            {
LABEL_92:
              if ( v45 )
                v12 = v45;
              v57 = *(_DWORD *)(v3 + 16) + 1;
              goto LABEL_75;
            }
          }
          else if ( v47 == -2 && v12[1] == -2 && !v45 )
          {
            v45 = (_DWORD *)(v44 + 16LL * i);
          }
          v48 = v43 + i;
          ++v43;
        }
        goto LABEL_90;
      }
LABEL_114:
      ++*(_DWORD *)(v3 + 16);
      BUG();
    }
    v9 = *(_DWORD *)(v4 + 4);
    v10 = 1;
    v11 = *(_QWORD *)(v3 + 8);
    v12 = 0;
    for ( j = (v7 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * ((unsigned int)(37 * v9) | ((unsigned __int64)(unsigned int)(37 * *(_DWORD *)v4) << 32))) >> 31)
             ^ (756364221 * v9)); ; j = (v7 - 1) & v50 )
    {
      v14 = (_DWORD *)(v11 + 16LL * j);
      v15 = *v14;
      if ( *(_DWORD *)v4 == *v14 && v9 == v14[1] )
      {
        v16 = v14 + 2;
        goto LABEL_15;
      }
      if ( v15 == -1 )
        break;
      if ( v15 == -2 && v14[1] == -2 && !v12 )
        v12 = (_DWORD *)(v11 + 16LL * j);
LABEL_52:
      v50 = v10 + j;
      ++v10;
    }
    if ( v14[1] != -1 )
      goto LABEL_52;
    v56 = *(_DWORD *)(v3 + 16);
    if ( !v12 )
      v12 = v14;
    ++*(_QWORD *)v3;
    v57 = v56 + 1;
    if ( 4 * (v56 + 1) >= 3 * v7 )
      goto LABEL_36;
    if ( v7 - *(_DWORD *)(v3 + 20) - v57 <= v7 >> 3 )
    {
      sub_311B460(v3, v7);
      v59 = *(_DWORD *)(v3 + 24);
      if ( v59 )
      {
        v60 = *(_DWORD *)(v4 + 4);
        v61 = v59 - 1;
        v62 = 1;
        v45 = 0;
        for ( k = v61
                & (((0xBF58476D1CE4E5B9LL
                   * ((unsigned int)(37 * v60) | ((unsigned __int64)(unsigned int)(37 * *(_DWORD *)v4) << 32))) >> 31)
                 ^ (756364221 * v60)); ; k = v61 & v66 )
        {
          v63 = *(_QWORD *)(v3 + 8);
          v12 = (_DWORD *)(v63 + 16LL * k);
          v65 = *v12;
          if ( *(_DWORD *)v4 == *v12 && v60 == v12[1] )
            break;
          if ( v65 == -1 )
          {
            if ( v12[1] == -1 )
              goto LABEL_92;
          }
          else if ( v65 == -2 && v12[1] == -2 && !v45 )
          {
            v45 = (_DWORD *)(v63 + 16LL * k);
          }
          v66 = v62 + k;
          ++v62;
        }
LABEL_90:
        v57 = *(_DWORD *)(v3 + 16) + 1;
        goto LABEL_75;
      }
      goto LABEL_114;
    }
LABEL_75:
    *(_DWORD *)(v3 + 16) = v57;
    if ( *v12 != -1 || v12[1] != -1 )
      --*(_DWORD *)(v3 + 20);
    v16 = v12 + 2;
    *((_DWORD *)v16 - 2) = *(_DWORD *)v4;
    v58 = *(_DWORD *)(v4 + 4);
    *v16 = 0;
    *((_DWORD *)v16 - 1) = v58;
LABEL_15:
    v4 += 16;
    *v16 = v8;
  }
  while ( v6 != v4 );
  v17 = *a2;
  v18 = *((_DWORD *)a2 + 18);
  v19 = sub_22077B0(0x20u);
  if ( v19 )
    goto LABEL_17;
LABEL_26:
  v19 = 0;
  sub_C7D6A0(*(_QWORD *)(v3 + 8), 16LL * *(unsigned int *)(v3 + 24), 8);
  j_j___libc_free_0(v3);
  v20 = a1;
  v21 = *(_DWORD *)(a1 + 24);
  if ( !v21 )
    goto LABEL_27;
LABEL_19:
  v22 = *(_QWORD *)v19;
  v23 = *(_QWORD *)(v20 + 8);
  v24 = 0;
  v25 = 1;
  v26 = (v21 - 1) & (((0xBF58476D1CE4E5B9LL * *(_QWORD *)v19) >> 31) ^ (484763065 * *(_DWORD *)v19));
  v27 = (__int64 *)(v23 + 72LL * v26);
  v28 = *v27;
  if ( *v27 == *(_QWORD *)v19 )
  {
LABEL_20:
    v29 = *((unsigned int *)v27 + 4);
    v30 = v27 + 1;
    v31 = v29;
    if ( *((_DWORD *)v27 + 5) > (unsigned int)v29 )
      goto LABEL_21;
    v51 = (_QWORD *)sub_C8D7D0((__int64)(v27 + 1), (__int64)(v27 + 3), 0, 8u, v78, v25);
    v52 = &v51[*((unsigned int *)v27 + 4)];
    if ( v52 )
    {
      *v52 = v19;
      v19 = 0;
    }
    sub_311AF30((__int64)(v27 + 1), v51);
    v53 = v27[1];
    v54 = v78[0];
    if ( v27 + 3 != (__int64 *)v53 )
      _libc_free(v53);
    ++*((_DWORD *)v27 + 4);
    v27[1] = (__int64)v51;
    *((_DWORD *)v27 + 5) = v54;
LABEL_47:
    if ( v19 )
    {
      v49 = *(_QWORD *)(v19 + 24);
      if ( v49 )
      {
        sub_C7D6A0(*(_QWORD *)(v49 + 8), 16LL * *(unsigned int *)(v49 + 24), 8);
        j_j___libc_free_0(v49);
      }
      j_j___libc_free_0(v19);
    }
    return;
  }
  while ( v28 != -1 )
  {
    if ( v28 == -2 && !v24 )
      v24 = v27;
    v26 = (v21 - 1) & (v25 + v26);
    v27 = (__int64 *)(v23 + 72LL * v26);
    v28 = *v27;
    if ( v22 == *v27 )
      goto LABEL_20;
    v25 = (unsigned int)(v25 + 1);
  }
  if ( !v24 )
    v24 = v27;
  ++*(_QWORD *)a1;
  v37 = *(_DWORD *)(a1 + 16) + 1;
  if ( 4 * v37 < 3 * v21 )
  {
    if ( v21 - *(_DWORD *)(a1 + 20) - v37 > v21 >> 3 )
      goto LABEL_68;
    sub_311AFF0(a1, v21);
    v67 = *(_DWORD *)(a1 + 24);
    if ( v67 )
    {
      v22 = *(_QWORD *)v19;
      v68 = v67 - 1;
      v69 = *(_QWORD *)(a1 + 8);
      v70 = 0;
      v71 = 1;
      v72 = (v67 - 1) & (((0xBF58476D1CE4E5B9LL * *(_QWORD *)v19) >> 31) ^ (484763065 * *(_DWORD *)v19));
      v24 = (_QWORD *)(v69 + 72LL * v72);
      v73 = *v24;
      v37 = *(_DWORD *)(a1 + 16) + 1;
      if ( *v24 != *(_QWORD *)v19 )
      {
        while ( v73 != -1 )
        {
          if ( !v70 && v73 == -2 )
            v70 = v24;
          v72 = v68 & (v71 + v72);
          v24 = (_QWORD *)(v69 + 72LL * v72);
          v73 = *v24;
          if ( v22 == *v24 )
            goto LABEL_68;
          ++v71;
        }
        if ( v70 )
          v24 = v70;
      }
      goto LABEL_68;
    }
LABEL_113:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_28:
  sub_311AFF0(a1, 2 * v21);
  v33 = *(_DWORD *)(a1 + 24);
  if ( !v33 )
    goto LABEL_113;
  v34 = v33 - 1;
  v35 = *(_QWORD *)(a1 + 8);
  v36 = (v33 - 1) & (((0xBF58476D1CE4E5B9LL * *(_QWORD *)v19) >> 31) ^ (484763065 * *(_DWORD *)v19));
  v37 = *(_DWORD *)(a1 + 16) + 1;
  v24 = (_QWORD *)(v35 + 72LL * v36);
  v22 = *v24;
  if ( *(_QWORD *)v19 != *v24 )
  {
    v38 = 1;
    v39 = 0;
    while ( v22 != -1 )
    {
      if ( v22 == -2 && !v39 )
        v39 = v24;
      v36 = v34 & (v38 + v36);
      v24 = (_QWORD *)(v35 + 72LL * v36);
      v22 = *v24;
      if ( *(_QWORD *)v19 == *v24 )
        goto LABEL_68;
      ++v38;
    }
    v22 = *(_QWORD *)v19;
    if ( v39 )
      v24 = v39;
  }
LABEL_68:
  *(_DWORD *)(a1 + 16) = v37;
  if ( *v24 != -1 )
    --*(_DWORD *)(a1 + 20);
  v55 = v24 + 3;
  *v24 = v22;
  v30 = v24 + 1;
  v29 = 0;
  *v30 = v55;
  v30[1] = 0x600000000LL;
  v31 = 0;
LABEL_21:
  v32 = (unsigned __int64 *)(*v30 + 8 * v29);
  if ( !v32 )
  {
    *((_DWORD *)v30 + 2) = v31 + 1;
    goto LABEL_47;
  }
  *v32 = v19;
  ++*((_DWORD *)v30 + 2);
}
