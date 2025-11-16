// Function: sub_29E6420
// Address: 0x29e6420
//
void __fastcall sub_29E6420(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD *v6; // r9
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // rax
  __int64 v9; // rdx
  _QWORD *v10; // rax
  _QWORD *i; // rdx
  _QWORD *v12; // r14
  __int64 v13; // r13
  unsigned int *v14; // rcx
  unsigned int v15; // edi
  __int64 v16; // r8
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rdx
  _QWORD *v20; // rax
  unsigned __int64 v21; // rdi
  _QWORD *v22; // r12
  __int64 v23; // rsi
  __int64 v24; // rdx
  __int64 v25; // rdx
  int v26; // eax
  __int64 v27; // rcx
  __int64 v28; // r15
  _QWORD *v29; // r12
  __int64 v30; // rax
  __int64 v31; // rcx
  __int64 v32; // rsi
  _QWORD *v33; // rax
  __int64 v34; // rdi
  __int64 v35; // rdx
  __int64 v36; // r14
  __int64 v37; // rdx
  __int64 v38; // rax
  int v39; // edi
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rdx
  char v43; // di
  __int64 v44; // r12
  __int64 v45; // rax
  __int64 v46; // rcx
  int *v47; // rdi
  int *v48; // r14
  unsigned int v49; // edx
  __int64 v50; // r15
  bool v51; // al
  __int64 j; // rcx
  __int64 v53; // rdi
  __int64 v54; // rsi
  int v55; // [rsp+18h] [rbp-48h]
  _QWORD *v57; // [rsp+28h] [rbp-38h]
  _QWORD *v58; // [rsp+28h] [rbp-38h]
  _QWORD *v59; // [rsp+28h] [rbp-38h]

  v6 = *(_QWORD **)a1;
  v7 = **(unsigned int **)(a1 + 8);
  v8 = *(unsigned int *)(a2 + 32);
  if ( v7 != v8 )
  {
    if ( v7 >= v8 )
    {
      if ( v7 > *(unsigned int *)(a2 + 36) )
      {
        v59 = *(_QWORD **)a1;
        sub_C8D5F0(a2 + 24, (const void *)(a2 + 40), v7, 8u, a5, (__int64)v6);
        v8 = *(unsigned int *)(a2 + 32);
        v6 = v59;
      }
      v9 = *(_QWORD *)(a2 + 24);
      v10 = (_QWORD *)(v9 + 8 * v8);
      for ( i = (_QWORD *)(v9 + 8 * v7); i != v10; ++v10 )
      {
        if ( v10 )
          *v10 = 0;
      }
    }
    *(_DWORD *)(a2 + 32) = v7;
  }
  v12 = *(_QWORD **)(a2 + 184);
  v13 = a2 + 176;
  v14 = *(unsigned int **)(a1 + 16);
  if ( !v12 )
    return;
  v15 = *v14;
  v16 = a2 + 176;
  v17 = *(_QWORD *)(a2 + 184);
  do
  {
    while ( 1 )
    {
      v18 = *(_QWORD *)(v17 + 16);
      v19 = *(_QWORD *)(v17 + 24);
      if ( *(_DWORD *)(v17 + 32) >= v15 )
        break;
      v17 = *(_QWORD *)(v17 + 24);
      if ( !v19 )
        goto LABEL_15;
    }
    v16 = v17;
    v17 = *(_QWORD *)(v17 + 16);
  }
  while ( v18 );
LABEL_15:
  if ( v13 == v16 )
    return;
  if ( v15 < *(_DWORD *)(v16 + 32) )
    return;
  v20 = *(_QWORD **)(v16 + 56);
  if ( !v20 )
    return;
  v21 = **(_QWORD **)(a1 + 24);
  v22 = (_QWORD *)(v16 + 48);
  do
  {
    while ( 1 )
    {
      v23 = v20[2];
      v24 = v20[3];
      if ( v20[4] >= v21 )
        break;
      v20 = (_QWORD *)v20[3];
      if ( !v24 )
        goto LABEL_22;
    }
    v22 = v20;
    v20 = (_QWORD *)v20[2];
  }
  while ( v23 );
LABEL_22:
  if ( v22 == (_QWORD *)(v16 + 48) || v21 < v22[4] )
    return;
  v25 = 0;
  v26 = 0;
  if ( !*((_DWORD *)v22 + 18) )
  {
    v28 = v22[29];
    v29 = v22 + 27;
    if ( v29 == (_QWORD *)v28 )
      goto LABEL_57;
    goto LABEL_31;
  }
  do
  {
    v27 = *(_QWORD *)(*v6 + 8 * v25);
    if ( v27 >= 0 )
      *(_QWORD *)(*(_QWORD *)(a2 + 24) + 8 * v27) = *(_QWORD *)(v22[8] + 8 * v25);
    v25 = (unsigned int)(v26 + 1);
    v26 = v25;
  }
  while ( *((_DWORD *)v22 + 18) > (unsigned int)v25 );
  v28 = v22[29];
  v29 = v22 + 27;
  v12 = *(_QWORD **)(a2 + 184);
  if ( (_QWORD *)v28 != v29 )
  {
    do
    {
LABEL_31:
      v31 = *(_QWORD *)(v6[3] + 8LL * *(unsigned int *)(v28 + 32));
      if ( v31 >= 0 )
      {
        v32 = a2 + 176;
        if ( !v12 )
          goto LABEL_39;
        v33 = v12;
        do
        {
          while ( 1 )
          {
            v34 = v33[2];
            v35 = v33[3];
            if ( (unsigned int)v31 <= *((_DWORD *)v33 + 8) )
              break;
            v33 = (_QWORD *)v33[3];
            if ( !v35 )
              goto LABEL_37;
          }
          v32 = (__int64)v33;
          v33 = (_QWORD *)v33[2];
        }
        while ( v34 );
LABEL_37:
        if ( v13 == v32 || (unsigned int)v31 < *(_DWORD *)(v32 + 32) )
        {
LABEL_39:
          v55 = *(_QWORD *)(v6[3] + 8LL * *(unsigned int *)(v28 + 32));
          v58 = v6;
          v36 = sub_22077B0(0x58u);
          v37 = v36 + 48;
          *(_DWORD *)(v36 + 32) = v55;
          v38 = *(_QWORD *)(v28 + 56);
          if ( v38 )
          {
            v39 = *(_DWORD *)(v28 + 48);
            *(_QWORD *)(v36 + 56) = v38;
            *(_DWORD *)(v36 + 48) = v39;
            *(_QWORD *)(v36 + 64) = *(_QWORD *)(v28 + 64);
            *(_QWORD *)(v36 + 72) = *(_QWORD *)(v28 + 72);
            *(_QWORD *)(v38 + 8) = v37;
            v40 = *(_QWORD *)(v28 + 80);
            *(_QWORD *)(v28 + 56) = 0;
            *(_QWORD *)(v36 + 80) = v40;
            *(_QWORD *)(v28 + 64) = v28 + 48;
            *(_QWORD *)(v28 + 72) = v28 + 48;
            *(_QWORD *)(v28 + 80) = 0;
          }
          else
          {
            *(_DWORD *)(v36 + 48) = 0;
            *(_QWORD *)(v36 + 56) = 0;
            *(_QWORD *)(v36 + 64) = v37;
            *(_QWORD *)(v36 + 72) = v37;
            *(_QWORD *)(v36 + 80) = 0;
          }
          v41 = sub_29A66E0((_QWORD *)(a2 + 168), v32, (unsigned int *)(v36 + 32));
          if ( v42 )
          {
            v43 = v41 || v13 == v42 || *(_DWORD *)(v36 + 32) < *(_DWORD *)(v42 + 32);
            sub_220F040(v43, v36, (_QWORD *)v42, (_QWORD *)(a2 + 176));
            ++*(_QWORD *)(a2 + 208);
            v6 = v58;
          }
          else
          {
            sub_29E1B90(*(_QWORD **)(v36 + 56));
            j_j___libc_free_0(v36);
            v6 = v58;
          }
          v12 = *(_QWORD **)(a2 + 184);
        }
      }
      v57 = v6;
      v30 = sub_220EEE0(v28);
      v6 = v57;
      v28 = v30;
    }
    while ( (_QWORD *)v30 != v29 );
  }
  v14 = *(unsigned int **)(a1 + 16);
  if ( v12 )
  {
LABEL_57:
    v49 = *v14;
    v50 = a2 + 176;
    v44 = (__int64)v12;
    while ( 1 )
    {
      while ( *(_DWORD *)(v44 + 32) < v49 )
      {
        v44 = *(_QWORD *)(v44 + 24);
        if ( !v44 )
          goto LABEL_62;
      }
      v45 = *(_QWORD *)(v44 + 16);
      if ( *(_DWORD *)(v44 + 32) <= v49 )
        break;
      v50 = v44;
      v44 = *(_QWORD *)(v44 + 16);
      if ( !v45 )
      {
LABEL_62:
        v51 = v13 == v50;
        goto LABEL_63;
      }
    }
    j = *(_QWORD *)(v44 + 24);
    if ( j )
    {
      v53 = *(_QWORD *)(j + 16);
      v54 = *(_QWORD *)(j + 24);
      if ( v49 < *(_DWORD *)(j + 32) )
        goto LABEL_73;
LABEL_70:
      for ( j = v54; j; j = v53 )
      {
        v53 = *(_QWORD *)(j + 16);
        v54 = *(_QWORD *)(j + 24);
        if ( v49 >= *(_DWORD *)(j + 32) )
          goto LABEL_70;
LABEL_73:
        v50 = j;
      }
    }
    while ( v45 )
    {
      while ( 1 )
      {
        v46 = *(_QWORD *)(v45 + 24);
        if ( v49 <= *(_DWORD *)(v45 + 32) )
          break;
        v45 = *(_QWORD *)(v45 + 24);
        if ( !v46 )
          goto LABEL_51;
      }
      v44 = v45;
      v45 = *(_QWORD *)(v45 + 16);
    }
LABEL_51:
    if ( *(_QWORD *)(a2 + 192) != v44 || v13 != v50 )
    {
      for ( ; v44 != v50; --*(_QWORD *)(a2 + 208) )
      {
        v47 = (int *)v44;
        v44 = sub_220EF30(v44);
        v48 = sub_220F330(v47, (_QWORD *)(a2 + 176));
        sub_29E1B90(*((_QWORD **)v48 + 7));
        j_j___libc_free_0((unsigned __int64)v48);
      }
      return;
    }
LABEL_65:
    sub_29E1940(v12);
    *(_QWORD *)(a2 + 192) = v13;
    *(_QWORD *)(a2 + 184) = 0;
    *(_QWORD *)(a2 + 200) = v13;
    *(_QWORD *)(a2 + 208) = 0;
    return;
  }
  v50 = a2 + 176;
  v51 = 1;
LABEL_63:
  if ( *(_QWORD *)(a2 + 192) == v50 && v51 )
    goto LABEL_65;
}
