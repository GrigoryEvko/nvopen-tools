// Function: sub_25A73A0
// Address: 0x25a73a0
//
__int64 __fastcall sub_25A73A0(__int64 a1, size_t *a2, unsigned __int64 a3)
{
  __int64 v6; // rax
  size_t v7; // rcx
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdx
  size_t v10; // r14
  unsigned __int64 v11; // rsi
  unsigned __int64 v12; // rbx
  char *v13; // rcx
  const void *v14; // rsi
  __int64 v15; // rbx
  int v17; // r8d
  const void *v18; // rax
  int v19; // r14d
  unsigned __int64 v20; // r8
  void *v21; // r9
  void *v22; // rax
  size_t v23; // r9
  int v24; // r11d
  unsigned int v25; // edi
  unsigned __int64 v26; // rax
  __int64 v27; // rcx
  unsigned __int64 v28; // r13
  __int64 v29; // rsi
  void *v30; // rdx
  __int64 v31; // rdx
  unsigned __int64 v32; // rbx
  char *v33; // rcx
  const void *v34; // rsi
  __int64 v35; // rbx
  void *v36; // rdi
  __int64 v37; // rax
  void *v38; // rdi
  int v39; // eax
  int v40; // eax
  size_t v41; // r8
  int v42; // ecx
  unsigned int v43; // eax
  int v44; // eax
  void *v45; // rax
  int v46; // edx
  unsigned __int64 v47; // rax
  int v48; // eax
  int v49; // eax
  int v50; // eax
  int v51; // r9d
  unsigned int v52; // r14d
  unsigned __int64 v53; // r8
  int v54; // r10d
  unsigned __int64 v55; // r9
  int v56; // [rsp+4h] [rbp-5Ch]
  size_t n; // [rsp+8h] [rbp-58h]
  size_t na; // [rsp+8h] [rbp-58h]
  size_t nb; // [rsp+8h] [rbp-58h]
  int v60; // [rsp+10h] [rbp-50h] BYREF
  void *s1; // [rsp+18h] [rbp-48h]
  __int64 v62; // [rsp+20h] [rbp-40h]
  unsigned __int64 v63; // [rsp+28h] [rbp-38h]

  v6 = *((unsigned int *)a2 + 8);
  v7 = a2[2];
  if ( (_DWORD)v6 )
  {
    v8 = (unsigned int)(v6 - 1);
    v9 = (unsigned int)v8 & ((unsigned int)a3 ^ (unsigned int)(a3 >> 9));
    v10 = v7 + 40LL * (unsigned int)v9;
    v11 = *(_QWORD *)v10;
    if ( a3 == *(_QWORD *)v10 )
    {
LABEL_3:
      if ( v10 != v7 + 40 * v6 )
      {
        *(_DWORD *)a1 = *(_DWORD *)(v10 + 8);
        v12 = *(_QWORD *)(v10 + 24) - *(_QWORD *)(v10 + 16);
        *(_QWORD *)(a1 + 8) = 0;
        *(_QWORD *)(a1 + 16) = 0;
        *(_QWORD *)(a1 + 24) = 0;
        if ( !v12 )
        {
          v13 = 0;
          goto LABEL_7;
        }
        if ( v12 <= 0x7FFFFFFFFFFFFFF8LL )
        {
          v13 = (char *)sub_22077B0(v12);
LABEL_7:
          *(_QWORD *)(a1 + 8) = v13;
          *(_QWORD *)(a1 + 24) = &v13[v12];
          *(_QWORD *)(a1 + 16) = v13;
          v14 = *(const void **)(v10 + 16);
          v15 = *(_QWORD *)(v10 + 24) - (_QWORD)v14;
          if ( *(const void **)(v10 + 24) != v14 )
            v13 = (char *)memmove(v13, v14, *(_QWORD *)(v10 + 24) - (_QWORD)v14);
          *(_QWORD *)(a1 + 16) = &v13[v15];
          return a1;
        }
LABEL_73:
        sub_4261EA(v8, v11, v9);
      }
    }
    else
    {
      v17 = 1;
      while ( v11 != -2 )
      {
        v9 = (unsigned int)v8 & (v17 + (_DWORD)v9);
        v10 = v7 + 40LL * (unsigned int)v9;
        v11 = *(_QWORD *)v10;
        if ( *(_QWORD *)v10 == a3 )
          goto LABEL_3;
        ++v17;
      }
    }
  }
  v8 = (unsigned __int64)&v60;
  (*(void (__fastcall **)(int *, size_t, unsigned __int64, size_t))(*(_QWORD *)*a2 + 24LL))(&v60, *a2, a3, v7);
  v9 = *a2;
  v18 = *(const void **)(*a2 + 88);
  v11 = *(_QWORD *)(*a2 + 80);
  n = *a2;
  v19 = *(_DWORD *)(*a2 + 72);
  v20 = (unsigned __int64)v18 - v11;
  if ( v18 == (const void *)v11 )
  {
    v21 = 0;
  }
  else
  {
    if ( v20 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_73;
    v21 = (void *)sub_22077B0(v20);
    v18 = *(const void **)(n + 88);
    v11 = *(_QWORD *)(n + 80);
    v20 = (unsigned __int64)v18 - v11;
  }
  if ( (const void *)v11 != v18 )
  {
    v56 = v60;
    na = v20;
    v22 = memmove(v21, (const void *)v11, v20);
    v20 = na;
    v21 = v22;
    if ( v19 == v56 )
    {
      v37 = v62;
      v38 = s1;
      if ( v62 - (_QWORD)s1 == na )
      {
LABEL_36:
        if ( !v20 )
        {
          if ( !v21 )
            goto LABEL_49;
          goto LABEL_38;
        }
        nb = (size_t)v21;
        v39 = memcmp(v38, v21, v20);
        v21 = (void *)nb;
        if ( !v39 )
        {
LABEL_38:
          j_j___libc_free_0((unsigned __int64)v21);
          v38 = s1;
          v37 = v62;
LABEL_49:
          *(_QWORD *)(a1 + 16) = v37;
          v46 = v60;
          v47 = v63;
          *(_QWORD *)(a1 + 8) = v38;
          *(_DWORD *)a1 = v46;
          *(_QWORD *)(a1 + 24) = v47;
          return a1;
        }
        goto LABEL_19;
      }
    }
    goto LABEL_19;
  }
  if ( v19 == v60 )
  {
    v37 = v62;
    v38 = s1;
    if ( v62 - (_QWORD)s1 == v20 )
      goto LABEL_36;
  }
  if ( v21 )
LABEL_19:
    j_j___libc_free_0((unsigned __int64)v21);
  v11 = *((unsigned int *)a2 + 8);
  if ( !(_DWORD)v11 )
  {
    ++a2[1];
    goto LABEL_43;
  }
  v23 = a2[2];
  v9 = 0;
  v24 = 1;
  v25 = (v11 - 1) & (a3 ^ (a3 >> 9));
  v26 = v23 + 40LL * v25;
  v27 = *(_QWORD *)v26;
  if ( a3 == *(_QWORD *)v26 )
  {
LABEL_22:
    v8 = *(_QWORD *)(v26 + 16);
    v28 = v26 + 8;
    v29 = *(_QWORD *)(v26 + 32);
    *(_DWORD *)(v26 + 8) = v60;
    v30 = s1;
    v11 = v29 - v8;
    s1 = 0;
    *(_QWORD *)(v26 + 16) = v30;
    v31 = v62;
    v62 = 0;
    *(_QWORD *)(v26 + 24) = v31;
    v9 = v63;
    v63 = 0;
    *(_QWORD *)(v26 + 32) = v9;
    if ( v8 )
      j_j___libc_free_0(v8);
    goto LABEL_24;
  }
  while ( v27 != -2 )
  {
    if ( !v9 && v27 == -16 )
      v9 = v26;
    v25 = (v11 - 1) & (v24 + v25);
    v26 = v23 + 40LL * v25;
    v27 = *(_QWORD *)v26;
    if ( *(_QWORD *)v26 == a3 )
      goto LABEL_22;
    ++v24;
  }
  if ( !v9 )
    v9 = v26;
  v48 = *((_DWORD *)a2 + 6);
  ++a2[1];
  v42 = v48 + 1;
  if ( 4 * (v48 + 1) >= (unsigned int)(3 * v11) )
  {
LABEL_43:
    sub_25A5BF0((__int64)(a2 + 1), 2 * v11);
    v40 = *((_DWORD *)a2 + 8);
    if ( v40 )
    {
      v11 = (unsigned int)(v40 - 1);
      v41 = a2[2];
      v42 = *((_DWORD *)a2 + 6) + 1;
      v43 = v11 & (a3 ^ (a3 >> 9));
      v9 = v41 + 40LL * v43;
      v8 = *(_QWORD *)v9;
      if ( a3 != *(_QWORD *)v9 )
      {
        v54 = 1;
        v55 = 0;
        while ( v8 != -2 )
        {
          if ( v8 == -16 && !v55 )
            v55 = v9;
          v43 = v11 & (v54 + v43);
          v9 = v41 + 40LL * v43;
          v8 = *(_QWORD *)v9;
          if ( *(_QWORD *)v9 == a3 )
            goto LABEL_45;
          ++v54;
        }
        if ( v55 )
          v9 = v55;
      }
      goto LABEL_45;
    }
    goto LABEL_84;
  }
  v8 = (unsigned int)v11 >> 3;
  if ( (int)v11 - *((_DWORD *)a2 + 7) - v42 <= (unsigned int)v8 )
  {
    sub_25A5BF0((__int64)(a2 + 1), v11);
    v49 = *((_DWORD *)a2 + 8);
    if ( v49 )
    {
      v50 = v49 - 1;
      v8 = a2[2];
      v51 = 1;
      v52 = v50 & (a3 ^ (a3 >> 9));
      v53 = 0;
      v42 = *((_DWORD *)a2 + 6) + 1;
      v9 = v8 + 40LL * v52;
      v11 = *(_QWORD *)v9;
      if ( *(_QWORD *)v9 != a3 )
      {
        while ( v11 != -2 )
        {
          if ( !v53 && v11 == -16 )
            v53 = v9;
          v52 = v50 & (v51 + v52);
          v9 = v8 + 40LL * v52;
          v11 = *(_QWORD *)v9;
          if ( a3 == *(_QWORD *)v9 )
            goto LABEL_45;
          ++v51;
        }
        if ( v53 )
          v9 = v53;
      }
      goto LABEL_45;
    }
LABEL_84:
    ++*((_DWORD *)a2 + 6);
    BUG();
  }
LABEL_45:
  *((_DWORD *)a2 + 6) = v42;
  if ( *(_QWORD *)v9 != -2 )
    --*((_DWORD *)a2 + 7);
  *(_QWORD *)v9 = a3;
  v44 = v60;
  v28 = v9 + 8;
  *(_OWORD *)(v9 + 8) = 0;
  *(_DWORD *)(v9 + 8) = v44;
  v45 = s1;
  *(_OWORD *)(v9 + 24) = 0;
  *(_QWORD *)(v9 + 16) = v45;
  s1 = 0;
  *(_QWORD *)(v9 + 24) = v62;
  v62 = 0;
  *(_QWORD *)(v9 + 32) = v63;
  v63 = 0;
LABEL_24:
  *(_DWORD *)a1 = *(_DWORD *)v28;
  v32 = *(_QWORD *)(v28 + 16) - *(_QWORD *)(v28 + 8);
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  if ( v32 )
  {
    if ( v32 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_73;
    v33 = (char *)sub_22077B0(v32);
  }
  else
  {
    v32 = 0;
    v33 = 0;
  }
  *(_QWORD *)(a1 + 8) = v33;
  *(_QWORD *)(a1 + 24) = &v33[v32];
  *(_QWORD *)(a1 + 16) = v33;
  v34 = *(const void **)(v28 + 8);
  v35 = *(_QWORD *)(v28 + 16) - (_QWORD)v34;
  if ( *(const void **)(v28 + 16) != v34 )
    v33 = (char *)memmove(v33, v34, *(_QWORD *)(v28 + 16) - (_QWORD)v34);
  v36 = s1;
  *(_QWORD *)(a1 + 16) = &v33[v35];
  if ( v36 )
    j_j___libc_free_0((unsigned __int64)v36);
  return a1;
}
