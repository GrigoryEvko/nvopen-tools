// Function: sub_183C910
// Address: 0x183c910
//
__int64 __fastcall sub_183C910(__int64 a1, _QWORD *a2, unsigned __int64 a3)
{
  __int64 v5; // rax
  __int64 v7; // rdi
  int v8; // r9d
  __int64 v9; // rdx
  _DWORD *v10; // r13
  _BYTE *v11; // rsi
  unsigned __int64 v12; // rbx
  char *v13; // rcx
  const void *v14; // rsi
  __int64 v15; // rbx
  _BYTE *v17; // rax
  int v18; // r14d
  size_t v19; // r13
  signed __int64 v20; // r15
  void *v21; // r8
  __int64 *v22; // rbx
  void *v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned __int64 v26; // r13
  char *v27; // rcx
  _BYTE *v28; // rax
  _BYTE *v29; // rsi
  size_t v30; // rbx
  void *v31; // rdi
  __int64 v32; // rsi
  __int64 v33; // rsi
  __int64 v34; // rax
  void *v35; // rdi
  size_t v36; // rdx
  int v37; // eax
  int v38; // edx
  __int64 v39; // rax
  __int64 v40; // [rsp+0h] [rbp-60h]
  int v41; // [rsp+0h] [rbp-60h]
  void *v42; // [rsp+0h] [rbp-60h]
  unsigned __int64 v43; // [rsp+8h] [rbp-58h] BYREF
  int v44; // [rsp+10h] [rbp-50h] BYREF
  void *s1; // [rsp+18h] [rbp-48h]
  __int64 v46; // [rsp+20h] [rbp-40h]
  __int64 v47; // [rsp+28h] [rbp-38h]

  v5 = *((unsigned int *)a2 + 8);
  v43 = a3;
  if ( (_DWORD)v5 )
  {
    v7 = a2[2];
    v8 = 1;
    v9 = ((_DWORD)v5 - 1) & ((unsigned int)a3 ^ (unsigned int)(a3 >> 9));
    v10 = (_DWORD *)(v7 + 40LL * (unsigned int)v9);
    v11 = *(_BYTE **)v10;
    if ( a3 == *(_QWORD *)v10 )
    {
LABEL_3:
      if ( v10 != (_DWORD *)(v7 + 40 * v5) )
      {
        *(_DWORD *)a1 = v10[2];
        v12 = *((_QWORD *)v10 + 3) - *((_QWORD *)v10 + 2);
        *(_QWORD *)(a1 + 8) = 0;
        *(_QWORD *)(a1 + 16) = 0;
        *(_QWORD *)(a1 + 24) = 0;
        if ( !v12 )
        {
          v12 = 0;
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
          v14 = (const void *)*((_QWORD *)v10 + 2);
          v15 = *((_QWORD *)v10 + 3) - (_QWORD)v14;
          if ( *((const void **)v10 + 3) != v14 )
            v13 = (char *)memmove(v13, v14, *((_QWORD *)v10 + 3) - (_QWORD)v14);
          *(_QWORD *)(a1 + 16) = &v13[v15];
          return a1;
        }
LABEL_43:
        sub_4261EA(v7, v11, v9);
      }
    }
    else
    {
      while ( v11 != (_BYTE *)-2LL )
      {
        v9 = ((_DWORD)v5 - 1) & (unsigned int)(v8 + v9);
        v10 = (_DWORD *)(v7 + 40LL * (unsigned int)v9);
        v11 = *(_BYTE **)v10;
        if ( a3 == *(_QWORD *)v10 )
          goto LABEL_3;
        ++v8;
      }
    }
  }
  v7 = (__int64)&v44;
  (*(void (__fastcall **)(int *, _QWORD, unsigned __int64))(*(_QWORD *)*a2 + 24LL))(&v44, *a2, v43);
  v9 = *a2;
  v17 = *(_BYTE **)(*a2 + 88LL);
  v11 = *(_BYTE **)(*a2 + 80LL);
  v40 = *a2;
  v18 = *(_DWORD *)(*a2 + 72LL);
  v19 = v17 - v11;
  v20 = v17 - v11;
  if ( v17 == v11 )
  {
    v21 = 0;
  }
  else
  {
    if ( v19 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_43;
    v21 = (void *)sub_22077B0(v19);
    v17 = *(_BYTE **)(v40 + 88);
    v11 = *(_BYTE **)(v40 + 80);
    v19 = v17 - v11;
  }
  if ( v17 != v11 )
  {
    v41 = v44;
    v21 = memmove(v21, v11, v19);
    if ( v18 == v41 )
    {
      v34 = v46;
      v35 = s1;
      v36 = v46 - (_QWORD)s1;
      if ( v19 == v46 - (_QWORD)s1 )
      {
LABEL_35:
        if ( !v36 )
        {
          if ( !v21 )
            goto LABEL_42;
          goto LABEL_37;
        }
        v42 = v21;
        v37 = memcmp(v35, v21, v36);
        v21 = v42;
        if ( !v37 )
        {
LABEL_37:
          j_j___libc_free_0(v21, v20);
          v35 = s1;
          v34 = v46;
LABEL_42:
          *(_QWORD *)(a1 + 16) = v34;
          v38 = v44;
          v39 = v47;
          *(_QWORD *)(a1 + 8) = v35;
          *(_DWORD *)a1 = v38;
          *(_QWORD *)(a1 + 24) = v39;
          return a1;
        }
        goto LABEL_20;
      }
    }
    goto LABEL_20;
  }
  if ( v18 == v44 )
  {
    v34 = v46;
    v35 = s1;
    v36 = v46 - (_QWORD)s1;
    if ( v19 == v46 - (_QWORD)s1 )
      goto LABEL_35;
  }
  if ( v21 )
LABEL_20:
    j_j___libc_free_0(v21, v20);
  v22 = sub_183BD80((__int64)(a2 + 1), (__int64 *)&v43);
  v7 = v22[2];
  v11 = (_BYTE *)v22[4];
  *((_DWORD *)v22 + 2) = v44;
  v23 = s1;
  s1 = 0;
  v22[2] = (__int64)v23;
  v24 = v46;
  v46 = 0;
  v22[3] = v24;
  v25 = v47;
  v47 = 0;
  v22[4] = v25;
  if ( v7 )
  {
    v11 -= v7;
    j_j___libc_free_0(v7, v11);
  }
  *(_DWORD *)a1 = *((_DWORD *)v22 + 2);
  v26 = v22[3] - v22[2];
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  if ( v26 )
  {
    if ( v26 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_43;
    v27 = (char *)sub_22077B0(v26);
  }
  else
  {
    v27 = 0;
  }
  *(_QWORD *)(a1 + 8) = v27;
  *(_QWORD *)(a1 + 16) = v27;
  *(_QWORD *)(a1 + 24) = &v27[v26];
  v28 = (_BYTE *)v22[3];
  v29 = (_BYTE *)v22[2];
  v30 = v28 - v29;
  if ( v28 != v29 )
    v27 = (char *)memmove(v27, v29, v30);
  v31 = s1;
  v32 = v47;
  *(_QWORD *)(a1 + 16) = &v27[v30];
  v33 = v32 - (_QWORD)v31;
  if ( v31 )
    j_j___libc_free_0(v31, v33);
  return a1;
}
