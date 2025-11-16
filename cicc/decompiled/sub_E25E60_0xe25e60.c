// Function: sub_E25E60
// Address: 0xe25e60
//
unsigned __int64 __fastcall sub_E25E60(__int64 a1, unsigned __int64 *a2)
{
  _QWORD *v3; // rdx
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // r12
  unsigned __int64 v6; // rax
  _BYTE *v7; // rdx
  unsigned __int64 v8; // rbx
  unsigned __int64 v9; // rax
  _BYTE *v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // r13
  _BYTE *v16; // rax
  size_t v17; // rsi
  size_t v18; // rdx
  void *v19; // rax
  size_t v20; // rsi
  char *v21; // rax
  char *v22; // rax
  _BYTE *v23; // r13
  size_t v24; // rdi
  unsigned __int64 v25; // rax
  size_t v26; // rbx
  _BYTE *v27; // r8
  size_t v28; // rsi
  size_t v29; // rax
  void *v30; // rdi
  __int64 v31; // rdx
  __int64 *v33; // rax
  __int64 *v34; // rbx
  __int64 v35; // rax
  __int64 v36; // rax
  char *v37; // r9
  _BYTE v38[11]; // [rsp+15h] [rbp-5Bh] BYREF
  void *src; // [rsp+20h] [rbp-50h] BYREF
  size_t v40; // [rsp+28h] [rbp-48h]
  unsigned __int64 v41; // [rsp+30h] [rbp-40h]
  __int64 v42; // [rsp+38h] [rbp-38h]
  int v43; // [rsp+40h] [rbp-30h]

  v3 = *(_QWORD **)(a1 + 16);
  v4 = (*v3 + v3[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
  v3[1] = v4 - *v3 + 40;
  if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
  {
    v33 = (__int64 *)sub_22077B0(32);
    v34 = v33;
    if ( v33 )
    {
      *v33 = 0;
      v33[1] = 0;
      v33[2] = 0;
      v33[3] = 0;
    }
    v35 = sub_2207820(4096);
    v34[2] = 4096;
    *v34 = v35;
    v5 = v35;
    v36 = *(_QWORD *)(a1 + 16);
    v34[1] = 40;
    v34[3] = v36;
    *(_QWORD *)(a1 + 16) = v34;
    if ( v5 )
      goto LABEL_4;
  }
  else
  {
    v5 = 0;
    if ( v4 )
    {
      v5 = v4;
LABEL_4:
      *(_DWORD *)(v5 + 8) = 5;
      *(_QWORD *)(v5 + 16) = 0;
      *(_QWORD *)(v5 + 24) = 0;
      *(_QWORD *)v5 = &unk_49E0F88;
      *(_QWORD *)(v5 + 32) = 0;
    }
  }
  v6 = *a2;
  if ( *a2 )
  {
    v7 = (_BYTE *)a2[1];
    if ( *v7 == 63 )
    {
      a2[1] = (unsigned __int64)(v7 + 1);
      *a2 = v6 - 1;
    }
  }
  v8 = sub_E219C0(a1, a2);
  v9 = *a2;
  if ( *a2 )
  {
    v10 = (_BYTE *)a2[1];
    if ( *v10 == 63 )
    {
      a2[1] = (unsigned __int64)(v10 + 1);
      *a2 = v9 - 1;
    }
  }
  v15 = sub_E25DD0(a1, a2);
  if ( !*(_BYTE *)(a1 + 8) )
  {
    v40 = 0;
    v41 = 993;
    v42 = -1;
    v43 = 1;
    v16 = (_BYTE *)malloc(993, a2, v11, v12, v13, v14);
    src = v16;
    if ( !v16 )
      goto LABEL_49;
    *v16 = 96;
    v40 = 1;
    (*(void (__fastcall **)(__int64, void **, _QWORD))(*(_QWORD *)v15 + 16LL))(v15, &src, 0);
    v17 = v40;
    v18 = v40 + 1;
    if ( v40 + 1 > v41 )
    {
      if ( v40 + 993 > 2 * v41 )
        v41 = v40 + 993;
      else
        v41 *= 2LL;
      v19 = (void *)realloc(src);
      src = v19;
      if ( !v19 )
        goto LABEL_49;
      v17 = v40;
      v18 = v40 + 1;
    }
    else
    {
      v19 = src;
    }
    v40 = v18;
    *((_BYTE *)v19 + v17) = 39;
    v20 = v40;
    if ( v40 + 3 <= v41 )
    {
      v21 = (char *)src;
    }
    else
    {
      if ( v40 + 995 > 2 * v41 )
        v41 = v40 + 995;
      else
        v41 *= 2LL;
      v21 = (char *)realloc(src);
      src = v21;
      if ( !v21 )
        goto LABEL_49;
      v20 = v40;
    }
    v22 = &v21[v20];
    *(_WORD *)v22 = 14906;
    v23 = v38;
    v22[2] = 96;
    v24 = v40 + 3;
    v40 += 3LL;
    do
    {
      *--v23 = v8 % 0xA + 48;
      v25 = v8;
      v8 /= 0xAu;
    }
    while ( v25 > 9 );
    v26 = v38 - v23;
    if ( v38 != v23 )
    {
      v37 = (char *)src;
      if ( v41 < v26 + v24 )
      {
        if ( v26 + v24 + 992 > 2 * v41 )
          v41 = v26 + v24 + 992;
        else
          v41 *= 2LL;
        src = (void *)realloc(src);
        v37 = (char *)src;
        if ( !src )
          goto LABEL_49;
        v24 = v40;
      }
      memcpy(&v37[v24], v23, v26);
      v24 = v26 + v40;
      v40 += v26;
    }
    v27 = src;
    if ( v24 + 1 <= v41 )
    {
LABEL_33:
      v27[v24] = 39;
      v28 = ++v40;
      v29 = sub_E213F0(a1, v40, src);
      v30 = src;
      *(_QWORD *)(v5 + 24) = v29;
      *(_QWORD *)(v5 + 32) = v31;
      _libc_free(v30, v28);
      return v5;
    }
    if ( v24 + 993 > 2 * v41 )
      v41 = v24 + 993;
    else
      v41 *= 2LL;
    src = (void *)realloc(src);
    v27 = src;
    if ( src )
    {
      v24 = v40;
      goto LABEL_33;
    }
LABEL_49:
    abort();
  }
  return 0;
}
