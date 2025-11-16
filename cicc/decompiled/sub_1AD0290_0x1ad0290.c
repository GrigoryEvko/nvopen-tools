// Function: sub_1AD0290
// Address: 0x1ad0290
//
void __fastcall sub_1AD0290(char *src, char *a2)
{
  char *v3; // r15
  size_t *v4; // r8
  size_t *v5; // rsi
  size_t v6; // r13
  __int64 v7; // rax
  int v8; // ecx
  bool v9; // zf
  bool v10; // sf
  bool v11; // of
  size_t **v12; // rbx
  const void *v13; // rdi
  bool v14; // al
  size_t *v15; // r12
  size_t v16; // rax
  int v17; // eax
  size_t v18; // rcx
  size_t v19; // r13
  const void *v20; // rsi
  int v21; // eax
  int v22; // eax
  size_t v23; // rbx
  size_t v24; // r12
  const void *v25; // rsi
  const void *v26; // rdi
  int v27; // eax
  size_t v28; // [rsp+0h] [rbp-50h]
  size_t *v29; // [rsp+0h] [rbp-50h]
  size_t *v30; // [rsp+8h] [rbp-48h]
  int v31; // [rsp+8h] [rbp-48h]
  size_t v32; // [rsp+8h] [rbp-48h]
  int v33; // [rsp+8h] [rbp-48h]
  size_t *v34; // [rsp+10h] [rbp-40h]
  size_t *v35; // [rsp+10h] [rbp-40h]
  size_t *v36; // [rsp+10h] [rbp-40h]

  if ( src == a2 || src + 8 == a2 )
    return;
  v3 = src + 8;
  do
  {
    while ( 1 )
    {
      v4 = *(size_t **)v3;
      v5 = *(size_t **)src;
      v6 = *(_QWORD *)(*(_QWORD *)v3 + 8LL);
      v7 = *(_QWORD *)(*(_QWORD *)src + 8LL);
      v8 = *(_DWORD *)(v6 + 80);
      v11 = __OFSUB__(v8, *(_DWORD *)(v7 + 80));
      v9 = v8 == *(_DWORD *)(v7 + 80);
      v10 = v8 - *(_DWORD *)(v7 + 80) < 0;
      if ( v8 != *(_DWORD *)(v7 + 80)
        || (v22 = *(_DWORD *)(v7 + 84),
            v11 = __OFSUB__(*(_DWORD *)(v6 + 84), v22),
            v9 = *(_DWORD *)(v6 + 84) == v22,
            v10 = *(_DWORD *)(v6 + 84) - v22 < 0,
            *(_DWORD *)(v6 + 84) != v22) )
      {
        if ( !(v10 ^ v11 | v9) )
          goto LABEL_4;
        goto LABEL_9;
      }
      v23 = *v5;
      v24 = *v4;
      v25 = v5 + 2;
      v26 = v4 + 2;
      if ( v23 >= *v4 )
        break;
      if ( !v23 )
        goto LABEL_9;
      v33 = *(_DWORD *)(v6 + 80);
      v36 = *(size_t **)v3;
      v27 = memcmp(v26, v25, v23);
      v4 = v36;
      v8 = v33;
      if ( v27 )
      {
LABEL_37:
        if ( v27 < 0 )
          goto LABEL_4;
        goto LABEL_9;
      }
LABEL_29:
      if ( v23 <= v24 )
        goto LABEL_9;
LABEL_4:
      if ( src != v3 )
      {
        v34 = v4;
        memmove(src + 8, src, v3 - src);
        v4 = v34;
      }
      *(_QWORD *)src = v4;
      v3 += 8;
      if ( a2 == v3 )
        return;
    }
    if ( v24 )
    {
      v31 = *(_DWORD *)(v6 + 80);
      v35 = *(size_t **)v3;
      v27 = memcmp(v26, v25, *v4);
      v4 = v35;
      v8 = v31;
      if ( v27 )
        goto LABEL_37;
    }
    if ( v23 != v24 )
      goto LABEL_29;
LABEL_9:
    v12 = (size_t **)v3;
    v13 = v4 + 2;
    while ( 1 )
    {
      v15 = *(v12 - 1);
      v16 = v15[1];
      if ( *(_DWORD *)(v16 + 80) != v8 )
      {
        v14 = *(_DWORD *)(v16 + 80) < v8;
        goto LABEL_11;
      }
      v17 = *(_DWORD *)(v16 + 84);
      if ( *(_DWORD *)(v6 + 84) != v17 )
      {
        v14 = *(_DWORD *)(v6 + 84) > v17;
LABEL_11:
        if ( !v14 )
          goto LABEL_21;
        goto LABEL_12;
      }
      v18 = *v15;
      v19 = *v4;
      v20 = v15 + 2;
      if ( *v15 >= *v4 )
      {
        if ( v19 )
        {
          v28 = *v15;
          v30 = v4;
          v21 = memcmp(v13, v20, *v4);
          v4 = v30;
          v18 = v28;
          if ( v21 )
            break;
        }
        if ( v18 == v19 )
          goto LABEL_21;
        goto LABEL_20;
      }
      if ( !v18 )
      {
LABEL_21:
        *v12 = v4;
        goto LABEL_22;
      }
      v29 = v4;
      v32 = *v15;
      v21 = memcmp(v13, v20, *v15);
      v18 = v32;
      v4 = v29;
      if ( v21 )
        break;
LABEL_20:
      if ( v18 <= v19 )
        goto LABEL_21;
LABEL_12:
      *v12 = v15;
      v6 = v4[1];
      --v12;
      v8 = *(_DWORD *)(v6 + 80);
    }
    if ( v21 < 0 )
      goto LABEL_12;
    *v12 = v4;
LABEL_22:
    v3 += 8;
  }
  while ( a2 != v3 );
}
