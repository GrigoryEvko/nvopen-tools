// Function: sub_36FC280
// Address: 0x36fc280
//
void __fastcall sub_36FC280(char *src, char *a2)
{
  char *i; // r15
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
  size_t v20; // rdx
  int v21; // eax
  int v22; // eax
  size_t v23; // r12
  size_t v24; // rbx
  size_t v25; // rdx
  int v26; // eax
  size_t v27; // [rsp+0h] [rbp-50h]
  size_t *v28; // [rsp+8h] [rbp-48h]
  int v29; // [rsp+8h] [rbp-48h]
  size_t *v30; // [rsp+10h] [rbp-40h]
  size_t *v31; // [rsp+10h] [rbp-40h]

  if ( src != a2 && src + 8 != a2 )
  {
    for ( i = src + 8; a2 != i; i += 8 )
    {
      while ( 1 )
      {
        v4 = *(size_t **)i;
        v5 = *(size_t **)src;
        v6 = *(_QWORD *)(*(_QWORD *)i + 8LL);
        v7 = *(_QWORD *)(*(_QWORD *)src + 8LL);
        v8 = *(_DWORD *)(v6 + 80);
        v11 = __OFSUB__(v8, *(_DWORD *)(v7 + 80));
        v9 = v8 == *(_DWORD *)(v7 + 80);
        v10 = v8 - *(_DWORD *)(v7 + 80) < 0;
        if ( v8 == *(_DWORD *)(v7 + 80) )
        {
          v22 = *(_DWORD *)(v7 + 84);
          v11 = __OFSUB__(*(_DWORD *)(v6 + 84), v22);
          v9 = *(_DWORD *)(v6 + 84) == v22;
          v10 = *(_DWORD *)(v6 + 84) - v22 < 0;
          if ( *(_DWORD *)(v6 + 84) == v22 )
            break;
        }
        if ( !(v10 ^ v11 | v9) )
          goto LABEL_4;
LABEL_9:
        v12 = (size_t **)i;
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
              goto LABEL_22;
            goto LABEL_12;
          }
          v18 = *v15;
          v19 = *v4;
          v20 = *v4;
          if ( *v15 <= *v4 )
            v20 = *v15;
          if ( !v20 )
            break;
          v27 = *v15;
          v28 = v4;
          v21 = memcmp(v13, v15 + 2, v20);
          v4 = v28;
          v18 = v27;
          if ( !v21 )
            break;
          if ( v21 >= 0 )
            goto LABEL_22;
LABEL_12:
          *v12 = v15;
          v6 = v4[1];
          --v12;
          v8 = *(_DWORD *)(v6 + 80);
        }
        if ( v18 != v19 && v18 > v19 )
          goto LABEL_12;
LABEL_22:
        *v12 = v4;
        i += 8;
        if ( a2 == i )
          return;
      }
      v23 = *v5;
      v24 = *v4;
      v25 = *v4;
      if ( *v5 <= *v4 )
        v25 = *v5;
      if ( !v25
        || (v29 = *(_DWORD *)(v6 + 80), v31 = *(size_t **)i, v26 = memcmp(v4 + 2, v5 + 2, v25), v4 = v31, v8 = v29, !v26) )
      {
        if ( v23 != v24 && v23 > v24 )
          goto LABEL_4;
        goto LABEL_9;
      }
      if ( v26 >= 0 )
        goto LABEL_9;
LABEL_4:
      if ( src != i )
      {
        v30 = v4;
        memmove(src + 8, src, i - src);
        v4 = v30;
      }
      *(_QWORD *)src = v4;
    }
  }
}
