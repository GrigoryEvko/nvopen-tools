// Function: sub_E6EBC0
// Address: 0xe6ebc0
//
__int64 __fastcall sub_E6EBC0(__int64 a1, __int64 a2, _QWORD *a3)
{
  int v4; // r15d
  __int64 v6; // r12
  unsigned __int64 v7; // rbx
  unsigned __int64 v8; // rax
  int v9; // r8d
  char *v10; // rdi
  int v11; // r9d
  __int64 v12; // r10
  size_t v13; // rdx
  unsigned int i; // ebx
  __int64 v15; // r15
  const void *v16; // rcx
  bool v17; // al
  int v18; // eax
  unsigned int v19; // ebx
  __int64 v20; // [rsp+8h] [rbp-58h]
  int v21; // [rsp+10h] [rbp-50h]
  int v22; // [rsp+14h] [rbp-4Ch]
  size_t v23; // [rsp+18h] [rbp-48h]
  const void *v24; // [rsp+20h] [rbp-40h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v7 = ((0xBF58476D1CE4E5B9LL
       * ((unsigned int)(1512728442 * *(_DWORD *)a2) | ((unsigned __int64)(unsigned int)(37 * *(_DWORD *)(a2 + 4)) << 32))) >> 31)
     ^ (0xBF58476D1CE4E5B9LL
      * ((unsigned int)(1512728442 * *(_DWORD *)a2) | ((unsigned __int64)(unsigned int)(37 * *(_DWORD *)(a2 + 4)) << 32)));
  v8 = sub_C94890(*(_QWORD **)(a2 + 8), *(_QWORD *)(a2 + 16));
  v9 = v4 - 1;
  v10 = *(char **)(a2 + 8);
  v11 = 1;
  v12 = 0;
  v13 = *(_QWORD *)(a2 + 16);
  for ( i = (v4 - 1) & (((0xBF58476D1CE4E5B9LL * ((unsigned int)v7 | (v8 << 32))) >> 31) ^ (484763065 * v7)); ; i = v9 & v19 )
  {
    v15 = v6 + 32LL * i;
    v16 = *(const void **)(v15 + 8);
    if ( v16 == (const void *)-1LL )
      break;
    v17 = v10 + 2 == 0;
    if ( v16 == (const void *)-2LL )
      goto LABEL_9;
    if ( *(_QWORD *)(v15 + 16) != v13 )
      goto LABEL_13;
    if ( v13 )
    {
      v21 = v11;
      v20 = v12;
      v22 = v9;
      v23 = v13;
      v24 = *(const void **)(v15 + 8);
      v18 = memcmp(v10, v24, v13);
      v16 = v24;
      v13 = v23;
      v9 = v22;
      v12 = v20;
      v11 = v21;
      v17 = v18 == 0;
LABEL_9:
      if ( !v17 )
        goto LABEL_11;
LABEL_10:
      if ( *(_DWORD *)(a2 + 4) != *(_DWORD *)(v15 + 4) )
        goto LABEL_11;
      goto LABEL_14;
    }
    if ( *(_DWORD *)(a2 + 4) != *(_DWORD *)(v15 + 4) )
      goto LABEL_23;
LABEL_14:
    if ( *(_DWORD *)a2 == *(_DWORD *)v15 )
    {
      *a3 = v15;
      return 1;
    }
LABEL_11:
    if ( v16 == (const void *)-1LL )
      goto LABEL_12;
LABEL_23:
    if ( v16 == (const void *)-2LL && *(_DWORD *)(v15 + 4) == -2 && *(_DWORD *)v15 == -2 && !v12 )
      v12 = v6 + 32LL * i;
LABEL_13:
    v19 = v11 + i;
    ++v11;
  }
  if ( v10 == (char *)-1LL )
    goto LABEL_10;
LABEL_12:
  if ( *(_DWORD *)(v15 + 4) != -1 || *(_DWORD *)v15 != -1 )
    goto LABEL_13;
  if ( !v12 )
    v12 = v6 + 32LL * i;
  *a3 = v12;
  return 0;
}
