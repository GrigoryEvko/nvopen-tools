// Function: sub_3581570
// Address: 0x3581570
//
__int64 __fastcall sub_3581570(__int64 a1, __int64 a2, _QWORD *a3)
{
  int v4; // r15d
  __int64 v6; // r13
  unsigned int v7; // eax
  unsigned __int64 v8; // r12
  unsigned __int64 v9; // rax
  int v10; // r8d
  char *v11; // rdi
  size_t v12; // r9
  int v13; // r11d
  __int64 v14; // r10
  unsigned int i; // r12d
  __int64 v16; // rcx
  __int64 v17; // r15
  bool v18; // al
  int v19; // eax
  unsigned int v20; // r12d
  __int64 v21; // [rsp+10h] [rbp-50h]
  int v22; // [rsp+18h] [rbp-48h]
  int v23; // [rsp+1Ch] [rbp-44h]
  size_t v24; // [rsp+20h] [rbp-40h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v7 = ((0xBF58476D1CE4E5B9LL
       * ((969526130 * ((484763065 * *(_DWORD *)a2) ^ (unsigned int)((0xBF58476D1CE4E5B9LL * *(_QWORD *)a2) >> 31)))
        | ((unsigned __int64)(unsigned int)(37 * *(_DWORD *)(a2 + 8)) << 32))) >> 31)
     ^ (-279380126 * ((484763065 * *(_DWORD *)a2) ^ ((0xBF58476D1CE4E5B9LL * *(_QWORD *)a2) >> 31)));
  v8 = ((0xBF58476D1CE4E5B9LL * (v7 | ((unsigned __int64)(unsigned int)(37 * *(_DWORD *)(a2 + 12)) << 32))) >> 31)
     ^ (0xBF58476D1CE4E5B9LL * (v7 | ((unsigned __int64)(unsigned int)(37 * *(_DWORD *)(a2 + 12)) << 32)));
  v9 = sub_C94890(*(_QWORD **)(a2 + 16), *(_QWORD *)(a2 + 24));
  v10 = v4 - 1;
  v11 = *(char **)(a2 + 16);
  v12 = *(_QWORD *)(a2 + 24);
  v13 = 1;
  v14 = 0;
  for ( i = (v4 - 1) & (((0xBF58476D1CE4E5B9LL * ((v9 << 32) | (unsigned int)v8)) >> 31) ^ (484763065 * v8));
        ;
        i = v10 & v20 )
  {
    v16 = v6 + 40LL * i;
    v17 = *(_QWORD *)(v16 + 16);
    if ( v17 == -1 )
      break;
    v18 = v11 + 2 == 0;
    if ( v17 == -2 )
      goto LABEL_9;
    if ( *(_QWORD *)(v16 + 24) != v12 )
      goto LABEL_13;
    if ( v12 )
    {
      v22 = v13;
      v21 = v14;
      v23 = v10;
      v24 = v12;
      v19 = memcmp(v11, *(const void **)(v16 + 16), v12);
      v12 = v24;
      v10 = v23;
      v14 = v21;
      v13 = v22;
      v16 = v6 + 40LL * i;
      v18 = v19 == 0;
LABEL_9:
      if ( !v18 )
        goto LABEL_11;
LABEL_10:
      if ( *(_DWORD *)(a2 + 12) != *(_DWORD *)(v16 + 12) )
        goto LABEL_11;
      goto LABEL_14;
    }
    if ( *(_DWORD *)(a2 + 12) != *(_DWORD *)(v16 + 12) )
      goto LABEL_25;
LABEL_14:
    if ( *(_DWORD *)(a2 + 8) == *(_DWORD *)(v16 + 8) && *(_QWORD *)a2 == *(_QWORD *)v16 )
    {
      *a3 = v16;
      return 1;
    }
LABEL_11:
    if ( v17 == -1 )
      goto LABEL_12;
LABEL_25:
    if ( v17 == -2 && *(_DWORD *)(v16 + 12) == -2 && *(_DWORD *)(v16 + 8) == -2 && *(_QWORD *)v16 == -2 && !v14 )
      v14 = v16;
LABEL_13:
    v20 = v13 + i;
    ++v13;
  }
  if ( v11 == (char *)-1LL )
    goto LABEL_10;
LABEL_12:
  if ( *(_DWORD *)(v16 + 12) != -1 || *(_DWORD *)(v16 + 8) != -1 || *(_QWORD *)v16 != -1 )
    goto LABEL_13;
  if ( !v14 )
    v14 = v16;
  *a3 = v14;
  return 0;
}
