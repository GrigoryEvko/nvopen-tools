// Function: sub_25CE2B0
// Address: 0x25ce2b0
//
__int64 __fastcall sub_25CE2B0(__int64 a1, char **a2, _QWORD *a3)
{
  int v4; // r15d
  __int64 v7; // r13
  unsigned __int64 v8; // r12
  unsigned __int64 v9; // rax
  int v10; // ecx
  char *v11; // rdi
  int v12; // r9d
  __int64 v13; // r8
  size_t v14; // rdx
  unsigned int i; // r12d
  __int64 v16; // r15
  const void *v17; // rsi
  bool v18; // al
  int v19; // eax
  unsigned int v20; // r12d
  __int64 v21; // [rsp+8h] [rbp-58h]
  int v22; // [rsp+10h] [rbp-50h]
  int v23; // [rsp+14h] [rbp-4Ch]
  size_t v24; // [rsp+18h] [rbp-48h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v7 = *(_QWORD *)(a1 + 8);
  v8 = ((0xBF58476D1CE4E5B9LL * (unsigned __int64)a2[2]) >> 31) ^ (0xBF58476D1CE4E5B9LL * (_QWORD)a2[2]);
  v9 = sub_C94890(*a2, (__int64)a2[1]);
  v10 = v4 - 1;
  v11 = *a2;
  v12 = 1;
  v13 = 0;
  v14 = (size_t)a2[1];
  for ( i = (v4 - 1) & (((0xBF58476D1CE4E5B9LL * ((unsigned int)v8 | (v9 << 32))) >> 31) ^ (484763065 * v8));
        ;
        i = v10 & v20 )
  {
    v16 = v7 + 32LL * i;
    v17 = *(const void **)v16;
    if ( *(_QWORD *)v16 != -1 )
    {
      v18 = v11 + 2 == 0;
      if ( v17 != (const void *)-2LL )
      {
        if ( *(_QWORD *)(v16 + 8) != v14 )
          goto LABEL_15;
        if ( !v14 )
          goto LABEL_10;
        v22 = v12;
        v21 = v13;
        v23 = v10;
        v24 = v14;
        v19 = memcmp(v11, v17, v14);
        v14 = v24;
        v10 = v23;
        v13 = v21;
        v12 = v22;
        v18 = v19 == 0;
      }
      if ( !v18 )
      {
LABEL_11:
        if ( v17 == (const void *)-2LL && *(_QWORD *)(v16 + 16) == -2 && !v13 )
          v13 = v7 + 32LL * i;
        goto LABEL_15;
      }
LABEL_10:
      if ( *(char **)(v16 + 16) == a2[2] )
        goto LABEL_16;
      goto LABEL_11;
    }
    if ( v11 == (char *)-1LL && a2[2] == *(char **)(v16 + 16) )
    {
LABEL_16:
      *a3 = v16;
      return 1;
    }
    if ( *(_QWORD *)(v16 + 16) == -1 )
      break;
LABEL_15:
    v20 = v12 + i;
    ++v12;
  }
  if ( !v13 )
    v13 = v7 + 32LL * i;
  *a3 = v13;
  return 0;
}
