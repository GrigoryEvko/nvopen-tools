// Function: sub_C94910
// Address: 0xc94910
//
char *__fastcall sub_C94910(__int64 a1, _QWORD *a2, size_t a3)
{
  unsigned int v6; // r13d
  __int64 v7; // rdi
  char **v8; // rbx
  __int64 v9; // r13
  int v10; // eax
  int v11; // r10d
  char **v12; // r9
  int v13; // r8d
  unsigned int i; // ecx
  char *v15; // rsi
  bool v16; // al
  unsigned int v17; // ecx
  unsigned int v18; // r13d
  int v19; // eax
  int v20; // r10d
  char **v21; // r9
  unsigned int j; // ecx
  char *v23; // rsi
  bool v24; // al
  unsigned int v25; // ecx
  int v26; // eax
  int v28; // eax
  int v29; // eax
  char *v30; // rdx
  int v31; // eax
  __int64 v32; // r13
  int v33; // eax
  int v34; // r10d
  int v35; // r8d
  unsigned int v36; // ecx
  char *v37; // rsi
  bool v38; // al
  unsigned int v39; // ecx
  int v40; // eax
  char **v41; // [rsp+8h] [rbp-48h]
  char **v42; // [rsp+8h] [rbp-48h]
  char **v43; // [rsp+8h] [rbp-48h]
  int v44; // [rsp+10h] [rbp-40h]
  int v45; // [rsp+10h] [rbp-40h]
  int v46; // [rsp+10h] [rbp-40h]
  unsigned int v47; // [rsp+14h] [rbp-3Ch]
  unsigned int v48; // [rsp+14h] [rbp-3Ch]
  unsigned int v49; // [rsp+14h] [rbp-3Ch]
  int v50; // [rsp+18h] [rbp-38h]
  __int64 v51; // [rsp+18h] [rbp-38h]
  int v52; // [rsp+18h] [rbp-38h]
  int v53; // [rsp+18h] [rbp-38h]
  int v54; // [rsp+18h] [rbp-38h]

  v6 = *(_DWORD *)(a1 + 32);
  if ( !v6 )
  {
    ++*(_QWORD *)(a1 + 8);
    v7 = a1 + 8;
LABEL_3:
    v8 = 0;
    sub_BA8070(v7, 2 * v6);
    v50 = *(_DWORD *)(a1 + 32);
    if ( !v50 )
      goto LABEL_24;
    v9 = *(_QWORD *)(a1 + 16);
    v10 = sub_C94890(a2, a3);
    v11 = 1;
    v12 = 0;
    v13 = v50 - 1;
    for ( i = (v50 - 1) & v10; ; i = v13 & v17 )
    {
      v8 = (char **)(v9 + 16LL * i);
      v15 = *v8;
      if ( *v8 == (char *)-1LL )
        goto LABEL_48;
      v16 = (_QWORD *)((char *)a2 + 2) == 0;
      if ( v15 != (char *)-2LL )
      {
        if ( (char *)a3 != v8[1] )
          goto LABEL_8;
        v45 = v11;
        v42 = v12;
        v48 = i;
        v52 = v13;
        if ( !a3 )
          goto LABEL_24;
        v28 = memcmp(a2, v15, a3);
        v13 = v52;
        i = v48;
        v12 = v42;
        v11 = v45;
        v16 = v28 == 0;
      }
      if ( v16 )
        goto LABEL_24;
      if ( !v12 && v15 == (char *)-2LL )
        v12 = v8;
LABEL_8:
      v17 = v11 + i;
      ++v11;
    }
  }
  v18 = v6 - 1;
  v51 = *(_QWORD *)(a1 + 16);
  v19 = sub_C94890(a2, a3);
  v20 = 1;
  v21 = 0;
  for ( j = v18 & v19; ; j = v18 & v25 )
  {
    v8 = (char **)(v51 + 16LL * j);
    v23 = *v8;
    v24 = (_QWORD *)((char *)a2 + 1) == 0;
    if ( *v8 != (char *)-1LL )
    {
      v24 = (_QWORD *)((char *)a2 + 2) == 0;
      if ( v23 != (char *)-2LL )
      {
        if ( (char *)a3 != v8[1] )
          goto LABEL_13;
        v44 = v20;
        v41 = v21;
        v47 = j;
        if ( !a3 )
          return *v8;
        v26 = memcmp(a2, v23, a3);
        j = v47;
        v21 = v41;
        v20 = v44;
        v24 = v26 == 0;
      }
    }
    if ( v24 )
      return *v8;
    if ( v23 == (char *)-1LL )
      break;
LABEL_13:
    if ( v23 == (char *)-2LL && !v21 )
      v21 = v8;
    v25 = v20 + j;
    ++v20;
  }
  v31 = *(_DWORD *)(a1 + 24);
  v6 = *(_DWORD *)(a1 + 32);
  v7 = a1 + 8;
  if ( v21 )
    v8 = v21;
  ++*(_QWORD *)(a1 + 8);
  v29 = v31 + 1;
  if ( 4 * v29 >= 3 * v6 )
    goto LABEL_3;
  if ( v6 - (v29 + *(_DWORD *)(a1 + 28)) > v6 >> 3 )
    goto LABEL_25;
  v8 = 0;
  sub_BA8070(v7, v6);
  v53 = *(_DWORD *)(a1 + 32);
  if ( !v53 )
    goto LABEL_24;
  v32 = *(_QWORD *)(a1 + 16);
  v33 = sub_C94890(a2, a3);
  v34 = 1;
  v12 = 0;
  v35 = v53 - 1;
  v36 = (v53 - 1) & v33;
  while ( 2 )
  {
    v8 = (char **)(v32 + 16LL * v36);
    v37 = *v8;
    if ( *v8 != (char *)-1LL )
    {
      v38 = (_QWORD *)((char *)a2 + 2) == 0;
      if ( v37 != (char *)-2LL )
      {
        if ( v8[1] != (char *)a3 )
        {
LABEL_38:
          if ( v12 || v37 != (char *)-2LL )
            v8 = v12;
          v39 = v34 + v36;
          v12 = v8;
          ++v34;
          v36 = v35 & v39;
          continue;
        }
        v46 = v34;
        v43 = v12;
        v49 = v36;
        v54 = v35;
        if ( !a3 )
          goto LABEL_24;
        v40 = memcmp(a2, v37, a3);
        v35 = v54;
        v36 = v49;
        v12 = v43;
        v34 = v46;
        v38 = v40 == 0;
      }
      if ( v38 )
        goto LABEL_24;
      if ( v37 == (char *)-1LL )
        goto LABEL_45;
      goto LABEL_38;
    }
    break;
  }
LABEL_48:
  if ( a2 == (_QWORD *)-1LL )
    goto LABEL_24;
LABEL_45:
  if ( v12 )
    v8 = v12;
LABEL_24:
  v29 = *(_DWORD *)(a1 + 24) + 1;
LABEL_25:
  *(_DWORD *)(a1 + 24) = v29;
  if ( *v8 != (char *)-1LL )
    --*(_DWORD *)(a1 + 28);
  *v8 = (char *)a2;
  v8[1] = (char *)a3;
  *v8 = sub_C948A0((char ***)a1, a2, a3);
  v8[1] = v30;
  return *v8;
}
