// Function: sub_C0EEA0
// Address: 0xc0eea0
//
void __fastcall sub_C0EEA0(__int64 a1, const char **a2, const char **a3)
{
  const char **v3; // rbx
  const char *v4; // r13
  size_t v5; // r14
  unsigned int v6; // r12d
  __int64 v7; // r9
  int v8; // eax
  unsigned int v9; // r12d
  int v10; // eax
  int v11; // r11d
  unsigned int j; // r10d
  __int64 v13; // rcx
  const void *v14; // rsi
  bool v15; // al
  unsigned int v16; // r10d
  int v17; // eax
  int v18; // eax
  __int64 v19; // r12
  int v20; // eax
  int v21; // r11d
  __int64 v22; // r10
  int v23; // r8d
  unsigned int v24; // ecx
  const void *v25; // rsi
  bool v26; // al
  unsigned int v27; // ecx
  __int64 v28; // r12
  int v29; // eax
  int v30; // r11d
  int v31; // r8d
  unsigned int i; // ecx
  const void *v33; // rsi
  bool v34; // al
  int v35; // eax
  unsigned int v36; // ecx
  int v37; // eax
  __int64 v38; // [rsp+8h] [rbp-68h]
  __int64 v39; // [rsp+8h] [rbp-68h]
  __int64 v40; // [rsp+10h] [rbp-60h]
  __int64 v41; // [rsp+18h] [rbp-58h]
  int v42; // [rsp+18h] [rbp-58h]
  int v43; // [rsp+18h] [rbp-58h]
  int v44; // [rsp+20h] [rbp-50h]
  __int64 v45; // [rsp+20h] [rbp-50h]
  __int64 v46; // [rsp+20h] [rbp-50h]
  unsigned int v47; // [rsp+2Ch] [rbp-44h]
  unsigned int v48; // [rsp+2Ch] [rbp-44h]
  unsigned int v49; // [rsp+2Ch] [rbp-44h]
  int v50; // [rsp+30h] [rbp-40h]
  __int64 v51; // [rsp+30h] [rbp-40h]
  int v52; // [rsp+30h] [rbp-40h]
  int v53; // [rsp+30h] [rbp-40h]
  int v54; // [rsp+30h] [rbp-40h]

  if ( a2 == a3 )
    return;
  v3 = a2;
  do
  {
LABEL_3:
    v4 = *v3;
    v5 = 0;
    if ( *v3 )
      v5 = strlen(*v3);
    v6 = *(_DWORD *)(a1 + 24);
    if ( !v6 )
    {
      ++*(_QWORD *)a1;
LABEL_7:
      sub_BA8070(a1, 2 * v6);
      v7 = 0;
      v50 = *(_DWORD *)(a1 + 24);
      if ( !v50 )
        goto LABEL_8;
      v28 = *(_QWORD *)(a1 + 8);
      v29 = sub_C94890(v4, v5);
      v30 = 1;
      v22 = 0;
      v31 = v50 - 1;
      for ( i = (v50 - 1) & v29; ; i = v31 & v36 )
      {
        v7 = v28 + 16LL * i;
        v33 = *(const void **)v7;
        if ( *(_QWORD *)v7 == -1 )
          goto LABEL_57;
        v34 = v4 + 2 == 0;
        if ( v33 != (const void *)-2LL )
        {
          if ( v5 != *(_QWORD *)(v7 + 8) )
            goto LABEL_49;
          v42 = v30;
          v45 = v22;
          v48 = i;
          v53 = v31;
          if ( !v5 )
            goto LABEL_8;
          v38 = v28 + 16LL * i;
          v35 = memcmp(v4, v33, v5);
          v7 = v38;
          v31 = v53;
          i = v48;
          v22 = v45;
          v30 = v42;
          v34 = v35 == 0;
        }
        if ( v34 )
          goto LABEL_8;
        if ( !v22 && v33 == (const void *)-2LL )
          v22 = v7;
LABEL_49:
        v36 = v30 + i;
        ++v30;
      }
    }
    v9 = v6 - 1;
    v51 = *(_QWORD *)(a1 + 8);
    v10 = sub_C94890(v4, v5);
    v11 = 1;
    v7 = 0;
    for ( j = v9 & v10; ; j = v9 & v16 )
    {
      v13 = v51 + 16LL * j;
      v14 = *(const void **)v13;
      v15 = v4 + 1 == 0;
      if ( *(_QWORD *)v13 != -1 )
      {
        v15 = v4 + 2 == 0;
        if ( v14 != (const void *)-2LL )
        {
          if ( v5 != *(_QWORD *)(v13 + 8) )
            goto LABEL_17;
          v40 = v51 + 16LL * j;
          v41 = v7;
          v44 = v11;
          v47 = j;
          if ( !v5 )
            goto LABEL_24;
          v17 = memcmp(v4, v14, v5);
          j = v47;
          v11 = v44;
          v7 = v41;
          v13 = v40;
          v15 = v17 == 0;
        }
      }
      if ( v15 )
      {
LABEL_24:
        if ( a3 == ++v3 )
          return;
        goto LABEL_3;
      }
      if ( v14 == (const void *)-1LL )
        break;
LABEL_17:
      if ( v14 == (const void *)-2LL && !v7 )
        v7 = v13;
      v16 = v11 + j;
      ++v11;
    }
    v18 = *(_DWORD *)(a1 + 16);
    v6 = *(_DWORD *)(a1 + 24);
    if ( !v7 )
      v7 = v13;
    ++*(_QWORD *)a1;
    v8 = v18 + 1;
    if ( 4 * v8 >= 3 * v6 )
      goto LABEL_7;
    if ( v6 - (v8 + *(_DWORD *)(a1 + 20)) > v6 >> 3 )
      goto LABEL_9;
    sub_BA8070(a1, v6);
    v7 = 0;
    v52 = *(_DWORD *)(a1 + 24);
    if ( !v52 )
      goto LABEL_8;
    v19 = *(_QWORD *)(a1 + 8);
    v20 = sub_C94890(v4, v5);
    v21 = 1;
    v22 = 0;
    v23 = v52 - 1;
    v24 = (v52 - 1) & v20;
    while ( 2 )
    {
      v7 = v19 + 16LL * v24;
      v25 = *(const void **)v7;
      if ( *(_QWORD *)v7 != -1 )
      {
        v26 = v4 + 2 == 0;
        if ( v25 != (const void *)-2LL )
        {
          if ( v5 != *(_QWORD *)(v7 + 8) )
          {
LABEL_36:
            if ( v22 || v25 != (const void *)-2LL )
              v7 = v22;
            v27 = v21 + v24;
            v22 = v7;
            ++v21;
            v24 = v23 & v27;
            continue;
          }
          v43 = v21;
          v46 = v22;
          v49 = v24;
          v54 = v23;
          if ( !v5 )
            goto LABEL_8;
          v39 = v19 + 16LL * v24;
          v37 = memcmp(v4, v25, v5);
          v7 = v39;
          v23 = v54;
          v24 = v49;
          v22 = v46;
          v21 = v43;
          v26 = v37 == 0;
        }
        if ( v26 )
          goto LABEL_8;
        if ( v25 == (const void *)-1LL )
          goto LABEL_54;
        goto LABEL_36;
      }
      break;
    }
LABEL_57:
    if ( v4 == (const char *)-1LL )
      goto LABEL_8;
LABEL_54:
    if ( v22 )
      v7 = v22;
LABEL_8:
    v8 = *(_DWORD *)(a1 + 16) + 1;
LABEL_9:
    *(_DWORD *)(a1 + 16) = v8;
    if ( *(_QWORD *)v7 != -1 )
      --*(_DWORD *)(a1 + 20);
    *(_QWORD *)v7 = v4;
    ++v3;
    *(_QWORD *)(v7 + 8) = v5;
  }
  while ( a3 != v3 );
}
