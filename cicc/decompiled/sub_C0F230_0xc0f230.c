// Function: sub_C0F230
// Address: 0xc0f230
//
__int64 __fastcall sub_C0F230(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rax
  __int64 v3; // rdx
  _QWORD *i; // rdx
  const char **v5; // rbx
  const char *v6; // r12
  size_t v7; // rax
  unsigned int v8; // ecx
  size_t v9; // r15
  __int64 v10; // r9
  int v11; // eax
  int v12; // r11d
  __int64 v13; // r10
  int v14; // ecx
  unsigned int j; // r8d
  const void *v16; // rsi
  bool v17; // al
  unsigned int v18; // r8d
  int v19; // eax
  int v20; // eax
  int v22; // eax
  int v23; // r11d
  int v24; // ecx
  unsigned int k; // r10d
  __int64 v26; // r8
  const void *v27; // rsi
  bool v28; // al
  unsigned int v29; // r10d
  int v30; // eax
  int v31; // eax
  int v32; // eax
  int v33; // r11d
  int v34; // ecx
  unsigned int v35; // r8d
  const void *v36; // rsi
  bool v37; // al
  int v38; // eax
  unsigned int v39; // r8d
  __int64 v40; // [rsp+8h] [rbp-22A8h]
  int v41; // [rsp+8h] [rbp-22A8h]
  __int64 v42; // [rsp+10h] [rbp-22A0h]
  __int64 v43; // [rsp+10h] [rbp-22A0h]
  int v44; // [rsp+18h] [rbp-2298h]
  __int64 v45; // [rsp+18h] [rbp-2298h]
  unsigned int v46; // [rsp+18h] [rbp-2298h]
  __int64 v47; // [rsp+20h] [rbp-2290h]
  int v48; // [rsp+20h] [rbp-2290h]
  int v49; // [rsp+20h] [rbp-2290h]
  unsigned int v50; // [rsp+28h] [rbp-2288h]
  unsigned int v51; // [rsp+28h] [rbp-2288h]
  __int64 v52; // [rsp+28h] [rbp-2288h]
  __int64 v53; // [rsp+30h] [rbp-2280h]
  int v54; // [rsp+30h] [rbp-2280h]
  int v55; // [rsp+30h] [rbp-2280h]
  __int64 v56; // [rsp+30h] [rbp-2280h]
  int v57; // [rsp+38h] [rbp-2278h]
  int v58; // [rsp+38h] [rbp-2278h]
  __int64 v59; // [rsp+38h] [rbp-2278h]
  int v60; // [rsp+38h] [rbp-2278h]
  _BYTE v61[5832]; // [rsp+40h] [rbp-2270h] BYREF
  char v62; // [rsp+1708h] [rbp-BA8h] BYREF

  *(_QWORD *)a1 = 0;
  *(_DWORD *)(a1 + 24) = 4;
  v2 = (_QWORD *)sub_C7D670(64, 8);
  v3 = *(unsigned int *)(a1 + 24);
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 8) = v2;
  for ( i = &v2[2 * v3]; i != v2; v2 += 2 )
  {
    if ( v2 )
    {
      *v2 = -1;
      v2[1] = 0;
    }
  }
  v5 = (const char **)v61;
  sub_C0EEA0(a1, (const char **)&off_4C5C700 - 2, (const char **)&off_4C5C700);
  sub_E42970(v61, a2);
  do
  {
    v6 = *v5;
    if ( !*v5 )
      goto LABEL_22;
    v7 = strlen(*v5);
    v8 = *(_DWORD *)(a1 + 24);
    v9 = v7;
    if ( !v8 )
    {
      ++*(_QWORD *)a1;
LABEL_9:
      sub_BA8070(a1, 2 * v8);
      v10 = 0;
      v57 = *(_DWORD *)(a1 + 24);
      if ( !v57 )
        goto LABEL_18;
      v53 = *(_QWORD *)(a1 + 8);
      v11 = sub_C94890(v6, v9);
      v12 = 1;
      v13 = 0;
      v14 = v57 - 1;
      for ( j = (v57 - 1) & v11; ; j = v14 & v18 )
      {
        v10 = v53 + 16LL * j;
        v16 = *(const void **)v10;
        if ( *(_QWORD *)v10 == -1 )
          goto LABEL_52;
        v17 = v6 + 2 == 0;
        if ( v16 != (const void *)-2LL )
        {
          if ( v9 != *(_QWORD *)(v10 + 8) )
            goto LABEL_14;
          v44 = v12;
          v47 = v13;
          v50 = j;
          v58 = v14;
          if ( !v9 )
            goto LABEL_18;
          v40 = v53 + 16LL * j;
          v19 = memcmp(v6, v16, v9);
          v10 = v40;
          v14 = v58;
          j = v50;
          v13 = v47;
          v12 = v44;
          v17 = v19 == 0;
        }
        if ( v17 )
          goto LABEL_18;
        if ( v16 == (const void *)-2LL && !v13 )
          v13 = v10;
LABEL_14:
        v18 = v12 + j;
        ++v12;
      }
    }
    v54 = *(_DWORD *)(a1 + 24);
    v59 = *(_QWORD *)(a1 + 8);
    v22 = sub_C94890(v6, v7);
    v23 = 1;
    v10 = 0;
    v24 = v54 - 1;
    for ( k = (v54 - 1) & v22; ; k = v24 & v29 )
    {
      v26 = v59 + 16LL * k;
      v27 = *(const void **)v26;
      v28 = v6 + 1 == 0;
      if ( *(_QWORD *)v26 != -1 )
      {
        v28 = v6 + 2 == 0;
        if ( v27 != (const void *)-2LL )
        {
          if ( v9 != *(_QWORD *)(v26 + 8) )
            goto LABEL_28;
          v42 = v59 + 16LL * k;
          v45 = v10;
          v48 = v23;
          v51 = k;
          v55 = v24;
          if ( !v9 )
            goto LABEL_22;
          v30 = memcmp(v6, v27, v9);
          v24 = v55;
          k = v51;
          v23 = v48;
          v10 = v45;
          v26 = v42;
          v28 = v30 == 0;
        }
      }
      if ( v28 )
        goto LABEL_22;
      if ( v27 == (const void *)-1LL )
        break;
LABEL_28:
      if ( !v10 && v27 == (const void *)-2LL )
        v10 = v26;
      v29 = v23 + k;
      ++v23;
    }
    v31 = *(_DWORD *)(a1 + 16);
    v8 = *(_DWORD *)(a1 + 24);
    if ( !v10 )
      v10 = v26;
    ++*(_QWORD *)a1;
    v20 = v31 + 1;
    if ( 4 * v20 >= 3 * v8 )
      goto LABEL_9;
    if ( v8 - (v20 + *(_DWORD *)(a1 + 20)) > v8 >> 3 )
      goto LABEL_19;
    sub_BA8070(a1, v8);
    v10 = 0;
    v60 = *(_DWORD *)(a1 + 24);
    if ( !v60 )
      goto LABEL_18;
    v56 = *(_QWORD *)(a1 + 8);
    v32 = sub_C94890(v6, v9);
    v33 = 1;
    v13 = 0;
    v34 = v60 - 1;
    v35 = (v60 - 1) & v32;
    while ( 2 )
    {
      v10 = v56 + 16LL * v35;
      v36 = *(const void **)v10;
      if ( *(_QWORD *)v10 != -1 )
      {
        v37 = v6 + 2 == 0;
        if ( v36 == (const void *)-2LL )
        {
LABEL_47:
          if ( v37 )
            goto LABEL_18;
          if ( v36 == (const void *)-1LL )
            goto LABEL_53;
        }
        else if ( v9 == *(_QWORD *)(v10 + 8) )
        {
          v41 = v33;
          v43 = v13;
          v46 = v35;
          v49 = v34;
          if ( !v9 )
            goto LABEL_18;
          v52 = v56 + 16LL * v35;
          v38 = memcmp(v6, v36, v9);
          v10 = v52;
          v34 = v49;
          v35 = v46;
          v13 = v43;
          v33 = v41;
          v37 = v38 == 0;
          goto LABEL_47;
        }
        if ( v13 || v36 != (const void *)-2LL )
          v10 = v13;
        v39 = v33 + v35;
        v13 = v10;
        ++v33;
        v35 = v34 & v39;
        continue;
      }
      break;
    }
LABEL_52:
    if ( v6 == (const char *)-1LL )
      goto LABEL_18;
LABEL_53:
    if ( v13 )
      v10 = v13;
LABEL_18:
    v20 = *(_DWORD *)(a1 + 16) + 1;
LABEL_19:
    *(_DWORD *)(a1 + 16) = v20;
    if ( *(_QWORD *)v10 != -1 )
      --*(_DWORD *)(a1 + 20);
    *(_QWORD *)v10 = v6;
    *(_QWORD *)(v10 + 8) = v9;
LABEL_22:
    ++v5;
  }
  while ( &v62 != (char *)v5 );
  return a1;
}
