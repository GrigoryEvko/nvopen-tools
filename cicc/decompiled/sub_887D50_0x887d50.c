// Function: sub_887D50
// Address: 0x887d50
//
char **sub_887D50()
{
  char *v0; // r14
  char **v1; // r12
  __int64 v2; // rdx
  __int64 v3; // r8
  __int64 v4; // r9
  int v5; // eax
  size_t v6; // rsi
  _QWORD *v7; // r12
  _QWORD *v8; // rax
  _QWORD *v9; // rcx
  _QWORD *v10; // rdx
  char **v11; // r12
  size_t v12; // rsi
  __int16 v13; // r15
  char *i; // r14
  unsigned __int64 v15; // r9
  _QWORD *v16; // r14
  int v17; // esi
  unsigned int v18; // edx
  __int64 v19; // rdi
  __int64 v20; // r10
  __int64 v21; // rax
  __int16 v22; // dx
  __int64 v23; // rdi
  unsigned int v24; // edx
  int v25; // eax
  char **result; // rax
  unsigned int v27; // edx
  int v28; // eax
  unsigned int v29; // r15d
  _QWORD *v30; // rax
  __int64 v31; // r8
  __int64 v32; // rdx
  unsigned __int64 v33; // r9
  __int64 v34; // rcx
  _QWORD *v35; // rsi
  __int64 v36; // r10
  unsigned __int64 *v37; // rsi
  __int64 v38; // r11
  unsigned __int64 v39; // rdi
  unsigned __int64 k; // rdx
  unsigned int v41; // edx
  __int64 v42; // rax
  _QWORD *v43; // rax
  _QWORD *v44; // rsi
  unsigned __int64 *v45; // rsi
  __int64 v46; // r11
  unsigned __int64 v47; // rdi
  unsigned __int64 j; // rdx
  unsigned int v49; // edx
  __int64 v50; // rax
  unsigned int v51; // [rsp+4h] [rbp-7Ch]
  unsigned int v52; // [rsp+4h] [rbp-7Ch]
  unsigned int v53; // [rsp+8h] [rbp-78h]
  unsigned int v54; // [rsp+8h] [rbp-78h]
  unsigned int v55; // [rsp+Ch] [rbp-74h]
  unsigned int v56; // [rsp+Ch] [rbp-74h]
  __int64 v57[14]; // [rsp+10h] [rbp-70h] BYREF

  v0 = "is_constant_evaluated";
  v1 = (char **)&off_49770C8;
  memset(qword_4D04BE0, 0, 0x1FFFA8u);
  memset(qword_4D04A60, 0, sizeof(qword_4D04A60));
  dword_4F04C30 = 0;
  v5 = sub_880E90((__int64)qword_4D04BE0, 0, v2, 0, v3, v4);
  v6 = 21;
  qword_4F60018 = 0;
  qword_4F60010 = 0;
  dword_4F066A8 = v5;
  qword_4F60008 = 0;
  unk_4D049E8 = 0;
  qword_4F60000 = 0;
  qword_4F5FFF8 = 0;
  qword_4F5FFF0 = 0;
  qword_4F5FFE8 = 0;
  qword_4F5FFD8 = 0;
  qword_4F5FFE0 = 0;
  qword_4F5FE50 = 0;
  qword_4F5FFD0 = 0;
  qword_4F600F0 = 0;
  qword_4F600E8 = 0;
  qword_4F600E0 = 0;
  qword_4F600C8 = 0;
  qword_4F600C0 = 0;
  qword_4F600D8 = 0;
  qword_4F600D0 = 0;
  qword_4F5FFA8 = 0;
  qword_4F5FFA0 = 0;
  qword_4F5FED0 = 0;
  qword_4F5FE58 = 0;
  while ( 1 )
  {
    sub_878540(v0, v6, v57);
    *(_BYTE *)(v57[0] + 73) |= 2u;
    if ( &unk_49772F0 == (_UNKNOWN *)v1 )
      break;
    v0 = *v1++;
    v6 = strlen(v0);
  }
  v7 = (_QWORD *)sub_823970(16);
  qword_4F5FE48 = v7;
  if ( v7 )
  {
    v8 = (_QWORD *)sub_823970(1024);
    v9 = v8;
    v10 = v8 + 128;
    do
    {
      if ( v8 )
        *v8 = 0;
      v8 += 2;
    }
    while ( v10 != v8 );
    *v7 = v9;
    v7[1] = 63;
  }
  v11 = (char **)&unk_4976FD0;
  v12 = 22;
  v13 = 339;
  for ( i = "__add_lvalue_reference"; ; v12 = strlen(i) )
  {
    sub_878540(i, v12, v57);
    v15 = v57[0];
    v16 = qword_4F5FE48;
    *(_BYTE *)(v57[0] + 73) |= 2u;
    v17 = *((_DWORD *)v16 + 2);
    v18 = v17 & (v15 >> 3);
    v19 = 16LL * v18;
    v20 = *v16 + v19;
    if ( !*(_QWORD *)v20 )
      break;
    do
    {
      v18 = v17 & (v18 + 1);
      v21 = *v16 + 16LL * v18;
    }
    while ( *(_QWORD *)v21 );
    v22 = *(_WORD *)(v20 + 8);
    *(_QWORD *)v21 = *(_QWORD *)v20;
    *(_WORD *)(v21 + 8) = v22;
    *(_QWORD *)v20 = 0;
    v23 = *v16 + v19;
    *(_QWORD *)v23 = v15;
    *(_WORD *)(v23 + 8) = v13;
    v24 = *((_DWORD *)v16 + 2);
    v25 = *((_DWORD *)v16 + 3) + 1;
    *((_DWORD *)v16 + 3) = v25;
    if ( 2 * v25 > v24 )
    {
      v54 = v24;
      v52 = v24 + 1;
      v29 = 2 * v24 + 1;
      v56 = 2 * v24 + 2;
      v43 = (_QWORD *)sub_823970(16LL * v56);
      v32 = v54;
      v33 = v52;
      v34 = (__int64)v43;
      if ( v56 )
      {
        v44 = &v43[2 * v29 + 2];
        do
        {
          if ( v43 )
            *v43 = 0;
          v43 += 2;
        }
        while ( v43 != v44 );
      }
      v36 = *v16;
      if ( v52 )
      {
        v32 = 16LL * v54;
        v45 = (unsigned __int64 *)*v16;
        v46 = v36 + v32 + 16;
        do
        {
          v47 = *v45;
          if ( *v45 )
          {
            for ( j = v47 >> 3; ; LODWORD(j) = v49 + 1 )
            {
              v49 = v29 & j;
              v50 = v34 + 16LL * v49;
              if ( !*(_QWORD *)v50 )
                break;
            }
            *(_QWORD *)v50 = v47;
            v32 = *((unsigned __int16 *)v45 + 4);
            *(_WORD *)(v50 + 8) = v32;
          }
          v45 += 2;
        }
        while ( (unsigned __int64 *)v46 != v45 );
      }
      goto LABEL_30;
    }
LABEL_14:
    result = &off_49770C0;
    if ( &off_49770C0 == v11 )
      return result;
LABEL_15:
    i = v11[1];
    v13 = *(_WORD *)v11;
    v11 += 2;
  }
  *(_QWORD *)v20 = v15;
  *(_WORD *)(v20 + 8) = v13;
  v27 = *((_DWORD *)v16 + 2);
  v28 = *((_DWORD *)v16 + 3) + 1;
  *((_DWORD *)v16 + 3) = v28;
  if ( 2 * v28 <= v27 )
    goto LABEL_14;
  v53 = v27;
  v51 = v27 + 1;
  v29 = 2 * v27 + 1;
  v55 = 2 * v27 + 2;
  v30 = (_QWORD *)sub_823970(16LL * v55);
  v32 = v53;
  v33 = v51;
  v34 = (__int64)v30;
  if ( v55 )
  {
    v35 = &v30[2 * v29 + 2];
    do
    {
      if ( v30 )
        *v30 = 0;
      v30 += 2;
    }
    while ( v35 != v30 );
  }
  v36 = *v16;
  if ( v51 )
  {
    v32 = 16LL * v53;
    v37 = (unsigned __int64 *)*v16;
    v38 = v36 + v32 + 16;
    do
    {
      while ( 1 )
      {
        v39 = *v37;
        if ( *v37 )
          break;
        v37 += 2;
        if ( (unsigned __int64 *)v38 == v37 )
          goto LABEL_30;
      }
      for ( k = v39 >> 3; ; LODWORD(k) = v41 + 1 )
      {
        v41 = v29 & k;
        v42 = v34 + 16LL * v41;
        if ( !*(_QWORD *)v42 )
          break;
      }
      *(_QWORD *)v42 = v39;
      v32 = *((unsigned __int16 *)v37 + 4);
      v37 += 2;
      *(_WORD *)(v42 + 8) = v32;
    }
    while ( (unsigned __int64 *)v38 != v37 );
  }
LABEL_30:
  *v16 = v34;
  *((_DWORD *)v16 + 2) = v29;
  sub_823A00(v36, 16LL * (unsigned int)v33, v32, v34, v31, (__int64 *)v33);
  result = &off_49770C0;
  if ( &off_49770C0 != v11 )
    goto LABEL_15;
  return result;
}
