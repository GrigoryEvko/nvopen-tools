// Function: sub_129F850
// Address: 0x129f850
//
__int64 __fastcall sub_129F850(__int64 a1, unsigned int a2)
{
  char *v3; // rdx
  char v4; // r9
  char *v5; // rax
  char *v6; // rcx
  __int64 v7; // rsi
  const char *v8; // r13
  int v9; // eax
  int v10; // edx
  int v11; // r8d
  int v12; // ebx
  const char *v13; // rsi
  int v14; // edx
  __int64 v15; // r12
  const char *v17; // r13
  const char **v18; // rax
  __int64 v19; // rax
  __int64 v20; // rsi
  unsigned int v21; // ecx
  const char **v22; // rdx
  const char *v23; // r8
  __int64 v24; // rax
  __int64 v25; // rcx
  const char *v26; // r12
  int v27; // edx
  int v28; // r8d
  int v29; // edx
  int v30; // eax
  __int64 v31; // rax
  unsigned int v32; // esi
  __int64 v33; // rdi
  unsigned int v34; // ecx
  const char **v35; // r14
  const char *v36; // rdx
  _QWORD *v37; // r13
  int v38; // edx
  int v39; // r9d
  int v40; // r10d
  const char **v41; // r9
  int v42; // ecx
  int v43; // ecx
  int v44; // eax
  int v45; // edx
  __int64 v46; // rdi
  unsigned int v47; // eax
  const char *v48; // rsi
  int v49; // r9d
  const char **v50; // r8
  int v51; // edx
  int v52; // edx
  __int64 v53; // rdi
  int v54; // r9d
  unsigned int v55; // eax
  const char *v56; // rsi
  int v57; // [rsp+8h] [rbp-78h]
  unsigned int v58; // [rsp+8h] [rbp-78h]
  _BYTE v59[16]; // [rsp+10h] [rbp-70h] BYREF
  char v60; // [rsp+20h] [rbp-60h]
  _BYTE v61[24]; // [rsp+30h] [rbp-50h] BYREF
  char v62; // [rsp+48h] [rbp-38h]

  if ( !a2 )
  {
    v3 = *(char **)(a1 + 8);
    v60 = 0;
    v62 = 0;
    v4 = *v3;
    v5 = v3;
    if ( *v3 == 15 )
    {
      v6 = v3;
    }
    else
    {
      v5 = *(char **)&v3[-8 * *((unsigned int *)v3 + 2)];
      if ( !v5 )
      {
        v8 = byte_3F871B3;
        v12 = 0;
        v14 = 0;
        v13 = byte_3F871B3;
        return sub_15A56E0((int)a1 + 16, (_DWORD)v13, v14, (_DWORD)v8, v12, (unsigned int)v61, (__int64)v59);
      }
      v6 = *(char **)&v3[-8 * *((unsigned int *)v3 + 2)];
    }
    v7 = *((unsigned int *)v5 + 2);
    v8 = *(const char **)&v6[8 * (1 - v7)];
    if ( v8 )
    {
      v9 = sub_161E970(*(_QWORD *)&v6[8 * (1 - v7)]);
      v11 = v10;
      v3 = *(char **)(a1 + 8);
      LODWORD(v8) = v9;
      v4 = *v3;
    }
    else
    {
      v11 = 0;
    }
    v12 = v11;
    v13 = *(const char **)&v3[-8 * *((unsigned int *)v3 + 2)];
    if ( v4 == 15 )
      goto LABEL_10;
    if ( v13 )
    {
      v13 = *(const char **)&v13[-8 * *((unsigned int *)v13 + 2)];
LABEL_10:
      if ( v13 )
        LODWORD(v13) = sub_161E970(v13);
      else
        v14 = 0;
      return sub_15A56E0((int)a1 + 16, (_DWORD)v13, v14, (_DWORD)v8, v12, (unsigned int)v61, (__int64)v59);
    }
    v14 = 0;
    v13 = byte_3F871B3;
    return sub_15A56E0((int)a1 + 16, (_DWORD)v13, v14, (_DWORD)v8, v12, (unsigned int)v61, (__int64)v59);
  }
  v17 = "<unknown>";
  v18 = (const char **)sub_129E300(a2, 0);
  if ( v18 )
    v17 = *v18;
  v19 = *(unsigned int *)(a1 + 600);
  if ( (_DWORD)v19 )
  {
    v20 = *(_QWORD *)(a1 + 584);
    v21 = (v19 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
    v22 = (const char **)(v20 + 16LL * v21);
    v23 = *v22;
    if ( v17 == *v22 )
    {
LABEL_18:
      if ( v22 != (const char **)(v20 + 16 * v19) )
      {
        v15 = (__int64)v22[1];
        if ( v15 )
          return v15;
      }
    }
    else
    {
      v38 = 1;
      while ( v23 != (const char *)-1LL )
      {
        v39 = v38 + 1;
        v21 = (v19 - 1) & (v38 + v21);
        v22 = (const char **)(v20 + 16LL * v21);
        v23 = *v22;
        if ( v17 == *v22 )
          goto LABEL_18;
        v38 = v39;
      }
    }
  }
  v24 = *(_QWORD *)(a1 + 8);
  v60 = 0;
  v62 = 0;
  if ( *(_BYTE *)v24 == 15 || (v24 = *(_QWORD *)(v24 - 8LL * *(unsigned int *)(v24 + 8))) != 0 )
  {
    v25 = *(unsigned int *)(v24 + 8);
    v26 = *(const char **)(v24 + 8 * (1 - v25));
    if ( v26 )
    {
      LODWORD(v26) = sub_161E970(*(_QWORD *)(v24 + 8 * (1 - v25)));
      v28 = v27;
    }
    else
    {
      v28 = 0;
    }
  }
  else
  {
    v28 = 0;
    v26 = byte_3F871B3;
  }
  v29 = 0;
  if ( v17 )
  {
    v57 = v28;
    v30 = strlen(v17);
    v28 = v57;
    v29 = v30;
  }
  v31 = sub_15A56E0((int)a1 + 16, (_DWORD)v17, v29, (_DWORD)v26, v28, (unsigned int)v61, (__int64)v59);
  v32 = *(_DWORD *)(a1 + 600);
  v15 = v31;
  if ( !v32 )
  {
    ++*(_QWORD *)(a1 + 576);
    goto LABEL_52;
  }
  v33 = *(_QWORD *)(a1 + 584);
  v34 = (v32 - 1) & (((unsigned int)v17 >> 4) ^ ((unsigned int)v17 >> 9));
  v35 = (const char **)(v33 + 16LL * v34);
  v36 = *v35;
  if ( v17 != *v35 )
  {
    v40 = 1;
    v41 = 0;
    while ( v36 != (const char *)-1LL )
    {
      if ( !v41 && v36 == (const char *)-2LL )
        v41 = v35;
      v34 = (v32 - 1) & (v40 + v34);
      v35 = (const char **)(v33 + 16LL * v34);
      v36 = *v35;
      if ( v17 == *v35 )
        goto LABEL_28;
      ++v40;
    }
    v42 = *(_DWORD *)(a1 + 592);
    if ( v41 )
      v35 = v41;
    ++*(_QWORD *)(a1 + 576);
    v43 = v42 + 1;
    if ( 4 * v43 < 3 * v32 )
    {
      if ( v32 - *(_DWORD *)(a1 + 596) - v43 > v32 >> 3 )
      {
LABEL_48:
        *(_DWORD *)(a1 + 592) = v43;
        if ( *v35 != (const char *)-1LL )
          --*(_DWORD *)(a1 + 596);
        *v35 = v17;
        v37 = v35 + 1;
        v35[1] = 0;
        goto LABEL_30;
      }
      v58 = ((unsigned int)v17 >> 4) ^ ((unsigned int)v17 >> 9);
      sub_129F630(a1 + 576, v32);
      v51 = *(_DWORD *)(a1 + 600);
      if ( v51 )
      {
        v52 = v51 - 1;
        v53 = *(_QWORD *)(a1 + 584);
        v50 = 0;
        v54 = 1;
        v55 = v52 & v58;
        v43 = *(_DWORD *)(a1 + 592) + 1;
        v35 = (const char **)(v53 + 16LL * (v52 & v58));
        v56 = *v35;
        if ( v17 == *v35 )
          goto LABEL_48;
        while ( v56 != (const char *)-1LL )
        {
          if ( !v50 && v56 == (const char *)-2LL )
            v50 = v35;
          v55 = v52 & (v54 + v55);
          v35 = (const char **)(v53 + 16LL * v55);
          v56 = *v35;
          if ( v17 == *v35 )
            goto LABEL_48;
          ++v54;
        }
        goto LABEL_56;
      }
      goto LABEL_78;
    }
LABEL_52:
    sub_129F630(a1 + 576, 2 * v32);
    v44 = *(_DWORD *)(a1 + 600);
    if ( v44 )
    {
      v45 = v44 - 1;
      v46 = *(_QWORD *)(a1 + 584);
      v47 = (v44 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v43 = *(_DWORD *)(a1 + 592) + 1;
      v35 = (const char **)(v46 + 16LL * v47);
      v48 = *v35;
      if ( v17 == *v35 )
        goto LABEL_48;
      v49 = 1;
      v50 = 0;
      while ( v48 != (const char *)-1LL )
      {
        if ( !v50 && v48 == (const char *)-2LL )
          v50 = v35;
        v47 = v45 & (v49 + v47);
        v35 = (const char **)(v46 + 16LL * v47);
        v48 = *v35;
        if ( v17 == *v35 )
          goto LABEL_48;
        ++v49;
      }
LABEL_56:
      if ( v50 )
        v35 = v50;
      goto LABEL_48;
    }
LABEL_78:
    ++*(_DWORD *)(a1 + 592);
    BUG();
  }
LABEL_28:
  v37 = v35 + 1;
  if ( v35[1] )
    sub_161E7C0(v35 + 1);
LABEL_30:
  v35[1] = (const char *)v15;
  if ( v15 )
    sub_1623A60(v37, v15, 2);
  return v15;
}
