// Function: sub_E2B920
// Address: 0xe2b920
//
void __fastcall sub_E2B920(__int64 a1, void **a2, unsigned int a3)
{
  __int16 v6; // ax
  __int16 v7; // ax
  __int64 v8; // rdi
  char *v9; // rsi
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  unsigned __int64 v12; // rsi
  unsigned __int64 v13; // rax
  char *v14; // rsi
  unsigned __int64 v15; // rax
  __int64 v16; // rdi
  unsigned __int64 v17; // rsi
  unsigned __int64 v18; // rax
  __int64 v19; // rax
  char *v20; // rsi
  unsigned __int64 v21; // rax
  char *v22; // rdi
  unsigned __int64 v23; // rsi
  unsigned __int64 v24; // rax
  __int64 v25; // rax
  char *v26; // rsi
  unsigned __int64 v27; // rax
  char *v28; // rdi
  unsigned __int64 v29; // rsi
  unsigned __int64 v30; // rax
  __int64 v31; // rax
  char *v32; // rdi
  char *v33; // rsi
  unsigned __int64 v34; // rax
  char *v35; // rdi
  unsigned __int64 v36; // rsi
  unsigned __int64 v37; // rax
  __int64 v38; // rax
  char *v39; // rsi
  unsigned __int64 v40; // rax
  __int64 v41; // rdi
  unsigned __int64 v42; // rsi
  unsigned __int64 v43; // rax
  __int64 v44; // rax
  char *v45; // rax
  unsigned __int64 v46; // rdx
  char *v47; // rdi
  unsigned __int64 v48; // rdx
  __int64 v49; // rax
  char *v50; // rdi

  if ( (a3 & 4) != 0 )
    goto LABEL_5;
  v6 = *(_WORD *)(a1 + 22);
  if ( (v6 & 1) != 0 )
  {
    v14 = (char *)a2[1];
    v15 = (unsigned __int64)a2[2];
    v16 = (__int64)*a2;
    if ( (unsigned __int64)(v14 + 8) > v15 )
    {
      v17 = (unsigned __int64)(v14 + 1000);
      v18 = 2 * v15;
      if ( v17 > v18 )
        a2[2] = (void *)v17;
      else
        a2[2] = (void *)v18;
      v19 = realloc((void *)v16);
      *a2 = (void *)v19;
      v16 = v19;
      if ( !v19 )
        goto LABEL_65;
      v14 = (char *)a2[1];
    }
    *(_QWORD *)&v14[v16] = 0x203A63696C627570LL;
    a2[1] = (char *)a2[1] + 8;
    v6 = *(_WORD *)(a1 + 22);
    if ( (v6 & 2) == 0 )
    {
LABEL_4:
      if ( (v6 & 4) == 0 )
        goto LABEL_5;
      goto LABEL_33;
    }
  }
  else if ( (v6 & 2) == 0 )
  {
    goto LABEL_4;
  }
  v20 = (char *)a2[1];
  v21 = (unsigned __int64)a2[2];
  v22 = (char *)*a2;
  if ( (unsigned __int64)(v20 + 11) > v21 )
  {
    v23 = (unsigned __int64)(v20 + 1003);
    v24 = 2 * v21;
    if ( v23 > v24 )
      a2[2] = (void *)v23;
    else
      a2[2] = (void *)v24;
    v25 = realloc(v22);
    *a2 = (void *)v25;
    v22 = (char *)v25;
    if ( !v25 )
      goto LABEL_65;
    v20 = (char *)a2[1];
  }
  qmemcpy(&v22[(_QWORD)v20], "protected: ", 11);
  a2[1] = (char *)a2[1] + 11;
  if ( (*(_WORD *)(a1 + 22) & 4) != 0 )
  {
LABEL_33:
    v26 = (char *)a2[1];
    v27 = (unsigned __int64)a2[2];
    v28 = (char *)*a2;
    if ( (unsigned __int64)(v26 + 9) > v27 )
    {
      v29 = (unsigned __int64)(v26 + 1001);
      v30 = 2 * v27;
      if ( v29 > v30 )
        a2[2] = (void *)v29;
      else
        a2[2] = (void *)v30;
      v31 = realloc(v28);
      *a2 = (void *)v31;
      v28 = (char *)v31;
      if ( !v31 )
        goto LABEL_65;
      v26 = (char *)a2[1];
    }
    v32 = &v28[(_QWORD)v26];
    *(_QWORD *)v32 = 0x3A65746176697270LL;
    v32[8] = 32;
    a2[1] = (char *)a2[1] + 9;
  }
LABEL_5:
  if ( (a3 & 8) != 0 )
    goto LABEL_10;
  v7 = *(_WORD *)(a1 + 22);
  if ( (v7 & 8) == 0 && (v7 & 0x10) != 0 )
  {
    v45 = (char *)a2[1];
    v46 = (unsigned __int64)a2[2];
    v47 = (char *)*a2;
    if ( (unsigned __int64)(v45 + 7) > v46 )
    {
      v48 = 2 * v46;
      if ( (unsigned __int64)(v45 + 999) > v48 )
        a2[2] = v45 + 999;
      else
        a2[2] = (void *)v48;
      v49 = realloc(v47);
      *a2 = (void *)v49;
      v47 = (char *)v49;
      if ( !v49 )
        goto LABEL_65;
      v45 = (char *)a2[1];
    }
    v50 = &v47[(_QWORD)v45];
    *(_DWORD *)v50 = 1952543859;
    *((_WORD *)v50 + 2) = 25449;
    v50[6] = 32;
    a2[1] = (char *)a2[1] + 7;
    v7 = *(_WORD *)(a1 + 22);
  }
  if ( (v7 & 0x20) == 0 )
  {
    if ( (v7 & 0x80u) == 0 )
      goto LABEL_10;
    goto LABEL_39;
  }
  v39 = (char *)a2[1];
  v40 = (unsigned __int64)a2[2];
  v41 = (__int64)*a2;
  if ( (unsigned __int64)(v39 + 8) > v40 )
  {
    v42 = (unsigned __int64)(v39 + 1000);
    v43 = 2 * v40;
    if ( v42 > v43 )
      a2[2] = (void *)v42;
    else
      a2[2] = (void *)v43;
    v44 = realloc((void *)v41);
    *a2 = (void *)v44;
    v41 = v44;
    if ( !v44 )
      goto LABEL_65;
    v39 = (char *)a2[1];
  }
  *(_QWORD *)&v39[v41] = 0x206C617574726976LL;
  a2[1] = (char *)a2[1] + 8;
  if ( (*(_WORD *)(a1 + 22) & 0x80u) != 0 )
  {
LABEL_39:
    v33 = (char *)a2[1];
    v34 = (unsigned __int64)a2[2];
    v35 = (char *)*a2;
    if ( (unsigned __int64)(v33 + 11) > v34 )
    {
      v36 = (unsigned __int64)(v33 + 1003);
      v37 = 2 * v34;
      if ( v36 > v37 )
        a2[2] = (void *)v36;
      else
        a2[2] = (void *)v37;
      v38 = realloc(v35);
      *a2 = (void *)v38;
      v35 = (char *)v38;
      if ( !v38 )
        goto LABEL_65;
      v33 = (char *)a2[1];
    }
    qmemcpy(&v35[(_QWORD)v33], "extern \"C\" ", 11);
    a2[1] = (char *)a2[1] + 11;
  }
LABEL_10:
  if ( (a3 & 0x10) != 0 )
    goto LABEL_15;
  v8 = *(_QWORD *)(a1 + 32);
  if ( !v8 )
    goto LABEL_15;
  (*(void (__fastcall **)(__int64, void **, _QWORD))(*(_QWORD *)v8 + 24LL))(v8, a2, a3);
  v9 = (char *)a2[1];
  v10 = (unsigned __int64)a2[2];
  if ( (unsigned __int64)(v9 + 1) > v10 )
  {
    v12 = (unsigned __int64)(v9 + 993);
    v13 = 2 * v10;
    if ( v12 > v13 )
      a2[2] = (void *)v12;
    else
      a2[2] = (void *)v13;
    v11 = realloc(*a2);
    *a2 = (void *)v11;
    if ( v11 )
    {
      v9 = (char *)a2[1];
      goto LABEL_14;
    }
LABEL_65:
    abort();
  }
  v11 = (__int64)*a2;
LABEL_14:
  v9[v11] = 32;
  a2[1] = (char *)a2[1] + 1;
LABEL_15:
  if ( (a3 & 1) == 0 )
    sub_E2B3D0((__int64 *)a2, *(_BYTE *)(a1 + 20));
}
