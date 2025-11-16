// Function: sub_E2E360
// Address: 0xe2e360
//
void __fastcall sub_E2E360(__int64 a1, __int64 *a2, unsigned int a3)
{
  _DWORD *v6; // rdi
  void (*v7)(void); // rax
  int v8; // eax
  __int64 *v9; // rdi
  unsigned __int64 (__fastcall *v10)(__int64, char **, unsigned int); // rax
  __int64 v11; // rsi
  unsigned __int64 v12; // rax
  char *v13; // rdi
  int v14; // eax
  char v15; // si
  __int64 v16; // rsi
  unsigned __int64 v17; // rax
  char *v18; // rdi
  unsigned __int64 v19; // rsi
  unsigned __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rsi
  unsigned __int64 v23; // rax
  void *v24; // rdi
  unsigned __int64 v25; // rsi
  unsigned __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rsi
  unsigned __int64 v29; // rax
  char *v30; // rdi
  unsigned __int64 v31; // rsi
  unsigned __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rsi
  unsigned __int64 v35; // rax
  void *v36; // rdi
  unsigned __int64 v37; // rsi
  unsigned __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rsi
  unsigned __int64 v41; // rax
  void *v42; // rdi
  unsigned __int64 v43; // rsi
  unsigned __int64 v44; // rax
  __int64 v45; // rax
  unsigned __int64 v46; // rsi
  unsigned __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rsi
  unsigned __int64 v50; // rax
  void *v51; // rdi
  unsigned __int64 v52; // rsi
  unsigned __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rsi
  unsigned __int64 v56; // rax
  unsigned __int64 v57; // rsi
  unsigned __int64 v58; // rax
  __int64 v59; // rax

  v6 = *(_DWORD **)(a1 + 32);
  v7 = *(void (**)(void))(*(_QWORD *)v6 + 24LL);
  if ( v6[2] != 3 )
  {
    v7();
    sub_E2A040((__int64)a2);
    if ( (*(_BYTE *)(a1 + 12) & 0x10) == 0 )
      goto LABEL_3;
LABEL_16:
    v16 = a2[1];
    v17 = a2[2];
    v18 = (char *)*a2;
    if ( v16 + 12 > v17 )
    {
      v19 = v16 + 1004;
      v20 = 2 * v17;
      if ( v19 > v20 )
        a2[2] = v19;
      else
        a2[2] = v20;
      v21 = realloc(v18);
      *a2 = v21;
      v18 = (char *)v21;
      if ( !v21 )
        goto LABEL_72;
      v16 = a2[1];
    }
    qmemcpy(&v18[v16], "__unaligned ", 12);
    a2[1] += 12;
    v8 = *(_DWORD *)(*(_QWORD *)(a1 + 32) + 8LL);
    if ( v8 != 16 )
      goto LABEL_4;
    goto LABEL_22;
  }
  ((void (__fastcall *)(_DWORD *, __int64 *, __int64))v7)(v6, a2, 1);
  sub_E2A040((__int64)a2);
  if ( (*(_BYTE *)(a1 + 12) & 0x10) != 0 )
    goto LABEL_16;
LABEL_3:
  v8 = *(_DWORD *)(*(_QWORD *)(a1 + 32) + 8LL);
  if ( v8 != 16 )
  {
LABEL_4:
    if ( v8 == 3 )
    {
      v49 = a2[1];
      v50 = a2[2];
      v51 = (void *)*a2;
      if ( v49 + 1 > v50 )
      {
        v52 = v49 + 993;
        v53 = 2 * v50;
        if ( v52 > v53 )
          a2[2] = v52;
        else
          a2[2] = v53;
        v54 = realloc(v51);
        *a2 = v54;
        v51 = (void *)v54;
        if ( !v54 )
          goto LABEL_72;
        v49 = a2[1];
      }
      *((_BYTE *)v51 + v49) = 40;
      ++a2[1];
      sub_E2B3D0(a2, *(_BYTE *)(*(_QWORD *)(a1 + 32) + 20LL));
      v55 = a2[1];
      v56 = a2[2];
      if ( v55 + 1 <= v56 )
      {
        v59 = *a2;
      }
      else
      {
        v57 = v55 + 993;
        v58 = 2 * v56;
        if ( v57 > v58 )
          a2[2] = v57;
        else
          a2[2] = v58;
        v59 = realloc((void *)*a2);
        *a2 = v59;
        if ( !v59 )
          goto LABEL_72;
        v55 = a2[1];
      }
      *(_BYTE *)(v59 + v55) = 32;
      ++a2[1];
    }
    goto LABEL_5;
  }
LABEL_22:
  v22 = a2[1];
  v23 = a2[2];
  v24 = (void *)*a2;
  if ( v22 + 1 > v23 )
  {
    v25 = v22 + 993;
    v26 = 2 * v23;
    if ( v25 > v26 )
      a2[2] = v25;
    else
      a2[2] = v26;
    v27 = realloc(v24);
    *a2 = v27;
    v24 = (void *)v27;
    if ( !v27 )
      goto LABEL_72;
    v22 = a2[1];
  }
  *((_BYTE *)v24 + v22) = 40;
  ++a2[1];
LABEL_5:
  v9 = *(__int64 **)(a1 + 24);
  if ( v9 )
  {
    v10 = *(unsigned __int64 (__fastcall **)(__int64, char **, unsigned int))(*v9 + 16);
    if ( v10 == sub_E2CA10 )
      sub_E2C8E0(v9[2], (char **)a2, a3, 2u, "::");
    else
      v10((__int64)v9, (char **)a2, a3);
    v11 = a2[1];
    v12 = a2[2];
    v13 = (char *)*a2;
    if ( v11 + 2 > v12 )
    {
      v46 = v11 + 994;
      v47 = 2 * v12;
      if ( v46 > v47 )
        a2[2] = v46;
      else
        a2[2] = v47;
      v48 = realloc(v13);
      *a2 = v48;
      v13 = (char *)v48;
      if ( !v48 )
        goto LABEL_72;
      v11 = a2[1];
    }
    *(_WORD *)&v13[v11] = 14906;
    a2[1] += 2;
  }
  v14 = *(_DWORD *)(a1 + 16);
  if ( v14 == 2 )
  {
    v34 = a2[1];
    v35 = a2[2];
    v36 = (void *)*a2;
    if ( v34 + 1 > v35 )
    {
      v37 = v34 + 993;
      v38 = 2 * v35;
      if ( v37 > v38 )
        a2[2] = v37;
      else
        a2[2] = v38;
      v39 = realloc(v36);
      *a2 = v39;
      v36 = (void *)v39;
      if ( !v39 )
        goto LABEL_72;
      v34 = a2[1];
    }
    *((_BYTE *)v36 + v34) = 38;
    ++a2[1];
    goto LABEL_13;
  }
  if ( v14 == 3 )
  {
    v28 = a2[1];
    v29 = a2[2];
    v30 = (char *)*a2;
    if ( v28 + 2 > v29 )
    {
      v31 = v28 + 994;
      v32 = 2 * v29;
      if ( v31 > v32 )
        a2[2] = v31;
      else
        a2[2] = v32;
      v33 = realloc(v30);
      *a2 = v33;
      v30 = (char *)v33;
      if ( !v33 )
        goto LABEL_72;
      v28 = a2[1];
    }
    *(_WORD *)&v30[v28] = 9766;
    a2[1] += 2;
    goto LABEL_13;
  }
  if ( v14 != 1 )
    goto LABEL_13;
  v40 = a2[1];
  v41 = a2[2];
  v42 = (void *)*a2;
  if ( v40 + 1 <= v41 )
    goto LABEL_46;
  v43 = v40 + 993;
  v44 = 2 * v41;
  if ( v43 > v44 )
    a2[2] = v43;
  else
    a2[2] = v44;
  v45 = realloc(v42);
  *a2 = v45;
  v42 = (void *)v45;
  if ( !v45 )
LABEL_72:
    abort();
  v40 = a2[1];
LABEL_46:
  *((_BYTE *)v42 + v40) = 42;
  ++a2[1];
LABEL_13:
  v15 = *(_BYTE *)(a1 + 12);
  if ( v15 )
    sub_E2A820(a2, v15, 0, 0);
}
