// Function: sub_E2AF50
// Address: 0xe2af50
//
void __fastcall sub_E2AF50(__int64 a1, char **a2)
{
  __int64 v4; // rdi
  char *v5; // rax
  unsigned __int64 v6; // rcx
  __int64 v7; // rdi
  unsigned __int64 v8; // rcx
  __int64 v9; // rax
  int v10; // r10d
  __int64 v11; // r9
  _BYTE *v12; // r12
  unsigned __int64 v13; // rcx
  _BYTE *v14; // rdi
  unsigned __int64 v15; // rax
  _BYTE *v16; // r8
  size_t v17; // r13
  __int64 v18; // r15
  char *v19; // rsi
  unsigned __int64 v20; // rax
  __int64 v21; // rdi
  unsigned __int64 v22; // rsi
  unsigned __int64 v23; // rax
  __int64 v24; // rax
  _BYTE *v25; // r12
  __int64 v26; // r8
  __int64 v27; // r9
  unsigned __int64 v28; // rcx
  _BYTE *v29; // rsi
  unsigned __int64 v30; // rax
  _BYTE *v31; // rdx
  int v32; // eax
  char *v33; // rax
  unsigned __int64 v34; // rdx
  char *v35; // rdi
  unsigned __int64 v36; // rdx
  __int64 v37; // rax
  unsigned __int64 v38; // rax
  _BYTE *v39; // rsi
  char *v40; // rdi
  char *v41; // rsi
  unsigned __int64 v42; // rax
  __int64 v43; // rax
  char *v44; // rax
  unsigned __int64 v45; // rdx
  __int64 v46; // rdi
  unsigned __int64 v47; // rdx
  __int64 v48; // rax
  char *v49; // rax
  unsigned __int64 v50; // rcx
  char *v51; // rdi
  unsigned __int64 v52; // rcx
  __int64 v53; // rax
  char *v54; // rcx
  unsigned __int64 v55; // rdx
  char *v56; // rdi
  char *v57; // rax
  unsigned __int64 v58; // rdx
  __int64 v59; // rax
  size_t n; // [rsp+8h] [rbp-78h]
  _BYTE v61[32]; // [rsp+25h] [rbp-5Bh] BYREF
  _BYTE v62[59]; // [rsp+45h] [rbp-3Bh] BYREF

  if ( *(int *)(a1 + 24) > 0 )
  {
    v5 = a2[1];
    v6 = (unsigned __int64)a2[2];
    v7 = (__int64)*a2;
    if ( (unsigned __int64)(v5 + 1) > v6 )
    {
      v8 = 2 * v6;
      if ( (unsigned __int64)(v5 + 993) > v8 )
        a2[2] = v5 + 993;
      else
        a2[2] = (char *)v8;
      v9 = realloc((void *)v7);
      *a2 = (char *)v9;
      v7 = v9;
      if ( !v9 )
        goto LABEL_73;
      v5 = a2[1];
    }
    v5[v7] = 123;
    ++a2[1];
LABEL_12:
    v4 = *(_QWORD *)(a1 + 16);
    if ( !v4 )
      goto LABEL_13;
    goto LABEL_4;
  }
  if ( *(_DWORD *)(a1 + 56) == 1 )
  {
    v49 = a2[1];
    v50 = (unsigned __int64)a2[2];
    v51 = *a2;
    if ( (unsigned __int64)(v49 + 1) > v50 )
    {
      v52 = 2 * v50;
      if ( (unsigned __int64)(v49 + 993) > v52 )
        a2[2] = v49 + 993;
      else
        a2[2] = (char *)v52;
      v53 = realloc(v51);
      *a2 = (char *)v53;
      v51 = (char *)v53;
      if ( !v53 )
        goto LABEL_73;
      v49 = a2[1];
    }
    v49[(_QWORD)v51] = 38;
    ++a2[1];
    goto LABEL_12;
  }
  v4 = *(_QWORD *)(a1 + 16);
  if ( !v4 )
  {
    if ( *(int *)(a1 + 24) > 1 )
      goto LABEL_20;
    return;
  }
LABEL_4:
  (*(void (__fastcall **)(__int64, char **))(*(_QWORD *)v4 + 16LL))(v4, a2);
  if ( *(int *)(a1 + 24) <= 0 )
    return;
  v44 = a2[1];
  v45 = (unsigned __int64)a2[2];
  v46 = (__int64)*a2;
  if ( (unsigned __int64)(v44 + 2) > v45 )
  {
    v47 = 2 * v45;
    if ( (unsigned __int64)(v44 + 994) > v47 )
      a2[2] = v44 + 994;
    else
      a2[2] = (char *)v47;
    v48 = realloc((void *)v46);
    *a2 = (char *)v48;
    v46 = v48;
    if ( !v48 )
      goto LABEL_73;
    v44 = a2[1];
  }
  *(_WORD *)&v44[v46] = 8236;
  a2[1] += 2;
LABEL_13:
  v10 = *(_DWORD *)(a1 + 24);
  if ( v10 > 0 )
  {
    v11 = *(_QWORD *)(a1 + 32);
    v12 = v61;
    v13 = abs64(v11);
    do
    {
      v14 = v12--;
      *v12 = v13 % 0xA + 48;
      v15 = v13;
      v13 /= 0xAu;
    }
    while ( v15 > 9 );
    if ( v11 < 0 )
    {
      *(v12 - 1) = 45;
      v12 = v14 - 2;
    }
    v16 = (_BYTE *)(v61 - v12);
    v17 = v61 - v12;
    if ( v61 == v12 )
    {
      if ( v10 != 1 )
      {
LABEL_20:
        v18 = 1;
        while ( 1 )
        {
          v19 = a2[1];
          v20 = (unsigned __int64)a2[2];
          v21 = (__int64)*a2;
          if ( (unsigned __int64)(v19 + 2) > v20 )
          {
            v22 = (unsigned __int64)(v19 + 994);
            v23 = 2 * v20;
            if ( v22 > v23 )
              a2[2] = (char *)v22;
            else
              a2[2] = (char *)v23;
            v24 = realloc((void *)v21);
            *a2 = (char *)v24;
            v21 = v24;
            if ( !v24 )
              goto LABEL_73;
            v19 = a2[1];
          }
          *(_WORD *)&v19[v21] = 8236;
          v25 = v62;
          v26 = (__int64)(a2[1] + 2);
          a2[1] = (char *)v26;
          v27 = *(_QWORD *)(a1 + 8 * v18 + 32);
          v28 = abs64(v27);
          do
          {
            v29 = v25--;
            *v25 = v28 % 0xA + 48;
            v30 = v28;
            v28 /= 0xAu;
          }
          while ( v30 > 9 );
          if ( v27 < 0 )
          {
            *(v25 - 1) = 45;
            v25 = v29 - 2;
          }
          v31 = (_BYTE *)(v62 - v25);
          if ( v62 != v25 )
          {
            v38 = (unsigned __int64)a2[2];
            v39 = (_BYTE *)(v26 + v62 - v25);
            v40 = *a2;
            if ( (unsigned __int64)v39 > v38 )
            {
              v41 = v39 + 992;
              v42 = 2 * v38;
              if ( (unsigned __int64)v41 > v42 )
                a2[2] = v41;
              else
                a2[2] = (char *)v42;
              v43 = realloc(v40);
              *a2 = (char *)v43;
              v40 = (char *)v43;
              if ( !v43 )
                goto LABEL_73;
              v26 = (__int64)a2[1];
              v31 = (_BYTE *)(v62 - v25);
            }
            n = (size_t)v31;
            memcpy(&v40[v26], v25, (size_t)v31);
            a2[1] += n;
          }
          v32 = *(_DWORD *)(a1 + 24);
          if ( v32 <= (int)++v18 )
            goto LABEL_32;
        }
      }
LABEL_33:
      v33 = a2[1];
      v34 = (unsigned __int64)a2[2];
      v35 = *a2;
      if ( (unsigned __int64)(v33 + 1) <= v34 )
      {
LABEL_38:
        v33[(_QWORD)v35] = 125;
        ++a2[1];
        return;
      }
      v36 = 2 * v34;
      if ( (unsigned __int64)(v33 + 993) > v36 )
        a2[2] = v33 + 993;
      else
        a2[2] = (char *)v36;
      v37 = realloc(v35);
      *a2 = (char *)v37;
      v35 = (char *)v37;
      if ( v37 )
      {
        v33 = a2[1];
        goto LABEL_38;
      }
LABEL_73:
      abort();
    }
    v54 = a2[1];
    v55 = (unsigned __int64)a2[2];
    v56 = *a2;
    v57 = &v54[(_QWORD)v16];
    if ( &v54[(_QWORD)v16] > (char *)v55 )
    {
      v58 = 2 * v55;
      if ( (unsigned __int64)(v57 + 992) > v58 )
        a2[2] = v57 + 992;
      else
        a2[2] = (char *)v58;
      v59 = realloc(v56);
      *a2 = (char *)v59;
      v56 = (char *)v59;
      if ( !v59 )
        goto LABEL_73;
      v54 = a2[1];
    }
    memcpy(&v56[(_QWORD)v54], v12, v17);
    a2[1] += v17;
    v32 = *(_DWORD *)(a1 + 24);
    if ( v32 > 1 )
      goto LABEL_20;
LABEL_32:
    if ( v32 > 0 )
      goto LABEL_33;
  }
}
