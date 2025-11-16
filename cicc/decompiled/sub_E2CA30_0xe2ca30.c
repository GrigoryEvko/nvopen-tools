// Function: sub_E2CA30
// Address: 0xe2ca30
//
unsigned __int64 __fastcall sub_E2CA30(__int64 a1, __int64 a2, unsigned int a3)
{
  char v6; // al
  unsigned __int64 result; // rax
  __int64 v8; // rdi
  __int64 v9; // rsi
  unsigned __int64 v10; // rax
  char *v11; // rdi
  unsigned __int64 v12; // rsi
  unsigned __int64 v13; // rax
  __int64 v14; // rax
  char *v15; // rdi
  __int64 v16; // rsi
  char *v17; // rdi
  unsigned __int64 v18; // rsi
  unsigned __int64 v19; // rax
  __int64 v20; // rsi
  unsigned __int64 v21; // rax
  char *v22; // rdi
  unsigned __int64 v23; // rsi
  unsigned __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rsi
  unsigned __int64 v27; // rax
  char *v28; // rdi
  unsigned __int64 v29; // rsi
  unsigned __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rsi
  unsigned __int64 v33; // rax
  char *v34; // rdi
  unsigned __int64 v35; // rsi
  unsigned __int64 v36; // rax
  __int64 v37; // rax
  char *v38; // rdi
  __int64 v39; // rsi
  unsigned __int64 v40; // rax
  char *v41; // rdi
  unsigned __int64 v42; // rsi
  unsigned __int64 v43; // rax
  __int64 v44; // rax
  char *v45; // rdi
  __int64 v46; // rsi
  unsigned __int64 v47; // rax
  char *v48; // rdi
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 v51; // rdi
  unsigned __int64 (__fastcall *v52)(__int64, char **, unsigned int); // rax
  __int64 v53; // rsi
  char *v54; // rdi
  unsigned __int64 v55; // rax
  unsigned __int64 v56; // rsi
  unsigned __int64 v57; // rax
  __int64 v58; // rax
  unsigned __int64 v59; // rax
  unsigned __int64 v60; // rsi
  unsigned __int64 v61; // rax
  __int64 v62; // rax
  char *v63; // rsi
  unsigned __int64 v64; // rax
  char *v65; // rdi
  unsigned __int64 v66; // rsi
  unsigned __int64 v67; // rax
  __int64 v68; // rax
  unsigned __int64 v69; // rsi
  unsigned __int64 v70; // rax
  __int64 v71; // rax
  __int64 v72; // rsi
  unsigned __int64 v73; // rax
  char *v74; // rdi
  unsigned __int64 v75; // rsi
  unsigned __int64 v76; // rax
  __int64 v77; // rax
  char *v78; // rdi
  unsigned __int64 v79; // rcx
  char *v80; // rdi
  unsigned __int64 v81; // rcx
  __int64 v82; // rax

  if ( (*(_BYTE *)(a1 + 23) & 1) == 0 )
  {
    v46 = *(_QWORD *)(a2 + 8);
    v47 = *(_QWORD *)(a2 + 16);
    v48 = *(char **)a2;
    if ( v46 + 1 > v47 )
    {
      v69 = v46 + 993;
      v70 = 2 * v47;
      if ( v69 > v70 )
        *(_QWORD *)(a2 + 16) = v69;
      else
        *(_QWORD *)(a2 + 16) = v70;
      v71 = realloc(v48);
      *(_QWORD *)a2 = v71;
      v48 = (char *)v71;
      if ( !v71 )
        goto LABEL_103;
      v46 = *(_QWORD *)(a2 + 8);
    }
    v48[v46] = 40;
    v49 = *(_QWORD *)(a2 + 8);
    v50 = v49 + 1;
    *(_QWORD *)(a2 + 8) = v49 + 1;
    v51 = *(_QWORD *)(a1 + 48);
    if ( v51 )
    {
      v52 = *(unsigned __int64 (__fastcall **)(__int64, char **, unsigned int))(*(_QWORD *)v51 + 16LL);
      if ( v52 == sub_E2C9F0 )
        sub_E2C8E0(v51, (char **)a2, a3, 2u, ", ");
      else
        v52(v51, (char **)a2, a3);
      v53 = *(_QWORD *)(a2 + 8);
    }
    else
    {
      v79 = *(_QWORD *)(a2 + 16);
      v80 = *(char **)a2;
      if ( v49 + 5 > v79 )
      {
        v81 = 2 * v79;
        if ( v49 + 997 > v81 )
          *(_QWORD *)(a2 + 16) = v49 + 997;
        else
          *(_QWORD *)(a2 + 16) = v81;
        v82 = realloc(v80);
        *(_QWORD *)a2 = v82;
        v80 = (char *)v82;
        if ( !v82 )
          goto LABEL_103;
        v50 = *(_QWORD *)(a2 + 8);
      }
      *(_DWORD *)&v80[v50] = 1684631414;
      v53 = *(_QWORD *)(a2 + 8) + 4LL;
      *(_QWORD *)(a2 + 8) = v53;
    }
    if ( *(_BYTE *)(a1 + 40) )
    {
      v54 = *(char **)a2;
      if ( *(_BYTE *)(*(_QWORD *)a2 + v53 - 1) != 40 )
      {
        v55 = *(_QWORD *)(a2 + 16);
        if ( v53 + 2 > v55 )
        {
          v56 = v53 + 994;
          v57 = 2 * v55;
          if ( v56 > v57 )
            *(_QWORD *)(a2 + 16) = v56;
          else
            *(_QWORD *)(a2 + 16) = v57;
          v58 = realloc(v54);
          *(_QWORD *)a2 = v58;
          v54 = (char *)v58;
          if ( !v58 )
            goto LABEL_103;
          v53 = *(_QWORD *)(a2 + 8);
        }
        *(_WORD *)&v54[v53] = 8236;
        v54 = *(char **)a2;
        v53 = *(_QWORD *)(a2 + 8) + 2LL;
        *(_QWORD *)(a2 + 8) = v53;
      }
      v59 = *(_QWORD *)(a2 + 16);
      if ( v53 + 3 > v59 )
      {
        v60 = v53 + 995;
        v61 = 2 * v59;
        if ( v60 > v61 )
          *(_QWORD *)(a2 + 16) = v60;
        else
          *(_QWORD *)(a2 + 16) = v61;
        v62 = realloc(v54);
        *(_QWORD *)a2 = v62;
        v54 = (char *)v62;
        if ( !v62 )
          goto LABEL_103;
        v53 = *(_QWORD *)(a2 + 8);
      }
      v63 = &v54[v53];
      *(_WORD *)v63 = 11822;
      v63[2] = 46;
      v53 = *(_QWORD *)(a2 + 8) + 3LL;
      *(_QWORD *)(a2 + 8) = v53;
    }
    v64 = *(_QWORD *)(a2 + 16);
    v65 = *(char **)a2;
    if ( v53 + 1 > v64 )
    {
      v66 = v53 + 993;
      v67 = 2 * v64;
      if ( v66 > v67 )
        *(_QWORD *)(a2 + 16) = v66;
      else
        *(_QWORD *)(a2 + 16) = v67;
      v68 = realloc(v65);
      *(_QWORD *)a2 = v68;
      v65 = (char *)v68;
      if ( !v68 )
        goto LABEL_103;
      v53 = *(_QWORD *)(a2 + 8);
    }
    v65[v53] = 41;
    ++*(_QWORD *)(a2 + 8);
  }
  v6 = *(_BYTE *)(a1 + 12);
  if ( (v6 & 1) != 0 )
  {
    v39 = *(_QWORD *)(a2 + 8);
    v40 = *(_QWORD *)(a2 + 16);
    v41 = *(char **)a2;
    if ( v39 + 6 > v40 )
    {
      v42 = v39 + 998;
      v43 = 2 * v40;
      if ( v42 > v43 )
        *(_QWORD *)(a2 + 16) = v42;
      else
        *(_QWORD *)(a2 + 16) = v43;
      v44 = realloc(v41);
      *(_QWORD *)a2 = v44;
      v41 = (char *)v44;
      if ( !v44 )
        goto LABEL_103;
      v39 = *(_QWORD *)(a2 + 8);
    }
    v45 = &v41[v39];
    *(_DWORD *)v45 = 1852793632;
    *((_WORD *)v45 + 2) = 29811;
    *(_QWORD *)(a2 + 8) += 6LL;
    v6 = *(_BYTE *)(a1 + 12);
  }
  if ( (v6 & 2) != 0 )
  {
    v32 = *(_QWORD *)(a2 + 8);
    v33 = *(_QWORD *)(a2 + 16);
    v34 = *(char **)a2;
    if ( v32 + 9 > v33 )
    {
      v35 = v32 + 1001;
      v36 = 2 * v33;
      if ( v35 > v36 )
        *(_QWORD *)(a2 + 16) = v35;
      else
        *(_QWORD *)(a2 + 16) = v36;
      v37 = realloc(v34);
      *(_QWORD *)a2 = v37;
      v34 = (char *)v37;
      if ( !v37 )
        goto LABEL_103;
      v32 = *(_QWORD *)(a2 + 8);
    }
    v38 = &v34[v32];
    *(_QWORD *)v38 = 0x6C6974616C6F7620LL;
    v38[8] = 101;
    *(_QWORD *)(a2 + 8) += 9LL;
    v6 = *(_BYTE *)(a1 + 12);
  }
  if ( (v6 & 0x20) != 0 )
  {
    v26 = *(_QWORD *)(a2 + 8);
    v27 = *(_QWORD *)(a2 + 16);
    v28 = *(char **)a2;
    if ( v26 + 11 > v27 )
    {
      v29 = v26 + 1003;
      v30 = 2 * v27;
      if ( v29 > v30 )
        *(_QWORD *)(a2 + 16) = v29;
      else
        *(_QWORD *)(a2 + 16) = v30;
      v31 = realloc(v28);
      *(_QWORD *)a2 = v31;
      v28 = (char *)v31;
      if ( !v31 )
        goto LABEL_103;
      v26 = *(_QWORD *)(a2 + 8);
    }
    qmemcpy(&v28[v26], " __restrict", 11);
    *(_QWORD *)(a2 + 8) += 11LL;
    v6 = *(_BYTE *)(a1 + 12);
  }
  if ( (v6 & 0x10) != 0 )
  {
    v20 = *(_QWORD *)(a2 + 8);
    v21 = *(_QWORD *)(a2 + 16);
    v22 = *(char **)a2;
    if ( v20 + 12 > v21 )
    {
      v23 = v20 + 1004;
      v24 = 2 * v21;
      if ( v23 > v24 )
        *(_QWORD *)(a2 + 16) = v23;
      else
        *(_QWORD *)(a2 + 16) = v24;
      v25 = realloc(v22);
      *(_QWORD *)a2 = v25;
      v22 = (char *)v25;
      if ( !v25 )
        goto LABEL_103;
      v20 = *(_QWORD *)(a2 + 8);
    }
    qmemcpy(&v22[v20], " __unaligned", 12);
    *(_QWORD *)(a2 + 8) += 12LL;
  }
  if ( !*(_BYTE *)(a1 + 56) )
  {
    result = *(unsigned int *)(a1 + 24);
    if ( (_DWORD)result != 1 )
      goto LABEL_8;
LABEL_19:
    v16 = *(_QWORD *)(a2 + 8);
    result = *(_QWORD *)(a2 + 16);
    v17 = *(char **)a2;
    if ( v16 + 2 > result )
    {
      v18 = v16 + 994;
      v19 = 2 * result;
      if ( v18 > v19 )
        *(_QWORD *)(a2 + 16) = v18;
      else
        *(_QWORD *)(a2 + 16) = v19;
      result = realloc(v17);
      *(_QWORD *)a2 = result;
      v17 = (char *)result;
      if ( !result )
        goto LABEL_103;
      v16 = *(_QWORD *)(a2 + 8);
    }
    *(_WORD *)&v17[v16] = 9760;
    *(_QWORD *)(a2 + 8) += 2LL;
    goto LABEL_9;
  }
  v9 = *(_QWORD *)(a2 + 8);
  v10 = *(_QWORD *)(a2 + 16);
  v11 = *(char **)a2;
  if ( v9 + 9 > v10 )
  {
    v12 = v9 + 1001;
    v13 = 2 * v10;
    if ( v12 > v13 )
      *(_QWORD *)(a2 + 16) = v12;
    else
      *(_QWORD *)(a2 + 16) = v13;
    v14 = realloc(v11);
    *(_QWORD *)a2 = v14;
    v11 = (char *)v14;
    if ( !v14 )
      goto LABEL_103;
    v9 = *(_QWORD *)(a2 + 8);
  }
  v15 = &v11[v9];
  *(_QWORD *)v15 = 0x70656378656F6E20LL;
  v15[8] = 116;
  *(_QWORD *)(a2 + 8) += 9LL;
  result = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)result == 1 )
    goto LABEL_19;
LABEL_8:
  if ( (_DWORD)result != 2 )
    goto LABEL_9;
  v72 = *(_QWORD *)(a2 + 8);
  v73 = *(_QWORD *)(a2 + 16);
  v74 = *(char **)a2;
  if ( v72 + 3 <= v73 )
    goto LABEL_83;
  v75 = v72 + 995;
  v76 = 2 * v73;
  if ( v75 > v76 )
    *(_QWORD *)(a2 + 16) = v75;
  else
    *(_QWORD *)(a2 + 16) = v76;
  v77 = realloc(v74);
  *(_QWORD *)a2 = v77;
  v74 = (char *)v77;
  if ( !v77 )
LABEL_103:
    abort();
  v72 = *(_QWORD *)(a2 + 8);
LABEL_83:
  v78 = &v74[v72];
  result = 9760;
  *(_WORD *)v78 = 9760;
  v78[2] = 38;
  *(_QWORD *)(a2 + 8) += 3LL;
LABEL_9:
  if ( (a3 & 0x10) == 0 )
  {
    v8 = *(_QWORD *)(a1 + 32);
    if ( v8 )
      return (*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v8 + 32LL))(v8, a2, a3);
  }
  return result;
}
