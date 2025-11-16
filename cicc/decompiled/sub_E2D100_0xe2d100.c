// Function: sub_E2D100
// Address: 0xe2d100
//
unsigned __int64 __fastcall sub_E2D100(__int64 a1, __int64 *a2, unsigned int a3)
{
  __int16 v6; // ax
  __int64 v7; // rdx
  unsigned __int64 v8; // rcx
  char *v9; // rdi
  unsigned __int64 v10; // rcx
  __int64 v11; // rax
  _BYTE *v12; // r13
  __int64 v13; // r8
  int v14; // r10d
  unsigned __int64 v15; // rcx
  _BYTE *v16; // rdi
  unsigned __int64 v17; // rax
  _BYTE *v18; // rbx
  unsigned __int64 v19; // rax
  char *v20; // rdi
  unsigned __int64 v21; // rax
  __int64 v22; // rax
  _BYTE *v23; // r13
  __int64 v24; // r8
  int v25; // r10d
  unsigned __int64 v26; // rcx
  _BYTE *v27; // rdi
  unsigned __int64 v28; // rax
  _BYTE *v29; // rbx
  unsigned __int64 v30; // rax
  char *v31; // rdi
  unsigned __int64 v32; // rax
  __int64 v33; // rax
  _BYTE *v34; // r13
  __int64 v35; // r8
  int v36; // r10d
  unsigned __int64 v37; // rcx
  _BYTE *v38; // rdi
  unsigned __int64 v39; // rax
  _BYTE *v40; // rbx
  unsigned __int64 v41; // rax
  char *v42; // rdi
  unsigned __int64 v43; // rax
  __int64 v44; // rax
  _BYTE *v45; // r13
  __int64 v46; // rdi
  unsigned __int64 v47; // rcx
  unsigned __int64 v48; // rax
  _BYTE *v49; // rbx
  unsigned __int64 v50; // rdx
  _BYTE *v51; // rax
  char *v52; // r9
  unsigned __int64 v53; // rsi
  unsigned __int64 v54; // rdx
  __int64 v55; // rax
  _BYTE *v56; // rbx
  __int64 v57; // rsi
  unsigned __int64 v58; // rax
  char *v59; // rdi
  unsigned __int64 v60; // rcx
  unsigned __int64 v61; // rax
  unsigned __int64 v62; // rax
  char *v63; // r8
  unsigned __int64 v64; // rax
  __int64 v65; // rax
  unsigned __int64 v67; // rcx
  __int64 v68; // rax
  _BYTE *v69; // r13
  __int64 v70; // r8
  int v71; // r10d
  unsigned __int64 v72; // rcx
  _BYTE *v73; // rdi
  unsigned __int64 v74; // rax
  _BYTE *v75; // rbx
  unsigned __int64 v76; // rax
  char *v77; // rdi
  unsigned __int64 v78; // rax
  __int64 v79; // rax
  unsigned __int64 v80; // rcx
  unsigned __int64 v81; // rax
  unsigned __int64 v82; // rsi
  unsigned __int64 v83; // rax
  __int64 v84; // rax
  unsigned __int64 v85; // rax
  unsigned __int64 v86; // rax
  unsigned __int64 v87; // rdx
  _BYTE *v88; // rax
  char *v89; // rdi
  unsigned __int64 v90; // rdx
  __int64 v91; // rax
  _BYTE *v92; // rbx
  unsigned __int64 v93; // rdx
  _BYTE *v94; // rax
  char *v95; // rdi
  unsigned __int64 v96; // rdx
  __int64 v97; // rax
  _BYTE *v98; // rbx
  unsigned __int64 v99; // rdx
  _BYTE *v100; // rax
  char *v101; // rdi
  unsigned __int64 v102; // rdx
  __int64 v103; // rax
  _BYTE *v104; // rbx
  unsigned __int64 v105; // rdx
  _BYTE *v106; // rax
  char *v107; // rdi
  unsigned __int64 v108; // rdx
  __int64 v109; // rax
  _BYTE *v110; // rbx
  _BYTE v111[32]; // [rsp+15h] [rbp-FBh] BYREF
  _BYTE v112[32]; // [rsp+35h] [rbp-DBh] BYREF
  _BYTE v113[32]; // [rsp+55h] [rbp-BBh] BYREF
  _BYTE v114[32]; // [rsp+75h] [rbp-9Bh] BYREF
  _BYTE v115[32]; // [rsp+95h] [rbp-7Bh] BYREF
  _BYTE v116[32]; // [rsp+B5h] [rbp-5Bh] BYREF
  _BYTE v117[59]; // [rsp+D5h] [rbp-3Bh] BYREF

  v6 = *(_WORD *)(a1 + 22);
  if ( (v6 & 0x800) != 0 )
  {
    v57 = a2[1];
    v58 = a2[2];
    v59 = (char *)*a2;
    if ( v57 + 10 > v58 )
    {
      v82 = v57 + 1002;
      v83 = 2 * v58;
      if ( v82 > v83 )
        a2[2] = v82;
      else
        a2[2] = v83;
      v84 = realloc(v59);
      *a2 = v84;
      v59 = (char *)v84;
      if ( !v84 )
        goto LABEL_123;
      v57 = a2[1];
    }
    qmemcpy(&v59[v57], "`adjustor{", 10);
    v45 = v111;
    v46 = a2[1] + 10;
    a2[1] = v46;
    v60 = *(unsigned int *)(a1 + 60);
    do
    {
      *--v45 = v60 % 0xA + 48;
      v61 = v60;
      v60 /= 0xAu;
    }
    while ( v61 > 9 );
    v49 = (_BYTE *)(v111 - v45);
    if ( v111 == v45 )
    {
LABEL_52:
      v62 = a2[2];
      v63 = (char *)*a2;
      if ( v46 + 2 <= v62 )
      {
LABEL_57:
        *(_WORD *)&v63[v46] = 10109;
        a2[1] += 2;
        return sub_E2CA30(a1, (__int64)a2, a3);
      }
      v64 = 2 * v62;
      if ( v46 + 994 > v64 )
        a2[2] = v46 + 994;
      else
        a2[2] = v64;
      v65 = realloc(v63);
      *a2 = v65;
      v63 = (char *)v65;
      if ( v65 )
      {
        v46 = a2[1];
        goto LABEL_57;
      }
LABEL_123:
      abort();
    }
    v85 = a2[2];
    v52 = (char *)*a2;
    if ( (unsigned __int64)&v49[v46] <= v85 )
    {
LABEL_47:
      memcpy(&v52[v46], v45, (size_t)v49);
      v56 = &v49[a2[1]];
      a2[1] = (__int64)v56;
      v46 = (__int64)v56;
      goto LABEL_52;
    }
    v53 = (unsigned __int64)&v49[v46 + 992];
    v86 = 2 * v85;
    if ( v53 <= v86 )
    {
      a2[2] = v86;
      goto LABEL_45;
    }
LABEL_44:
    a2[2] = v53;
LABEL_45:
    v55 = realloc(v52);
    *a2 = v55;
    v52 = (char *)v55;
    if ( !v55 )
      goto LABEL_123;
    v46 = a2[1];
    goto LABEL_47;
  }
  if ( (v6 & 0x200) != 0 )
  {
    v7 = a2[1];
    v8 = a2[2];
    v9 = (char *)*a2;
    if ( (v6 & 0x400) != 0 )
    {
      if ( v7 + 12 > v8 )
      {
        v10 = 2 * v8;
        if ( v7 + 1004 > v10 )
          a2[2] = v7 + 1004;
        else
          a2[2] = v10;
        v11 = realloc(v9);
        *a2 = v11;
        v9 = (char *)v11;
        if ( !v11 )
          goto LABEL_123;
        v7 = a2[1];
      }
      qmemcpy(&v9[v7], "`vtordispex{", 12);
      v12 = v115;
      v13 = a2[1] + 12;
      a2[1] = v13;
      v14 = *(_DWORD *)(a1 + 64);
      v15 = abs32(v14);
      do
      {
        v16 = v12--;
        *v12 = v15 % 0xA + 48;
        v17 = v15;
        v15 /= 0xAu;
      }
      while ( v17 > 9 );
      if ( v14 < 0 )
      {
        *(v12 - 1) = 45;
        v12 = v16 - 2;
      }
      v18 = (_BYTE *)(v115 - v12);
      if ( v115 != v12 )
      {
        v105 = a2[2];
        v106 = &v18[v13];
        v107 = (char *)*a2;
        if ( (unsigned __int64)&v18[v13] > v105 )
        {
          v108 = 2 * v105;
          if ( (unsigned __int64)(v106 + 992) > v108 )
            a2[2] = (__int64)(v106 + 992);
          else
            a2[2] = v108;
          v109 = realloc(v107);
          *a2 = v109;
          v107 = (char *)v109;
          if ( !v109 )
            goto LABEL_123;
          v13 = a2[1];
        }
        memcpy(&v107[v13], v12, v115 - v12);
        v110 = &v18[a2[1]];
        a2[1] = (__int64)v110;
        v13 = (__int64)v110;
      }
      v19 = a2[2];
      v20 = (char *)*a2;
      if ( v13 + 2 > v19 )
      {
        v21 = 2 * v19;
        if ( v13 + 994 <= v21 )
          a2[2] = v21;
        else
          a2[2] = v13 + 994;
        v22 = realloc(v20);
        *a2 = v22;
        v20 = (char *)v22;
        if ( !v22 )
          goto LABEL_123;
        v13 = a2[1];
      }
      *(_WORD *)&v20[v13] = 8236;
      v23 = v114;
      v24 = a2[1] + 2;
      a2[1] = v24;
      v25 = *(_DWORD *)(a1 + 68);
      v26 = abs32(v25);
      do
      {
        v27 = v23--;
        *v23 = v26 % 0xA + 48;
        v28 = v26;
        v26 /= 0xAu;
      }
      while ( v28 > 9 );
      if ( v25 < 0 )
      {
        *(v23 - 1) = 45;
        v23 = v27 - 2;
      }
      v29 = (_BYTE *)(v114 - v23);
      if ( v114 != v23 )
      {
        v99 = a2[2];
        v100 = &v29[v24];
        v101 = (char *)*a2;
        if ( (unsigned __int64)&v29[v24] > v99 )
        {
          v102 = 2 * v99;
          if ( (unsigned __int64)(v100 + 992) > v102 )
            a2[2] = (__int64)(v100 + 992);
          else
            a2[2] = v102;
          v103 = realloc(v101);
          *a2 = v103;
          v101 = (char *)v103;
          if ( !v103 )
            goto LABEL_123;
          v24 = a2[1];
        }
        memcpy(&v101[v24], v23, v114 - v23);
        v104 = &v29[a2[1]];
        a2[1] = (__int64)v104;
        v24 = (__int64)v104;
      }
      v30 = a2[2];
      v31 = (char *)*a2;
      if ( v24 + 2 > v30 )
      {
        v32 = 2 * v30;
        if ( v24 + 994 > v32 )
          a2[2] = v24 + 994;
        else
          a2[2] = v32;
        v33 = realloc(v31);
        *a2 = v33;
        v31 = (char *)v33;
        if ( !v33 )
          goto LABEL_123;
        v24 = a2[1];
      }
      *(_WORD *)&v31[v24] = 8236;
      v34 = v113;
      v35 = a2[1] + 2;
      a2[1] = v35;
      v36 = *(_DWORD *)(a1 + 72);
      v37 = abs32(v36);
      do
      {
        v38 = v34--;
        *v34 = v37 % 0xA + 48;
        v39 = v37;
        v37 /= 0xAu;
      }
      while ( v39 > 9 );
      if ( v36 < 0 )
      {
        *(v34 - 1) = 45;
        v34 = v38 - 2;
      }
      v40 = (_BYTE *)(v113 - v34);
      if ( v113 != v34 )
      {
        v87 = a2[2];
        v88 = &v40[v35];
        v89 = (char *)*a2;
        if ( (unsigned __int64)&v40[v35] > v87 )
        {
          v90 = 2 * v87;
          if ( (unsigned __int64)(v88 + 992) > v90 )
            a2[2] = (__int64)(v88 + 992);
          else
            a2[2] = v90;
          v91 = realloc(v89);
          *a2 = v91;
          v89 = (char *)v91;
          if ( !v91 )
            goto LABEL_123;
          v35 = a2[1];
        }
        memcpy(&v89[v35], v34, v113 - v34);
        v92 = &v40[a2[1]];
        a2[1] = (__int64)v92;
        v35 = (__int64)v92;
      }
      v41 = a2[2];
      v42 = (char *)*a2;
      if ( v35 + 2 > v41 )
      {
        v43 = 2 * v41;
        if ( v35 + 994 > v43 )
          a2[2] = v35 + 994;
        else
          a2[2] = v43;
        v44 = realloc(v42);
        *a2 = v44;
        v42 = (char *)v44;
        if ( !v44 )
          goto LABEL_123;
        v35 = a2[1];
      }
      *(_WORD *)&v42[v35] = 8236;
      v45 = v112;
      v46 = a2[1] + 2;
      a2[1] = v46;
      v47 = *(unsigned int *)(a1 + 60);
      do
      {
        *--v45 = v47 % 0xA + 48;
        v48 = v47;
        v47 /= 0xAu;
      }
      while ( v48 > 9 );
      v49 = (_BYTE *)(v112 - v45);
      if ( v112 == v45 )
        goto LABEL_52;
      v50 = a2[2];
      v51 = &v49[v46];
      v52 = (char *)*a2;
      if ( v50 >= (unsigned __int64)&v49[v46] )
        goto LABEL_47;
    }
    else
    {
      if ( v7 + 10 > v8 )
      {
        v67 = 2 * v8;
        if ( v7 + 1002 > v67 )
          a2[2] = v7 + 1002;
        else
          a2[2] = v67;
        v68 = realloc(v9);
        *a2 = v68;
        v9 = (char *)v68;
        if ( !v68 )
          goto LABEL_123;
        v7 = a2[1];
      }
      qmemcpy(&v9[v7], "`vtordisp{", 10);
      v69 = v117;
      v70 = a2[1] + 10;
      a2[1] = v70;
      v71 = *(_DWORD *)(a1 + 72);
      v72 = abs32(v71);
      do
      {
        v73 = v69--;
        *v69 = v72 % 0xA + 48;
        v74 = v72;
        v72 /= 0xAu;
      }
      while ( v74 > 9 );
      if ( v71 < 0 )
      {
        *(v69 - 1) = 45;
        v69 = v73 - 2;
      }
      v75 = (_BYTE *)(v117 - v69);
      if ( v117 != v69 )
      {
        v93 = a2[2];
        v94 = &v75[v70];
        v95 = (char *)*a2;
        if ( (unsigned __int64)&v75[v70] > v93 )
        {
          v96 = 2 * v93;
          if ( (unsigned __int64)(v94 + 992) > v96 )
            a2[2] = (__int64)(v94 + 992);
          else
            a2[2] = v96;
          v97 = realloc(v95);
          *a2 = v97;
          v95 = (char *)v97;
          if ( !v97 )
            goto LABEL_123;
          v70 = a2[1];
        }
        memcpy(&v95[v70], v69, v117 - v69);
        v98 = &v75[a2[1]];
        a2[1] = (__int64)v98;
        v70 = (__int64)v98;
      }
      v76 = a2[2];
      v77 = (char *)*a2;
      if ( v70 + 2 > v76 )
      {
        v78 = 2 * v76;
        if ( v70 + 994 <= v78 )
          a2[2] = v78;
        else
          a2[2] = v70 + 994;
        v79 = realloc(v77);
        *a2 = v79;
        v77 = (char *)v79;
        if ( !v79 )
          goto LABEL_123;
        v70 = a2[1];
      }
      *(_WORD *)&v77[v70] = 8236;
      v45 = v116;
      v46 = a2[1] + 2;
      a2[1] = v46;
      v80 = *(unsigned int *)(a1 + 60);
      do
      {
        *--v45 = v80 % 0xA + 48;
        v81 = v80;
        v80 /= 0xAu;
      }
      while ( v81 > 9 );
      v49 = (_BYTE *)(v116 - v45);
      if ( v116 == v45 )
        goto LABEL_52;
      v50 = a2[2];
      v51 = &v49[v46];
      v52 = (char *)*a2;
      if ( (unsigned __int64)&v49[v46] <= v50 )
        goto LABEL_47;
    }
    v53 = (unsigned __int64)(v51 + 992);
    v54 = 2 * v50;
    if ( (unsigned __int64)(v51 + 992) <= v54 )
    {
      a2[2] = v54;
      goto LABEL_45;
    }
    goto LABEL_44;
  }
  return sub_E2CA30(a1, (__int64)a2, a3);
}
