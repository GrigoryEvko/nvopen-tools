// Function: sub_1E343B0
// Address: 0x1e343b0
//
_BYTE *__fastcall sub_1E343B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5, __int64 a6, __int64 a7)
{
  _BYTE *v11; // rax
  __int16 v12; // ax
  __int64 v13; // r15
  void *v14; // rdx
  unsigned __int8 v15; // al
  unsigned __int8 v16; // dl
  unsigned __int8 v17; // al
  __int64 v18; // rax
  unsigned __int64 v19; // r14
  __int16 v20; // ax
  char *v21; // r15
  size_t v22; // r8
  _QWORD *v23; // rax
  unsigned __int8 v24; // al
  unsigned __int8 *v25; // rbx
  unsigned __int8 *v26; // r15
  unsigned __int8 *v27; // r14
  _QWORD *v28; // rdx
  void *v29; // rdx
  void *v30; // rdx
  __int64 v31; // rdx
  unsigned int v32; // ebx
  _BYTE *result; // rax
  __int64 v34; // rdx
  char *v35; // rbx
  size_t v36; // rax
  _BYTE *v37; // rdi
  size_t v38; // r14
  unsigned __int64 v39; // rax
  __int64 v40; // r15
  char *v41; // rbx
  size_t v42; // rax
  _BYTE *v43; // rdi
  size_t v44; // r14
  unsigned __int64 v45; // rax
  __int64 v46; // r15
  _QWORD *v47; // rdx
  __int64 v48; // rdi
  unsigned __int64 v49; // r14
  __int16 v50; // ax
  char *v51; // r15
  size_t v52; // r8
  _QWORD *v53; // rax
  void *v54; // rdx
  __int64 v55; // rdx
  __int64 v56; // rdx
  _BYTE *v57; // rax
  __int64 v58; // r8
  const char *v59; // rax
  __int64 v60; // r8
  char *v61; // r15
  size_t v62; // rax
  size_t v63; // rdx
  char *v64; // rdi
  unsigned __int64 v65; // rax
  __int64 v66; // rax
  _BYTE *v67; // rax
  __int64 v68; // r8
  const char *v69; // rax
  __int64 v70; // r8
  size_t v71; // rax
  char *v72; // rsi
  size_t v73; // rdx
  char *v74; // rdi
  unsigned __int64 v75; // rax
  __int64 v76; // rax
  _BYTE *v77; // rax
  __int64 v78; // r8
  const char *v79; // rax
  __int64 v80; // r8
  size_t v81; // rax
  char *v82; // rsi
  size_t v83; // rdx
  char *v84; // rdi
  unsigned __int64 v85; // rax
  __int64 v86; // rax
  void *v87; // rdx
  __m128i *v88; // rdx
  void *v89; // rdx
  __int64 v90; // rdx
  char *v91; // rsi
  __int64 v92; // rax
  __int64 v93; // rax
  __int64 v94; // rax
  unsigned __int64 v95; // rdx
  _BYTE *v96; // rcx
  unsigned __int64 v97; // rdx
  int v98; // eax
  unsigned __int64 v99; // rdi
  char *v100; // rax
  char *v101; // r15
  unsigned int v102; // ecx
  unsigned int v103; // esi
  __int64 v104; // rax
  void *v105; // rdx
  void *v106; // rdx
  __int64 v107; // rdx
  void *v108; // rdx
  const char *v109; // r14
  size_t v110; // rdx
  __int64 v111; // rdx
  __int64 v112; // rdx
  void *v113; // rdx
  unsigned __int64 v114; // rdx
  int v115; // ecx
  signed __int64 v116; // rax
  unsigned int v117; // ecx
  unsigned int v118; // ecx
  unsigned int v119; // esi
  __int64 v120; // rdi
  _BYTE *v121; // rax
  const char *v122; // rax
  size_t v123; // rdx
  _QWORD *v124; // [rsp+0h] [rbp-50h]
  __int64 v125; // [rsp+0h] [rbp-50h]
  __int64 v126; // [rsp+0h] [rbp-50h]
  _QWORD *src; // [rsp+8h] [rbp-48h]
  char *srce; // [rsp+8h] [rbp-48h]
  void *srca; // [rsp+8h] [rbp-48h]
  char *srcb; // [rsp+8h] [rbp-48h]
  char *srcf; // [rsp+8h] [rbp-48h]
  void *srcc; // [rsp+8h] [rbp-48h]
  char *srcd; // [rsp+8h] [rbp-48h]
  char *srcg; // [rsp+8h] [rbp-48h]

  v11 = *(_BYTE **)(a2 + 24);
  if ( (unsigned __int64)v11 >= *(_QWORD *)(a2 + 16) )
  {
    sub_16E7DE0(a2, 40);
  }
  else
  {
    *(_QWORD *)(a2 + 24) = v11 + 1;
    *v11 = 40;
  }
  v12 = *(_WORD *)(a1 + 32);
  if ( (v12 & 4) != 0 )
  {
    v90 = *(_QWORD *)(a2 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v90) <= 8 )
    {
      sub_16E7EE0(a2, "volatile ", 9u);
    }
    else
    {
      *(_BYTE *)(v90 + 8) = 32;
      *(_QWORD *)v90 = 0x656C6974616C6F76LL;
      *(_QWORD *)(a2 + 24) += 9LL;
    }
    v12 = *(_WORD *)(a1 + 32);
  }
  if ( (v12 & 8) != 0 )
  {
    v89 = *(void **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v89 <= 0xCu )
    {
      sub_16E7EE0(a2, "non-temporal ", 0xDu);
    }
    else
    {
      qmemcpy(v89, "non-temporal ", 13);
      *(_QWORD *)(a2 + 24) += 13LL;
    }
    v12 = *(_WORD *)(a1 + 32);
  }
  if ( (v12 & 0x10) != 0 )
  {
    v88 = *(__m128i **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v88 <= 0xFu )
    {
      sub_16E7EE0(a2, "dereferenceable ", 0x10u);
    }
    else
    {
      *v88 = _mm_load_si128((const __m128i *)&xmmword_42EB9C0);
      *(_QWORD *)(a2 + 24) += 16LL;
    }
    v12 = *(_WORD *)(a1 + 32);
  }
  if ( (v12 & 0x20) != 0 )
  {
    v87 = *(void **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v87 <= 9u )
    {
      sub_16E7EE0(a2, "invariant ", 0xAu);
    }
    else
    {
      qmemcpy(v87, "invariant ", 10);
      *(_QWORD *)(a2 + 24) += 10LL;
    }
    v12 = *(_WORD *)(a1 + 32);
  }
  if ( (v12 & 0x40) != 0 )
  {
    v77 = *(_BYTE **)(a2 + 24);
    if ( (unsigned __int64)v77 >= *(_QWORD *)(a2 + 16) )
    {
      v78 = sub_16E7DE0(a2, 34);
    }
    else
    {
      v78 = a2;
      *(_QWORD *)(a2 + 24) = v77 + 1;
      *v77 = 34;
    }
    srcc = (void *)v78;
    v79 = (const char *)sub_1E30920(a7, 64);
    v80 = (__int64)srcc;
    if ( v79 )
    {
      v126 = (__int64)srcc;
      srcd = (char *)v79;
      v81 = strlen(v79);
      v80 = v126;
      v82 = srcd;
      v83 = v81;
      v84 = *(char **)(v126 + 24);
      v85 = *(_QWORD *)(v126 + 16) - (_QWORD)v84;
      if ( v83 <= v85 )
      {
        if ( v83 )
        {
          srcg = (char *)v83;
          memcpy(v84, v82, v83);
          v80 = v126;
          v86 = *(_QWORD *)(v126 + 16);
          v84 = &srcg[*(_QWORD *)(v126 + 24)];
          *(_QWORD *)(v126 + 24) = v84;
          v85 = v86 - (_QWORD)v84;
        }
        goto LABEL_114;
      }
      v80 = sub_16E7EE0(v126, srcd, v83);
    }
    v84 = *(char **)(v80 + 24);
    v85 = *(_QWORD *)(v80 + 16) - (_QWORD)v84;
LABEL_114:
    if ( v85 <= 1 )
    {
      sub_16E7EE0(v80, "\" ", 2u);
    }
    else
    {
      *(_WORD *)v84 = 8226;
      *(_QWORD *)(v80 + 24) += 2LL;
    }
    v12 = *(_WORD *)(a1 + 32);
  }
  if ( (v12 & 0x80u) == 0 )
    goto LABEL_9;
  v67 = *(_BYTE **)(a2 + 24);
  if ( (unsigned __int64)v67 >= *(_QWORD *)(a2 + 16) )
  {
    v68 = sub_16E7DE0(a2, 34);
  }
  else
  {
    v68 = a2;
    *(_QWORD *)(a2 + 24) = v67 + 1;
    *v67 = 34;
  }
  srca = (void *)v68;
  v69 = (const char *)sub_1E30920(a7, 128);
  v70 = (__int64)srca;
  if ( v69 )
  {
    v125 = (__int64)srca;
    srcb = (char *)v69;
    v71 = strlen(v69);
    v70 = v125;
    v72 = srcb;
    v73 = v71;
    v74 = *(char **)(v125 + 24);
    v75 = *(_QWORD *)(v125 + 16) - (_QWORD)v74;
    if ( v73 <= v75 )
    {
      if ( v73 )
      {
        srcf = (char *)v73;
        memcpy(v74, v72, v73);
        v70 = v125;
        v76 = *(_QWORD *)(v125 + 16);
        v74 = &srcf[*(_QWORD *)(v125 + 24)];
        *(_QWORD *)(v125 + 24) = v74;
        v75 = v76 - (_QWORD)v74;
      }
      goto LABEL_124;
    }
    v70 = sub_16E7EE0(v125, srcb, v73);
  }
  v74 = *(char **)(v70 + 24);
  v75 = *(_QWORD *)(v70 + 16) - (_QWORD)v74;
LABEL_124:
  if ( v75 <= 1 )
  {
    sub_16E7EE0(v70, "\" ", 2u);
  }
  else
  {
    *(_WORD *)v74 = 8226;
    *(_QWORD *)(v70 + 24) += 2LL;
  }
  v12 = *(_WORD *)(a1 + 32);
LABEL_9:
  if ( (v12 & 0x100) == 0 )
    goto LABEL_10;
  v57 = *(_BYTE **)(a2 + 24);
  if ( (unsigned __int64)v57 >= *(_QWORD *)(a2 + 16) )
  {
    v58 = sub_16E7DE0(a2, 34);
  }
  else
  {
    v58 = a2;
    *(_QWORD *)(a2 + 24) = v57 + 1;
    *v57 = 34;
  }
  src = (_QWORD *)v58;
  v59 = (const char *)sub_1E30920(a7, 256);
  v60 = (__int64)src;
  v61 = (char *)v59;
  if ( v59 )
  {
    v62 = strlen(v59);
    v60 = (__int64)src;
    v63 = v62;
    v64 = (char *)src[3];
    v65 = src[2] - (_QWORD)v64;
    if ( v63 <= v65 )
    {
      if ( v63 )
      {
        v124 = src;
        srce = (char *)v63;
        memcpy(v64, v61, v63);
        v60 = (__int64)v124;
        v66 = v124[2];
        v64 = &srce[v124[3]];
        v124[3] = v64;
        v65 = v66 - (_QWORD)v64;
      }
      goto LABEL_119;
    }
    v60 = sub_16E7EE0((__int64)src, v61, v63);
  }
  v64 = *(char **)(v60 + 24);
  v65 = *(_QWORD *)(v60 + 16) - (_QWORD)v64;
LABEL_119:
  if ( v65 <= 1 )
  {
    sub_16E7EE0(v60, "\" ", 2u);
  }
  else
  {
    *(_WORD *)v64 = 8226;
    *(_QWORD *)(v60 + 24) += 2LL;
  }
  v12 = *(_WORD *)(a1 + 32);
LABEL_10:
  if ( (v12 & 1) != 0 )
  {
    v56 = *(_QWORD *)(a2 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v56) <= 4 )
    {
      sub_16E7EE0(a2, "load ", 5u);
    }
    else
    {
      *(_DWORD *)v56 = 1684107116;
      *(_BYTE *)(v56 + 4) = 32;
      *(_QWORD *)(a2 + 24) += 5LL;
    }
    v12 = *(_WORD *)(a1 + 32);
  }
  if ( (v12 & 2) != 0 )
  {
    v55 = *(_QWORD *)(a2 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v55) <= 5 )
    {
      sub_16E7EE0(a2, "store ", 6u);
    }
    else
    {
      *(_DWORD *)v55 = 1919906931;
      *(_WORD *)(v55 + 4) = 8293;
      *(_QWORD *)(a2 + 24) += 6LL;
    }
  }
  v13 = *(unsigned __int8 *)(a1 + 36);
  if ( (_BYTE)v13 == 1 )
  {
LABEL_18:
    v15 = *(_BYTE *)(a1 + 37);
    v16 = v15 & 0xF;
    if ( (v15 & 0xF) == 0 )
      goto LABEL_19;
    goto LABEL_52;
  }
  if ( !*(_DWORD *)(a4 + 8) )
    sub_16032E0(a5);
  v14 = *(void **)(a2 + 24);
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)v14 <= 0xAu )
  {
    sub_16E7EE0(a2, "syncscope(\"", 0xBu);
  }
  else
  {
    qmemcpy(v14, "syncscope(\"", 11);
    *(_QWORD *)(a2 + 24) += 11LL;
  }
  sub_16D16F0(*(unsigned __int8 **)(*(_QWORD *)a4 + 16 * v13), *(_QWORD *)(*(_QWORD *)a4 + 16 * v13 + 8), a2);
  v34 = *(_QWORD *)(a2 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v34) <= 2 )
  {
    sub_16E7EE0(a2, "\") ", 3u);
    goto LABEL_18;
  }
  *(_BYTE *)(v34 + 2) = 32;
  *(_WORD *)v34 = 10530;
  *(_QWORD *)(a2 + 24) += 3LL;
  v15 = *(_BYTE *)(a1 + 37);
  v16 = v15 & 0xF;
  if ( (v15 & 0xF) == 0 )
    goto LABEL_19;
LABEL_52:
  v35 = (&off_4C6F320)[v16];
  if ( !v35 )
  {
    v37 = *(_BYTE **)(a2 + 24);
    v39 = *(_QWORD *)(a2 + 16);
    v40 = a2;
    goto LABEL_56;
  }
  v36 = strlen((&off_4C6F320)[v16]);
  v37 = *(_BYTE **)(a2 + 24);
  v38 = v36;
  v39 = *(_QWORD *)(a2 + 16);
  if ( v38 <= v39 - (unsigned __int64)v37 )
  {
    v40 = a2;
    if ( v38 )
    {
      memcpy(v37, v35, v38);
      v39 = *(_QWORD *)(a2 + 16);
      v37 = (_BYTE *)(v38 + *(_QWORD *)(a2 + 24));
      *(_QWORD *)(a2 + 24) = v37;
    }
LABEL_56:
    if ( v39 > (unsigned __int64)v37 )
      goto LABEL_57;
    goto LABEL_135;
  }
  v92 = sub_16E7EE0(a2, v35, v38);
  v37 = *(_BYTE **)(v92 + 24);
  v40 = v92;
  if ( *(_QWORD *)(v92 + 16) <= (unsigned __int64)v37 )
  {
LABEL_135:
    sub_16E7DE0(v40, 32);
    v15 = *(_BYTE *)(a1 + 37);
LABEL_19:
    v17 = v15 >> 4;
    if ( !v17 )
      goto LABEL_20;
    goto LABEL_58;
  }
LABEL_57:
  *(_QWORD *)(v40 + 24) = v37 + 1;
  *v37 = 32;
  v17 = *(_BYTE *)(a1 + 37) >> 4;
  if ( !v17 )
    goto LABEL_20;
LABEL_58:
  v41 = (&off_4C6F320)[v17];
  if ( v41 )
  {
    v42 = strlen((&off_4C6F320)[v17]);
    v43 = *(_BYTE **)(a2 + 24);
    v44 = v42;
    v45 = *(_QWORD *)(a2 + 16);
    if ( v44 > v45 - (unsigned __int64)v43 )
    {
      v93 = sub_16E7EE0(a2, v41, v44);
      v43 = *(_BYTE **)(v93 + 24);
      v46 = v93;
      if ( (unsigned __int64)v43 < *(_QWORD *)(v93 + 16) )
        goto LABEL_63;
      goto LABEL_137;
    }
    v46 = a2;
    if ( v44 )
    {
      memcpy(v43, v41, v44);
      v45 = *(_QWORD *)(a2 + 16);
      v43 = (_BYTE *)(v44 + *(_QWORD *)(a2 + 24));
      *(_QWORD *)(a2 + 24) = v43;
    }
  }
  else
  {
    v43 = *(_BYTE **)(a2 + 24);
    v45 = *(_QWORD *)(a2 + 16);
    v46 = a2;
  }
  if ( (unsigned __int64)v43 < v45 )
  {
LABEL_63:
    *(_QWORD *)(v46 + 24) = v43 + 1;
    *v43 = 32;
    goto LABEL_20;
  }
LABEL_137:
  sub_16E7DE0(v46, 32);
LABEL_20:
  sub_16E7A90(a2, *(_QWORD *)(a1 + 24));
  v18 = *(_QWORD *)a1;
  if ( (*(_QWORD *)a1 & 4) != 0 )
  {
    v49 = v18 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v18 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v50 = *(_WORD *)(a1 + 32);
      v51 = " into ";
      if ( (v50 & 1) != 0 )
      {
        v91 = " from ";
        if ( (v50 & 2) != 0 )
          v91 = " on ";
        v51 = v91;
      }
      v52 = strlen(v51);
      v53 = *(_QWORD **)(a2 + 24);
      if ( v52 <= *(_QWORD *)(a2 + 16) - (_QWORD)v53 )
      {
        if ( (unsigned int)v52 >= 8 )
        {
          *v53 = *(_QWORD *)v51;
          *(_QWORD *)((char *)v53 + (unsigned int)v52 - 8) = *(_QWORD *)&v51[(unsigned int)v52 - 8];
          v114 = (unsigned __int64)(v53 + 1) & 0xFFFFFFFFFFFFFFF8LL;
          v115 = (_DWORD)v53 - v114;
          v116 = v51 - ((char *)v53 - v114);
          v117 = (v52 + v115) & 0xFFFFFFF8;
          if ( v117 >= 8 )
          {
            v118 = v117 & 0xFFFFFFF8;
            v119 = 0;
            do
            {
              v120 = v119;
              v119 += 8;
              *(_QWORD *)(v114 + v120) = *(_QWORD *)(v116 + v120);
            }
            while ( v119 < v118 );
          }
        }
        else if ( (v52 & 4) != 0 )
        {
          *(_DWORD *)v53 = *(_DWORD *)v51;
          *(_DWORD *)((char *)v53 + (unsigned int)v52 - 4) = *(_DWORD *)&v51[(unsigned int)v52 - 4];
        }
        else if ( (_DWORD)v52 )
        {
          *(_BYTE *)v53 = *v51;
          if ( (v52 & 2) != 0 )
            *(_WORD *)((char *)v53 + (unsigned int)v52 - 2) = *(_WORD *)&v51[(unsigned int)v52 - 2];
        }
        *(_QWORD *)(a2 + 24) += v52;
      }
      else
      {
        sub_16E7EE0(a2, v51, v52);
      }
      switch ( *(_DWORD *)(v49 + 8) )
      {
        case 0:
          v111 = *(_QWORD *)(a2 + 24);
          if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v111) <= 4 )
          {
            sub_16E7EE0(a2, "stack", 5u);
          }
          else
          {
            *(_DWORD *)v111 = 1667331187;
            *(_BYTE *)(v111 + 4) = 107;
            *(_QWORD *)(a2 + 24) += 5LL;
          }
          break;
        case 1:
          v107 = *(_QWORD *)(a2 + 24);
          if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v107) <= 2 )
          {
            sub_16E7EE0(a2, "got", 3u);
          }
          else
          {
            *(_BYTE *)(v107 + 2) = 116;
            *(_WORD *)v107 = 28519;
            *(_QWORD *)(a2 + 24) += 3LL;
          }
          break;
        case 2:
          v106 = *(void **)(a2 + 24);
          if ( *(_QWORD *)(a2 + 16) - (_QWORD)v106 <= 9u )
          {
            sub_16E7EE0(a2, "jump-table", 0xAu);
          }
          else
          {
            qmemcpy(v106, "jump-table", 10);
            *(_QWORD *)(a2 + 24) += 10LL;
          }
          break;
        case 3:
          v105 = *(void **)(a2 + 24);
          if ( *(_QWORD *)(a2 + 16) - (_QWORD)v105 <= 0xCu )
          {
            sub_16E7EE0(a2, "constant-pool", 0xDu);
          }
          else
          {
            qmemcpy(v105, "constant-pool", 13);
            *(_QWORD *)(a2 + 24) += 13LL;
          }
          break;
        case 4:
          sub_1E32090(a2, *(_DWORD *)(v49 + 16), 1, a6);
          break;
        case 5:
          v113 = *(void **)(a2 + 24);
          if ( *(_QWORD *)(a2 + 16) - (_QWORD)v113 <= 0xAu )
          {
            sub_16E7EE0(a2, "call-entry ", 0xBu);
          }
          else
          {
            qmemcpy(v113, "call-entry ", 11);
            *(_QWORD *)(a2 + 24) += 11LL;
          }
          sub_1553920(*(__int64 **)(v49 + 16), a2, 0, a3);
          break;
        case 6:
          v108 = *(void **)(a2 + 24);
          if ( *(_QWORD *)(a2 + 16) - (_QWORD)v108 <= 0xBu )
          {
            sub_16E7EE0(a2, "call-entry &", 0xCu);
          }
          else
          {
            qmemcpy(v108, "call-entry &", 12);
            *(_QWORD *)(a2 + 24) += 12LL;
          }
          v109 = *(const char **)(v49 + 16);
          v110 = 0;
          if ( v109 )
            v110 = strlen(v109);
          sub_154B650(a2, v109, v110);
          break;
        case 7:
          v112 = *(_QWORD *)(a2 + 24);
          if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v112) <= 6 )
          {
            sub_16E7EE0(a2, "custom ", 7u);
          }
          else
          {
            *(_DWORD *)v112 = 1953723747;
            *(_WORD *)(v112 + 4) = 28015;
            *(_BYTE *)(v112 + 6) = 32;
            *(_QWORD *)(a2 + 24) += 7LL;
          }
          (**(void (__fastcall ***)(unsigned __int64, __int64))v49)(v49, a2);
          break;
        default:
          break;
      }
    }
  }
  else
  {
    v19 = v18 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v18 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v20 = *(_WORD *)(a1 + 32);
      v21 = " into ";
      if ( (v20 & 1) != 0 )
      {
        v21 = " on ";
        if ( (v20 & 2) == 0 )
          v21 = " from ";
      }
      v22 = strlen(v21);
      v23 = *(_QWORD **)(a2 + 24);
      if ( v22 <= *(_QWORD *)(a2 + 16) - (_QWORD)v23 )
      {
        if ( (unsigned int)v22 >= 8 )
        {
          v99 = (unsigned __int64)(v23 + 1) & 0xFFFFFFFFFFFFFFF8LL;
          *v23 = *(_QWORD *)v21;
          *(_QWORD *)((char *)v23 + (unsigned int)v22 - 8) = *(_QWORD *)&v21[(unsigned int)v22 - 8];
          v100 = (char *)v23 - v99;
          v101 = (char *)(v21 - v100);
          if ( (((_DWORD)v22 + (_DWORD)v100) & 0xFFFFFFF8) >= 8 )
          {
            v102 = (v22 + (_DWORD)v100) & 0xFFFFFFF8;
            v103 = 0;
            do
            {
              v104 = v103;
              v103 += 8;
              *(_QWORD *)(v99 + v104) = *(_QWORD *)&v101[v104];
            }
            while ( v103 < v102 );
          }
        }
        else if ( (v22 & 4) != 0 )
        {
          *(_DWORD *)v23 = *(_DWORD *)v21;
          *(_DWORD *)((char *)v23 + (unsigned int)v22 - 4) = *(_DWORD *)&v21[(unsigned int)v22 - 4];
        }
        else if ( (_DWORD)v22 )
        {
          *(_BYTE *)v23 = *v21;
          if ( (v22 & 2) != 0 )
            *(_WORD *)((char *)v23 + (unsigned int)v22 - 2) = *(_WORD *)&v21[(unsigned int)v22 - 2];
        }
        *(_QWORD *)(a2 + 24) += v22;
      }
      else
      {
        sub_16E7EE0(a2, v21, v22);
      }
      v24 = *(_BYTE *)(v19 + 16);
      if ( v24 > 3u )
      {
        v95 = *(_QWORD *)(a2 + 16);
        v96 = *(_BYTE **)(a2 + 24);
        if ( v24 <= 0x10u )
        {
          if ( v95 <= (unsigned __int64)v96 )
          {
            sub_16E7DE0(a2, 96);
          }
          else
          {
            *(_QWORD *)(a2 + 24) = v96 + 1;
            *v96 = 96;
          }
          sub_1553920((__int64 *)v19, a2, 1, a3);
          v121 = *(_BYTE **)(a2 + 24);
          if ( (unsigned __int64)v121 >= *(_QWORD *)(a2 + 16) )
          {
            sub_16E7DE0(a2, 96);
          }
          else
          {
            *(_QWORD *)(a2 + 24) = v121 + 1;
            *v121 = 96;
          }
        }
        else
        {
          v97 = v95 - (_QWORD)v96;
          if ( v97 <= 3 )
          {
            sub_16E7EE0(a2, "%ir.", 4u);
          }
          else
          {
            *(_DWORD *)v96 = 779249957;
            *(_QWORD *)(a2 + 24) += 4LL;
          }
          if ( (*(_BYTE *)(v19 + 23) & 0x20) != 0 )
          {
            v122 = sub_1649960(v19);
            sub_154B650(a2, v122, v123);
          }
          else
          {
            v98 = sub_154F480(a3, v19, v97, (__int64)v96);
            sub_1E32200(a2, v98);
          }
        }
      }
      else
      {
        sub_1553920((__int64 *)v19, a2, 0, a3);
      }
    }
  }
  sub_1E32140(a2, *(_QWORD *)(a1 + 8));
  if ( *(_QWORD *)(a1 + 24) != (unsigned int)(1 << *(_WORD *)(a1 + 34)) >> 1 )
  {
    v47 = *(_QWORD **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v47 <= 7u )
    {
      v48 = sub_16E7EE0(a2, ", align ", 8u);
    }
    else
    {
      v48 = a2;
      *v47 = 0x206E67696C61202CLL;
      *(_QWORD *)(a2 + 24) += 8LL;
    }
    sub_16E7A90(v48, (unsigned int)(1 << *(_WORD *)(a1 + 34)) >> 1);
  }
  v25 = *(unsigned __int8 **)(a1 + 40);
  v26 = *(unsigned __int8 **)(a1 + 48);
  v27 = *(unsigned __int8 **)(a1 + 56);
  if ( v25 )
  {
    v28 = *(_QWORD **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v28 <= 7u )
    {
      sub_16E7EE0(a2, ", !tbaa ", 8u);
    }
    else
    {
      *v28 = 0x206161627421202CLL;
      *(_QWORD *)(a2 + 24) += 8LL;
    }
    sub_1556260(v25, a2, a3, 0);
  }
  if ( v26 )
  {
    v29 = *(void **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v29 <= 0xEu )
    {
      sub_16E7EE0(a2, ", !alias.scope ", 0xFu);
    }
    else
    {
      qmemcpy(v29, ", !alias.scope ", 15);
      *(_QWORD *)(a2 + 24) += 15LL;
    }
    sub_1556260(v26, a2, a3, 0);
  }
  if ( v27 )
  {
    v30 = *(void **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v30 <= 0xAu )
    {
      sub_16E7EE0(a2, ", !noalias ", 0xBu);
    }
    else
    {
      qmemcpy(v30, ", !noalias ", 11);
      *(_QWORD *)(a2 + 24) += 11LL;
    }
    sub_1556260(v27, a2, a3, 0);
  }
  if ( *(_QWORD *)(a1 + 64) )
  {
    v31 = *(_QWORD *)(a2 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v31) <= 8 )
    {
      sub_16E7EE0(a2, ", !range ", 9u);
    }
    else
    {
      *(_BYTE *)(v31 + 8) = 32;
      *(_QWORD *)v31 = 0x65676E617221202CLL;
      *(_QWORD *)(a2 + 24) += 9LL;
    }
    sub_1556260(*(unsigned __int8 **)(a1 + 64), a2, a3, 0);
  }
  v32 = sub_1E340A0(a1);
  if ( v32 )
  {
    v54 = *(void **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v54 <= 0xBu )
    {
      v94 = sub_16E7EE0(a2, ", addrspace ", 0xCu);
      sub_16E7A90(v94, v32);
    }
    else
    {
      qmemcpy(v54, ", addrspace ", 12);
      *(_QWORD *)(a2 + 24) += 12LL;
      sub_16E7A90(a2, v32);
    }
    result = *(_BYTE **)(a2 + 24);
    if ( (unsigned __int64)result < *(_QWORD *)(a2 + 16) )
      goto LABEL_48;
  }
  else
  {
    result = *(_BYTE **)(a2 + 24);
    if ( (unsigned __int64)result < *(_QWORD *)(a2 + 16) )
    {
LABEL_48:
      *(_QWORD *)(a2 + 24) = result + 1;
      *result = 41;
      return result;
    }
  }
  return (_BYTE *)sub_16E7DE0(a2, 41);
}
