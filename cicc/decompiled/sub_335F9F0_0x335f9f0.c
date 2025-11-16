// Function: sub_335F9F0
// Address: 0x335f9f0
//
void __fastcall sub_335F9F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int *v6; // rax
  __int64 v7; // r8
  __int64 v8; // r11
  char *v9; // rdx
  int v11; // eax
  __int64 v12; // rcx
  __int64 v13; // rcx
  unsigned int v14; // edi
  __int64 v15; // rsi
  unsigned int v16; // eax
  __int64 v17; // r12
  __int64 v18; // rsi
  void *v19; // rdi
  int v20; // r15d
  int v21; // r13d
  __int64 v22; // rbx
  char *v23; // rax
  __int64 *v24; // rdi
  __int64 v25; // r10
  __int64 v26; // r8
  int v27; // ebx
  unsigned int v28; // edi
  __int64 *v29; // rax
  __int64 v30; // rcx
  __int64 v31; // rdx
  __int64 *v32; // rax
  int v33; // r15d
  unsigned int v34; // r12d
  unsigned int v35; // r14d
  __int64 v36; // rbx
  _QWORD *v37; // rdx
  int v38; // r10d
  unsigned int v39; // edi
  _QWORD *v40; // rax
  __int64 v41; // rcx
  __int64 v42; // r13
  __int64 v43; // rdi
  __int64 (*v44)(); // rax
  __int64 v45; // rdi
  __int64 (*v46)(); // rax
  unsigned int v47; // eax
  __int64 v48; // r10
  unsigned int v49; // edi
  __int64 v50; // rsi
  unsigned int v51; // eax
  unsigned int v52; // eax
  __int64 v53; // rsi
  unsigned int v54; // ecx
  __int64 *v55; // rdx
  unsigned int v56; // ecx
  __int64 *v57; // rdx
  __int64 v58; // r15
  __int64 v59; // rax
  unsigned __int64 v60; // rdx
  int v61; // eax
  __int64 v62; // r8
  __int64 v63; // r9
  __int64 v64; // rax
  unsigned __int64 v65; // rdx
  unsigned __int64 v66; // r12
  __int64 v67; // r14
  unsigned int v68; // r15d
  unsigned int v69; // ebx
  __int64 v70; // r13
  unsigned int v71; // esi
  __int64 v72; // rdi
  int v73; // r11d
  _QWORD *v74; // r9
  _QWORD *v75; // r8
  unsigned int v76; // r13d
  int v77; // r10d
  __int64 v78; // rsi
  __int64 *v79; // rdi
  _QWORD *v80; // rax
  __int64 v81; // rdx
  __int64 v82; // rax
  __int64 *v83; // r11
  _QWORD *v84; // rax
  __int64 v85; // r9
  int v86; // ecx
  unsigned int v87; // edx
  unsigned __int64 v88; // rdx
  int v89; // r10d
  int v90; // edi
  __int64 *v91; // rsi
  __int64 v92; // rdx
  int v93; // r15d
  int v94; // edi
  unsigned __int8 v95; // [rsp+17h] [rbp-189h]
  __int64 v96; // [rsp+18h] [rbp-188h]
  char v97; // [rsp+20h] [rbp-180h]
  unsigned __int8 v98; // [rsp+20h] [rbp-180h]
  unsigned __int8 v99; // [rsp+20h] [rbp-180h]
  __int64 v100; // [rsp+20h] [rbp-180h]
  __int64 v101; // [rsp+28h] [rbp-178h]
  __int64 v102; // [rsp+28h] [rbp-178h]
  int v103; // [rsp+28h] [rbp-178h]
  __int64 v105; // [rsp+40h] [rbp-160h] BYREF
  __int64 v106; // [rsp+48h] [rbp-158h] BYREF
  __int64 v107; // [rsp+50h] [rbp-150h] BYREF
  __int64 v108; // [rsp+58h] [rbp-148h]
  __int64 v109; // [rsp+60h] [rbp-140h]
  unsigned int v110; // [rsp+68h] [rbp-138h]
  void *base; // [rsp+70h] [rbp-130h] BYREF
  __int64 v112; // [rsp+78h] [rbp-128h]
  _BYTE v113[32]; // [rsp+80h] [rbp-120h] BYREF
  __int64 *v114; // [rsp+A0h] [rbp-100h] BYREF
  __int64 v115; // [rsp+A8h] [rbp-F8h]
  _BYTE v116[32]; // [rsp+B0h] [rbp-F0h] BYREF
  __int64 v117; // [rsp+D0h] [rbp-D0h] BYREF
  char *v118; // [rsp+D8h] [rbp-C8h]
  __int64 v119; // [rsp+E0h] [rbp-C0h]
  int v120; // [rsp+E8h] [rbp-B8h]
  char v121; // [rsp+ECh] [rbp-B4h]
  char v122; // [rsp+F0h] [rbp-B0h] BYREF

  v6 = (unsigned int *)(*(_QWORD *)(a2 + 40) + 40LL * (unsigned int)(*(_DWORD *)(a2 + 64) - 1));
  v7 = *(_QWORD *)v6;
  v8 = v6[2];
  if ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)v6 + 48LL) + 16 * v8) != 1 )
    return;
  v9 = *(char **)(a1 + 16);
  v119 = 16;
  v118 = &v122;
  base = v113;
  v112 = 0x400000000LL;
  v11 = *(_DWORD *)(a2 + 24);
  v120 = 0;
  v121 = 1;
  v108 = 0;
  v109 = 0;
  v110 = 0;
  v12 = *((_QWORD *)v9 + 1);
  v117 = 0;
  v107 = 0;
  v13 = v12 - 40LL * (unsigned int)~v11;
  v14 = *(unsigned __int16 *)(v13 + 2);
  if ( *(_WORD *)(v13 + 2) )
  {
    v15 = 0;
    v16 = 0;
    while ( 1 )
    {
      if ( v16 < v14 )
      {
        a6 = 5LL * *(unsigned __int16 *)v13 + 5;
        v9 = (char *)(v13 + 6LL * *(unsigned __int16 *)(v13 + 16) + 8 * a6);
        if ( (v9[v15 + 4] & 1) != 0 )
          break;
      }
      ++v16;
      v15 += 6;
      if ( v16 == v14 )
        goto LABEL_13;
    }
    v17 = 0;
LABEL_7:
    v18 = 0;
    goto LABEL_8;
  }
LABEL_13:
  v17 = *(_QWORD *)(v7 + 56);
  v20 = 0;
  v21 = v8;
  v101 = a2;
  v97 = 0;
  if ( !v17 )
    goto LABEL_7;
  do
  {
    if ( v21 != *(_DWORD *)(v17 + 8) )
      goto LABEL_21;
    v22 = *(_QWORD *)(v17 + 16);
    if ( a2 == v22 )
      goto LABEL_21;
    if ( !v121 )
      goto LABEL_40;
    v23 = v118;
    v13 = HIDWORD(v119);
    v9 = &v118[8 * HIDWORD(v119)];
    if ( v118 != v9 )
    {
      while ( v22 != *(_QWORD *)v23 )
      {
        v23 += 8;
        if ( v9 == v23 )
          goto LABEL_58;
      }
      goto LABEL_21;
    }
LABEL_58:
    if ( HIDWORD(v119) < (unsigned int)v119 )
    {
      v13 = (unsigned int)++HIDWORD(v119);
      *(_QWORD *)v9 = v22;
      ++v117;
    }
    else
    {
LABEL_40:
      sub_C8CC70((__int64)&v117, *(_QWORD *)(v17 + 16), (__int64)v9, v13, v7, a6);
      if ( !(_BYTE)v9 )
        goto LABEL_21;
    }
    v45 = *(_QWORD *)(a1 + 16);
    v46 = *(__int64 (**)())(*(_QWORD *)v45 + 800LL);
    if ( v46 == sub_2FDC690 )
      goto LABEL_21;
    v47 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64 *, __int64 *))v46)(v45, v101, v22, &v105, &v106);
    v7 = v47;
    if ( !(_BYTE)v47 )
      goto LABEL_21;
    v48 = v106;
    if ( v105 == v106 )
      goto LABEL_21;
    v13 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) - 40LL * (unsigned int)~*(_DWORD *)(v22 + 24);
    v49 = *(unsigned __int16 *)(v13 + 2);
    if ( *(_WORD *)(v13 + 2) )
    {
      v50 = 0;
      v51 = 0;
      while ( 1 )
      {
        if ( v49 > v51 )
        {
          a6 = 5LL * *(unsigned __int16 *)v13 + 5;
          v9 = (char *)(v13 + 6LL * *(unsigned __int16 *)(v13 + 16) + 8 * a6);
          if ( (v9[v50 + 4] & 1) != 0 )
            break;
        }
        ++v51;
        v50 += 6;
        if ( v51 == v49 )
          goto LABEL_49;
      }
LABEL_21:
      LOBYTE(v7) = (unsigned int)++v20 <= 0x63;
      goto LABEL_22;
    }
LABEL_49:
    v114 = (__int64 *)v105;
    v115 = v101;
    if ( v110 )
    {
      v52 = v110 - 1;
      v53 = v108;
      v54 = (v110 - 1) & (37 * v105);
      v55 = (__int64 *)(v108 + 16LL * v54);
      a6 = *v55;
      if ( v105 == *v55 )
      {
LABEL_51:
        v114 = (__int64 *)v48;
        v115 = v22;
        goto LABEL_52;
      }
      v93 = 1;
      v79 = 0;
      while ( a6 != 0x7FFFFFFFFFFFFFFFLL )
      {
        if ( !v79 && a6 == 0x8000000000000000LL )
          v79 = v55;
        v54 = v52 & (v93 + v54);
        v55 = (__int64 *)(v108 + 16LL * v54);
        a6 = *v55;
        if ( v105 == *v55 )
        {
          v48 = v106;
          goto LABEL_51;
        }
        ++v93;
      }
      if ( !v79 )
        v79 = v55;
    }
    else
    {
      v79 = 0;
    }
    v95 = v7;
    v80 = sub_335F830((__int64)&v107, (__int64 *)&v114, v79);
    a6 = (__int64)&v114;
    v7 = v95;
    *v80 = v114;
    v80[1] = v115;
    v81 = (unsigned int)v112;
    v82 = v105;
    if ( (unsigned __int64)(unsigned int)v112 + 1 > HIDWORD(v112) )
    {
      v100 = v105;
      sub_C8D5F0((__int64)&base, v113, (unsigned int)v112 + 1LL, 8u, v95, (__int64)&v114);
      v81 = (unsigned int)v112;
      a6 = (__int64)&v114;
      v7 = v95;
      v82 = v100;
    }
    *((_QWORD *)base + v81) = v82;
    v48 = v106;
    LODWORD(v112) = v112 + 1;
    v114 = (__int64 *)v106;
    v53 = v108;
    v115 = v22;
    if ( !v110 )
    {
      v83 = 0;
LABEL_113:
      v98 = v7;
      v84 = sub_335F830((__int64)&v107, (__int64 *)&v114, v83);
      v7 = v98;
      *v84 = v114;
      v58 = v106;
      v84[1] = v115;
      goto LABEL_53;
    }
    v52 = v110 - 1;
LABEL_52:
    v56 = v52 & (37 * v48);
    v57 = (__int64 *)(v53 + 16LL * v56);
    v58 = *v57;
    if ( v48 != *v57 )
    {
      a6 = 1;
      v83 = 0;
      while ( v58 != 0x7FFFFFFFFFFFFFFFLL )
      {
        if ( v58 == 0x8000000000000000LL && !v83 )
          v83 = v57;
        v56 = v52 & (a6 + v56);
        v57 = (__int64 *)(v53 + 16LL * v56);
        v58 = *v57;
        if ( v48 == *v57 )
          goto LABEL_53;
        a6 = (unsigned int)(a6 + 1);
      }
      if ( !v83 )
        v83 = v57;
      goto LABEL_113;
    }
LABEL_53:
    v59 = (unsigned int)v112;
    v13 = HIDWORD(v112);
    v60 = (unsigned int)v112 + 1LL;
    if ( v60 > HIDWORD(v112) )
    {
      v99 = v7;
      sub_C8D5F0((__int64)&base, v113, v60, 8u, v7, a6);
      v59 = (unsigned int)v112;
      v7 = v99;
    }
    v9 = (char *)base;
    v97 = v7;
    *((_QWORD *)base + v59) = v58;
    v20 = 1;
    LODWORD(v112) = v112 + 1;
    if ( v106 >= v105 )
      v22 = v101;
    v101 = v22;
LABEL_22:
    v17 = *(_QWORD *)(v17 + 32);
  }
  while ( v17 && (_BYTE)v7 );
  if ( !v97 )
    goto LABEL_38;
  v24 = (__int64 *)base;
  if ( (unsigned int)v112 > 1uLL )
  {
    qsort(base, (8LL * (unsigned int)v112) >> 3, 8u, (__compar_fn_t)sub_F8E3D0);
    v24 = (__int64 *)base;
  }
  v114 = (__int64 *)v116;
  v115 = 0x400000000LL;
  v25 = *v24;
  v102 = *v24;
  if ( !v110 )
  {
    ++v107;
    goto LABEL_116;
  }
  v26 = v110 - 1;
  v27 = 37 * v25;
  v28 = v26 & (37 * v25);
  v29 = (__int64 *)(v108 + 16LL * v28);
  v30 = *v29;
  if ( v25 != *v29 )
  {
    v89 = 1;
    v85 = 0;
    while ( v30 != 0x7FFFFFFFFFFFFFFFLL )
    {
      if ( v30 == 0x8000000000000000LL && !v85 )
        v85 = (__int64)v29;
      v28 = v26 & (v89 + v28);
      v29 = (__int64 *)(v108 + 16LL * v28);
      v30 = *v29;
      if ( v102 == *v29 )
        goto LABEL_29;
      ++v89;
    }
    if ( v85 )
      v29 = (__int64 *)v85;
    ++v107;
    v86 = v109 + 1;
    if ( 4 * ((int)v109 + 1) < 3 * v110 )
    {
      if ( v110 - HIDWORD(v109) - v86 > v110 >> 3 )
        goto LABEL_118;
      sub_335F610((__int64)&v107, v110);
      if ( v110 )
      {
        v85 = v110 - 1;
        v90 = 1;
        v91 = 0;
        v92 = (unsigned int)v85 & v27;
        v29 = (__int64 *)(v108 + 16 * v92);
        v86 = v109 + 1;
        v26 = *v29;
        if ( v102 != *v29 )
        {
          while ( v26 != 0x7FFFFFFFFFFFFFFFLL )
          {
            if ( !v91 && v26 == 0x8000000000000000LL )
              v91 = v29;
            LODWORD(v92) = v85 & (v90 + v92);
            v29 = (__int64 *)(v108 + 16LL * (unsigned int)v92);
            v26 = *v29;
            if ( v102 == *v29 )
              goto LABEL_118;
            ++v90;
          }
LABEL_132:
          if ( v91 )
            v29 = v91;
          goto LABEL_118;
        }
        goto LABEL_118;
      }
      goto LABEL_185;
    }
LABEL_116:
    sub_335F610((__int64)&v107, 2 * v110);
    if ( v110 )
    {
      v85 = v110 - 1;
      v86 = v109 + 1;
      v87 = v85 & (37 * v102);
      v29 = (__int64 *)(v108 + 16LL * v87);
      v26 = *v29;
      if ( v102 != *v29 )
      {
        v94 = 1;
        v91 = 0;
        while ( v26 != 0x7FFFFFFFFFFFFFFFLL )
        {
          if ( v26 == 0x8000000000000000LL && !v91 )
            v91 = v29;
          v87 = v85 & (v94 + v87);
          v29 = (__int64 *)(v108 + 16LL * v87);
          v26 = *v29;
          if ( v102 == *v29 )
            goto LABEL_118;
          ++v94;
        }
        goto LABEL_132;
      }
LABEL_118:
      LODWORD(v109) = v86;
      if ( *v29 != 0x7FFFFFFFFFFFFFFFLL )
        --HIDWORD(v109);
      v29[1] = 0;
      *v29 = v102;
      v88 = (unsigned int)v115 + 1LL;
      if ( v88 > HIDWORD(v115) )
        sub_C8D5F0((__int64)&v114, v116, v88, 8u, v26, v85);
      v96 = 0;
      v32 = &v114[(unsigned int)v115];
      v31 = 0;
      goto LABEL_30;
    }
LABEL_185:
    LODWORD(v109) = v109 + 1;
    BUG();
  }
LABEL_29:
  v96 = v29[1];
  v31 = v96;
  v32 = (__int64 *)v116;
LABEL_30:
  *v32 = v31;
  v33 = v112;
  v34 = 1;
  LODWORD(v115) = v115 + 1;
  if ( (_DWORD)v112 == 1 )
    goto LABEL_36;
  while ( 2 )
  {
    v35 = v34 - 1;
    v36 = *((_QWORD *)base + v34);
    if ( v110 )
    {
      v37 = 0;
      v38 = 1;
      v39 = (v110 - 1) & (37 * v36);
      v40 = (_QWORD *)(v108 + 16LL * v39);
      v41 = *v40;
      if ( v36 == *v40 )
      {
LABEL_33:
        v42 = v40[1];
        goto LABEL_34;
      }
      while ( v41 != 0x7FFFFFFFFFFFFFFFLL )
      {
        if ( v41 == 0x8000000000000000LL && !v37 )
          v37 = v40;
        v39 = (v110 - 1) & (v38 + v39);
        v40 = (_QWORD *)(v108 + 16LL * v39);
        v41 = *v40;
        if ( v36 == *v40 )
          goto LABEL_33;
        ++v38;
      }
      if ( !v37 )
        v37 = v40;
      ++v107;
      v61 = v109 + 1;
      if ( 4 * ((int)v109 + 1) < 3 * v110 )
      {
        if ( v110 - HIDWORD(v109) - v61 <= v110 >> 3 )
        {
          sub_335F610((__int64)&v107, v110);
          if ( !v110 )
          {
LABEL_184:
            LODWORD(v109) = v109 + 1;
            BUG();
          }
          v75 = 0;
          v76 = (v110 - 1) & (37 * v36);
          v77 = 1;
          v61 = v109 + 1;
          v37 = (_QWORD *)(v108 + 16LL * v76);
          v78 = *v37;
          if ( v36 != *v37 )
          {
            while ( v78 != 0x7FFFFFFFFFFFFFFFLL )
            {
              if ( v78 == 0x8000000000000000LL && !v75 )
                v75 = v37;
              v76 = (v110 - 1) & (v76 + v77);
              v37 = (_QWORD *)(v108 + 16LL * v76);
              v78 = *v37;
              if ( v36 == *v37 )
                goto LABEL_70;
              ++v77;
            }
            if ( v75 )
              v37 = v75;
          }
        }
        goto LABEL_70;
      }
    }
    else
    {
      ++v107;
    }
    sub_335F610((__int64)&v107, 2 * v110);
    if ( !v110 )
      goto LABEL_184;
    v71 = (v110 - 1) & (37 * v36);
    v61 = v109 + 1;
    v37 = (_QWORD *)(v108 + 16LL * v71);
    v72 = *v37;
    if ( v36 != *v37 )
    {
      v73 = 1;
      v74 = 0;
      while ( v72 != 0x7FFFFFFFFFFFFFFFLL )
      {
        if ( v72 == 0x8000000000000000LL && !v74 )
          v74 = v37;
        v71 = (v110 - 1) & (v71 + v73);
        v37 = (_QWORD *)(v108 + 16LL * v71);
        v72 = *v37;
        if ( v36 == *v37 )
          goto LABEL_70;
        ++v73;
      }
      if ( v74 )
        v37 = v74;
    }
LABEL_70:
    LODWORD(v109) = v61;
    if ( *v37 != 0x7FFFFFFFFFFFFFFFLL )
      --HIDWORD(v109);
    *v37 = v36;
    v42 = 0;
    v37[1] = 0;
LABEL_34:
    v43 = *(_QWORD *)(a1 + 16);
    v44 = *(__int64 (**)())(*(_QWORD *)v43 + 808LL);
    if ( v44 == sub_2FDC6A0
      || !((unsigned __int8 (__fastcall *)(__int64, __int64, __int64, __int64, __int64, _QWORD))v44)(
            v43,
            v96,
            v42,
            v102,
            v36,
            v35) )
    {
      goto LABEL_35;
    }
    v64 = (unsigned int)v115;
    v65 = (unsigned int)v115 + 1LL;
    if ( v65 > HIDWORD(v115) )
    {
      sub_C8D5F0((__int64)&v114, v116, v65, 8u, v62, v63);
      v64 = (unsigned int)v115;
    }
    v114[v64] = v42;
    LODWORD(v115) = v115 + 1;
    if ( v34 + 1 != v33 )
    {
      ++v34;
      continue;
    }
    break;
  }
  v35 = v34;
LABEL_35:
  if ( v35 )
  {
    v66 = 0;
    v67 = *v114;
    if ( (unsigned __int8)sub_335C750(*v114, 0, 0, 1, *(_QWORD *)(a1 + 592)) )
      v66 = (unsigned int)(*(_DWORD *)(v67 + 68) - 1);
    else
      v67 = 0;
    v68 = 1;
    v103 = v115;
    v69 = v115 - 1;
    if ( (_DWORD)v115 != 1 )
    {
      do
      {
        v70 = v114[v68];
        if ( (unsigned __int8)sub_335C750(v70, v67, v66, v68 < v69, *(_QWORD *)(a1 + 592)) )
        {
          if ( v68 < v69 )
          {
            v67 = v70;
            v66 = (unsigned int)(*(_DWORD *)(v70 + 68) - 1) | v66 & 0xFFFFFFFF00000000LL;
          }
        }
        else if ( v68 >= v69 && v67 )
        {
          sub_335C390(v67, *(_QWORD *)(a1 + 592), *(void **)(v67 + 48), (unsigned int)(*(_DWORD *)(v67 + 68) - 1), 0, 0);
        }
        ++v68;
      }
      while ( v68 != v103 );
    }
    if ( v114 != (__int64 *)v116 )
      _libc_free((unsigned __int64)v114);
    sub_C7D6A0(v108, 16LL * v110, 8);
    v19 = base;
    if ( base != v113 )
      goto LABEL_9;
  }
  else
  {
LABEL_36:
    if ( v114 != (__int64 *)v116 )
      _libc_free((unsigned __int64)v114);
LABEL_38:
    v17 = v108;
    v18 = 16LL * v110;
LABEL_8:
    sub_C7D6A0(v17, v18, 8);
    v19 = base;
    if ( base != v113 )
LABEL_9:
      _libc_free((unsigned __int64)v19);
  }
  if ( !v121 )
    _libc_free((unsigned __int64)v118);
}
