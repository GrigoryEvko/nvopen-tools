// Function: sub_1E181D0
// Address: 0x1e181d0
//
void __fastcall sub_1E181D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, _BYTE *a6, char a7, __int64 a8)
{
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // r13
  __int64 (*v12)(void); // rax
  __int64 (*v13)(void); // rax
  __int64 (*v14)(void); // rax
  __int64 v15; // r15
  unsigned int v16; // r14d
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rcx
  int v20; // eax
  __int64 v21; // rdx
  _BYTE *v22; // r13
  char v23; // al
  _WORD *v24; // rdx
  __int16 v25; // ax
  __int64 v26; // rdx
  char *v27; // rsi
  size_t v28; // rax
  void *v29; // rdi
  size_t v30; // r15
  __int64 v31; // rdx
  __int64 v32; // r8
  _BYTE *v33; // rax
  __int64 v34; // r15
  __int64 v35; // r13
  __int16 v36; // ax
  __int64 v37; // rcx
  int v38; // eax
  __int64 v39; // r8
  __int16 v40; // dx
  _QWORD *v41; // r13
  __m128i *v42; // rdx
  unsigned __int8 *v43; // rax
  __int64 v44; // rax
  __int64 *v45; // rax
  __int64 v46; // r14
  __int64 *v47; // r15
  _DWORD *v48; // rdx
  __int64 *v49; // r10
  __int64 v50; // r11
  __int64 *v51; // rbx
  _WORD *v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // rdi
  _QWORD *v55; // rcx
  __int64 v56; // r15
  __int64 v57; // rcx
  __int64 v58; // rdi
  _BYTE *v59; // rax
  __int64 v60; // r13
  __int64 v61; // rdi
  __int64 v62; // rax
  size_t v63; // rdx
  char *v64; // rdi
  void *v65; // rsi
  unsigned __int64 v66; // rax
  _BYTE *v67; // rax
  __int64 v68; // rdi
  unsigned int v69; // eax
  __int64 v70; // rcx
  unsigned int v71; // r13d
  _BYTE *v72; // rax
  _BYTE *v73; // rax
  unsigned __int64 v74; // r12
  _QWORD *v75; // rdx
  _DWORD *v76; // rdx
  __int64 v77; // rdx
  __int64 v78; // rdx
  void *v79; // rdx
  void *v80; // rdx
  __int64 v81; // rdx
  __int64 v82; // rdx
  _DWORD *v83; // rdx
  __int64 v84; // rdx
  _BYTE *v85; // rax
  _WORD *v86; // rdx
  _BYTE *v87; // rax
  __int64 v88; // rcx
  int v89; // eax
  __int64 v90; // rdi
  __int64 v91; // r13
  _BYTE *v92; // rax
  __int64 v93; // r14
  __int64 v94; // rax
  _BYTE *v95; // rdx
  __int64 v96; // r14
  __int64 v97; // rdi
  __int64 v98; // rax
  __int64 v99; // rsi
  _BYTE *v100; // rax
  __int64 v101; // rax
  __int64 *v102; // rax
  __int64 v103; // rdx
  unsigned __int64 v104; // rax
  unsigned __int64 v105; // rsi
  __int64 v106; // rdx
  int v107; // edx
  unsigned int v108; // r13d
  __int64 v109; // rdi
  __int64 v110; // rdx
  unsigned __int64 v111; // rdx
  __int64 v112; // r15
  char *v113; // rsi
  size_t v114; // rax
  void *v115; // rdi
  size_t v116; // r13
  void *v117; // rdx
  _QWORD *v118; // rdx
  _QWORD *v119; // rdx
  __int64 v120; // rax
  _DWORD *v121; // rdx
  __int64 v122; // rdi
  __int64 v123; // rax
  __int64 v124; // rax
  void *v125; // rdx
  void *v126; // rdx
  void *v127; // rdx
  void *v128; // rdx
  void *v129; // rdx
  void *v130; // rdx
  void *v131; // rax
  __int64 v132; // rax
  __int64 v133; // rax
  __int64 v134; // rax
  __int16 v135; // [rsp+0h] [rbp-B0h]
  _QWORD *src; // [rsp+8h] [rbp-A8h]
  int srcc; // [rsp+8h] [rbp-A8h]
  _QWORD *srca; // [rsp+8h] [rbp-A8h]
  char *srcd; // [rsp+8h] [rbp-A8h]
  void *srcb; // [rsp+8h] [rbp-A8h]
  __int16 srce; // [rsp+8h] [rbp-A8h]
  __int16 srcf; // [rsp+8h] [rbp-A8h]
  __int16 srcg; // [rsp+8h] [rbp-A8h]
  int srch; // [rsp+8h] [rbp-A8h]
  __int16 srci; // [rsp+8h] [rbp-A8h]
  int srcj; // [rsp+8h] [rbp-A8h]
  int srck; // [rsp+8h] [rbp-A8h]
  int srcl; // [rsp+8h] [rbp-A8h]
  int srcm; // [rsp+8h] [rbp-A8h]
  int srcn; // [rsp+8h] [rbp-A8h]
  int srco; // [rsp+8h] [rbp-A8h]
  int srcp; // [rsp+8h] [rbp-A8h]
  __int16 srcq; // [rsp+8h] [rbp-A8h]
  int srcr; // [rsp+8h] [rbp-A8h]
  int srcs; // [rsp+8h] [rbp-A8h]
  int srct; // [rsp+8h] [rbp-A8h]
  int srcu; // [rsp+8h] [rbp-A8h]
  int srcv; // [rsp+8h] [rbp-A8h]
  int srcw; // [rsp+8h] [rbp-A8h]
  int srcx; // [rsp+8h] [rbp-A8h]
  int srcy; // [rsp+8h] [rbp-A8h]
  int srcz; // [rsp+8h] [rbp-A8h]
  int srcba; // [rsp+8h] [rbp-A8h]
  int srcbb; // [rsp+8h] [rbp-A8h]
  int srcbc; // [rsp+8h] [rbp-A8h]
  int srcbd; // [rsp+8h] [rbp-A8h]
  int srcbe; // [rsp+8h] [rbp-A8h]
  int srcbf; // [rsp+8h] [rbp-A8h]
  int srcbg; // [rsp+8h] [rbp-A8h]
  int srcbh; // [rsp+8h] [rbp-A8h]
  int srcbi; // [rsp+8h] [rbp-A8h]
  int srcbj; // [rsp+8h] [rbp-A8h]
  int srcbk; // [rsp+8h] [rbp-A8h]
  int srcbl; // [rsp+8h] [rbp-A8h]
  unsigned int v175; // [rsp+14h] [rbp-9Ch]
  char v176; // [rsp+1Ch] [rbp-94h]
  int v177; // [rsp+20h] [rbp-90h]
  int v178; // [rsp+20h] [rbp-90h]
  int v179; // [rsp+20h] [rbp-90h]
  char v180; // [rsp+28h] [rbp-88h]
  unsigned __int8 v181; // [rsp+2Ch] [rbp-84h]
  int v182; // [rsp+2Ch] [rbp-84h]
  unsigned int v183; // [rsp+30h] [rbp-80h]
  unsigned __int8 v184; // [rsp+36h] [rbp-7Ah]
  unsigned __int8 v185; // [rsp+37h] [rbp-79h]
  __int64 v186; // [rsp+38h] [rbp-78h]
  __int64 v187; // [rsp+38h] [rbp-78h]
  __int64 v188; // [rsp+40h] [rbp-70h]
  __int64 v189; // [rsp+40h] [rbp-70h]
  __int64 v191; // [rsp+50h] [rbp-60h]
  __int64 *v192; // [rsp+50h] [rbp-60h]
  unsigned __int8 v193; // [rsp+58h] [rbp-58h]
  __int64 *v194; // [rsp+58h] [rbp-58h]
  unsigned __int64 v195; // [rsp+68h] [rbp-48h] BYREF
  __int64 v196[2]; // [rsp+70h] [rbp-40h] BYREF
  _BYTE v197[48]; // [rsp+80h] [rbp-30h] BYREF

  v9 = a1;
  v181 = a4;
  v10 = *(_QWORD *)(a1 + 24);
  v180 = a5;
  v176 = (char)a6;
  v185 = a4;
  v184 = a5;
  v188 = v10;
  if ( v10 )
  {
    v11 = *(_QWORD *)(v10 + 56);
    if ( v11 )
    {
      v188 = 0;
      v12 = *(__int64 (**)(void))(**(_QWORD **)(v11 + 16) + 112LL);
      if ( v12 != sub_1D00B10 )
        v188 = v12();
      v186 = 0;
      v191 = *(_QWORD *)(v11 + 40);
      v13 = *(__int64 (**)(void))(**(_QWORD **)(v11 + 8) + 32LL);
      if ( v13 != sub_16FF770 )
        v186 = v13();
      a8 = 0;
      v14 = *(__int64 (**)(void))(**(_QWORD **)(v11 + 16) + 40LL);
      if ( v14 != sub_1D00B00 )
        a8 = v14();
    }
    else
    {
      v186 = 0;
      v191 = 0;
      v188 = 0;
    }
  }
  else
  {
    v186 = 0;
    v191 = 0;
  }
  v15 = 0;
  v16 = 0;
  v195 = 0x2000000000000001LL;
  v193 = sub_1E17EA0(a1, a2, a3, a4, a5, a6);
  v183 = *(_DWORD *)(a1 + 40);
  if ( v183 )
  {
    while ( 1 )
    {
      v22 = (_BYTE *)(v15 + *(_QWORD *)(a1 + 32));
      if ( *v22 )
        break;
      v23 = v22[3];
      if ( (v23 & 0x10) == 0 || (v23 & 0x20) != 0 )
        break;
      if ( v16 )
      {
        v24 = *(_WORD **)(a2 + 24);
        if ( *(_QWORD *)(a2 + 16) - (_QWORD)v24 <= 1u )
        {
          sub_16E7EE0(a2, ", ", 2u);
        }
        else
        {
          *v24 = 8236;
          *(_QWORD *)(a2 + 24) += 2LL;
        }
      }
      v19 = 0;
      if ( v191 )
        v19 = sub_1E17F60(a1, v16, &v195, v191);
      v20 = 0;
      if ( v193 )
      {
        v21 = v15 + *(_QWORD *)(a1 + 32);
        if ( !*(_BYTE *)v21 && (*(_WORD *)(v21 + 2) & 0xFF0) != 0 && (*(_BYTE *)(v21 + 3) & 0x10) == 0 )
        {
          v178 = v19;
          v20 = sub_1E16AB0(a1, v16, v21, v19, v17, (_BYTE *)v18);
          LODWORD(v19) = v178;
        }
      }
      ++v16;
      v15 += 40;
      sub_1E32250((_DWORD)v22, a2, a3, v19, 0, v185, v193, v20, v188, v186);
      if ( v16 == v183 )
        goto LABEL_170;
    }
    if ( !v16 )
      goto LABEL_27;
LABEL_170:
    v84 = *(_QWORD *)(a2 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v84) <= 2 )
    {
      sub_16E7EE0(a2, " = ", 3u);
    }
    else
    {
      *(_BYTE *)(v84 + 2) = 32;
      *(_WORD *)v84 = 15648;
      *(_QWORD *)(a2 + 24) += 3LL;
    }
  }
  else
  {
LABEL_27:
    v16 = 0;
  }
  v25 = *(_WORD *)(a1 + 46);
  if ( (v25 & 1) != 0 )
  {
    v79 = *(void **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v79 <= 0xBu )
    {
      sub_16E7EE0(a2, "frame-setup ", 0xCu);
    }
    else
    {
      qmemcpy(v79, "frame-setup ", 12);
      *(_QWORD *)(a2 + 24) += 12LL;
    }
    v25 = *(_WORD *)(a1 + 46);
    if ( (v25 & 2) == 0 )
    {
LABEL_30:
      if ( (v25 & 0x10) == 0 )
        goto LABEL_31;
      goto LABEL_161;
    }
  }
  else if ( (v25 & 2) == 0 )
  {
    goto LABEL_30;
  }
  v80 = *(void **)(a2 + 24);
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)v80 <= 0xDu )
  {
    sub_16E7EE0(a2, "frame-destroy ", 0xEu);
  }
  else
  {
    qmemcpy(v80, "frame-destroy ", 14);
    *(_QWORD *)(a2 + 24) += 14LL;
  }
  v25 = *(_WORD *)(a1 + 46);
  if ( (v25 & 0x10) == 0 )
  {
LABEL_31:
    if ( (v25 & 0x20) == 0 )
      goto LABEL_32;
    goto LABEL_164;
  }
LABEL_161:
  v81 = *(_QWORD *)(a2 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v81) <= 4 )
  {
    sub_16E7EE0(a2, "nnan ", 5u);
  }
  else
  {
    *(_DWORD *)v81 = 1851879022;
    *(_BYTE *)(v81 + 4) = 32;
    *(_QWORD *)(a2 + 24) += 5LL;
  }
  v25 = *(_WORD *)(a1 + 46);
  if ( (v25 & 0x20) == 0 )
  {
LABEL_32:
    if ( (v25 & 0x40) == 0 )
      goto LABEL_33;
    goto LABEL_167;
  }
LABEL_164:
  v82 = *(_QWORD *)(a2 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v82) <= 4 )
  {
    sub_16E7EE0(a2, "ninf ", 5u);
  }
  else
  {
    *(_DWORD *)v82 = 1718511982;
    *(_BYTE *)(v82 + 4) = 32;
    *(_QWORD *)(a2 + 24) += 5LL;
  }
  v25 = *(_WORD *)(a1 + 46);
  if ( (v25 & 0x40) != 0 )
  {
LABEL_167:
    v83 = *(_DWORD **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v83 <= 3u )
    {
      sub_16E7EE0(a2, "nsz ", 4u);
    }
    else
    {
      *v83 = 544895854;
      *(_QWORD *)(a2 + 24) += 4LL;
    }
    v25 = *(_WORD *)(a1 + 46);
  }
LABEL_33:
  if ( (v25 & 0x80u) != 0 )
  {
    v78 = *(_QWORD *)(a2 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v78) <= 4 )
    {
      sub_16E7EE0(a2, "arcp ", 5u);
    }
    else
    {
      *(_DWORD *)v78 = 1885565537;
      *(_BYTE *)(v78 + 4) = 32;
      *(_QWORD *)(a2 + 24) += 5LL;
    }
    v25 = *(_WORD *)(a1 + 46);
  }
  if ( (v25 & 0x100) != 0 )
  {
    v77 = *(_QWORD *)(a2 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v77) <= 8 )
    {
      sub_16E7EE0(a2, "contract ", 9u);
    }
    else
    {
      *(_BYTE *)(v77 + 8) = 32;
      *(_QWORD *)v77 = 0x74636172746E6F63LL;
      *(_QWORD *)(a2 + 24) += 9LL;
    }
    v25 = *(_WORD *)(a1 + 46);
  }
  if ( (v25 & 0x200) != 0 )
  {
    v76 = *(_DWORD **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v76 <= 3u )
    {
      sub_16E7EE0(a2, "afn ", 4u);
    }
    else
    {
      *v76 = 544106081;
      *(_QWORD *)(a2 + 24) += 4LL;
    }
    v25 = *(_WORD *)(a1 + 46);
  }
  if ( (v25 & 0x400) != 0 )
  {
    v75 = *(_QWORD **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v75 <= 7u )
    {
      sub_16E7EE0(a2, "reassoc ", 8u);
    }
    else
    {
      *v75 = 0x20636F7373616572LL;
      *(_QWORD *)(a2 + 24) += 8LL;
    }
  }
  if ( a8 )
  {
    v26 = **(unsigned __int16 **)(a1 + 16);
    v27 = (char *)(*(_QWORD *)(a8 + 24) + *(unsigned int *)(*(_QWORD *)(a8 + 16) + 4 * v26));
    if ( v27 )
    {
      v28 = strlen(v27);
      v29 = *(void **)(a2 + 24);
      v30 = v28;
      if ( *(_QWORD *)(a2 + 16) - (_QWORD)v29 < v28 )
      {
        sub_16E7EE0(a2, v27, v28);
      }
      else if ( v28 )
      {
        memcpy(v29, v27, v28);
        *(_QWORD *)(a2 + 24) += v30;
      }
    }
  }
  else
  {
    v26 = *(_QWORD *)(a2 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v26) <= 6 )
    {
      sub_16E7EE0(a2, "UNKNOWN", 7u);
    }
    else
    {
      *(_DWORD *)v26 = 1313558101;
      *(_WORD *)(v26 + 4) = 22351;
      *(_BYTE *)(v26 + 6) = 78;
      *(_QWORD *)(a2 + 24) += 7LL;
    }
  }
  if ( !v180 )
  {
    if ( **(_WORD **)(v9 + 16) == 1 && v183 > 1 )
    {
      v87 = *(_BYTE **)(a2 + 24);
      if ( *(_BYTE **)(a2 + 16) == v87 )
      {
        sub_16E7EE0(a2, " ", 1u);
      }
      else
      {
        *v87 = 32;
        ++*(_QWORD *)(a2 + 24);
      }
      v88 = 0;
      if ( v191 )
        v88 = sub_1E17F60(v9, 0, &v195, v191);
      v89 = 0;
      v90 = *(_QWORD *)(v9 + 32);
      if ( v193 && !*(_BYTE *)v90 && (*(_WORD *)(v90 + 2) & 0xFF0) != 0 && (*(_BYTE *)(v90 + 3) & 0x10) == 0 )
      {
        v179 = v88;
        v89 = sub_1E16AB0(v9, 0, v26, v88, v17, (_BYTE *)v18);
        v90 = *(_QWORD *)(v9 + 32);
        LODWORD(v88) = v179;
      }
      sub_1E32250(v90, a2, a3, v88, 1, v181, v193, v89, v188, v186);
      v91 = *(_QWORD *)(*(_QWORD *)(v9 + 32) + 64LL);
      if ( (v91 & 1) != 0 )
      {
        v129 = *(void **)(a2 + 24);
        if ( *(_QWORD *)(a2 + 16) - (_QWORD)v129 <= 0xCu )
        {
          sub_16E7EE0(a2, " [sideeffect]", 0xDu);
        }
        else
        {
          qmemcpy(v129, " [sideeffect]", 13);
          *(_QWORD *)(a2 + 24) += 13LL;
        }
      }
      if ( (v91 & 8) != 0 )
      {
        v130 = *(void **)(a2 + 24);
        if ( *(_QWORD *)(a2 + 16) - (_QWORD)v130 <= 9u )
        {
          sub_16E7EE0(a2, " [mayload]", 0xAu);
        }
        else
        {
          qmemcpy(v130, " [mayload]", 10);
          *(_QWORD *)(a2 + 24) += 10LL;
        }
      }
      if ( (v91 & 0x10) != 0 )
      {
        v127 = *(void **)(a2 + 24);
        if ( *(_QWORD *)(a2 + 16) - (_QWORD)v127 <= 0xAu )
        {
          sub_16E7EE0(a2, " [maystore]", 0xBu);
        }
        else
        {
          qmemcpy(v127, " [maystore]", 11);
          *(_QWORD *)(a2 + 24) += 11LL;
        }
      }
      if ( (v91 & 0x20) != 0 )
      {
        v128 = *(void **)(a2 + 24);
        if ( *(_QWORD *)(a2 + 16) - (_QWORD)v128 <= 0xEu )
        {
          sub_16E7EE0(a2, " [isconvergent]", 0xFu);
        }
        else
        {
          qmemcpy(v128, " [isconvergent]", 15);
          *(_QWORD *)(a2 + 24) += 15LL;
        }
      }
      if ( (v91 & 2) != 0 )
      {
        v126 = *(void **)(a2 + 24);
        if ( *(_QWORD *)(a2 + 16) - (_QWORD)v126 <= 0xCu )
        {
          sub_16E7EE0(a2, " [alignstack]", 0xDu);
        }
        else
        {
          qmemcpy(v126, " [alignstack]", 13);
          *(_QWORD *)(a2 + 24) += 13LL;
        }
      }
      if ( !(unsigned int)sub_1E16470(v9) )
      {
        v125 = *(void **)(a2 + 24);
        if ( *(_QWORD *)(a2 + 16) - (_QWORD)v125 <= 0xCu )
        {
          sub_16E7EE0(a2, " [attdialect]", 0xDu);
        }
        else
        {
          qmemcpy(v125, " [attdialect]", 13);
          *(_QWORD *)(a2 + 24) += 13LL;
        }
      }
      if ( (unsigned int)sub_1E16470(v9) == 1 )
      {
        v131 = *(void **)(a2 + 24);
        if ( *(_QWORD *)(a2 + 16) - (_QWORD)v131 <= 0xEu )
        {
          sub_16E7EE0(a2, " [inteldialect]", 0xFu);
        }
        else
        {
          qmemcpy(v131, " [inteldialect]", 15);
          *(_QWORD *)(a2 + 24) += 15LL;
        }
      }
      v177 = 2;
      v31 = 0;
      v16 = 2;
    }
    else
    {
      v177 = -1;
      v31 = 1;
    }
    v182 = *(_DWORD *)(v9 + 40);
    if ( v16 != v182 )
    {
      v175 = 0;
      while ( 1 )
      {
        v32 = 5LL * v16;
        v33 = *(_BYTE **)(a2 + 24);
        v34 = 40LL * v16;
        v35 = v34 + *(_QWORD *)(v9 + 32);
        if ( !(_BYTE)v31 )
        {
          if ( v33 == *(_BYTE **)(a2 + 16) )
          {
            sub_16E7EE0(a2, ",", 1u);
            v33 = *(_BYTE **)(a2 + 24);
          }
          else
          {
            *v33 = 44;
            v33 = (_BYTE *)(*(_QWORD *)(a2 + 24) + 1LL);
            *(_QWORD *)(a2 + 24) = v33;
          }
        }
        if ( *(_BYTE **)(a2 + 16) == v33 )
        {
          sub_16E7EE0(a2, " ", 1u);
        }
        else
        {
          *v33 = 32;
          ++*(_QWORD *)(a2 + 24);
        }
        v36 = **(_WORD **)(v9 + 16);
        if ( v36 == 12 )
        {
          if ( *(_BYTE *)v35 == 14 )
          {
            v53 = *(_QWORD *)(v35 + 24);
            if ( *(_BYTE *)v53 != 25 )
              goto LABEL_91;
            v31 = *(unsigned int *)(v53 + 8);
            src = *(_QWORD **)(v35 + 24);
            v54 = *(_QWORD *)(v53 + 8 * (1 - v31));
            if ( !v54 )
              goto LABEL_91;
            sub_161E970(v54);
            v55 = src;
            if ( !v31 )
              goto LABEL_91;
            v86 = *(_WORD **)(a2 + 24);
            if ( *(_QWORD *)(a2 + 16) - (_QWORD)v86 <= 1u )
            {
              v123 = sub_16E7EE0(a2, "!\"", 2u);
              v55 = src;
              v60 = v123;
            }
            else
            {
              v60 = a2;
              *v86 = 8737;
              *(_QWORD *)(a2 + 24) += 2LL;
            }
LABEL_108:
            v61 = v55[1LL - *((unsigned int *)v55 + 2)];
            if ( v61 )
            {
              v62 = sub_161E970(v61);
              v64 = *(char **)(v60 + 24);
              v65 = (void *)v62;
              v66 = *(_QWORD *)(v60 + 16);
              if ( v66 - (unsigned __int64)v64 >= v63 )
              {
                if ( v63 )
                {
                  srcd = (char *)v63;
                  memcpy(v64, v65, v63);
                  v66 = *(_QWORD *)(v60 + 16);
                  v64 = &srcd[*(_QWORD *)(v60 + 24)];
                  *(_QWORD *)(v60 + 24) = v64;
                }
                goto LABEL_112;
              }
              v60 = sub_16E7EE0(v60, (char *)v65, v63);
            }
            v64 = *(char **)(v60 + 24);
            v66 = *(_QWORD *)(v60 + 16);
LABEL_112:
            if ( (unsigned __int64)v64 >= v66 )
            {
              sub_16E7DE0(v60, 34);
            }
            else
            {
              *(_QWORD *)(v60 + 24) = v64 + 1;
              *v64 = 34;
            }
            goto LABEL_66;
          }
        }
        else if ( v36 == 13 && *(_BYTE *)v35 == 14 )
        {
          v57 = *(_QWORD *)(v35 + 24);
          if ( *(_BYTE *)v57 == 26 )
          {
            v31 = *(unsigned int *)(v57 + 8);
            srca = *(_QWORD **)(v35 + 24);
            v58 = *(_QWORD *)(v57 + 8 * (1 - v31));
            if ( v58 )
            {
              sub_161E970(v58);
              v55 = srca;
              if ( v31 )
              {
                v59 = *(_BYTE **)(a2 + 24);
                if ( *(_BYTE **)(a2 + 16) == v59 )
                {
                  v124 = sub_16E7EE0(a2, "\"", 1u);
                  v55 = srca;
                  v60 = v124;
                }
                else
                {
                  *v59 = 34;
                  v60 = a2;
                  ++*(_QWORD *)(a2 + 24);
                }
                goto LABEL_108;
              }
            }
          }
LABEL_91:
          if ( v191 )
            v37 = sub_1E17F60(v9, v16, &v195, v191);
          else
            v37 = 0;
          v38 = 0;
          if ( v193 )
          {
            v56 = *(_QWORD *)(v9 + 32) + v34;
            if ( !*(_BYTE *)v56 && (*(_WORD *)(v56 + 2) & 0xFF0) != 0 && (*(_BYTE *)(v56 + 3) & 0x10) == 0 )
            {
              srcc = v37;
              v38 = sub_1E16AB0(v9, v16, v31, v37, v32, (_BYTE *)v18);
              LODWORD(v37) = srcc;
            }
          }
LABEL_65:
          sub_1E32250(v35, a2, a3, v37, 1, v185, v193, v38, v188, v186);
LABEL_66:
          if ( ++v16 == v182 )
            goto LABEL_70;
          goto LABEL_67;
        }
        if ( v177 == v16 && *(_BYTE *)v35 == 1 )
        {
          v67 = *(_BYTE **)(a2 + 24);
          if ( (unsigned __int64)v67 >= *(_QWORD *)(a2 + 16) )
          {
            v68 = sub_16E7DE0(a2, 36);
          }
          else
          {
            v68 = a2;
            *(_QWORD *)(a2 + 24) = v67 + 1;
            *v67 = 36;
          }
          v69 = v175++;
          sub_16E7A90(v68, v69);
          v70 = *(_QWORD *)(v35 + 24);
          v71 = v70 & 7;
          switch ( v70 & 7 )
          {
            case 1LL:
              v118 = *(_QWORD **)(a2 + 24);
              if ( *(_QWORD *)(a2 + 16) - (_QWORD)v118 <= 7u )
              {
                srck = v70;
                sub_16E7EE0(a2, ":[reguse", 8u);
                v104 = *(_QWORD *)(a2 + 24);
                LODWORD(v70) = srck;
              }
              else
              {
                *v118 = 0x6573756765725B3ALL;
                v104 = *(_QWORD *)(a2 + 24) + 8LL;
                *(_QWORD *)(a2 + 24) = v104;
              }
              goto LABEL_251;
            case 2LL:
              v119 = *(_QWORD **)(a2 + 24);
              if ( *(_QWORD *)(a2 + 16) - (_QWORD)v119 <= 7u )
              {
                srcn = v70;
                sub_16E7EE0(a2, ":[regdef", 8u);
                v104 = *(_QWORD *)(a2 + 24);
                LODWORD(v70) = srcn;
              }
              else
              {
                *v119 = 0x6665646765725B3ALL;
                v104 = *(_QWORD *)(a2 + 24) + 8LL;
                *(_QWORD *)(a2 + 24) = v104;
              }
              goto LABEL_251;
            case 3LL:
              v117 = *(void **)(a2 + 24);
              if ( *(_QWORD *)(a2 + 16) - (_QWORD)v117 <= 0xAu )
              {
                srcl = v70;
                sub_16E7EE0(a2, ":[regdef-ec", 0xBu);
                v104 = *(_QWORD *)(a2 + 24);
                LODWORD(v70) = srcl;
              }
              else
              {
                qmemcpy(v117, ":[regdef-ec", 11);
                v104 = *(_QWORD *)(a2 + 24) + 11LL;
                *(_QWORD *)(a2 + 24) = v104;
              }
              goto LABEL_251;
            case 4LL:
              v110 = *(_QWORD *)(a2 + 24);
              if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v110) <= 8 )
              {
                srcj = v70;
                sub_16E7EE0(a2, ":[clobber", 9u);
                v104 = *(_QWORD *)(a2 + 24);
                LODWORD(v70) = srcj;
              }
              else
              {
                *(_BYTE *)(v110 + 8) = 114;
                *(_QWORD *)v110 = 0x6562626F6C635B3ALL;
                v104 = *(_QWORD *)(a2 + 24) + 9LL;
                *(_QWORD *)(a2 + 24) = v104;
              }
              goto LABEL_251;
            case 5LL:
              v106 = *(_QWORD *)(a2 + 24);
              if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v106) <= 4 )
              {
                srco = v70;
                sub_16E7EE0(a2, ":[imm", 5u);
                v104 = *(_QWORD *)(a2 + 24);
                LODWORD(v70) = srco;
              }
              else
              {
                *(_DWORD *)v106 = 1835621178;
                *(_BYTE *)(v106 + 4) = 109;
                v104 = *(_QWORD *)(a2 + 24) + 5LL;
                *(_QWORD *)(a2 + 24) = v104;
              }
              goto LABEL_244;
            case 6LL:
              v103 = *(_QWORD *)(a2 + 24);
              if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v103) <= 4 )
              {
                srcp = v70;
                sub_16E7EE0(a2, ":[mem", 5u);
                v104 = *(_QWORD *)(a2 + 24);
                LODWORD(v70) = srcp;
              }
              else
              {
                *(_DWORD *)v103 = 1701665594;
                *(_BYTE *)(v103 + 4) = 109;
                v104 = *(_QWORD *)(a2 + 24) + 5LL;
                *(_QWORD *)(a2 + 24) = v104;
              }
              goto LABEL_241;
            default:
              v121 = *(_DWORD **)(a2 + 24);
              if ( *(_QWORD *)(a2 + 16) - (_QWORD)v121 <= 3u )
              {
                srcm = v70;
                v133 = sub_16E7EE0(a2, ":[??", 4u);
                LODWORD(v70) = srcm;
                v122 = v133;
              }
              else
              {
                *v121 = 1061116730;
                v122 = a2;
                *(_QWORD *)(a2 + 24) += 4LL;
              }
              srch = v70;
              sub_16E7A90(v122, v71);
              LODWORD(v70) = srch;
              v104 = *(_QWORD *)(a2 + 24);
              v107 = srch;
              if ( v71 == 5 )
                goto LABEL_245;
              if ( v71 == 6 )
              {
LABEL_241:
                v105 = *(_QWORD *)(a2 + 16) - v104;
                switch ( WORD1(v70) & 0x7FFF )
                {
                  case 1:
                    if ( v105 <= 2 )
                    {
                      srcx = v70;
                      sub_16E7EE0(a2, ":es", 3u);
                      v104 = *(_QWORD *)(a2 + 24);
                      LODWORD(v70) = srcx;
                    }
                    else
                    {
                      *(_BYTE *)(v104 + 2) = 115;
                      *(_WORD *)v104 = 25914;
                      v104 = *(_QWORD *)(a2 + 24) + 3LL;
                      *(_QWORD *)(a2 + 24) = v104;
                    }
                    break;
                  case 2:
                    if ( v105 <= 1 )
                    {
                      srcbh = v70;
                      sub_16E7EE0(a2, ":i", 2u);
                      v104 = *(_QWORD *)(a2 + 24);
                      LODWORD(v70) = srcbh;
                    }
                    else
                    {
                      *(_WORD *)v104 = 26938;
                      v104 = *(_QWORD *)(a2 + 24) + 2LL;
                      *(_QWORD *)(a2 + 24) = v104;
                    }
                    break;
                  case 3:
                    if ( v105 <= 1 )
                    {
                      srcbf = v70;
                      sub_16E7EE0(a2, ":m", 2u);
                      v104 = *(_QWORD *)(a2 + 24);
                      LODWORD(v70) = srcbf;
                    }
                    else
                    {
                      *(_WORD *)v104 = 27962;
                      v104 = *(_QWORD *)(a2 + 24) + 2LL;
                      *(_QWORD *)(a2 + 24) = v104;
                    }
                    break;
                  case 4:
                    if ( v105 <= 1 )
                    {
                      srcbl = v70;
                      sub_16E7EE0(a2, ":o", 2u);
                      v104 = *(_QWORD *)(a2 + 24);
                      LODWORD(v70) = srcbl;
                    }
                    else
                    {
                      *(_WORD *)v104 = 28474;
                      v104 = *(_QWORD *)(a2 + 24) + 2LL;
                      *(_QWORD *)(a2 + 24) = v104;
                    }
                    break;
                  case 5:
                    if ( v105 <= 1 )
                    {
                      srcu = v70;
                      sub_16E7EE0(a2, ":v", 2u);
                      v104 = *(_QWORD *)(a2 + 24);
                      LODWORD(v70) = srcu;
                    }
                    else
                    {
                      *(_WORD *)v104 = 30266;
                      v104 = *(_QWORD *)(a2 + 24) + 2LL;
                      *(_QWORD *)(a2 + 24) = v104;
                    }
                    break;
                  case 6:
                    if ( v105 <= 1 )
                    {
                      srcbg = v70;
                      sub_16E7EE0(a2, ":Q", 2u);
                      v104 = *(_QWORD *)(a2 + 24);
                      LODWORD(v70) = srcbg;
                    }
                    else
                    {
                      *(_WORD *)v104 = 20794;
                      v104 = *(_QWORD *)(a2 + 24) + 2LL;
                      *(_QWORD *)(a2 + 24) = v104;
                    }
                    break;
                  case 7:
                    if ( v105 <= 1 )
                    {
                      srcw = v70;
                      sub_16E7EE0(a2, ":R", 2u);
                      v104 = *(_QWORD *)(a2 + 24);
                      LODWORD(v70) = srcw;
                    }
                    else
                    {
                      *(_WORD *)v104 = 21050;
                      v104 = *(_QWORD *)(a2 + 24) + 2LL;
                      *(_QWORD *)(a2 + 24) = v104;
                    }
                    break;
                  case 8:
                    if ( v105 <= 1 )
                    {
                      srcz = v70;
                      sub_16E7EE0(a2, ":S", 2u);
                      v104 = *(_QWORD *)(a2 + 24);
                      LODWORD(v70) = srcz;
                    }
                    else
                    {
                      *(_WORD *)v104 = 21306;
                      v104 = *(_QWORD *)(a2 + 24) + 2LL;
                      *(_QWORD *)(a2 + 24) = v104;
                    }
                    break;
                  case 9:
                    if ( v105 <= 1 )
                    {
                      srcbd = v70;
                      sub_16E7EE0(a2, ":T", 2u);
                      v104 = *(_QWORD *)(a2 + 24);
                      LODWORD(v70) = srcbd;
                    }
                    else
                    {
                      v18 = 21562;
                      *(_WORD *)v104 = 21562;
                      v104 = *(_QWORD *)(a2 + 24) + 2LL;
                      *(_QWORD *)(a2 + 24) = v104;
                    }
                    break;
                  case 0xA:
                    if ( v105 <= 2 )
                    {
                      srcbk = v70;
                      sub_16E7EE0(a2, ":Um", 3u);
                      v104 = *(_QWORD *)(a2 + 24);
                      LODWORD(v70) = srcbk;
                    }
                    else
                    {
                      *(_BYTE *)(v104 + 2) = 109;
                      *(_WORD *)v104 = 21818;
                      v104 = *(_QWORD *)(a2 + 24) + 3LL;
                      *(_QWORD *)(a2 + 24) = v104;
                    }
                    break;
                  case 0xB:
                    if ( v105 <= 2 )
                    {
                      srct = v70;
                      sub_16E7EE0(a2, ":Un", 3u);
                      v104 = *(_QWORD *)(a2 + 24);
                      LODWORD(v70) = srct;
                    }
                    else
                    {
                      *(_BYTE *)(v104 + 2) = 110;
                      *(_WORD *)v104 = 21818;
                      v104 = *(_QWORD *)(a2 + 24) + 3LL;
                      *(_QWORD *)(a2 + 24) = v104;
                    }
                    break;
                  case 0xC:
                    if ( v105 <= 2 )
                    {
                      srcbj = v70;
                      sub_16E7EE0(a2, ":Uq", 3u);
                      v104 = *(_QWORD *)(a2 + 24);
                      LODWORD(v70) = srcbj;
                    }
                    else
                    {
                      *(_BYTE *)(v104 + 2) = 113;
                      *(_WORD *)v104 = 21818;
                      v104 = *(_QWORD *)(a2 + 24) + 3LL;
                      *(_QWORD *)(a2 + 24) = v104;
                    }
                    break;
                  case 0xD:
                    if ( v105 <= 2 )
                    {
                      srcbb = v70;
                      sub_16E7EE0(a2, ":Us", 3u);
                      v104 = *(_QWORD *)(a2 + 24);
                      LODWORD(v70) = srcbb;
                    }
                    else
                    {
                      *(_BYTE *)(v104 + 2) = 115;
                      *(_WORD *)v104 = 21818;
                      v104 = *(_QWORD *)(a2 + 24) + 3LL;
                      *(_QWORD *)(a2 + 24) = v104;
                    }
                    break;
                  case 0xE:
                    if ( v105 <= 2 )
                    {
                      srcbi = v70;
                      sub_16E7EE0(a2, ":Ut", 3u);
                      v104 = *(_QWORD *)(a2 + 24);
                      LODWORD(v70) = srcbi;
                    }
                    else
                    {
                      *(_BYTE *)(v104 + 2) = 116;
                      *(_WORD *)v104 = 21818;
                      v104 = *(_QWORD *)(a2 + 24) + 3LL;
                      *(_QWORD *)(a2 + 24) = v104;
                    }
                    break;
                  case 0xF:
                    if ( v105 <= 2 )
                    {
                      srcs = v70;
                      sub_16E7EE0(a2, ":Uv", 3u);
                      v104 = *(_QWORD *)(a2 + 24);
                      LODWORD(v70) = srcs;
                    }
                    else
                    {
                      *(_BYTE *)(v104 + 2) = 118;
                      *(_WORD *)v104 = 21818;
                      v104 = *(_QWORD *)(a2 + 24) + 3LL;
                      *(_QWORD *)(a2 + 24) = v104;
                    }
                    break;
                  case 0x10:
                    if ( v105 <= 2 )
                    {
                      srcy = v70;
                      sub_16E7EE0(a2, ":Uy", 3u);
                      v104 = *(_QWORD *)(a2 + 24);
                      LODWORD(v70) = srcy;
                    }
                    else
                    {
                      *(_BYTE *)(v104 + 2) = 121;
                      *(_WORD *)v104 = 21818;
                      v104 = *(_QWORD *)(a2 + 24) + 3LL;
                      *(_QWORD *)(a2 + 24) = v104;
                    }
                    break;
                  case 0x11:
                    if ( v105 <= 1 )
                    {
                      srcbc = v70;
                      sub_16E7EE0(a2, ":X", 2u);
                      v104 = *(_QWORD *)(a2 + 24);
                      LODWORD(v70) = srcbc;
                    }
                    else
                    {
                      *(_WORD *)v104 = 22586;
                      v104 = *(_QWORD *)(a2 + 24) + 2LL;
                      *(_QWORD *)(a2 + 24) = v104;
                    }
                    break;
                  case 0x12:
                    if ( v105 <= 1 )
                    {
                      srcba = v70;
                      sub_16E7EE0(a2, ":Z", 2u);
                      v104 = *(_QWORD *)(a2 + 24);
                      LODWORD(v70) = srcba;
                    }
                    else
                    {
                      v18 = 23098;
                      *(_WORD *)v104 = 23098;
                      v104 = *(_QWORD *)(a2 + 24) + 2LL;
                      *(_QWORD *)(a2 + 24) = v104;
                    }
                    break;
                  case 0x13:
                    if ( v105 <= 2 )
                    {
                      srcbe = v70;
                      sub_16E7EE0(a2, ":ZC", 3u);
                      v104 = *(_QWORD *)(a2 + 24);
                      LODWORD(v70) = srcbe;
                    }
                    else
                    {
                      *(_BYTE *)(v104 + 2) = 67;
                      *(_WORD *)v104 = 23098;
                      v104 = *(_QWORD *)(a2 + 24) + 3LL;
                      *(_QWORD *)(a2 + 24) = v104;
                    }
                    break;
                  case 0x14:
                    if ( v105 <= 2 )
                    {
                      srcr = v70;
                      sub_16E7EE0(a2, ":Zy", 3u);
                      v104 = *(_QWORD *)(a2 + 24);
                      LODWORD(v70) = srcr;
                    }
                    else
                    {
                      *(_BYTE *)(v104 + 2) = 121;
                      *(_WORD *)v104 = 23098;
                      v104 = *(_QWORD *)(a2 + 24) + 3LL;
                      *(_QWORD *)(a2 + 24) = v104;
                    }
                    break;
                  default:
                    if ( v105 <= 1 )
                    {
                      srcv = v70;
                      sub_16E7EE0(a2, ":?", 2u);
                      v104 = *(_QWORD *)(a2 + 24);
                      LODWORD(v70) = srcv;
                    }
                    else
                    {
                      *(_WORD *)v104 = 16186;
                      v104 = *(_QWORD *)(a2 + 24) + 2LL;
                      *(_QWORD *)(a2 + 24) = v104;
                    }
                    break;
                }
LABEL_244:
                v107 = v70;
LABEL_245:
                if ( v107 < 0 )
                {
LABEL_246:
                  v108 = WORD1(v70) & 0x7FFF;
                  if ( *(_QWORD *)(a2 + 16) - v104 <= 8 )
                  {
                    srcg = v70;
                    v120 = sub_16E7EE0(a2, " tiedto:$", 9u);
                    LOWORD(v70) = srcg;
                    v109 = v120;
                  }
                  else
                  {
                    *(_BYTE *)(v104 + 8) = 36;
                    v109 = a2;
                    *(_QWORD *)v104 = 0x3A6F746465697420LL;
                    *(_QWORD *)(a2 + 24) += 9LL;
                  }
                  goto LABEL_248;
                }
              }
              else
              {
LABEL_251:
                if ( (int)v70 < 0 )
                  goto LABEL_246;
                if ( WORD1(v70) )
                {
                  v111 = *(_QWORD *)(a2 + 16);
                  v108 = WORD1(v70) - 1;
                  if ( !v188 )
                  {
                    if ( v111 - v104 <= 2 )
                    {
                      srcq = v70;
                      v134 = sub_16E7EE0(a2, ":RC", 3u);
                      LOWORD(v70) = srcq;
                      v109 = v134;
                    }
                    else
                    {
                      *(_BYTE *)(v104 + 2) = 67;
                      v109 = a2;
                      *(_WORD *)v104 = 21050;
                      *(_QWORD *)(a2 + 24) += 3LL;
                    }
LABEL_248:
                    srce = v70;
                    sub_16E7A90(v109, v108);
                    v104 = *(_QWORD *)(a2 + 24);
                    LOWORD(v70) = srce;
                    goto LABEL_261;
                  }
                  if ( v111 <= v104 )
                  {
                    srci = v70;
                    v132 = sub_16E7DE0(a2, 58);
                    LOWORD(v70) = srci;
                    v112 = v132;
                  }
                  else
                  {
                    v112 = a2;
                    *(_QWORD *)(a2 + 24) = v104 + 1;
                    *(_BYTE *)v104 = 58;
                  }
                  v113 = (char *)(*(_QWORD *)(v188 + 80)
                                + *(unsigned int *)(**(_QWORD **)(*(_QWORD *)(v188 + 256) + 8LL * v108) + 16LL));
                  if ( v113 )
                  {
                    v135 = v70;
                    v114 = strlen(v113);
                    v115 = *(void **)(v112 + 24);
                    v116 = v114;
                    LOWORD(v70) = v135;
                    if ( v114 > *(_QWORD *)(v112 + 16) - (_QWORD)v115 )
                    {
                      sub_16E7EE0(v112, v113, v114);
                      LOWORD(v70) = v135;
                    }
                    else if ( v114 )
                    {
                      memcpy(v115, v113, v114);
                      *(_QWORD *)(v112 + 24) += v116;
                      LOWORD(v70) = v135;
                    }
                  }
                  v104 = *(_QWORD *)(a2 + 24);
                }
              }
LABEL_261:
              if ( *(_QWORD *)(a2 + 16) <= v104 )
              {
                srcf = v70;
                sub_16E7DE0(a2, 93);
                LOWORD(v70) = srcf;
              }
              else
              {
                *(_QWORD *)(a2 + 24) = v104 + 1;
                *(_BYTE *)v104 = 93;
              }
              v177 += ((unsigned __int16)v70 >> 3) + 1;
              break;
          }
          goto LABEL_66;
        }
        if ( v191 )
          v37 = sub_1E17F60(v9, v16, &v195, v191);
        else
          v37 = 0;
        v38 = 0;
        if ( v193 )
        {
          v39 = v34 + *(_QWORD *)(v9 + 32);
          if ( !*(_BYTE *)v39 && (*(_WORD *)(v39 + 2) & 0xFF0) != 0 && (*(_BYTE *)(v39 + 3) & 0x10) == 0 )
          {
            srcb = (void *)v37;
            v38 = sub_1E16AB0(v9, v16, v31, v37, v39, (_BYTE *)v18);
            v37 = (__int64)srcb;
          }
        }
        if ( *(_BYTE *)v35 != 1 )
          goto LABEL_65;
        v40 = **(_WORD **)(v9 + 16);
        switch ( v40 )
        {
          case 7:
            if ( v16 != 2 )
              goto LABEL_65;
            break;
          case 8:
            if ( v16 != 3 )
              goto LABEL_65;
            break;
          case 14:
            if ( v16 <= 1 || (v16 & 1) != 0 )
              goto LABEL_65;
            break;
          default:
            if ( v40 != 10 || v16 != 3 )
              goto LABEL_65;
            break;
        }
        sub_1E31810(a2, *(_QWORD *)(v35 + 24), v188, v37);
        if ( ++v16 == v182 )
        {
LABEL_70:
          LOBYTE(v31) = 0;
          break;
        }
LABEL_67:
        v31 = v184;
      }
    }
    if ( v176 )
    {
      if ( !*(_BYTE *)(v9 + 49) )
        goto LABEL_133;
    }
    else
    {
      v41 = (_QWORD *)(v9 + 64);
      if ( *(_QWORD *)(v9 + 64) )
      {
        if ( !(_BYTE)v31 )
        {
          v92 = *(_BYTE **)(a2 + 24);
          if ( (unsigned __int64)v92 >= *(_QWORD *)(a2 + 16) )
          {
            sub_16E7DE0(a2, 44);
          }
          else
          {
            *(_QWORD *)(a2 + 24) = v92 + 1;
            *v92 = 44;
          }
        }
        v42 = *(__m128i **)(a2 + 24);
        if ( *(_QWORD *)(a2 + 16) - (_QWORD)v42 <= 0xFu )
        {
          sub_16E7EE0(a2, " debug-location ", 0x10u);
        }
        else
        {
          *v42 = _mm_load_si128((const __m128i *)&xmmword_42EB2D0);
          *(_QWORD *)(a2 + 24) += 16LL;
        }
        v43 = (unsigned __int8 *)sub_15C70A0(v9 + 64);
        sub_1556260(v43, a2, a3, 0);
        if ( !*(_BYTE *)(v9 + 49) )
          goto LABEL_125;
      }
      else if ( !*(_BYTE *)(v9 + 49) )
      {
        v184 = 0;
LABEL_131:
        if ( **(_WORD **)(v9 + 16) == 12 )
        {
          v93 = 40LL * (v183 - 2);
          v94 = v93 + *(_QWORD *)(v9 + 32);
          if ( *(_BYTE *)v94 == 14 )
          {
            v95 = *(_BYTE **)(a2 + 24);
            if ( !v184 )
            {
              if ( *(_BYTE **)(a2 + 16) == v95 )
              {
                sub_16E7EE0(a2, ";", 1u);
                v95 = *(_BYTE **)(a2 + 24);
              }
              else
              {
                *v95 = 59;
                v95 = (_BYTE *)(*(_QWORD *)(a2 + 24) + 1LL);
                *(_QWORD *)(a2 + 24) = v95;
              }
              v94 = v93 + *(_QWORD *)(v9 + 32);
            }
            v96 = *(_QWORD *)(v94 + 24);
            if ( *(_QWORD *)(a2 + 16) - (_QWORD)v95 <= 8u )
            {
              v97 = sub_16E7EE0(a2, " line no:", 9u);
            }
            else
            {
              v95[8] = 58;
              v97 = a2;
              *(_QWORD *)v95 = 0x6F6E20656E696C20LL;
              *(_QWORD *)(a2 + 24) += 9LL;
            }
            sub_16E7A90(v97, *(unsigned int *)(v96 + 24));
            v98 = sub_15C70A0((__int64)v41);
            if ( *(_DWORD *)(v98 + 8) == 2 )
            {
              v99 = *(_QWORD *)(v98 - 8);
              if ( v99 )
              {
                sub_15C7080(v196, v99);
                if ( v196[0] )
                  sub_161E7C0((__int64)v196, v196[0]);
              }
            }
            if ( **(_WORD **)(v9 + 16) == 12 )
            {
              v100 = *(_BYTE **)(v9 + 32);
              if ( !*v100 && v100[40] == 1 )
              {
                v101 = *(_QWORD *)(a2 + 24);
                if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v101) <= 8 )
                {
                  sub_16E7EE0(a2, " indirect", 9u);
                }
                else
                {
                  *(_BYTE *)(v101 + 8) = 116;
                  *(_QWORD *)v101 = 0x63657269646E6920LL;
                  *(_QWORD *)(a2 + 24) += 9LL;
                }
              }
            }
          }
        }
        if ( a7 )
        {
          v85 = *(_BYTE **)(a2 + 24);
          if ( (unsigned __int64)v85 >= *(_QWORD *)(a2 + 16) )
          {
            sub_16E7DE0(a2, 10);
          }
          else
          {
            *(_QWORD *)(a2 + 24) = v85 + 1;
            *v85 = 10;
          }
        }
        goto LABEL_133;
      }
    }
    v196[1] = 0;
    v196[0] = (__int64)v197;
    v44 = *(_QWORD *)(v9 + 24);
    if ( v44 && (v45 = *(__int64 **)(v44 + 56)) != 0 )
    {
      v46 = v45[7];
      v192 = 0;
      LODWORD(v47) = sub_15E0530(*v45);
    }
    else
    {
      v102 = (__int64 *)sub_22077B0(8);
      v47 = v102;
      if ( v102 )
        sub_1602D10(v102);
      v192 = v47;
      LODWORD(v46) = 0;
    }
    v48 = *(_DWORD **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v48 <= 3u )
    {
      sub_16E7EE0(a2, " :: ", 4u);
    }
    else
    {
      *v48 = 540686880;
      *(_QWORD *)(a2 + 24) += 4LL;
    }
    v49 = *(__int64 **)(v9 + 56);
    v194 = &v49[*(unsigned __int8 *)(v9 + 49)];
    if ( v49 != v194 )
    {
      v50 = *v49;
      v189 = v9;
      v51 = v49 + 1;
      while ( 1 )
      {
        sub_1E343B0(v50, a2, a3, (unsigned int)v196, (_DWORD)v47, v46, a8);
        if ( v194 == v51 )
          break;
        v52 = *(_WORD **)(a2 + 24);
        v50 = *v51;
        if ( *(_QWORD *)(a2 + 16) - (_QWORD)v52 <= 1u )
        {
          v187 = *v51++;
          sub_16E7EE0(a2, ", ", 2u);
          LODWORD(v50) = v187;
        }
        else
        {
          ++v51;
          *v52 = 8236;
          *(_QWORD *)(a2 + 24) += 2LL;
        }
      }
      v9 = v189;
    }
    if ( v192 )
    {
      sub_16025D0(v192);
      j_j___libc_free_0(v192, 8);
    }
    if ( (_BYTE *)v196[0] != v197 )
      _libc_free(v196[0]);
    if ( !v176 )
    {
      v41 = (_QWORD *)(v9 + 64);
LABEL_125:
      if ( *(_QWORD *)(v9 + 64) )
      {
        v72 = *(_BYTE **)(a2 + 24);
        if ( (unsigned __int64)v72 >= *(_QWORD *)(a2 + 16) )
        {
          sub_16E7DE0(a2, 59);
        }
        else
        {
          *(_QWORD *)(a2 + 24) = v72 + 1;
          *v72 = 59;
        }
        v73 = *(_BYTE **)(a2 + 24);
        if ( (unsigned __int64)v73 >= *(_QWORD *)(a2 + 16) )
        {
          sub_16E7DE0(a2, 32);
        }
        else
        {
          *(_QWORD *)(a2 + 24) = v73 + 1;
          *v73 = 32;
        }
        sub_15C7170(v41, a2);
        v184 = 1;
      }
      goto LABEL_131;
    }
  }
LABEL_133:
  v74 = v195;
  if ( (v195 & 1) == 0 && v195 )
  {
    _libc_free(*(_QWORD *)v195);
    j_j___libc_free_0(v74, 24);
  }
}
