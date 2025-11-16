// Function: sub_2B26DC0
// Address: 0x2b26dc0
//
void __fastcall sub_2B26DC0(
        __int64 a1,
        __int64 a2,
        char *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        _QWORD *a8,
        int a9,
        int **a10)
{
  int v12; // esi
  int *v13; // rdx
  signed int v14; // esi
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rsi
  int v19; // edx
  int v20; // ecx
  _DWORD *v21; // r11
  unsigned int v22; // eax
  __int64 v23; // rax
  char v24; // dl
  char v25; // r15
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rax
  unsigned int v29; // r13d
  __int64 v30; // r9
  int *v31; // rcx
  _DWORD *v32; // r10
  char v33; // dl
  char v34; // r11
  __int64 v35; // rax
  int *v36; // rdx
  int v37; // esi
  unsigned int *v38; // rax
  int v39; // esi
  int v40; // esi
  unsigned int v41; // edx
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // r9
  __int64 v45; // rdx
  __int64 v46; // r8
  unsigned __int64 *v47; // rdx
  unsigned __int64 v48; // rax
  __int64 v49; // r13
  __int64 v50; // rax
  unsigned int v51; // ebx
  _DWORD *v52; // r11
  __int64 v53; // rdi
  int *v54; // r15
  _DWORD *v55; // r10
  char v56; // dl
  char v57; // r9
  __int64 v58; // rax
  int v59; // edx
  int *v60; // rsi
  unsigned int *v61; // rax
  int v62; // esi
  __int64 v63; // rdi
  int v64; // eax
  __int64 v65; // rax
  __int64 v66; // rax
  __int64 v67; // rdx
  int *v68; // rdx
  unsigned __int64 v69; // rax
  unsigned __int64 v70; // r12
  unsigned __int64 v71; // r12
  unsigned int v72; // esi
  unsigned int v73; // edi
  _QWORD *v74; // rax
  __int64 v75; // rax
  char v76; // dl
  unsigned int v77; // r13d
  __int64 v78; // r10
  __int64 v79; // rax
  int v80; // edx
  int *v81; // rdi
  int v82; // esi
  unsigned __int64 v83; // r15
  int *v84; // rcx
  int v85; // esi
  __int64 v86; // rax
  __int64 v87; // rax
  __int64 v88; // rdx
  int *v89; // rdx
  unsigned __int64 v90; // rax
  int *v91; // r13
  int v92; // ecx
  unsigned int v93; // edx
  int v94; // esi
  __int64 v95; // rdi
  __int64 v96; // r8
  unsigned __int64 v97; // rax
  int v98; // edx
  unsigned int *v99; // rdi
  unsigned __int64 v100; // r15
  int v101; // r8d
  int *v102; // rcx
  __int64 v103; // rax
  __int64 v104; // rax
  __int64 v105; // rdx
  __int64 v106; // rcx
  char v107; // dl
  int *v108; // rcx
  _DWORD *v109; // rax
  unsigned int v110; // ebx
  int v111; // esi
  int v112; // edx
  __int64 v113; // r12
  char v114; // al
  unsigned int *v115; // r8
  unsigned int v116; // esi
  __int64 v117; // rax
  int v118; // edx
  int v119; // r8d
  __int64 v120; // rcx
  int *v121; // rdx
  unsigned __int64 v122; // rax
  bool v123; // cc
  __int64 v124; // [rsp-8h] [rbp-E8h]
  __int64 v125; // [rsp+8h] [rbp-D8h]
  __int64 v126; // [rsp+18h] [rbp-C8h]
  int *v127; // [rsp+18h] [rbp-C8h]
  __int64 v128; // [rsp+18h] [rbp-C8h]
  char v129; // [rsp+18h] [rbp-C8h]
  int *v130; // [rsp+18h] [rbp-C8h]
  int *v131; // [rsp+18h] [rbp-C8h]
  _DWORD *v133; // [rsp+28h] [rbp-B8h]
  unsigned __int64 *v134; // [rsp+48h] [rbp-98h] BYREF
  unsigned __int64 v135; // [rsp+50h] [rbp-90h] BYREF
  unsigned __int64 *v136; // [rsp+58h] [rbp-88h] BYREF
  unsigned __int64 v137; // [rsp+60h] [rbp-80h] BYREF
  __int64 v138; // [rsp+68h] [rbp-78h]
  int *v139; // [rsp+70h] [rbp-70h] BYREF
  __int64 v140; // [rsp+78h] [rbp-68h]
  _BYTE v141[96]; // [rsp+80h] [rbp-60h] BYREF

  v139 = (int *)v141;
  v12 = *(_DWORD *)(a1 + 16);
  v140 = 0xC00000000LL;
  if ( v12 )
  {
    sub_2B0D670((__int64)&v139, a1 + 8, (__int64)a3, a4, a5, a6);
    v15 = (unsigned int)v140;
    v13 = v139;
    v14 = v140;
  }
  else
  {
    v13 = (int *)v141;
    v14 = 0;
    v15 = 0;
  }
  sub_2B23C00((__int64 *)&v134, v14, (__int64)v13, v15, 2);
  sub_2B25A00(&v135, a3, (unsigned __int64 *)&v134);
  if ( (v135 & 1) != 0 )
  {
    if ( (~(-1LL << (v135 >> 58)) & (v135 >> 1)) == (1LL << (v135 >> 58)) - 1 )
      goto LABEL_5;
LABEL_71:
    v75 = sub_2B203A0(a8, *(_QWORD *)a1, v139, (unsigned int)v140, v16, v17);
    v129 = v76;
    v133 = (_DWORD *)v75;
    v125 = v75;
    sub_2B25530(&v137, (__int64)a3, (unsigned __int64 *)&v134);
    v77 = v140;
    v78 = v125;
    if ( (_DWORD)v140 )
    {
      v79 = 0;
      do
      {
        while ( 1 )
        {
          v80 = v79;
          v81 = &v139[v79];
          v82 = *v81;
          if ( *v81 != -1 )
            break;
          if ( (v137 & 1) != 0 )
            v83 = (~(-1LL << (v137 >> 58)) & (v137 >> 1)) >> v79;
          else
            v83 = *(_QWORD *)(*(_QWORD *)v137 + 8LL * ((unsigned int)v79 >> 6)) >> v79;
          if ( (v83 & 1) == 0 )
            v82 = v79;
          ++v79;
          *v81 = v82;
          if ( v77 == v79 )
            goto LABEL_82;
        }
        if ( !v129 )
          v80 = *v81;
        ++v79;
        *v81 = v77 + v80;
      }
      while ( v77 != v79 );
LABEL_82:
      v78 = v125;
      v77 = v140;
    }
    v84 = v139;
    v85 = **a10;
    if ( !v85 )
    {
      **a10 = v77;
      v85 = **a10;
    }
    v130 = v84;
    v86 = sub_2B08680(*(_QWORD *)(**(_QWORD **)v78 + 8LL), v85);
    v87 = sub_2B097B0(*((__int64 **)a10[1] + 412), 6, v86, v130, v77, 0, 0, 0);
    v46 = v88;
    v89 = a10[2];
    if ( (_DWORD)v46 == 1 )
      v89[2] = 1;
    if ( __OFADD__(*(_QWORD *)v89, v87) )
    {
      v123 = v87 <= 0;
      v90 = 0x7FFFFFFFFFFFFFFFLL;
      if ( v123 )
        v90 = 0x8000000000000000LL;
    }
    else
    {
      v90 = *(_QWORD *)v89 + v87;
    }
    *(_QWORD *)v89 = v90;
    **a10 = v77;
    v91 = (int *)v137;
    if ( (v137 & 1) == 0 && v137 )
    {
      if ( *(_QWORD *)v137 != v137 + 16 )
        _libc_free(*(_QWORD *)v137);
      j_j___libc_free_0((unsigned __int64)v91);
    }
    v49 = a1 + 72;
    goto LABEL_34;
  }
  v72 = *(_DWORD *)(v135 + 64);
  v73 = v72 >> 6;
  if ( v72 >> 6 )
  {
    v74 = *(_QWORD **)v135;
    while ( *v74 == -1 )
    {
      if ( (_QWORD *)(*(_QWORD *)v135 + 8LL * (v73 - 1) + 8) == ++v74 )
        goto LABEL_107;
    }
    goto LABEL_71;
  }
LABEL_107:
  v94 = v72 & 0x3F;
  if ( v94 && *(_QWORD *)(*(_QWORD *)v135 + 8LL * v73) != (1LL << v94) - 1 )
    goto LABEL_71;
LABEL_5:
  v18 = *(_QWORD *)a1;
  if ( a2 == 1 )
  {
    sub_2B203A0(a8, v18, v139, (unsigned int)v140, (unsigned int)v140, (__int64)v139);
    if ( !v107 )
    {
      v108 = *a10;
      v109 = *(_DWORD **)a1;
      v137 = (unsigned __int64)v139;
      v110 = v140;
      v138 = (unsigned int)v140;
      v111 = *v108;
      if ( !*v108 )
      {
        v112 = v109[30];
        if ( !v112 )
          v112 = v109[2];
        *v108 = v112;
        v111 = **a10;
      }
      v113 = sub_2B08680(*(_QWORD *)(**(_QWORD **)v109 + 8LL), v111);
      v114 = sub_B4ED80((int *)v137, v138, **a10);
      v115 = (unsigned int *)*a10;
      if ( !v114 )
      {
        v116 = *v115;
        v136 = &v137;
        if ( !sub_2B09200(&v136, v116) )
        {
          v117 = sub_DFBC30(*((__int64 **)a10[1] + 412), 7, v113, v137, v138, 0, 0, 0, 0, 0, 0);
          v119 = v118;
          v120 = v117;
          v121 = a10[2];
          if ( v119 == 1 )
            v121[2] = 1;
          v122 = *(_QWORD *)v121 + v117;
          if ( __OFADD__(*(_QWORD *)v121, v120) )
          {
            v122 = 0x7FFFFFFFFFFFFFFFLL;
            if ( v120 <= 0 )
              v122 = 0x8000000000000000LL;
          }
          *(_QWORD *)v121 = v122;
          v115 = (unsigned int *)*a10;
        }
      }
      *v115 = v110;
    }
    goto LABEL_54;
  }
  v19 = *(_DWORD *)(v18 + 120);
  v20 = v19;
  if ( !v19 )
    v20 = *(_DWORD *)(v18 + 8);
  v21 = *(_DWORD **)(a1 + 72);
  v22 = v21[30];
  v133 = v21;
  if ( !v22 )
  {
    if ( v21[2] != v20 )
      goto LABEL_10;
LABEL_111:
    v95 = *(_QWORD *)(a1 + 80);
    if ( (_DWORD)v140 )
    {
      v96 = 4LL * (unsigned int)v140;
      v97 = 0;
      do
      {
        v98 = *(_DWORD *)(v95 + v97);
        if ( v98 != -1 )
          v139[v97 / 4] = v20 + v98;
        v97 += 4LL;
      }
      while ( v96 != v97 );
      v99 = (unsigned int *)*a10;
      v100 = (unsigned int)v140;
      v18 = *(_QWORD *)a1;
      v101 = **a10;
      v102 = v139;
      v133 = *(_DWORD **)(a1 + 72);
      v29 = v140;
      if ( v101 )
        goto LABEL_125;
      if ( !v18 )
        goto LABEL_124;
      v19 = *(_DWORD *)(v18 + 120);
      v22 = *(_DWORD *)(*(_QWORD *)(a1 + 72) + 120LL);
    }
    else
    {
      v99 = (unsigned int *)*a10;
      v102 = v139;
      v100 = 0;
      v29 = **a10;
      if ( v29 )
      {
        v101 = **a10;
        v29 = 0;
        goto LABEL_125;
      }
    }
    if ( !v19 )
      v19 = *(_DWORD *)(v18 + 8);
    if ( !v22 )
      v22 = v133[2];
    if ( v22 == v19 )
    {
      *v99 = v22;
      v101 = **a10;
      goto LABEL_125;
    }
LABEL_124:
    *v99 = v29;
    v101 = **a10;
LABEL_125:
    v131 = v102;
    v103 = sub_2B08680(*(_QWORD *)(**(_QWORD **)v133 + 8LL), v101);
    v104 = sub_2B097B0(*((__int64 **)a10[1] + 412), 6, v103, v131, v100, 0, 0, 0);
    v46 = v105;
    v106 = v104;
    v47 = (unsigned __int64 *)a10[2];
    if ( (_DWORD)v46 == 1 )
      *((_DWORD *)v47 + 2) = 1;
    v48 = *v47 + v104;
    if ( __OFADD__(*v47, v106) )
    {
      v48 = 0x7FFFFFFFFFFFFFFFLL;
      if ( v106 <= 0 )
        v48 = 0x8000000000000000LL;
    }
    goto LABEL_33;
  }
  if ( v22 == v20 )
    goto LABEL_111;
LABEL_10:
  v23 = sub_2B203A0(a8, v18, v139, (unsigned int)v140, (unsigned int)v140, (__int64)v139);
  v25 = v24;
  v126 = v23;
  v28 = sub_2B203A0(a8, *(_QWORD *)(a1 + 72), *(int **)(a1 + 80), *(unsigned int *)(a1 + 88), v26, v27);
  v29 = v140;
  v30 = *(_QWORD *)(a1 + 80);
  v133 = (_DWORD *)v28;
  v31 = v139;
  v32 = (_DWORD *)v28;
  v34 = v33;
  if ( !(_DWORD)v140 )
    goto LABEL_21;
  v35 = 0;
  do
  {
    while ( 1 )
    {
      v36 = &v31[v35];
      v37 = v35;
      if ( *v36 != -1 )
      {
        if ( v25 )
        {
          *v36 = v35;
          v31 = v139;
        }
        goto LABEL_14;
      }
      if ( *(_DWORD *)(v30 + 4 * v35) != -1 )
        break;
LABEL_14:
      if ( v29 == ++v35 )
        goto LABEL_20;
    }
    if ( !v34 )
      v37 = *(_DWORD *)(v30 + 4 * v35);
    ++v35;
    *v36 = v29 + v37;
    v31 = v139;
  }
  while ( v29 != v35 );
LABEL_20:
  v29 = v140;
LABEL_21:
  v38 = (unsigned int *)*a10;
  v39 = **a10;
  if ( !v39 )
  {
    if ( !v126 )
      goto LABEL_28;
    v40 = *(_DWORD *)(v126 + 120);
    if ( !v40 )
      v40 = *(_DWORD *)(v126 + 8);
    v41 = v32[30];
    if ( !v41 )
      v41 = v32[2];
    if ( v41 == v40 )
    {
      *v38 = v41;
      v39 = **a10;
    }
    else
    {
LABEL_28:
      *v38 = v29;
      v39 = **a10;
    }
  }
  v127 = v31;
  v42 = sub_2B08680(*(_QWORD *)(**(_QWORD **)v32 + 8LL), v39);
  v43 = sub_2B097B0(*((__int64 **)a10[1] + 412), 6, v42, v127, v29, 0, 0, 0);
  v44 = v124;
  v46 = v45;
  v47 = (unsigned __int64 *)a10[2];
  if ( (_DWORD)v46 == 1 )
    *((_DWORD *)v47 + 2) = 1;
  if ( __OFADD__(*v47, v43) )
  {
    v123 = v43 <= 0;
    v48 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v123 )
      v48 = 0x8000000000000000LL;
  }
  else
  {
    v48 = *v47 + v43;
  }
LABEL_33:
  *v47 = v48;
  **a10 = v29;
  v49 = a1 + 144;
LABEL_34:
  v128 = a1 + 72 * a2;
  if ( v128 == v49 )
    goto LABEL_54;
  while ( 2 )
  {
    v50 = sub_2B203A0(a8, *(_QWORD *)v49, *(int **)(v49 + 8), *(unsigned int *)(v49 + 16), v46, v44);
    v51 = v140;
    v52 = v133;
    v53 = *(_QWORD *)(v49 + 8);
    v54 = v139;
    v55 = (_DWORD *)v50;
    v57 = v56;
    v133 = (_DWORD *)v50;
    if ( !(_DWORD)v140 )
      goto LABEL_45;
    v58 = 0;
    while ( 2 )
    {
      while ( 2 )
      {
        v59 = v58;
        v60 = &v54[v58];
        if ( *(_DWORD *)(v53 + 4 * v58) != -1 )
        {
          if ( !v57 )
            v59 = *(_DWORD *)(v53 + 4 * v58);
          *v60 = v51 + v59;
          v54 = v139;
LABEL_40:
          if ( v51 == ++v58 )
            goto LABEL_44;
          continue;
        }
        break;
      }
      if ( *v60 == -1 )
        goto LABEL_40;
      *v60 = v58++;
      v54 = v139;
      if ( v51 != v58 )
        continue;
      break;
    }
LABEL_44:
    v51 = v140;
LABEL_45:
    v61 = (unsigned int *)*a10;
    v62 = **a10;
    if ( !v62 )
    {
      v92 = v52[30];
      if ( !v92 )
        v92 = v52[2];
      v93 = v55[30];
      if ( !v93 )
        v93 = v55[2];
      if ( v93 == v92 )
        *v61 = v93;
      else
        *v61 = v51;
      v62 = **a10;
    }
    v63 = *(_QWORD *)(**(_QWORD **)v55 + 8LL);
    v64 = *(unsigned __int8 *)(v63 + 8);
    if ( (_BYTE)v64 == 17 )
    {
      v62 *= *(_DWORD *)(v63 + 32);
      goto LABEL_48;
    }
    if ( (unsigned int)(v64 - 17) <= 1 )
LABEL_48:
      v63 = **(_QWORD **)(v63 + 16);
    v65 = sub_BCDA70((__int64 *)v63, v62);
    v66 = sub_2B097B0(*((__int64 **)a10[1] + 412), 6, v65, v54, v51, 0, 0, 0);
    v46 = v67;
    v68 = a10[2];
    if ( (_DWORD)v46 == 1 )
      v68[2] = 1;
    if ( __OFADD__(*(_QWORD *)v68, v66) )
    {
      v123 = v66 <= 0;
      v69 = 0x8000000000000000LL;
      if ( !v123 )
        v69 = 0x7FFFFFFFFFFFFFFFLL;
    }
    else
    {
      v69 = *(_QWORD *)v68 + v66;
    }
    *(_QWORD *)v68 = v69;
    v49 += 72;
    **a10 = v51;
    if ( v128 != v49 )
      continue;
    break;
  }
LABEL_54:
  v70 = v135;
  if ( (v135 & 1) == 0 && v135 )
  {
    if ( *(_QWORD *)v135 != v135 + 16 )
      _libc_free(*(_QWORD *)v135);
    j_j___libc_free_0(v70);
  }
  v71 = (unsigned __int64)v134;
  if ( ((unsigned __int8)v134 & 1) == 0 && v134 )
  {
    if ( (unsigned __int64 *)*v134 != v134 + 2 )
      _libc_free(*v134);
    j_j___libc_free_0(v71);
  }
  if ( v139 != (int *)v141 )
    _libc_free((unsigned __int64)v139);
}
