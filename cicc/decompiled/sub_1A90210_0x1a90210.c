// Function: sub_1A90210
// Address: 0x1a90210
//
__int64 __fastcall sub_1A90210(_QWORD *a1)
{
  __int64 v2; // r13
  char *v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  int v6; // r8d
  int v7; // r9d
  __int64 v8; // r13
  _QWORD *v9; // rbx
  _QWORD *v10; // r15
  _QWORD *i; // r14
  __int64 *v12; // rdi
  char *v13; // r8
  char *v14; // r9
  __int64 v15; // rdx
  __int64 v16; // r12
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rbx
  __int64 v20; // rbx
  __int64 v21; // r13
  __int64 v22; // rbx
  __int64 v23; // rax
  unsigned int *v24; // rax
  __int64 v25; // rcx
  unsigned __int8 *v26; // r13
  unsigned __int8 **v27; // rax
  char *v28; // rdi
  unsigned __int8 *v29; // rsi
  char *v30; // r13
  _QWORD *v31; // rbx
  __int64 v32; // r14
  unsigned __int64 v33; // rax
  _QWORD *v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rbx
  int v38; // ebx
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // r13
  __int64 v42; // rdx
  int v43; // ecx
  __int64 *v44; // r13
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 *v47; // rbx
  unsigned __int64 v48; // r14
  __int64 *v49; // rax
  int v50; // ecx
  int v51; // eax
  __int64 v52; // r13
  unsigned __int64 v53; // r13
  __int64 v54; // rdi
  __int64 v55; // rdi
  __int64 v56; // rdx
  __int64 v57; // rax
  char **v58; // rdi
  __int64 v59; // rax
  __int64 v60; // rdx
  unsigned __int8 *v61; // rsi
  void *v62; // rbx
  unsigned __int8 *v63; // r13
  unsigned __int8 *v64; // rax
  void *v65; // rax
  void *v66; // rsi
  unsigned __int64 v67; // rbx
  __int64 v68; // rax
  char *v69; // rcx
  size_t v70; // r13
  __int64 v71; // rbx
  _QWORD *v72; // rax
  _QWORD *v73; // r14
  __int64 v74; // rsi
  __int64 v75; // rax
  __int64 v76; // rsi
  unsigned __int8 *v77; // rsi
  unsigned int v78; // r12d
  _QWORD *v80; // rax
  size_t v81; // rdx
  size_t v82; // rdx
  unsigned __int64 v83; // rsi
  bool v84; // cf
  unsigned __int64 v85; // rax
  __int64 v86; // rsi
  __int64 v87; // rax
  char *v88; // r10
  char *v89; // rcx
  char *v90; // rax
  char *v91; // rdx
  _QWORD *v92; // rax
  char *v93; // rsi
  char *v94; // rdi
  char *v95; // rax
  char *v96; // rbx
  unsigned __int8 **v97; // rdi
  __int64 v98; // rax
  _QWORD *v99; // rax
  __int64 v100; // rax
  char *v101; // [rsp+8h] [rbp-238h]
  __int64 v102; // [rsp+28h] [rbp-218h]
  __int64 *v103; // [rsp+28h] [rbp-218h]
  __int64 v104; // [rsp+28h] [rbp-218h]
  char *v105; // [rsp+28h] [rbp-218h]
  char *v106; // [rsp+28h] [rbp-218h]
  unsigned __int64 v107; // [rsp+30h] [rbp-210h]
  __int64 *v108; // [rsp+38h] [rbp-208h]
  size_t v109; // [rsp+40h] [rbp-200h]
  char *v110; // [rsp+40h] [rbp-200h]
  __int64 v111; // [rsp+48h] [rbp-1F8h]
  _QWORD *v112; // [rsp+50h] [rbp-1F0h]
  __int64 v113; // [rsp+58h] [rbp-1E8h]
  char *v114; // [rsp+60h] [rbp-1E0h]
  char *v115; // [rsp+60h] [rbp-1E0h]
  char *v116; // [rsp+60h] [rbp-1E0h]
  _QWORD *v117; // [rsp+70h] [rbp-1D0h]
  __int64 v118; // [rsp+70h] [rbp-1D0h]
  __int64 *v119; // [rsp+88h] [rbp-1B8h]
  __int64 v120; // [rsp+90h] [rbp-1B0h] BYREF
  size_t v121; // [rsp+98h] [rbp-1A8h] BYREF
  __int64 v122[2]; // [rsp+A0h] [rbp-1A0h] BYREF
  __int16 v123; // [rsp+B0h] [rbp-190h]
  __int64 *v124; // [rsp+C0h] [rbp-180h] BYREF
  __int64 v125; // [rsp+C8h] [rbp-178h]
  _BYTE v126[32]; // [rsp+D0h] [rbp-170h] BYREF
  void *dest; // [rsp+F0h] [rbp-150h]
  size_t n; // [rsp+F8h] [rbp-148h]
  _QWORD v129[2]; // [rsp+100h] [rbp-140h] BYREF
  void *src; // [rsp+110h] [rbp-130h]
  void *v131; // [rsp+118h] [rbp-128h]
  __int64 v132; // [rsp+120h] [rbp-120h]
  unsigned __int8 *v133[2]; // [rsp+130h] [rbp-110h] BYREF
  _QWORD v134[2]; // [rsp+140h] [rbp-100h] BYREF
  char *v135; // [rsp+150h] [rbp-F0h]
  char *v136; // [rsp+158h] [rbp-E8h]
  char *v137; // [rsp+160h] [rbp-E0h]
  char *v138; // [rsp+170h] [rbp-D0h] BYREF
  size_t v139; // [rsp+178h] [rbp-C8h]
  unsigned __int8 *v140; // [rsp+180h] [rbp-C0h] BYREF
  __int64 v141; // [rsp+188h] [rbp-B8h]
  __int64 v142; // [rsp+190h] [rbp-B0h]
  int v143; // [rsp+198h] [rbp-A8h]
  __int64 v144; // [rsp+1A0h] [rbp-A0h]
  __int64 v145; // [rsp+1A8h] [rbp-98h]
  __int64 *v146; // [rsp+1C0h] [rbp-80h] BYREF
  __int64 v147; // [rsp+1C8h] [rbp-78h]
  _BYTE v148[112]; // [rsp+1D0h] [rbp-70h] BYREF

  v2 = a1[5];
  v3 = sub_15E0FD0(79);
  v5 = sub_16321A0(v2, (__int64)v3, v4);
  if ( !v5 )
    return 0;
  v8 = v5;
  if ( !*(_QWORD *)(v5 + 8) )
    return 0;
  v9 = (_QWORD *)a1[10];
  v10 = a1 + 9;
  v146 = (__int64 *)v148;
  v147 = 0x800000000LL;
  if ( a1 + 9 == v9 )
  {
    i = 0;
  }
  else
  {
    while ( 1 )
    {
      if ( !v9 )
LABEL_168:
        BUG();
      i = (_QWORD *)v9[3];
      if ( i != v9 + 2 )
        break;
      v9 = (_QWORD *)v9[1];
      if ( v10 == v9 )
        goto LABEL_7;
    }
  }
  if ( v10 != v9 )
  {
    if ( !i )
LABEL_152:
      BUG();
    while ( 1 )
    {
      if ( *((_BYTE *)i - 8) == 78 )
      {
        v98 = *(i - 6);
        if ( !*(_BYTE *)(v98 + 16) && *(_DWORD *)(v98 + 36) == 79 )
        {
          v100 = (unsigned int)v147;
          if ( (unsigned int)v147 >= HIDWORD(v147) )
          {
            sub_16CD150((__int64)&v146, v148, 0, 8, v6, v7);
            v100 = (unsigned int)v147;
          }
          v146[v100] = (__int64)(i - 3);
          LODWORD(v147) = v147 + 1;
        }
      }
      for ( i = (_QWORD *)i[1]; ; i = (_QWORD *)v9[3] )
      {
        v99 = v9 - 3;
        if ( !v9 )
          v99 = 0;
        if ( i != v99 + 5 )
          break;
        v9 = (_QWORD *)v9[1];
        if ( v10 == v9 )
          goto LABEL_7;
        if ( !v9 )
          goto LABEL_168;
      }
      if ( v10 == v9 )
        break;
      if ( !i )
        goto LABEL_152;
    }
  }
LABEL_7:
  if ( !(_DWORD)v147 )
  {
    v78 = 0;
    v108 = v146;
    goto LABEL_91;
  }
  v12 = (__int64 *)a1[5];
  v138 = **(char ***)(a1[3] + 16LL);
  v113 = sub_15E26F0(v12, 75, (__int64 *)&v138, 1);
  v15 = (unsigned int)v147;
  *(_WORD *)(v113 + 18) = *(_WORD *)(v113 + 18) & 0xC00F | *(_WORD *)(v8 + 18) & 0x3FF0;
  v108 = &v146[v15];
  if ( v146 == v108 )
  {
    v78 = 1;
    goto LABEL_91;
  }
  v119 = v146;
  do
  {
    v16 = *v119;
    if ( *(char *)(*v119 + 23) < 0 )
    {
      v17 = sub_1648A40(*v119);
      v19 = v17 + v18;
      if ( *(char *)(v16 + 23) < 0 )
        v19 -= sub_1648A40(v16);
      v20 = v19 >> 4;
      if ( (_DWORD)v20 )
      {
        v21 = 0;
        v22 = 16LL * (unsigned int)v20;
        while ( 1 )
        {
          v23 = 0;
          if ( *(char *)(v16 + 23) < 0 )
            v23 = sub_1648A40(v16);
          v24 = (unsigned int *)(v21 + v23);
          LODWORD(v14) = *(_DWORD *)(*(_QWORD *)v24 + 8LL);
          if ( !(_DWORD)v14 )
            break;
          v21 += 16;
          if ( v22 == v21 )
            goto LABEL_20;
        }
        v111 = *(_QWORD *)v24;
        v25 = 24LL * v24[2];
        v112 = (_QWORD *)(v16 + v25 - 24LL * (*(_DWORD *)(v16 + 20) & 0xFFFFFFF));
        v107 = 0xAAAAAAAAAAAAAAABLL * ((24LL * v24[3] - v25) >> 3);
      }
    }
LABEL_20:
    n = 0;
    LOBYTE(v129[0]) = 0;
    dest = v129;
    src = 0;
    v131 = 0;
    v132 = 0;
    v26 = *(unsigned __int8 **)v111;
    v133[0] = v26;
    v138 = (char *)&v140;
    if ( (unsigned __int64)v26 > 0xF )
    {
      v138 = (char *)sub_22409D0(&v138, v133, 0);
      v97 = (unsigned __int8 **)v138;
      v140 = v133[0];
    }
    else
    {
      if ( v26 == (unsigned __int8 *)1 )
      {
        LOBYTE(v140) = *(_BYTE *)(v111 + 16);
        v27 = &v140;
        goto LABEL_23;
      }
      if ( !v26 )
      {
        v27 = &v140;
        goto LABEL_23;
      }
      v97 = &v140;
    }
    memcpy(v97, (const void *)(v111 + 16), (size_t)v26);
    v26 = v133[0];
    v27 = (unsigned __int8 **)v138;
LABEL_23:
    v139 = (size_t)v26;
    v26[(_QWORD)v27] = 0;
    v28 = (char *)dest;
    if ( v138 == (char *)&v140 )
    {
      v81 = v139;
      if ( v139 )
      {
        if ( v139 == 1 )
          *(_BYTE *)dest = (_BYTE)v140;
        else
          memcpy(dest, &v140, v139);
        v81 = v139;
        v28 = (char *)dest;
      }
      n = v81;
      v28[v81] = 0;
      v28 = v138;
    }
    else
    {
      if ( dest == v129 )
      {
        dest = v138;
        n = v139;
        v129[0] = v140;
      }
      else
      {
        v29 = (unsigned __int8 *)v129[0];
        dest = v138;
        n = v139;
        v129[0] = v140;
        if ( v28 )
        {
          v138 = v28;
          v140 = v29;
          goto LABEL_27;
        }
      }
      v28 = (char *)&v140;
      v138 = (char *)&v140;
    }
LABEL_27:
    v139 = 0;
    *v28 = 0;
    if ( v138 != (char *)&v140 )
      j_j___libc_free_0(v138, v140 + 1);
    v30 = (char *)v131;
    v31 = &v112[3 * v107];
    if ( v31 != v112 )
    {
      v32 = v132;
      v33 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(24 * v107) >> 3);
      if ( v33 <= (v132 - (__int64)v131) >> 3 )
      {
        v34 = v112;
        do
        {
          if ( v30 )
            *(_QWORD *)v30 = *v34;
          v34 += 3;
          v30 += 8;
        }
        while ( v31 != v34 );
        v131 = (char *)v131 + 0x5555555555555558LL * ((__int64)(24 * v107) >> 3);
        goto LABEL_36;
      }
      v13 = (char *)src;
      v82 = (_BYTE *)v131 - (_BYTE *)src;
      v83 = ((_BYTE *)v131 - (_BYTE *)src) >> 3;
      if ( v33 > 0xFFFFFFFFFFFFFFFLL - v83 )
        sub_4262D8((__int64)"vector::_M_range_insert");
      if ( v33 < v83 )
        v33 = ((_BYTE *)v131 - (_BYTE *)src) >> 3;
      v84 = __CFADD__(v83, v33);
      v85 = v83 + v33;
      v14 = (char *)v85;
      if ( v84 )
      {
        v86 = 0x7FFFFFFFFFFFFFF8LL;
      }
      else
      {
        if ( !v85 )
        {
          v118 = 0;
          v88 = (char *)v131;
          v89 = 0;
          goto LABEL_119;
        }
        if ( v85 > 0xFFFFFFFFFFFFFFFLL )
          v85 = 0xFFFFFFFFFFFFFFFLL;
        v86 = 8 * v85;
      }
      v87 = sub_22077B0(v86);
      v88 = (char *)v131;
      v89 = (char *)v87;
      v13 = (char *)src;
      v32 = v132;
      v118 = v86 + v87;
      v82 = v30 - (_BYTE *)src;
      v14 = (char *)((_BYTE *)v131 - v30);
LABEL_119:
      if ( v30 != v13 )
      {
        v101 = v88;
        v105 = v14;
        v109 = v82;
        v114 = v13;
        v90 = (char *)memmove(v89, v13, v82);
        v88 = v101;
        v14 = v105;
        v82 = v109;
        v13 = v114;
        v89 = v90;
      }
      v91 = &v89[v82];
      v92 = v112;
      v93 = v91;
      do
      {
        if ( v93 )
          *(_QWORD *)v93 = *v92;
        v92 += 3;
        v93 += 8;
      }
      while ( v31 != v92 );
      v94 = &v91[0x5555555555555558LL * ((24 * v107 - 24) >> 3) + 8];
      if ( v30 != v88 )
      {
        v106 = v89;
        v110 = v13;
        v115 = v14;
        v95 = (char *)memcpy(v94, v30, (size_t)v14);
        v89 = v106;
        v13 = v110;
        v14 = v115;
        v94 = v95;
      }
      v96 = &v14[(_QWORD)v94];
      if ( v13 )
      {
        v116 = v89;
        j_j___libc_free_0(v13, v32 - (_QWORD)v13);
        v89 = v116;
      }
      src = v89;
      v131 = v96;
      v132 = v118;
    }
LABEL_36:
    if ( *(char *)(v16 + 23) >= 0 )
      goto LABEL_95;
    v35 = sub_1648A40(v16);
    v37 = v35 + v36;
    if ( *(char *)(v16 + 23) >= 0 )
    {
      if ( (unsigned int)(v37 >> 4) )
LABEL_166:
        BUG();
LABEL_95:
      v42 = -24;
      v41 = -24;
      goto LABEL_42;
    }
    if ( !(unsigned int)((v37 - sub_1648A40(v16)) >> 4) )
      goto LABEL_95;
    if ( *(char *)(v16 + 23) >= 0 )
      goto LABEL_166;
    v38 = *(_DWORD *)(sub_1648A40(v16) + 8);
    if ( *(char *)(v16 + 23) >= 0 )
      BUG();
    v39 = sub_1648A40(v16);
    v41 = -24 - 24LL * (unsigned int)(*(_DWORD *)(v39 + v40 - 4) - v38);
    v42 = v41;
LABEL_42:
    v43 = *(_DWORD *)(v16 + 20);
    v44 = (__int64 *)(v16 + v41);
    v125 = 0x400000000LL;
    v45 = 24 * (1LL - (v43 & 0xFFFFFFF));
    v124 = (__int64 *)v126;
    v46 = v42 - v45;
    v47 = (__int64 *)(v16 + v45);
    v48 = 0xAAAAAAAAAAAAAAABLL * (v46 >> 3);
    v49 = (__int64 *)v126;
    v50 = 0;
    if ( (unsigned __int64)v46 > 0x60 )
    {
      sub_16CD150((__int64)&v124, v126, 0xAAAAAAAAAAAAAAABLL * (v46 >> 3), 8, (int)v13, (int)v14);
      v50 = v125;
      v49 = &v124[(unsigned int)v125];
    }
    if ( v44 != v47 )
    {
      do
      {
        if ( v49 )
          *v49 = *v47;
        v47 += 3;
        ++v49;
      }
      while ( v44 != v47 );
      v50 = v125;
    }
    v51 = *(_DWORD *)(v16 + 20);
    v52 = *(_QWORD *)(v16 + 40);
    LODWORD(v125) = v50 + v48;
    v117 = (_QWORD *)sub_1AA92B0(*(_QWORD *)(v16 - 24LL * (v51 & 0xFFFFFFF)), v16, 1, 0, 0, 0);
    v53 = sub_157EBA0(v52);
    sub_15F89F0(v53);
    v54 = *(_QWORD *)(v53 - 24);
    v138 = "guarded";
    LOWORD(v140) = 259;
    sub_164B780(v54, (__int64 *)&v138);
    v55 = *(_QWORD *)(v53 - 48);
    v138 = "deopt";
    LOWORD(v140) = 259;
    sub_164B780(v55, (__int64 *)&v138);
    if ( *(_QWORD *)(v16 + 48) || *(__int16 *)(v16 + 18) < 0 )
    {
      v56 = sub_1625790(v16, 14);
      if ( v56 )
        sub_1625C10(v53, 14, v56);
    }
    v120 = sub_16498A0(v16);
    v57 = sub_161BE60(&v120, dword_4FB57C0, 1u);
    sub_1625C10(v53, 2, v57);
    v58 = (char **)v117;
    v59 = sub_16498A0((__int64)v117);
    v138 = 0;
    v141 = v59;
    v142 = 0;
    v143 = 0;
    v144 = 0;
    v145 = 0;
    v139 = v117[5];
    v140 = (unsigned __int8 *)(v117 + 3);
    v61 = (unsigned __int8 *)v117[6];
    v133[0] = v61;
    if ( v61 )
    {
      v58 = (char **)v133;
      sub_1623A60((__int64)v133, (__int64)v61, 2);
      if ( v138 )
      {
        v58 = &v138;
        sub_161E7C0((__int64)&v138, (__int64)v138);
      }
      v138 = (char *)v133[0];
      if ( v133[0] )
      {
        v58 = (char **)v133;
        sub_1623210((__int64)v133, v133[0], (__int64)&v138);
      }
    }
    v62 = dest;
    v63 = (unsigned __int8 *)n;
    v123 = 257;
    v133[0] = (unsigned __int8 *)v134;
    if ( (char *)dest + n && !dest )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v121 = n;
    if ( n > 0xF )
    {
      v133[0] = (unsigned __int8 *)sub_22409D0(v133, &v121, 0);
      v58 = (char **)v133[0];
      v134[0] = v121;
    }
    else
    {
      if ( n == 1 )
      {
        LOBYTE(v134[0]) = *(_BYTE *)dest;
        v64 = (unsigned __int8 *)v134;
        goto LABEL_62;
      }
      if ( !n )
      {
        v64 = (unsigned __int8 *)v134;
        goto LABEL_62;
      }
      v58 = (char **)v134;
    }
    memcpy(v58, v62, (size_t)v63);
    v63 = (unsigned __int8 *)v121;
    v64 = v133[0];
LABEL_62:
    v133[1] = v63;
    v63[(_QWORD)v64] = 0;
    v65 = v131;
    v66 = src;
    v135 = 0;
    v136 = 0;
    v137 = 0;
    v67 = (_BYTE *)v131 - (_BYTE *)src;
    if ( v131 == src )
    {
      v70 = 0;
      v67 = 0;
      v69 = 0;
    }
    else
    {
      if ( v67 > 0x7FFFFFFFFFFFFFF8LL )
        sub_4261EA(v58, src, v60);
      v68 = sub_22077B0((_BYTE *)v131 - (_BYTE *)src);
      v66 = src;
      v69 = (char *)v68;
      v65 = v131;
      v70 = (_BYTE *)v131 - (_BYTE *)src;
    }
    v135 = v69;
    v136 = v69;
    v137 = &v69[v67];
    if ( v66 != v65 )
      v69 = (char *)memmove(v69, v66, v70);
    v136 = &v69[v70];
    v71 = sub_1A8FF40((__int64)&v138, (_QWORD *)v113, v124, (unsigned int)v125, (__int64 *)v133, 1, v122, 0);
    if ( v135 )
      j_j___libc_free_0(v135, v137 - v135);
    if ( (_QWORD *)v133[0] != v134 )
      j_j___libc_free_0(v133[0], v134[0] + 1LL);
    if ( *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(v113 + 24) + 16LL) + 8LL) )
    {
      v133[0] = "deoptcall";
      LOWORD(v134[0]) = 259;
      sub_164B780(v71, (__int64 *)v133);
      LOWORD(v134[0]) = 257;
      v104 = v141;
      v80 = sub_1648A60(56, v71 != 0);
      v73 = v80;
      if ( v80 )
        sub_15F6F90((__int64)v80, v104, v71, 0);
    }
    else
    {
      LOWORD(v134[0]) = 257;
      v102 = v141;
      v72 = sub_1648A60(56, 0);
      v73 = v72;
      if ( v72 )
        sub_15F6F90((__int64)v72, v102, 0, 0);
    }
    if ( v139 )
    {
      v103 = (__int64 *)v140;
      sub_157E9D0(v139 + 40, (__int64)v73);
      v74 = *v103;
      v75 = v73[3] & 7LL;
      v73[4] = v103;
      v74 &= 0xFFFFFFFFFFFFFFF8LL;
      v73[3] = v74 | v75;
      *(_QWORD *)(v74 + 8) = v73 + 3;
      *v103 = *v103 & 7 | (unsigned __int64)(v73 + 3);
    }
    sub_164B780((__int64)v73, (__int64 *)v133);
    if ( v138 )
    {
      v122[0] = (__int64)v138;
      sub_1623A60((__int64)v122, (__int64)v138, 2);
      v76 = v73[6];
      if ( v76 )
        sub_161E7C0((__int64)(v73 + 6), v76);
      v77 = (unsigned __int8 *)v122[0];
      v73[6] = v122[0];
      if ( v77 )
        sub_1623210((__int64)v122, v77, (__int64)(v73 + 6));
    }
    *(_WORD *)(v71 + 18) = *(_WORD *)(v71 + 18) & 0x8000
                         | *(_WORD *)(v71 + 18) & 3
                         | (4 * ((*(_WORD *)(v16 + 18) >> 2) & 0xDFFF));
    sub_15F20C0(v117);
    if ( v138 )
      sub_161E7C0((__int64)&v138, (__int64)v138);
    if ( v124 != (__int64 *)v126 )
      _libc_free((unsigned __int64)v124);
    if ( src )
      j_j___libc_free_0(src, v132 - (_QWORD)src);
    if ( dest != v129 )
      j_j___libc_free_0(dest, v129[0] + 1LL);
    sub_15F20C0((_QWORD *)v16);
    ++v119;
  }
  while ( v108 != v119 );
  v78 = 1;
  v108 = v146;
LABEL_91:
  if ( v108 != (__int64 *)v148 )
    _libc_free((unsigned __int64)v108);
  return v78;
}
