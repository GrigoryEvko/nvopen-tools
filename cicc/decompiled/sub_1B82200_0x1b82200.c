// Function: sub_1B82200
// Address: 0x1b82200
//
void __fastcall sub_1B82200(__int64 a1, __int64 a2, __int64 a3, __int64 **a4, __int64 a5)
{
  __int64 v6; // r12
  __int64 v7; // rbx
  unsigned __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r13
  __int64 v11; // rdx
  __int64 v12; // rcx
  int v13; // r8d
  int v14; // r9d
  int v15; // r11d
  int v16; // r10d
  __int64 **v17; // rcx
  __int64 v18; // r8
  char v19; // al
  __int64 v20; // rdx
  bool v21; // zf
  int v22; // eax
  __int64 v23; // rax
  int v24; // r9d
  __int64 v25; // rdx
  int v26; // r14d
  __int64 v27; // r13
  __int64 v28; // rax
  int v29; // r8d
  __int64 v30; // rax
  __int64 v31; // r8
  int v32; // r9d
  __int64 v33; // rax
  int v34; // eax
  __int64 v35; // rax
  int v36; // r9d
  unsigned int v37; // r13d
  __int64 v38; // r14
  __int64 v39; // rsi
  __int64 v40; // r13
  __int64 v41; // rax
  int v42; // r8d
  __int64 v43; // rax
  __int64 v44; // r8
  int v45; // r9d
  __int64 v46; // rax
  __int64 v47; // rdx
  int v48; // eax
  __int64 v49; // rax
  unsigned __int64 **v50; // r14
  __int64 v51; // rdx
  __int64 v52; // rcx
  int v53; // r8d
  int v54; // r9d
  unsigned __int64 **v55; // r10
  int v56; // esi
  __int64 v57; // rcx
  unsigned __int64 *v58; // r13
  __int64 v59; // rdx
  __int64 v60; // rax
  char v61; // al
  unsigned int *v62; // rbx
  __int64 *v63; // r15
  _QWORD *v64; // r14
  _QWORD *v65; // rax
  _QWORD *v66; // rbx
  __int64 v67; // rdi
  unsigned __int64 *v68; // r14
  __int64 v69; // rax
  unsigned __int64 v70; // rcx
  __int64 v71; // rsi
  __int64 v72; // rsi
  unsigned __int8 *v73; // rsi
  _QWORD *v74; // rax
  __int64 v75; // rdx
  __int64 v76; // rax
  __int64 v77; // rdx
  unsigned __int64 v78; // rax
  __int64 v79; // rax
  __int64 v80; // rax
  unsigned __int64 *v81; // rbx
  __int64 v82; // rax
  unsigned __int64 v83; // rcx
  __int64 v84; // rsi
  __int64 v85; // rsi
  unsigned __int8 *v86; // rsi
  __int64 v87; // rax
  __int64 v88; // rcx
  __int64 v89; // rax
  __int64 v90; // rsi
  bool v91; // cf
  unsigned __int64 v92; // rax
  __int64 v93; // rsi
  __int64 v94; // rax
  _QWORD *v95; // rcx
  _QWORD *v96; // rdx
  __int64 v97; // rcx
  __int64 v98; // rax
  unsigned __int64 *v99; // rbx
  __int64 v100; // r12
  __int64 v101; // r14
  __int64 v102; // r13
  __int64 v103; // rax
  unsigned __int64 *v104; // r13
  unsigned __int64 **v105; // r10
  unsigned __int64 *v106; // rbx
  unsigned __int64 **v107; // r14
  unsigned __int64 v108; // rdi
  unsigned __int64 *v109; // rax
  unsigned __int64 *v110; // rdi
  char *v111; // [rsp+8h] [rbp-168h]
  unsigned __int64 **v112; // [rsp+8h] [rbp-168h]
  unsigned __int64 **v113; // [rsp+8h] [rbp-168h]
  unsigned __int64 **v114; // [rsp+10h] [rbp-160h]
  __int64 v115; // [rsp+10h] [rbp-160h]
  __int64 v116; // [rsp+10h] [rbp-160h]
  _QWORD *v117; // [rsp+10h] [rbp-160h]
  _QWORD *v118; // [rsp+10h] [rbp-160h]
  __int64 v119; // [rsp+18h] [rbp-158h]
  __int64 v120; // [rsp+20h] [rbp-150h]
  unsigned __int64 **v121; // [rsp+20h] [rbp-150h]
  __int64 v122; // [rsp+28h] [rbp-148h]
  unsigned __int64 *v123; // [rsp+28h] [rbp-148h]
  unsigned __int64 **v124; // [rsp+28h] [rbp-148h]
  __int64 v125; // [rsp+30h] [rbp-140h]
  unsigned __int64 **v126; // [rsp+30h] [rbp-140h]
  __int64 v127; // [rsp+30h] [rbp-140h]
  unsigned __int64 **v128; // [rsp+30h] [rbp-140h]
  unsigned __int64 **v129; // [rsp+30h] [rbp-140h]
  _BOOL4 v130; // [rsp+38h] [rbp-138h]
  __int64 v131; // [rsp+40h] [rbp-130h]
  __int64 v132; // [rsp+40h] [rbp-130h]
  int v133; // [rsp+48h] [rbp-128h]
  __int64 v134; // [rsp+48h] [rbp-128h]
  __int64 v135; // [rsp+48h] [rbp-128h]
  __int64 v136; // [rsp+50h] [rbp-120h]
  int v137; // [rsp+50h] [rbp-120h]
  int v138; // [rsp+58h] [rbp-118h]
  unsigned __int64 **v139; // [rsp+58h] [rbp-118h]
  unsigned __int64 **v140; // [rsp+58h] [rbp-118h]
  unsigned __int64 **v141; // [rsp+58h] [rbp-118h]
  __int64 v144; // [rsp+68h] [rbp-108h]
  __int64 v145; // [rsp+68h] [rbp-108h]
  __int64 v146; // [rsp+78h] [rbp-F8h] BYREF
  _QWORD v147[2]; // [rsp+80h] [rbp-F0h] BYREF
  __int16 v148; // [rsp+90h] [rbp-E0h]
  __int64 v149[2]; // [rsp+A0h] [rbp-D0h] BYREF
  __int16 v150; // [rsp+B0h] [rbp-C0h]
  __int64 v151[2]; // [rsp+C0h] [rbp-B0h] BYREF
  __int16 v152; // [rsp+D0h] [rbp-A0h]
  char *v153; // [rsp+E0h] [rbp-90h] BYREF
  __int64 v154; // [rsp+E8h] [rbp-88h]
  _BYTE v155[16]; // [rsp+F0h] [rbp-80h] BYREF
  __int64 **v156; // [rsp+100h] [rbp-70h] BYREF
  __int64 v157; // [rsp+108h] [rbp-68h]
  _BYTE v158[32]; // [rsp+110h] [rbp-60h] BYREF
  __int64 v159; // [rsp+130h] [rbp-40h]

  v6 = a2;
  v7 = a3;
  v8 = *(unsigned __int8 *)(a3 + 8);
  v9 = 100990;
  if ( !_bittest64(&v9, v8) )
  {
    v21 = (_BYTE)v8 == 14;
    v22 = *(_DWORD *)(a1 + 104);
    if ( v21 )
    {
      v133 = *(_DWORD *)(a1 + 104);
      *(_DWORD *)(a1 + 104) = (*(_DWORD *)(a1 + 108) | v22) & -(*(_DWORD *)(a1 + 108) | v22);
      v23 = sub_127FA20(*(_QWORD *)a1, *(_QWORD *)(v7 + 24));
      v25 = *(_QWORD *)(v7 + 32);
      v138 = (unsigned __int64)(v23 + 7) >> 3;
      if ( (_DWORD)v25 )
      {
        v26 = 0;
        v27 = 0;
        v136 = (unsigned int)v25;
        v28 = *(unsigned int *)(a1 + 16);
        do
        {
          v29 = v27;
          if ( (unsigned int)v28 >= *(_DWORD *)(a1 + 20) )
          {
            sub_16CD150(a1 + 8, (const void *)(a1 + 24), 0, 4, v27, v24);
            v28 = *(unsigned int *)(a1 + 16);
            v29 = v27;
          }
          *(_DWORD *)(*(_QWORD *)(a1 + 8) + 4 * v28) = v29;
          ++*(_DWORD *)(a1 + 16);
          v30 = sub_1643350(*(_QWORD **)(a2 + 24));
          v31 = sub_159C470(v30, v27, 0);
          v33 = *(unsigned int *)(a1 + 48);
          if ( (unsigned int)v33 >= *(_DWORD *)(a1 + 52) )
          {
            v122 = v31;
            sub_16CD150(a1 + 40, (const void *)(a1 + 56), 0, 8, v31, v32);
            v33 = *(unsigned int *)(a1 + 48);
            v31 = v122;
          }
          ++v27;
          *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v33) = v31;
          ++*(_DWORD *)(a1 + 48);
          *(_DWORD *)(a1 + 108) = v26;
          sub_1B82200(a1, a2, *(_QWORD *)(v7 + 24), a4, a5);
          v34 = *(_DWORD *)(a1 + 16);
          --*(_DWORD *)(a1 + 48);
          v26 += v138;
          v28 = (unsigned int)(v34 - 1);
          *(_DWORD *)(a1 + 16) = v28;
        }
        while ( v136 != v27 );
      }
      *(_DWORD *)(a1 + 104) = v133;
    }
    else
    {
      v137 = *(_DWORD *)(a1 + 104);
      *(_DWORD *)(a1 + 104) = (*(_DWORD *)(a1 + 108) | v22) & -(*(_DWORD *)(a1 + 108) | v22);
      v35 = sub_15A9930(*(_QWORD *)a1, v7);
      v37 = *(_DWORD *)(v7 + 12);
      v38 = v35;
      if ( v37 )
      {
        v39 = v37;
        v40 = 0;
        v41 = *(unsigned int *)(a1 + 16);
        do
        {
          v42 = v40;
          if ( *(_DWORD *)(a1 + 20) <= (unsigned int)v41 )
          {
            sub_16CD150(a1 + 8, (const void *)(a1 + 24), 0, 4, v40, v36);
            v41 = *(unsigned int *)(a1 + 16);
            v42 = v40;
          }
          *(_DWORD *)(*(_QWORD *)(a1 + 8) + 4 * v41) = v42;
          ++*(_DWORD *)(a1 + 16);
          v43 = sub_1643350(*(_QWORD **)(v6 + 24));
          v44 = sub_159C470(v43, v40, 0);
          v46 = *(unsigned int *)(a1 + 48);
          if ( (unsigned int)v46 >= *(_DWORD *)(a1 + 52) )
          {
            v125 = v44;
            sub_16CD150(a1 + 40, (const void *)(a1 + 56), 0, 8, v44, v45);
            v46 = *(unsigned int *)(a1 + 48);
            v44 = v125;
          }
          *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v46) = v44;
          ++*(_DWORD *)(a1 + 48);
          *(_DWORD *)(a1 + 108) = *(_QWORD *)(v38 + 8 * v40 + 16);
          v47 = *(_QWORD *)(*(_QWORD *)(v7 + 16) + 8 * v40++);
          sub_1B82200(a1, v6, v47, a4, a5);
          v48 = *(_DWORD *)(a1 + 16);
          --*(_DWORD *)(a1 + 48);
          v41 = (unsigned int)(v48 - 1);
          *(_DWORD *)(a1 + 16) = v41;
        }
        while ( v39 != v40 );
      }
      *(_DWORD *)(a1 + 104) = v137;
    }
    return;
  }
  v10 = -(__int64)(unsigned int)(*(_DWORD *)(a1 + 104) | *(_DWORD *)(a1 + 108))
      & (unsigned int)(*(_DWORD *)(a1 + 104) | *(_DWORD *)(a1 + 108));
  if ( !byte_4FB7940
    || v7 != sub_1643330(*(_QWORD **)(a2 + 24)) && v7 != sub_1643340(*(_QWORD **)(a2 + 24))
    || (v7 == sub_1643330(*(_QWORD **)(a2 + 24))
      ? (v49 = 0x2E8BA2E8BA2E8BA3LL * ((__int64)(*(_QWORD *)(a1 + 120) - *(_QWORD *)(a1 + 112)) >> 3))
      : (v49 = 0x2E8BA2E8BA2E8BA3LL * ((__int64)(*(_QWORD *)(a1 + 144) - *(_QWORD *)(a1 + 136)) >> 3)),
        !v49 && (v10 & 3) != 0) )
  {
    sub_1B81290(a1, (__int64 *)a2, a4, a5, (unsigned __int64 **)(a1 + 112), 0);
    sub_1B81290(a1, (__int64 *)a2, a4, a5, (unsigned __int64 **)(a1 + 136), 1);
    v15 = *(_DWORD *)(a1 + 16);
    v153 = v155;
    v154 = 0x400000000LL;
    if ( v15 )
      sub_1B7D5D0((__int64)&v153, a1 + 8, v11, v12, v13, v14);
    v16 = *(_DWORD *)(a1 + 48);
    v17 = (__int64 **)v158;
    v18 = 0;
    v156 = (__int64 **)v158;
    v157 = 0x400000000LL;
    if ( v16 )
    {
      sub_1B7D3B0((__int64)&v156, a1 + 40, v11, (__int64)v158, 0, v14);
      v17 = v156;
      v18 = (unsigned int)v157;
    }
    v159 = v10;
    v19 = *(_BYTE *)(a5 + 16);
    if ( v19 )
    {
      if ( v19 == 1 )
      {
        v147[0] = ".gep";
        v148 = 259;
      }
      else
      {
        if ( *(_BYTE *)(a5 + 17) == 1 )
        {
          v20 = *(_QWORD *)a5;
        }
        else
        {
          v20 = a5;
          v19 = 2;
        }
        v147[0] = v20;
        v147[1] = ".gep";
        LOBYTE(v148) = v19;
        HIBYTE(v148) = 3;
      }
    }
    else
    {
      v148 = 256;
    }
    v135 = sub_128B460((__int64 *)a2, *(_QWORD *)(a1 + 96), *(_BYTE **)(a1 + 88), v17, v18, (__int64)v147);
    v61 = *(_BYTE *)(a5 + 16);
    if ( v61 )
    {
      if ( v61 == 1 )
      {
        v149[0] = (__int64)".extract";
        v150 = 259;
      }
      else
      {
        if ( *(_BYTE *)(a5 + 17) == 1 )
          a5 = *(_QWORD *)a5;
        else
          v61 = 2;
        LOBYTE(v150) = v61;
        HIBYTE(v150) = 3;
        v149[0] = a5;
        v149[1] = (__int64)".extract";
      }
    }
    else
    {
      v150 = 256;
    }
    v62 = (unsigned int *)v153;
    v63 = *a4;
    if ( *((_BYTE *)*a4 + 16) > 0x10u )
    {
      v144 = (unsigned int)v154;
      v152 = 257;
      v74 = sub_1648A60(88, 1u);
      v64 = v74;
      if ( v74 )
      {
        v75 = v144;
        v132 = v144;
        v145 = (__int64)v74;
        v76 = sub_15FB2A0(*v63, v62, v75);
        sub_15F1EA0((__int64)v64, v76, 62, (__int64)(v64 - 3), 1, 0);
        if ( *(v64 - 3) )
        {
          v77 = *(v64 - 2);
          v78 = *(v64 - 1) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v78 = v77;
          if ( v77 )
            *(_QWORD *)(v77 + 16) = *(_QWORD *)(v77 + 16) & 3LL | v78;
        }
        *(v64 - 3) = v63;
        v79 = v63[1];
        *(v64 - 2) = v79;
        if ( v79 )
          *(_QWORD *)(v79 + 16) = (unsigned __int64)(v64 - 2) | *(_QWORD *)(v79 + 16) & 3LL;
        *(v64 - 1) = (unsigned __int64)(v63 + 1) | *(v64 - 1) & 3LL;
        v63[1] = (__int64)(v64 - 3);
        v64[7] = v64 + 9;
        v64[8] = 0x400000000LL;
        sub_15FB110((__int64)v64, v62, v132, (__int64)v151);
      }
      else
      {
        v145 = 0;
      }
      v80 = *(_QWORD *)(a2 + 8);
      if ( v80 )
      {
        v81 = *(unsigned __int64 **)(a2 + 16);
        sub_157E9D0(v80 + 40, (__int64)v64);
        v82 = v64[3];
        v83 = *v81;
        v64[4] = v81;
        v83 &= 0xFFFFFFFFFFFFFFF8LL;
        v64[3] = v83 | v82 & 7;
        *(_QWORD *)(v83 + 8) = v64 + 3;
        *v81 = *v81 & 7 | (unsigned __int64)(v64 + 3);
      }
      sub_164B780(v145, v149);
      v84 = *(_QWORD *)a2;
      if ( *(_QWORD *)v6 )
      {
        v146 = *(_QWORD *)v6;
        sub_1623A60((__int64)&v146, v84, 2);
        v85 = v64[6];
        if ( v85 )
          sub_161E7C0((__int64)(v64 + 6), v85);
        v86 = (unsigned __int8 *)v146;
        v64[6] = v146;
        if ( v86 )
          sub_1623210((__int64)&v146, v86, (__int64)(v64 + 6));
      }
    }
    else
    {
      v64 = (_QWORD *)sub_15A3AE0(*a4, (unsigned int *)v153, (unsigned int)v154, 0);
    }
    v152 = 257;
    v65 = sub_1648A60(64, 2u);
    v66 = v65;
    if ( v65 )
      sub_15F9650((__int64)v65, (__int64)v64, v135, 0, 0);
    v67 = *(_QWORD *)(v6 + 8);
    if ( v67 )
    {
      v68 = *(unsigned __int64 **)(v6 + 16);
      sub_157E9D0(v67 + 40, (__int64)v66);
      v69 = v66[3];
      v70 = *v68;
      v66[4] = v68;
      v70 &= 0xFFFFFFFFFFFFFFF8LL;
      v66[3] = v70 | v69 & 7;
      *(_QWORD *)(v70 + 8) = v66 + 3;
      *v68 = *v68 & 7 | (unsigned __int64)(v66 + 3);
    }
    sub_164B780((__int64)v66, v151);
    v71 = *(_QWORD *)v6;
    if ( *(_QWORD *)v6 )
    {
      v146 = *(_QWORD *)v6;
      sub_1623A60((__int64)&v146, v71, 2);
      v72 = v66[6];
      if ( v72 )
        sub_161E7C0((__int64)(v66 + 6), v72);
      v73 = (unsigned __int8 *)v146;
      v66[6] = v146;
      if ( v73 )
        sub_1623210((__int64)&v146, v73, (__int64)(v66 + 6));
    }
    sub_15F9450((__int64)v66, v10);
    if ( v156 != (__int64 **)v158 )
      _libc_free((unsigned __int64)v156);
    if ( v153 != v155 )
      _libc_free((unsigned __int64)v153);
    return;
  }
  v50 = (unsigned __int64 **)(a1 + 112);
  v131 = sub_1643330(*(_QWORD **)(a2 + 24));
  v130 = v7 != v131;
  if ( v7 == v131 )
  {
    sub_1B81290(a1, (__int64 *)a2, a4, a5, (unsigned __int64 **)(a1 + 136), 1);
    v134 = sub_15A9FF0(*(_QWORD *)a1, **a4, *(_QWORD **)(a1 + 40), *(unsigned int *)(a1 + 48));
    if ( !*(_BYTE *)(a1 + 184) )
    {
LABEL_40:
      v55 = v50;
      goto LABEL_41;
    }
    v55 = (unsigned __int64 **)(a1 + 112);
    if ( *(_QWORD *)(a1 + 176) + 1LL == v134 )
      goto LABEL_41;
LABEL_39:
    sub_1B81290(a1, (__int64 *)a2, a4, a5, v50, v130);
    goto LABEL_40;
  }
  sub_1B81290(a1, (__int64 *)a2, a4, a5, (unsigned __int64 **)(a1 + 112), 0);
  v87 = sub_15A9FF0(*(_QWORD *)a1, **a4, *(_QWORD **)(a1 + 40), *(unsigned int *)(a1 + 48));
  v55 = (unsigned __int64 **)(a1 + 136);
  v134 = v87;
  if ( *(_BYTE *)(a1 + 184) )
  {
    v50 = (unsigned __int64 **)(a1 + 136);
    if ( v87 != *(_QWORD *)(a1 + 176) + 2LL )
      goto LABEL_39;
  }
LABEL_41:
  v56 = *(_DWORD *)(a1 + 16);
  v153 = v155;
  v154 = 0x400000000LL;
  if ( v56 )
  {
    v141 = v55;
    sub_1B7D5D0((__int64)&v153, a1 + 8, v51, v52, v53, v54);
    v55 = v141;
  }
  v57 = *(unsigned int *)(a1 + 48);
  v156 = (__int64 **)v158;
  v157 = 0x400000000LL;
  if ( (_DWORD)v57 )
  {
    v126 = v55;
    sub_1B7D3B0((__int64)&v156, a1 + 40, v51, v57, v53, v54);
    v55 = v126;
  }
  v159 = v10;
  v58 = v55[1];
  if ( v58 != v55[2] )
  {
    if ( v58 )
    {
      *v58 = (unsigned __int64)(v58 + 2);
      v58[1] = 0x400000000LL;
      v59 = (unsigned int)v154;
      if ( (_DWORD)v154 )
      {
        v129 = v55;
        sub_1B7D490((__int64)v58, &v153, (unsigned int)v154, v57, v53, v54);
        v55 = v129;
      }
      v58[4] = (unsigned __int64)(v58 + 6);
      v58[5] = 0x400000000LL;
      if ( (_DWORD)v157 )
      {
        v128 = v55;
        sub_1B7CFB0((__int64)(v58 + 4), (char **)&v156, v59, v57, v53, v54);
        v55 = v128;
      }
      v58[10] = v159;
      v58 = v55[1];
    }
    v55[1] = v58 + 11;
    goto LABEL_53;
  }
  v88 = (char *)v58 - (char *)*v55;
  v123 = *v55;
  v89 = 0x2E8BA2E8BA2E8BA3LL * (v88 >> 3);
  if ( v89 == 0x1745D1745D1745DLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v90 = 1;
  if ( v89 )
    v90 = 0x2E8BA2E8BA2E8BA3LL * (v88 >> 3);
  v91 = __CFADD__(v90, v89);
  v92 = v90 + v89;
  if ( v91 )
  {
    v93 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v92 )
    {
      v120 = 88;
      v119 = 0;
      v127 = 0;
      goto LABEL_115;
    }
    if ( v92 > 0x1745D1745D1745DLL )
      v92 = 0x1745D1745D1745DLL;
    v93 = 88 * v92;
  }
  v111 = (char *)((char *)v58 - (char *)*v55);
  v114 = v55;
  v94 = sub_22077B0(v93);
  v55 = v114;
  v127 = v94;
  v88 = (__int64)v111;
  v119 = v94 + v93;
  v120 = v94 + 88;
LABEL_115:
  v21 = v127 + v88 == 0;
  v95 = (_QWORD *)(v127 + v88);
  v96 = v95;
  if ( !v21 )
  {
    *v95 = v95 + 2;
    v95[1] = 0x400000000LL;
    if ( (_DWORD)v154 )
    {
      v113 = v55;
      v118 = v95;
      sub_1B7D490((__int64)v95, &v153, (__int64)v95, (__int64)v95, v53, v54);
      v55 = v113;
      v96 = v118;
    }
    v96[4] = v96 + 6;
    v96[5] = 0x400000000LL;
    if ( (_DWORD)v157 )
    {
      v112 = v55;
      v117 = v96;
      sub_1B7CFB0((__int64)(v96 + 4), (char **)&v156, (__int64)v96, (__int64)v95, v53, v54);
      v55 = v112;
      v96 = v117;
    }
    v96[10] = v159;
  }
  v97 = (__int64)v123;
  if ( v58 != v123 )
  {
    v98 = v7;
    v115 = v6;
    v99 = v58;
    v100 = (__int64)v123;
    v121 = v55;
    v101 = v127;
    v102 = v98;
    while ( 1 )
    {
      if ( v101 )
      {
        *(_DWORD *)(v101 + 8) = 0;
        *(_QWORD *)v101 = v101 + 16;
        *(_DWORD *)(v101 + 12) = 4;
        if ( *(_DWORD *)(v100 + 8) )
          sub_1B7D5D0(v101, v100, (__int64)v96, v97, v53, v54);
        *(_DWORD *)(v101 + 40) = 0;
        *(_QWORD *)(v101 + 32) = v101 + 48;
        *(_DWORD *)(v101 + 44) = 4;
        if ( *(_DWORD *)(v100 + 40) )
          sub_1B7D3B0(v101 + 32, v100 + 32, (__int64)v96, v97, v53, v54);
        *(_QWORD *)(v101 + 80) = *(_QWORD *)(v100 + 80);
      }
      v100 += 88;
      if ( v99 == (unsigned __int64 *)v100 )
        break;
      v101 += 88;
    }
    v103 = v102;
    v104 = v99;
    v105 = v121;
    v6 = v115;
    v120 = v101 + 176;
    v116 = v103;
    v106 = v123;
    v107 = v105;
    do
    {
      v108 = v106[4];
      if ( (unsigned __int64 *)v108 != v106 + 6 )
        _libc_free(v108);
      if ( (unsigned __int64 *)*v106 != v106 + 2 )
        _libc_free(*v106);
      v106 += 11;
    }
    while ( v104 != v106 );
    v7 = v116;
    v55 = v107;
  }
  v109 = v123;
  if ( v123 )
  {
    v110 = v123;
    v124 = v55;
    j_j___libc_free_0(v110, (char *)v55[2] - (char *)v109);
    v55 = v124;
  }
  *v55 = (unsigned __int64 *)v127;
  v55[1] = (unsigned __int64 *)v120;
  v55[2] = (unsigned __int64 *)v119;
LABEL_53:
  if ( v156 != (__int64 **)v158 )
  {
    v139 = v55;
    _libc_free((unsigned __int64)v156);
    v55 = v139;
  }
  if ( v153 != v155 )
  {
    v140 = v55;
    _libc_free((unsigned __int64)v153);
    v55 = v140;
  }
  v21 = *(_BYTE *)(a1 + 184) == 0;
  *(_QWORD *)(a1 + 176) = v134;
  if ( v21 )
    *(_BYTE *)(a1 + 184) = 1;
  v60 = 0x2E8BA2E8BA2E8BA3LL * (v55[1] - *v55);
  if ( v7 == v131 )
  {
    if ( v60 == *(_QWORD *)(a1 + 160) )
      goto LABEL_61;
  }
  else if ( v60 == *(_QWORD *)(a1 + 168) )
  {
LABEL_61:
    sub_1B81290(a1, (__int64 *)v6, a4, a5, v55, v130);
  }
}
