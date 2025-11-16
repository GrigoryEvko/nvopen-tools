// Function: sub_1BF8840
// Address: 0x1bf8840
//
__int64 __fastcall sub_1BF8840(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        unsigned __int8 a4,
        unsigned __int8 a5,
        __m128i a6,
        __m128i a7)
{
  __int64 v7; // r10
  __int64 *v8; // r15
  __int16 v11; // ax
  __int64 *v12; // r12
  __int64 *v14; // rax
  __int64 v15; // r13
  __int64 v16; // rbx
  __int16 v17; // ax
  unsigned int v18; // r14d
  unsigned int v19; // ebx
  __int64 v20; // rax
  __int64 v21; // rcx
  __int16 v22; // ax
  __int64 v23; // r13
  __int64 v24; // rax
  int v25; // r8d
  int v26; // r9d
  __int64 v27; // r15
  __int64 v28; // rax
  unsigned int v29; // edx
  __int64 v30; // r13
  __int16 v31; // ax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // r15
  __int64 v35; // r14
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rcx
  __int64 v39; // rdx
  unsigned int v40; // r8d
  __int64 v41; // rsi
  __int64 v42; // r10
  unsigned __int8 v43; // r8
  unsigned int v44; // r13d
  __int64 v45; // r15
  __int64 v46; // r12
  __int64 v47; // rax
  __int64 v48; // r14
  __int64 v49; // rax
  int v50; // r8d
  int v51; // r9d
  __int64 v52; // r13
  __int64 v53; // rax
  __int64 *v54; // rax
  __int64 v55; // r10
  __int64 v56; // r15
  unsigned __int64 v57; // rbx
  __int64 v58; // rax
  __int64 v59; // r8
  int v60; // r9d
  __int64 v61; // rax
  __int64 *v62; // rdi
  char *v63; // rdx
  char *v64; // rcx
  __int64 v65; // r10
  unsigned __int8 v66; // r8
  char *v67; // rdx
  __int64 v68; // rbx
  unsigned int v69; // r14d
  char v70; // r15
  __int64 v71; // r13
  int v72; // r8d
  int v73; // r9d
  __int64 v74; // r11
  __int64 v75; // rax
  char v76; // r13
  __int64 *v77; // rdi
  __int64 v78; // rax
  __int64 v79; // rax
  __int64 v80; // r10
  unsigned __int8 v81; // r8
  __int64 v82; // r15
  __int64 v83; // r15
  __int64 v84; // r14
  __int64 v85; // r10
  unsigned __int8 v86; // r8
  __int64 v87; // r13
  __int64 v88; // r15
  __int64 v89; // rax
  __int64 v90; // r8
  int v91; // r9d
  __int64 v92; // rax
  __int64 v93; // rax
  __int64 v94; // rax
  __int64 v95; // rax
  __int64 v96; // rax
  __int64 v97; // r10
  __int64 v98; // r15
  __int64 v99; // r15
  __int64 v100; // [rsp+8h] [rbp-F8h]
  __int64 v101; // [rsp+8h] [rbp-F8h]
  __int64 v102; // [rsp+8h] [rbp-F8h]
  __int64 v103; // [rsp+10h] [rbp-F0h]
  __int64 v104; // [rsp+10h] [rbp-F0h]
  __int64 v105; // [rsp+10h] [rbp-F0h]
  char v107; // [rsp+18h] [rbp-E8h]
  __int64 v108; // [rsp+20h] [rbp-E0h]
  __int64 v110; // [rsp+20h] [rbp-E0h]
  __int64 v111; // [rsp+20h] [rbp-E0h]
  unsigned int v113; // [rsp+20h] [rbp-E0h]
  unsigned __int8 v114; // [rsp+20h] [rbp-E0h]
  __int64 v115; // [rsp+20h] [rbp-E0h]
  unsigned __int8 v116; // [rsp+28h] [rbp-D8h]
  __int64 v118; // [rsp+30h] [rbp-D0h]
  unsigned __int8 v119; // [rsp+30h] [rbp-D0h]
  __int64 v121; // [rsp+38h] [rbp-C8h]
  __int64 v122; // [rsp+38h] [rbp-C8h]
  __int64 v123; // [rsp+38h] [rbp-C8h]
  __int64 v124; // [rsp+38h] [rbp-C8h]
  __int64 v126; // [rsp+38h] [rbp-C8h]
  __int64 v127; // [rsp+38h] [rbp-C8h]
  unsigned int v128; // [rsp+38h] [rbp-C8h]
  __int64 *v129; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v130; // [rsp+48h] [rbp-B8h]
  __int64 v131; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v132; // [rsp+58h] [rbp-A8h]
  char *v133; // [rsp+70h] [rbp-90h] BYREF
  __int64 v134; // [rsp+78h] [rbp-88h]
  char v135[32]; // [rsp+80h] [rbp-80h] BYREF
  __int64 *v136; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v137; // [rsp+A8h] [rbp-58h]
  __int64 v138; // [rsp+B0h] [rbp-50h] BYREF
  __int64 v139; // [rsp+B8h] [rbp-48h]

  v7 = a3;
  v8 = (__int64 *)a1;
  v11 = *(_WORD *)(a1 + 24);
  if ( v11 != 5 )
  {
    if ( v11 == 3 )
    {
      v30 = *(_QWORD *)(a1 + 32);
      if ( a5 && !sub_1BF8240(*(_QWORD *)(a1 + 32), a3, a4) )
        return a1;
      v31 = *(_WORD *)(v30 + 24);
      if ( v31 != 7 )
      {
        if ( v31 == 4 )
        {
          if ( !a4 && (*(_BYTE *)(v30 + 26) & 5) == 0 )
            return a1;
          v116 = a5;
          v136 = &v138;
          v129 = &v131;
          v130 = 0x400000000LL;
          v133 = v135;
          v134 = 0x400000000LL;
          v137 = 0x400000000LL;
          v123 = v7;
          sub_1BF7FA0(
            (__int64)&v129,
            (char *)&v131,
            *(char **)(v30 + 32),
            (char *)(*(_QWORD *)(v30 + 32) + 8LL * *(_QWORD *)(v30 + 40)));
          v55 = v123;
          if ( (_DWORD)v130 )
          {
            v124 = 8LL * (unsigned int)v130;
            v56 = v55;
            v57 = 0;
            do
            {
              v58 = sub_147B0D0((__int64)a2, v129[v57 / 8], *(_QWORD *)(a1 + 40), 0);
              v59 = sub_1BF8840(v58, a2, v56, a4, v116);
              v61 = (unsigned int)v137;
              if ( (unsigned int)v137 >= HIDWORD(v137) )
              {
                v101 = v59;
                sub_16CD150((__int64)&v136, &v138, 0, 8, v59, v60);
                v61 = (unsigned int)v137;
                v59 = v101;
              }
              v57 += 8LL;
              v136[v61] = v59;
              LODWORD(v137) = v137 + 1;
            }
            while ( v124 != v57 );
          }
          v12 = sub_147DD40((__int64)a2, (__int64 *)&v136, *(_WORD *)(v30 + 26) & 7, 0, a6, a7);
          if ( v136 != &v138 )
            _libc_free((unsigned __int64)v136);
          v62 = v129;
          if ( v129 == &v131 )
            return (__int64)v12;
LABEL_71:
          _libc_free((unsigned __int64)v62);
          return (__int64)v12;
        }
        if ( v31 != 5 || !a4 && (*(_BYTE *)(v30 + 26) & 5) == 0 )
          return a1;
        v114 = a5;
        v133 = v135;
        v134 = 0x400000000LL;
        v136 = &v138;
        v137 = 0x400000000LL;
        v127 = v7;
        sub_1BF7FA0(
          (__int64)&v133,
          v135,
          *(char **)(v30 + 32),
          (char *)(*(_QWORD *)(v30 + 32) + 8LL * *(_QWORD *)(v30 + 40)));
        v85 = v127;
        v86 = v114;
        if ( (_DWORD)v134 )
        {
          v105 = v30;
          v115 = 8LL * (unsigned int)v134;
          v128 = v86;
          v87 = 0;
          v88 = v85;
          do
          {
            v89 = sub_147B0D0((__int64)a2, *(_QWORD *)&v133[v87], *(_QWORD *)(a1 + 40), 0);
            v90 = sub_1BF8840(v89, a2, v88, a4, v128);
            v92 = (unsigned int)v137;
            if ( (unsigned int)v137 >= HIDWORD(v137) )
            {
              v102 = v90;
              sub_16CD150((__int64)&v136, &v138, 0, 8, v90, v91);
              v92 = (unsigned int)v137;
              v90 = v102;
            }
            v87 += 8;
            v136[v92] = v90;
            LODWORD(v137) = v137 + 1;
          }
          while ( v115 != v87 );
          v93 = sub_147EE30(a2, &v136, *(_WORD *)(v105 + 26) & 7, 0, a6, a7);
        }
        else
        {
          v93 = sub_147EE30(a2, &v136, *(_WORD *)(v30 + 26) & 7, 0, a6, a7);
        }
        v77 = v136;
        v12 = (__int64 *)v93;
        if ( v136 == &v138 )
        {
LABEL_70:
          v62 = (__int64 *)v133;
          if ( v133 == v135 )
            return (__int64)v12;
          goto LABEL_71;
        }
LABEL_69:
        _libc_free((unsigned __int64)v77);
        goto LABEL_70;
      }
      if ( v7 != *(_QWORD *)(v30 + 48) || !a4 && (*(_BYTE *)(v30 + 26) & 5) == 0 )
        return a1;
      v119 = a5;
      v122 = v7;
      v32 = sub_147B0D0((__int64)a2, **(_QWORD **)(v30 + 32), *(_QWORD *)(a1 + 40), 0);
      v33 = sub_1BF8840(v32, a2, v122, a4, v119);
      v34 = *(_QWORD *)(a1 + 40);
      v35 = v33;
      v36 = sub_13A5BC0((_QWORD *)v30, (__int64)a2);
      v37 = sub_147B0D0((__int64)a2, v36, v34, 0);
      v38 = *(_QWORD *)(v30 + 48);
      v39 = v37;
      v40 = *(_WORD *)(v30 + 26) & 7;
LABEL_26:
      v41 = v35;
      return sub_14799E0((__int64)a2, v41, v39, v38, v40);
    }
    if ( v11 != 4 )
      return a1;
    v63 = *(char **)(a1 + 32);
    v134 = 0x400000000LL;
    v137 = 0x400000000LL;
    v64 = &v63[8 * *(_QWORD *)(a1 + 40)];
    v133 = v135;
    v136 = &v138;
    sub_1BF7FA0((__int64)&v133, v135, v63, v64);
    v65 = a3;
    v66 = a5;
    if ( (_DWORD)v134 )
    {
      v126 = 8LL * (unsigned int)v134;
      v67 = v133;
      v113 = a4;
      v68 = 0;
      v69 = v66;
      v70 = 0;
      v71 = v65;
      do
      {
        v74 = sub_1BF8840(*(_QWORD *)&v67[v68], a2, v71, v113, v69);
        v75 = (unsigned int)v137;
        if ( (unsigned int)v137 >= HIDWORD(v137) )
        {
          v100 = v74;
          sub_16CD150((__int64)&v136, &v138, 0, 8, v72, v73);
          v75 = (unsigned int)v137;
          v74 = v100;
        }
        v136[v75] = v74;
        v67 = v133;
        LODWORD(v137) = v137 + 1;
        if ( v74 != *(_QWORD *)&v133[v68] )
          v70 = 1;
        v68 += 8;
      }
      while ( v126 != v68 );
      v76 = v70;
      v8 = (__int64 *)a1;
      if ( v76 )
      {
        v29 = *(_WORD *)(a1 + 26) & 7;
LABEL_18:
        v12 = sub_147DD40((__int64)a2, (__int64 *)&v136, v29, 0, a6, a7);
        goto LABEL_68;
      }
    }
    goto LABEL_67;
  }
  if ( *(_QWORD *)(a1 + 40) != 2 )
    return a1;
  v14 = *(__int64 **)(a1 + 32);
  v15 = v14[1];
  v16 = *v14;
  v17 = *(_WORD *)(v15 + 24);
  v121 = v16;
  if ( v17 == 3 )
  {
    v18 = a4;
    if ( a5 && !sub_1BF8240(*(_QWORD *)(v15 + 32), a3, a4) )
      return a1;
    v19 = a5;
    v118 = v7;
    v20 = sub_1BF8840(v15, a2, v7, v18, a5);
    v103 = v20;
    v21 = v20;
    if ( v15 == v20 )
      return a1;
    v22 = *(_WORD *)(v20 + 24);
    if ( v22 == 4 )
    {
      v133 = v135;
      v134 = 0x400000000LL;
      v136 = &v138;
      v137 = 0x400000000LL;
      sub_1BF7FA0(
        (__int64)&v133,
        v135,
        *(char **)(v21 + 32),
        (char *)(*(_QWORD *)(v21 + 32) + 8LL * *(_QWORD *)(v21 + 40)));
      if ( (_DWORD)v134 )
      {
        v23 = 0;
        v108 = 8LL * (unsigned int)v134;
        do
        {
          v24 = *(_QWORD *)&v133[v23];
          v129 = &v131;
          v131 = v121;
          v132 = v24;
          v130 = 0x200000002LL;
          v27 = sub_147EE30(a2, &v129, 0, 0, a6, a7);
          if ( v129 != &v131 )
            _libc_free((unsigned __int64)v129);
          v28 = (unsigned int)v137;
          if ( (unsigned int)v137 >= HIDWORD(v137) )
          {
            sub_16CD150((__int64)&v136, &v138, 0, 8, v25, v26);
            v28 = (unsigned int)v137;
          }
          v23 += 8;
          v136[v28] = v27;
          LODWORD(v137) = v137 + 1;
        }
        while ( v23 != v108 );
      }
      v29 = *(_WORD *)(v103 + 26) & 7;
      goto LABEL_18;
    }
    if ( v22 != 7 || v118 != *(_QWORD *)(v103 + 48) || !sub_146CEE0((__int64)a2, v121, v118) )
    {
      v138 = v121;
      v136 = &v138;
      v139 = v103;
      v137 = 0x200000002LL;
      v94 = sub_147EE30(a2, &v136, 0, 0, a6, a7);
      v62 = v136;
      v12 = (__int64 *)v94;
      if ( v136 == &v138 )
        return (__int64)v12;
      goto LABEL_71;
    }
    v95 = **(_QWORD **)(v103 + 32);
    v138 = v121;
    v136 = &v138;
    v139 = v95;
    v137 = 0x200000002LL;
    v96 = sub_147EE30(a2, &v136, 0, 0, a6, a7);
    v97 = v118;
    v98 = v96;
    if ( v136 != &v138 )
    {
      _libc_free((unsigned __int64)v136);
      v97 = v118;
    }
    v35 = sub_1BF8840(v98, a2, v97, v18, v19);
    v139 = sub_13A5BC0((_QWORD *)v103, (__int64)a2);
    v138 = v121;
    v136 = &v138;
    v137 = 0x200000002LL;
    v99 = sub_147EE30(a2, &v136, 0, 0, a6, a7);
    if ( v136 != &v138 )
      _libc_free((unsigned __int64)v136);
    v39 = v99;
    v38 = *(_QWORD *)(v103 + 48);
    v40 = *(_WORD *)(v103 + 26) & 7;
    goto LABEL_26;
  }
  if ( v17 == 4 )
  {
    v133 = v135;
    v134 = 0x400000000LL;
    v136 = &v138;
    v137 = 0x400000000LL;
    sub_1BF7FA0(
      (__int64)&v133,
      v135,
      *(char **)(v15 + 32),
      (char *)(*(_QWORD *)(v15 + 32) + 8LL * *(_QWORD *)(v15 + 40)));
    v42 = a3;
    v43 = a5;
    if ( (_DWORD)v134 )
    {
      v107 = 0;
      v104 = v15;
      v44 = v43;
      v110 = 8LL * (unsigned int)v134;
      v45 = 0;
      v46 = v42;
      do
      {
        v47 = sub_1BF8840(*(_QWORD *)&v133[v45], a2, v46, a4, v44);
        if ( *(_QWORD *)&v133[v45] != v47 )
        {
          *(_QWORD *)&v133[v45] = v47;
          v107 = 1;
        }
        v45 += 8;
      }
      while ( v110 != v45 );
      v8 = (__int64 *)a1;
      if ( v107 )
      {
        if ( (_DWORD)v134 )
        {
          v111 = 8LL * (unsigned int)v134;
          v48 = 0;
          do
          {
            v49 = *(_QWORD *)&v133[v48];
            v129 = &v131;
            v131 = v16;
            v132 = v49;
            v130 = 0x200000002LL;
            v52 = sub_147EE30(a2, &v129, 0, 0, a6, a7);
            if ( v129 != &v131 )
              _libc_free((unsigned __int64)v129);
            v53 = (unsigned int)v137;
            if ( (unsigned int)v137 >= HIDWORD(v137) )
            {
              sub_16CD150((__int64)&v136, &v138, 0, 8, v50, v51);
              v53 = (unsigned int)v137;
            }
            v48 += 8;
            v136[v53] = v52;
            LODWORD(v137) = v137 + 1;
          }
          while ( v111 != v48 );
          v54 = sub_147DD40((__int64)a2, (__int64 *)&v136, *(_WORD *)(v104 + 26) & 7, 0, a6, a7);
        }
        else
        {
          v54 = sub_147DD40((__int64)a2, (__int64 *)&v136, *(_WORD *)(v104 + 26) & 7, 0, a6, a7);
        }
        v12 = v54;
LABEL_68:
        v77 = v136;
        if ( v136 == &v138 )
          goto LABEL_70;
        goto LABEL_69;
      }
    }
LABEL_67:
    v12 = v8;
    goto LABEL_68;
  }
  if ( v17 != 7 || a3 != *(_QWORD *)(v15 + 48) || !sub_146CEE0((__int64)a2, v16, a3) )
    return a1;
  v78 = **(_QWORD **)(v15 + 32);
  v138 = v16;
  v139 = v78;
  v136 = &v138;
  v137 = 0x200000002LL;
  v79 = sub_147EE30(a2, &v136, 0, 0, a6, a7);
  v80 = a3;
  v81 = a5;
  v82 = v79;
  if ( v136 != &v138 )
  {
    _libc_free((unsigned __int64)v136);
    v81 = a5;
    v80 = a3;
  }
  v83 = sub_1BF8840(v82, a2, v80, a4, v81);
  v139 = sub_13A5BC0((_QWORD *)v15, (__int64)a2);
  v138 = v16;
  v136 = &v138;
  v137 = 0x200000002LL;
  v84 = sub_147EE30(a2, &v136, 0, 0, a6, a7);
  if ( v136 != &v138 )
    _libc_free((unsigned __int64)v136);
  v38 = *(_QWORD *)(v15 + 48);
  v39 = v84;
  v41 = v83;
  v40 = *(_WORD *)(v15 + 26) & 7;
  return sub_14799E0((__int64)a2, v41, v39, v38, v40);
}
