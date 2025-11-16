// Function: sub_2C515C0
// Address: 0x2c515c0
//
unsigned __int8 *__fastcall sub_2C515C0(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8)
{
  int v8; // r14d
  __int64 **v11; // rbx
  __int64 v12; // r15
  __int64 v13; // r8
  __int64 v14; // r9
  _QWORD *v15; // rax
  _QWORD *v16; // rdx
  unsigned __int8 *v17; // r13
  __int64 *v19; // rax
  _QWORD *v20; // rax
  _QWORD *v21; // rdx
  unsigned __int64 v22; // rbx
  _BYTE *v23; // rax
  _BYTE *v24; // r12
  __int64 v25; // rax
  __int64 v26; // rdi
  _BYTE *v27; // r14
  __int64 (__fastcall *v28)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64); // rax
  unsigned __int8 **v29; // rdi
  __int64 *v30; // rax
  _QWORD *v31; // rax
  _QWORD *v32; // rdx
  unsigned __int64 v33; // rcx
  unsigned __int64 v34; // r12
  unsigned int v35; // eax
  int v36; // edx
  __int64 v37; // r8
  unsigned int v38; // ecx
  int v39; // eax
  __int64 v40; // rsi
  __int64 v41; // rdx
  size_t v42; // r12
  _BYTE *v43; // rbx
  _BYTE *v44; // rsi
  __int64 v45; // rax
  __int64 v46; // rdx
  unsigned __int64 v47; // rbx
  unsigned int v48; // edx
  __int64 v49; // rax
  unsigned int v50; // edx
  unsigned int v51; // edx
  unsigned int v52; // eax
  __int64 v53; // rcx
  unsigned int v54; // ebx
  __int64 v55; // rax
  __int64 v56; // r11
  void *v57; // r10
  __int64 v58; // r14
  int v59; // r12d
  __int64 v60; // rdi
  __int64 v61; // rax
  _BYTE *v62; // r12
  _BYTE *v63; // r13
  __int64 (__fastcall *v64)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64); // rax
  _QWORD *v65; // rax
  __int64 v66; // r13
  __int64 v67; // r12
  __int64 v68; // rdx
  unsigned int v69; // esi
  __int64 *v70; // rax
  unsigned __int8 *v71; // rax
  _QWORD *v72; // r11
  __int64 v73; // r8
  _BYTE *v74; // rdx
  char *v75; // rax
  char *i; // rdx
  _QWORD *v77; // r15
  __int64 v78; // r12
  __int64 v79; // rax
  __int64 v80; // rax
  __int64 **v81; // r12
  __int64 v82; // r13
  __int64 v83; // rax
  unsigned __int64 v84; // rdx
  int v85; // esi
  __int64 v86; // rdi
  __int64 **v87; // rax
  __int64 v88; // r8
  __int64 v89; // rsi
  int v90; // edx
  unsigned __int8 *v91; // rax
  __int64 v92; // rax
  __int64 v93; // rdx
  __int64 v94; // rbx
  _BYTE *v95; // rdi
  __int64 v96; // rdx
  unsigned __int8 *v97; // rax
  __int64 v98; // r9
  __int64 v99; // rbx
  __int64 v100; // r12
  __int64 v101; // rdx
  unsigned int v102; // esi
  char *v103; // rax
  char *v104; // rdx
  __int64 v105; // rcx
  __int64 v106; // rdx
  unsigned int v107; // eax
  unsigned int v108; // eax
  unsigned int v109; // edx
  __int64 v110; // rsi
  __int64 v111; // rdx
  __int64 v112; // rax
  __int64 v113; // rdi
  unsigned __int8 *v114; // r14
  unsigned int v115; // r12d
  __int64 (__fastcall *v116)(__int64, unsigned int, _BYTE *); // rax
  __int64 v117; // rbx
  __int64 v118; // r12
  __int64 v119; // rdx
  unsigned int v120; // esi
  __int64 v121; // rdx
  int v122; // r12d
  __int64 v123; // [rsp-10h] [rbp-1D0h]
  int v124; // [rsp+28h] [rbp-198h]
  int v125; // [rsp+28h] [rbp-198h]
  unsigned int v126; // [rsp+38h] [rbp-188h]
  unsigned __int8 *v127; // [rsp+38h] [rbp-188h]
  unsigned int s; // [rsp+40h] [rbp-180h]
  unsigned int sa; // [rsp+40h] [rbp-180h]
  _QWORD *sb; // [rsp+40h] [rbp-180h]
  __int64 v131; // [rsp+48h] [rbp-178h]
  __int64 v132; // [rsp+48h] [rbp-178h]
  void *v133; // [rsp+48h] [rbp-178h]
  __int64 v134; // [rsp+48h] [rbp-178h]
  __int64 v135; // [rsp+48h] [rbp-178h]
  __int64 v136; // [rsp+48h] [rbp-178h]
  void *v137; // [rsp+48h] [rbp-178h]
  __int64 v138; // [rsp+50h] [rbp-170h]
  void *v139; // [rsp+50h] [rbp-170h]
  __int64 v140; // [rsp+50h] [rbp-170h]
  _QWORD *v141; // [rsp+50h] [rbp-170h]
  __int64 v142; // [rsp+50h] [rbp-170h]
  _BYTE *v144; // [rsp+58h] [rbp-168h]
  __int64 v145; // [rsp+58h] [rbp-168h]
  _QWORD *v146; // [rsp+58h] [rbp-168h]
  int v147; // [rsp+58h] [rbp-168h]
  char v148[32]; // [rsp+60h] [rbp-160h] BYREF
  __int16 v149; // [rsp+80h] [rbp-140h]
  __int64 v150[4]; // [rsp+90h] [rbp-130h] BYREF
  __int16 v151; // [rsp+B0h] [rbp-110h]
  void *dest; // [rsp+C0h] [rbp-100h] BYREF
  __int64 v153; // [rsp+C8h] [rbp-F8h]
  _BYTE v154[16]; // [rsp+D0h] [rbp-F0h] BYREF
  __int16 v155; // [rsp+E0h] [rbp-E0h]
  void *src; // [rsp+100h] [rbp-C0h] BYREF
  __int64 v157; // [rsp+108h] [rbp-B8h]
  _BYTE v158[16]; // [rsp+110h] [rbp-B0h] BYREF
  __int16 v159; // [rsp+120h] [rbp-A0h]
  void *v160; // [rsp+140h] [rbp-80h] BYREF
  __int64 v161; // [rsp+148h] [rbp-78h]
  _BYTE v162[112]; // [rsp+150h] [rbp-70h] BYREF

  v8 = a4;
  v11 = (__int64 **)a1;
  v12 = (__int64)a7;
  v13 = *(_QWORD *)a1;
  v14 = *(unsigned int *)(a1 + 8);
  if ( !*(_BYTE *)(a4 + 28) )
  {
    s = *(_DWORD *)(a1 + 8);
    v131 = *(_QWORD *)a1;
    v19 = sub_C8CA60(a4, v13);
    v13 = v131;
    v14 = s;
    if ( !v19 )
      goto LABEL_9;
    return *(unsigned __int8 **)v13;
  }
  v15 = *(_QWORD **)(a4 + 8);
  v16 = &v15[*(unsigned int *)(a4 + 20)];
  if ( v15 != v16 )
  {
    while ( v13 != *v15 )
    {
      if ( v16 == ++v15 )
        goto LABEL_9;
    }
    return *(unsigned __int8 **)v13;
  }
LABEL_9:
  if ( !*(_BYTE *)(a5 + 28) )
  {
    sa = v14;
    v132 = v13;
    v30 = sub_C8CA60(a5, v13);
    v13 = v132;
    v14 = sa;
    if ( !v30 )
      goto LABEL_27;
LABEL_14:
    v22 = *(unsigned int *)(a3 + 32);
    v160 = v162;
    v161 = 0x1000000000LL;
    if ( (unsigned int)v22 > 0x10 )
    {
      v142 = v13;
      v147 = v14;
      sub_C8D5F0((__int64)&v160, v162, v22, 4u, v13, v14);
      v103 = (char *)v160;
      v13 = v142;
      v104 = (char *)v160 + 4 * v22;
      do
      {
        *(_DWORD *)v103 = v147;
        v103 += 4;
      }
      while ( v104 != v103 );
      LODWORD(v161) = v22;
      v144 = v160;
    }
    else
    {
      v23 = v162;
      v144 = v162;
      if ( v22 )
      {
        do
        {
          *(_DWORD *)v23 = v14;
          v23 += 4;
        }
        while ( &v162[4 * v22] != v23 );
        v144 = v160;
      }
      LODWORD(v161) = v22;
    }
    v155 = 257;
    v24 = *(_BYTE **)v13;
    v25 = sub_ACADE0(*(__int64 ***)(*(_QWORD *)v13 + 8LL));
    v26 = a7[10];
    v27 = (_BYTE *)v25;
    v28 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)v26 + 112LL);
    if ( v28 == sub_9B6630 )
    {
      if ( *v24 > 0x15u || *v27 > 0x15u )
      {
LABEL_132:
        v159 = 257;
        v97 = (unsigned __int8 *)sub_BD2C40(112, unk_3F1FE60);
        v17 = v97;
        if ( v97 )
        {
          sub_B4E9E0((__int64)v97, (__int64)v24, (__int64)v27, v144, v22, (__int64)&src, 0, 0);
          v98 = v123;
        }
        (*(void (__fastcall **)(__int64, unsigned __int8 *, void **, __int64, __int64, __int64))(*(_QWORD *)a7[11] + 16LL))(
          a7[11],
          v17,
          &dest,
          a7[7],
          a7[8],
          v98);
        v99 = *a7;
        v100 = *a7 + 16LL * *((unsigned int *)a7 + 2);
        if ( *a7 != v100 )
        {
          do
          {
            v101 = *(_QWORD *)(v99 + 8);
            v102 = *(_DWORD *)v99;
            v99 += 16;
            sub_B99FD0((__int64)v17, v102, v101);
          }
          while ( v100 != v99 );
        }
LABEL_24:
        v29 = (unsigned __int8 **)v160;
        if ( v160 == v162 )
          return v17;
LABEL_25:
        _libc_free((unsigned __int64)v29);
        return v17;
      }
      v17 = (unsigned __int8 *)sub_AD5CE0((__int64)v24, (__int64)v27, v144, v22, 0);
    }
    else
    {
      v17 = (unsigned __int8 *)((__int64 (__fastcall *)(__int64, _BYTE *, _BYTE *, _BYTE *, unsigned __int64))v28)(
                                 v26,
                                 v24,
                                 v27,
                                 v144,
                                 v22);
    }
    if ( v17 )
      goto LABEL_24;
    goto LABEL_132;
  }
  v20 = *(_QWORD **)(a5 + 8);
  v21 = &v20[*(unsigned int *)(a5 + 20)];
  if ( v20 != v21 )
  {
    while ( v13 != *v20 )
    {
      if ( v21 == ++v20 )
        goto LABEL_27;
    }
    goto LABEL_14;
  }
LABEL_27:
  if ( *(_BYTE *)(a6 + 28) )
  {
    v31 = *(_QWORD **)(a6 + 8);
    v32 = &v31[*(unsigned int *)(a6 + 20)];
    if ( v31 == v32 )
      goto LABEL_72;
    while ( v13 != *v31 )
    {
      if ( v32 == ++v31 )
        goto LABEL_72;
    }
    goto LABEL_32;
  }
  v135 = v13;
  v70 = sub_C8CA60(a6, v13);
  v13 = v135;
  if ( v70 )
  {
LABEL_32:
    v33 = *(unsigned int *)(*(_QWORD *)(*(_QWORD *)v13 + 8LL) + 32LL);
    v126 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v13 + 8LL) + 32LL);
    v34 = a2 / v33;
    dest = v154;
    v153 = 0x600000000LL;
    if ( a2 / v33 > 6 )
    {
      sub_C8D5F0((__int64)&dest, v154, v34, 8u, v13, v14);
      v29 = (unsigned __int8 **)((char *)dest + 8 * v34);
      if ( dest != v29 )
      {
        memset(dest, 0, 8 * v34);
        v29 = (unsigned __int8 **)dest;
      }
      LODWORD(v153) = v34;
      v36 = v34;
    }
    else
    {
      if ( v33 <= a2 )
      {
        v35 = 8 * v34;
        if ( 8 * v34 )
        {
          if ( v35 >= 8 )
          {
            v106 = v35;
            v107 = v35 - 1;
            *(_QWORD *)&v154[v106 - 8] = 0;
            if ( v107 >= 8 )
            {
              v108 = v107 & 0xFFFFFFF8;
              v109 = 0;
              do
              {
                v110 = v109;
                v109 += 8;
                *(_QWORD *)&v154[v110] = 0;
              }
              while ( v109 < v108 );
            }
          }
          else if ( v35 )
          {
            v154[0] = 0;
          }
        }
      }
      LODWORD(v153) = v34;
      v29 = (unsigned __int8 **)v154;
      v36 = v34;
    }
    v37 = v126;
    v38 = 0;
    v39 = 0;
    v40 = 0;
    if ( !v36 )
    {
LABEL_123:
      v17 = *v29;
      if ( v29 == (unsigned __int8 **)v154 )
        return v17;
      goto LABEL_25;
    }
    while ( 1 )
    {
      v41 = v38;
      v38 += v126;
      v29[v40] = (unsigned __int8 *)*v11[2 * v41];
      v40 = (unsigned int)(v39 + 1);
      v39 = v40;
      if ( (unsigned int)v153 <= (unsigned int)v40 )
        break;
      v29 = (unsigned __int8 **)dest;
    }
    if ( (unsigned int)v153 <= 1 )
      goto LABEL_122;
LABEL_44:
    v126 *= 2;
    v160 = v162;
    v161 = 0x1000000000LL;
    v42 = 4LL * v126;
    if ( v126 > 0x10uLL )
    {
      sub_C8D5F0((__int64)&v160, v162, v126, 4u, v37, v14);
      memset(v160, 0, v42);
      v44 = v160;
      LODWORD(v161) = v126;
      v43 = (char *)v160 + v42;
    }
    else
    {
      v43 = &v162[v42];
      if ( v126 && v43 != v162 )
        memset(v162, 0, 4LL * v126);
      v44 = v162;
      LODWORD(v161) = v126;
    }
    if ( v44 != v43 )
    {
      v45 = 0;
      do
      {
        v46 = v45;
        *(_DWORD *)&v44[4 * v45] = v45;
        ++v45;
      }
      while ( (unsigned __int64)(v43 - 4 - v44) >> 2 != v46 );
    }
    src = v158;
    v157 = 0x600000000LL;
    v47 = (unsigned __int64)(unsigned int)v153 >> 1;
    if ( (unsigned int)v153 > 0xDuLL )
    {
      sub_C8D5F0((__int64)&src, v158, (unsigned __int64)(unsigned int)v153 >> 1, 8u, v37, v14);
      memset(src, 0, 8 * v47);
      LODWORD(v157) = v47;
    }
    else
    {
      if ( !v47 )
        goto LABEL_125;
      v48 = 8 * v47;
      if ( 8 * v47 )
      {
        v49 = v48;
        v50 = v48 - 1;
        *(_QWORD *)&v158[v49 - 8] = 0;
        if ( v50 >= 8 )
        {
          v51 = v50 & 0xFFFFFFF8;
          v52 = 0;
          do
          {
            v53 = v52;
            v52 += 8;
            *(_QWORD *)&v158[v53] = 0;
          }
          while ( v52 < v51 );
        }
      }
      LODWORD(v157) = v47;
    }
    v145 = 0;
    v54 = 0;
    while ( 1 )
    {
      v60 = a7[10];
      v61 = 2 * v54;
      v149 = 257;
      v57 = v160;
      v62 = (_BYTE *)*((_QWORD *)dest + v61);
      v56 = (unsigned int)v161;
      v63 = (_BYTE *)*((_QWORD *)dest + (unsigned int)(v61 + 1));
      v64 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)v60 + 112LL);
      if ( v64 == sub_9B6630 )
      {
        if ( *v62 > 0x15u || *v63 > 0x15u )
        {
LABEL_66:
          v134 = v56;
          v151 = 257;
          v139 = v57;
          v65 = sub_BD2C40(112, unk_3F1FE60);
          v58 = (__int64)v65;
          if ( v65 )
            sub_B4E9E0((__int64)v65, (__int64)v62, (__int64)v63, v139, v134, (__int64)v150, 0, 0);
          (*(void (__fastcall **)(__int64, __int64, char *, __int64, __int64))(*(_QWORD *)a7[11] + 16LL))(
            a7[11],
            v58,
            v148,
            a7[7],
            a7[8]);
          v66 = *a7;
          v67 = *a7 + 16LL * *((unsigned int *)a7 + 2);
          if ( *a7 != v67 )
          {
            do
            {
              v68 = *(_QWORD *)(v66 + 8);
              v69 = *(_DWORD *)v66;
              v66 += 16;
              sub_B99FD0(v58, v69, v68);
            }
            while ( v67 != v66 );
          }
          goto LABEL_63;
        }
        v133 = v160;
        v138 = (unsigned int)v161;
        v55 = sub_AD5CE0((__int64)v62, (__int64)v63, v160, (unsigned int)v161, 0);
        v56 = v138;
        v57 = v133;
        v58 = v55;
      }
      else
      {
        v137 = v160;
        v140 = (unsigned int)v161;
        v92 = v64(v60, v62, v63, (__int64)v160, (unsigned int)v161, v14);
        v57 = v137;
        v56 = v140;
        v58 = v92;
      }
      if ( !v58 )
        goto LABEL_66;
LABEL_63:
      *((_QWORD *)src + v145) = v58;
      v59 = v157;
      v145 = ++v54;
      if ( v54 >= (unsigned int)v157 )
      {
        v93 = (unsigned int)v157;
        if ( (unsigned int)v157 <= (unsigned __int64)(unsigned int)v153 )
        {
          if ( (_DWORD)v157 )
            memmove(dest, src, 8LL * (unsigned int)v157);
          else
LABEL_125:
            v59 = 0;
          LODWORD(v153) = v59;
          v95 = src;
          goto LABEL_117;
        }
        if ( (unsigned int)v157 > (unsigned __int64)HIDWORD(v153) )
        {
          v94 = 0;
          LODWORD(v153) = 0;
          sub_C8D5F0((__int64)&dest, v154, (unsigned int)v157, 8u, v37, v14);
          v93 = (unsigned int)v157;
        }
        else
        {
          v94 = 8LL * (unsigned int)v153;
          if ( (_DWORD)v153 )
          {
            memmove(dest, src, 8LL * (unsigned int)v153);
            v93 = (unsigned int)v157;
          }
        }
        v95 = src;
        v96 = 8 * v93;
        if ( (char *)src + v94 != (char *)src + v96 )
        {
          memcpy((char *)dest + v94, (char *)src + v94, v96 - v94);
          v95 = src;
        }
        LODWORD(v153) = v59;
LABEL_117:
        if ( v95 != v158 )
          _libc_free((unsigned __int64)v95);
        if ( v160 != v162 )
          _libc_free((unsigned __int64)v160);
        if ( (unsigned int)v153 <= 1 )
        {
LABEL_122:
          v29 = (unsigned __int8 **)dest;
          goto LABEL_123;
        }
        goto LABEL_44;
      }
    }
  }
LABEL_72:
  v71 = *(unsigned __int8 **)v13;
  v72 = 0;
  v127 = v71;
  v73 = *(_DWORD *)(*(_QWORD *)v13 + 4LL) & 0x7FFFFFF;
  if ( *v71 == 85 )
  {
    v72 = (_QWORD *)*((_QWORD *)v71 - 4);
    if ( v72 )
    {
      if ( !*(_BYTE *)v72 && v72[3] == *((_QWORD *)v71 + 10) && (*((_BYTE *)v72 + 33) & 0x20) != 0 )
      {
        v72 = v71;
        v73 = (unsigned int)(v73 - 1);
      }
      else
      {
        v72 = 0;
      }
    }
  }
  v136 = (unsigned int)v73;
  src = v158;
  v157 = 0x600000000LL;
  if ( (_DWORD)v73 )
  {
    v74 = v158;
    v75 = v158;
    if ( (unsigned int)v73 > 6uLL )
    {
      v125 = v73;
      sb = v72;
      sub_C8D5F0((__int64)&src, v158, (unsigned int)v73, 8u, v73, v14);
      v74 = src;
      LODWORD(v73) = v125;
      v72 = sb;
      v75 = (char *)src + 8 * (unsigned int)v157;
    }
    for ( i = &v74[8 * v136]; i != v75; v75 += 8 )
    {
      if ( v75 )
        *(_QWORD *)v75 = 0;
    }
    v77 = v72;
    LODWORD(v157) = v73;
    v124 = a5;
    v78 = 0;
    do
    {
      if ( !v77 )
        goto LABEL_86;
      v79 = *(v77 - 4);
      if ( !v79 || *(_BYTE *)v79 || *(_QWORD *)(v79 + 24) != v77[10] )
        BUG();
      if ( sub_9B75A0(*(unsigned int *)(v79 + 36), v78, a8) )
      {
        *((_QWORD *)src + v78) = v77[4 * (v78 - (*((_DWORD *)v77 + 1) & 0x7FFFFFF))];
      }
      else
      {
LABEL_86:
        sub_2C4FBF0((__int64)&v160, a1, a2, v78);
        v80 = sub_2C515C0((_DWORD)v160, v161, a3, v8, v124, a6, (__int64)a7, a8);
        *((_QWORD *)src + v78) = v80;
        if ( v160 != v162 )
          _libc_free((unsigned __int64)v160);
      }
      ++v78;
    }
    while ( v78 != v136 );
    v72 = v77;
    v12 = (__int64)a7;
  }
  v160 = v162;
  v81 = (__int64 **)(a1 + 16 * a2);
  v161 = 0x800000000LL;
  if ( v81 != (__int64 **)a1 )
  {
    do
    {
      if ( *v11 )
      {
        v82 = **v11;
        v83 = (unsigned int)v161;
        v84 = (unsigned int)v161 + 1LL;
        if ( v84 > HIDWORD(v161) )
        {
          v141 = v72;
          sub_C8D5F0((__int64)&v160, v162, v84, 8u, v73, v14);
          v83 = (unsigned int)v161;
          v72 = v141;
        }
        *((_QWORD *)v160 + v83) = v82;
        LODWORD(v161) = v161 + 1;
      }
      v11 += 2;
    }
    while ( v81 != v11 );
  }
  v85 = *(_DWORD *)(a3 + 32);
  v86 = *((_QWORD *)v127 + 1);
  if ( (unsigned int)*(unsigned __int8 *)(v86 + 8) - 17 <= 1 )
    v86 = **(_QWORD **)(v86 + 16);
  v146 = v72;
  v87 = (__int64 **)sub_BCDA70((__int64 *)v86, v85);
  v89 = (__int64)v87;
  v90 = *v127;
  if ( (unsigned int)(v90 - 42) <= 0x11 )
  {
    v155 = 257;
    v91 = (unsigned __int8 *)sub_2C51350(
                               (__int64 *)v12,
                               *v127 - 29,
                               *(unsigned __int8 **)src,
                               *((unsigned __int8 **)src + 1),
                               v150[0],
                               0,
                               (__int64)&dest,
                               0);
LABEL_100:
    v17 = v91;
    sub_F70480(v91, (unsigned __int8 **)v160, (unsigned int)v161, 0, 1);
    goto LABEL_101;
  }
  v105 = (unsigned int)(v90 - 82);
  if ( (unsigned __int8)(v90 - 82) > 1u )
  {
    if ( (_BYTE)v90 == 86 )
    {
      v155 = 257;
      v17 = (unsigned __int8 *)sub_B36550(
                                 (unsigned int **)v12,
                                 *(_QWORD *)src,
                                 *((_QWORD *)src + 1),
                                 *((_QWORD *)src + 2),
                                 (__int64)&dest,
                                 (__int64)v127);
      goto LABEL_143;
    }
    v111 = (unsigned int)(v90 - 67);
    if ( (unsigned int)v111 <= 0xC )
    {
      v155 = 257;
      v91 = (unsigned __int8 *)sub_2C511B0(
                                 (__int64 *)v12,
                                 (unsigned int)*v127 - 29,
                                 *(_QWORD *)src,
                                 v87,
                                 (__int64)&dest,
                                 0,
                                 v150[0],
                                 0);
      goto LABEL_100;
    }
    if ( v146 )
    {
      v155 = 257;
      HIDWORD(v150[0]) = 0;
      v112 = *(v146 - 4);
      if ( !v112 || *(_BYTE *)v112 || *(_QWORD *)(v112 + 24) != v146[10] )
        BUG();
      v91 = (unsigned __int8 *)sub_B35180(
                                 v12,
                                 v89,
                                 *(_DWORD *)(v112 + 36),
                                 (__int64)src,
                                 (unsigned int)v157,
                                 v150[0],
                                 (__int64)&dest);
      goto LABEL_100;
    }
    v113 = *(_QWORD *)(v12 + 80);
    v151 = 257;
    v114 = *(unsigned __int8 **)src;
    v115 = *v127 - 29;
    v116 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *))(*(_QWORD *)v113 + 48LL);
    if ( v116 == sub_9288C0 )
    {
      if ( *v114 > 0x15u )
        goto LABEL_167;
      v17 = (unsigned __int8 *)sub_AAAFF0(v115, *(unsigned __int8 **)src, v111, v105, v88);
    }
    else
    {
      v17 = (unsigned __int8 *)((__int64 (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD))v116)(
                                 v113,
                                 v115,
                                 *(_QWORD *)src,
                                 *(unsigned int *)(v12 + 104));
    }
    if ( v17 )
      goto LABEL_143;
LABEL_167:
    v155 = 257;
    v17 = (unsigned __int8 *)sub_B50340(v115, (__int64)v114, (__int64)&dest, 0, 0);
    if ( (unsigned __int8)sub_920620((__int64)v17) )
    {
      v121 = *(_QWORD *)(v12 + 96);
      v122 = *(_DWORD *)(v12 + 104);
      if ( v121 )
        sub_B99FD0((__int64)v17, 3u, v121);
      sub_B45150((__int64)v17, v122);
    }
    (*(void (__fastcall **)(_QWORD, unsigned __int8 *, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v12 + 88) + 16LL))(
      *(_QWORD *)(v12 + 88),
      v17,
      v150,
      *(_QWORD *)(v12 + 56),
      *(_QWORD *)(v12 + 64));
    v117 = *(_QWORD *)v12;
    v118 = *(_QWORD *)v12 + 16LL * *(unsigned int *)(v12 + 8);
    if ( *(_QWORD *)v12 != v118 )
    {
      do
      {
        v119 = *(_QWORD *)(v117 + 8);
        v120 = *(_DWORD *)v117;
        v117 += 16;
        sub_B99FD0((__int64)v17, v120, v119);
      }
      while ( v118 != v117 );
    }
    goto LABEL_143;
  }
  v155 = 257;
  v17 = (unsigned __int8 *)sub_2B22A00(
                             v12,
                             *((_WORD *)v127 + 1) & 0x3F,
                             *(_QWORD *)src,
                             *((_QWORD *)src + 1),
                             (__int64)&dest,
                             0);
LABEL_143:
  sub_F70480(v17, (unsigned __int8 **)v160, (unsigned int)v161, 0, 1);
LABEL_101:
  if ( v160 != v162 )
    _libc_free((unsigned __int64)v160);
  v29 = (unsigned __int8 **)src;
  if ( src != v158 )
    goto LABEL_25;
  return v17;
}
