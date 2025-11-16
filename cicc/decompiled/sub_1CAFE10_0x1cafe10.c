// Function: sub_1CAFE10
// Address: 0x1cafe10
//
__int64 __fastcall sub_1CAFE10(
        __int64 a1,
        __m128 si128,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9,
        __int64 a10,
        size_t a11,
        __int64 a12,
        int a13,
        int a14)
{
  __int64 v14; // rdi
  __int64 v15; // rbx
  __int64 v16; // r12
  __int64 v17; // r15
  __int64 v18; // rbx
  _BYTE *v19; // rsi
  _BYTE *v20; // rdx
  _BYTE *v21; // rax
  __int64 v22; // rdx
  char v23; // r15
  const char *v24; // rdx
  __int64 v25; // rbx
  unsigned int v26; // eax
  __int64 v27; // r13
  bool v28; // r12
  __int64 i; // r14
  int v30; // r8d
  int v31; // r9d
  _BYTE *v32; // r13
  unsigned __int8 v33; // al
  const char *v34; // rdi
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 **v37; // r14
  _BYTE *v38; // rsi
  bool v39; // r12
  __int64 v40; // rax
  _BYTE *v41; // rsi
  __int64 v42; // rax
  __int64 *v43; // r14
  __int64 v44; // rax
  __int64 v45; // r13
  _QWORD *v46; // rax
  double v47; // xmm4_8
  double v48; // xmm5_8
  _QWORD *v49; // r12
  __int64 v50; // rax
  __int64 *v51; // rax
  __int64 *v52; // rax
  int v53; // r8d
  __int64 *v54; // r10
  __int64 *v55; // rax
  __int64 v56; // rdx
  __int64 *v57; // rax
  __int64 *v58; // r14
  __int64 v59; // rsi
  __int64 v60; // rsi
  unsigned __int8 *v61; // rsi
  char v62; // bl
  __int64 v63; // rdi
  __int64 v64; // rdi
  __int64 *v66; // rax
  void *v67; // rax
  __m128 *v68; // rdx
  __int64 v69; // r12
  const char *v70; // rax
  _BYTE *v71; // rdi
  char *v72; // rsi
  size_t v73; // r13
  unsigned __int64 v74; // rax
  void *v75; // rax
  __m128 *v76; // rdx
  __int64 v77; // rdi
  __int64 v78; // rax
  _WORD *v79; // rdx
  __int64 v80; // rdi
  __int64 v81; // rax
  __int64 v82; // rdx
  __int64 v83; // r12
  unsigned __int64 v84; // rdx
  const char *v85; // r14
  size_t v86; // r13
  __int64 v87; // rax
  const char *v88; // rdx
  size_t v89; // rdx
  char *v90; // rsi
  __int64 v91; // r12
  __int64 v92; // rax
  const char *v93; // rax
  size_t v94; // rdx
  void *v95; // rdi
  char *v96; // rsi
  size_t v97; // r13
  unsigned __int64 v98; // rax
  __int64 v99; // rax
  __int64 v100; // rdi
  _BYTE *v101; // rax
  __int64 v102; // rax
  char *v103; // rdi
  __int64 v104; // rax
  __int64 v105; // rax
  __int64 *v107; // [rsp+10h] [rbp-180h]
  unsigned __int8 v108; // [rsp+18h] [rbp-178h]
  __int64 *v109; // [rsp+18h] [rbp-178h]
  int v110; // [rsp+18h] [rbp-178h]
  __int64 v111; // [rsp+28h] [rbp-168h]
  __int64 v112; // [rsp+38h] [rbp-158h]
  __int64 v113; // [rsp+50h] [rbp-140h]
  __int64 v114; // [rsp+58h] [rbp-138h]
  bool v115; // [rsp+67h] [rbp-129h]
  bool v116; // [rsp+67h] [rbp-129h]
  unsigned __int8 v117; // [rsp+68h] [rbp-128h]
  __int64 v118; // [rsp+68h] [rbp-128h]
  __int64 *v119; // [rsp+78h] [rbp-118h] BYREF
  _BYTE *v120; // [rsp+80h] [rbp-110h] BYREF
  _BYTE *v121; // [rsp+88h] [rbp-108h]
  _BYTE *v122; // [rsp+90h] [rbp-100h]
  __int64 *v123; // [rsp+A0h] [rbp-F0h] BYREF
  _BYTE *v124; // [rsp+A8h] [rbp-E8h]
  _BYTE *v125; // [rsp+B0h] [rbp-E0h]
  __int64 v126; // [rsp+C0h] [rbp-D0h] BYREF
  __int64 v127; // [rsp+C8h] [rbp-C8h]
  __int64 v128; // [rsp+D0h] [rbp-C0h]
  int v129; // [rsp+D8h] [rbp-B8h]
  __int64 *v130; // [rsp+E0h] [rbp-B0h] BYREF
  int v131; // [rsp+E8h] [rbp-A8h] BYREF
  __int64 v132; // [rsp+F0h] [rbp-A0h]
  int *v133; // [rsp+F8h] [rbp-98h]
  int *v134; // [rsp+100h] [rbp-90h]
  __int64 v135; // [rsp+108h] [rbp-88h]
  const char *v136; // [rsp+110h] [rbp-80h] BYREF
  __int64 v137; // [rsp+118h] [rbp-78h]
  _QWORD v138[14]; // [rsp+120h] [rbp-70h] BYREF

  if ( byte_4FBE620 )
  {
    v67 = sub_16E8CB0();
    v68 = (__m128 *)*((_QWORD *)v67 + 3);
    v69 = (__int64)v67;
    if ( *((_QWORD *)v67 + 2) - (_QWORD)v68 <= 0x14u )
    {
      v69 = sub_16E7EE0((__int64)v67, "Normalizing function ", 0x15u);
    }
    else
    {
      si128 = (__m128)_mm_load_si128((const __m128i *)&xmmword_42DFE50);
      v68[1].m128_i32[0] = 1852795252;
      v68[1].m128_i8[4] = 32;
      *v68 = si128;
      *((_QWORD *)v67 + 3) += 21LL;
    }
    v70 = sub_1649960(a1);
    v71 = *(_BYTE **)(v69 + 24);
    v72 = (char *)v70;
    v73 = a11;
    v74 = *(_QWORD *)(v69 + 16) - (_QWORD)v71;
    if ( v74 < a11 )
    {
      v102 = sub_16E7EE0(v69, v72, a11);
      v71 = *(_BYTE **)(v102 + 24);
      v69 = v102;
      if ( *(_QWORD *)(v102 + 16) - (_QWORD)v71 > 4u )
      {
LABEL_99:
        *(_DWORD *)v71 = 774778400;
        v71[4] = 10;
        *(_QWORD *)(v69 + 24) += 5LL;
        goto LABEL_2;
      }
    }
    else
    {
      if ( a11 )
      {
        memcpy(v71, v72, a11);
        v99 = *(_QWORD *)(v69 + 16);
        v71 = (_BYTE *)(v73 + *(_QWORD *)(v69 + 24));
        *(_QWORD *)(v69 + 24) = v71;
        v74 = v99 - (_QWORD)v71;
      }
      if ( v74 > 4 )
        goto LABEL_99;
    }
    sub_16E7EE0(v69, " ...\n", 5u);
  }
LABEL_2:
  v137 = 0x2000000000LL;
  v136 = (const char *)v138;
  sub_1CAD270(a1, (__int64)&v136, a11, a12, a13, a14);
  v14 = *(_QWORD *)(a1 + 40);
  LOWORD(v132) = 261;
  v126 = (__int64)v136;
  v130 = &v126;
  v127 = (unsigned int)v137;
  v15 = sub_1632310(v14, (__int64)&v130);
  if ( v136 != (const char *)v138 )
    _libc_free((unsigned __int64)v136);
  v120 = 0;
  v121 = 0;
  v16 = *(_QWORD *)(a1 + 80);
  v122 = 0;
  v126 = 0;
  v127 = 0;
  v128 = 0;
  v129 = 0;
  v115 = v15 != 0;
  if ( v16 == a1 + 72 )
  {
    v108 = 0;
    v64 = 0;
    goto LABEL_87;
  }
  do
  {
    if ( !v16 )
      BUG();
    v17 = *(_QWORD *)(v16 + 24);
    v18 = v16 + 16;
    if ( v17 != v16 + 16 )
    {
      while ( 1 )
      {
        if ( !v17 )
          BUG();
        if ( *(_BYTE *)(v17 - 8) != 56 )
          goto LABEL_8;
        v136 = (const char *)(v17 - 24);
        if ( (unsigned __int8)sub_1CAF920(
                                *(_QWORD *)(v17 - 24LL * (*(_DWORD *)(v17 - 4) & 0xFFFFFFF) - 24),
                                (__int64)&v126) )
          goto LABEL_8;
        v19 = v121;
        if ( v121 == v122 )
        {
          sub_1CADA60((__int64)&v120, v121, &v136);
LABEL_8:
          v17 = *(_QWORD *)(v17 + 8);
          if ( v18 == v17 )
            break;
        }
        else
        {
          if ( v121 )
          {
            *(_QWORD *)v121 = v136;
            v19 = v121;
          }
          v121 = v19 + 8;
          v17 = *(_QWORD *)(v17 + 8);
          if ( v18 == v17 )
            break;
        }
      }
    }
    v16 = *(_QWORD *)(v16 + 8);
  }
  while ( a1 + 72 != v16 );
  v20 = v121;
  v21 = v120;
  if ( v121 != v120 )
  {
    if ( !byte_4FBE620 )
      goto LABEL_19;
    v75 = sub_16E8CB0();
    v76 = (__m128 *)*((_QWORD *)v75 + 3);
    v77 = (__int64)v75;
    if ( *((_QWORD *)v75 + 2) - (_QWORD)v76 <= 0x14u )
    {
      v77 = sub_16E7EE0((__int64)v75, "Normalize GEP Index (", 0x15u);
    }
    else
    {
      si128 = (__m128)_mm_load_si128((const __m128i *)&xmmword_42DFE60);
      v76[1].m128_i32[0] = 544761188;
      v76[1].m128_i8[4] = 40;
      *v76 = si128;
      *((_QWORD *)v75 + 3) += 21LL;
    }
    v78 = sub_16E7A90(v77, (v121 - v120) >> 3);
    v79 = *(_WORD **)(v78 + 24);
    v80 = v78;
    if ( *(_QWORD *)(v78 + 16) - (_QWORD)v79 <= 1u )
    {
      v80 = sub_16E7EE0(v78, ", ", 2u);
    }
    else
    {
      *v79 = 8236;
      *(_QWORD *)(v78 + 24) += 2LL;
    }
    v81 = sub_16E7AB0(v80, v115);
    v82 = *(_QWORD *)(v81 + 24);
    v83 = v81;
    if ( (unsigned __int64)(*(_QWORD *)(v81 + 16) - v82) <= 4 )
    {
      v83 = sub_16E7EE0(v81, " ) : ", 5u);
    }
    else
    {
      *(_DWORD *)v82 = 975186208;
      *(_BYTE *)(v82 + 4) = 32;
      *(_QWORD *)(v81 + 24) += 5LL;
    }
    v85 = sub_1649960(a1);
    v86 = v84;
    if ( !v85 )
    {
      v137 = 0;
      v89 = 0;
      LOBYTE(v138[0]) = 0;
      v136 = (const char *)v138;
      v90 = (char *)v138;
      goto LABEL_121;
    }
    v130 = (__int64 *)v84;
    v87 = v84;
    v136 = (const char *)v138;
    if ( v84 > 0xF )
    {
      v136 = (const char *)sub_22409D0(&v136, &v130, 0);
      v103 = (char *)v136;
      v138[0] = v130;
    }
    else
    {
      if ( v84 == 1 )
      {
        LOBYTE(v138[0]) = *v85;
        v88 = (const char *)v138;
LABEL_110:
        v137 = v87;
        v88[v87] = 0;
        v89 = v137;
        v90 = (char *)v136;
LABEL_121:
        v100 = sub_16E7EE0(v83, v90, v89);
        v101 = *(_BYTE **)(v100 + 24);
        if ( *(_BYTE **)(v100 + 16) == v101 )
        {
          sub_16E7EE0(v100, "\n", 1u);
        }
        else
        {
          *v101 = 10;
          ++*(_QWORD *)(v100 + 24);
        }
        if ( v136 != (const char *)v138 )
          j_j___libc_free_0(v136, v138[0] + 1LL);
        v20 = v121;
        v21 = v120;
LABEL_19:
        v131 = 0;
        v22 = (v20 - v21) >> 3;
        v133 = &v131;
        v132 = 0;
        v134 = &v131;
        v135 = 0;
        if ( !(_DWORD)v22 )
        {
          v108 = 0;
          v63 = 0;
LABEL_86:
          sub_1CAD430(v63);
          v64 = v127;
          goto LABEL_87;
        }
        v108 = 0;
        v23 = v115;
        v114 = 0;
        v111 = 8LL * (unsigned int)(v22 - 1);
        while ( 2 )
        {
          v24 = (const char *)v138;
          v25 = *(_QWORD *)&v21[v114];
          v138[0] = v25;
          v136 = (const char *)v138;
          v137 = 0x800000001LL;
          v26 = 1;
          while ( 1 )
          {
            v27 = *(_QWORD *)&v24[8 * v26 - 8];
            LODWORD(v137) = v26 - 1;
            v28 = sub_1642F90(*(_QWORD *)(v25 + 56), 8);
            if ( !v28 )
              break;
            for ( i = *(_QWORD *)(v27 + 8); i; i = *(_QWORD *)(i + 8) )
            {
              v32 = sub_1648700(i);
              v33 = v32[16];
              if ( v33 <= 0x17u )
                goto LABEL_27;
              if ( v33 == 56 )
              {
                v42 = (unsigned int)v137;
                if ( (unsigned int)v137 >= HIDWORD(v137) )
                {
                  sub_16CD150((__int64)&v136, v138, 0, 8, v30, v31);
                  v42 = (unsigned int)v137;
                }
                *(_QWORD *)&v136[8 * v42] = v32;
                LODWORD(v137) = v137 + 1;
              }
              else if ( v33 != 71 || *(_BYTE *)(*(_QWORD *)v32 + 8LL) != 15 )
              {
                goto LABEL_27;
              }
            }
            v34 = v136;
            v26 = v137;
            v24 = v136;
            if ( !(_DWORD)v137 )
              goto LABEL_28;
          }
LABEL_27:
          v34 = v136;
          v28 = 0;
LABEL_28:
          if ( v34 != (const char *)v138 )
            _libc_free((unsigned __int64)v34);
          v35 = *(_QWORD *)(v25 - 24LL * (*(_DWORD *)(v25 + 20) & 0xFFFFFFF));
          v125 = 0;
          v123 = 0;
          v113 = v35;
          v36 = *(_DWORD *)(v25 + 20) & 0xFFFFFFF;
          v124 = 0;
          v37 = (__int64 **)(v25 + 24 * (1 - v36));
          if ( (__int64 **)v25 == v37 )
          {
LABEL_47:
            if ( v111 != v114 )
            {
              v21 = v120;
              v114 += 8;
              continue;
            }
            if ( v108 && (v62 = byte_4FBE620) != 0 )
            {
              v91 = (__int64)sub_16E8CB0();
              v92 = *(_QWORD *)(v91 + 24);
              if ( (unsigned __int64)(*(_QWORD *)(v91 + 16) - v92) <= 8 )
              {
                v91 = sub_16E7EE0(v91, "Function ", 9u);
              }
              else
              {
                *(_BYTE *)(v92 + 8) = 32;
                *(_QWORD *)v92 = 0x6E6F6974636E7546LL;
                *(_QWORD *)(v91 + 24) += 9LL;
              }
              v93 = sub_1649960(a1);
              v95 = *(void **)(v91 + 24);
              v96 = (char *)v93;
              v97 = v94;
              v98 = *(_QWORD *)(v91 + 16) - (_QWORD)v95;
              if ( v94 > v98 )
              {
                v104 = sub_16E7EE0(v91, v96, v94);
                v95 = *(void **)(v104 + 24);
                v91 = v104;
                v98 = *(_QWORD *)(v104 + 16) - (_QWORD)v95;
              }
              else if ( v94 )
              {
                memcpy(v95, v96, v94);
                v105 = *(_QWORD *)(v91 + 16);
                v95 = (void *)(v97 + *(_QWORD *)(v91 + 24));
                *(_QWORD *)(v91 + 24) = v95;
                v98 = v105 - (_QWORD)v95;
              }
              if ( v98 <= 0xE )
              {
                sub_16E7EE0(v91, " is normalized\n", 0xFu);
              }
              else
              {
                qmemcpy(v95, " is normalized\n", 15);
                *(_QWORD *)(v91 + 24) += 15LL;
              }
              v108 = v62;
              v63 = v132;
            }
            else
            {
              v63 = v132;
            }
            goto LABEL_86;
          }
          break;
        }
        v116 = 0;
        v117 = v28;
        do
        {
          while ( 1 )
          {
            v119 = *v37;
            v39 = sub_1642F90(*v119, 64);
            if ( !v39 )
              break;
            v40 = sub_1CADBF0((_QWORD ***)v119, &v130, v117, v23);
            v136 = (const char *)v40;
            if ( !v40 )
              break;
            v41 = v124;
            if ( v124 == v125 )
            {
              sub_1287830((__int64)&v123, v124, &v136);
            }
            else
            {
              if ( v124 )
              {
                *(_QWORD *)v124 = v40;
                v41 = v124;
              }
              v124 = v41 + 8;
            }
            v37 += 3;
            v116 = v39;
            if ( (__int64 **)v25 == v37 )
              goto LABEL_44;
          }
          v38 = v124;
          if ( v124 == v125 )
          {
            sub_1287830((__int64)&v123, v124, &v119);
          }
          else
          {
            if ( v124 )
            {
              *(_QWORD *)v124 = v119;
              v38 = v124;
            }
            v124 = v38 + 8;
          }
          v37 += 3;
        }
        while ( (__int64 **)v25 != v37 );
LABEL_44:
        if ( !v116 )
        {
LABEL_45:
          if ( v123 )
            j_j___libc_free_0(v123, v125 - (_BYTE *)v123);
          goto LABEL_47;
        }
        v43 = v123;
        v136 = "newGep";
        LOWORD(v138[0]) = 259;
        v118 = (v124 - (_BYTE *)v123) >> 3;
        v44 = *(_QWORD *)v113;
        if ( *(_BYTE *)(*(_QWORD *)v113 + 8LL) == 16 )
          v44 = **(_QWORD **)(v44 + 16);
        v109 = (__int64 *)v124;
        v45 = *(_QWORD *)(v44 + 24);
        v46 = sub_1648A60(72, (int)v118 + 1);
        v49 = v46;
        if ( v46 )
        {
          v112 = (__int64)&v46[-3 * (unsigned int)(v118 + 1)];
          v50 = *(_QWORD *)v113;
          if ( *(_BYTE *)(*(_QWORD *)v113 + 8LL) == 16 )
            v50 = **(_QWORD **)(v50 + 16);
          v107 = v109;
          v110 = *(_DWORD *)(v50 + 8) >> 8;
          v51 = (__int64 *)sub_15F9F50(v45, (__int64)v43, v118);
          v52 = (__int64 *)sub_1646BA0(v51, v110);
          v53 = v118 + 1;
          v54 = v52;
          if ( *(_BYTE *)(*(_QWORD *)v113 + 8LL) == 16 )
          {
            v66 = sub_16463B0(v52, *(_QWORD *)(*(_QWORD *)v113 + 32LL));
            v53 = v118 + 1;
            v54 = v66;
          }
          else if ( v107 != v43 )
          {
            v55 = v43;
            while ( 1 )
            {
              v56 = *(_QWORD *)*v55;
              if ( *(_BYTE *)(v56 + 8) == 16 )
                break;
              if ( v107 == ++v55 )
                goto LABEL_69;
            }
            v57 = sub_16463B0(v54, *(_QWORD *)(v56 + 32));
            v53 = v118 + 1;
            v54 = v57;
          }
LABEL_69:
          sub_15F1EA0((__int64)v49, (__int64)v54, 32, v112, v53, v25);
          v49[7] = v45;
          v49[8] = sub_15F9F50(v45, (__int64)v43, v118);
          sub_15F9CE0((__int64)v49, v113, v43, v118, (__int64)&v136);
        }
        v58 = v49 + 6;
        sub_164D160(v25, (__int64)v49, si128, a3, a4, a5, v47, v48, a8, a9);
        v59 = *(_QWORD *)(v25 + 48);
        v136 = (const char *)v59;
        if ( v59 )
        {
          sub_1623A60((__int64)&v136, v59, 2);
          if ( v58 == (__int64 *)&v136 )
          {
            if ( v136 )
              sub_161E7C0((__int64)&v136, (__int64)v136);
            goto LABEL_74;
          }
          v60 = v49[6];
          if ( !v60 )
          {
LABEL_81:
            v61 = (unsigned __int8 *)v136;
            v49[6] = v136;
            if ( v61 )
              sub_1623210((__int64)&v136, v61, (__int64)(v49 + 6));
            goto LABEL_74;
          }
        }
        else if ( v58 == (__int64 *)&v136 || (v60 = v49[6]) == 0 )
        {
LABEL_74:
          v108 = v116;
          goto LABEL_45;
        }
        sub_161E7C0((__int64)(v49 + 6), v60);
        goto LABEL_81;
      }
      if ( !v84 )
      {
        v88 = (const char *)v138;
        goto LABEL_110;
      }
      v103 = (char *)v138;
    }
    memcpy(v103, v85, v86);
    v87 = (__int64)v130;
    v88 = v136;
    goto LABEL_110;
  }
  v108 = 0;
  v64 = v127;
LABEL_87:
  j___libc_free_0(v64);
  if ( v120 )
    j_j___libc_free_0(v120, v122 - v120);
  return v108;
}
