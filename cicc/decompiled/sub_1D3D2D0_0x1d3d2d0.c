// Function: sub_1D3D2D0
// Address: 0x1d3d2d0
//
__int64 *__fastcall sub_1D3D2D0(
        __int64 *a1,
        __int64 a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        unsigned __int64 a6,
        __m128 a7,
        __m128i a8,
        __m128i a9,
        __int64 a10,
        unsigned __int64 a11,
        unsigned __int64 a12,
        unsigned int a13,
        char a14,
        char a15,
        __int64 a16,
        __int64 a17,
        __int64 a18,
        __int64 a19,
        __int64 a20,
        __int64 a21)
{
  __int64 *v21; // r13
  __int64 *v23; // r15
  __int64 v24; // r14
  __int64 v25; // rax
  __int64 v26; // r12
  __int64 v27; // rax
  __int64 v28; // rsi
  __int64 v29; // rax
  bool v30; // bl
  int v31; // eax
  unsigned int v32; // eax
  int v33; // eax
  unsigned int v34; // r14d
  unsigned int v35; // r13d
  unsigned int v36; // eax
  unsigned int v37; // r8d
  __int64 v38; // rdx
  __int64 v39; // rsi
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // rax
  __int64 v43; // rdx
  unsigned __int64 v44; // rbx
  __int64 v45; // rcx
  __int64 v46; // rdx
  unsigned int v47; // eax
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // r9
  unsigned __int64 v52; // r14
  unsigned __int8 v53; // al
  unsigned __int8 v54; // dl
  __int64 v55; // r13
  unsigned int v56; // edx
  __int64 v57; // r12
  unsigned __int64 v58; // rdx
  __int64 *v59; // rax
  __int64 v60; // rdx
  __int64 v61; // r12
  __int64 v62; // rax
  __int64 v63; // r13
  __int64 *v64; // rax
  __int64 v65; // r13
  unsigned int v66; // r12d
  unsigned __int64 v67; // rdx
  char v68; // al
  unsigned __int16 v69; // r9
  unsigned __int64 v70; // rdx
  __int64 *v71; // rax
  __int64 v72; // rdx
  int v73; // r8d
  int v74; // r9d
  __int64 v75; // r13
  unsigned int v76; // edx
  __int64 v77; // r12
  __int64 v78; // rax
  __int64 *v79; // rax
  unsigned __int64 v80; // rdx
  __int64 *v81; // rax
  __int64 v82; // rdx
  __int64 v83; // r12
  __int64 v84; // rax
  __int64 v85; // r13
  __int64 *v86; // rax
  unsigned int v87; // r12d
  __int64 v88; // r13
  unsigned int v89; // eax
  unsigned int v90; // edi
  unsigned int v91; // r15d
  unsigned int v92; // r15d
  unsigned int v93; // eax
  int v94; // ebx
  unsigned int v95; // r12d
  unsigned __int64 v96; // rax
  __int64 v97; // rax
  __int64 (*v98)(); // rcx
  unsigned int v99; // ebx
  __int64 (__fastcall *v100)(__int64); // rax
  __int64 v101; // rax
  unsigned int v102; // esi
  char v103; // bl
  unsigned int v104; // r15d
  unsigned __int64 v105; // rax
  __int64 v106; // rax
  bool v107; // zf
  __int64 v108; // rax
  __int64 *v109; // rax
  __int64 v110; // rax
  __int64 v111; // rax
  _QWORD *v112; // rcx
  __int64 v113; // r12
  __int64 v114; // rbx
  __int64 v115; // rax
  const __m128i *v116; // r14
  const __m128i *v117; // r14
  unsigned int v118; // edx
  __int64 v119; // rsi
  __int64 v120; // rdi
  unsigned int v121; // r13d
  __int64 (*v122)(); // rax
  unsigned int v123; // eax
  __int64 v124; // rax
  __int64 v125; // rax
  unsigned int v126; // edx
  __int64 v127; // rax
  unsigned int v128; // eax
  unsigned int v129; // edx
  unsigned int v130; // r12d
  unsigned int v131; // ebx
  unsigned int v132; // r14d
  unsigned int v133; // r8d
  unsigned int v134; // ecx
  unsigned int v135; // r12d
  char v136; // r8
  unsigned __int8 v137; // al
  unsigned int v138; // r13d
  __int64 v139; // r15
  __int64 v140; // rax
  const void **v141; // r8
  __int64 v142; // rcx
  __int128 v143; // rax
  unsigned int v144; // edx
  const void **v145; // rdx
  __int128 v146; // [rsp-10h] [rbp-640h]
  __int64 v147; // [rsp+0h] [rbp-630h]
  unsigned int v148; // [rsp+20h] [rbp-610h]
  unsigned __int64 v149; // [rsp+20h] [rbp-610h]
  __int64 v150; // [rsp+30h] [rbp-600h]
  unsigned __int16 v151; // [rsp+38h] [rbp-5F8h]
  unsigned int v152; // [rsp+40h] [rbp-5F0h]
  unsigned int v153; // [rsp+44h] [rbp-5ECh]
  unsigned __int128 v156; // [rsp+60h] [rbp-5D0h]
  __int64 v157; // [rsp+70h] [rbp-5C0h]
  __int64 v158; // [rsp+78h] [rbp-5B8h]
  unsigned int v159; // [rsp+78h] [rbp-5B8h]
  char v160; // [rsp+80h] [rbp-5B0h]
  char v161; // [rsp+87h] [rbp-5A9h]
  __int64 v162; // [rsp+90h] [rbp-5A0h]
  char v163; // [rsp+98h] [rbp-598h]
  __int64 *v164; // [rsp+98h] [rbp-598h]
  unsigned int v165; // [rsp+98h] [rbp-598h]
  __int64 v166; // [rsp+A0h] [rbp-590h]
  unsigned int v167; // [rsp+A0h] [rbp-590h]
  _DWORD *v168; // [rsp+A8h] [rbp-588h]
  __int64 v170; // [rsp+B8h] [rbp-578h]
  __m128i v171; // [rsp+F0h] [rbp-540h] BYREF
  __m128i v172; // [rsp+100h] [rbp-530h] BYREF
  __int64 v173; // [rsp+110h] [rbp-520h] BYREF
  __int64 v174; // [rsp+118h] [rbp-518h]
  __int64 v175; // [rsp+120h] [rbp-510h]
  __int64 v176; // [rsp+130h] [rbp-500h] BYREF
  __int64 v177; // [rsp+138h] [rbp-4F8h]
  unsigned __int64 v178; // [rsp+140h] [rbp-4F0h]
  __int128 v179; // [rsp+150h] [rbp-4E0h]
  __int64 v180; // [rsp+160h] [rbp-4D0h]
  __int128 v181; // [rsp+170h] [rbp-4C0h]
  __int64 v182; // [rsp+180h] [rbp-4B0h]
  __int128 v183; // [rsp+190h] [rbp-4A0h]
  __int64 v184; // [rsp+1A0h] [rbp-490h]
  unsigned __int64 v185; // [rsp+1B0h] [rbp-480h] BYREF
  __int64 v186; // [rsp+1B8h] [rbp-478h]
  __int64 v187; // [rsp+1C0h] [rbp-470h]
  _BYTE *v188; // [rsp+1D0h] [rbp-460h] BYREF
  __int64 v189; // [rsp+1D8h] [rbp-458h]
  _BYTE v190[256]; // [rsp+1E0h] [rbp-450h] BYREF
  _BYTE *v191; // [rsp+2E0h] [rbp-350h] BYREF
  __int64 v192; // [rsp+2E8h] [rbp-348h]
  _BYTE v193[256]; // [rsp+2F0h] [rbp-340h] BYREF
  _BYTE *v194; // [rsp+3F0h] [rbp-240h] BYREF
  __int64 v195; // [rsp+3F8h] [rbp-238h]
  _BYTE v196[560]; // [rsp+400h] [rbp-230h] BYREF

  v156 = __PAIR128__(a4, a3);
  v161 = a15;
  if ( *(_WORD *)(a10 + 24) == 48 )
    return (__int64 *)a3;
  v23 = a1;
  v24 = a10;
  v168 = (_DWORD *)a1[2];
  v25 = sub_1E0A0C0(a1[4]);
  v26 = a1[4];
  v174 = 0;
  v150 = v25;
  v27 = a1[6];
  v175 = 0;
  v28 = *(_QWORD *)(v26 + 8);
  v162 = v27;
  v29 = *(_QWORD *)(v26 + 56);
  v173 = 0;
  v158 = v29;
  v163 = sub_1D139C0(*(_QWORD *)v26, v28);
  v166 = a5;
  v30 = *(_WORD *)(a5 + 24) == 14 || *(_WORD *)(a5 + 24) == 36;
  if ( v30 )
  {
    v31 = *(_DWORD *)(a5 + 84);
    if ( v31 < 0 )
      v30 = v31 < -*(_DWORD *)(v158 + 32);
  }
  else
  {
    v166 = 0;
  }
  v32 = sub_1D1FC50((__int64)a1, a10);
  if ( v32 < a13 )
    v32 = a13;
  v153 = v32;
  v33 = *(unsigned __int16 *)(a10 + 24);
  if ( v33 == 12 )
  {
    v112 = 0;
  }
  else
  {
    if ( v33 != 52 )
      goto LABEL_11;
    v109 = *(__int64 **)(a10 + 32);
    v24 = *v109;
    if ( *(_WORD *)(*v109 + 24) != 12 )
      goto LABEL_11;
    v110 = v109[5];
    if ( *(_WORD *)(v110 + 24) != 10 )
      goto LABEL_11;
    v111 = *(_QWORD *)(v110 + 88);
    v112 = *(_QWORD **)(v111 + 24);
    if ( *(_DWORD *)(v111 + 32) > 0x40u )
      v112 = (_QWORD *)*v112;
  }
  v160 = sub_14ACAF0(*(_QWORD *)(v24 + 88), (__int64)&v176, 8u, (__int64)v112 + *(_QWORD *)(v24 + 96));
  if ( !v160 )
  {
LABEL_11:
    v160 = 0;
    v34 = 0;
    goto LABEL_12;
  }
  if ( v176 )
  {
    v34 = 1;
LABEL_12:
    if ( a15 )
    {
      v35 = -1;
      sub_1E340A0(&a19);
      v36 = sub_1E340A0(&a16);
      v161 = 0;
      v37 = v153;
      goto LABEL_14;
    }
    goto LABEL_98;
  }
  if ( a15 )
  {
    v34 = 1;
    sub_1E340A0(&a19);
    v36 = sub_1E340A0(&a16);
    v160 = a15;
    v37 = 0;
    v35 = -1;
    goto LABEL_14;
  }
  v34 = 1;
  v161 = v160;
LABEL_98:
  if ( v163 )
    v35 = v168[20379];
  else
    v35 = v168[20377];
  sub_1E340A0(&a19);
  v36 = sub_1E340A0(&a16);
  v37 = 0;
  if ( !v161 )
    v37 = v153;
LABEL_14:
  v38 = a1[4];
  if ( !v30 )
  {
    v39 = v35;
    if ( !(unsigned __int8)sub_1D26C30((__int64)&v173, v35, a12, a13, v37, 0, 0, v34, 1u, v36, v38, (__int64)v168) )
      goto LABEL_16;
    goto LABEL_20;
  }
  if ( !(unsigned __int8)sub_1D26C30((__int64)&v173, v35, a12, 0, v37, 0, 0, v34, 1u, v36, v38, (__int64)v168) )
  {
LABEL_16:
    v21 = 0;
    goto LABEL_17;
  }
  v119 = sub_1F58E60(v173, v162);
  v120 = 0;
  v121 = sub_15A9FE0(v150, v119);
  v122 = *(__int64 (**)())(**(_QWORD **)(v26 + 16) + 112LL);
  if ( v122 != sub_1D00B10 )
    v120 = ((__int64 (__fastcall *)(_QWORD))v122)(*(_QWORD *)(v26 + 16));
  v39 = v26;
  if ( (unsigned __int8)sub_1F4B450(v120, v26) )
  {
    if ( a13 < v121 )
    {
LABEL_162:
      a13 = v121;
      v127 = *(_QWORD *)(v158 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v158 + 32) + *(_DWORD *)(v166 + 84));
      if ( *(_DWORD *)(v127 + 16) < v121 )
      {
        *(_DWORD *)(v127 + 16) = v121;
        v39 = v121;
        sub_1E08740(v158, v121);
      }
    }
  }
  else if ( a13 < v121 )
  {
    v123 = *(_DWORD *)(v150 + 8);
    if ( v123 )
    {
      while ( v121 > v123 )
      {
        v121 >>= 1;
        if ( a13 >= v121 )
          goto LABEL_20;
      }
    }
    goto LABEL_162;
  }
LABEL_20:
  v188 = v190;
  v189 = 0x1000000000LL;
  v192 = 0x1000000000LL;
  v194 = v196;
  v195 = 0x2000000000LL;
  v42 = v173;
  v43 = (v174 - v173) >> 4;
  v191 = v193;
  if ( !(_DWORD)v43 )
    goto LABEL_85;
  v44 = 0;
  v152 = (unsigned __int16)(4 * (a14 != 0));
  v45 = v156;
  v46 = 16LL * (unsigned int)(v43 - 1);
  v157 = v46;
  v170 = 0;
  v164 = v23;
  while ( 1 )
  {
    a8 = _mm_loadu_si128((const __m128i *)(v42 + v170));
    v171 = a8;
    if ( a8.m128i_i8[0] )
      v47 = sub_1D13440(a8.m128i_i8[0]);
    else
      v47 = sub_1F58D40(&v171, v39, v46, v45, v40, v41);
    v52 = v47 >> 3;
    v159 = v47 >> 3;
    if ( v52 > a12 )
      v44 = v44 + a12 - v52;
    if ( !v160 )
    {
LABEL_46:
      sub_1F40D10(&v185, v168, v162, v171.m128i_i64[0], v171.m128i_i64[1]);
      v65 = v187;
      v66 = (unsigned __int8)v186;
      v67 = a19 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (a19 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        if ( (a19 & 4) != 0 )
        {
          v186 = v44 + a20;
          LOBYTE(v187) = a21;
          v185 = v67 | 4;
          HIDWORD(v187) = *(_DWORD *)(v67 + 12);
        }
        else
        {
          v185 = a19 & 0xFFFFFFFFFFFFFFF8LL;
          v186 = v44 + a20;
          LOBYTE(v187) = a21;
          v125 = *(_QWORD *)v67;
          if ( *(_BYTE *)(*(_QWORD *)v67 + 8LL) == 16 )
            v125 = **(_QWORD **)(v125 + 16);
          HIDWORD(v187) = *(_DWORD *)(v125 + 8) >> 8;
        }
      }
      else
      {
        LODWORD(v187) = 0;
        v185 = 0;
        v186 = 0;
        HIDWORD(v187) = HIDWORD(a21);
      }
      v68 = sub_1E340B0(&v185, v159, v162, v150);
      v185 = 0;
      v186 = 0;
      v187 = 0;
      v69 = (4 * (a14 != 0)) | 0x10;
      if ( !v68 )
        v69 = 4 * (a14 != 0);
      v70 = a19 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (a19 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        if ( (a19 & 4) != 0 )
        {
          *((_QWORD *)&v181 + 1) = v44 + a20;
          LOBYTE(v182) = a21;
          *(_QWORD *)&v181 = v70 | 4;
          HIDWORD(v182) = *(_DWORD *)(v70 + 12);
        }
        else
        {
          v106 = *(_QWORD *)v70;
          *(_QWORD *)&v181 = a19 & 0xFFFFFFFFFFFFFFF8LL;
          *((_QWORD *)&v181 + 1) = v44 + a20;
          v107 = *(_BYTE *)(v106 + 8) == 16;
          LOBYTE(v182) = a21;
          if ( v107 )
            v106 = **(_QWORD **)(v106 + 16);
          HIDWORD(v182) = *(_DWORD *)(v106 + 8) >> 8;
        }
      }
      else
      {
        LODWORD(v182) = 0;
        v181 = 0u;
        HIDWORD(v182) = HIDWORD(a21);
      }
      v151 = v69;
      v71 = sub_1D3D250(v164, a10, a11, v44, a2, (__m128i)a7, *(double *)a8.m128i_i64, a9);
      v75 = sub_1D2B810(
              v164,
              1u,
              a2,
              v66,
              v65,
              (v44 | v153) & -(v44 | v153),
              v156,
              (__int64)v71,
              v72,
              v181,
              v182,
              v171.m128i_i64[0],
              v171.m128i_i64[1],
              v151,
              (__int64)&v185);
      v77 = v76;
      v78 = (unsigned int)v189;
      if ( (unsigned int)v189 >= HIDWORD(v189) )
      {
        sub_16CD150((__int64)&v188, v190, 0, 16, v73, v74);
        v78 = (unsigned int)v189;
      }
      v79 = (__int64 *)&v188[16 * v78];
      *v79 = v75;
      v79[1] = 1;
      LODWORD(v189) = v189 + 1;
      v185 = 0;
      v186 = 0;
      v187 = 0;
      v80 = a16 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (a16 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        if ( (a16 & 4) != 0 )
        {
          *((_QWORD *)&v183 + 1) = v44 + a17;
          LOBYTE(v184) = a18;
          *(_QWORD *)&v183 = v80 | 4;
          HIDWORD(v184) = *(_DWORD *)(v80 + 12);
        }
        else
        {
          v124 = *(_QWORD *)v80;
          *(_QWORD *)&v183 = a16 & 0xFFFFFFFFFFFFFFF8LL;
          *((_QWORD *)&v183 + 1) = v44 + a17;
          v107 = *(_BYTE *)(v124 + 8) == 16;
          LOBYTE(v184) = a18;
          if ( v107 )
            v124 = **(_QWORD **)(v124 + 16);
          HIDWORD(v184) = *(_DWORD *)(v124 + 8) >> 8;
        }
      }
      else
      {
        LODWORD(v184) = 0;
        v183 = 0u;
        HIDWORD(v184) = HIDWORD(a18);
      }
      v81 = sub_1D3D250(v164, a5, a6, v44, a2, (__m128i)a7, *(double *)a8.m128i_i64, a9);
      v83 = sub_1D2C750(
              v164,
              v156,
              *((__int64 *)&v156 + 1),
              a2,
              v75,
              v77,
              (__int64)v81,
              v82,
              v183,
              v184,
              v171.m128i_i64[0],
              v171.m128i_i64[1],
              a13,
              v152,
              (__int64)&v185);
      v84 = (unsigned int)v192;
      v85 = (unsigned int)v46;
      if ( (unsigned int)v192 >= HIDWORD(v192) )
      {
        sub_16CD150((__int64)&v191, v193, 0, 16, v40, v41);
        v84 = (unsigned int)v192;
      }
      v86 = (__int64 *)&v191[16 * v84];
      *v86 = v83;
      v86[1] = v85;
      LODWORD(v192) = v192 + 1;
      goto LABEL_59;
    }
    if ( !v161 )
    {
      if ( a8.m128i_i8[0] )
      {
        v48 = (unsigned int)a8.m128i_u8[0] - 2;
        v53 = a8.m128i_i8[0] - 14;
        if ( (unsigned __int8)(a8.m128i_i8[0] - 2) > 5u && v53 > 0x47u || v53 <= 0x5Fu )
          goto LABEL_46;
      }
      else if ( !(unsigned __int8)sub_1F58CF0(&v171) || (unsigned __int8)sub_1F58D20(&v171) )
      {
        goto LABEL_46;
      }
    }
    if ( v178 > v44 )
    {
      a9 = _mm_load_si128(&v171);
      v87 = v178 - v44;
      v88 = v176;
      v172 = a9;
      v148 = v44 + v177;
      if ( !v176 )
        goto LABEL_34;
      if ( a8.m128i_i8[0] )
        v89 = sub_1D13440(a8.m128i_i8[0]);
      else
        v89 = sub_1F58D40(&v172, v39, v48, v49, v50, v51);
      v90 = v87;
      LODWORD(v186) = v89;
      v91 = v89 >> 3;
      if ( v89 >> 3 < v87 )
        v90 = v89 >> 3;
      if ( v89 > 0x40 )
        sub_16A4EF0((__int64)&v185, 0, 0);
      else
        v185 = 0;
      if ( *(_BYTE *)sub_1E0A0C0(v164[4]) )
      {
        if ( !v90 )
          goto LABEL_76;
        v102 = v148;
        v149 = v44;
        v103 = 8 * v91 - 8;
        v104 = v102;
        do
        {
          v105 = (unsigned __int64)(unsigned __int8)sub_1595A50(v88, v104) << v103;
          if ( (unsigned int)v186 > 0x40 )
            *(_QWORD *)v185 |= v105;
          else
            v185 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v186) & (v185 | v105);
          ++v104;
          v103 -= 8;
        }
        while ( v102 + v90 != v104 );
      }
      else
      {
        v92 = v148;
        v93 = v90 + v148;
        if ( !v90 )
        {
LABEL_76:
          v97 = sub_1F58E60(&v172, v164[6]);
          v98 = *(__int64 (**)())(*(_QWORD *)v168 + 936LL);
          if ( v98 == sub_1D12E20
            || !((unsigned __int8 (__fastcall *)(_DWORD *, unsigned __int64 *, __int64))v98)(v168, &v185, v97) )
          {
            v57 = 0;
            v55 = 0;
          }
          else
          {
            v55 = sub_1D38970(
                    (__int64)v164,
                    (__int64)&v185,
                    a2,
                    v172.m128i_u32[0],
                    (const void **)v172.m128i_i64[1],
                    0,
                    (__m128i)a7,
                    *(double *)a8.m128i_i64,
                    a9,
                    0);
            v57 = v126;
          }
          if ( (unsigned int)v186 > 0x40 && v185 )
            j_j___libc_free_0_0(v185);
          goto LABEL_40;
        }
        v149 = v44;
        v94 = 0;
        v95 = v93;
        do
        {
          while ( 1 )
          {
            v96 = (unsigned __int64)(unsigned __int8)sub_1595A50(v88, v92) << v94;
            if ( (unsigned int)v186 <= 0x40 )
              break;
            ++v92;
            v94 += 8;
            *(_QWORD *)v185 |= v96;
            if ( v92 == v95 )
              goto LABEL_75;
          }
          ++v92;
          v94 += 8;
          v185 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v186) & (v185 | v96);
        }
        while ( v92 != v95 );
      }
LABEL_75:
      v44 = v149;
      goto LABEL_76;
    }
    v172 = _mm_load_si128(&v171);
LABEL_34:
    if ( a8.m128i_i8[0] )
    {
      v54 = a8.m128i_i8[0] - 14;
      if ( (unsigned __int8)(a8.m128i_i8[0] - 2) <= 5u || v54 <= 0x47u )
      {
LABEL_134:
        v55 = sub_1D38BB0(
                (__int64)v164,
                0,
                a2,
                v172.m128i_u32[0],
                (const void **)v172.m128i_i64[1],
                0,
                (__m128i)a7,
                *(double *)a8.m128i_i64,
                a9,
                0);
        v57 = v118;
        goto LABEL_40;
      }
      if ( (unsigned __int8)(a8.m128i_i8[0] - 9) <= 1u || a8.m128i_i8[0] == 12 )
      {
        a7 = 0;
        v55 = (__int64)sub_1D364E0(
                         (__int64)v164,
                         a2,
                         v172.m128i_u32[0],
                         (const void **)v172.m128i_i64[1],
                         0,
                         0.0,
                         *(double *)a8.m128i_i64,
                         a9);
        v57 = v56;
        goto LABEL_40;
      }
      v135 = word_42E7700[v54];
      switch ( a8.m128i_i8[0] )
      {
        case 'Y':
        case 'Z':
        case '[':
        case '\\':
        case ']':
        case 'e':
        case 'f':
        case 'g':
        case 'h':
        case 'i':
          v137 = 5;
          break;
        default:
          goto LABEL_175;
      }
    }
    else
    {
      if ( (unsigned __int8)sub_1F58CF0(&v172) )
        goto LABEL_134;
      v135 = sub_1F58D30(&v172);
      v136 = sub_1F596B0(&v172);
      v137 = 5;
      if ( v136 != 9 )
LABEL_175:
        v137 = 6;
    }
    v138 = v137;
    v139 = v164[6];
    LOBYTE(v140) = sub_1D15020(v137, v135);
    v141 = 0;
    if ( !(_BYTE)v140 )
    {
      v140 = sub_1F593D0(v139, v138, 0, v135);
      v147 = v140;
      v141 = v145;
    }
    v142 = v147;
    LOBYTE(v142) = v140;
    v147 = v142;
    *(_QWORD *)&v143 = sub_1D38BB0((__int64)v164, 0, a2, v142, v141, 0, (__m128i)a7, *(double *)a8.m128i_i64, a9, 0);
    v55 = sub_1D309E0(
            v164,
            158,
            a2,
            v172.m128i_u32[0],
            (const void **)v172.m128i_i64[1],
            0,
            *(double *)a7.m128_u64,
            *(double *)a8.m128i_i64,
            *(double *)a9.m128i_i64,
            v143);
    v57 = v144;
LABEL_40:
    if ( !v55 )
      goto LABEL_46;
    v185 = 0;
    v186 = 0;
    v187 = 0;
    v58 = a16 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (a16 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      if ( (a16 & 4) != 0 )
      {
        *((_QWORD *)&v179 + 1) = v44 + a17;
        LOBYTE(v180) = a18;
        *(_QWORD *)&v179 = v58 | 4;
        HIDWORD(v180) = *(_DWORD *)(v58 + 12);
      }
      else
      {
        v108 = *(_QWORD *)v58;
        *(_QWORD *)&v179 = a16 & 0xFFFFFFFFFFFFFFF8LL;
        *((_QWORD *)&v179 + 1) = v44 + a17;
        v107 = *(_BYTE *)(v108 + 8) == 16;
        LOBYTE(v180) = a18;
        if ( v107 )
          v108 = **(_QWORD **)(v108 + 16);
        HIDWORD(v180) = *(_DWORD *)(v108 + 8) >> 8;
      }
    }
    else
    {
      LODWORD(v180) = 0;
      v179 = 0u;
      HIDWORD(v180) = HIDWORD(a18);
    }
    v59 = sub_1D3D250(v164, a5, a6, v44, a2, (__m128i)a7, *(double *)a8.m128i_i64, a9);
    v61 = sub_1D2BF40(
            v164,
            v156,
            *((__int64 *)&v156 + 1),
            a2,
            v55,
            v57,
            (__int64)v59,
            v60,
            v179,
            v180,
            a13,
            v152,
            (__int64)&v185);
    v62 = (unsigned int)v195;
    v63 = (unsigned int)v46;
    if ( (unsigned int)v195 >= HIDWORD(v195) )
    {
      sub_16CD150((__int64)&v194, v196, 0, 16, v40, v41);
      v62 = (unsigned int)v195;
    }
    v64 = (__int64 *)&v194[16 * v62];
    *v64 = v61;
    v64[1] = v63;
    LODWORD(v195) = v195 + 1;
    if ( !v61 )
      goto LABEL_46;
LABEL_59:
    a12 -= v52;
    v44 += v52;
    if ( v157 == v170 )
      break;
    v39 = v170 + 16;
    v42 = v173;
    v170 += 16;
  }
  v23 = v164;
LABEL_85:
  v99 = dword_4FC1700;
  if ( !dword_4FC1700 )
  {
    v100 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v168 + 432LL);
    if ( v100 == sub_1D12D50 )
      v99 = v168[20378];
    else
      v99 = v100((__int64)v168);
  }
  if ( (_DWORD)v192 )
  {
    if ( v99 > 1 && byte_4FC17E0 )
    {
      if ( v99 >= (unsigned int)v192 )
      {
        sub_1D36A70(v23, a2, (__int64)&v194, 0, v192, &v188, a7, *(double *)a8.m128i_i64, a9, &v191);
      }
      else
      {
        v128 = (unsigned int)v192 / v99;
        v129 = (unsigned int)v192 % v99;
        v130 = v99;
        v131 = v192 - v99;
        v167 = v128;
        v165 = v129;
        v132 = 0;
        do
        {
          v133 = v130 + v131;
          v134 = v131;
          v131 -= v130;
          ++v132;
          sub_1D36A70(v23, a2, (__int64)&v194, v134, v133, &v188, a7, *(double *)a8.m128i_i64, a9, &v191);
        }
        while ( v167 > v132 );
        if ( !v165 )
          goto LABEL_89;
        sub_1D36A70(v23, a2, (__int64)&v194, 0, v165, &v188, a7, *(double *)a8.m128i_i64, a9, &v191);
      }
      LODWORD(v101) = v195;
    }
    else
    {
      v101 = (unsigned int)v195;
      v113 = 16LL * (unsigned int)v192;
      v114 = 0;
      do
      {
        v117 = (const __m128i *)&v188[v114];
        if ( (unsigned int)v101 >= HIDWORD(v195) )
        {
          sub_16CD150((__int64)&v194, v196, 0, 16, v40, v41);
          v101 = (unsigned int)v195;
        }
        *(__m128i *)&v194[16 * v101] = _mm_loadu_si128(v117);
        v115 = (unsigned int)(v195 + 1);
        v116 = (const __m128i *)&v191[v114];
        LODWORD(v195) = v115;
        if ( HIDWORD(v195) <= (unsigned int)v115 )
        {
          sub_16CD150((__int64)&v194, v196, 0, 16, v40, v41);
          v115 = (unsigned int)v195;
        }
        a7 = (__m128)_mm_loadu_si128(v116);
        v114 += 16;
        *(__m128 *)&v194[16 * v115] = a7;
        v101 = (unsigned int)(v195 + 1);
        LODWORD(v195) = v195 + 1;
      }
      while ( v113 != v114 );
    }
  }
  else
  {
LABEL_89:
    LODWORD(v101) = v195;
  }
  *((_QWORD *)&v146 + 1) = (unsigned int)v101;
  *(_QWORD *)&v146 = v194;
  v21 = sub_1D359D0(v23, 2, a2, 1, 0, 0, *(double *)a7.m128_u64, *(double *)a8.m128i_i64, a9, v146);
  if ( v194 != v196 )
    _libc_free((unsigned __int64)v194);
  if ( v191 != v193 )
    _libc_free((unsigned __int64)v191);
  if ( v188 != v190 )
    _libc_free((unsigned __int64)v188);
LABEL_17:
  if ( v173 )
    j_j___libc_free_0(v173, v175 - v173);
  return v21;
}
