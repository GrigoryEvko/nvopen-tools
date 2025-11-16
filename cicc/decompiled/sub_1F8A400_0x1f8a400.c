// Function: sub_1F8A400
// Address: 0x1f8a400
//
__int64 __fastcall sub_1F8A400(
        __int64 **a1,
        __int64 a2,
        __int64 a3,
        const void **a4,
        __m128i a5,
        double a6,
        __m128i a7)
{
  __int64 v7; // r12
  char *v8; // rdx
  char v9; // al
  __int64 v10; // rdx
  char v11; // al
  const void **v12; // rdx
  char v13; // r13
  char v14; // bl
  unsigned int v15; // r15d
  unsigned int v16; // r14d
  __int64 v17; // rsi
  int v18; // ebx
  char v19; // al
  __int64 v20; // r9
  unsigned int v21; // ecx
  int v22; // r12d
  char v23; // dl
  int v24; // r13d
  unsigned __int64 v25; // rsi
  __int64 v26; // rax
  char *v27; // rdx
  char v28; // al
  __int64 v29; // rdx
  unsigned int v30; // r15d
  unsigned int v31; // ebx
  const void **v32; // r14
  _QWORD *v33; // r13
  unsigned int v34; // eax
  unsigned int v35; // ebx
  __int64 *v36; // rbx
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // r9
  __int64 v40; // r8
  __int64 v41; // rdx
  __int64 *v42; // rdx
  __int64 v43; // r15
  __int64 v44; // rax
  __int64 v45; // r12
  __int64 v46; // r13
  __int64 v47; // rdx
  __int64 *v48; // r15
  __int64 v49; // rsi
  unsigned int v50; // edx
  unsigned int v52; // esi
  unsigned int v53; // eax
  __int64 v54; // rcx
  unsigned int v55; // ebx
  __int64 v56; // rax
  __int64 v57; // r8
  __int64 v58; // rdx
  __int64 v59; // r9
  __int64 v60; // rax
  __int64 *v61; // rax
  unsigned int v62; // ebx
  const void **v63; // r13
  _QWORD *v64; // r12
  unsigned __int8 v65; // al
  const void **v66; // r8
  __int64 *v67; // rax
  unsigned __int64 v68; // rdi
  __int64 v69; // rax
  const void **v70; // rbx
  __int64 v71; // r13
  __int64 v72; // rax
  const void **v73; // rdx
  __int64 v74; // rdx
  _QWORD *v75; // rax
  __int64 v76; // rdx
  const void **v77; // rdx
  unsigned int v78; // ebx
  int v79; // r13d
  __int64 v80; // r9
  __int64 v81; // rdx
  int v82; // ebx
  __int64 v83; // rax
  __int64 v84; // rdx
  __int64 v85; // r9
  __int64 v86; // r8
  __int64 v87; // rdx
  __int64 *v88; // rdx
  _BYTE *v89; // r8
  __int64 v90; // rsi
  __int64 *v91; // rax
  _QWORD *v92; // rax
  int v93; // edx
  int v94; // ebx
  __int64 v95; // rax
  int v96; // r8d
  __int64 v97; // rbx
  int v98; // edx
  _BYTE *v99; // rax
  __m128i *v100; // rax
  const __m128i *v101; // rdx
  _BYTE *v102; // r15
  __int64 v103; // rbx
  __int64 v104; // rsi
  __int64 *v105; // r14
  const void **v106; // rdx
  const void **v107; // rdx
  __int64 *v108; // r13
  __int128 v109; // rax
  __int64 v110; // rsi
  __int128 v111; // rcx
  __int128 v112; // [rsp-10h] [rbp-1C0h]
  __int128 v113; // [rsp-10h] [rbp-1C0h]
  __int128 v114; // [rsp-10h] [rbp-1C0h]
  __int128 v115; // [rsp-10h] [rbp-1C0h]
  __int64 v116; // [rsp+8h] [rbp-1A8h]
  const void **v117; // [rsp+10h] [rbp-1A0h]
  unsigned int v118; // [rsp+2Ch] [rbp-184h]
  unsigned __int8 v119; // [rsp+2Ch] [rbp-184h]
  char v120; // [rsp+30h] [rbp-180h]
  const void **v121; // [rsp+30h] [rbp-180h]
  unsigned int v122; // [rsp+38h] [rbp-178h]
  unsigned int v123; // [rsp+38h] [rbp-178h]
  __int64 v124; // [rsp+40h] [rbp-170h]
  __int64 v125; // [rsp+40h] [rbp-170h]
  _QWORD *v126; // [rsp+40h] [rbp-170h]
  int v127; // [rsp+40h] [rbp-170h]
  __int64 v128; // [rsp+40h] [rbp-170h]
  __int64 v129; // [rsp+48h] [rbp-168h]
  __int64 v130; // [rsp+48h] [rbp-168h]
  __int64 v131; // [rsp+58h] [rbp-158h]
  __int64 *v132; // [rsp+58h] [rbp-158h]
  const void **v133; // [rsp+58h] [rbp-158h]
  __int64 v134; // [rsp+58h] [rbp-158h]
  __int64 v135; // [rsp+60h] [rbp-150h]
  char v136; // [rsp+60h] [rbp-150h]
  _QWORD *v138; // [rsp+60h] [rbp-150h]
  __int64 v139; // [rsp+60h] [rbp-150h]
  _QWORD *v140; // [rsp+60h] [rbp-150h]
  __int64 v141; // [rsp+60h] [rbp-150h]
  __int64 v142; // [rsp+68h] [rbp-148h]
  __int64 v143; // [rsp+68h] [rbp-148h]
  unsigned int v144; // [rsp+70h] [rbp-140h]
  int i; // [rsp+70h] [rbp-140h]
  unsigned int v146; // [rsp+70h] [rbp-140h]
  __int128 v147; // [rsp+70h] [rbp-140h]
  __int64 v148; // [rsp+90h] [rbp-120h] BYREF
  const void **v149; // [rsp+98h] [rbp-118h]
  __int64 v150; // [rsp+A0h] [rbp-110h] BYREF
  const void **v151; // [rsp+A8h] [rbp-108h]
  __int64 v152; // [rsp+B0h] [rbp-100h] BYREF
  int v153; // [rsp+B8h] [rbp-F8h]
  unsigned __int64 v154; // [rsp+C0h] [rbp-F0h] BYREF
  unsigned int v155; // [rsp+C8h] [rbp-E8h]
  unsigned __int64 v156; // [rsp+D0h] [rbp-E0h] BYREF
  unsigned int v157; // [rsp+D8h] [rbp-D8h]
  __int64 v158; // [rsp+E0h] [rbp-D0h] BYREF
  unsigned int v159; // [rsp+E8h] [rbp-C8h]
  _BYTE *v160; // [rsp+F0h] [rbp-C0h] BYREF
  __int64 v161; // [rsp+F8h] [rbp-B8h]
  _BYTE v162[176]; // [rsp+100h] [rbp-B0h] BYREF

  v7 = a2;
  v148 = a3;
  v8 = *(char **)(a2 + 40);
  v149 = a4;
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  LOBYTE(v160) = v9;
  v161 = v10;
  if ( v9 )
  {
    switch ( v9 )
    {
      case 14:
      case 15:
      case 16:
      case 17:
      case 18:
      case 19:
      case 20:
      case 21:
      case 22:
      case 23:
      case 56:
      case 57:
      case 58:
      case 59:
      case 60:
      case 61:
        v13 = v148;
        LOBYTE(v150) = 2;
        v151 = 0;
        if ( (_BYTE)v148 != 2 )
        {
          v14 = 2;
          goto LABEL_64;
        }
        break;
      case 24:
      case 25:
      case 26:
      case 27:
      case 28:
      case 29:
      case 30:
      case 31:
      case 32:
      case 62:
      case 63:
      case 64:
      case 65:
      case 66:
      case 67:
        v13 = v148;
        LOBYTE(v150) = 3;
        v151 = 0;
        if ( (_BYTE)v148 != 3 )
        {
          v14 = 3;
          goto LABEL_64;
        }
        break;
      case 33:
      case 34:
      case 35:
      case 36:
      case 37:
      case 38:
      case 39:
      case 40:
      case 68:
      case 69:
      case 70:
      case 71:
      case 72:
      case 73:
        v13 = v148;
        LOBYTE(v150) = 4;
        v151 = 0;
        if ( (_BYTE)v148 != 4 )
        {
          v14 = 4;
          goto LABEL_64;
        }
        break;
      case 41:
      case 42:
      case 43:
      case 44:
      case 45:
      case 46:
      case 47:
      case 48:
      case 74:
      case 75:
      case 76:
      case 77:
      case 78:
      case 79:
        v13 = v148;
        LOBYTE(v150) = 5;
        v151 = 0;
        if ( (_BYTE)v148 != 5 )
        {
          v14 = 5;
          goto LABEL_64;
        }
        break;
      case 49:
      case 50:
      case 51:
      case 52:
      case 53:
      case 54:
      case 80:
      case 81:
      case 82:
      case 83:
      case 84:
      case 85:
        v13 = v148;
        LOBYTE(v150) = 6;
        v151 = 0;
        if ( (_BYTE)v148 != 6 )
        {
          v14 = 6;
          goto LABEL_64;
        }
        break;
      case 55:
        v13 = v148;
        LOBYTE(v150) = 7;
        v151 = 0;
        if ( (_BYTE)v148 != 7 )
        {
          v14 = 7;
          goto LABEL_64;
        }
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        v13 = v148;
        LOBYTE(v150) = 8;
        v151 = 0;
        if ( (_BYTE)v148 != 8 )
        {
          v14 = 8;
          goto LABEL_64;
        }
        break;
      case 89:
      case 90:
      case 91:
      case 92:
      case 93:
      case 101:
      case 102:
      case 103:
      case 104:
      case 105:
        v13 = v148;
        LOBYTE(v150) = 9;
        v151 = 0;
        if ( (_BYTE)v148 != 9 )
        {
          v14 = 9;
          goto LABEL_64;
        }
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        v13 = v148;
        LOBYTE(v150) = 10;
        v151 = 0;
        if ( (_BYTE)v148 != 10 )
        {
          v14 = 10;
          goto LABEL_64;
        }
        break;
    }
    return v7;
  }
  v11 = sub_1F596B0((__int64)&v160);
  v13 = v148;
  LOBYTE(v150) = v11;
  v14 = v11;
  v151 = v12;
  if ( (_BYTE)v148 == v11 )
  {
    if ( v11 || v149 == v12 )
      return v7;
    v15 = sub_1F58D40((__int64)&v150);
  }
  else
  {
    if ( v11 )
LABEL_64:
      v15 = sub_1F6C8D0(v14);
    else
      v15 = sub_1F58D40((__int64)&v150);
    if ( v13 )
    {
      v16 = sub_1F6C8D0(v13);
      if ( v16 != v15 )
        goto LABEL_8;
LABEL_43:
      v27 = *(char **)(a2 + 40);
      v28 = *v27;
      v29 = *((_QWORD *)v27 + 1);
      LOBYTE(v160) = v28;
      v161 = v29;
      if ( v28 )
        v30 = word_42FA680[(unsigned __int8)(v28 - 14)];
      else
        v30 = sub_1F58D30((__int64)&v160);
      v31 = v148;
      v32 = v149;
      v33 = (_QWORD *)(*a1)[6];
      LOBYTE(v34) = sub_1D15020(v148, v30);
      v121 = 0;
      if ( !(_BYTE)v34 )
      {
        v34 = sub_1F593D0(v33, v31, (__int64)v32, v30);
        v122 = v34;
        v121 = v106;
      }
      v35 = v122;
      LOBYTE(v35) = v34;
      v123 = v35;
      v36 = *(__int64 **)(a2 + 32);
      if ( *(_WORD *)(a2 + 24) == 111 )
      {
        v108 = *a1;
        *(_QWORD *)&v109 = sub_1D32840(
                             *a1,
                             v148,
                             v149,
                             *v36,
                             v36[1],
                             *(double *)a5.m128i_i64,
                             a6,
                             *(double *)a7.m128i_i64);
        v110 = *(_QWORD *)(a2 + 72);
        v111 = v109;
        v160 = (_BYTE *)v110;
        if ( v110 )
        {
          v147 = v109;
          sub_1623A60((__int64)&v160, v110, 2);
          v111 = v147;
        }
        LODWORD(v161) = *(_DWORD *)(v7 + 64);
        v7 = sub_1D309E0(
               v108,
               111,
               (__int64)&v160,
               v123,
               v121,
               0,
               *(double *)a5.m128i_i64,
               a6,
               *(double *)a7.m128i_i64,
               v111);
        if ( v160 )
          sub_161E7C0((__int64)&v160, (__int64)v160);
      }
      else
      {
        v160 = v162;
        v161 = 0x800000000LL;
        v132 = &v36[5 * *(unsigned int *)(a2 + 56)];
        if ( v132 == v36 )
        {
          v102 = v162;
          v103 = 0;
        }
        else
        {
          do
          {
            v43 = *((unsigned int *)v36 + 2);
            v44 = *v36;
            v45 = *v36;
            v46 = v36[1];
            v47 = *(_QWORD *)(*v36 + 40) + 16 * v43;
            if ( (_BYTE)v150 != *(_BYTE *)v47 || *(const void ***)(v47 + 8) != v151 && !*(_BYTE *)v47 )
            {
              v48 = *a1;
              v49 = *(_QWORD *)(a2 + 72);
              v158 = v49;
              if ( v49 )
                sub_1623A60((__int64)&v158, v49, 2);
              *((_QWORD *)&v112 + 1) = v46;
              *(_QWORD *)&v112 = v45;
              v159 = *(_DWORD *)(a2 + 64);
              v44 = sub_1D309E0(
                      v48,
                      145,
                      (__int64)&v158,
                      (unsigned int)v150,
                      v151,
                      0,
                      *(double *)a5.m128i_i64,
                      a6,
                      *(double *)a7.m128i_i64,
                      v112);
              v43 = v50;
              if ( v158 )
              {
                v124 = v44;
                sub_161E7C0((__int64)&v158, v158);
                v44 = v124;
              }
            }
            v37 = sub_1D32840(
                    *a1,
                    v148,
                    v149,
                    v44,
                    v43 | v46 & 0xFFFFFFFF00000000LL,
                    *(double *)a5.m128i_i64,
                    a6,
                    *(double *)a7.m128i_i64);
            v39 = v38;
            v40 = v37;
            v41 = (unsigned int)v161;
            if ( (unsigned int)v161 >= HIDWORD(v161) )
            {
              v128 = v37;
              v130 = v39;
              sub_16CD150((__int64)&v160, v162, 0, 16, v37, v39);
              v41 = (unsigned int)v161;
              v40 = v128;
              v39 = v130;
            }
            v42 = (__int64 *)&v160[16 * v41];
            v36 += 5;
            *v42 = v40;
            v42[1] = v39;
            LODWORD(v161) = v161 + 1;
            sub_1F81BC0((__int64)a1, *(_QWORD *)&v160[16 * (unsigned int)v161 - 16]);
          }
          while ( v132 != v36 );
          v7 = a2;
          v102 = v160;
          v103 = (unsigned int)v161;
        }
        v104 = *(_QWORD *)(v7 + 72);
        v105 = *a1;
        v158 = v104;
        if ( v104 )
          sub_1623A60((__int64)&v158, v104, 2);
        *((_QWORD *)&v115 + 1) = v103;
        *(_QWORD *)&v115 = v102;
        v159 = *(_DWORD *)(v7 + 64);
        v7 = (__int64)sub_1D359D0(v105, 104, (__int64)&v158, v123, v121, 0, *(double *)a5.m128i_i64, a6, a7, v115);
        if ( v158 )
          sub_161E7C0((__int64)&v158, v158);
        if ( v160 != v162 )
          _libc_free((unsigned __int64)v160);
      }
      return v7;
    }
  }
  v13 = 0;
  v16 = sub_1F58D40((__int64)&v148);
  if ( v16 == v15 )
    goto LABEL_43;
LABEL_8:
  if ( v14 )
  {
    if ( (unsigned __int8)(v14 - 86) > 0x17u && (unsigned __int8)(v14 - 8) > 5u )
      goto LABEL_10;
  }
  else if ( !sub_1F58CD0((__int64)&v150) )
  {
    goto LABEL_10;
  }
  if ( v15 == 32 )
  {
    LOBYTE(v69) = 5;
  }
  else if ( v15 > 0x20 )
  {
    if ( v15 == 64 )
    {
      LOBYTE(v69) = 6;
    }
    else
    {
      if ( v15 != 128 )
      {
LABEL_120:
        v69 = sub_1F58CC0((_QWORD *)(*a1)[6], v15);
        v135 = v69;
        v70 = v77;
        goto LABEL_92;
      }
      LOBYTE(v69) = 7;
    }
  }
  else if ( v15 == 8 )
  {
    LOBYTE(v69) = 3;
  }
  else
  {
    LOBYTE(v69) = 4;
    if ( v15 != 16 )
    {
      LOBYTE(v69) = 2;
      if ( v15 != 1 )
        goto LABEL_120;
    }
  }
  v70 = 0;
LABEL_92:
  v71 = v135;
  LOBYTE(v71) = v69;
  v72 = sub_1F8A400(a1, a2, (unsigned int)v71, v70);
  v150 = v71;
  v13 = v148;
  v151 = v70;
  v7 = v72;
LABEL_10:
  if ( v13 )
  {
    if ( (unsigned __int8)(v13 - 8) > 5u && (unsigned __int8)(v13 - 86) > 0x17u )
      goto LABEL_13;
    v52 = sub_1F6C8D0(v13);
LABEL_67:
    if ( v52 == 32 )
    {
      LOBYTE(v53) = 5;
    }
    else if ( v52 > 0x20 )
    {
      if ( v52 == 64 )
      {
        LOBYTE(v53) = 6;
      }
      else
      {
        if ( v52 != 128 )
        {
LABEL_102:
          v53 = sub_1F58CC0((_QWORD *)(*a1)[6], v52);
          v144 = v53;
          v54 = v74;
          goto LABEL_72;
        }
        LOBYTE(v53) = 7;
      }
    }
    else if ( v52 == 8 )
    {
      LOBYTE(v53) = 3;
    }
    else
    {
      LOBYTE(v53) = 4;
      if ( v52 != 16 )
      {
        LOBYTE(v53) = 2;
        if ( v52 != 1 )
          goto LABEL_102;
      }
    }
    v54 = 0;
LABEL_72:
    v55 = v144;
    LOBYTE(v55) = v53;
    v56 = sub_1F8A400(a1, v7, v55, v54);
    return sub_1F8A400(a1, v56, (unsigned int)v148, v149);
  }
  if ( sub_1F58CD0((__int64)&v148) )
  {
    v52 = sub_1F58D40((__int64)&v148);
    goto LABEL_67;
  }
LABEL_13:
  v17 = *(_QWORD *)(v7 + 72);
  v152 = v17;
  if ( v17 )
    sub_1623A60((__int64)&v152, v17, 2);
  v153 = *(_DWORD *)(v7 + 64);
  if ( v16 <= v15 )
  {
    v78 = v148;
    v146 = v15 / v16;
    v79 = *(_DWORD *)(v7 + 56) * (v15 / v16);
    v140 = (_QWORD *)(*a1)[6];
    v133 = v149;
    v117 = 0;
    v119 = sub_1D15020(v148, v79);
    if ( !v119 )
    {
      v119 = sub_1F593D0(v140, v78, (__int64)v133, v79);
      v117 = v107;
    }
    v160 = v162;
    v161 = 0x800000000LL;
    v81 = *(_QWORD *)(v7 + 32);
    v134 = v81 + 40LL * *(unsigned int *)(v7 + 56);
    if ( v81 == v134 )
    {
      v89 = v162;
      v90 = 0;
    }
    else
    {
      v141 = *(_QWORD *)(v7 + 32);
      do
      {
        if ( *(_WORD *)(*(_QWORD *)v141 + 24LL) == 48 )
        {
          v158 = 0;
          v159 = 0;
          v92 = sub_1D2B300(*a1, 0x30u, (__int64)&v158, v148, (__int64)v149, v80);
          v80 = (__int64)v92;
          v94 = v93;
          if ( v158 )
          {
            v126 = v92;
            sub_161E7C0((__int64)&v158, v158);
            v80 = (__int64)v126;
          }
          v95 = (unsigned int)v161;
          v96 = v94;
          v97 = v146;
          v98 = v161;
          if ( v146 > HIDWORD(v161) - (unsigned __int64)(unsigned int)v161 )
          {
            v116 = v80;
            v127 = v96;
            sub_16CD150((__int64)&v160, v162, v146 + (unsigned __int64)(unsigned int)v161, 16, v96, v80);
            v95 = (unsigned int)v161;
            v80 = v116;
            v96 = v127;
            v98 = v161;
          }
          v99 = &v160[16 * v95];
          if ( v146 )
          {
            do
            {
              if ( v99 )
              {
                *(_QWORD *)v99 = v80;
                *((_DWORD *)v99 + 2) = v96;
              }
              v99 += 16;
              --v97;
            }
            while ( v97 );
            v98 = v161;
          }
          LODWORD(v161) = v146 + v98;
        }
        else
        {
          v82 = 0;
          sub_16A5D10((__int64)&v156, *(_QWORD *)(*(_QWORD *)v141 + 88LL) + 24LL, v15);
          do
          {
            sub_16A5A50((__int64)&v158, (__int64 *)&v156, v16);
            v83 = sub_1D38970((__int64)*a1, (__int64)&v158, (__int64)&v152, v148, v149, 0, a5, a6, a7, 0);
            v85 = v84;
            v86 = v83;
            v87 = (unsigned int)v161;
            if ( (unsigned int)v161 >= HIDWORD(v161) )
            {
              v125 = v83;
              v129 = v85;
              sub_16CD150((__int64)&v160, v162, 0, 16, v83, v85);
              v87 = (unsigned int)v161;
              v86 = v125;
              v85 = v129;
            }
            v88 = (__int64 *)&v160[16 * v87];
            *v88 = v86;
            v88[1] = v85;
            LODWORD(v161) = v161 + 1;
            if ( v157 <= 0x40 )
            {
              if ( v16 == v157 )
                v156 = 0;
              else
                v156 >>= v16;
            }
            else
            {
              sub_16A8110((__int64)&v156, v16);
            }
            if ( v159 > 0x40 && v158 )
              j_j___libc_free_0_0(v158);
            ++v82;
          }
          while ( v146 != v82 );
          if ( *(_BYTE *)sub_1E0A0C0((*a1)[4]) )
          {
            v100 = (__m128i *)&v160[16 * (unsigned int)v161];
            v101 = (const __m128i *)&v160[16 * ((unsigned int)v161 - (unsigned __int64)v146)];
            if ( v101 != v100 )
            {
              while ( v101 < --v100 )
              {
                a5 = _mm_loadu_si128(v101++);
                v101[-1].m128i_i64[0] = v100->m128i_i64[0];
                v101[-1].m128i_i32[2] = v100->m128i_i32[2];
                v100->m128i_i64[0] = a5.m128i_i64[0];
                v100->m128i_i32[2] = a5.m128i_i32[2];
              }
            }
          }
          if ( v157 > 0x40 && v156 )
            j_j___libc_free_0_0(v156);
        }
        v141 += 40;
      }
      while ( v134 != v141 );
      v89 = v160;
      v90 = (unsigned int)v161;
    }
    *((_QWORD *)&v114 + 1) = v90;
    *(_QWORD *)&v114 = v89;
    v91 = sub_1D359D0(*a1, 104, (__int64)&v152, v119, v117, 0, *(double *)a5.m128i_i64, a6, a7, v114);
    v68 = (unsigned __int64)v160;
    v7 = (__int64)v91;
    if ( v160 != v162 )
      goto LABEL_83;
    goto LABEL_84;
  }
  v18 = v16 / v15;
  v160 = v162;
  v161 = 0x800000000LL;
  v118 = *(_DWORD *)(v7 + 56);
  if ( !v118 )
    goto LABEL_80;
  v131 = v7;
  for ( i = 0; i != v118; i += v22 )
  {
    v19 = *(_BYTE *)sub_1E0A0C0((*a1)[4]);
    v155 = v16;
    v136 = v19;
    if ( v16 > 0x40 )
    {
      sub_16A4EF0((__int64)&v154, 0, 0);
      v21 = v155;
    }
    else
    {
      v154 = 0;
      v21 = v16;
    }
    v22 = 1;
    v23 = 1;
    v24 = 0;
    if ( v21 <= 0x40 )
    {
LABEL_21:
      v25 = 0;
      if ( v15 != v21 )
        v25 = (v154 << v15) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v21);
      v154 = v25;
      goto LABEL_24;
    }
    while ( 1 )
    {
      v120 = v23;
      sub_16A7DC0((__int64 *)&v154, v15);
      v23 = v120;
LABEL_24:
      if ( !v136 )
        v24 = v18 - v22;
      v26 = *(_QWORD *)(*(_QWORD *)(v131 + 32) + 40LL * (unsigned int)(v24 + i));
      if ( *(_WORD *)(v26 + 24) != 48 )
      {
        sub_16A5D10((__int64)&v156, *(_QWORD *)(v26 + 88) + 24LL, v15);
        sub_16A5C50((__int64)&v158, (const void **)&v156, v16);
        if ( v155 > 0x40 )
          sub_16A89F0((__int64 *)&v154, &v158);
        else
          v154 |= v158;
        if ( v159 > 0x40 && v158 )
          j_j___libc_free_0_0(v158);
        if ( v157 > 0x40 && v156 )
          j_j___libc_free_0_0(v156);
        v23 = 0;
      }
      if ( v18 == v22 )
        break;
      v21 = v155;
      v24 = v22++;
      if ( v155 <= 0x40 )
        goto LABEL_21;
    }
    if ( v23 )
    {
      v158 = 0;
      v159 = 0;
      v75 = sub_1D2B300(*a1, 0x30u, (__int64)&v158, v148, (__int64)v149, v20);
      v57 = (__int64)v75;
      v59 = v76;
      if ( v158 )
      {
        v138 = v75;
        v142 = v76;
        sub_161E7C0((__int64)&v158, v158);
        v57 = (__int64)v138;
        v59 = v142;
      }
      v60 = (unsigned int)v161;
      if ( (unsigned int)v161 >= HIDWORD(v161) )
      {
LABEL_114:
        v139 = v57;
        v143 = v59;
        sub_16CD150((__int64)&v160, v162, 0, 16, v57, v59);
        v60 = (unsigned int)v161;
        v57 = v139;
        v59 = v143;
      }
    }
    else
    {
      v57 = sub_1D38970((__int64)*a1, (__int64)&v154, (__int64)&v152, v148, v149, 0, a5, a6, a7, 0);
      v59 = v58;
      v60 = (unsigned int)v161;
      if ( (unsigned int)v161 >= HIDWORD(v161) )
        goto LABEL_114;
    }
    v61 = (__int64 *)&v160[16 * v60];
    *v61 = v57;
    v61[1] = v59;
    LODWORD(v161) = v161 + 1;
    if ( v155 > 0x40 && v154 )
      j_j___libc_free_0_0(v154);
  }
  v118 = v161;
LABEL_80:
  v62 = v148;
  v63 = v149;
  v64 = (_QWORD *)(*a1)[6];
  v65 = sub_1D15020(v148, v118);
  v66 = 0;
  if ( !v65 )
  {
    v65 = sub_1F593D0(v64, v62, (__int64)v63, v118);
    v66 = v73;
  }
  *((_QWORD *)&v113 + 1) = (unsigned int)v161;
  *(_QWORD *)&v113 = v160;
  v67 = sub_1D359D0(*a1, 104, (__int64)&v152, v65, v66, 0, *(double *)a5.m128i_i64, a6, a7, v113);
  v68 = (unsigned __int64)v160;
  v7 = (__int64)v67;
  if ( v160 != v162 )
LABEL_83:
    _libc_free(v68);
LABEL_84:
  if ( v152 )
    sub_161E7C0((__int64)&v152, v152);
  return v7;
}
