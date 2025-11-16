// Function: sub_119A9E0
// Address: 0x119a9e0
//
unsigned __int8 *__fastcall sub_119A9E0(__m128i *a1, __int64 a2)
{
  __m128i v4; // xmm1
  unsigned __int64 v5; // xmm2_8
  __m128i v6; // xmm3
  __int64 v7; // rax
  char v8; // al
  __int64 v9; // rax
  unsigned __int8 *v11; // r15
  __int64 v12; // r15
  __int64 v13; // rbx
  unsigned int v14; // eax
  __int64 v15; // rdx
  __m128i v16; // xmm5
  __int64 v17; // rax
  unsigned __int64 v18; // xmm6_8
  __m128i v19; // xmm7
  _BYTE *v20; // r15
  __int64 v21; // rax
  __int64 v22; // rcx
  __m128i v23; // xmm5
  __int64 v24; // rax
  unsigned __int64 v25; // xmm6_8
  __m128i v26; // xmm7
  char v27; // al
  __int64 v28; // rax
  char v29; // al
  __int64 v30; // rdx
  unsigned int **v31; // r14
  __m128i v32; // rax
  __int64 v33; // rax
  __int64 v34; // rdx
  _BYTE *v35; // rax
  unsigned __int8 v36; // cl
  __int64 v37; // rax
  __int64 v38; // rax
  unsigned __int8 v39; // cl
  int v40; // eax
  __int32 v41; // eax
  bool v42; // al
  unsigned __int64 v43; // r11
  bool v44; // r15
  _BYTE *v45; // r15
  int v46; // eax
  unsigned int **v47; // r15
  bool v48; // al
  unsigned __int8 v49; // r13
  _BYTE *v50; // rax
  __int64 v51; // rax
  __int64 v52; // rax
  _BYTE *v53; // rdi
  __int64 v54; // rbx
  unsigned __int8 *v55; // r13
  _BYTE *v56; // rax
  __int64 v57; // rax
  __int64 v58; // r14
  _BYTE *v59; // rbx
  __int64 v60; // r15
  _BYTE *v61; // r13
  __int64 v62; // rdx
  _BYTE *v63; // rax
  _BYTE *v64; // rax
  int v65; // eax
  unsigned __int8 *v66; // rdi
  __int64 v67; // rdx
  __int64 v68; // rdi
  _QWORD *v69; // rax
  __int64 v70; // rdi
  unsigned int v71; // edx
  _QWORD **v72; // rdi
  _QWORD *v73; // rax
  unsigned int v74; // eax
  __int64 v75; // rax
  _BYTE *v76; // rdx
  __int64 v77; // r15
  int v78; // eax
  unsigned int **v79; // r13
  unsigned int v80; // eax
  _BYTE *v81; // rax
  __int64 v82; // rax
  __int64 v83; // r13
  unsigned __int8 *v84; // rax
  __int64 v85; // rax
  unsigned int **v86; // rdi
  __int64 v87; // rdx
  _BYTE *v88; // rsi
  int v89; // edi
  bool v90; // al
  __int64 v91; // rdx
  _BYTE *v92; // rax
  __int64 v93; // rbx
  __int64 v94; // r14
  __int64 v95; // rdx
  unsigned int v96; // esi
  __int64 v97; // rax
  __int64 v98; // rdx
  unsigned __int8 *v99; // rax
  char v100; // al
  unsigned __int8 *v101; // rdx
  __int64 *v102; // rdi
  _BYTE *v103; // rax
  __int64 v104; // rax
  __int64 *v105; // rdi
  __int64 v106; // rax
  __int64 v107; // rax
  unsigned __int64 v108; // [rsp+8h] [rbp-158h]
  unsigned int v109; // [rsp+18h] [rbp-148h]
  int v110; // [rsp+18h] [rbp-148h]
  unsigned __int64 v111; // [rsp+28h] [rbp-138h]
  unsigned __int64 **v112; // [rsp+28h] [rbp-138h]
  int v113; // [rsp+28h] [rbp-138h]
  int v114; // [rsp+30h] [rbp-130h]
  unsigned int v115; // [rsp+30h] [rbp-130h]
  __int64 v116; // [rsp+38h] [rbp-128h]
  __int64 v117; // [rsp+40h] [rbp-120h]
  unsigned int v118; // [rsp+48h] [rbp-118h]
  char v119; // [rsp+48h] [rbp-118h]
  _BYTE *v120; // [rsp+50h] [rbp-110h] BYREF
  _BYTE *v121; // [rsp+58h] [rbp-108h] BYREF
  unsigned __int64 v122; // [rsp+60h] [rbp-100h] BYREF
  unsigned int v123; // [rsp+68h] [rbp-F8h]
  __int16 v124; // [rsp+80h] [rbp-E0h]
  __m128i v125[2]; // [rsp+90h] [rbp-D0h] BYREF
  unsigned __int64 v126; // [rsp+B0h] [rbp-B0h]
  __int64 v127; // [rsp+B8h] [rbp-A8h]
  __m128i v128; // [rsp+C0h] [rbp-A0h]
  __int64 v129; // [rsp+D0h] [rbp-90h]
  __m128i v130; // [rsp+E0h] [rbp-80h] BYREF
  __m128i v131; // [rsp+F0h] [rbp-70h]
  unsigned __int64 v132; // [rsp+100h] [rbp-60h]
  __int64 v133; // [rsp+108h] [rbp-58h]
  __m128i v134; // [rsp+110h] [rbp-50h]
  __int64 v135; // [rsp+120h] [rbp-40h]

  v4 = _mm_loadu_si128(a1 + 7);
  v5 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
  v6 = _mm_loadu_si128(a1 + 9);
  v7 = a1[10].m128i_i64[0];
  v130 = _mm_loadu_si128(a1 + 6);
  v132 = v5;
  v135 = v7;
  v133 = a2;
  v131 = v4;
  v134 = v6;
  v8 = sub_B44E60(a2);
  v9 = sub_101D670(*(unsigned __int8 **)(a2 - 64), *(_BYTE **)(a2 - 32), v8, &v130);
  if ( v9 )
    return sub_F162A0((__int64)a1, a2, v9);
  v11 = sub_F0F270((__int64)a1, (unsigned __int8 *)a2);
  if ( v11 )
    return v11;
  v11 = (unsigned __int8 *)sub_F11DB0(a1->m128i_i64, (unsigned __int8 *)a2);
  if ( v11 )
    return v11;
  v11 = (unsigned __int8 *)sub_11993B0(a1, (unsigned __int8 *)a2);
  if ( v11 )
    return v11;
  v12 = *(_QWORD *)(a2 - 32);
  v13 = *(_QWORD *)(a2 - 64);
  v117 = v12;
  v116 = *(_QWORD *)(a2 + 8);
  v14 = sub_BCB060(v116);
  v15 = v12 + 24;
  v118 = v14;
  if ( *(_BYTE *)v12 != 17 )
  {
    v34 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v12 + 8) + 8LL) - 17;
    if ( (unsigned int)v34 > 1 )
      goto LABEL_10;
    if ( *(_BYTE *)v12 > 0x15u )
      goto LABEL_10;
    v35 = sub_AD7630(v12, 0, v34);
    if ( !v35 )
      goto LABEL_10;
    v15 = (__int64)(v35 + 24);
    if ( *v35 != 17 )
      goto LABEL_10;
  }
  if ( *(_DWORD *)(v15 + 8) > 0x40u )
  {
    v112 = (unsigned __int64 **)v15;
    v114 = *(_DWORD *)(v15 + 8);
    if ( v114 - (unsigned int)sub_C444A0(v15) > 0x40 )
      goto LABEL_10;
    v111 = **v112;
    if ( v118 <= v111 )
      goto LABEL_10;
  }
  else
  {
    v111 = *(_QWORD *)v15;
    if ( (unsigned __int64)v118 <= *(_QWORD *)v15 )
      goto LABEL_10;
  }
  v36 = *(_BYTE *)v13;
  if ( *(_BYTE *)v13 == 54 )
  {
    v64 = *(_BYTE **)(v13 - 64);
    if ( *v64 != 68 || (v97 = *((_QWORD *)v64 - 4)) == 0 )
    {
LABEL_81:
      v65 = v36 - 29;
      goto LABEL_82;
    }
    v120 = (_BYTE *)v97;
    v98 = *(_QWORD *)(v13 - 32);
    if ( v12 == v98 && v98 && (_DWORD)v111 == v118 - (unsigned int)sub_BCB060(*(_QWORD *)(v97 + 8)) )
    {
      LOWORD(v132) = 257;
      v99 = (unsigned __int8 *)sub_BD2C40(72, unk_3F10A14);
      v11 = v99;
      if ( v99 )
        sub_B51650((__int64)v99, (__int64)v120, v116, (__int64)&v130, 0, 0);
      return v11;
    }
    v36 = *(_BYTE *)v13;
  }
  if ( v36 > 0x1Cu )
  {
    if ( v36 > 0x36u || ((0x40540000000000uLL >> v36) & 1) == 0 )
      goto LABEL_40;
    goto LABEL_81;
  }
  if ( v36 != 5 )
    goto LABEL_40;
  v65 = *(unsigned __int16 *)(v13 + 2);
  if ( (*(_WORD *)(v13 + 2) & 0xFFFD) != 0xD && (v65 & 0xFFF7) != 0x11 )
    goto LABEL_41;
LABEL_82:
  if ( v65 == 25 && (*(_BYTE *)(v13 + 1) & 4) != 0 && *(_QWORD *)(v13 - 64) )
  {
    v120 = *(_BYTE **)(v13 - 64);
    v66 = *(unsigned __int8 **)(v13 - 32);
    v67 = *v66;
    if ( (_BYTE)v67 == 17 )
    {
      v68 = (__int64)(v66 + 24);
    }
    else
    {
      if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v66 + 1) + 8LL) - 17 > 1 )
        goto LABEL_91;
      if ( (unsigned __int8)v67 > 0x15u )
        goto LABEL_91;
      v103 = sub_AD7630((__int64)v66, 0, v67);
      if ( !v103 )
        goto LABEL_91;
      v68 = (__int64)(v103 + 24);
      if ( *v103 != 17 )
        goto LABEL_91;
    }
    if ( *(_DWORD *)(v68 + 8) > 0x40u )
    {
      v110 = *(_DWORD *)(v68 + 8);
      if ( v110 - (unsigned int)sub_C444A0(v68) <= 0x40 )
      {
        v69 = **(_QWORD ***)v68;
        if ( v118 > (unsigned __int64)v69 )
          goto LABEL_89;
      }
    }
    else
    {
      v69 = *(_QWORD **)v68;
      if ( (unsigned __int64)v118 > *(_QWORD *)v68 )
      {
LABEL_89:
        if ( (unsigned int)v111 > (unsigned int)v69 )
        {
          v104 = sub_AD64C0(v116, (unsigned int)(v111 - (_DWORD)v69), 0);
          LOWORD(v132) = 257;
          v88 = v120;
          v87 = v104;
          v89 = 27;
          goto LABEL_125;
        }
        if ( (unsigned int)v111 < (unsigned int)v69 )
        {
          v106 = sub_AD64C0(v116, (unsigned int)((_DWORD)v69 - v111), 0);
          LOWORD(v132) = 257;
          v11 = (unsigned __int8 *)sub_B504D0(25, (__int64)v120, v106, (__int64)&v130, 0, 0);
          sub_B44850(v11, 1);
          return v11;
        }
      }
    }
LABEL_91:
    v36 = *(_BYTE *)v13;
  }
LABEL_40:
  if ( v36 == 56 )
  {
    if ( *(_QWORD *)(v13 - 64) )
    {
      v120 = *(_BYTE **)(v13 - 64);
      v70 = *(_QWORD *)(v13 - 32);
      if ( *(_BYTE *)v70 == 17
        || (v91 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v70 + 8) + 8LL) - 17, (unsigned int)v91 <= 1)
        && *(_BYTE *)v70 <= 0x15u
        && (v92 = sub_AD7630(v70, 0, v91), (v70 = (__int64)v92) != 0)
        && *v92 == 17 )
      {
        v71 = *(_DWORD *)(v70 + 32);
        v72 = (_QWORD **)(v70 + 24);
        if ( v71 > 0x40 )
        {
          if ( v71 - (unsigned int)sub_C444A0((__int64)v72) > 0x40 )
            goto LABEL_41;
          v73 = (_QWORD *)**v72;
          if ( v118 <= (unsigned __int64)v73 )
            goto LABEL_41;
        }
        else
        {
          v73 = *v72;
          if ( v118 <= (unsigned __int64)*v72 )
            goto LABEL_41;
        }
        v74 = v111 + (_DWORD)v73;
        LOWORD(v132) = 257;
        if ( v74 > v118 - 1 )
          v74 = v118 - 1;
        v75 = sub_AD64C0(v116, v74, 0);
        return (unsigned __int8 *)sub_B504D0(27, (__int64)v120, v75, (__int64)&v130, 0, 0);
      }
    }
  }
LABEL_41:
  v37 = *(_QWORD *)(v13 + 16);
  if ( v37 )
  {
    if ( !*(_QWORD *)(v37 + 8) && *(_BYTE *)v13 == 69 )
    {
      v76 = *(_BYTE **)(v13 - 32);
      if ( v76 )
      {
        v120 = *(_BYTE **)(v13 - 32);
        if ( (unsigned int)*(unsigned __int8 *)(v116 + 8) - 17 <= 1 )
        {
LABEL_103:
          v77 = *((_QWORD *)v76 + 1);
          v78 = sub_BCB060(v77);
          v79 = (unsigned int **)a1[2].m128i_i64[0];
          v80 = v78 - 1;
          LOWORD(v132) = 257;
          if ( v80 > (unsigned int)v111 )
            v80 = v111;
          v81 = (_BYTE *)sub_AD64C0(v77, v80, 0);
          v82 = sub_920F70(v79, v120, v81, (__int64)&v130, 0);
          goto LABEL_106;
        }
        if ( (unsigned __int8)sub_F0C890((__int64)a1, v116, *((_QWORD *)v76 + 1)) )
        {
          v76 = v120;
          goto LABEL_103;
        }
      }
    }
  }
  if ( (_DWORD)v111 == v118 - 1 )
  {
    v130.m128i_i64[0] = 0;
    v130.m128i_i64[1] = (__int64)&v120;
    v131.m128i_i64[0] = (__int64)&v120;
    v85 = *(_QWORD *)(v13 + 16);
    if ( v85
      && !*(_QWORD *)(v85 + 8)
      && *(_BYTE *)v13 == 58
      && ((v100 = sub_10A7530((__int64 **)&v130, 15, *(unsigned __int8 **)(v13 - 64)),
           v101 = *(unsigned __int8 **)(v13 - 32),
           v100)
       && v101 == *(unsigned __int8 **)v131.m128i_i64[0]
       || (unsigned __int8)sub_10A7530((__int64 **)&v130, 15, v101)
       && *(_QWORD *)(v13 - 64) == *(_QWORD *)v131.m128i_i64[0]) )
    {
      v102 = (__int64 *)a1[2].m128i_i64[0];
      LOWORD(v126) = 257;
      v82 = (__int64)sub_10A0740(v102, (__int64)v120, (__int64)v125);
    }
    else
    {
      v130.m128i_i64[0] = (__int64)&v120;
      v130.m128i_i64[1] = (__int64)&v122;
      if ( (unsigned __int8)sub_11960C0(&v130, (unsigned __int8 *)v13) )
      {
        v86 = (unsigned int **)a1[2].m128i_i64[0];
        LOWORD(v126) = 257;
        v82 = sub_92B530(v86, 0x28u, (__int64)v120, (_BYTE *)v122, (__int64)v125);
      }
      else
      {
        v130 = (__m128i)(unsigned __int64)&v120;
        v131.m128i_i64[0] = 0;
        v131.m128i_i64[1] = (__int64)&v120;
        if ( !sub_1196720(&v130, v13) )
          goto LABEL_43;
        v105 = (__int64 *)a1[2].m128i_i64[0];
        LOWORD(v126) = 257;
        v82 = (__int64)sub_10BE4E0(v105, (__int64)v120, (__int64)v125);
      }
    }
LABEL_106:
    v83 = v82;
    LOWORD(v132) = 257;
    v84 = (unsigned __int8 *)sub_BD2C40(72, unk_3F10A14);
    v11 = v84;
    if ( v84 )
      sub_B51650((__int64)v84, v83, v116, (__int64)&v130, 0, 0);
    return v11;
  }
LABEL_43:
  v130.m128i_i64[0] = (__int64)&v120;
  v130.m128i_i64[1] = (__int64)&v121;
  v131.m128i_i8[0] = 0;
  v38 = *(_QWORD *)(v13 + 16);
  if ( v38 && !*(_QWORD *)(v38 + 8) )
  {
    v39 = *(_BYTE *)v13;
    if ( *(_BYTE *)v13 <= 0x1Cu )
    {
      if ( v39 != 5 )
        goto LABEL_10;
      v40 = *(unsigned __int16 *)(v13 + 2);
      if ( (*(_WORD *)(v13 + 2) & 0xFFFD) != 0xD && (v40 & 0xFFF7) != 0x11 )
        goto LABEL_10;
    }
    else
    {
      if ( v39 > 0x36u || ((0x40540000000000uLL >> v39) & 1) == 0 )
        goto LABEL_10;
      v40 = v39 - 29;
    }
    if ( v40 == 17 && (*(_BYTE *)(v13 + 1) & 4) != 0 )
    {
      if ( *(_QWORD *)(v13 - 64) )
      {
        v120 = *(_BYTE **)(v13 - 64);
        if ( (unsigned __int8)sub_991580((__int64)&v130.m128i_i64[1], *(_QWORD *)(v13 - 32)) )
        {
          if ( v118 > 2 )
          {
            sub_9865C0((__int64)&v122, (__int64)v121);
            sub_C46F20((__int64)&v122, 1u);
            v41 = v123;
            v123 = 0;
            v109 = v41;
            v125[0].m128i_i64[0] = v122;
            v108 = v122;
            v125[0].m128i_i32[2] = v41;
            v42 = sub_986BA0((__int64)v125);
            v43 = v108;
            v44 = v42;
            if ( v42 )
            {
              v45 = v121;
              v46 = sub_9871A0((__int64)v121);
              v43 = v108;
              v44 = *((_DWORD *)v45 + 2) + ~v46 == (_DWORD)v111 && (unsigned int)v111 < v118 - 1;
            }
            if ( v109 > 0x40 && v43 )
              j_j___libc_free_0_0(v43);
            if ( v123 > 0x40 && v122 )
              j_j___libc_free_0_0(v122);
            if ( v44 )
            {
              v47 = (unsigned int **)a1[2].m128i_i64[0];
              LOWORD(v132) = 257;
              v48 = sub_B44E60(a2);
              LOWORD(v126) = 257;
              v49 = v48;
              v50 = (_BYTE *)sub_AD64C0(v116, (unsigned int)v111, 0);
              v51 = sub_920F70(v47, v120, v50, (__int64)v125, v49);
              v11 = (unsigned __int8 *)sub_B504D0(13, (__int64)v120, v51, (__int64)&v130, 0, 0);
              sub_B44850(v11, 1);
              sub_B447F0(v11, (*(_BYTE *)(v13 + 1) & 2) != 0);
              return v11;
            }
          }
        }
      }
    }
  }
LABEL_10:
  v11 = (unsigned __int8 *)a2;
  v16 = _mm_loadu_si128(a1 + 7);
  v17 = a1[10].m128i_i64[0];
  v18 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
  v125[0] = _mm_loadu_si128(a1 + 6);
  v19 = _mm_loadu_si128(a1 + 9);
  v129 = v17;
  v126 = v18;
  v125[1] = v16;
  v127 = a2;
  v128 = v19;
  if ( (unsigned __int8)sub_11948A0((unsigned __int8 *)a2, v125) )
    return v11;
  v20 = (_BYTE *)v117;
  v115 = v118 - 1;
  if ( *(_BYTE *)v117 == 17
    || (v62 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v117 + 8) + 8LL) - 17, (unsigned int)v62 <= 1)
    && *(_BYTE *)v117 <= 0x15u
    && (v63 = sub_AD7630(v117, 1, v62), (v20 = v63) != 0)
    && *v63 == 17 )
  {
    if ( *((_DWORD *)v20 + 8) > 0x40u )
    {
      v113 = *((_DWORD *)v20 + 8);
      if ( v113 - (unsigned int)sub_C444A0((__int64)(v20 + 24)) > 0x40 )
        goto LABEL_15;
      v21 = **((_QWORD **)v20 + 3);
    }
    else
    {
      v21 = *((_QWORD *)v20 + 3);
    }
    if ( v115 == v21 )
    {
      v130.m128i_i64[1] = v21;
      v130.m128i_i64[0] = (__int64)&v121;
      v52 = *(_QWORD *)(v13 + 16);
      if ( v52 )
      {
        if ( !*(_QWORD *)(v52 + 8) && *(_BYTE *)v13 == 54 )
        {
          if ( *(_QWORD *)(v13 - 64) )
          {
            v121 = *(_BYTE **)(v13 - 64);
            if ( sub_1196010(&v130.m128i_i64[1], *(_QWORD *)(v13 - 32)) )
            {
              v53 = (_BYTE *)sub_AD64C0(v116, 1, 0);
              if ( (*(_BYTE *)(v13 + 7) & 0x40) != 0 )
                v54 = *(_QWORD *)(v13 - 8);
              else
                v54 = v13 - 32LL * (*(_DWORD *)(v13 + 4) & 0x7FFFFFF);
              v55 = *(unsigned __int8 **)(v54 + 32);
              v56 = (_BYTE *)sub_AD7180(v53, (unsigned __int8 *)v117);
              v57 = sub_AD7180(v56, v55);
              v58 = a1[2].m128i_i64[0];
              v59 = v121;
              v124 = 257;
              v60 = v57;
              v61 = (_BYTE *)(*(__int64 (__fastcall **)(_QWORD, __int64, _BYTE *, __int64))(**(_QWORD **)(v58 + 80)
                                                                                          + 16LL))(
                               *(_QWORD *)(v58 + 80),
                               28,
                               v121,
                               v57);
              if ( !v61 )
              {
                LOWORD(v132) = 257;
                v61 = (_BYTE *)sub_B504D0(28, (__int64)v59, v60, (__int64)&v130, 0, 0);
                (*(void (__fastcall **)(_QWORD, _BYTE *, unsigned __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v58 + 88)
                                                                                            + 16LL))(
                  *(_QWORD *)(v58 + 88),
                  v61,
                  &v122,
                  *(_QWORD *)(v58 + 56),
                  *(_QWORD *)(v58 + 64));
                v93 = *(_QWORD *)v58;
                v94 = *(_QWORD *)v58 + 16LL * *(unsigned int *)(v58 + 8);
                while ( v94 != v93 )
                {
                  v95 = *(_QWORD *)(v93 + 8);
                  v96 = *(_DWORD *)v93;
                  v93 += 16;
                  sub_B99FD0((__int64)v61, v96, v95);
                }
              }
              v121 = v61;
              LOWORD(v132) = 257;
              return (unsigned __int8 *)sub_B50550((__int64)v61, (__int64)&v130, 0, 0);
            }
          }
        }
      }
    }
  }
LABEL_15:
  v11 = sub_1196190((__int64)a1, (unsigned __int8 *)a2);
  if ( v11 )
    return v11;
  v123 = v118;
  v22 = 1LL << v115;
  if ( v118 > 0x40 )
  {
    sub_C43690((__int64)&v122, 0, 0);
    v22 = 1LL << v115;
    if ( v123 > 0x40 )
    {
      *(_QWORD *)(v122 + 8LL * (v115 >> 6)) |= 1LL << v115;
      goto LABEL_19;
    }
  }
  else
  {
    v122 = 0;
  }
  v122 |= v22;
LABEL_19:
  v23 = _mm_loadu_si128(a1 + 7);
  v24 = a1[10].m128i_i64[0];
  v25 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
  v130 = _mm_loadu_si128(a1 + 6);
  v26 = _mm_loadu_si128(a1 + 9);
  v135 = v24;
  v132 = v25;
  v131 = v23;
  v133 = a2;
  v134 = v26;
  v27 = sub_9AC230(v13, (__int64)&v122, &v130, 0);
  if ( v123 > 0x40 && v122 )
  {
    v119 = v27;
    j_j___libc_free_0_0(v122);
    v27 = v119;
  }
  if ( v27 )
  {
    v87 = v117;
    LOWORD(v132) = 257;
    v88 = (_BYTE *)v13;
    v89 = 26;
LABEL_125:
    v11 = (unsigned __int8 *)sub_B504D0(v89, (__int64)v88, v87, (__int64)&v130, 0, 0);
    v90 = sub_B44E60(a2);
    sub_B448B0((__int64)v11, v90);
    return v11;
  }
  v130.m128i_i64[0] = 0;
  v130.m128i_i64[1] = (__int64)&v121;
  v28 = *(_QWORD *)(v13 + 16);
  if ( !v28 || *(_QWORD *)(v28 + 8) || *(_BYTE *)v13 != 59 )
    return v11;
  v29 = sub_995B10(&v130, *(_QWORD *)(v13 - 64));
  v30 = *(_QWORD *)(v13 - 32);
  if ( v29 && v30 )
  {
    *(_QWORD *)v130.m128i_i64[1] = v30;
  }
  else
  {
    if ( !(unsigned __int8)sub_995B10(&v130, *(_QWORD *)(v13 - 32)) )
      return v11;
    v107 = *(_QWORD *)(v13 - 64);
    if ( !v107 )
      return v11;
    *(_QWORD *)v130.m128i_i64[1] = v107;
  }
  v31 = (unsigned int **)a1[2].m128i_i64[0];
  v32.m128i_i64[0] = (__int64)sub_BD5D20(v13);
  LOWORD(v132) = 773;
  v130 = v32;
  v131.m128i_i64[0] = (__int64)".not";
  v33 = sub_920F70(v31, v121, (_BYTE *)v117, (__int64)&v130, 0);
  LOWORD(v132) = 257;
  return (unsigned __int8 *)sub_B50640(v33, (__int64)&v130, 0, 0);
}
