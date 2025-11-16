// Function: sub_1B4CB10
// Address: 0x1b4cb10
//
__int64 __fastcall sub_1B4CB10(
        __int64 a1,
        __int64 **a2,
        unsigned __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v13; // r12
  __int64 v14; // rdi
  __int64 v15; // r8
  __int64 v16; // r14
  __int64 v17; // rax
  int v18; // edx
  _QWORD *v19; // rax
  __int64 v20; // rcx
  int v21; // eax
  int v22; // eax
  unsigned int v23; // r14d
  int v25; // edx
  __int64 v26; // rax
  unsigned int v27; // eax
  char v28; // al
  __int64 v29; // r15
  __int64 i; // rbx
  __int64 v32; // r13
  __int64 v33; // r14
  __int64 v34; // rsi
  double v35; // xmm4_8
  double v36; // xmm5_8
  __int64 *v37; // rax
  __int64 v38; // r11
  __int64 v39; // rax
  __int64 v40; // r13
  __int64 *v41; // rax
  __int64 v42; // r13
  __int64 v43; // r15
  _QWORD *v44; // rax
  __int64 v45; // r14
  _QWORD *v46; // rax
  __int64 v47; // r14
  __int64 *v48; // rbx
  __int64 *v49; // rcx
  __int64 v50; // rdi
  unsigned __int64 *v51; // r8
  __int64 *v52; // rbx
  __int64 *v53; // rcx
  __int64 v54; // rdi
  unsigned __int64 *v55; // r8
  __int64 v56; // r10
  _QWORD *v57; // r14
  __int64 *v58; // rax
  __int64 v59; // r15
  __int64 v60; // r13
  __int64 v61; // rax
  _QWORD *v62; // r13
  double v63; // xmm4_8
  double v64; // xmm5_8
  _QWORD *v65; // r14
  _QWORD *v66; // rax
  __int64 v67; // rbx
  __int64 v68; // rax
  unsigned __int64 v69; // rax
  __int64 v70; // rax
  unsigned __int64 v71; // rax
  __int64 v72; // rax
  unsigned __int64 v73; // rax
  __int64 v74; // rcx
  __int64 v75; // rdx
  unsigned __int64 v76; // rax
  __int64 v77; // rax
  __int64 v78; // rax
  _QWORD *v79; // rax
  _QWORD *v80; // rax
  double v81; // xmm4_8
  double v82; // xmm5_8
  __int64 v83; // rax
  __int64 v84; // rsi
  __int64 v85; // rax
  unsigned __int64 v86; // rax
  __int64 v87; // rcx
  __int64 v88; // rdx
  unsigned __int64 v89; // rax
  __int64 v90; // rax
  __int64 v91; // rax
  __int64 v92; // rax
  unsigned __int64 v93; // rax
  __int64 v94; // rcx
  __int64 v95; // rdx
  unsigned __int64 v96; // rax
  unsigned __int64 v97; // rax
  __int64 v98; // rax
  __int64 v99; // rax
  __int64 v100; // rax
  unsigned __int64 v101; // rax
  __int64 v102; // rcx
  __int64 v103; // rdx
  unsigned __int64 v104; // rax
  unsigned __int64 v105; // rax
  __int64 v106; // rax
  __int64 v107; // rax
  _QWORD *v108; // rax
  double v109; // xmm4_8
  double v110; // xmm5_8
  __int64 *v111; // [rsp+8h] [rbp-148h]
  __int64 *v112; // [rsp+8h] [rbp-148h]
  __int64 v113; // [rsp+10h] [rbp-140h]
  unsigned __int64 *v114; // [rsp+10h] [rbp-140h]
  __int64 v115; // [rsp+10h] [rbp-140h]
  __int64 v116; // [rsp+10h] [rbp-140h]
  __int64 v117; // [rsp+10h] [rbp-140h]
  __int64 v118; // [rsp+18h] [rbp-138h]
  __int64 v119; // [rsp+18h] [rbp-138h]
  __int64 v120; // [rsp+18h] [rbp-138h]
  __int64 v121; // [rsp+18h] [rbp-138h]
  __int64 v122; // [rsp+20h] [rbp-130h]
  unsigned __int64 v123; // [rsp+20h] [rbp-130h]
  __int64 v124; // [rsp+20h] [rbp-130h]
  __int64 v125; // [rsp+20h] [rbp-130h]
  __int64 v126; // [rsp+28h] [rbp-128h]
  unsigned __int64 v127; // [rsp+28h] [rbp-128h]
  __int64 v128; // [rsp+28h] [rbp-128h]
  __int64 v129; // [rsp+30h] [rbp-120h]
  __int64 v130; // [rsp+38h] [rbp-118h]
  __int64 v131; // [rsp+38h] [rbp-118h]
  int v132; // [rsp+48h] [rbp-108h] BYREF
  int v133; // [rsp+4Ch] [rbp-104h] BYREF
  __int64 v134; // [rsp+50h] [rbp-100h] BYREF
  __int64 v135; // [rsp+58h] [rbp-F8h] BYREF
  __int64 v136[2]; // [rsp+60h] [rbp-F0h] BYREF
  __int16 v137; // [rsp+70h] [rbp-E0h]
  __int64 v138; // [rsp+80h] [rbp-D0h] BYREF
  _BYTE *v139; // [rsp+88h] [rbp-C8h]
  _BYTE *v140; // [rsp+90h] [rbp-C0h]
  __int64 v141; // [rsp+98h] [rbp-B8h]
  int v142; // [rsp+A0h] [rbp-B0h]
  _BYTE v143[40]; // [rsp+A8h] [rbp-A8h] BYREF
  __m128i v144; // [rsp+D0h] [rbp-80h] BYREF
  __int64 *v145; // [rsp+E0h] [rbp-70h]
  __int64 v146; // [rsp+E8h] [rbp-68h]
  __int64 v147; // [rsp+F0h] [rbp-60h]

  v13 = *(_QWORD *)(a1 + 40);
  v14 = *(_QWORD *)(v13 + 56);
  if ( v14 && (unsigned __int8)sub_1560180(v14 + 112, 33) )
    return 0;
  v130 = sub_1AA6DC0(v13, &v134, &v135);
  if ( !v130 || *(_BYTE *)(v130 + 16) == 13 )
    return 0;
  v16 = *(_QWORD *)a1;
  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 15 )
  {
    v19 = (*(_BYTE *)(a1 + 23) & 0x40) != 0
        ? *(_QWORD **)(a1 - 8)
        : (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    v20 = v19[3];
    if ( v20 != *v19 )
    {
      v21 = *(_DWORD *)(*(_QWORD *)*v19 + 8LL) >> 8;
      if ( v21 == 5 )
        return 0;
      if ( !v21 )
        return 0;
      v22 = *(_DWORD *)(*(_QWORD *)v20 + 8LL) >> 8;
      if ( v22 == 5 || !v22 )
        return 0;
    }
  }
  if ( (*(_DWORD *)(a1 + 20) & 0xFFFFFFF) == 2 )
  {
    if ( v134 )
    {
      v15 = v135;
      if ( v134 != v135 )
      {
        if ( v135 )
        {
          v25 = 0;
          v126 = *(_QWORD *)(a1 + 40);
          v26 = *(_QWORD *)(v126 + 48);
          while ( 1 )
          {
            if ( !v26 )
              BUG();
            if ( *(_BYTE *)(v26 - 8) != 77 )
              break;
            v26 = *(_QWORD *)(v26 + 8);
            ++v25;
          }
          if ( v25 == 1 )
          {
            v129 = v135;
            v122 = sub_1455EB0(a1, v134);
            v113 = sub_1455EB0(a1, v129);
            if ( sub_1642F90(v16, 1) && sub_1642F90(*(_QWORD *)v130, 1) && sub_1642F90(*(_QWORD *)v122, 1) )
            {
              LOBYTE(v27) = sub_1642F90(*(_QWORD *)v113, 1);
              v23 = v27;
              if ( (_BYTE)v27 )
              {
                v28 = *(_BYTE *)(v113 + 16);
                if ( *(_BYTE *)(v122 + 16) == 13 )
                {
                  v118 = v122 + 24;
                  if ( v28 == 13 )
                  {
                    sub_1B47690((__int64)&v144, a1, 0, 0, 0);
                    if ( sub_1455000(v118) )
                    {
                      if ( sub_1455000(v113 + 24) )
                        goto LABEL_39;
                      if ( sub_13D01C0(v113 + 24) )
                      {
                        v122 = v130;
                        goto LABEL_39;
                      }
                      goto LABEL_173;
                    }
                    if ( sub_13D01C0(v118) )
                    {
                      if ( !sub_1455000(v113 + 24) )
                      {
                        if ( sub_13D01C0(v113 + 24) )
                        {
LABEL_39:
                          if ( *(_BYTE *)(v122 + 16) == 77 && a1 == v122 )
                            goto LABEL_41;
                          sub_164B7C0(v122, a1);
                          sub_164D160(a1, v122, a4, a5, a6, a7, v109, v110, a10, a11);
                          sub_15F20C0((_QWORD *)a1);
LABEL_150:
                          sub_17CD270(v144.m128i_i64);
                          return v23;
                        }
LABEL_195:
                        v100 = 0;
                        v125 = v135;
                        if ( *(_BYTE *)(v113 + 16) > 0x17u )
                          v100 = v113;
                        v121 = v100;
                        v101 = sub_157EBA0(v135);
                        if ( *(_BYTE *)(v101 + 16) != 26 )
                          v101 = 0;
                        v102 = *(_QWORD *)(v125 + 48);
                        v103 = 0;
                        while ( v125 + 40 != v102 )
                        {
                          v102 = *(_QWORD *)(v102 + 8);
                          ++v103;
                        }
                        if ( v103 > 2 )
                          goto LABEL_41;
                        if ( !v101 )
                          goto LABEL_41;
                        if ( (*(_DWORD *)(v101 + 20) & 0xFFFFFFF) != 1 )
                          goto LABEL_41;
                        v104 = sub_157EBA0(v134);
                        if ( (unsigned int)sub_15F4D60(v104) != 2 )
                          goto LABEL_41;
                        v105 = sub_157EBA0(v134);
                        v106 = sub_15F4DF0(v105, 1u);
                        if ( !v121 )
                          goto LABEL_41;
                        if ( v106 != v135 )
                          goto LABEL_41;
                        if ( *(_BYTE *)(v121 + 16) == 77 )
                          goto LABEL_41;
                        if ( v135 != *(_QWORD *)(v121 + 40) )
                          goto LABEL_41;
                        if ( sub_1B46960(v121) )
                          goto LABEL_41;
                        v107 = *(_QWORD *)(v113 + 8);
                        if ( !v107 || *(_QWORD *)(v107 + 8) )
                          goto LABEL_41;
                        LOWORD(v140) = 257;
                        v137 = 257;
                        v108 = sub_1B47DD0(v144.m128i_i64, v130, v136);
                        v80 = sub_1B47F80(v144.m128i_i64, (__int64)v108, v113, &v138);
                        goto LABEL_144;
                      }
                      LOWORD(v140) = 257;
                      v80 = sub_1B47DD0(v144.m128i_i64, v130, &v138);
                      v121 = 0;
LABEL_144:
                      if ( !v80 || *((_BYTE *)v80 + 16) == 77 && v80 == (_QWORD *)a1 )
                        goto LABEL_41;
                      v131 = (__int64)v80;
                      sub_164B7C0((__int64)v80, a1);
                      sub_164D160(a1, v131, a4, a5, a6, a7, v81, v82, a10, a11);
                      sub_15F20C0((_QWORD *)a1);
                      if ( v121 )
                      {
                        v83 = sub_157EE30(v126);
                        v84 = v83;
                        if ( v83 )
                          v84 = v83 - 24;
                        sub_15F22F0((_QWORD *)v121, v84);
                      }
                      goto LABEL_150;
                    }
LABEL_125:
                    if ( sub_1455000(v113 + 24) )
                    {
                      v72 = 0;
                      v116 = v134;
                      if ( *(_BYTE *)(v122 + 16) > 0x17u )
                        v72 = v122;
                      v121 = v72;
                      v73 = sub_157EBA0(v134);
                      if ( *(_BYTE *)(v73 + 16) != 26 )
                        v73 = 0;
                      v74 = *(_QWORD *)(v116 + 48);
                      v75 = 0;
                      while ( v74 != v116 + 40 )
                      {
                        v74 = *(_QWORD *)(v74 + 8);
                        ++v75;
                      }
                      if ( v75 > 2 )
                        goto LABEL_41;
                      if ( !v73 )
                        goto LABEL_41;
                      if ( (*(_DWORD *)(v73 + 20) & 0xFFFFFFF) != 1 )
                        goto LABEL_41;
                      v76 = sub_157EBA0(v135);
                      v77 = sub_15F4DF0(v76, 0);
                      if ( !v121 )
                        goto LABEL_41;
                      if ( v77 != v134 )
                        goto LABEL_41;
                      if ( *(_BYTE *)(v121 + 16) == 77 )
                        goto LABEL_41;
                      if ( v134 != *(_QWORD *)(v121 + 40) )
                        goto LABEL_41;
                      if ( sub_1B46960(v121) )
                        goto LABEL_41;
                      v78 = *(_QWORD *)(v122 + 8);
                      if ( !v78 || *(_QWORD *)(v78 + 8) )
                        goto LABEL_41;
                      LOWORD(v140) = 257;
                      v137 = 257;
                      v79 = sub_1B47DD0(v144.m128i_i64, v130, v136);
                      v80 = sub_1B47BD0(v144.m128i_i64, (__int64)v79, v122, &v138);
                    }
                    else
                    {
                      if ( !sub_13D01C0(v113 + 24) )
                        goto LABEL_41;
                      v85 = 0;
                      v117 = v134;
                      if ( *(_BYTE *)(v122 + 16) > 0x17u )
                        v85 = v122;
                      v121 = v85;
                      v86 = sub_157EBA0(v134);
                      if ( *(_BYTE *)(v86 + 16) != 26 )
                        v86 = 0;
                      v87 = *(_QWORD *)(v117 + 48);
                      v88 = 0;
                      while ( v117 + 40 != v87 )
                      {
                        v87 = *(_QWORD *)(v87 + 8);
                        ++v88;
                      }
                      if ( !v86 )
                        goto LABEL_41;
                      if ( v88 > 2 )
                        goto LABEL_41;
                      if ( (*(_DWORD *)(v86 + 20) & 0xFFFFFFF) != 1 )
                        goto LABEL_41;
                      v89 = sub_157EBA0(v135);
                      v90 = sub_15F4DF0(v89, 0);
                      if ( !v121 )
                        goto LABEL_41;
                      if ( v90 != v134 )
                        goto LABEL_41;
                      if ( *(_BYTE *)(v121 + 16) == 77 )
                        goto LABEL_41;
                      if ( v134 != *(_QWORD *)(v121 + 40) )
                        goto LABEL_41;
                      if ( sub_1B46960(v121) )
                        goto LABEL_41;
                      v91 = *(_QWORD *)(v122 + 8);
                      if ( !v91 || *(_QWORD *)(v91 + 8) )
                        goto LABEL_41;
                      LOWORD(v140) = 257;
                      v80 = sub_1B47F80(v144.m128i_i64, v130, v122, &v138);
                    }
                    goto LABEL_144;
                  }
                  sub_1B47690((__int64)&v144, a1, 0, 0, 0);
                  if ( sub_1455000(v118) )
                  {
LABEL_173:
                    v92 = 0;
                    v124 = v135;
                    if ( *(_BYTE *)(v113 + 16) > 0x17u )
                      v92 = v113;
                    v121 = v92;
                    v93 = sub_157EBA0(v135);
                    if ( *(_BYTE *)(v93 + 16) != 26 )
                      v93 = 0;
                    v94 = *(_QWORD *)(v124 + 48);
                    v95 = 0;
                    while ( v94 != v124 + 40 )
                    {
                      v94 = *(_QWORD *)(v94 + 8);
                      ++v95;
                    }
                    if ( v95 > 2 )
                      goto LABEL_41;
                    if ( !v93 )
                      goto LABEL_41;
                    if ( (*(_DWORD *)(v93 + 20) & 0xFFFFFFF) != 1 )
                      goto LABEL_41;
                    v96 = sub_157EBA0(v134);
                    if ( (unsigned int)sub_15F4D60(v96) != 2 )
                      goto LABEL_41;
                    v97 = sub_157EBA0(v134);
                    v98 = sub_15F4DF0(v97, 1u);
                    if ( !v121 )
                      goto LABEL_41;
                    if ( v98 != v135 )
                      goto LABEL_41;
                    if ( *(_BYTE *)(v121 + 16) == 77 )
                      goto LABEL_41;
                    if ( v135 != *(_QWORD *)(v121 + 40) )
                      goto LABEL_41;
                    if ( sub_1B46960(v121) )
                      goto LABEL_41;
                    v99 = *(_QWORD *)(v113 + 8);
                    if ( !v99 || *(_QWORD *)(v99 + 8) )
                      goto LABEL_41;
                    LOWORD(v140) = 257;
                    v80 = sub_1B47BD0(v144.m128i_i64, v130, v113, &v138);
                    goto LABEL_144;
                  }
                  if ( sub_13D01C0(v118) )
                    goto LABEL_195;
                }
                else
                {
                  if ( v28 == 13 )
                  {
                    sub_1B47690((__int64)&v144, a1, 0, 0, 0);
                    goto LABEL_125;
                  }
                  sub_1B47690((__int64)&v144, a1, 0, 0, 0);
                }
LABEL_41:
                sub_17CD270(v144.m128i_i64);
              }
            }
          }
        }
      }
    }
  }
  v17 = *(_QWORD *)(v13 + 48);
  v18 = 4;
  if ( !v17 )
LABEL_220:
    BUG();
  while ( *(_BYTE *)(v17 - 8) == 77 )
  {
    if ( !--v18 )
      return 0;
    v17 = *(_QWORD *)(v17 + 8);
    if ( !v17 )
      goto LABEL_220;
  }
  v138 = 0;
  v139 = v143;
  v140 = v143;
  v141 = 4;
  v142 = 0;
  v29 = *(_QWORD *)(v13 + 48);
  v132 = dword_4FB7680;
  v133 = dword_4FB7680;
  v127 = a3;
  for ( i = v29; ; i = v33 )
  {
    if ( !i )
      BUG();
    v32 = i - 24;
    if ( *(_BYTE *)(i - 8) != 77 )
      break;
    v33 = *(_QWORD *)(i + 8);
    v144 = (__m128i)v127;
    v145 = 0;
    v146 = 0;
    v147 = i - 24;
    v34 = sub_13E3350(i - 24, &v144, 0, 1, v15);
    if ( v34 )
    {
      sub_164D160(i - 24, v34, a4, a5, a6, a7, v35, v36, a10, a11);
      sub_15F20C0((_QWORD *)(i - 24));
    }
    else
    {
      if ( (*(_BYTE *)(i - 1) & 0x40) != 0 )
        v37 = *(__int64 **)(i - 32);
      else
        v37 = (__int64 *)(v32 - 24LL * (*(_DWORD *)(i - 4) & 0xFFFFFFF));
      if ( !(unsigned __int8)sub_1B47330(*v37, v13, (__int64)&v138, (unsigned int *)&v132, a2, 0) )
        goto LABEL_55;
      v38 = (*(_BYTE *)(i - 1) & 0x40) != 0 ? *(_QWORD *)(i - 32) : v32 - 24LL * (*(_DWORD *)(i - 4) & 0xFFFFFFF);
      if ( !(unsigned __int8)sub_1B47330(*(_QWORD *)(v38 + 24), v13, (__int64)&v138, (unsigned int *)&v133, a2, 0) )
        goto LABEL_55;
    }
  }
  v39 = *(_QWORD *)(v13 + 48);
  if ( !v39 )
    BUG();
  if ( *(_BYTE *)(v39 - 8) == 77 )
  {
    v40 = v39 - 24;
    if ( sub_1642F90(*(_QWORD *)(v39 - 24), 1)
      && ((unsigned __int8)(*(_BYTE *)(sub_1455F60(v40, 0) + 16) - 35) <= 0x11u
       || (unsigned __int8)(*(_BYTE *)(sub_1455F60(v40, 1u) + 16) - 35) <= 0x11u
       || (unsigned __int8)(*(_BYTE *)(v130 + 16) - 35) <= 0x11u) )
    {
LABEL_55:
      v23 = 0;
      goto LABEL_56;
    }
    v41 = (__int64 *)sub_193FF80(v40);
    v42 = *v41;
    v43 = v41[1];
    if ( (*(_DWORD *)(sub_157EBA0(*v41) + 20) & 0xFFFFFFF) == 3 )
    {
      if ( (*(_DWORD *)(sub_157EBA0(v43) + 20) & 0xFFFFFFF) == 3 )
      {
        v43 = 0;
        v42 = 0;
        v123 = sub_157EBA0(0);
        sub_1B47690((__int64)&v144, v123, 0, 0, 0);
        v128 = 0;
LABEL_96:
        v120 = v42;
        v115 = v43;
        v56 = *(_QWORD *)(v13 + 48);
        if ( !v56 )
LABEL_222:
          BUG();
        while ( *(_BYTE *)(v56 - 8) == 77 )
        {
          v57 = (_QWORD *)(v56 - 24);
          v58 = (__int64 *)sub_193FF80(v56 - 24);
          v59 = *v58;
          v60 = sub_1455F60((__int64)v57, v135 == *v58);
          v61 = sub_1455F60((__int64)v57, v134 == v59);
          v137 = 257;
          v62 = sub_1B47760(v144.m128i_i64, v130, v60, v61, v136, v123);
          sub_164D160((__int64)v57, (__int64)v62, a4, a5, a6, a7, v63, v64, a10, a11);
          sub_164B7C0((__int64)v62, (__int64)v57);
          sub_15F20C0(v57);
          v56 = *(_QWORD *)(v13 + 48);
          if ( !v56 )
            goto LABEL_222;
        }
        v65 = (_QWORD *)sub_157EBA0(v128);
        sub_17050D0(v144.m128i_i64, (__int64)v65);
        v137 = 257;
        v66 = sub_1648A60(56, 1u);
        v67 = (__int64)v66;
        if ( v66 )
          sub_15F8320((__int64)v66, v13, 0);
        sub_1B43510(v67, v136, v144.m128i_i64[1], v145);
        sub_12A86E0(v144.m128i_i64, v67);
        sub_15F20C0(v65);
        if ( v120 )
        {
          v136[0] = *(_QWORD *)(v120 + 8);
          sub_15CDD40(v136);
          if ( !v136[0] )
          {
            v70 = *(_QWORD *)(*(_QWORD *)(v120 + 56) + 80LL);
            if ( !v70 || v120 != v70 - 24 )
            {
              v71 = sub_157EBA0(v120);
              if ( *(_BYTE *)(v71 + 16) == 26 && (*(_DWORD *)(v71 + 20) & 0xFFFFFFF) != 3 )
                sub_1593B40((_QWORD *)(v71 - 24), v120);
            }
          }
        }
        if ( v115 )
        {
          v136[0] = *(_QWORD *)(v115 + 8);
          sub_15CDD40(v136);
          if ( !v136[0] )
          {
            v68 = *(_QWORD *)(*(_QWORD *)(v115 + 56) + 80LL);
            if ( !v68 || v115 != v68 - 24 )
            {
              v69 = sub_157EBA0(v115);
              if ( *(_BYTE *)(v69 + 16) == 26 && (*(_DWORD *)(v69 + 20) & 0xFFFFFFF) != 3 )
                sub_1593B40((_QWORD *)(v69 - 24), v115);
            }
          }
        }
        v23 = 1;
        sub_17CD270(v144.m128i_i64);
        goto LABEL_56;
      }
      v42 = 0;
    }
    else
    {
      v144.m128i_i64[0] = *(_QWORD *)(v42 + 8);
      sub_15CDD40(v144.m128i_i64);
      v44 = sub_1648700(v144.m128i_i64[0]);
      v45 = *(_QWORD *)(v42 + 48);
      v128 = v44[5];
      while ( 1 )
      {
        if ( !v45 )
          BUG();
        if ( (unsigned int)*(unsigned __int8 *)(v45 - 8) - 25 <= 9 )
          break;
        if ( !sub_13A0E30((__int64)&v138, v45 - 24) && !sub_1B44350(v45) )
          goto LABEL_55;
        v45 = *(_QWORD *)(v45 + 8);
      }
      if ( (*(_DWORD *)(sub_157EBA0(v43) + 20) & 0xFFFFFFF) == 3 )
      {
        v43 = 0;
        v123 = sub_157EBA0(v128);
        sub_1B47690((__int64)&v144, v123, 0, 0, 0);
        goto LABEL_81;
      }
    }
    v144.m128i_i64[0] = *(_QWORD *)(v43 + 8);
    sub_15CDD40(v144.m128i_i64);
    v46 = sub_1648700(v144.m128i_i64[0]);
    v47 = *(_QWORD *)(v43 + 48);
    v128 = v46[5];
    while ( 1 )
    {
      if ( !v47 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v47 - 8) - 25 <= 9 )
        break;
      if ( !sub_13A0E30((__int64)&v138, v47 - 24) && !sub_1B44350(v47) )
        goto LABEL_55;
      v47 = *(_QWORD *)(v47 + 8);
    }
    v123 = sub_157EBA0(v128);
    sub_1B47690((__int64)&v144, v123, 0, 0, 0);
    if ( !v42 )
    {
      v119 = v128 + 40;
      v114 = (unsigned __int64 *)(v123 + 24);
LABEL_89:
      v52 = *(__int64 **)(v43 + 48);
      v53 = v52;
      if ( (__int64 *)(v43 + 40) != v52 )
      {
        do
        {
          v54 = (__int64)(v52 - 3);
          if ( !v52 )
            v54 = 0;
          sub_1624960(v54, 0, 0);
          v52 = (__int64 *)v52[1];
        }
        while ( (__int64 *)(v43 + 40) != v52 );
        v53 = *(__int64 **)(v43 + 48);
      }
      v112 = v53;
      v55 = (unsigned __int64 *)(sub_157EBA0(v43) + 24);
      if ( v55 != (unsigned __int64 *)v112 )
        sub_1AD56A0(v119, v114, v43 + 40, v112, v55);
      goto LABEL_96;
    }
LABEL_81:
    v48 = *(__int64 **)(v42 + 48);
    v49 = v48;
    if ( (__int64 *)(v42 + 40) != v48 )
    {
      do
      {
        v50 = (__int64)(v48 - 3);
        if ( !v48 )
          v50 = 0;
        sub_1624960(v50, 0, 0);
        v48 = (__int64 *)v48[1];
      }
      while ( (__int64 *)(v42 + 40) != v48 );
      v49 = *(__int64 **)(v42 + 48);
    }
    v111 = v49;
    v119 = v128 + 40;
    v51 = (unsigned __int64 *)(sub_157EBA0(v42) + 24);
    v114 = (unsigned __int64 *)(v123 + 24);
    if ( v51 != (unsigned __int64 *)v111 )
      sub_1AD56A0(v119, v114, v42 + 40, v111, v51);
    if ( !v43 )
      goto LABEL_96;
    goto LABEL_89;
  }
  v23 = 1;
LABEL_56:
  if ( v140 != v139 )
    _libc_free((unsigned __int64)v140);
  return v23;
}
