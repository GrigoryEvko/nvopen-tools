// Function: sub_32BDCD0
// Address: 0x32bdcd0
//
__int64 __fastcall sub_32BDCD0(__int64 *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // rax
  unsigned int v6; // edx
  int v7; // r11d
  int v8; // r14d
  __int64 v9; // rsi
  __int64 v10; // r13
  __int64 v11; // rcx
  __int64 v12; // rax
  __int16 v13; // cx
  __int64 v14; // rax
  bool v15; // al
  int v16; // r10d
  __int64 v17; // rax
  int v18; // ecx
  const __m128i *v19; // rax
  char v20; // al
  __int64 v21; // rax
  int v22; // r14d
  __int64 v23; // rax
  __int64 v24; // r15
  __m128i v25; // rax
  unsigned int v26; // r13d
  __int64 v27; // rsi
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  int v31; // r10d
  __int64 v32; // rax
  unsigned __int64 v33; // rax
  int v34; // r11d
  unsigned __int64 v35; // rax
  int v36; // eax
  __int64 v37; // rdx
  unsigned int v38; // r13d
  __int64 v39; // rdx
  __int64 v40; // rdx
  unsigned int v41; // r15d
  __int64 v42; // rdx
  _BYTE *v43; // r8
  unsigned int v44; // eax
  char v45; // al
  __int64 v46; // rdi
  unsigned __int64 v47; // rax
  unsigned __int64 v48; // rdx
  char v49; // dl
  bool v50; // zf
  char v51; // al
  __int64 *v52; // rdi
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r8
  int v57; // r10d
  char v58; // al
  __int64 v59; // r15
  int v60; // r10d
  unsigned __int64 v61; // rax
  unsigned __int64 v62; // rdx
  char v63; // dl
  char v64; // al
  char v65; // r13
  __int64 v66; // rdx
  __int16 v67; // ax
  __int64 v68; // r15
  __int64 v69; // rsi
  __int64 v70; // rdx
  __int128 v71; // rax
  __int64 v72; // r15
  int v73; // edx
  __int64 v74; // rdi
  __int64 *v75; // rsi
  __int64 v76; // rbx
  __int64 v77; // r8
  __int64 v78; // r9
  __int64 v79; // r8
  __int64 v80; // r9
  __int64 v81; // r8
  __int64 v82; // r9
  __int64 v83; // rdx
  __int64 v84; // rdi
  __int128 v85; // [rsp-20h] [rbp-1D8h]
  __int64 v86; // [rsp+10h] [rbp-1A8h]
  unsigned __int8 (__fastcall *v87)(__int64, _QWORD, __int64, _QWORD, __int64, _QWORD, _QWORD, _QWORD, __m128i *, __int64 *); // [rsp+18h] [rbp-1A0h]
  __int64 v88; // [rsp+20h] [rbp-198h]
  unsigned int v89; // [rsp+20h] [rbp-198h]
  int v90; // [rsp+20h] [rbp-198h]
  __int64 v91; // [rsp+28h] [rbp-190h]
  unsigned __int8 v92; // [rsp+28h] [rbp-190h]
  int v93; // [rsp+28h] [rbp-190h]
  unsigned int v94; // [rsp+30h] [rbp-188h]
  __int16 v95; // [rsp+30h] [rbp-188h]
  __int64 v96; // [rsp+30h] [rbp-188h]
  unsigned int v97; // [rsp+38h] [rbp-180h]
  unsigned int v98; // [rsp+38h] [rbp-180h]
  unsigned int v99; // [rsp+3Ch] [rbp-17Ch]
  char v100; // [rsp+3Ch] [rbp-17Ch]
  unsigned int v101; // [rsp+40h] [rbp-178h]
  unsigned int v102; // [rsp+40h] [rbp-178h]
  unsigned int v103; // [rsp+50h] [rbp-168h]
  int v104; // [rsp+50h] [rbp-168h]
  unsigned int v105; // [rsp+50h] [rbp-168h]
  int v106; // [rsp+50h] [rbp-168h]
  int v107; // [rsp+50h] [rbp-168h]
  unsigned int v108; // [rsp+50h] [rbp-168h]
  int v109; // [rsp+50h] [rbp-168h]
  __int64 v110; // [rsp+58h] [rbp-160h]
  int v111; // [rsp+60h] [rbp-158h]
  unsigned int v112; // [rsp+60h] [rbp-158h]
  __int64 v113; // [rsp+60h] [rbp-158h]
  int v114; // [rsp+68h] [rbp-150h]
  __int64 v115; // [rsp+68h] [rbp-150h]
  int v116; // [rsp+70h] [rbp-148h]
  __int64 v117; // [rsp+70h] [rbp-148h]
  __int64 v118; // [rsp+78h] [rbp-140h]
  __int128 v119; // [rsp+80h] [rbp-138h]
  int v120; // [rsp+80h] [rbp-138h]
  int v121; // [rsp+90h] [rbp-128h]
  _QWORD *v122; // [rsp+90h] [rbp-128h]
  __int64 v123; // [rsp+90h] [rbp-128h]
  __int64 v124; // [rsp+A0h] [rbp-118h]
  __int64 v125; // [rsp+A8h] [rbp-110h]
  __int64 v126; // [rsp+A8h] [rbp-110h]
  unsigned int v127; // [rsp+B0h] [rbp-108h] BYREF
  __int64 v128; // [rsp+B8h] [rbp-100h]
  __m128i v129; // [rsp+C0h] [rbp-F8h] BYREF
  __int64 v130; // [rsp+D0h] [rbp-E8h] BYREF
  int v131; // [rsp+D8h] [rbp-E0h]
  unsigned int v132; // [rsp+E0h] [rbp-D8h] BYREF
  __int64 v133; // [rsp+E8h] [rbp-D0h]
  __int64 v134[2]; // [rsp+F0h] [rbp-C8h] BYREF
  __int64 v135; // [rsp+100h] [rbp-B8h] BYREF
  int v136; // [rsp+108h] [rbp-B0h]
  __int64 v137; // [rsp+110h] [rbp-A8h]
  int v138; // [rsp+118h] [rbp-A0h]
  __int128 v139; // [rsp+120h] [rbp-98h] BYREF
  __int64 v140; // [rsp+130h] [rbp-88h]
  __int128 v141; // [rsp+140h] [rbp-78h] BYREF
  __int64 v142; // [rsp+150h] [rbp-68h]
  __m128i v143; // [rsp+160h] [rbp-58h] BYREF
  __m128i v144; // [rsp+170h] [rbp-48h]
  __int64 v145; // [rsp+1B0h] [rbp-8h] BYREF

  if ( (*(_BYTE *)(*(_QWORD *)(a2 + 112) + 37LL) & 0xF) != 0 )
    return 0;
  if ( (*(_BYTE *)(a2 + 32) & 8) != 0 )
    return 0;
  v4 = *(_QWORD *)(a2 + 40);
  v6 = *(_DWORD *)(v4 + 48);
  v7 = *(_DWORD *)(v4 + 8);
  v8 = *(_DWORD *)(v4 + 88);
  v125 = *(_QWORD *)v4;
  v9 = *(_QWORD *)(v4 + 80);
  v10 = *(_QWORD *)(v4 + 40);
  v124 = v10;
  v11 = *(_QWORD *)(v4 + 88);
  v12 = *(_QWORD *)(v10 + 48) + 16LL * v6;
  v116 = v9;
  v121 = v11;
  v13 = *(_WORD *)v12;
  v14 = *(_QWORD *)(v12 + 8);
  LOWORD(v127) = v13;
  v128 = v14;
  if ( (*(_BYTE *)(a2 + 33) & 4) != 0 )
    return 0;
  if ( v13 )
  {
    if ( (unsigned __int16)(v13 - 17) <= 0xD3u )
      return 0;
  }
  else
  {
    v103 = v6;
    v114 = v7;
    v15 = sub_30070B0((__int64)&v127);
    v7 = v114;
    v6 = v103;
    if ( v15 )
      return 0;
  }
  v16 = *(_DWORD *)(v10 + 24);
  if ( (unsigned int)(v16 - 186) > 2 )
    return 0;
  v17 = *(_QWORD *)(v10 + 56);
  if ( !v17 )
    return 0;
  v18 = 1;
  do
  {
    if ( *(_DWORD *)(v17 + 8) == v6 )
    {
      if ( !v18 )
        return 0;
      v17 = *(_QWORD *)(v17 + 32);
      if ( !v17 )
        goto LABEL_21;
      if ( v6 == *(_DWORD *)(v17 + 8) )
        return 0;
      v18 = 0;
    }
    v17 = *(_QWORD *)(v17 + 32);
  }
  while ( v17 );
  if ( v18 == 1 )
    return 0;
LABEL_21:
  if ( v16 != 187 || !(_BYTE)qword_5037D08 )
    goto LABEL_73;
  v106 = v7;
  v33 = sub_3287910(**(_QWORD **)(v10 + 40), *(_QWORD *)(*(_QWORD *)(v10 + 40) + 8LL), v9, v8, v125);
  v34 = v106;
  if ( !(_DWORD)v33
    || (result = sub_3273860(
                   v33,
                   HIDWORD(v33),
                   *(_QWORD *)(*(_QWORD *)(v10 + 40) + 40LL),
                   *(_QWORD *)(*(_QWORD *)(v10 + 40) + 48LL),
                   a2,
                   a1),
        v34 = v106,
        !result) )
  {
    v107 = v34;
    v35 = sub_3287910(
            *(_QWORD *)(*(_QWORD *)(v10 + 40) + 40LL),
            *(_QWORD *)(*(_QWORD *)(v10 + 40) + 48LL),
            v9,
            v8,
            v125);
    v16 = 187;
    v7 = v107;
    if ( !(_DWORD)v35
      || (result = sub_3273860(
                     v35,
                     HIDWORD(v35),
                     **(_QWORD **)(v10 + 40),
                     *(_QWORD *)(*(_QWORD *)(v10 + 40) + 8LL),
                     a2,
                     a1),
          v16 = 187,
          v7 = v107,
          !result) )
    {
LABEL_73:
      if ( (_BYTE)qword_5037EC8 )
      {
        v19 = *(const __m128i **)(v10 + 40);
        if ( *(_DWORD *)(v19[2].m128i_i64[1] + 24) == 11 )
        {
          v111 = v7;
          v104 = v16;
          v129 = _mm_loadu_si128(v19);
          if ( *(_DWORD *)(v129.m128i_i64[0] + 24) == 298 && (*(_BYTE *)(v129.m128i_i64[0] + 33) & 0xC) == 0 )
          {
            v115 = v129.m128i_i64[0];
            if ( (*(_WORD *)(v129.m128i_i64[0] + 32) & 0x380) == 0 )
            {
              v20 = sub_3286E00(&v129);
              if ( v125 == v115 && v111 == 1 )
              {
                if ( v20 )
                {
                  v21 = *(_QWORD *)(v115 + 40);
                  if ( v9 == *(_QWORD *)(v21 + 40) && v8 == *(_DWORD *)(v21 + 48) )
                  {
                    v22 = sub_2EAC1E0(*(_QWORD *)(v115 + 112));
                    if ( v22 == (unsigned int)sub_2EAC1E0(*(_QWORD *)(a2 + 112)) )
                    {
                      v23 = *(_QWORD *)(v10 + 40);
                      v24 = *(_QWORD *)(v23 + 40);
                      v25.m128i_i64[0] = sub_3262090(v24, *(_DWORD *)(v23 + 48));
                      v143 = v25;
                      v26 = sub_CA1930(&v143);
                      v27 = *(_QWORD *)(v24 + 96) + 24LL;
                      sub_9865C0((__int64)&v130, v27);
                      v31 = v104;
                      if ( v104 == 186 )
                      {
                        sub_987160((__int64)&v130, v27, v28, v29, v30);
                        v31 = 186;
                      }
                      v105 = v31;
                      if ( !sub_D94970((__int64)&v130, 0) && !sub_986760((__int64)&v130) )
                      {
                        v112 = v105;
                        v101 = v26;
                        v99 = sub_D949C0((__int64)&v130) & 0xFFFFFFF8;
                        v36 = sub_9871A0((__int64)&v130);
                        v94 = (v131 - 1 - v36) | 7;
                        v108 = sub_AF1560(v94 - v99);
                        v132 = sub_327FC40(*(_QWORD **)(*a1 + 64), v108);
                        v133 = v37;
                        v38 = v112;
                        while ( v108 < v101 )
                        {
                          v143.m128i_i64[0] = sub_3285A80((unsigned __int16 *)&v132);
                          v143.m128i_i64[1] = v40;
                          if ( sub_CA1930(&v143) == v108 )
                          {
                            v88 = v133;
                            v97 = v132;
                            v91 = a1[1];
                            if ( (unsigned __int8)sub_328A020(v91, v112, v132, v133, 0) )
                            {
                              if ( (_BYTE)qword_5037DE8
                                || (*(unsigned __int8 (__fastcall **)(__int64, __int64, _QWORD, __int64, _QWORD, __int64))(*(_QWORD *)v91 + 1656LL))(
                                     v91,
                                     a2,
                                     v127,
                                     v128,
                                     v97,
                                     v88) )
                              {
                                v41 = 0;
                                v143.m128i_i64[0] = sub_3285A80((unsigned __int16 *)&v127);
                                v143.m128i_i64[1] = v42;
                                v102 = v143.m128i_i32[0];
                                while ( v102 >= v41 + v108 && v41 <= v99 )
                                {
                                  if ( v94 <= v41 + v108 )
                                  {
                                    v43 = (_BYTE *)sub_2E79000(*(__int64 **)(*a1 + 40));
                                    v44 = v41;
                                    if ( *v43 )
                                      v44 = v102 - v108 - v41;
                                    v143.m128i_i32[0] = 0;
                                    v113 = v44 >> 3;
                                    v45 = sub_2EAC4F0(*(_QWORD *)(v115 + 112));
                                    v86 = a1[1];
                                    v46 = *(_QWORD *)(v115 + 112);
                                    v47 = ((1LL << v45) | v113) & -((1LL << v45) | v113);
                                    _BitScanReverse64(&v48, v47);
                                    v98 = *(unsigned __int16 *)(v46 + 32);
                                    v49 = v48 ^ 0x3F;
                                    v50 = v47 == 0;
                                    v51 = 64;
                                    if ( !v50 )
                                      v51 = v49;
                                    v92 = 63 - v51;
                                    v87 = *(unsigned __int8 (__fastcall **)(__int64, _QWORD, __int64, _QWORD, __int64, _QWORD, _QWORD, _QWORD, __m128i *, __int64 *))(*(_QWORD *)a1[1] + 824LL);
                                    v89 = sub_2EAC1E0(v46);
                                    v52 = *(__int64 **)(*a1 + 40);
                                    v53 = sub_2E79000(v52);
                                    if ( v87(v86, *(_QWORD *)(*a1 + 64), v53, v132, v133, v89, v92, v98, &v143, v52) )
                                    {
                                      if ( v143.m128i_i32[0] )
                                      {
                                        sub_9865C0((__int64)&v143, (__int64)&v130);
                                        sub_986C30((__int64)&v143, v41);
                                        sub_C44740((__int64)v134, (char **)&v143, v108);
                                        sub_969240(v143.m128i_i64);
                                        v57 = v38;
                                        if ( v38 == 186 )
                                        {
                                          sub_987160((__int64)v134, (__int64)&v143, v54, v55, v56);
                                          v57 = 186;
                                        }
                                        v109 = v57;
                                        v58 = sub_2EAC4F0(*(_QWORD *)(v115 + 112));
                                        v59 = *a1;
                                        v60 = v109;
                                        v61 = (v113 | (1LL << v58)) & -(v113 | (1LL << v58));
                                        _BitScanReverse64(&v62, v61);
                                        v63 = v62 ^ 0x3F;
                                        v50 = v61 == 0;
                                        v64 = 64;
                                        if ( !v50 )
                                          v64 = v63;
                                        v100 = 63 - v64;
                                        v65 = 63 - v64;
                                        v143.m128i_i64[0] = *(_QWORD *)(v115 + 80);
                                        if ( v143.m128i_i64[0] )
                                        {
                                          sub_325F5D0(v143.m128i_i64);
                                          v60 = v109;
                                        }
                                        LOBYTE(v138) = 0;
                                        v143.m128i_i32[2] = *(_DWORD *)(v115 + 72);
                                        v90 = v60;
                                        v137 = v113;
                                        v117 = sub_3409320(v59, v116, v121, v113, v138, (unsigned int)&v145 - 80, 0);
                                        v118 = v66;
                                        sub_9C6650(&v143);
                                        HIBYTE(v67) = 1;
                                        v68 = *a1;
                                        LOBYTE(v67) = v65;
                                        v69 = *(_QWORD *)(v115 + 112);
                                        v95 = v67;
                                        v143 = _mm_loadu_si128((const __m128i *)(v69 + 40));
                                        v144 = _mm_loadu_si128((const __m128i *)(v69 + 56));
                                        v93 = *(unsigned __int16 *)(v69 + 32);
                                        sub_327C6E0((__int64)&v139, (__int64 *)v69, v113);
                                        v122 = *(_QWORD **)(v115 + 40);
                                        sub_3285E70((__int64)&v141, v129.m128i_i64[0]);
                                        v123 = sub_33F1F00(
                                                 v68,
                                                 v132,
                                                 v133,
                                                 (unsigned int)&v145 - 112,
                                                 *v122,
                                                 v122[1],
                                                 v117,
                                                 v118,
                                                 v139,
                                                 v140,
                                                 v95,
                                                 v93,
                                                 (__int64)&v143,
                                                 0);
                                        v110 = v70;
                                        sub_9C6650(&v141);
                                        v96 = *a1;
                                        sub_3285E70((__int64)&v143, v124);
                                        *(_QWORD *)&v71 = sub_34007B0(
                                                            v96,
                                                            (unsigned int)v134,
                                                            (unsigned int)&v145 - 80,
                                                            v132,
                                                            v133,
                                                            0,
                                                            0);
                                        v119 = v71;
                                        sub_3285E70((__int64)&v141, v124);
                                        *((_QWORD *)&v85 + 1) = v110;
                                        *(_QWORD *)&v85 = v123;
                                        v72 = sub_3406EB0(
                                                v96,
                                                v90,
                                                (unsigned int)&v145 - 112,
                                                v132,
                                                v133,
                                                DWORD2(v119),
                                                v85,
                                                v119);
                                        v120 = v73;
                                        sub_9C6650(&v141);
                                        sub_9C6650(&v143);
                                        v74 = *a1;
                                        v75 = *(__int64 **)(a2 + 112);
                                        v143 = 0u;
                                        v144 = 0u;
                                        sub_327C6E0((__int64)&v141, v75, v113);
                                        v135 = *(_QWORD *)(a2 + 80);
                                        if ( v135 )
                                          sub_325F5D0(&v135);
                                        v136 = *(_DWORD *)(a2 + 72);
                                        v76 = sub_33F4560(
                                                v74,
                                                v125,
                                                1,
                                                (unsigned int)&v145 - 176,
                                                v72,
                                                v120,
                                                v117,
                                                v118,
                                                v141,
                                                v142,
                                                v100,
                                                0,
                                                (__int64)&v143);
                                        sub_9C6650(&v135);
                                        sub_32B3E80((__int64)a1, v117, 1, 0, v77, v78);
                                        sub_32B3E80((__int64)a1, v123, 1, 0, v79, v80);
                                        sub_32B3E80((__int64)a1, v72, 1, 0, v81, v82);
                                        v83 = *(_QWORD *)(*a1 + 768);
                                        v144.m128i_i64[0] = *a1;
                                        v143.m128i_i64[1] = v83;
                                        *(_QWORD *)(v144.m128i_i64[0] + 768) = &v143;
                                        v84 = *a1;
                                        v143.m128i_i64[0] = (__int64)off_4A360B8;
                                        v144.m128i_i64[1] = (__int64)a1;
                                        sub_34161C0(v84, v115, 1, v123, 1);
                                        *(_QWORD *)(v144.m128i_i64[0] + 768) = v143.m128i_i64[1];
                                        sub_969240(v134);
                                        v32 = v76;
                                        goto LABEL_37;
                                      }
                                    }
                                  }
                                  v41 += 8;
                                }
                                break;
                              }
                            }
                          }
                          v108 = sub_AF1560(v108);
                          v132 = sub_327FC40(*(_QWORD **)(*a1 + 64), v108);
                          v133 = v39;
                        }
                      }
                      v32 = 0;
LABEL_37:
                      v126 = v32;
                      sub_969240(&v130);
                      return v126;
                    }
                  }
                }
              }
            }
          }
        }
      }
      return 0;
    }
  }
  return result;
}
