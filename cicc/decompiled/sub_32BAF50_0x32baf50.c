// Function: sub_32BAF50
// Address: 0x32baf50
//
__int64 __fastcall sub_32BAF50(__int64 *a1, __int64 a2)
{
  const __m128i *v4; // rax
  __int64 v5; // r14
  __int64 *v6; // r15
  __int64 v7; // rax
  __int64 v8; // rsi
  __int16 *v9; // rax
  __int16 v10; // cx
  __int64 v11; // rax
  __int32 v12; // ecx
  __int64 v13; // rbx
  __int64 v14; // rdx
  int v15; // ebx
  _QWORD *v16; // rdi
  int v17; // esi
  __int64 v18; // rax
  __int64 v19; // r12
  __int64 v20; // rsi
  _QWORD *v22; // rdi
  __m128i v23; // xmm2
  __int64 v24; // rcx
  __int64 v25; // r8
  int v26; // r9d
  __int64 v27; // rdx
  __int64 v28; // rcx
  bool v29; // al
  _QWORD *v30; // rdi
  __int64 v31; // r12
  __int128 v32; // rax
  int v33; // r9d
  unsigned int v34; // ecx
  int v35; // r8d
  unsigned int v36; // edx
  __int64 v37; // rax
  char v38; // al
  __int64 v39; // r9
  bool v40; // zf
  int v41; // eax
  char v42; // al
  __int64 v43; // rax
  __int128 v44; // rax
  __int64 v45; // rcx
  __int64 v46; // r9
  __int64 v47; // r8
  __int128 v48; // rax
  int v49; // r9d
  __int64 v50; // rax
  __int64 v51; // rsi
  __m128i v52; // xmm5
  __m128i v53; // xmm6
  __int16 *v54; // rax
  __int16 v55; // dx
  __int64 v56; // rax
  _QWORD *v57; // rcx
  __int64 v58; // r14
  __int64 v59; // rax
  char v60; // dl
  __int64 *v61; // rdi
  __int64 v62; // rax
  _QWORD *v63; // rdi
  __int64 v64; // rax
  char v65; // bl
  __int64 (__fastcall *v66)(__int64, __int64, __int64, __int64, unsigned int); // rax
  __int64 v67; // rdx
  __int64 v68; // rax
  int v69; // eax
  __int64 (*v70)(); // rax
  __int64 v71; // r8
  __int64 v72; // r9
  __int64 v73; // r14
  __int64 (*v74)(); // rax
  __int64 v75; // rax
  char v76; // al
  _QWORD *v77; // rax
  int v78; // ecx
  __int64 v79; // rbx
  int v80; // edx
  __int64 v81; // rdx
  __int64 v82; // rax
  __int64 v83; // rax
  int v84; // ecx
  __int128 v85; // rax
  int v86; // r9d
  __int128 v87; // rax
  __int64 v88; // rbx
  char v89; // al
  __int128 v90; // rax
  char v91; // al
  __int64 v92; // rax
  __int64 v93; // rbx
  _DWORD *v94; // r14
  void *v95; // rax
  int v96; // r9d
  void *v97; // r14
  __int64 v98; // rax
  __int64 v99; // rax
  int v100; // r9d
  __int64 v101; // r12
  __int128 v102; // rax
  int v103; // r9d
  __int64 v104; // rax
  __int64 v105; // rax
  __int64 v106; // rbx
  int v107; // r9d
  __int64 v108; // rax
  _QWORD *i; // rbx
  __int64 v110; // rdi
  int v111; // r9d
  __int64 v112; // r14
  __int128 v113; // rax
  int v114; // r9d
  int v115; // r9d
  __int64 v116; // r14
  __int128 v117; // rax
  int v118; // r9d
  __int128 v119; // [rsp-20h] [rbp-1E0h]
  __int128 v120; // [rsp-10h] [rbp-1D0h]
  __int128 v121; // [rsp-10h] [rbp-1D0h]
  __int128 v122; // [rsp-10h] [rbp-1D0h]
  __int128 v123; // [rsp-10h] [rbp-1D0h]
  __int128 v124; // [rsp-10h] [rbp-1D0h]
  __int64 v125; // [rsp-8h] [rbp-1C8h]
  __int64 v126; // [rsp+8h] [rbp-1B8h]
  void *v127; // [rsp+8h] [rbp-1B8h]
  __int64 v128; // [rsp+18h] [rbp-1A8h]
  __int64 v129; // [rsp+20h] [rbp-1A0h]
  unsigned int v130; // [rsp+20h] [rbp-1A0h]
  unsigned int v131; // [rsp+20h] [rbp-1A0h]
  __int64 v132; // [rsp+30h] [rbp-190h]
  __int64 v133; // [rsp+30h] [rbp-190h]
  __int128 v134; // [rsp+30h] [rbp-190h]
  __int128 v135; // [rsp+30h] [rbp-190h]
  __int64 v136; // [rsp+30h] [rbp-190h]
  int v137; // [rsp+30h] [rbp-190h]
  unsigned __int32 v138; // [rsp+44h] [rbp-17Ch]
  bool v139; // [rsp+44h] [rbp-17Ch]
  __int64 v140; // [rsp+48h] [rbp-178h]
  __int64 v141; // [rsp+48h] [rbp-178h]
  __int64 v142; // [rsp+48h] [rbp-178h]
  __int16 v143; // [rsp+50h] [rbp-170h]
  __int64 v144; // [rsp+50h] [rbp-170h]
  __int64 v145; // [rsp+50h] [rbp-170h]
  __int128 v146; // [rsp+50h] [rbp-170h]
  __int64 v147; // [rsp+50h] [rbp-170h]
  __m128i v148; // [rsp+50h] [rbp-170h]
  __int64 v149; // [rsp+50h] [rbp-170h]
  char v150; // [rsp+63h] [rbp-15Dh] BYREF
  int v151; // [rsp+64h] [rbp-15Ch] BYREF
  int v152; // [rsp+68h] [rbp-158h] BYREF
  int v153; // [rsp+6Ch] [rbp-154h] BYREF
  __int128 v154; // [rsp+70h] [rbp-150h] BYREF
  unsigned int v155; // [rsp+80h] [rbp-140h] BYREF
  __int64 v156; // [rsp+88h] [rbp-138h]
  __int64 v157; // [rsp+90h] [rbp-130h] BYREF
  int v158; // [rsp+98h] [rbp-128h]
  unsigned int v159; // [rsp+A0h] [rbp-120h] BYREF
  __int64 v160; // [rsp+A8h] [rbp-118h]
  __int64 v161; // [rsp+B0h] [rbp-110h] BYREF
  int v162; // [rsp+B8h] [rbp-108h]
  _QWORD *v163; // [rsp+C0h] [rbp-100h] BYREF
  int v164; // [rsp+C8h] [rbp-F8h]
  __int64 v165; // [rsp+D0h] [rbp-F0h]
  void *v166; // [rsp+E0h] [rbp-E0h] BYREF
  _QWORD *v167; // [rsp+E8h] [rbp-D8h]
  __m128i v168; // [rsp+100h] [rbp-C0h] BYREF
  _DWORD *v169; // [rsp+110h] [rbp-B0h]
  __int64 *v170; // [rsp+118h] [rbp-A8h]
  unsigned int *v171; // [rsp+120h] [rbp-A0h]

  v4 = *(const __m128i **)(a2 + 40);
  v5 = v4[2].m128i_i64[1];
  v6 = (__int64 *)v4[3].m128i_i64[0];
  v140 = v5;
  v138 = v4[3].m128i_u32[0];
  v154 = (__int128)_mm_loadu_si128(v4);
  v7 = sub_33E1790(v5, v6, 1);
  v8 = *(_QWORD *)(a2 + 80);
  v129 = v7;
  v9 = *(__int16 **)(a2 + 48);
  v10 = *v9;
  v11 = *((_QWORD *)v9 + 1);
  v157 = v8;
  v143 = v10;
  LOWORD(v155) = v10;
  v156 = v11;
  if ( v8 )
    sub_B96E90((__int64)&v157, v8, 1);
  v12 = DWORD2(v154);
  v158 = *(_DWORD *)(a2 + 72);
  v13 = *(_QWORD *)*a1;
  v14 = *(_QWORD *)(*a1 + 1024);
  v163 = (_QWORD *)*a1;
  v132 = v13;
  v15 = *(_DWORD *)(a2 + 28);
  v165 = v14;
  v163[128] = &v163;
  v16 = (_QWORD *)*a1;
  v17 = *(_DWORD *)(a2 + 24);
  v164 = v15;
  v18 = sub_33FE9E0((_DWORD)v16, v17, v154, v12, v5, (_DWORD)v6, v15);
  if ( v18 )
    goto LABEL_4;
  v22 = (_QWORD *)*a1;
  v23 = _mm_loadu_si128((const __m128i *)&v154);
  v169 = (_DWORD *)v5;
  v170 = v6;
  v168 = v23;
  v18 = sub_3402EA0((_DWORD)v22, 98, (unsigned int)&v157, v155, v156, 0, (__int64)&v168, 2);
  if ( v18 )
    goto LABEL_4;
  if ( (unsigned __int8)sub_33E2470(*a1, v154, *((_QWORD *)&v154 + 1)) && !(unsigned __int8)sub_33E2470(*a1, v5, v6) )
  {
    v34 = v155;
    v35 = v156;
    v120 = v154;
    v36 = (unsigned int)&v157;
    *((_QWORD *)&v119 + 1) = v6;
    *(_QWORD *)&v119 = v5;
LABEL_26:
    v37 = sub_3406EB0(*a1, 98, v36, v34, v35, v33, v119, v120);
LABEL_27:
    v19 = v37;
    goto LABEL_5;
  }
  if ( v143 )
  {
    if ( (unsigned __int16)(v143 - 17) > 0xD3u )
      goto LABEL_13;
  }
  else if ( !sub_30070B0((__int64)&v155) )
  {
    goto LABEL_13;
  }
  v18 = sub_3295970(a1, a2, (__int64)&v157, v24, v25);
  if ( v18 )
    goto LABEL_4;
LABEL_13:
  v18 = sub_329BF20(a1, a2);
  if ( v18 )
    goto LABEL_4;
  if ( (*(_BYTE *)(v132 + 864) & 1) == 0 && (v15 & 0x800) == 0 )
    goto LABEL_16;
  v38 = sub_33E2470(*a1, v5, v6);
  v39 = v154;
  v40 = v38 == 0;
  v41 = *(_DWORD *)(v154 + 24);
  if ( !v40 && v41 == 98 )
  {
    v75 = *(_QWORD *)(v154 + 40);
    v126 = v154;
    v135 = (__int128)_mm_loadu_si128((const __m128i *)(v75 + 40));
    v148 = _mm_loadu_si128((const __m128i *)v75);
    v76 = sub_33E2470(*a1, v135, *((_QWORD *)&v135 + 1));
    v39 = v126;
    if ( v76 )
    {
      v89 = sub_33E2470(*a1, v148.m128i_i64[0], v148.m128i_i64[1]);
      v39 = v126;
      if ( !v89 )
      {
        *((_QWORD *)&v122 + 1) = v6;
        *(_QWORD *)&v122 = v5;
        *(_QWORD *)&v90 = sub_3406EB0(*a1, 98, (unsigned int)&v157, v155, v156, v126, v135, v122);
        v120 = v90;
        v119 = (__int128)v148;
        goto LABEL_100;
      }
    }
    v41 = *(_DWORD *)(v39 + 24);
  }
  if ( v41 == 96 )
  {
    v145 = v39;
    v42 = sub_3286E00(&v154);
    v39 = v145;
    if ( v42 )
    {
      v43 = *(_QWORD *)(v145 + 40);
      if ( *(_QWORD *)v43 == *(_QWORD *)(v43 + 40) && *(_DWORD *)(v43 + 8) == *(_DWORD *)(v43 + 48) )
      {
        *(_QWORD *)&v85 = sub_33FE730(*a1, &v157, v155, v156, 0, 2.0);
        *((_QWORD *)&v121 + 1) = v6;
        *(_QWORD *)&v121 = v5;
        *(_QWORD *)&v87 = sub_3406EB0(*a1, 98, (unsigned int)&v157, v155, v156, v86, v85, v121);
        v33 = v145;
        v120 = v87;
        v119 = *(_OWORD *)*(_QWORD *)(v145 + 40);
LABEL_100:
        v34 = v155;
        v36 = (unsigned int)&v157;
        v35 = v156;
        goto LABEL_26;
      }
    }
  }
  v18 = sub_328C120(a1, 0x179u, 0x62u, (int)&v157, v155, v156, v39, v5, v15);
  if ( v18 )
  {
LABEL_4:
    v19 = v18;
    goto LABEL_5;
  }
LABEL_16:
  if ( !v129 )
    goto LABEL_35;
  if ( (unsigned __int8)sub_11DB340((_DWORD **)(*(_QWORD *)(v129 + 96) + 24LL), 2.0) )
  {
    v37 = sub_3406EB0(*a1, 96, (unsigned int)&v157, v155, v156, v26, v154, v154);
    goto LABEL_27;
  }
  if ( (unsigned __int8)sub_11DB340((_DWORD **)(*(_QWORD *)(v129 + 96) + 24LL), -1.0) )
  {
    v27 = v155;
    v28 = v156;
    if ( !*((_BYTE *)a1 + 33)
      || (v130 = v155,
          v133 = v156,
          v144 = a1[1],
          v29 = sub_328D6E0(v144, 0x61u, v155),
          v30 = (_QWORD *)v144,
          v28 = v133,
          v27 = v130,
          v29) )
    {
      v31 = *a1;
      *(_QWORD *)&v32 = sub_33FE730(v31, &v157, v27, v28, 0, -0.0);
      v19 = sub_3405C90(v31, 97, (unsigned int)&v157, v155, v156, v15, v32, v154);
      goto LABEL_5;
    }
  }
  else
  {
LABEL_35:
    v30 = (_QWORD *)a1[1];
  }
  v151 = 2;
  v152 = 2;
  *(_QWORD *)&v44 = (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64, __int64, _QWORD, _QWORD, int *, _QWORD))(*v30 + 2264LL))(
                      v30,
                      v154,
                      *((_QWORD *)&v154 + 1),
                      *a1,
                      *((unsigned __int8 *)a1 + 33),
                      *((unsigned __int8 *)a1 + 35),
                      &v151,
                      0);
  v47 = v125;
  if ( (_QWORD)v44 )
  {
    v146 = v44;
    sub_3287B60((__int64)&v168, v44, *((__int64 *)&v44 + 1), v45, v125, v46);
    *(_QWORD *)&v48 = (*(__int64 (__fastcall **)(__int64, __int64, __int64 *, __int64, _QWORD, _QWORD, int *))(*(_QWORD *)a1[1] + 2264LL))(
                        a1[1],
                        v5,
                        v6,
                        *a1,
                        *((unsigned __int8 *)a1 + 33),
                        *((unsigned __int8 *)a1 + 35),
                        &v152);
    if ( (_QWORD)v48 && (!v151 || !v152) )
    {
      v19 = sub_3406EB0(*a1, 98, (unsigned int)&v157, v155, v156, v49, v146, v48);
      sub_33CF710(&v168);
      goto LABEL_5;
    }
    sub_33CF710(&v168);
  }
  if ( (v15 & 0xA0) == 0xA0 )
  {
    v149 = v154;
    if ( *(_DWORD *)(v154 + 24) == 205 )
    {
      if ( !sub_328D6E0(a1[1], 0xF5u, v155) )
        goto LABEL_42;
      v88 = v149;
      v149 = v5;
      v140 = v88;
    }
    else
    {
      if ( *(_DWORD *)(v5 + 24) != 205 || !sub_328D6E0(a1[1], 0xF5u, v155) )
        goto LABEL_42;
      v138 = DWORD2(v154);
    }
    v77 = *(_QWORD **)(v140 + 40);
    v142 = v77[5];
    v78 = *(_DWORD *)(v142 + 24);
    v79 = v77[10];
    v80 = *(_DWORD *)(v79 + 24);
    if ( (v78 == 36 || v78 == 12) && (v80 == 12 || v80 == 36) )
    {
      v81 = *v77;
      if ( *(_DWORD *)(*v77 + 24LL) == 208 )
      {
        v82 = *(_QWORD *)(v81 + 40);
        if ( *(_QWORD *)v82 == v149 && *(_DWORD *)(v82 + 8) == v138 )
        {
          v83 = *(_QWORD *)(v82 + 40);
          v84 = *(_DWORD *)(v83 + 24);
          if ( v84 == 12 || v84 == 36 )
          {
            v136 = v81;
            if ( (unsigned __int8)sub_11DB340((_DWORD **)(*(_QWORD *)(v83 + 96) + 24LL), 0.0) )
            {
              switch ( *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v136 + 40) + 80LL) + 96LL) )
              {
                case 2:
                case 3:
                case 0xA:
                case 0xB:
                case 0x12:
                case 0x13:
                  break;
                case 4:
                case 5:
                case 0xC:
                case 0xD:
                case 0x14:
                case 0x15:
                  v99 = v142;
                  v142 = v79;
                  v79 = v99;
                  break;
                default:
                  goto LABEL_42;
              }
              if ( (unsigned __int8)sub_11DB340((_DWORD **)(*(_QWORD *)(v142 + 96) + 24LL), -1.0) )
              {
                if ( (unsigned __int8)sub_11DB340((_DWORD **)(*(_QWORD *)(v79 + 96) + 24LL), 1.0) )
                {
                  v131 = v155;
                  v137 = v156;
                  if ( sub_328D6E0(a1[1], 0xF4u, v155) )
                  {
                    v101 = *a1;
                    *((_QWORD *)&v124 + 1) = v138 | (unsigned __int64)v6 & 0xFFFFFFFF00000000LL;
                    *(_QWORD *)&v124 = v149;
                    *(_QWORD *)&v102 = sub_33FAF80(v101, 245, (unsigned int)&v157, v131, v137, v131, v124);
                    v19 = sub_33FAF80(v101, 244, (unsigned int)&v157, v155, v156, v103, v102);
                    goto LABEL_5;
                  }
                }
              }
              if ( (unsigned __int8)sub_11DB340((_DWORD **)(*(_QWORD *)(v142 + 96) + 24LL), 1.0)
                && (unsigned __int8)sub_11DB340((_DWORD **)(*(_QWORD *)(v79 + 96) + 24LL), -1.0) )
              {
                *((_QWORD *)&v123 + 1) = v138 | (unsigned __int64)v6 & 0xFFFFFFFF00000000LL;
                *(_QWORD *)&v123 = v149;
                v19 = sub_33FAF80(*a1, 245, (unsigned int)&v157, v155, v156, v100, v123);
                goto LABEL_5;
              }
            }
          }
        }
      }
    }
  }
LABEL_42:
  v50 = *(_QWORD *)(a2 + 40);
  v51 = *(_QWORD *)(a2 + 80);
  v52 = _mm_loadu_si128((const __m128i *)v50);
  v53 = _mm_loadu_si128((const __m128i *)(v50 + 40));
  v147 = *(_QWORD *)v50;
  v141 = *(_QWORD *)(v50 + 40);
  v54 = *(__int16 **)(a2 + 48);
  v134 = (__int128)v52;
  v55 = *v54;
  v56 = *((_QWORD *)v54 + 1);
  v161 = v51;
  LOWORD(v159) = v55;
  v160 = v56;
  if ( v51 )
    sub_B96E90((__int64)&v161, v51, 1);
  v57 = (_QWORD *)*a1;
  v162 = *(_DWORD *)(a2 + 72);
  v58 = *v57;
  v59 = v141;
  v60 = *(_BYTE *)(*v57 + 864LL);
  if ( *(_DWORD *)(v147 + 24) == 96 )
    v59 = v147;
  if ( (v60 & 2) == 0 && (*(_BYTE *)(v59 + 28) & 0x40) == 0
    || *(_DWORD *)(v58 + 952) && (v60 & 1) == 0 && (*(_BYTE *)(a2 + 29) & 2) == 0 )
  {
    goto LABEL_48;
  }
  v63 = (_QWORD *)a1[1];
  if ( !*((_BYTE *)a1 + 33)
    || ((v64 = 1, (_WORD)v159 == 1) || (_WORD)v159 && (v64 = (unsigned __int16)v159, v63[(unsigned __int16)v159 + 14]))
    && (*((_BYTE *)v63 + 500 * v64 + 6564) & 0xFB) == 0 )
  {
    v74 = *(__int64 (**)())(*v63 + 1608LL);
    if ( v74 == sub_2FE3540 )
    {
      if ( (v60 & 1) == 0 )
        goto LABEL_48;
      v65 = 0;
      if ( !*((_BYTE *)a1 + 33) )
        goto LABEL_48;
    }
    else
    {
      v65 = ((__int64 (__fastcall *)(_QWORD *, _QWORD, _QWORD, __int64, __int64))v74)(v63, v57[5], v159, v160, v47);
      if ( (*(_BYTE *)(v58 + 864) & 1) == 0 || !*((_BYTE *)a1 + 33) )
        goto LABEL_101;
      v63 = (_QWORD *)a1[1];
    }
  }
  else
  {
    v65 = 0;
    if ( (v60 & 1) == 0 )
      goto LABEL_48;
  }
  v66 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, unsigned int))(*v63 + 1640LL);
  if ( v66 == sub_2FE40B0 )
  {
    v67 = **(unsigned __int16 **)(a2 + 48);
    v68 = 1;
    if ( (_WORD)v67 == 1 || (_WORD)v67 && (v68 = (unsigned __int16)v67, v63[v67 + 14]) )
    {
      if ( !*((_BYTE *)v63 + 500 * v68 + 6565) )
      {
        v69 = 151;
        goto LABEL_68;
      }
    }
LABEL_101:
    if ( v65 )
    {
      v63 = (_QWORD *)a1[1];
      v69 = 150;
      goto LABEL_68;
    }
LABEL_48:
    if ( v161 )
      sub_B91220((__int64)&v161, v161);
    v61 = a1;
    v19 = 0;
    v62 = sub_32899A0(v61, a2);
    if ( v62 )
      v19 = v62;
    goto LABEL_5;
  }
  if ( !(unsigned __int8)v66((__int64)v63, *a1, a2, (__int64)v57, v47) )
    goto LABEL_101;
  v63 = (_QWORD *)a1[1];
  v69 = 151;
LABEL_68:
  v153 = v69;
  v70 = *(__int64 (**)())(*v63 + 512LL);
  if ( v70 == sub_2FE30F0 )
  {
    v150 = 0;
    if ( *(_DWORD *)(v147 + 24) != 96 )
      goto LABEL_70;
LABEL_112:
    v92 = *(_QWORD *)(v147 + 56);
    if ( !v92 || *(_QWORD *)(v92 + 32) )
      goto LABEL_70;
    goto LABEL_114;
  }
  v91 = ((__int64 (__fastcall *)(_QWORD *, _QWORD, __int64, _QWORD *, __int64))v70)(v63, v159, v160, v57, v47);
  v150 = v91;
  if ( *(_DWORD *)(v147 + 24) != 96 )
    goto LABEL_70;
  if ( !v91 )
    goto LABEL_112;
LABEL_114:
  v128 = sub_33E1790(*(_QWORD *)(*(_QWORD *)(v147 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(v147 + 40) + 48LL), 1);
  if ( !v128 )
    goto LABEL_70;
  v93 = *(_QWORD *)(v128 + 96);
  v94 = sub_C33320();
  sub_C3B1B0((__int64)&v168, 1.0);
  sub_C407B0(&v166, v168.m128i_i64, v94);
  sub_C338F0((__int64)&v168);
  sub_C41640((__int64 *)&v166, *(_DWORD **)(v93 + 24), 1, (bool *)v168.m128i_i8);
  v127 = *(void **)(v93 + 24);
  v95 = sub_C33340();
  v139 = 0;
  v97 = v95;
  if ( v127 == v166 )
  {
    v110 = v93 + 24;
    if ( v95 == v127 )
      v139 = sub_C3E590(v110, (__int64)&v166);
    else
      v139 = sub_C33D00(v110, (__int64)&v166);
  }
  if ( v97 == v166 )
  {
    if ( v167 )
    {
      for ( i = &v167[3 * *(v167 - 1)]; v167 != i; sub_91D830(i) )
        i -= 3;
      j_j_j___libc_free_0_0((unsigned __int64)(i - 1));
    }
  }
  else
  {
    sub_C338F0((__int64)&v166);
  }
  if ( v139 )
  {
    v98 = sub_340F900(
            *a1,
            v153,
            (unsigned int)&v161,
            v159,
            v160,
            v96,
            *(_OWORD *)*(_QWORD *)(v147 + 40),
            *(_OWORD *)&v53,
            *(_OWORD *)&v53);
    v73 = v98;
  }
  else
  {
    if ( !(unsigned __int8)sub_11DB340((_DWORD **)(*(_QWORD *)(v128 + 96) + 24LL), -1.0) )
      goto LABEL_70;
    v112 = *a1;
    *(_QWORD *)&v113 = sub_33FAF80(*a1, 244, (unsigned int)&v161, v159, v160, v111, *(_OWORD *)&v53);
    v98 = sub_340F900(
            v112,
            v153,
            (unsigned int)&v161,
            v159,
            v160,
            v114,
            *(_OWORD *)*(_QWORD *)(v147 + 40),
            *(_OWORD *)&v53,
            v113);
    v73 = v98;
  }
  if ( v98 )
    goto LABEL_72;
LABEL_70:
  if ( *(_DWORD *)(v141 + 24) == 96 && (v150 || (v104 = *(_QWORD *)(v141 + 56)) != 0 && !*(_QWORD *)(v104 + 32)) )
  {
    v105 = sub_33E1790(*(_QWORD *)(*(_QWORD *)(v141 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(v141 + 40) + 48LL), 1);
    v106 = v105;
    if ( v105 )
    {
      if ( (unsigned __int8)sub_11DB340((_DWORD **)(*(_QWORD *)(v105 + 96) + 24LL), 1.0) )
      {
        v108 = sub_340F900(
                 *a1,
                 v153,
                 (unsigned int)&v161,
                 v159,
                 v160,
                 v107,
                 *(_OWORD *)*(_QWORD *)(v141 + 40),
                 v134,
                 v134);
        v73 = v108;
        goto LABEL_138;
      }
      if ( (unsigned __int8)sub_11DB340((_DWORD **)(*(_QWORD *)(v106 + 96) + 24LL), -1.0) )
      {
        v116 = *a1;
        *(_QWORD *)&v117 = sub_33FAF80(*a1, 244, (unsigned int)&v161, v159, v160, v115, v134);
        v108 = sub_340F900(
                 v116,
                 v153,
                 (unsigned int)&v161,
                 v159,
                 v160,
                 v118,
                 *(_OWORD *)*(_QWORD *)(v141 + 40),
                 v134,
                 v117);
        v73 = v108;
LABEL_138:
        if ( v108 )
          goto LABEL_72;
      }
    }
  }
  v168.m128i_i64[1] = (__int64)a1;
  v168.m128i_i64[0] = (__int64)&v150;
  v169 = &v153;
  v170 = &v161;
  v171 = &v159;
  v73 = sub_327F470((__int64)&v168, v147, v53.m128i_i64[0], v53.m128i_i64[1]);
  if ( !v73 )
  {
    v73 = sub_327F470((__int64)&v168, v141, v134, *((__int64 *)&v134 + 1));
    if ( !v73 )
      goto LABEL_48;
  }
LABEL_72:
  if ( v161 )
    sub_B91220((__int64)&v161, v161);
  sub_32B3E80((__int64)a1, v73, 1, 0, v71, v72);
  v19 = v73;
LABEL_5:
  v20 = v157;
  v163[128] = v165;
  if ( v20 )
    sub_B91220((__int64)&v157, v20);
  return v19;
}
