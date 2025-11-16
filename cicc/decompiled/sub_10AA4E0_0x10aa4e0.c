// Function: sub_10AA4E0
// Address: 0x10aa4e0
//
unsigned __int8 *__fastcall sub_10AA4E0(const __m128i *a1, __int64 a2)
{
  __int64 v2; // r14
  unsigned __int8 *v3; // r15
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 *v10; // r9
  unsigned __int8 v11; // al
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rax
  char v15; // al
  __int64 v16; // rax
  __m128i v17; // xmm1
  unsigned __int8 *v18; // r15
  __int64 v19; // rax
  unsigned __int64 v20; // xmm2_8
  __m128i v21; // xmm3
  int v22; // eax
  _BYTE *v23; // r13
  __int64 v24; // rax
  bool v25; // al
  unsigned int v26; // eax
  __int64 v27; // rsi
  unsigned int v28; // ecx
  __int64 v29; // r8
  unsigned int v30; // r14d
  int v31; // eax
  bool v32; // al
  __int64 v33; // rax
  unsigned __int64 v34; // rax
  int v35; // eax
  __int64 v36; // rax
  unsigned int v37; // r14d
  bool v38; // al
  _BYTE *v39; // rax
  __m128i v40; // xmm5
  unsigned __int64 v41; // xmm6_8
  __int64 v42; // rax
  __m128i v43; // xmm7
  unsigned __int8 *v44; // rax
  _BYTE *v45; // rdi
  __int64 v46; // rbx
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rdi
  unsigned __int8 *v50; // rax
  __int64 v51; // rax
  unsigned __int64 v52; // rax
  int v53; // eax
  __int64 v54; // rdi
  __int64 v55; // rax
  __int64 v56; // r13
  __int64 v57; // r15
  __int64 v58; // r14
  __int64 v59; // rax
  __int64 v60; // rdi
  int v61; // eax
  int v62; // ecx
  __int64 v63; // r8
  __int64 v64; // rax
  __int64 v65; // rdi
  bool v66; // al
  _BYTE *v67; // r15
  __int64 v68; // r14
  __int64 v69; // r13
  __int64 v70; // r12
  unsigned __int8 *v71; // rax
  unsigned __int8 *v72; // rax
  _BYTE *v73; // rax
  bool v74; // dl
  __int64 v75; // rbx
  __int64 v76; // r13
  __int64 v77; // rdx
  unsigned int v78; // esi
  _QWORD **v79; // rdx
  int v80; // ecx
  int v81; // eax
  __int64 *v82; // rax
  __int64 v83; // rsi
  __int64 v84; // r13
  __int64 v85; // r14
  __int64 v86; // rdx
  unsigned int v87; // esi
  unsigned int v88; // ecx
  __int64 v89; // rax
  unsigned int v90; // ecx
  int v91; // eax
  __m128i v92; // xmm5
  unsigned __int64 v93; // xmm6_8
  __m128i v94; // xmm7
  __int64 v95; // rax
  __int64 v96; // rdx
  __int64 v97; // rcx
  unsigned int v98; // r8d
  __int64 *v99; // r14
  __int64 v100; // rax
  unsigned __int64 v101; // rdx
  __int32 v102; // eax
  char v103; // r14
  unsigned int v104; // r14d
  int v105; // eax
  int v106; // r9d
  __m128i *v107; // rdi
  const __m128i *v108; // rsi
  __int64 i; // rcx
  __int64 v110; // r14
  unsigned __int64 v111; // rdx
  int v112; // eax
  __int64 v113; // rdi
  __int64 v114; // rax
  __int64 v115; // rax
  _BYTE *v116; // rsi
  __int64 v117; // rdx
  __int64 v118; // rcx
  unsigned int v119; // r8d
  __int64 *v120; // r14
  unsigned int v121; // eax
  _BYTE *v122; // rax
  __int64 *v123; // r14
  _BYTE *v124; // rax
  int v125; // eax
  __int64 *v126; // r12
  int v127; // eax
  __int64 v128; // rax
  const __m128i *v129; // rsi
  __int64 v130; // rcx
  __m128i *v131; // rdi
  __int32 v132; // eax
  bool v133; // r14
  __int64 *v134; // r14
  _BYTE *v135; // r13
  int v136; // eax
  __int64 v137; // rax
  bool v138; // al
  int v139; // eax
  __int64 v140; // r14
  __int64 v141; // rdi
  int v142; // eax
  unsigned int **v143; // rdi
  int v144; // eax
  unsigned int **v145; // rdi
  __int64 v146; // r12
  __int64 v147; // rax
  _BYTE *v148; // rax
  unsigned int **v149; // rdi
  __int64 v150; // r12
  __int64 v151; // rax
  int v152; // [rsp+8h] [rbp-148h]
  __int64 v153; // [rsp+8h] [rbp-148h]
  __int64 v154; // [rsp+10h] [rbp-140h]
  char v155; // [rsp+10h] [rbp-140h]
  int v156; // [rsp+10h] [rbp-140h]
  __int64 v157; // [rsp+10h] [rbp-140h]
  unsigned int v158; // [rsp+10h] [rbp-140h]
  _QWORD *v159; // [rsp+10h] [rbp-140h]
  __int64 *v160; // [rsp+10h] [rbp-140h]
  __int64 v161; // [rsp+10h] [rbp-140h]
  bool v162; // [rsp+18h] [rbp-138h]
  int v163; // [rsp+18h] [rbp-138h]
  int v164; // [rsp+18h] [rbp-138h]
  _BYTE *v165; // [rsp+20h] [rbp-130h]
  int v166; // [rsp+20h] [rbp-130h]
  __int64 v167; // [rsp+20h] [rbp-130h]
  unsigned int v168; // [rsp+20h] [rbp-130h]
  int v169; // [rsp+20h] [rbp-130h]
  __int64 v170; // [rsp+20h] [rbp-130h]
  __int64 v171; // [rsp+28h] [rbp-128h]
  __int64 v172; // [rsp+30h] [rbp-120h]
  __int64 v173; // [rsp+38h] [rbp-118h]
  int v174; // [rsp+38h] [rbp-118h]
  _BYTE *v175; // [rsp+40h] [rbp-110h] BYREF
  __int64 *v176; // [rsp+48h] [rbp-108h] BYREF
  __int64 v177; // [rsp+50h] [rbp-100h] BYREF
  __int64 *v178; // [rsp+58h] [rbp-F8h] BYREF
  unsigned __int64 v179; // [rsp+60h] [rbp-F0h] BYREF
  int v180; // [rsp+68h] [rbp-E8h]
  unsigned __int64 v181; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v182; // [rsp+78h] [rbp-D8h] BYREF
  int v183; // [rsp+80h] [rbp-D0h]
  __int64 v184; // [rsp+88h] [rbp-C8h] BYREF
  int v185; // [rsp+90h] [rbp-C0h]
  unsigned __int64 v186; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v187; // [rsp+A8h] [rbp-A8h] BYREF
  __int64 v188; // [rsp+B0h] [rbp-A0h] BYREF
  __int64 v189; // [rsp+B8h] [rbp-98h] BYREF
  int v190; // [rsp+C0h] [rbp-90h]
  __m128i v191; // [rsp+D0h] [rbp-80h] BYREF
  __m128i v192; // [rsp+E0h] [rbp-70h] BYREF
  unsigned __int64 v193; // [rsp+F0h] [rbp-60h]
  __int64 v194; // [rsp+F8h] [rbp-58h]
  __m128i v195; // [rsp+100h] [rbp-50h]
  __int64 v196; // [rsp+110h] [rbp-40h]

  v2 = *(_QWORD *)(a2 - 32);
  if ( *(_BYTE *)v2 > 0x15u )
    return 0;
  if ( *(_BYTE *)v2 == 5 )
    return 0;
  v173 = *(_QWORD *)(a2 - 64);
  v171 = *(_QWORD *)(a2 + 8);
  if ( (unsigned __int8)sub_AD6CA0(*(_QWORD *)(a2 - 32)) )
    return 0;
  v3 = sub_F28360((__int64)a1, (_BYTE *)a2, v7, v8, v9, v10);
  if ( v3 )
    return v3;
  v11 = *(_BYTE *)v173;
  if ( *(_BYTE *)v173 == 44 )
  {
    v45 = *(_BYTE **)(v173 - 64);
    if ( *v45 <= 0x15u )
    {
      v46 = *(_QWORD *)(v173 - 32);
      if ( v46 )
      {
        v175 = *(_BYTE **)(v173 - 32);
        LOWORD(v193) = 257;
        v47 = sub_AD57C0((__int64)v45, (unsigned __int8 *)v2, 0, 0);
        return (unsigned __int8 *)sub_B504D0(15, v47, v46, (__int64)&v191, 0, 0);
      }
    }
  }
  v12 = *(_QWORD *)(v173 + 16);
  if ( v12 && !*(_QWORD *)(v12 + 8) && v11 == 44 && *(_QWORD *)(v173 - 64) )
  {
    v175 = *(_BYTE **)(v173 - 64);
    v172 = *(_QWORD *)(v173 - 32);
    if ( v172 )
    {
      v191.m128i_i64[0] = 0;
      if ( (unsigned __int8)sub_995B10(&v191, v2) )
      {
        v56 = a1[2].m128i_i64[0];
        LOWORD(v190) = 257;
        LOWORD(v185) = 257;
        v57 = sub_AD62B0(*(_QWORD *)(v172 + 8));
        v58 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v56 + 80) + 16LL))(
                *(_QWORD *)(v56 + 80),
                30,
                v172,
                v57);
        if ( !v58 )
        {
          LOWORD(v193) = 257;
          v58 = sub_B504D0(30, v172, v57, (__int64)&v191, 0, 0);
          (*(void (__fastcall **)(_QWORD, __int64, unsigned __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v56 + 88) + 16LL))(
            *(_QWORD *)(v56 + 88),
            v58,
            &v181,
            *(_QWORD *)(v56 + 56),
            *(_QWORD *)(v56 + 64));
          v75 = *(_QWORD *)v56;
          v76 = *(_QWORD *)v56 + 16LL * *(unsigned int *)(v56 + 8);
          while ( v76 != v75 )
          {
            v77 = *(_QWORD *)(v75 + 8);
            v78 = *(_DWORD *)v75;
            v75 += 16;
            sub_B99FD0(v58, v78, v77);
          }
        }
        return (unsigned __int8 *)sub_B504D0(13, v58, (__int64)v175, (__int64)&v186, 0, 0);
      }
      v11 = *(_BYTE *)v173;
    }
    else
    {
      v11 = *(_BYTE *)v173;
    }
  }
  if ( v11 == 68 )
  {
    v59 = *(_QWORD *)(v173 - 32);
    if ( !v59 )
      goto LABEL_11;
    v60 = *(_QWORD *)(v59 + 8);
    v175 = *(_BYTE **)(v173 - 32);
    if ( (unsigned int)sub_BCB060(v60) == 1 )
    {
      LOWORD(v193) = 257;
      v72 = (unsigned __int8 *)sub_AD64C0(*(_QWORD *)(v2 + 8), 1, 0);
      v51 = sub_AD57C0(v2, v72, 0, 0);
      return sub_109FEA0((__int64)v175, v51, v2, (const char **)&v191, 0, 0, 0);
    }
    v11 = *(_BYTE *)v173;
  }
  if ( v11 == 69 )
  {
    v48 = *(_QWORD *)(v173 - 32);
    if ( v48 )
    {
      v49 = *(_QWORD *)(v48 + 8);
      v175 = *(_BYTE **)(v173 - 32);
      if ( (unsigned int)sub_BCB060(v49) == 1 )
      {
        LOWORD(v193) = 257;
        v50 = (unsigned __int8 *)sub_AD64C0(*(_QWORD *)(v2 + 8), 1, 0);
        v51 = sub_AD57F0(v2, v50, 0, 0);
        return sub_109FEA0((__int64)v175, v51, v2, (const char **)&v191, 0, 0, 0);
      }
    }
  }
LABEL_11:
  v191.m128i_i64[0] = 0;
  v191.m128i_i64[1] = (__int64)&v175;
  if ( (unsigned __int8)sub_996420(&v191, 30, (unsigned __int8 *)v173) )
  {
    v16 = sub_AD64C0(*(_QWORD *)(v2 + 8), 1, 0);
    v17 = _mm_loadu_si128(a1 + 7);
    v18 = (unsigned __int8 *)v16;
    v19 = a1[10].m128i_i64[0];
    v20 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
    v191 = _mm_loadu_si128(a1 + 6);
    v21 = _mm_loadu_si128(a1 + 9);
    v192 = v17;
    v193 = v20;
    v195 = v21;
    v196 = v19;
    v194 = a2;
    v22 = sub_9AFB10(v2, (__int64)v18, &v191);
    v23 = v175;
    LOWORD(v193) = 257;
    v174 = v22;
    v24 = sub_AD57F0(v2, v18, 0, 0);
    v3 = (unsigned __int8 *)sub_B504D0(15, v24, (__int64)v23, (__int64)&v191, 0, 0);
    v25 = sub_B44900(a2);
    sub_B44850(v3, v25 && v174 == 3);
    return v3;
  }
  v13 = *(_QWORD *)(v173 + 16);
  if ( !v13 )
    goto LABEL_13;
  if ( *(_QWORD *)(v13 + 8) )
    goto LABEL_13;
  if ( *(_BYTE *)v173 != 56 )
    goto LABEL_13;
  v165 = *(_BYTE **)(v173 - 64);
  if ( !v165 )
    goto LABEL_13;
  v61 = sub_BCB060(v171);
  v62 = v61;
  v175 = v165;
  v63 = *(_QWORD *)(v173 - 32);
  if ( !v63 )
    BUG();
  if ( *(_BYTE *)v63 != 17 )
  {
    v169 = v61;
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v63 + 8) + 8LL) - 17 > 1 )
      goto LABEL_13;
    if ( *(_BYTE *)v63 > 0x15u )
      goto LABEL_13;
    v124 = sub_AD7630(v63, 1, 0);
    v63 = (__int64)v124;
    if ( !v124 )
      goto LABEL_13;
    v62 = v169;
    if ( *v124 != 17 )
      goto LABEL_13;
  }
  if ( *(_DWORD *)(v63 + 32) > 0x40u )
  {
    v152 = *(_DWORD *)(v63 + 32);
    v164 = v62;
    v170 = v63;
    v125 = sub_C444A0(v63 + 24);
    v62 = v164;
    if ( (unsigned int)(v152 - v125) > 0x40 )
      goto LABEL_13;
    v64 = **(_QWORD **)(v170 + 24);
  }
  else
  {
    v64 = *(_QWORD *)(v63 + 24);
  }
  if ( v62 - 1 != v64 )
    goto LABEL_13;
  if ( *(_BYTE *)v2 == 17 )
  {
    if ( *(_DWORD *)(v2 + 32) > 0x40u )
    {
      v166 = *(_DWORD *)(v2 + 32);
      v65 = v2 + 24;
LABEL_84:
      v66 = v166 - 1 == (unsigned int)sub_C444A0(v65);
      goto LABEL_85;
    }
    v66 = *(_QWORD *)(v2 + 24) == 1;
  }
  else
  {
    v167 = *(_QWORD *)(v2 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v167 + 8) - 17 > 1 || *(_BYTE *)v2 > 0x15u )
      goto LABEL_13;
    v73 = sub_AD7630(v2, 0, 0);
    v74 = 0;
    if ( !v73 || *v73 != 17 )
    {
      if ( *(_BYTE *)(v167 + 8) == 17 )
      {
        v156 = *(_DWORD *)(v167 + 32);
        if ( v156 )
        {
          v88 = 0;
          while ( 1 )
          {
            v162 = v74;
            v168 = v88;
            v89 = sub_AD69F0((unsigned __int8 *)v2, v88);
            if ( !v89 )
              break;
            v90 = v168;
            v74 = v162;
            if ( *(_BYTE *)v89 != 13 )
            {
              if ( *(_BYTE *)v89 != 17 )
                break;
              if ( *(_DWORD *)(v89 + 32) <= 0x40u )
              {
                v74 = *(_QWORD *)(v89 + 24) == 1;
              }
              else
              {
                v163 = *(_DWORD *)(v89 + 32);
                v91 = sub_C444A0(v89 + 24);
                v90 = v168;
                v74 = v163 - 1 == v91;
              }
              if ( !v74 )
                break;
            }
            v88 = v90 + 1;
            if ( v156 == v88 )
            {
              if ( v74 )
                goto LABEL_86;
              goto LABEL_13;
            }
          }
        }
      }
      goto LABEL_13;
    }
    if ( *((_DWORD *)v73 + 8) > 0x40u )
    {
      v166 = *((_DWORD *)v73 + 8);
      v65 = (__int64)(v73 + 24);
      goto LABEL_84;
    }
    v66 = *((_QWORD *)v73 + 3) == 1;
  }
LABEL_85:
  if ( v66 )
  {
LABEL_86:
    v67 = v175;
    v68 = a1[2].m128i_i64[0];
    v186 = (unsigned __int64)"isnotneg";
    LOWORD(v190) = 259;
    v69 = sub_AD62B0(*((_QWORD *)v175 + 1));
    v70 = (*(__int64 (__fastcall **)(_QWORD, __int64, _BYTE *, __int64))(**(_QWORD **)(v68 + 80) + 56LL))(
            *(_QWORD *)(v68 + 80),
            38,
            v67,
            v69);
    if ( !v70 )
    {
      LOWORD(v193) = 257;
      v70 = (__int64)sub_BD2C40(72, unk_3F10FD0);
      if ( v70 )
      {
        v79 = (_QWORD **)*((_QWORD *)v67 + 1);
        v80 = *((unsigned __int8 *)v79 + 8);
        if ( (unsigned int)(v80 - 17) > 1 )
        {
          v83 = sub_BCB2A0(*v79);
        }
        else
        {
          v81 = *((_DWORD *)v79 + 8);
          BYTE4(v181) = (_BYTE)v80 == 18;
          LODWORD(v181) = v81;
          v82 = (__int64 *)sub_BCB2A0(*v79);
          v83 = sub_BCE1B0(v82, v181);
        }
        sub_B523C0(v70, v83, 53, 38, (__int64)v67, v69, (__int64)&v191, 0, 0, 0);
      }
      (*(void (__fastcall **)(_QWORD, __int64, unsigned __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v68 + 88) + 16LL))(
        *(_QWORD *)(v68 + 88),
        v70,
        &v186,
        *(_QWORD *)(v68 + 56),
        *(_QWORD *)(v68 + 64));
      v84 = *(_QWORD *)v68;
      v85 = *(_QWORD *)v68 + 16LL * *(unsigned int *)(v68 + 8);
      while ( v85 != v84 )
      {
        v86 = *(_QWORD *)(v84 + 8);
        v87 = *(_DWORD *)v84;
        v84 += 16;
        sub_B99FD0(v70, v87, v86);
      }
    }
    goto LABEL_87;
  }
LABEL_13:
  v191.m128i_i8[8] = 0;
  v191.m128i_i64[0] = (__int64)&v176;
  if ( !(unsigned __int8)sub_991580((__int64)&v191, v2) )
    return v3;
  v191.m128i_i64[0] = (__int64)&v175;
  v191.m128i_i64[1] = (__int64)&v177;
  if ( *(_BYTE *)v173 == 58 && (*(_BYTE *)(v173 + 1) & 2) != 0 )
  {
    if ( *(_QWORD *)(v173 - 64) )
    {
      v175 = *(_BYTE **)(v173 - 64);
      if ( (unsigned __int8)sub_F11D70((_QWORD **)&v191.m128i_i64[1], *(_BYTE **)(v173 - 32)) )
      {
        LOWORD(v193) = 257;
        v14 = sub_AD57C0(v177, (unsigned __int8 *)v2, 0, 0);
        v3 = (unsigned __int8 *)sub_B504D0(13, (__int64)v175, v14, (__int64)&v191, 0, 0);
        if ( sub_B44900(a2) )
        {
          v92 = _mm_loadu_si128(a1 + 7);
          v93 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
          v191 = _mm_loadu_si128(a1 + 6);
          v94 = _mm_loadu_si128(a1 + 9);
          v181 = v177 & 0xFFFFFFFFFFFFFFFBLL;
          v95 = a1[10].m128i_i64[0];
          v193 = v93;
          v186 = v2 & 0xFFFFFFFFFFFFFFFBLL;
          LODWORD(v188) = 1;
          v187 = 0;
          v190 = 1;
          v189 = 0;
          v183 = 1;
          v182 = 0;
          v185 = 1;
          v184 = 0;
          v196 = v95;
          v194 = a2;
          v192 = v92;
          v195 = v94;
          if ( (unsigned int)sub_9B0100((__int64 *)&v181, (__int64 *)&v186, &v191) == 3 )
            sub_B44850(v3, 1);
          else
            sub_B44850(v3, 0);
          sub_969240(&v184);
          sub_969240(&v182);
          sub_969240(&v189);
          sub_969240(&v187);
        }
        else
        {
          sub_B44850(v3, 0);
        }
        v15 = sub_B448F0(a2);
        sub_B447F0(v3, v15);
        return v3;
      }
    }
  }
  v192.m128i_i8[0] = 0;
  v191.m128i_i64[1] = (__int64)&v178;
  if ( *(_BYTE *)v173 == 58 && (unsigned __int8)sub_991580((__int64)&v191.m128i_i64[1], *(_QWORD *)(v173 - 32)) )
  {
    sub_9865C0((__int64)&v181, (__int64)v176);
    if ( (unsigned int)v182 > 0x40 )
    {
      sub_C43D10((__int64)&v181);
    }
    else
    {
      v52 = 0;
      if ( (_DWORD)v182 )
        v52 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v182;
      v181 = ~v181 & v52;
    }
    sub_C46250((__int64)&v181);
    v53 = v182;
    LODWORD(v182) = 0;
    LODWORD(v187) = v53;
    v186 = v181;
    v155 = sub_AAD8B0((__int64)v178, &v186);
    sub_969240((__int64 *)&v186);
    sub_969240((__int64 *)&v181);
    if ( v155 )
    {
      v54 = *(_QWORD *)(a2 + 8);
      LOWORD(v193) = 257;
      v55 = sub_AD8D80(v54, (__int64)v178);
      return (unsigned __int8 *)sub_B504D0(30, v173, v55, (__int64)&v191, 0, 0);
    }
  }
  v26 = *((_DWORD *)v176 + 2);
  v27 = *v176;
  v28 = v26 - 1;
  if ( v26 > 0x40 )
  {
    if ( (*(_QWORD *)(v27 + 8LL * (v28 >> 6)) & (1LL << v28)) == 0 || (unsigned int)sub_C44590((__int64)v176) != v28 )
      goto LABEL_32;
LABEL_97:
    if ( sub_B44900(a2) || sub_B448F0(a2) )
    {
      LOWORD(v193) = 257;
      return (unsigned __int8 *)sub_B504D0(29, v173, v2, (__int64)&v191, 0, 0);
    }
    else
    {
      LOWORD(v193) = 257;
      return (unsigned __int8 *)sub_B504D0(30, v173, v2, (__int64)&v191, 0, 0);
    }
  }
  if ( v27 == 1LL << v28 )
    goto LABEL_97;
LABEL_32:
  v192.m128i_i8[0] = 0;
  v191.m128i_i64[0] = (__int64)&v175;
  v191.m128i_i64[1] = (__int64)&v178;
  if ( *(_BYTE *)v173 == 68 )
  {
    v116 = *(_BYTE **)(v173 - 32);
    if ( *v116 == 59 )
    {
      if ( (unsigned __int8)sub_109D400((__int64)&v191, (__int64)v116) )
      {
        v120 = v178;
        if ( (unsigned __int8)sub_986B30(v178, (__int64)v116, v117, v118, v119) )
        {
          v159 = v176;
          v121 = sub_BCB060(v171);
          sub_C44830((__int64)&v186, v120, v121);
          if ( sub_AAD8B0((__int64)&v186, v159) )
          {
            sub_969240((__int64 *)&v186);
            LOWORD(v193) = 257;
            return (unsigned __int8 *)sub_B51D30(40, (__int64)v175, v171, (__int64)&v191, 0, 0);
          }
          sub_969240((__int64 *)&v186);
        }
      }
    }
  }
  v192.m128i_i8[0] = 0;
  v191.m128i_i64[0] = (__int64)&v175;
  v191.m128i_i64[1] = (__int64)&v178;
  if ( *(_BYTE *)v173 != 59 || !(unsigned __int8)sub_109D400((__int64)&v191, v173) )
    goto LABEL_34;
  v99 = v178;
  if ( (unsigned __int8)sub_986B30(v178, v173, v96, v97, v98) )
  {
    LOWORD(v193) = 257;
    v126 = v176;
    sub_9865C0((__int64)&v181, (__int64)v99);
    if ( (unsigned int)v182 > 0x40 )
      sub_C43C10(&v181, v126);
    else
      v181 ^= *v126;
    v127 = v182;
    LODWORD(v182) = 0;
    LODWORD(v187) = v127;
    v186 = v181;
    v128 = sub_AD8D80(v171, (__int64)&v186);
    v3 = (unsigned __int8 *)sub_B504D0(13, (__int64)v175, v128, (__int64)&v191, 0, 0);
    sub_969240((__int64 *)&v186);
    sub_969240((__int64 *)&v181);
    return v3;
  }
  if ( sub_1002450((__int64)v99) )
  {
    v129 = a1 + 6;
    v130 = 18;
    v131 = &v191;
    while ( v130 )
    {
      v131->m128i_i32[0] = v129->m128i_i32[0];
      v129 = (const __m128i *)((char *)v129 + 4);
      v131 = (__m128i *)((char *)v131 + 4);
      --v130;
    }
    v194 = a2;
    sub_9AC330((__int64)&v186, (__int64)v175, 0, &v191);
    sub_9865C0((__int64)&v181, (__int64)v178);
    if ( (unsigned int)v182 > 0x40 )
      sub_C43BD0(&v181, (__int64 *)&v186);
    else
      v181 |= v186;
    v132 = v182;
    LODWORD(v182) = 0;
    v191.m128i_i32[2] = v132;
    v191.m128i_i64[0] = v181;
    v133 = sub_986760((__int64)&v191);
    sub_969240(v191.m128i_i64);
    sub_969240((__int64 *)&v181);
    if ( v133 )
    {
      v134 = v176;
      LOWORD(v193) = 257;
      v135 = v175;
      sub_9865C0((__int64)&v179, (__int64)v178);
      sub_C45EE0((__int64)&v179, v134);
      v136 = v180;
      v180 = 0;
      LODWORD(v182) = v136;
      v181 = v179;
      v137 = sub_AD8D80(v171, (__int64)&v181);
      v3 = (unsigned __int8 *)sub_B504D0(15, v137, (__int64)v135, (__int64)&v191, 0, 0);
      sub_969240((__int64 *)&v181);
      sub_969240((__int64 *)&v179);
      sub_969240(&v188);
      sub_969240((__int64 *)&v186);
      return v3;
    }
    sub_969240(&v188);
    sub_969240((__int64 *)&v186);
  }
  v100 = *(_QWORD *)(v173 + 16);
  if ( !v100 || *(_QWORD *)(v100 + 8) )
    goto LABEL_34;
  sub_9865C0((__int64)&v186, (__int64)v176);
  if ( (unsigned int)v187 > 0x40 )
  {
    sub_C43D10((__int64)&v186);
  }
  else
  {
    v101 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v187;
    if ( !(_DWORD)v187 )
      v101 = 0;
    v186 = v101 & ~v186;
  }
  sub_C46250((__int64)&v186);
  v102 = v187;
  LODWORD(v187) = 0;
  v191.m128i_i32[2] = v102;
  v191.m128i_i64[0] = v186;
  v103 = sub_AAD8B0((__int64)v178, &v191);
  sub_969240(v191.m128i_i64);
  sub_969240((__int64 *)&v186);
  if ( !v103 )
    goto LABEL_34;
  v104 = sub_BCB060(v171);
  v157 = (__int64)v176;
  if ( sub_986BA0((__int64)v176) )
  {
    v105 = sub_9871A0(v157);
    v29 = v157;
    v106 = v105 + v104 - *(_DWORD *)(v157 + 8);
  }
  else
  {
    v153 = v157;
    v160 = v178;
    v138 = sub_986BA0((__int64)v178);
    v29 = v153;
    if ( !v138 )
      goto LABEL_35;
    v139 = sub_9871A0((__int64)v160);
    v29 = v153;
    v106 = v139 + v104 - *((_DWORD *)v160 + 2);
  }
  if ( v106 )
  {
    v158 = v106;
    sub_109DDE0((__int64)&v186, v104, v106);
    v107 = &v191;
    v108 = a1 + 6;
    for ( i = 18; i; --i )
    {
      v107->m128i_i32[0] = v108->m128i_i32[0];
      v108 = (const __m128i *)((char *)v108 + 4);
      v107 = (__m128i *)((char *)v107 + 4);
    }
    v194 = a2;
    if ( (unsigned __int8)sub_9AC230((__int64)v175, (__int64)&v186, &v191, 0) )
    {
      sub_969240((__int64 *)&v186);
      v148 = (_BYTE *)sub_AD64C0(v171, v158, 0);
      v149 = (unsigned int **)a1[2].m128i_i64[0];
      v150 = (__int64)v148;
      LOWORD(v193) = 259;
      v191.m128i_i64[0] = (__int64)"sext";
      v151 = sub_920A70(v149, v175, v148, (__int64)&v191, 0, 0);
      LOWORD(v193) = 257;
      return (unsigned __int8 *)sub_B504D0(27, v151, v150, (__int64)&v191, 0, 0);
    }
    sub_969240((__int64 *)&v186);
LABEL_34:
    v29 = (__int64)v176;
  }
LABEL_35:
  v30 = *(_DWORD *)(v29 + 8);
  if ( v30 <= 0x40 )
  {
    v32 = *(_QWORD *)v29 == 1;
  }
  else
  {
    v154 = v29;
    v31 = sub_C444A0(v29);
    v29 = v154;
    v32 = v30 - 1 == v31;
  }
  if ( v32 )
  {
    v33 = *(_QWORD *)(v173 + 16);
    if ( v33 )
    {
      if ( !*(_QWORD *)(v33 + 8) )
      {
        if ( *(_BYTE *)v173 == 69 )
        {
          v140 = *(_QWORD *)(v173 - 32);
          if ( v140 )
          {
            v141 = *(_QWORD *)(v140 + 8);
            v161 = v29;
            v175 = *(_BYTE **)(v173 - 32);
            v142 = sub_BCB060(v141);
            v29 = v161;
            if ( v142 == 1 )
            {
              v143 = (unsigned int **)a1[2].m128i_i64[0];
              LOWORD(v190) = 257;
              v70 = sub_A82B60(v143, v140, (__int64)&v186);
LABEL_87:
              LOWORD(v193) = 257;
              v71 = (unsigned __int8 *)sub_BD2C40(72, unk_3F10A14);
              v3 = v71;
              if ( v71 )
                sub_B515B0((__int64)v71, v70, v171, (__int64)&v191, 0, 0);
              return v3;
            }
          }
        }
        v192.m128i_i8[0] = 0;
        LOBYTE(v193) = 0;
        v191.m128i_i64[0] = (__int64)&v175;
        v191.m128i_i64[1] = (__int64)&v178;
        v192.m128i_i64[1] = (__int64)&v186;
        if ( *(_BYTE *)v173 == 56 )
        {
          v122 = *(_BYTE **)(v173 - 64);
          if ( *v122 == 54 )
          {
            if ( *((_QWORD *)v122 - 8) )
            {
              v175 = (_BYTE *)*((_QWORD *)v122 - 8);
              if ( (unsigned __int8)sub_991580((__int64)&v191.m128i_i64[1], *((_QWORD *)v122 - 4)) )
              {
                if ( (unsigned __int8)sub_991580((__int64)&v192.m128i_i64[1], *(_QWORD *)(v173 - 32)) )
                {
                  v123 = v178;
                  if ( v178 == (__int64 *)v186 )
                  {
                    v144 = sub_BCB060(v171);
                    if ( sub_D94970((__int64)v123, (_QWORD *)(unsigned int)(v144 - 1)) )
                    {
                      v145 = (unsigned int **)a1[2].m128i_i64[0];
                      LOWORD(v193) = 257;
                      v146 = sub_A82B60(v145, (__int64)v175, (__int64)&v191);
                      LOWORD(v193) = 257;
                      v147 = sub_AD64C0(v171, 1, 0);
                      return (unsigned __int8 *)sub_B504D0(28, v146, v147, (__int64)&v191, 0, 0);
                    }
                  }
                }
              }
              v29 = (__int64)v176;
            }
          }
        }
      }
    }
  }
  sub_9865C0((__int64)&v181, v29);
  if ( (unsigned int)v182 > 0x40 )
  {
    sub_C43D10((__int64)&v181);
  }
  else
  {
    v34 = 0;
    if ( (_DWORD)v182 )
      v34 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v182;
    v181 = ~v181 & v34;
  }
  sub_C46250((__int64)&v181);
  v35 = v182;
  LODWORD(v182) = 0;
  LODWORD(v187) = v35;
  v186 = v181;
  v191.m128i_i64[0] = (__int64)&v175;
  v191.m128i_i64[1] = (__int64)&v186;
  v36 = *(_QWORD *)(v173 + 16);
  if ( v36 && !*(_QWORD *)(v36 + 8) && sub_10AA3B0((__int64)&v191, (char *)v173) )
  {
    sub_969240((__int64 *)&v186);
    sub_969240((__int64 *)&v181);
    LOWORD(v193) = 257;
    v110 = a1[2].m128i_i64[0];
    HIDWORD(v179) = 0;
    sub_9865C0((__int64)&v181, (__int64)v176);
    if ( (unsigned int)v182 > 0x40 )
    {
      sub_C43D10((__int64)&v181);
    }
    else
    {
      v111 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v182;
      if ( !(_DWORD)v182 )
        v111 = 0;
      v181 = v111 & ~v181;
    }
    sub_C46250((__int64)&v181);
    v112 = v182;
    v113 = *(_QWORD *)(a2 + 8);
    LODWORD(v182) = 0;
    LODWORD(v187) = v112;
    v186 = v181;
    v114 = sub_AD8D80(v113, (__int64)&v186);
    v115 = sub_B33C40(v110, 0x173u, (__int64)v175, v114, v179, (__int64)&v191);
    v3 = sub_F162A0((__int64)a1, a2, v115);
    sub_969240((__int64 *)&v186);
    sub_969240((__int64 *)&v181);
  }
  else
  {
    sub_969240((__int64 *)&v186);
    sub_969240((__int64 *)&v181);
    v37 = *((_DWORD *)v176 + 2);
    if ( v37 <= 0x40 )
      v38 = *v176 == 1;
    else
      v38 = v37 - 1 == (unsigned int)sub_C444A0((__int64)v176);
    if ( v38 )
    {
      v191 = (__m128i)(unsigned __int64)&v175;
      if ( *(_BYTE *)v173 == 68 )
      {
        v39 = *(_BYTE **)(v173 - 32);
        if ( *v39 == 42 )
        {
          if ( *((_QWORD *)v39 - 8) )
          {
            v175 = (_BYTE *)*((_QWORD *)v39 - 8);
            if ( (unsigned __int8)sub_995B10((_QWORD **)&v191.m128i_i64[1], *((_QWORD *)v39 - 4)) )
            {
              v40 = _mm_loadu_si128(a1 + 7);
              v41 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
              v42 = a1[10].m128i_i64[0];
              v43 = _mm_loadu_si128(a1 + 9);
              v191 = _mm_loadu_si128(a1 + 6);
              v193 = v41;
              v196 = v42;
              v194 = a2;
              v192 = v40;
              v195 = v43;
              if ( (unsigned __int8)sub_9B6260((__int64)v175, &v191, 0) )
              {
                LOWORD(v190) = 257;
                v44 = (unsigned __int8 *)sub_BD2C40(72, unk_3F10A14);
                v3 = v44;
                if ( v44 )
                  sub_B515B0((__int64)v44, (__int64)v175, v171, (__int64)&v186, 0, 0);
              }
            }
          }
        }
      }
    }
  }
  return v3;
}
