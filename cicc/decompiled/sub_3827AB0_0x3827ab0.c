// Function: sub_3827AB0
// Address: 0x3827ab0
//
__int64 __fastcall sub_3827AB0(__int64 *a1, unsigned __int64 *a2, __int64 a3, unsigned int *a4, __int64 a5, __m128i a6)
{
  unsigned __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // r9
  unsigned int v14; // eax
  unsigned __int64 v15; // rcx
  int v16; // edx
  unsigned int v17; // r12d
  __int64 v18; // r9
  __int64 v19; // rdi
  __int64 v20; // rax
  unsigned __int16 v21; // cx
  __int64 v22; // r8
  unsigned int v23; // eax
  __int64 v24; // rdx
  __int128 v25; // rax
  unsigned int v26; // edx
  __int64 v27; // rax
  unsigned __int16 v28; // cx
  unsigned int *v29; // r12
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // r12
  unsigned int v34; // edx
  _QWORD *v35; // r12
  __int64 v36; // rdx
  unsigned int v37; // eax
  __int64 v38; // rdx
  __int64 v39; // r9
  unsigned int v40; // edx
  int v41; // eax
  __int64 v42; // rcx
  int v43; // edx
  char v44; // al
  __int64 v45; // rdi
  unsigned int v46; // edx
  bool v47; // al
  unsigned __int16 *v48; // rax
  unsigned __int16 v49; // ax
  unsigned int v50; // r8d
  __int64 v51; // rdi
  unsigned int v52; // eax
  __m128i v53; // xmm0
  __int32 v54; // eax
  __m128i v55; // xmm0
  unsigned __int16 *v56; // rax
  unsigned int v57; // r12d
  unsigned int v58; // eax
  __int64 v59; // rdx
  unsigned int *v60; // rax
  __int64 v61; // rdx
  __int64 v62; // r9
  unsigned __int8 *v63; // rax
  _QWORD *v64; // r12
  __int64 v65; // rdx
  unsigned int v66; // eax
  __int64 v67; // rdx
  __int64 v68; // r9
  __int64 result; // rax
  unsigned int v70; // edx
  __int64 v71; // rdi
  unsigned int v72; // r12d
  unsigned __int16 *v74; // rax
  unsigned __int8 *v75; // rax
  int v76; // edx
  __int64 v77; // rdx
  unsigned __int16 *v78; // rax
  __int64 v79; // r9
  int v80; // edx
  unsigned __int16 *v81; // rax
  __int64 v82; // r9
  unsigned __int8 *v83; // rax
  unsigned int v84; // edx
  int v85; // edx
  __int64 v86; // rdx
  unsigned int v87; // esi
  __int64 v88; // rdi
  __int64 v89; // rax
  __int64 v90; // rax
  __int64 v91; // rdx
  unsigned int v92; // edx
  __int64 v93; // rax
  __int64 v94; // rdi
  unsigned int v95; // r12d
  bool v96; // al
  int v97; // eax
  unsigned __int16 *v98; // rax
  int v99; // edx
  __int64 v100; // rax
  __int64 v101; // rdx
  unsigned __int8 *v102; // rax
  int v103; // edx
  unsigned __int16 *v104; // rax
  unsigned int v105; // edx
  __int64 v106; // rcx
  unsigned int v107; // eax
  __int64 v108; // rdx
  __int128 v109; // rax
  int v110; // edx
  __int128 v111; // [rsp-30h] [rbp-200h]
  __int128 v112; // [rsp-20h] [rbp-1F0h]
  __int64 v113; // [rsp-10h] [rbp-1E0h]
  __int64 v114; // [rsp+8h] [rbp-1C8h]
  unsigned int v115; // [rsp+14h] [rbp-1BCh]
  __int64 v116; // [rsp+18h] [rbp-1B8h]
  unsigned int v117; // [rsp+18h] [rbp-1B8h]
  unsigned int v118; // [rsp+18h] [rbp-1B8h]
  unsigned int v119; // [rsp+20h] [rbp-1B0h]
  unsigned int *v120; // [rsp+20h] [rbp-1B0h]
  int v121; // [rsp+20h] [rbp-1B0h]
  __int128 v122; // [rsp+20h] [rbp-1B0h]
  __int128 v123; // [rsp+30h] [rbp-1A0h]
  unsigned int v124; // [rsp+30h] [rbp-1A0h]
  _QWORD *v125; // [rsp+40h] [rbp-190h]
  __int128 v126; // [rsp+40h] [rbp-190h]
  __int64 v127; // [rsp+40h] [rbp-190h]
  int v128; // [rsp+40h] [rbp-190h]
  __int64 v129; // [rsp+40h] [rbp-190h]
  _QWORD *v130; // [rsp+40h] [rbp-190h]
  unsigned __int64 v131; // [rsp+58h] [rbp-178h]
  __int128 v132; // [rsp+60h] [rbp-170h]
  __int64 v133; // [rsp+60h] [rbp-170h]
  __int64 v134; // [rsp+60h] [rbp-170h]
  unsigned __int8 *v135; // [rsp+60h] [rbp-170h]
  unsigned int *v136; // [rsp+60h] [rbp-170h]
  unsigned int v137; // [rsp+70h] [rbp-160h]
  __int64 *v138; // [rsp+70h] [rbp-160h]
  __int128 v139; // [rsp+70h] [rbp-160h]
  unsigned __int64 v140; // [rsp+78h] [rbp-158h]
  __int128 v142; // [rsp+80h] [rbp-150h]
  __int128 v143; // [rsp+80h] [rbp-150h]
  int v144; // [rsp+138h] [rbp-98h]
  __int128 v145; // [rsp+140h] [rbp-90h] BYREF
  __int128 v146; // [rsp+150h] [rbp-80h] BYREF
  __m128i v147; // [rsp+160h] [rbp-70h] BYREF
  __m128i v148; // [rsp+170h] [rbp-60h] BYREF
  _QWORD v149[10]; // [rsp+180h] [rbp-50h] BYREF

  v11 = *a2;
  v12 = a2[1];
  *(_QWORD *)&v146 = 0;
  *(_QWORD *)&v145 = 0;
  DWORD2(v145) = 0;
  DWORD2(v146) = 0;
  v147.m128i_i64[0] = 0;
  v147.m128i_i32[2] = 0;
  v148.m128i_i64[0] = 0;
  v148.m128i_i32[2] = 0;
  sub_375E510((__int64)a1, v11, v12, (__int64)&v145, (__int64)&v146);
  sub_375E510((__int64)a1, *(_QWORD *)a3, *(_QWORD *)(a3 + 8), (__int64)&v147, (__int64)&v148);
  v14 = *a4;
  if ( *a4 == 17 || v14 == 22 )
  {
    if ( v148.m128i_i64[0] == v147.m128i_i64[0]
      && v148.m128i_i32[2] == v147.m128i_i32[2]
      && sub_33CF460(v148.m128i_i64[0]) )
    {
      v98 = (unsigned __int16 *)(*(_QWORD *)(v145 + 48) + 16LL * DWORD2(v145));
      *a2 = (unsigned __int64)sub_3406EB0((_QWORD *)a1[1], 0xBAu, a5, *v98, *((_QWORD *)v98 + 1), v13, v145, v146);
      *((_DWORD *)a2 + 2) = v99;
      *(_QWORD *)a3 = v147.m128i_i64[0];
      result = v147.m128i_u32[2];
      *(_DWORD *)(a3 + 8) = v147.m128i_i32[2];
    }
    else
    {
      v74 = (unsigned __int16 *)(*(_QWORD *)(v145 + 48) + 16LL * DWORD2(v145));
      v75 = sub_3406EB0((_QWORD *)a1[1], 0xBCu, a5, *v74, *((_QWORD *)v74 + 1), v13, v145, *(_OWORD *)&v147);
      v144 = v76;
      v77 = v145;
      *a2 = (unsigned __int64)v75;
      *((_DWORD *)a2 + 2) = v144;
      v78 = (unsigned __int16 *)(*(_QWORD *)(v77 + 48) + 16LL * DWORD2(v145));
      *(_QWORD *)a3 = sub_3406EB0((_QWORD *)a1[1], 0xBCu, a5, *v78, *((_QWORD *)v78 + 1), v79, v146, *(_OWORD *)&v148);
      *(_DWORD *)(a3 + 8) = v80;
      v81 = (unsigned __int16 *)(*(_QWORD *)(*a2 + 48) + 16LL * *((unsigned int *)a2 + 2));
      v83 = sub_3406EB0((_QWORD *)a1[1], 0xBBu, a5, *v81, *((_QWORD *)v81 + 1), v82, *(_OWORD *)a2, *(_OWORD *)a3);
      *a2 = (unsigned __int64)v83;
      *((_DWORD *)a2 + 2) = v84;
      *(_QWORD *)a3 = sub_3400BD0(
                        a1[1],
                        0,
                        a5,
                        *(unsigned __int16 *)(*((_QWORD *)v83 + 6) + 16LL * v84),
                        *(_QWORD *)(*((_QWORD *)v83 + 6) + 16LL * v84 + 8),
                        0,
                        a6,
                        0);
      *(_DWORD *)(a3 + 8) = v85;
      return v113;
    }
    return result;
  }
  v15 = *(_QWORD *)a3;
  v16 = *(_DWORD *)(*(_QWORD *)a3 + 24LL);
  if ( v16 != 11 && v16 != 35 )
    goto LABEL_5;
  if ( v14 != 20 )
  {
    if ( v14 != 18 )
    {
LABEL_5:
      switch ( v14 )
      {
        case 0xAu:
        case 0x12u:
          goto LABEL_40;
        case 0xBu:
        case 0x13u:
          v17 = 11;
          goto LABEL_7;
        case 0xCu:
        case 0x14u:
          goto LABEL_33;
        case 0xDu:
        case 0x15u:
          v17 = 13;
          goto LABEL_7;
        default:
          BUG();
      }
    }
    v71 = *(_QWORD *)(v15 + 96);
    v72 = *(_DWORD *)(v71 + 32);
    if ( v72 )
    {
      if ( !(v72 <= 0x40
           ? 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v72) == *(_QWORD *)(v71 + 24)
           : v72 == (unsigned int)sub_C445E0(v71 + 24)) )
      {
LABEL_40:
        v17 = 10;
        goto LABEL_7;
      }
    }
LABEL_56:
    *a2 = v146;
    *((_DWORD *)a2 + 2) = DWORD2(v146);
    *(_QWORD *)a3 = v148.m128i_i64[0];
    result = v148.m128i_u32[2];
    *(_DWORD *)(a3 + 8) = v148.m128i_i32[2];
    return result;
  }
  v94 = *(_QWORD *)(v15 + 96);
  v95 = *(_DWORD *)(v94 + 32);
  if ( v95 <= 0x40 )
    v96 = *(_QWORD *)(v94 + 24) == 0;
  else
    v96 = v95 == (unsigned int)sub_C444A0(v94 + 24);
  if ( v96 )
    goto LABEL_56;
LABEL_33:
  v17 = 12;
LABEL_7:
  v18 = a1[1];
  v149[0] = 0;
  v149[1] = 0x100000001LL;
  v149[2] = v18;
  v19 = *a1;
  v20 = *(_QWORD *)(v145 + 48) + 16LL * DWORD2(v145);
  v21 = *(_WORD *)v20;
  v22 = *(_QWORD *)(v20 + 8);
  if ( *(_WORD *)v20 )
  {
    if ( *(_QWORD *)(v19 + 8LL * v21 + 112) )
    {
      v89 = *(unsigned __int16 *)(*(_QWORD *)(v147.m128i_i64[0] + 48) + 16LL * v147.m128i_u32[2]);
      if ( (_WORD)v89 )
      {
        if ( *(_QWORD *)(v19 + 8 * v89 + 112) )
        {
          v136 = (unsigned int *)*a1;
          v90 = sub_38137B0(v19, v18, v21, v22);
          v133 = (__int64)sub_348D3E0(
                            v136,
                            v90,
                            v91,
                            v145,
                            *((__int64 *)&v145 + 1),
                            v17,
                            a6,
                            *(_OWORD *)&v147,
                            0,
                            (__int64)v149,
                            a5);
          v115 = v92;
          v131 = v92;
          if ( v133 )
            goto LABEL_9;
          v18 = a1[1];
          v19 = *a1;
          v93 = *(_QWORD *)(v145 + 48) + 16LL * DWORD2(v145);
          v21 = *(_WORD *)v93;
          v22 = *(_QWORD *)(v93 + 8);
        }
      }
    }
  }
  v125 = (_QWORD *)v18;
  v23 = sub_38137B0(v19, v18, v21, v22);
  v116 = v24;
  v119 = v23;
  v123 = v145;
  v132 = (__int128)_mm_loadu_si128(&v147);
  *(_QWORD *)&v25 = sub_33ED040(v125, v17);
  v133 = sub_340F900(v125, 0xD0u, a5, v119, v116, (__int64)v125, v123, v132, v25);
  v115 = v26;
  v131 = v26 | v131 & 0xFFFFFFFF00000000LL;
LABEL_9:
  v27 = *(_QWORD *)(v146 + 48) + 16LL * DWORD2(v146);
  v28 = *(_WORD *)v27;
  if ( !*(_WORD *)v27
    || (v29 = (unsigned int *)*a1, !*(_QWORD *)(*a1 + 8LL * v28 + 112))
    || (v30 = *(unsigned __int16 *)(*(_QWORD *)(v148.m128i_i64[0] + 48) + 16LL * v148.m128i_u32[2]), !(_WORD)v30)
    || !*(_QWORD *)&v29[2 * v30 + 28]
    || (v137 = *a4,
        v31 = sub_38137B0(*a1, a1[1], v28, *(_QWORD *)(v27 + 8)),
        v33 = (__int64)sub_348D3E0(
                         v29,
                         v31,
                         v32,
                         v146,
                         *((__int64 *)&v146 + 1),
                         v137,
                         a6,
                         *(_OWORD *)&v148,
                         0,
                         (__int64)v149,
                         a5),
        v124 = v34,
        v140 = v34,
        !v33) )
  {
    v35 = (_QWORD *)a1[1];
    *(_QWORD *)&v126 = sub_33ED040(v35, *a4);
    *((_QWORD *)&v126 + 1) = v36;
    v37 = sub_38137B0(
            *a1,
            a1[1],
            *(unsigned __int16 *)(*(_QWORD *)(v146 + 48) + 16LL * DWORD2(v146)),
            *(_QWORD *)(*(_QWORD *)(v146 + 48) + 16LL * DWORD2(v146) + 8));
    v33 = sub_340F900(v35, 0xD0u, a5, v37, v38, v39, v146, *(_OWORD *)&v148, v126);
    v124 = v40;
    v140 = v40 | v140 & 0xFFFFFFFF00000000LL;
  }
  v41 = *(_DWORD *)(v133 + 24);
  if ( v41 == 35 || (v42 = 0, v41 == 11) )
    v42 = v133;
  v43 = *(_DWORD *)(v33 + 24);
  v44 = *a4 & 1;
  if ( v43 == 11 || v43 == 35 )
  {
    v86 = *(_QWORD *)(v33 + 96);
    v87 = *(_DWORD *)(v86 + 32);
    v88 = v86 + 24;
    if ( v44 )
    {
      if ( v87 <= 0x40 )
      {
        v47 = *(_QWORD *)(v86 + 24) == 0;
      }
      else
      {
        v128 = *(_DWORD *)(v86 + 32);
        v47 = (unsigned int)sub_C444A0(v88) == v128;
      }
      goto LABEL_23;
    }
    if ( v87 <= 0x40 )
    {
      if ( *(_QWORD *)(v86 + 24) == 1 )
        goto LABEL_60;
    }
    else
    {
      v121 = *(_DWORD *)(v86 + 32);
      v129 = v42;
      v97 = sub_C444A0(v88);
      v42 = v129;
      if ( v97 == v121 - 1 )
        goto LABEL_60;
    }
  }
  else if ( v44 )
  {
    goto LABEL_24;
  }
  if ( !v42 )
    goto LABEL_24;
  v45 = *(_QWORD *)(v42 + 96);
  v46 = *(_DWORD *)(v45 + 32);
  if ( v46 <= 0x40 )
    v47 = *(_QWORD *)(v45 + 24) == 0;
  else
    v47 = v46 == (unsigned int)sub_C444A0(v45 + 24);
LABEL_23:
  if ( v47 )
  {
LABEL_60:
    *a2 = v33;
    *((_DWORD *)a2 + 2) = v124;
    *(_QWORD *)a3 = 0;
    *(_DWORD *)(a3 + 8) = 0;
    return v124;
  }
LABEL_24:
  if ( (_QWORD)v146 == v148.m128i_i64[0] && v148.m128i_i32[2] == DWORD2(v146) )
  {
    *a2 = v133;
    *((_DWORD *)a2 + 2) = v115;
    *(_QWORD *)a3 = 0;
    *(_DWORD *)(a3 + 8) = 0;
    return v115;
  }
  v48 = (unsigned __int16 *)(*(_QWORD *)(v146 + 48) + 16LL * DWORD2(v146));
  v127 = *((_QWORD *)v48 + 1);
  v117 = *v48;
  v49 = sub_3814400(*a1, *(_QWORD *)(a1[1] + 64), v117, v127);
  v120 = (unsigned int *)*a1;
  v51 = *a1;
  if ( (unsigned __int8)sub_3813820(*a1, 0xD1u, v49, 0, v50) )
  {
    v52 = *a4;
    if ( *a4 == 18 )
    {
      *a4 = 20;
    }
    else if ( v52 > 0x12 )
    {
      if ( v52 != 21 )
        goto LABEL_32;
      *a4 = 19;
    }
    else if ( v52 == 10 )
    {
      *a4 = 12;
    }
    else
    {
      if ( v52 != 13 )
      {
LABEL_32:
        v56 = (unsigned __int16 *)(*(_QWORD *)(v145 + 48) + 16LL * DWORD2(v145));
        v57 = *v56;
        v138 = (__int64 *)a1[1];
        v134 = *((_QWORD *)v56 + 1);
        v58 = sub_38137B0(v51, (__int64)v138, v57, v134);
        v60 = (unsigned int *)sub_33E5110(v138, v57, v134, v58, v59);
        v63 = sub_3411F20((_QWORD *)a1[1], 79, a5, v60, v61, v62, v145, *(_OWORD *)&v147);
        v64 = (_QWORD *)a1[1];
        v135 = v63;
        *(_QWORD *)&v139 = sub_33ED040(v64, *a4);
        *((_QWORD *)&v139 + 1) = v65;
        *(_QWORD *)&v142 = v135;
        *((_QWORD *)&v142 + 1) = 1;
        v66 = sub_38137B0(*a1, a1[1], v117, v127);
        *a2 = (unsigned __int64)sub_33FC130(v64, 209, a5, v66, v67, v68, v146, *(_OWORD *)&v148, v142, v139);
        *((_DWORD *)a2 + 2) = v70;
        *(_QWORD *)a3 = 0;
        *(_DWORD *)(a3 + 8) = 0;
        return v70;
      }
      *a4 = 11;
    }
    v53 = _mm_loadu_si128((const __m128i *)&v145);
    *(_QWORD *)&v145 = v147.m128i_i64[0];
    DWORD2(v145) = v147.m128i_i32[2];
    v54 = v53.m128i_i32[2];
    v147.m128i_i64[0] = v53.m128i_i64[0];
    v55 = _mm_loadu_si128((const __m128i *)&v146);
    v147.m128i_i32[2] = v54;
    v51 = *a1;
    *(_QWORD *)&v146 = v148.m128i_i64[0];
    v148.m128i_i64[0] = v55.m128i_i64[0];
    DWORD2(v146) = v148.m128i_i32[2];
    v148.m128i_i32[2] = v55.m128i_i32[2];
    goto LABEL_32;
  }
  v100 = sub_38137B0((__int64)v120, a1[1], v117, v127);
  v102 = sub_348D3E0(v120, v100, v101, v146, *((__int64 *)&v146 + 1), 0x11u, a6, *(_OWORD *)&v148, 0, (__int64)v149, a5);
  *a2 = (unsigned __int64)v102;
  *((_DWORD *)a2 + 2) = v103;
  if ( !v102 )
  {
    v106 = v127;
    v130 = (_QWORD *)a1[1];
    v107 = sub_38137B0(*a1, (__int64)v130, v117, v106);
    v114 = v108;
    v118 = v107;
    v122 = v146;
    v143 = (__int128)_mm_loadu_si128(&v148);
    *(_QWORD *)&v109 = sub_33ED040(v130, 0x11u);
    *a2 = sub_340F900(v130, 0xD0u, a5, v118, v114, (__int64)v130, v122, v143, v109);
    *((_DWORD *)a2 + 2) = v110;
  }
  v104 = (unsigned __int16 *)(*(_QWORD *)(v133 + 48) + 16LL * v115);
  *((_QWORD *)&v112 + 1) = v124 | v140 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v112 = v33;
  *((_QWORD *)&v111 + 1) = v115 | v131 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v111 = v133;
  *a2 = sub_3288B20(a1[1], a5, *v104, *((_QWORD *)v104 + 1), *a2, a2[1], v111, v112, 0);
  *((_DWORD *)a2 + 2) = v105;
  *(_QWORD *)a3 = 0;
  *(_DWORD *)(a3 + 8) = 0;
  return v105;
}
