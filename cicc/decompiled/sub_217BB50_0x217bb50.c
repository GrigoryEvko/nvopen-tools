// Function: sub_217BB50
// Address: 0x217bb50
//
__int64 __fastcall sub_217BB50(__int64 a1, __int64 a2, unsigned int a3, __m128i a4, double a5, __m128i a6)
{
  __int64 v8; // rdx
  char v9; // al
  __int64 v10; // rdx
  unsigned __int8 v11; // dl
  __int64 result; // rax
  __int64 *v13; // rax
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  __int64 v16; // r13
  __int64 v17; // r11
  unsigned int v18; // edx
  unsigned __int64 v19; // r14
  unsigned __int64 v20; // r15
  unsigned int v21; // eax
  __int64 v22; // r11
  unsigned int v23; // ecx
  unsigned int v24; // r15d
  unsigned int v25; // edx
  __int64 v26; // rax
  char v27; // di
  __int64 v28; // rax
  unsigned int v29; // eax
  __int64 v30; // rcx
  unsigned int v31; // edx
  __int64 v32; // r11
  __int64 v33; // rsi
  __int64 *v34; // r10
  __int64 v35; // rax
  unsigned int v36; // edx
  __int64 v37; // rsi
  __int64 *v38; // r10
  __int64 v39; // rax
  unsigned int v40; // edx
  __int64 v41; // r8
  int v42; // eax
  __int64 v43; // rsi
  __int64 v44; // r12
  __int64 v45; // rdx
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // rax
  __int64 *v49; // rax
  __int64 v50; // rsi
  __int64 v51; // r12
  __int64 v52; // rdx
  __int64 v53; // r8
  __int64 v54; // r9
  __int64 v55; // rax
  __int64 *v56; // rax
  unsigned int v57; // eax
  __int64 v58; // rsi
  __int64 v59; // r9
  __int64 v60; // r8
  _QWORD *v61; // r12
  __int64 v62; // rax
  __int64 v63; // r15
  unsigned __int8 *v64; // rcx
  __int64 v65; // rax
  unsigned __int8 v66; // dl
  const void **v67; // r14
  __int64 v68; // r8
  unsigned int v69; // edx
  unsigned int v70; // edx
  __int64 v71; // rsi
  __int64 *v72; // r10
  int v73; // eax
  __int64 v74; // rcx
  int v75; // r9d
  unsigned int v76; // eax
  __int64 v77; // r12
  __int64 v78; // rdx
  __int64 v79; // rcx
  int v80; // r8d
  int v81; // r9d
  __int64 v82; // r12
  __int64 v83; // r13
  _QWORD *v84; // r9
  const __m128i *v85; // rsi
  __int64 v86; // rax
  __int64 v87; // rdx
  __int64 v88; // rcx
  int v89; // r8d
  int v90; // r9d
  __int64 v91; // r12
  __int64 v92; // r13
  _QWORD *v93; // r9
  __int128 v94; // [rsp-10h] [rbp-1B0h]
  __int128 v95; // [rsp-10h] [rbp-1B0h]
  __int128 v96; // [rsp-10h] [rbp-1B0h]
  __int64 v97; // [rsp+0h] [rbp-1A0h]
  __int64 *v98; // [rsp+8h] [rbp-198h]
  __int64 v99; // [rsp+8h] [rbp-198h]
  __int64 v100; // [rsp+8h] [rbp-198h]
  unsigned int v101; // [rsp+8h] [rbp-198h]
  __int64 v102; // [rsp+8h] [rbp-198h]
  __int64 v103; // [rsp+8h] [rbp-198h]
  __int64 v104; // [rsp+10h] [rbp-190h]
  int v105; // [rsp+1Ch] [rbp-184h]
  __int64 v106; // [rsp+20h] [rbp-180h]
  __int64 *v107; // [rsp+20h] [rbp-180h]
  __int64 v108; // [rsp+20h] [rbp-180h]
  __int64 v109; // [rsp+20h] [rbp-180h]
  __int64 v110; // [rsp+20h] [rbp-180h]
  __int64 v111; // [rsp+28h] [rbp-178h]
  unsigned int v112; // [rsp+28h] [rbp-178h]
  unsigned __int64 v113; // [rsp+28h] [rbp-178h]
  unsigned __int64 v114; // [rsp+30h] [rbp-170h]
  __int64 v115; // [rsp+30h] [rbp-170h]
  __int64 v116; // [rsp+30h] [rbp-170h]
  _QWORD *v117; // [rsp+30h] [rbp-170h]
  __int64 v118; // [rsp+30h] [rbp-170h]
  __int64 v119; // [rsp+30h] [rbp-170h]
  __int64 v120; // [rsp+30h] [rbp-170h]
  __int64 v121; // [rsp+30h] [rbp-170h]
  __int64 v122; // [rsp+38h] [rbp-168h]
  __int64 v123; // [rsp+38h] [rbp-168h]
  __int64 v124; // [rsp+38h] [rbp-168h]
  __int64 v125; // [rsp+40h] [rbp-160h]
  char v126; // [rsp+40h] [rbp-160h]
  __int64 *v128; // [rsp+48h] [rbp-158h]
  __int64 v129; // [rsp+48h] [rbp-158h]
  _QWORD *v130; // [rsp+48h] [rbp-158h]
  unsigned int v131; // [rsp+D4h] [rbp-CCh] BYREF
  __int64 v132; // [rsp+D8h] [rbp-C8h] BYREF
  unsigned __int64 v133; // [rsp+E0h] [rbp-C0h] BYREF
  unsigned __int64 v134; // [rsp+E8h] [rbp-B8h] BYREF
  char v135[8]; // [rsp+F0h] [rbp-B0h] BYREF
  __int64 v136; // [rsp+F8h] [rbp-A8h]
  __int64 v137; // [rsp+100h] [rbp-A0h] BYREF
  int v138; // [rsp+108h] [rbp-98h]
  __m128i v139; // [rsp+110h] [rbp-90h] BYREF
  _QWORD *v140; // [rsp+120h] [rbp-80h] BYREF
  __int64 v141; // [rsp+128h] [rbp-78h]
  _QWORD v142[14]; // [rsp+130h] [rbp-70h] BYREF

  v8 = *(_QWORD *)(a1 + 40);
  v9 = *(_BYTE *)v8;
  v10 = *(_QWORD *)(v8 + 8);
  v135[0] = v9;
  v136 = v10;
  if ( v9 )
  {
    v11 = v9 - 14;
    if ( (unsigned __int8)(v9 - 2) > 5u && v11 > 0x47u || v11 <= 0x5Fu )
      return 0;
  }
  else if ( !sub_1F58CF0((__int64)v135) || sub_1F58D20((__int64)v135) )
  {
    return 0;
  }
  v13 = *(__int64 **)(a1 + 32);
  v132 = 0;
  v133 = 0;
  v14 = sub_2171180(*v13, v13[1], &v133);
  v114 = v15;
  v16 = v14;
  if ( !v14
    || (v17 = sub_2170FD0(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 40LL), *(_DWORD *)(*(_QWORD *)(a1 + 32) + 48LL), &v132),
        v19 = v18,
        !v17) )
  {
    v16 = sub_2171180(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a1 + 32) + 48LL), &v133);
    v114 = v69 | v114 & 0xFFFFFFFF00000000LL;
    if ( !v16 )
      return 0;
    v17 = sub_2170FD0(**(_QWORD **)(a1 + 32), *(_DWORD *)(*(_QWORD *)(a1 + 32) + 8LL), &v132);
    v19 = v70;
    if ( !v17 )
      return 0;
  }
  v20 = v133 + v132;
  if ( v135[0] )
  {
    v23 = sub_216FFF0(v135[0]);
  }
  else
  {
    v125 = v17;
    v21 = sub_1F58D40((__int64)v135);
    v22 = v125;
    v23 = v21;
  }
  if ( v20 < v23 )
    return 0;
  if ( v23 == 32 )
  {
LABEL_16:
    v126 = 5;
    v24 = 5;
    v105 = 164;
    goto LABEL_17;
  }
  if ( v23 != 64 )
  {
    result = 0;
    if ( v23 != 16 )
      return result;
    goto LABEL_16;
  }
  if ( !byte_4FD2B00 || a3 <= 0x31 )
    return 0;
  v126 = 6;
  v24 = 6;
  v105 = 165;
LABEL_17:
  v111 = v22;
  v134 = 0;
  v106 = sub_2171180(v22, v19, &v134);
  if ( v106 )
  {
    v26 = *(_QWORD *)(v111 + 40) + 16LL * (unsigned int)v19;
    v27 = *(_BYTE *)v26;
    v28 = *(_QWORD *)(v26 + 8);
    LOBYTE(v140) = v27;
    v141 = v28;
    if ( v27 )
    {
      v112 = v25;
      v29 = sub_216FFF0(v27);
      v31 = v112;
    }
    else
    {
      v101 = v25;
      v29 = sub_1F58D40((__int64)&v140);
      v31 = v101;
      v30 = v106;
    }
    v32 = v30;
    v113 = v29 - v134;
    v19 = v31 | v19 & 0xFFFFFFFF00000000LL;
  }
  else
  {
    if ( v135[0] )
    {
      v76 = sub_216FFF0(v135[0]);
    }
    else
    {
      v76 = sub_1F58D40((__int64)v135);
      v32 = v111;
    }
    v106 = v32;
    v113 = v76 - v132;
  }
  if ( *(_BYTE *)(*(_QWORD *)(v16 + 40) + 16LL * (unsigned int)v114) != v126 )
  {
    v33 = *(_QWORD *)(a1 + 72);
    v140 = (_QWORD *)v33;
    v34 = *(__int64 **)(a2 + 16);
    if ( v33 )
    {
      v97 = v32;
      v98 = *(__int64 **)(a2 + 16);
      sub_1623A60((__int64)&v140, v33, 2);
      v32 = v97;
      v34 = v98;
    }
    LOBYTE(v24) = v126;
    v99 = v32;
    LODWORD(v141) = *(_DWORD *)(a1 + 64);
    v35 = sub_1D323C0(v34, v16, v114, (__int64)&v140, v24, 0, *(double *)a4.m128i_i64, a5, *(double *)a6.m128i_i64);
    v32 = v99;
    v16 = v35;
    v114 = v36 | v114 & 0xFFFFFFFF00000000LL;
    if ( v140 )
    {
      sub_161E7C0((__int64)&v140, (__int64)v140);
      v32 = v99;
    }
  }
  if ( *(_BYTE *)(*(_QWORD *)(v106 + 40) + 16LL * (unsigned int)v19) != v126 )
  {
    v37 = *(_QWORD *)(a1 + 72);
    v140 = (_QWORD *)v37;
    v38 = *(__int64 **)(a2 + 16);
    if ( v37 )
    {
      v100 = v32;
      v107 = *(__int64 **)(a2 + 16);
      sub_1623A60((__int64)&v140, v37, 2);
      v32 = v100;
      v38 = v107;
    }
    LOBYTE(v24) = v126;
    LODWORD(v141) = *(_DWORD *)(a1 + 64);
    v39 = sub_1D321C0(v38, v32, v19, (__int64)&v140, v24, 0, *(double *)a4.m128i_i64, a5, *(double *)a6.m128i_i64);
    v32 = v39;
    v19 = v40 | v19 & 0xFFFFFFFF00000000LL;
    if ( v140 )
    {
      v108 = v39;
      sub_161E7C0((__int64)&v140, (__int64)v140);
      v32 = v108;
    }
  }
  if ( v133 == 16 && v132 == 16 )
  {
    if ( v135[0] )
    {
      v73 = sub_216FFF0(v135[0]);
    }
    else
    {
      v103 = v32;
      v110 = v132;
      v73 = sub_1F58D40((__int64)v135);
      v32 = v103;
      v41 = v110;
    }
    if ( v73 == 32 )
    {
      v140 = v142;
      v141 = 0x400000000LL;
      v85 = *(const __m128i **)(v16 + 32);
      if ( *(_BYTE *)(*(_QWORD *)(v85->m128i_i64[0] + 40) + 16LL * v85->m128i_u32[2]) == 4 )
      {
        v86 = *(_QWORD *)(v32 + 32);
        if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v86 + 40LL) + 16LL * *(unsigned int *)(v86 + 8)) == 4 )
        {
          v118 = v32;
          sub_1D23890((__int64)&v140, v85, *(_QWORD *)v86, v74, v41, v75);
          sub_1D23890((__int64)&v140, *(const __m128i **)(v118 + 32), v87, v88, v89, v90);
          v91 = (__int64)v140;
          v92 = (unsigned int)v141;
          v93 = *(_QWORD **)(a2 + 16);
          v139.m128i_i64[0] = *(_QWORD *)(a1 + 72);
          if ( v139.m128i_i64[0] )
          {
            v130 = v93;
            sub_21700C0(v139.m128i_i64);
            v93 = v130;
          }
          *((_QWORD *)&v96 + 1) = v92;
          LOBYTE(v24) = v126;
          *(_QWORD *)&v96 = v91;
          v139.m128i_i32[2] = *(_DWORD *)(a1 + 64);
          v119 = sub_1D2CDB0(v93, 4449, (__int64)&v139, v24, 0, (__int64)v93, v96);
          sub_17CD270(v139.m128i_i64);
          result = v119;
          if ( v140 != v142 )
          {
            _libc_free((unsigned __int64)v140);
            return v119;
          }
          return result;
        }
      }
    }
    if ( !v113 )
      goto LABEL_79;
LABEL_37:
    v142[0] = v32;
    v140 = v142;
    v142[1] = v19;
    v142[3] = v114;
    v142[2] = v16;
    v141 = 0x400000002LL;
    if ( v105 == 164 && (unsigned __int8)sub_216FE40(v41, v113, (int *)&v131) )
    {
      v77 = *(_QWORD *)(a2 + 16);
      v137 = *(_QWORD *)(a1 + 72);
      if ( v137 )
        sub_21700C0(&v137);
      v138 = *(_DWORD *)(a1 + 64);
      v139.m128i_i64[0] = sub_1D38BB0(v77, v131, (__int64)&v137, 5, 0, 1, a4, a5, a6, 0);
      v139.m128i_i64[1] = v78;
      sub_1D23890((__int64)&v140, &v139, v78, v79, v80, v81);
      sub_17CD270(&v137);
      v82 = (__int64)v140;
      v83 = (unsigned int)v141;
      v84 = *(_QWORD **)(a2 + 16);
      v139.m128i_i64[0] = *(_QWORD *)(a1 + 72);
      if ( v139.m128i_i64[0] )
      {
        v117 = v84;
        sub_21700C0(v139.m128i_i64);
        v84 = v117;
      }
      *((_QWORD *)&v95 + 1) = v83;
      *(_QWORD *)&v95 = v82;
      LOBYTE(v24) = v126;
      v139.m128i_i32[2] = *(_DWORD *)(a1 + 64);
      v16 = sub_1D2CDB0(v84, 3243, (__int64)&v139, v24, 0, (__int64)v84, v95);
      sub_17CD270(v139.m128i_i64);
    }
    else
    {
      v43 = *(_QWORD *)(a1 + 72);
      v139.m128i_i64[0] = v43;
      v44 = *(_QWORD *)(a2 + 16);
      if ( v43 )
        sub_1623A60((__int64)&v139, v43, 2);
      v139.m128i_i32[2] = *(_DWORD *)(a1 + 64);
      v46 = sub_1D38BB0(v44, v132, (__int64)&v139, 5, 0, 1, a4, a5, a6, 0);
      v47 = v45;
      v48 = (unsigned int)v141;
      if ( (unsigned int)v141 >= HIDWORD(v141) )
      {
        v124 = v45;
        v121 = v46;
        sub_16CD150((__int64)&v140, v142, 0, 16, v46, v45);
        v48 = (unsigned int)v141;
        v46 = v121;
        v47 = v124;
      }
      v49 = &v140[2 * v48];
      *v49 = v46;
      v49[1] = v47;
      LODWORD(v141) = v141 + 1;
      if ( v139.m128i_i64[0] )
        sub_161E7C0((__int64)&v139, v139.m128i_i64[0]);
      v50 = *(_QWORD *)(a1 + 72);
      v51 = *(_QWORD *)(a2 + 16);
      v139.m128i_i64[0] = v50;
      if ( v50 )
        sub_1623A60((__int64)&v139, v50, 2);
      v139.m128i_i32[2] = *(_DWORD *)(a1 + 64);
      v53 = sub_1D38BB0(v51, v113, (__int64)&v139, 5, 0, 1, a4, a5, a6, 0);
      v54 = v52;
      v55 = (unsigned int)v141;
      if ( (unsigned int)v141 >= HIDWORD(v141) )
      {
        v123 = v52;
        v120 = v53;
        sub_16CD150((__int64)&v140, v142, 0, 16, v53, v52);
        v55 = (unsigned int)v141;
        v53 = v120;
        v54 = v123;
      }
      v56 = &v140[2 * v55];
      *v56 = v53;
      v56[1] = v54;
      v57 = v141 + 1;
      LODWORD(v141) = v141 + 1;
      if ( v139.m128i_i64[0] )
      {
        sub_161E7C0((__int64)&v139, v139.m128i_i64[0]);
        v57 = v141;
      }
      v58 = *(_QWORD *)(a1 + 72);
      v59 = v57;
      v60 = (__int64)v140;
      v139.m128i_i64[0] = v58;
      v61 = *(_QWORD **)(a2 + 16);
      if ( v58 )
      {
        v115 = (__int64)v140;
        v122 = v57;
        sub_1623A60((__int64)&v139, v58, 2);
        v60 = v115;
        v59 = v122;
      }
      *((_QWORD *)&v94 + 1) = v59;
      LOBYTE(v24) = v126;
      *(_QWORD *)&v94 = v60;
      v139.m128i_i32[2] = *(_DWORD *)(a1 + 64);
      v16 = sub_1D2CDB0(v61, v105, (__int64)&v139, v24, 0, v59, v94);
      if ( v139.m128i_i64[0] )
        sub_161E7C0((__int64)&v139, v139.m128i_i64[0]);
    }
    if ( v140 != v142 )
      _libc_free((unsigned __int64)v140);
    v62 = 0;
    v63 = 0;
    goto LABEL_57;
  }
  if ( v113 )
  {
    v41 = v132;
    if ( !v132 )
    {
      if ( v135[0] )
      {
        v42 = sub_216FFF0(v135[0]);
      }
      else
      {
        v102 = v32;
        v109 = v132;
        v42 = sub_1F58D40((__int64)v135);
        v32 = v102;
        v41 = v109;
      }
      if ( v42 == v113 )
      {
        v16 = v32;
        v63 = (unsigned int)v19;
        v104 = (unsigned int)v19;
        v62 = 16LL * (unsigned int)v19;
        goto LABEL_57;
      }
    }
    goto LABEL_37;
  }
LABEL_79:
  v63 = (unsigned int)v114;
  v104 = (unsigned int)v114;
  v62 = 16LL * (unsigned int)v114;
LABEL_57:
  v64 = *(unsigned __int8 **)(a1 + 40);
  v65 = *(_QWORD *)(v16 + 40) + v62;
  v66 = *v64;
  v67 = (const void **)*((_QWORD *)v64 + 1);
  v68 = *v64;
  if ( *(_BYTE *)v65 == *v64 && (*(const void ***)(v65 + 8) == v67 || *(_BYTE *)v65) )
    return v16;
  v71 = *(_QWORD *)(a1 + 72);
  v140 = (_QWORD *)v71;
  v72 = *(__int64 **)(a2 + 16);
  if ( v71 )
  {
    v116 = v66;
    v128 = *(__int64 **)(a2 + 16);
    sub_1623A60((__int64)&v140, v71, 2);
    v68 = v116;
    v72 = v128;
  }
  LODWORD(v141) = *(_DWORD *)(a1 + 64);
  result = sub_1D321C0(
             v72,
             v16,
             v63 | v104 & 0xFFFFFFFF00000000LL,
             (__int64)&v140,
             v68,
             v67,
             *(double *)a4.m128i_i64,
             a5,
             *(double *)a6.m128i_i64);
  if ( v140 )
  {
    v129 = result;
    sub_161E7C0((__int64)&v140, (__int64)v140);
    return v129;
  }
  return result;
}
