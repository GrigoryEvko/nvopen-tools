// Function: sub_23D1580
// Address: 0x23d1580
//
__int64 __fastcall sub_23D1580(__int64 a1, __m128i *a2, _BYTE *a3, __int64 *a4)
{
  __int64 v5; // rax
  unsigned int v6; // r15d
  __int64 v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // r13
  __int64 v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // rax
  int v21; // edx
  unsigned int v22; // eax
  unsigned __int8 *v23; // r10
  unsigned int v24; // eax
  unsigned __int8 *v25; // r10
  __int64 v26; // rax
  _QWORD *v27; // rdx
  __int64 v28; // rax
  _QWORD *v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // rax
  __int64 *v32; // rsi
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rcx
  __int64 v37; // r13
  __int64 v38; // rax
  unsigned __int8 v39; // al
  unsigned __int8 *v40; // rdi
  __int64 v41; // rdx
  _QWORD *v42; // rdx
  _QWORD *v43; // rdx
  _QWORD *v44; // rdx
  _QWORD *v45; // rdx
  __int64 v46; // rdi
  __int64 v47; // rax
  int v48; // ebx
  __int64 v49; // r12
  __int64 v50; // rax
  __int64 v51; // r14
  __int64 *v52; // rax
  __int64 *v53; // rax
  __int64 v54; // rax
  __int64 v55; // rdi
  __int64 v56; // rax
  __int64 v57; // rdi
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rcx
  __int64 v61; // rax
  __int64 v62; // rcx
  unsigned __int8 *v63; // rax
  char v64; // dl
  unsigned __int8 *v65; // rcx
  char *v66; // r8
  char *v67; // r9
  __int64 v68; // rax
  __int64 v69; // rsi
  __int64 v70; // rax
  _QWORD *v71; // rax
  __int64 v72; // rax
  __int64 v73; // rax
  _QWORD *v74; // rdx
  unsigned __int32 v75; // ecx
  __int64 v76; // rdx
  __m128i v77; // xmm1
  __m128i v78; // xmm5
  __int64 v79; // rax
  _BYTE *v80; // rax
  __int64 v81; // rt1
  unsigned __int64 v82; // rdi
  unsigned int v83; // ecx
  __int64 v84; // rsi
  __m128i v85; // xmm3
  _BYTE *v86; // [rsp+8h] [rbp-458h]
  __m128i *v87; // [rsp+10h] [rbp-450h]
  __int64 v88; // [rsp+20h] [rbp-440h]
  __int64 v89; // [rsp+30h] [rbp-430h]
  __int64 v90; // [rsp+38h] [rbp-428h]
  __int64 v91; // [rsp+40h] [rbp-420h]
  __int64 v92; // [rsp+48h] [rbp-418h]
  __int64 v93; // [rsp+50h] [rbp-410h]
  __int64 v94; // [rsp+58h] [rbp-408h]
  char v95; // [rsp+60h] [rbp-400h]
  char v96; // [rsp+60h] [rbp-400h]
  unsigned __int64 v97; // [rsp+60h] [rbp-400h]
  char v98; // [rsp+68h] [rbp-3F8h]
  __int64 v99; // [rsp+68h] [rbp-3F8h]
  __int64 v100; // [rsp+70h] [rbp-3F0h]
  __int64 v101; // [rsp+78h] [rbp-3E8h]
  unsigned __int8 *v102; // [rsp+80h] [rbp-3E0h]
  __int64 v103; // [rsp+80h] [rbp-3E0h]
  __int64 v104; // [rsp+80h] [rbp-3E0h]
  char v105; // [rsp+80h] [rbp-3E0h]
  char *v106; // [rsp+80h] [rbp-3E0h]
  unsigned __int8 *v107; // [rsp+88h] [rbp-3D8h]
  unsigned __int64 v108; // [rsp+88h] [rbp-3D8h]
  unsigned __int64 v109; // [rsp+88h] [rbp-3D8h]
  __int64 v110; // [rsp+88h] [rbp-3D8h]
  char *v111; // [rsp+88h] [rbp-3D8h]
  unsigned __int8 *v112; // [rsp+A0h] [rbp-3C0h]
  __int64 v113; // [rsp+A0h] [rbp-3C0h]
  __int64 v114; // [rsp+A8h] [rbp-3B8h]
  unsigned __int8 *v115; // [rsp+B0h] [rbp-3B0h]
  __int64 v117; // [rsp+B8h] [rbp-3A8h]
  __int64 *v118; // [rsp+B8h] [rbp-3A8h]
  __m128i v119; // [rsp+C0h] [rbp-3A0h] BYREF
  __m128i v120; // [rsp+D0h] [rbp-390h] BYREF
  unsigned __int8 *v121; // [rsp+E8h] [rbp-378h] BYREF
  __int64 v122; // [rsp+F0h] [rbp-370h] BYREF
  _BYTE *v123; // [rsp+F8h] [rbp-368h] BYREF
  unsigned __int64 v124; // [rsp+100h] [rbp-360h] BYREF
  unsigned int v125; // [rsp+108h] [rbp-358h]
  unsigned __int64 v126; // [rsp+110h] [rbp-350h] BYREF
  unsigned __int32 v127; // [rsp+118h] [rbp-348h]
  __m128i v128; // [rsp+120h] [rbp-340h] BYREF
  __int64 v129; // [rsp+130h] [rbp-330h]
  __int64 v130; // [rsp+138h] [rbp-328h]
  __int64 v131; // [rsp+140h] [rbp-320h]
  __int64 v132; // [rsp+148h] [rbp-318h]
  __m128i v133; // [rsp+150h] [rbp-310h] BYREF
  __m128i v134; // [rsp+160h] [rbp-300h]
  __int64 v135; // [rsp+170h] [rbp-2F0h]
  __int64 v136; // [rsp+178h] [rbp-2E8h]
  char v137; // [rsp+180h] [rbp-2E0h]
  __int64 *v138; // [rsp+190h] [rbp-2D0h] BYREF
  _QWORD *v139; // [rsp+198h] [rbp-2C8h]
  __int64 v140; // [rsp+1A0h] [rbp-2C0h] BYREF
  __int64 v141; // [rsp+1A8h] [rbp-2B8h] BYREF
  unsigned int v142; // [rsp+1B0h] [rbp-2B0h]
  _QWORD v143[2]; // [rsp+2E8h] [rbp-178h] BYREF
  char v144; // [rsp+2F8h] [rbp-168h]
  _BYTE *v145; // [rsp+300h] [rbp-160h]
  __int64 v146; // [rsp+308h] [rbp-158h]
  _BYTE v147[128]; // [rsp+310h] [rbp-150h] BYREF
  __int16 v148; // [rsp+390h] [rbp-D0h]
  _QWORD v149[2]; // [rsp+398h] [rbp-C8h] BYREF
  __int64 v150; // [rsp+3A8h] [rbp-B8h]
  __int64 v151; // [rsp+3B0h] [rbp-B0h] BYREF
  unsigned int v152; // [rsp+3B8h] [rbp-A8h]
  char v153; // [rsp+430h] [rbp-30h] BYREF

  v139 = &v123;
  v140 = (__int64)&v121;
  v5 = *(_QWORD *)(a1 + 16);
  v121 = 0;
  v138 = &v122;
  LOBYTE(v141) = 0;
  if ( !v5 )
    return 0;
  if ( !*(_QWORD *)(v5 + 8) && *(_BYTE *)a1 == 58 )
  {
    v30 = *(_QWORD *)(a1 - 32);
    if ( *(_QWORD *)(a1 - 64) )
    {
      v122 = *(_QWORD *)(a1 - 64);
      v31 = *(_QWORD *)(v30 + 16);
      v32 = &v122;
      if ( !v31 )
        goto LABEL_45;
      if ( *(_QWORD *)(v31 + 8) )
        goto LABEL_45;
      if ( *(_BYTE *)v30 != 54 )
        goto LABEL_45;
      v54 = *(_QWORD *)(v30 - 64);
      v55 = *(_QWORD *)(v54 + 16);
      if ( !v55 )
        goto LABEL_45;
      if ( *(_QWORD *)(v55 + 8) )
        goto LABEL_45;
      if ( *(_BYTE *)v54 != 68 )
        goto LABEL_45;
      v56 = *(_QWORD *)(v54 - 32);
      v57 = *(_QWORD *)(v56 + 16);
      if ( !v57 || *(_QWORD *)(v57 + 8) || *(_BYTE *)v56 <= 0x1Cu )
        goto LABEL_45;
      v123 = (_BYTE *)v56;
      if ( (unsigned __int8)sub_991580((__int64)&v140, *(_QWORD *)(v30 - 32)) )
      {
LABEL_121:
        v10 = v122;
        goto LABEL_15;
      }
      v30 = *(_QWORD *)(a1 - 32);
    }
    if ( !v30 )
    {
LABEL_46:
      v5 = *(_QWORD *)(a1 + 16);
      if ( !v5 )
        return 0;
      goto LABEL_5;
    }
    v32 = v138;
LABEL_45:
    *v32 = v30;
    v33 = *(_QWORD *)(a1 - 64);
    v34 = *(_QWORD *)(v33 + 16);
    if ( v34 )
    {
      if ( !*(_QWORD *)(v34 + 8) && *(_BYTE *)v33 == 54 )
      {
        v59 = *(_QWORD *)(v33 - 64);
        v60 = *(_QWORD *)(v59 + 16);
        if ( v60 )
        {
          if ( !*(_QWORD *)(v60 + 8) && *(_BYTE *)v59 == 68 )
          {
            v61 = *(_QWORD *)(v59 - 32);
            v62 = *(_QWORD *)(v61 + 16);
            if ( v62 )
            {
              if ( !*(_QWORD *)(v62 + 8) && *(_BYTE *)v61 > 0x1Cu )
              {
                *v139 = v61;
                if ( (unsigned __int8)sub_991580((__int64)&v140, *(_QWORD *)(v33 - 32)) )
                  goto LABEL_121;
              }
            }
          }
        }
      }
    }
    goto LABEL_46;
  }
LABEL_5:
  if ( *(_QWORD *)(v5 + 8) )
    return 0;
  if ( *(_BYTE *)a1 != 58 )
    return 0;
  v10 = *(_QWORD *)(a1 - 64);
  if ( !v10 )
    return 0;
  v11 = *(_QWORD *)(a1 - 32);
  v122 = *(_QWORD *)(a1 - 64);
  v12 = *(_QWORD *)(v11 + 16);
  if ( !v12 )
    return 0;
  if ( *(_QWORD *)(v12 + 8) )
    return 0;
  if ( *(_BYTE *)v11 != 68 )
    return 0;
  v13 = *(_QWORD *)(v11 - 32);
  v14 = *(_QWORD *)(v13 + 16);
  if ( !v14 || *(_QWORD *)(v14 + 8) || *(_BYTE *)v13 <= 0x1Cu )
    return 0;
  v123 = (_BYTE *)v13;
LABEL_15:
  if ( !(unsigned __int8)sub_23D1580(v10, a2, a3, a4) )
  {
    if ( !a2[1].m128i_i8[0] )
    {
      v15 = a2->m128i_i64[0];
      v115 = (unsigned __int8 *)a2[2].m128i_i64[0];
      goto LABEL_18;
    }
    return 0;
  }
  v15 = a2->m128i_i64[0];
  v115 = (unsigned __int8 *)a2[2].m128i_i64[0];
  if ( a2[1].m128i_i8[0] )
    goto LABEL_19;
LABEL_18:
  v16 = *(_QWORD *)(v122 + 16);
  if ( !v16 || *(_QWORD *)(v16 + 8) )
    goto LABEL_19;
  if ( *(_BYTE *)v122 == 68 )
  {
    v39 = **(_BYTE **)(v122 - 32);
    if ( v39 <= 0x1Cu )
      goto LABEL_19;
    v15 = *(_QWORD *)(v122 - 32);
  }
  else
  {
    if ( *(_BYTE *)v122 != 54 )
      goto LABEL_19;
    v35 = *(_QWORD *)(v122 - 64);
    v36 = *(_QWORD *)(v35 + 16);
    if ( !v36 )
      goto LABEL_19;
    if ( *(_QWORD *)(v36 + 8) )
      goto LABEL_19;
    if ( *(_BYTE *)v35 != 68 )
      goto LABEL_19;
    v37 = *(_QWORD *)(v35 - 32);
    v38 = *(_QWORD *)(v37 + 16);
    if ( !v38 )
      goto LABEL_19;
    if ( *(_QWORD *)(v38 + 8) )
      goto LABEL_19;
    v39 = *(_BYTE *)v37;
    if ( *(_BYTE *)v37 <= 0x1Cu )
      goto LABEL_19;
    v40 = *(unsigned __int8 **)(v122 - 32);
    v41 = *v40;
    if ( (_BYTE)v41 == 17 )
    {
      v15 = v37;
      v115 = v40 + 24;
    }
    else
    {
      if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v40 + 1) + 8LL) - 17 > 1 )
        goto LABEL_19;
      if ( (unsigned __int8)v41 > 0x15u )
        goto LABEL_19;
      v80 = sub_AD7630((__int64)v40, 0, v41);
      if ( !v80 || *v80 != 17 )
        goto LABEL_19;
      v15 = v37;
      v115 = v80 + 24;
      v39 = *(_BYTE *)v37;
    }
  }
  if ( v39 != 61 )
    return 0;
LABEL_19:
  v17 = (__int64)v123;
  if ( v123 == (_BYTE *)v15 || v15 == 0 )
    return 0;
  if ( *v123 != 61 )
    return 0;
  v114 = v15;
  if ( sub_B46500((unsigned __int8 *)v15) )
    return 0;
  if ( (*(_BYTE *)(v15 + 2) & 1) != 0 )
    return 0;
  if ( sub_B46500((unsigned __int8 *)v17) )
    return 0;
  v6 = *(_BYTE *)(v17 + 2) & 1;
  if ( (*(_BYTE *)(v17 + 2) & 1) != 0 )
    return 0;
  v18 = *(_QWORD *)(*(_QWORD *)(v15 - 32) + 8LL);
  v19 = v18;
  if ( (unsigned int)*(unsigned __int8 *)(v18 + 8) - 17 <= 1 )
    v19 = **(_QWORD **)(v18 + 16);
  v20 = *(_QWORD *)(*(_QWORD *)(v17 - 32) + 8LL);
  v21 = *(_DWORD *)(v19 + 8) >> 8;
  if ( (unsigned int)*(unsigned __int8 *)(v20 + 8) - 17 <= 1 )
    v20 = **(_QWORD **)(v20 + 16);
  v112 = *(unsigned __int8 **)(v15 - 32);
  if ( *(_DWORD *)(v20 + 8) >> 8 != v21 || *(_QWORD *)(v15 + 40) != *(_QWORD *)(v17 + 40) )
    return 0;
  v98 = *a3;
  v22 = sub_AE43F0((__int64)a3, v18);
  v23 = v112;
  v125 = v22;
  if ( v22 > 0x40 )
  {
    sub_C43690((__int64)&v124, 0, 0);
    v23 = v112;
  }
  else
  {
    v124 = 0;
  }
  v102 = sub_BD45C0(v23, (__int64)a3, (__int64)&v124, 1, 0, 0, 0, 0);
  v113 = *(_QWORD *)(v17 - 32);
  v24 = sub_AE43F0((__int64)a3, *(_QWORD *)(v113 + 8));
  v25 = (unsigned __int8 *)v113;
  v127 = v24;
  if ( v24 > 0x40 )
  {
    sub_C43690((__int64)&v126, 0, 0);
    v25 = (unsigned __int8 *)v113;
  }
  else
  {
    v126 = 0;
  }
  v107 = sub_BD45C0(v25, (__int64)a3, (__int64)&v126, 1, 0, 0, 0, 0);
  v26 = sub_BCAE30(*(_QWORD *)(v15 + 8));
  v139 = v27;
  v138 = (__int64 *)v26;
  v101 = sub_CA1930(&v138);
  v28 = sub_BCAE30(*(_QWORD *)(v17 + 8));
  v139 = v29;
  v138 = (__int64 *)v28;
  v100 = sub_CA1930(&v138);
  if ( v107 == v102 )
  {
    v103 = *(_QWORD *)(v15 + 8);
    v138 = (__int64 *)sub_9208B0((__int64)a3, v103);
    v139 = v42;
    v108 = ((unsigned __int64)v138 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    v95 = (char)v42;
    v138 = (__int64 *)sub_9208B0((__int64)a3, v103);
    v139 = v43;
    if ( v138 == (__int64 *)v108 && (_BYTE)v139 == v95 )
    {
      v104 = *(_QWORD *)(v17 + 8);
      v138 = (__int64 *)sub_9208B0((__int64)a3, v104);
      v139 = v44;
      v109 = ((unsigned __int64)v138 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      v96 = (char)v44;
      v138 = (__int64 *)sub_9208B0((__int64)a3, v104);
      v139 = v45;
      if ( v138 == (__int64 *)v109 && (_BYTE)v139 == v96 )
      {
        if ( a2[1].m128i_i8[0] )
        {
          v89 = a2->m128i_i64[1];
          v46 = v89;
        }
        else
        {
          v89 = v15;
          v46 = v15;
        }
        if ( sub_B445A0(v46, v17) )
        {
          sub_D665A0(&v128, v17);
          v97 = v128.m128i_u64[1];
          v91 = v128.m128i_i64[0];
          v93 = v129;
          v92 = v130;
          v94 = v131;
          v90 = v132;
          v47 = v17;
        }
        else
        {
          sub_D665A0(&v128, v89);
          v91 = v128.m128i_i64[0];
          v93 = v129;
          v92 = v130;
          v94 = v131;
          v90 = v132;
          if ( a2[1].m128i_i8[0] )
          {
            v97 = a2[1].m128i_u64[1];
            v47 = v89;
            v89 = v17;
            if ( v97 > 0x3FFFFFFFFFFFFFFBLL )
              v97 = 0xBFFFFFFFFFFFFFFELL;
          }
          else
          {
            v47 = v89;
            v89 = v17;
            v97 = v128.m128i_u64[1];
          }
        }
        v110 = v47 + 24;
        if ( v47 + 24 != v89 + 24 )
        {
          v87 = a2;
          v48 = 0;
          v86 = a3;
          v49 = v89 + 24;
          v88 = v15;
          do
          {
            v50 = 0;
            if ( v49 )
              v50 = v49 - 24;
            v51 = v50;
            if ( (unsigned __int8)sub_B46490(v50) )
            {
              v137 = 1;
              v139 = 0;
              v133.m128i_i64[0] = v91;
              v140 = 1;
              v133.m128i_i64[1] = v97;
              v134.m128i_i64[0] = v93;
              v134.m128i_i64[1] = v92;
              v135 = v94;
              v136 = v90;
              v138 = a4;
              v52 = &v141;
              do
              {
                *v52 = -4;
                v52 += 5;
                *(v52 - 4) = -3;
                *(v52 - 3) = -4;
                *(v52 - 2) = -3;
              }
              while ( v52 != v143 );
              v143[1] = 0;
              v146 = 0x400000000LL;
              v148 = 256;
              v143[0] = v149;
              v144 = 0;
              v145 = v147;
              v149[1] = 0;
              v150 = 1;
              v149[0] = &unk_49DDBE8;
              v53 = &v151;
              do
              {
                *v53 = -4096;
                v53 += 2;
              }
              while ( v53 != (__int64 *)&v153 );
              v105 = sub_CF63E0(a4, (unsigned __int8 *)v51, &v133, (__int64)&v138);
              v149[0] = &unk_49DDBE8;
              if ( (v150 & 1) == 0 )
                sub_C7D6A0(v151, 16LL * v152, 8);
              nullsub_184();
              if ( v145 != v147 )
                _libc_free((unsigned __int64)v145);
              if ( (v140 & 1) == 0 )
                sub_C7D6A0(v141, 40LL * v142, 8);
              if ( (v105 & 2) != 0 )
                goto LABEL_93;
            }
            if ( (*(_BYTE *)v51 != 85
               || (v58 = *(_QWORD *)(v51 - 32)) == 0
               || *(_BYTE *)v58
               || *(_QWORD *)(v58 + 24) != *(_QWORD *)(v51 + 80)
               || (*(_BYTE *)(v58 + 33) & 0x20) == 0
               || (unsigned int)(*(_DWORD *)(v58 + 36) - 68) > 3)
              && ++v48 > (unsigned int)qword_4FDF6C8 )
            {
LABEL_93:
              v6 = (unsigned __int8)v6;
              goto LABEL_36;
            }
            v49 = *(_QWORD *)(v49 + 8);
          }
          while ( v110 != v49 );
          v15 = v88;
          v6 = (unsigned __int8)v6;
          a2 = v87;
          a3 = v86;
        }
        if ( (int)sub_C4C880((__int64)&v126, (__int64)&v124) < 0 )
        {
          v114 = v17;
          v81 = v17;
          v17 = v15;
          v15 = v81;
          v82 = v126;
          v63 = v115;
          v83 = v125;
          v126 = v124;
          v124 = v82;
          v84 = v100;
          v115 = v121;
          v121 = v63;
          v125 = v127;
          v100 = v101;
          v64 = 1;
          v127 = v83;
          v101 = v84;
        }
        else
        {
          v63 = v121;
          v64 = 0;
        }
        if ( v98 )
        {
          v65 = v63;
          v63 = v115;
          v115 = v65;
          v121 = v63;
        }
        v66 = 0;
        if ( v115 )
        {
          v66 = *(char **)v115;
          if ( *((_DWORD *)v115 + 2) > 0x40u )
            v66 = **(char ***)v115;
        }
        v67 = 0;
        if ( v63 )
        {
          v67 = *(char **)v63;
          if ( *((_DWORD *)v63 + 2) > 0x40u )
            v67 = **(char ***)v63;
        }
        if ( a2[1].m128i_i8[0] )
        {
          v68 = a2[1].m128i_i64[1];
          v69 = v101;
          if ( !v64 )
          {
            v69 = a2[1].m128i_i64[1];
            v68 = v100;
          }
          v101 = v69;
          v100 = v68;
        }
        v70 = v101;
        v106 = v67;
        if ( v98 )
          v70 = v100;
        v111 = v66;
        v117 = v70;
        v71 = (_QWORD *)sub_BD5C60(v114);
        v72 = sub_BCCE00(v71, v101);
        v73 = sub_9208B0((__int64)a3, v72);
        v139 = v74;
        v138 = (__int64 *)((unsigned __int64)(v73 + 7) >> 3);
        v99 = sub_CA1930(&v138);
        if ( v106 - v111 == v117 )
        {
          v133.m128i_i32[2] = v127;
          if ( v127 > 0x40 )
            sub_C43780((__int64)&v133, (const void **)&v126);
          else
            v133.m128i_i64[0] = v126;
          sub_C46B40((__int64)&v133, (__int64 *)&v124);
          v75 = v133.m128i_u32[2];
          v76 = v133.m128i_i64[0];
          v133.m128i_i32[2] = 0;
          LODWORD(v139) = v75;
          v138 = (__int64 *)v133.m128i_i64[0];
          if ( v75 > 0x40 )
          {
            v118 = (__int64 *)v133.m128i_i64[0];
            if ( v75 - (unsigned int)sub_C444A0((__int64)&v138) > 0x40 )
            {
LABEL_148:
              sub_969240((__int64 *)&v138);
              sub_969240(v133.m128i_i64);
              goto LABEL_36;
            }
            v76 = *v118;
          }
          if ( v99 == v76 )
          {
            sub_969240((__int64 *)&v138);
            sub_969240(v133.m128i_i64);
            v77 = _mm_loadu_si128(a2 + 4);
            v133 = _mm_loadu_si128(a2 + 3);
            v134 = v77;
            sub_B91FC0((__int64 *)&v138, v17);
            if ( !a2[1].m128i_i8[0] )
            {
              a2[1].m128i_i8[0] = 1;
              sub_B91FC0(v119.m128i_i64, v114);
              v85 = _mm_loadu_si128(&v120);
              v133 = _mm_loadu_si128(&v119);
              v134 = v85;
            }
            a2[1].m128i_i64[1] = v100 + v101;
            v6 = 1;
            a2->m128i_i64[1] = v89;
            sub_E00020(&v119, (__int64)&v133, (__int64)&v138);
            a2[3] = _mm_loadu_si128(&v119);
            v78 = _mm_loadu_si128(&v120);
            a2[2].m128i_i64[0] = (__int64)v115;
            v79 = v122;
            a2->m128i_i64[0] = v15;
            a2[4] = v78;
            a2[2].m128i_i64[1] = *(_QWORD *)(v79 + 8);
            goto LABEL_36;
          }
          goto LABEL_148;
        }
      }
    }
  }
LABEL_36:
  if ( v127 > 0x40 && v126 )
    j_j___libc_free_0_0(v126);
  if ( v125 > 0x40 && v124 )
    j_j___libc_free_0_0(v124);
  return v6;
}
