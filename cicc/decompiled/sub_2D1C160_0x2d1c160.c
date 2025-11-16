// Function: sub_2D1C160
// Address: 0x2d1c160
//
__int64 __fastcall sub_2D1C160(_QWORD *a1, unsigned __int64 a2)
{
  _QWORD *v3; // rbx
  _QWORD *v4; // rax
  _QWORD *v5; // rsi
  __int64 v6; // rcx
  __int64 v7; // rdx
  unsigned __int64 v8; // r12
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  unsigned __int64 v11; // r13
  __int64 v12; // rdi
  __int64 v13; // rdx
  unsigned int v14; // r13d
  unsigned __int64 v15; // rdx
  __int64 v16; // rdi
  _QWORD *v17; // rax
  _QWORD *v18; // rdi
  __int64 v19; // rsi
  __int64 v20; // rcx
  _QWORD *v22; // r14
  unsigned __int8 *v23; // r13
  unsigned __int8 v24; // bl
  __int64 v25; // rax
  int v26; // eax
  unsigned __int8 **v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  unsigned __int8 **v31; // rax
  unsigned __int8 **v32; // r15
  unsigned __int8 **v33; // r12
  __int64 v34; // rax
  unsigned __int8 **v35; // rax
  _QWORD *v36; // r14
  __m128i v37; // xmm0
  __m128i v38; // xmm1
  __m128i v39; // xmm2
  __int64 **v40; // rax
  unsigned __int8 *v41; // rsi
  __int64 *v42; // rax
  unsigned __int8 **v43; // rax
  __int64 v44; // r12
  __int64 v45; // rax
  unsigned __int64 v46; // rdi
  __int64 v47; // rax
  __int64 *v48; // r13
  __int64 *v49; // rax
  __int64 *v50; // rdx
  char v51; // di
  __int64 v52; // rsi
  __int64 v53; // r14
  __int64 *v54; // rax
  unsigned __int64 v55; // r13
  __int64 *v56; // rdx
  __int64 *v57; // rsi
  unsigned __int64 v58; // rbx
  _QWORD *v59; // rax
  _QWORD *v60; // rsi
  __int64 v61; // rcx
  __int64 v62; // rdx
  __int64 *v63; // rax
  __int64 v64; // rax
  __int64 *v65; // rax
  __int64 *v66; // rdx
  __int64 *v67; // rbx
  char v68; // di
  __int64 *v69; // r9
  _QWORD *v70; // r12
  __int64 v71; // rax
  unsigned __int64 v72; // r12
  _QWORD *v73; // rax
  __int64 *v74; // rdx
  char v75; // di
  __int64 v76; // rax
  __int64 *v77; // r9
  __int64 *v78; // rax
  __int64 v79; // rax
  __int64 *v80; // rax
  __int64 *v81; // rdx
  char v82; // di
  __int64 v83; // rsi
  char v84; // dh
  char v85; // al
  char v86; // dl
  __int64 v87; // rax
  unsigned __int64 v88; // rdi
  __int64 v89; // [rsp+8h] [rbp-428h]
  int v90; // [rsp+1Ch] [rbp-414h]
  _QWORD *v91; // [rsp+28h] [rbp-408h]
  unsigned __int64 v92; // [rsp+30h] [rbp-400h]
  _QWORD *v93; // [rsp+48h] [rbp-3E8h]
  __int64 v94; // [rsp+50h] [rbp-3E0h]
  __int64 v95; // [rsp+50h] [rbp-3E0h]
  unsigned __int8 v96; // [rsp+58h] [rbp-3D8h]
  unsigned __int64 v97; // [rsp+58h] [rbp-3D8h]
  _QWORD *v98; // [rsp+58h] [rbp-3D8h]
  __int64 v99; // [rsp+58h] [rbp-3D8h]
  __int64 *v100; // [rsp+58h] [rbp-3D8h]
  _QWORD *v101; // [rsp+60h] [rbp-3D0h]
  _QWORD *v102; // [rsp+68h] [rbp-3C8h]
  char v103; // [rsp+68h] [rbp-3C8h]
  __int64 v104; // [rsp+68h] [rbp-3C8h]
  unsigned __int64 v105; // [rsp+70h] [rbp-3C0h]
  __int64 v107; // [rsp+80h] [rbp-3B0h]
  __int64 v108; // [rsp+80h] [rbp-3B0h]
  unsigned __int8 *v109; // [rsp+80h] [rbp-3B0h]
  __int64 *v110; // [rsp+80h] [rbp-3B0h]
  int v111; // [rsp+88h] [rbp-3A8h]
  unsigned __int8 *v112; // [rsp+88h] [rbp-3A8h]
  __int64 v113; // [rsp+88h] [rbp-3A8h]
  __m128i v114; // [rsp+90h] [rbp-3A0h] BYREF
  __m128i v115; // [rsp+A0h] [rbp-390h] BYREF
  __m128i v116; // [rsp+B0h] [rbp-380h] BYREF
  __m128i v117; // [rsp+C0h] [rbp-370h] BYREF
  __m128i v118; // [rsp+D0h] [rbp-360h]
  __m128i v119; // [rsp+E0h] [rbp-350h]
  char v120; // [rsp+F0h] [rbp-340h]
  __int64 v121; // [rsp+100h] [rbp-330h] BYREF
  unsigned __int8 **v122; // [rsp+108h] [rbp-328h]
  __int64 v123; // [rsp+110h] [rbp-320h]
  int v124; // [rsp+118h] [rbp-318h]
  char v125; // [rsp+11Ch] [rbp-314h]
  char v126; // [rsp+120h] [rbp-310h] BYREF
  _QWORD *v127; // [rsp+160h] [rbp-2D0h] BYREF
  __int64 v128; // [rsp+168h] [rbp-2C8h] BYREF
  unsigned __int64 v129; // [rsp+170h] [rbp-2C0h]
  __int64 *v130; // [rsp+178h] [rbp-2B8h] BYREF
  __int64 *v131; // [rsp+180h] [rbp-2B0h]
  __int64 v132; // [rsp+188h] [rbp-2A8h]
  _QWORD v133[2]; // [rsp+2B8h] [rbp-178h] BYREF
  char v134; // [rsp+2C8h] [rbp-168h]
  _BYTE *v135; // [rsp+2D0h] [rbp-160h]
  __int64 v136; // [rsp+2D8h] [rbp-158h]
  _BYTE v137[128]; // [rsp+2E0h] [rbp-150h] BYREF
  __int16 v138; // [rsp+360h] [rbp-D0h]
  _QWORD v139[2]; // [rsp+368h] [rbp-C8h] BYREF
  __int64 v140; // [rsp+378h] [rbp-B8h]
  __int64 v141; // [rsp+380h] [rbp-B0h] BYREF
  unsigned int v142; // [rsp+388h] [rbp-A8h]
  char v143; // [rsp+400h] [rbp-30h] BYREF

  v3 = a1 + 9;
  v4 = (_QWORD *)a1[10];
  if ( v4 )
  {
    v5 = a1 + 9;
    do
    {
      while ( 1 )
      {
        v6 = v4[2];
        v7 = v4[3];
        if ( v4[4] >= a2 )
          break;
        v4 = (_QWORD *)v4[3];
        if ( !v7 )
          goto LABEL_6;
      }
      v5 = v4;
      v4 = (_QWORD *)v4[2];
    }
    while ( v6 );
LABEL_6:
    if ( v3 != v5 && v5[4] <= a2 )
      return 0;
  }
  v8 = a2 + 48;
  v9 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a2 + 48 == v9 )
    return 0;
  if ( !v9 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v9 - 24) - 30 <= 0xA )
  {
    if ( (unsigned int)sub_B46E30(v9 - 24) > 1 )
      goto LABEL_12;
    return 0;
  }
  v96 = 0;
  if ( (unsigned int)sub_B46E30(0) <= 1 )
    return v96;
LABEL_12:
  v10 = (unsigned int)(*(_DWORD *)(a2 + 44) + 1);
  if ( (unsigned int)v10 >= *(_DWORD *)(*a1 + 32LL) || !*(_QWORD *)(*(_QWORD *)(*a1 + 24LL) + 8 * v10) )
    return 0;
  sub_2D1B830(a1[4]);
  a1[4] = 0;
  a1[5] = a1 + 3;
  a1[6] = a1 + 3;
  a1[7] = 0;
  v11 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v8 == v11 )
  {
    v12 = 0;
  }
  else
  {
    if ( !v11 )
      BUG();
    v12 = v11 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v11 - 24) - 30 >= 0xB )
      v12 = 0;
  }
  v107 = *(_QWORD *)(a2 + 48);
  v111 = sub_B46E30(v12);
  if ( v111 )
  {
    v13 = v107;
    v14 = 0;
    while ( 1 )
    {
      v15 = v13 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v8 == v15 )
      {
        v16 = 0;
      }
      else
      {
        if ( !v15 )
          BUG();
        v16 = v15 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v15 - 24) - 30 >= 0xB )
          v16 = 0;
      }
      v127 = (_QWORD *)sub_B46EC0(v16, v14);
      v17 = (_QWORD *)a1[10];
      if ( !v17 )
        return 0;
      v18 = v3;
      do
      {
        while ( 1 )
        {
          v19 = v17[2];
          v20 = v17[3];
          if ( v17[4] >= (unsigned __int64)v127 )
            break;
          v17 = (_QWORD *)v17[3];
          if ( !v20 )
            goto LABEL_29;
        }
        v18 = v17;
        v17 = (_QWORD *)v17[2];
      }
      while ( v19 );
LABEL_29:
      if ( v3 == v18 || v18[4] > (unsigned __int64)v127 )
        return 0;
      ++v14;
      sub_22DC880(a1 + 2, (unsigned __int64 *)&v127);
      if ( v111 == v14 )
        break;
      v13 = *(_QWORD *)(a2 + 48);
    }
    v11 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  }
  v22 = (_QWORD *)v11;
  v121 = 0;
  v122 = (unsigned __int8 **)&v126;
  v123 = 8;
  v124 = 0;
  v125 = 1;
  v90 = 0;
  v96 = 0;
  v92 = a2;
  while ( 1 )
  {
    v23 = 0;
    v105 = (unsigned __int64)v22;
    if ( v22 )
      v23 = (unsigned __int8 *)(v22 - 3);
    v101 = *(_QWORD **)(v92 + 56);
    if ( v22 != v101 )
      v105 = *v22 & 0xFFFFFFFFFFFFFFF8LL;
    v24 = *v23;
    if ( *v23 == 85 )
    {
      v34 = *((_QWORD *)v23 - 4);
      if ( v34
        && !*(_BYTE *)v34
        && *(_QWORD *)(v34 + 24) == *((_QWORD *)v23 + 10)
        && (*(_BYTE *)(v34 + 33) & 0x20) != 0
        && (unsigned int)(*(_DWORD *)(v34 + 36) - 68) <= 3 )
      {
        goto LABEL_60;
      }
      if ( (unsigned __int8)sub_B46490((__int64)v23) )
        goto LABEL_66;
    }
    else
    {
      if ( v24 == 61 )
      {
        v25 = *(_QWORD *)(*((_QWORD *)v23 - 4) + 8LL);
        if ( (unsigned int)*(unsigned __int8 *)(v25 + 8) - 17 <= 1 )
          v25 = **(_QWORD **)(v25 + 16);
        v26 = *(_DWORD *)(v25 + 8) >> 8;
        if ( v26 == 1 || v26 == 5 || !v26 )
          goto LABEL_60;
        v102 = (_QWORD *)a1[1];
        if ( (unsigned __int8)sub_B46490((__int64)v23) )
        {
LABEL_66:
          if ( !v125 )
            goto LABEL_151;
          v35 = v122;
          v28 = HIDWORD(v123);
          v27 = &v122[HIDWORD(v123)];
          if ( v122 != v27 )
          {
            do
            {
              if ( v23 == *v35 )
                goto LABEL_60;
              ++v35;
            }
            while ( v27 != v35 );
          }
          if ( HIDWORD(v123) < (unsigned int)v123 )
          {
            ++HIDWORD(v123);
            *v27 = v23;
            ++v121;
          }
          else
          {
LABEL_151:
            sub_C8CC70((__int64)&v121, (__int64)v23, (__int64)v27, v28, v29, v30);
          }
          goto LABEL_60;
        }
        sub_D665A0(&v114, (__int64)v23);
        v31 = v122;
        if ( v125 )
          v32 = &v122[HIDWORD(v123)];
        else
          v32 = &v122[(unsigned int)v123];
        if ( v122 != v32 )
        {
          while ( 1 )
          {
            v33 = v31;
            if ( (unsigned __int64)*v31 < 0xFFFFFFFFFFFFFFFELL )
              break;
            if ( v32 == ++v31 )
              goto LABEL_57;
          }
          if ( v32 != v31 )
          {
            v91 = v22;
            v36 = v102;
            do
            {
              v37 = _mm_loadu_si128(&v114);
              v38 = _mm_loadu_si128(&v115);
              v120 = 1;
              v39 = _mm_loadu_si128(&v116);
              v40 = &v130;
              v117 = v37;
              v118 = v38;
              v119 = v39;
              v41 = *v33;
              v127 = v36;
              v128 = 0;
              v129 = 1;
              do
              {
                *v40 = (__int64 *)-4LL;
                v40 += 5;
                *(v40 - 4) = (__int64 *)-3LL;
                *(v40 - 3) = (__int64 *)-4LL;
                *(v40 - 2) = (__int64 *)-3LL;
              }
              while ( v40 != v133 );
              v133[0] = v139;
              v133[1] = 0;
              v135 = v137;
              v136 = 0x400000000LL;
              v138 = 256;
              v134 = 0;
              v139[1] = 0;
              v140 = 1;
              v139[0] = &unk_49DDBE8;
              v42 = &v141;
              do
              {
                *v42 = -4096;
                v42 += 2;
              }
              while ( v42 != (__int64 *)&v143 );
              v103 = sub_CF63E0(v36, v41, &v117, (__int64)&v127);
              v139[0] = &unk_49DDBE8;
              if ( (v140 & 1) == 0 )
                sub_C7D6A0(v141, 16LL * v142, 8);
              nullsub_184();
              if ( v135 != v137 )
                _libc_free((unsigned __int64)v135);
              if ( (v129 & 1) == 0 )
                sub_C7D6A0((__int64)v130, 40LL * (unsigned int)v131, 8);
              if ( (v103 & 2) != 0 )
              {
                v22 = v91;
                goto LABEL_60;
              }
              v43 = v33 + 1;
              if ( v33 + 1 == v32 )
                break;
              while ( 1 )
              {
                v33 = v43;
                if ( (unsigned __int64)*v43 < 0xFFFFFFFFFFFFFFFELL )
                  break;
                if ( v32 == ++v43 )
                  goto LABEL_88;
              }
            }
            while ( v32 != v43 );
LABEL_88:
            v22 = v91;
          }
        }
LABEL_57:
        v24 = *v23;
      }
      else if ( (unsigned __int8)sub_B46490((__int64)v23) )
      {
        goto LABEL_66;
      }
      if ( (unsigned int)v24 - 30 <= 0xA || v24 == 84 )
        goto LABEL_60;
    }
    if ( sub_CEA920((__int64)v23) )
      goto LABEL_60;
    LODWORD(v128) = 0;
    v129 = 0;
    v130 = &v128;
    v131 = &v128;
    v132 = 0;
    v117 = 0u;
    v118.m128i_i64[0] = 0;
    v44 = *((_QWORD *)v23 + 2);
    if ( v44 )
    {
      v112 = v23;
      do
      {
        v45 = *(_QWORD *)(v44 + 24);
        if ( *(_BYTE *)v45 > 0x1Cu )
        {
          v114.m128i_i64[0] = *(_QWORD *)(v44 + 24);
          if ( *(_BYTE *)v45 == 84 )
            goto LABEL_96;
          v58 = *(_QWORD *)(v45 + 40);
          v59 = (_QWORD *)a1[4];
          if ( !v59 )
            goto LABEL_96;
          v60 = a1 + 3;
          do
          {
            while ( 1 )
            {
              v61 = v59[2];
              v62 = v59[3];
              if ( v59[4] >= v58 )
                break;
              v59 = (_QWORD *)v59[3];
              if ( !v62 )
                goto LABEL_126;
            }
            v60 = v59;
            v59 = (_QWORD *)v59[2];
          }
          while ( v61 );
LABEL_126:
          if ( a1 + 3 == v60 || v60[4] > v58 )
          {
LABEL_96:
            v46 = v117.m128i_i64[0];
LABEL_97:
            if ( v46 )
              j_j___libc_free_0(v46);
LABEL_99:
            sub_2D1B660(v129);
            goto LABEL_60;
          }
          v63 = (__int64 *)v129;
          v48 = &v128;
          if ( !v129 )
            goto LABEL_102;
          do
          {
            if ( v63[4] < v58 )
            {
              v63 = (__int64 *)v63[3];
            }
            else
            {
              v48 = v63;
              v63 = (__int64 *)v63[2];
            }
          }
          while ( v63 );
          if ( v48 == &v128 || v48[4] > v58 )
          {
LABEL_102:
            v108 = (__int64)v48;
            v47 = sub_22077B0(0x30u);
            *(_QWORD *)(v47 + 32) = v58;
            v48 = (__int64 *)v47;
            *(_QWORD *)(v47 + 40) = 0;
            v49 = sub_2D1C060(&v127, v108, (unsigned __int64 *)(v47 + 32));
            if ( v50 )
            {
              v51 = &v128 == v50 || v49 || v50[4] > v58;
              sub_220F040(v51, (__int64)v48, v50, &v128);
              ++v132;
            }
            else
            {
              v110 = v49;
              j_j___libc_free_0((unsigned __int64)v48);
              v48 = v110;
            }
          }
          v48[5] = 0;
          v52 = v117.m128i_i64[1];
          if ( v117.m128i_i64[1] == v118.m128i_i64[0] )
          {
            sub_24454E0((__int64)&v117, (_BYTE *)v117.m128i_i64[1], &v114);
          }
          else
          {
            if ( v117.m128i_i64[1] )
            {
              *(_QWORD *)v117.m128i_i64[1] = v114.m128i_i64[0];
              v52 = v117.m128i_i64[1];
            }
            v117.m128i_i64[1] = v52 + 8;
          }
        }
        v44 = *(_QWORD *)(v44 + 8);
      }
      while ( v44 );
      v23 = v112;
      v46 = v117.m128i_i64[0];
      if ( a1[7] != v132 )
        goto LABEL_97;
      v104 = v117.m128i_i64[1];
      if ( v117.m128i_i64[1] == v117.m128i_i64[0] )
        goto LABEL_146;
      v113 = v117.m128i_i64[0];
      v93 = v22;
      v109 = v23;
      while ( 2 )
      {
        v53 = *(_QWORD *)v113;
        v54 = (__int64 *)v129;
        v55 = *(_QWORD *)(*(_QWORD *)v113 + 40LL);
        if ( v129 )
        {
          v56 = (__int64 *)v129;
          v57 = &v128;
          do
          {
            if ( v56[4] < v55 )
            {
              v56 = (__int64 *)v56[3];
            }
            else
            {
              v57 = v56;
              v56 = (__int64 *)v56[2];
            }
          }
          while ( v56 );
          if ( v57 != &v128 && v57[4] <= v55 )
          {
            v70 = (_QWORD *)v57[5];
            goto LABEL_155;
          }
        }
        else
        {
          v57 = &v128;
        }
        v71 = sub_22077B0(0x30u);
        *(_QWORD *)(v71 + 32) = v55;
        v72 = v71;
        *(_QWORD *)(v71 + 40) = 0;
        v73 = sub_2D1C060(&v127, (__int64)v57, (unsigned __int64 *)(v71 + 32));
        if ( v74 )
        {
          v75 = &v128 == v74 || v73 || v55 < v74[4];
          sub_220F040(v75, v72, v74, &v128);
          ++v132;
        }
        else
        {
          v98 = v73;
          j_j___libc_free_0(v72);
          v72 = (unsigned __int64)v98;
        }
        v54 = (__int64 *)v129;
        v70 = *(_QWORD **)(v72 + 40);
        if ( !v129 )
        {
          v69 = &v128;
          goto LABEL_138;
        }
LABEL_155:
        v69 = &v128;
        do
        {
          if ( v54[4] < v55 )
          {
            v54 = (__int64 *)v54[3];
          }
          else
          {
            v69 = v54;
            v54 = (__int64 *)v54[2];
          }
        }
        while ( v54 );
        if ( v69 == &v128 || v69[4] > v55 )
        {
LABEL_138:
          v94 = (__int64)v69;
          v64 = sub_22077B0(0x30u);
          *(_QWORD *)(v64 + 32) = v55;
          *(_QWORD *)(v64 + 40) = 0;
          v97 = v64;
          v65 = sub_2D1C060(&v127, v94, (unsigned __int64 *)(v64 + 32));
          v67 = v65;
          if ( v66 )
          {
            v68 = &v128 == v66 || v65 || v55 < v66[4];
            sub_220F040(v68, v97, v66, &v128);
            ++v132;
            v69 = (__int64 *)v97;
          }
          else
          {
            j_j___libc_free_0(v97);
            v69 = v67;
          }
        }
        if ( !v69[5] )
        {
          v76 = sub_B47F80(v109);
          v77 = &v128;
          v70 = (_QWORD *)v76;
          v78 = (__int64 *)v129;
          if ( !v129 )
            goto LABEL_184;
          do
          {
            if ( v78[4] < v55 )
            {
              v78 = (__int64 *)v78[3];
            }
            else
            {
              v77 = v78;
              v78 = (__int64 *)v78[2];
            }
          }
          while ( v78 );
          if ( v77 == &v128 || v77[4] > v55 )
          {
LABEL_184:
            v95 = (__int64)v77;
            v79 = sub_22077B0(0x30u);
            *(_QWORD *)(v79 + 32) = v55;
            *(_QWORD *)(v79 + 40) = 0;
            v99 = v79;
            v80 = sub_2D1C060(&v127, v95, (unsigned __int64 *)(v79 + 32));
            if ( v81 )
            {
              v82 = v80 || &v128 == v81 || v55 < v81[4];
              sub_220F040(v82, v99, v81, &v128);
              ++v132;
              v77 = (__int64 *)v99;
            }
            else
            {
              v88 = v99;
              v100 = v80;
              j_j___libc_free_0(v88);
              v77 = v100;
            }
          }
          v77[5] = (__int64)v70;
          v83 = sub_AA4FF0(v55);
          v85 = v84;
          if ( !v83 )
            v85 = 0;
          v86 = v85;
          v87 = v89;
          LOBYTE(v87) = 1;
          BYTE1(v87) = v86;
          v89 = v87;
          sub_B44220(v70, v83, v87);
        }
        sub_BD2ED0(v53, (__int64)v109, (__int64)v70);
        v113 += 8;
        if ( v104 == v113 )
        {
          v22 = v93;
          v23 = v109;
          goto LABEL_146;
        }
        continue;
      }
    }
    if ( a1[7] )
      goto LABEL_99;
LABEL_146:
    sub_B43D60(v23);
    if ( v117.m128i_i64[0] )
      j_j___libc_free_0(v117.m128i_u64[0]);
    sub_2D1B660(v129);
    ++v90;
    v96 = 1;
    if ( v90 >= SLODWORD(qword_5016348[8]) )
      break;
LABEL_60:
    if ( v22 == v101 )
      break;
    v22 = (_QWORD *)v105;
  }
  if ( !v125 )
    _libc_free((unsigned __int64)v122);
  return v96;
}
