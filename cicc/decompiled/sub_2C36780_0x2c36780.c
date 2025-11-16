// Function: sub_2C36780
// Address: 0x2c36780
//
void __fastcall sub_2C36780(__int64 *a1, _QWORD *a2)
{
  __int64 v3; // rsi
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // r12
  bool v7; // zf
  __int64 v8; // r8
  __int64 v9; // r9
  int v10; // eax
  int i; // r14d
  bool v12; // r15
  __int64 v13; // rdx
  bool v14; // r15
  __int64 v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rdx
  int v19; // r15d
  __int64 v20; // rsi
  __int64 v21; // rsi
  int v22; // r14d
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  int v26; // eax
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rsi
  __int64 v33; // rax
  __int64 v34; // rcx
  __int64 *v35; // rdx
  __int64 v36; // r8
  __int64 v37; // r10
  __int64 v38; // r14
  __int64 v39; // rsi
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r8
  __int64 v43; // r9
  __int64 v44; // rdx
  _QWORD *v45; // r15
  __int64 v46; // rdi
  unsigned int v47; // r14d
  unsigned int v48; // eax
  __m128i v49; // xmm1
  __m128i v50; // xmm2
  __m128i v51; // xmm3
  _QWORD *v52; // rax
  _QWORD *v53; // rax
  __int64 *v54; // rax
  __int64 v55; // rdx
  __int64 v56; // rsi
  __int64 *v57; // rcx
  _QWORD *v58; // rax
  __int64 *v59; // rax
  __int64 v60; // r15
  __int64 v61; // r14
  int v62; // r14d
  __int64 v63; // rax
  char v64; // dl
  _QWORD *v65; // rax
  __int64 v66; // r8
  __int64 v67; // r9
  __int64 v68; // rcx
  __int64 v69; // r14
  __int64 v70; // rax
  __int64 v71; // r8
  __int64 v72; // r9
  _QWORD *v73; // rdi
  __int64 v74; // rax
  _QWORD *v75; // rax
  _QWORD *v76; // rax
  __int64 v77; // [rsp+10h] [rbp-2F0h]
  __int64 v78; // [rsp+18h] [rbp-2E8h]
  __int64 v79; // [rsp+20h] [rbp-2E0h]
  __int64 v80; // [rsp+28h] [rbp-2D8h]
  __int64 v81; // [rsp+30h] [rbp-2D0h]
  __int64 v82; // [rsp+38h] [rbp-2C8h]
  __int64 *v83; // [rsp+38h] [rbp-2C8h]
  __int64 v84; // [rsp+40h] [rbp-2C0h]
  __int64 v85; // [rsp+40h] [rbp-2C0h]
  _QWORD *v86; // [rsp+40h] [rbp-2C0h]
  __int64 v87; // [rsp+40h] [rbp-2C0h]
  __int64 v88; // [rsp+48h] [rbp-2B8h]
  _QWORD *v89; // [rsp+48h] [rbp-2B8h]
  __int64 v90; // [rsp+50h] [rbp-2B0h]
  __int64 v91; // [rsp+58h] [rbp-2A8h]
  _QWORD *v92; // [rsp+58h] [rbp-2A8h]
  _QWORD *v93; // [rsp+58h] [rbp-2A8h]
  __int64 *v94; // [rsp+60h] [rbp-2A0h]
  char v95; // [rsp+60h] [rbp-2A0h]
  __int64 v96; // [rsp+60h] [rbp-2A0h]
  __int64 v97; // [rsp+60h] [rbp-2A0h]
  __int64 v98; // [rsp+68h] [rbp-298h]
  __int64 v99; // [rsp+70h] [rbp-290h] BYREF
  __int64 v100; // [rsp+78h] [rbp-288h] BYREF
  __int64 v101; // [rsp+80h] [rbp-280h] BYREF
  __int64 v102; // [rsp+88h] [rbp-278h] BYREF
  __int64 v103; // [rsp+90h] [rbp-270h] BYREF
  __int64 *v104; // [rsp+98h] [rbp-268h] BYREF
  __int64 v105; // [rsp+A0h] [rbp-260h] BYREF
  int v106; // [rsp+A8h] [rbp-258h]
  __int64 v107; // [rsp+B0h] [rbp-250h] BYREF
  int v108; // [rsp+B8h] [rbp-248h]
  __int64 v109; // [rsp+C0h] [rbp-240h] BYREF
  int v110; // [rsp+C8h] [rbp-238h]
  __int64 v111; // [rsp+D0h] [rbp-230h] BYREF
  __int64 v112; // [rsp+D8h] [rbp-228h]
  __int64 v113; // [rsp+E0h] [rbp-220h]
  __int64 v114; // [rsp+E8h] [rbp-218h]
  _QWORD *v115; // [rsp+F0h] [rbp-210h]
  __int64 v116; // [rsp+F8h] [rbp-208h]
  __m128i v117; // [rsp+100h] [rbp-200h] BYREF
  __int64 v118[2]; // [rsp+110h] [rbp-1F0h] BYREF
  __int64 *v119; // [rsp+120h] [rbp-1E0h]
  __int64 v120; // [rsp+128h] [rbp-1D8h]
  __m128i v121; // [rsp+130h] [rbp-1D0h] BYREF
  __int64 v122[2]; // [rsp+140h] [rbp-1C0h] BYREF
  __int64 *v123; // [rsp+150h] [rbp-1B0h]
  __int64 v124; // [rsp+158h] [rbp-1A8h]
  __m128i v125; // [rsp+160h] [rbp-1A0h] BYREF
  __int64 v126[2]; // [rsp+170h] [rbp-190h] BYREF
  __int64 v127; // [rsp+180h] [rbp-180h]
  __int64 v128; // [rsp+188h] [rbp-178h]
  __m128i v129; // [rsp+190h] [rbp-170h] BYREF
  __int64 *v130; // [rsp+1A0h] [rbp-160h] BYREF
  __int64 *v131; // [rsp+1A8h] [rbp-158h] BYREF
  __int64 *v132; // [rsp+1B0h] [rbp-150h]
  __int64 v133; // [rsp+1B8h] [rbp-148h]
  __int64 *v134; // [rsp+1C0h] [rbp-140h] BYREF
  __int64 v135; // [rsp+1C8h] [rbp-138h]
  _BYTE v136[64]; // [rsp+1D0h] [rbp-130h] BYREF
  _QWORD v137[6]; // [rsp+210h] [rbp-F0h] BYREF
  __int64 v138; // [rsp+240h] [rbp-C0h]
  __m128i v139; // [rsp+270h] [rbp-90h] BYREF
  __int64 *v140; // [rsp+280h] [rbp-80h] BYREF
  __int64 *v141; // [rsp+288h] [rbp-78h] BYREF
  _QWORD v142[2]; // [rsp+290h] [rbp-70h] BYREF
  __m128i v143; // [rsp+2A0h] [rbp-60h]
  __int64 v144[10]; // [rsp+2B0h] [rbp-50h] BYREF

  v3 = *a1;
  v134 = (__int64 *)v136;
  v135 = 0x800000000LL;
  sub_2C363F0((__int64)&v134, v3);
  v4 = *a2;
  v115 = a2;
  v111 = 0;
  v116 = v4;
  v112 = 0;
  v130 = v134;
  v129.m128i_i64[0] = (__int64)&v134[(unsigned int)v135];
  v113 = 0;
  v114 = 0;
  v129.m128i_i8[9] = 1;
  BYTE1(v131) = 1;
  sub_2C26110((__int64)&v139, v129.m128i_i64);
  sub_2C25F40((__int64)v137, v139.m128i_i64);
  v78 = v137[2];
  v79 = v137[0];
  v77 = v138;
  if ( v138 == v137[0] )
    goto LABEL_53;
  do
  {
    v5 = *(_QWORD *)(v79 - 8);
    v90 = v5 + 112;
    v98 = *(_QWORD *)(v5 + 120);
    if ( v5 + 112 == v98 )
      goto LABEL_48;
    do
    {
      v6 = v98;
      v7 = *(_BYTE *)(v98 - 16) == 25;
      v98 = *(_QWORD *)(v98 + 8);
      v94 = (__int64 *)(v6 - 24);
      if ( !v7 )
      {
        v139.m128i_i64[0] = (__int64)&v99;
        v139.m128i_i64[1] = (__int64)&v99;
        if ( !(unsigned __int8)sub_2C2BD50((__int64)&v139, (__int64)v94) )
        {
LABEL_64:
          v125.m128i_i32[2] = 64;
          v125.m128i_i64[0] = 0;
          sub_9865C0((__int64)&v139, (__int64)&v125);
          v141 = &v102;
          v140 = &v103;
          sub_9865C0((__int64)&v129, (__int64)&v139);
          v130 = v140;
          v131 = v141;
          sub_969240(v139.m128i_i64);
          v121.m128i_i64[0] = (__int64)&v103;
          v121.m128i_i64[1] = (__int64)&v102;
          sub_9865C0((__int64)v122, (__int64)&v129);
          v123 = v130;
          v124 = (__int64)v131;
          sub_969240(v129.m128i_i64);
          sub_969240(v125.m128i_i64);
          v125.m128i_i32[2] = 64;
          v125.m128i_i64[0] = 0;
          sub_9865C0((__int64)&v139, (__int64)&v125);
          v141 = &v100;
          v140 = &v101;
          sub_9865C0((__int64)&v129, (__int64)&v139);
          v130 = v140;
          v131 = v141;
          sub_969240(v139.m128i_i64);
          v117.m128i_i64[1] = (__int64)&v100;
          v117.m128i_i64[0] = (__int64)&v101;
          sub_9865C0((__int64)v118, (__int64)&v129);
          v119 = v130;
          v120 = (__int64)v131;
          sub_969240(v129.m128i_i64);
          sub_969240(v125.m128i_i64);
          v129 = _mm_loadu_si128(&v121);
          sub_9865C0((__int64)&v130, (__int64)v122);
          v49 = _mm_loadu_si128(&v117);
          v132 = v123;
          v125 = v49;
          v133 = v124;
          sub_9865C0((__int64)v126, (__int64)v118);
          v50 = _mm_loadu_si128(&v129);
          v127 = (__int64)v119;
          v139 = v50;
          v128 = v120;
          sub_9865C0((__int64)&v140, (__int64)&v130);
          v51 = _mm_loadu_si128(&v125);
          v142[0] = v132;
          v143 = v51;
          v142[1] = v133;
          sub_9865C0((__int64)v144, (__int64)v126);
          v144[2] = v127;
          v144[3] = v128;
          sub_969240(v126);
          sub_969240((__int64 *)&v130);
          if ( (unsigned __int8)sub_2C2C3F0(&v139, (__int64)v94) && v100 == v102 && v101 == v103 )
          {
            sub_969240(v144);
            sub_969240((__int64 *)&v140);
            sub_969240(v118);
            sub_969240(v122);
            v58 = sub_2C2A370((_QWORD *)(v6 - 8), 0);
            sub_2BF1250((__int64)v58, v100);
            sub_2C19E60(v94);
            continue;
          }
          sub_969240(v144);
          sub_969240((__int64 *)&v140);
          sub_969240(v118);
          sub_969240(v122);
          v125.m128i_i32[2] = 64;
          v125.m128i_i64[0] = 1;
          sub_9865C0((__int64)&v129, (__int64)&v125);
          sub_9865C0((__int64)&v139, (__int64)&v129);
          v140 = &v99;
          sub_969240(v129.m128i_i64);
          v95 = sub_2C2C7C0((__int64)&v139, (__int64)v94);
          sub_969240(v139.m128i_i64);
          sub_969240(v125.m128i_i64);
          if ( v95
            || *(_BYTE *)(v6 - 16) == 4
            && *(_BYTE *)(v6 + 136) == 70
            && (v104 = &v99, (unsigned __int8)sub_2C2BCF0(&v104, **(_QWORD **)(v6 + 24))) )
          {
            v53 = sub_2C2A370((_QWORD *)(v6 - 8), 0);
            sub_2BF1250((__int64)v53, v99);
            continue;
          }
          v117.m128i_i32[2] = 64;
          v117.m128i_i64[0] = 1;
          v110 = 64;
          v109 = 0;
          sub_9865C0((__int64)&v139, (__int64)&v117);
          v140 = &v99;
          sub_9865C0((__int64)&v141, (__int64)&v109);
          sub_9865C0((__int64)&v129, (__int64)&v139);
          v130 = v140;
          sub_9865C0((__int64)&v131, (__int64)&v141);
          sub_969240((__int64 *)&v141);
          sub_969240(v139.m128i_i64);
          if ( *(_BYTE *)(v6 - 16) == 1 )
          {
            sub_9865C0((__int64)&v125, (__int64)&v131);
            if ( sub_2C2C640((__int64)&v125, **(_QWORD **)(v6 + 24))
              && (v74 = *(_QWORD *)(*(_QWORD *)(v6 + 24) + 8LL)) != 0 )
            {
              *v130 = v74;
              sub_9865C0((__int64)&v139, (__int64)&v129);
              if ( sub_2C2C640((__int64)&v139, *(_QWORD *)(*(_QWORD *)(v6 + 24) + 16LL)) )
              {
                sub_969240(v139.m128i_i64);
                sub_969240(v125.m128i_i64);
                v96 = sub_2BFD6A0((__int64)&v111, *(_QWORD *)(*(_QWORD *)(v6 + 24) + 8LL));
                v75 = sub_2C2A370((_QWORD *)(v6 - 8), 0);
                v95 = v96 == sub_2BFD6A0((__int64)&v111, (__int64)v75);
LABEL_76:
                sub_969240((__int64 *)&v131);
                sub_969240(v129.m128i_i64);
                sub_969240(&v109);
                sub_969240(v117.m128i_i64);
                if ( v95 )
                {
                  v52 = sub_2C2A370((_QWORD *)(v6 - 8), 0);
                  sub_2BF1250((__int64)v52, *(_QWORD *)(*(_QWORD *)(v6 + 24) + 8LL));
                }
                continue;
              }
              sub_969240(v139.m128i_i64);
              sub_969240(v125.m128i_i64);
            }
            else
            {
              sub_969240(v125.m128i_i64);
            }
          }
          v108 = 64;
          v107 = 0;
          v106 = 64;
          v105 = 0;
          sub_9865C0((__int64)&v139, (__int64)&v107);
          sub_9865C0((__int64)&v140, (__int64)&v105);
          sub_9865C0((__int64)&v125, (__int64)&v139);
          sub_9865C0((__int64)v126, (__int64)&v140);
          sub_969240((__int64 *)&v140);
          sub_969240(v139.m128i_i64);
          if ( *(_BYTE *)(v6 - 16) == 1 )
          {
            sub_9865C0((__int64)&v121, (__int64)v126);
            v95 = sub_2C2C640((__int64)&v121, **(_QWORD **)(v6 + 24));
            if ( v95 )
            {
              sub_9865C0((__int64)&v139, (__int64)&v125);
              v95 = sub_2C2C640((__int64)&v139, *(_QWORD *)(*(_QWORD *)(v6 + 24) + 8LL));
              sub_969240(v139.m128i_i64);
              sub_969240(v121.m128i_i64);
              if ( v95 )
              {
                v97 = sub_2BFD6A0((__int64)&v111, *(_QWORD *)(*(_QWORD *)(v6 + 24) + 8LL));
                v76 = sub_2C2A370((_QWORD *)(v6 - 8), 0);
                v95 = v97 == sub_2BFD6A0((__int64)&v111, (__int64)v76);
              }
            }
            else
            {
              sub_969240(v121.m128i_i64);
            }
          }
          sub_969240(v126);
          sub_969240(v125.m128i_i64);
          sub_969240(&v105);
          sub_969240(&v107);
          goto LABEL_76;
        }
        v45 = sub_2C2A370((_QWORD *)(v6 - 8), 0);
        v91 = sub_2BFD6A0((__int64)&v111, (__int64)v45);
        v46 = sub_2BFD6A0((__int64)&v111, v99);
        if ( v91 == v46 )
        {
          sub_2BF1250((__int64)v45, v99);
          goto LABEL_64;
        }
        if ( *(_BYTE *)(v6 - 16) == 9 )
          continue;
        v47 = sub_BCB060(v46);
        v48 = sub_BCB060(v91);
        if ( v47 >= v48 )
        {
          if ( v47 <= v48 )
            goto LABEL_64;
          v70 = sub_22077B0(0xB0u);
          v69 = v70;
          if ( v70 )
          {
            sub_2C272E0(v70, 38, v99, v91, v71, v72);
            v73 = (_QWORD *)v69;
            v69 += 96;
            sub_2C19D60(v73, (__int64)v94);
            goto LABEL_106;
          }
LABEL_126:
          sub_2C19D60(0, (__int64)v94);
          goto LABEL_106;
        }
        v62 = 39;
        v63 = sub_2BF04A0(**(_QWORD **)(v6 + 24));
        if ( v63 )
        {
          v64 = *(_BYTE *)(v63 + 8);
          switch ( v64 )
          {
            case 23:
              goto LABEL_100;
            case 9:
              v62 = (**(_BYTE **)(v63 + 136) == 69) + 39;
              break;
            case 16:
LABEL_100:
              v62 = *(_DWORD *)(v63 + 160);
              if ( v62 != 40 )
                v62 = 39;
              break;
            case 4:
              v62 = (*(_BYTE *)(v63 + 160) == 40) + 39;
              break;
            default:
              v62 = 39;
              break;
          }
        }
        v65 = (_QWORD *)sub_22077B0(0xB0u);
        if ( v65 )
        {
          v68 = v91;
          v92 = v65;
          sub_2C272E0((__int64)v65, v62, v99, v68, v66, v67);
          v65 = v92;
          v69 = *(_QWORD *)(**(_QWORD **)(v6 + 24) + 40LL);
          if ( !v69 )
          {
LABEL_105:
            v93 = v65;
            sub_2C19D60(v65, (__int64)v94);
            v69 = (__int64)(v93 + 12);
LABEL_106:
            sub_2BF1250((__int64)v45, v69);
            goto LABEL_64;
          }
        }
        else
        {
          v69 = *(_QWORD *)(**(_QWORD **)(v6 + 24) + 40LL);
          if ( !v69 )
            goto LABEL_126;
        }
        v65[17] = v69;
        goto LABEL_105;
      }
      v140 = (__int64 *)4;
      v139.m128i_i64[0] = 0;
      v139.m128i_i64[1] = (__int64)v142;
      LODWORD(v141) = 0;
      BYTE4(v141) = 1;
      if ( (*(_BYTE *)(v6 + 32) & 1) == 0 )
      {
        v129.m128i_i32[2] = 64;
        v129.m128i_i64[0] = 0;
        if ( sub_2C23EC0((__int64)&v129, *(_QWORD *)(*(_QWORD *)(v6 + 24) + 8LL)) )
        {
          sub_969240(v129.m128i_i64);
          goto LABEL_6;
        }
        sub_969240(v129.m128i_i64);
      }
      sub_AE6EC0((__int64)&v139, **(_QWORD **)(v6 + 24));
LABEL_6:
      v10 = *(_DWORD *)(v6 + 32);
      if ( (unsigned int)(v10 + 1) >> 1 != 1 )
      {
        for ( i = 1; (unsigned int)(v10 + 1) >> 1 != i; ++i )
        {
          v129.m128i_i32[2] = 64;
          v129.m128i_i64[0] = 0;
          v13 = *(_QWORD *)(v6 + 24);
          if ( i )
          {
            v14 = sub_2C23EC0((__int64)&v129, *(_QWORD *)(v13 + 8LL * (2 * i + (unsigned int)((v10 & 1) == 0))));
            sub_969240(v129.m128i_i64);
            if ( !v14 )
            {
              v15 = *(_QWORD *)(*(_QWORD *)(v6 + 24) + 8LL * (2 * i - (*(_DWORD *)(v6 + 32) & 1u)));
LABEL_13:
              sub_AE6EC0((__int64)&v139, v15);
            }
          }
          else
          {
            v12 = sub_2C23EC0((__int64)&v129, *(_QWORD *)(v13 + 8));
            sub_969240(v129.m128i_i64);
            if ( !v12 )
            {
              v15 = **(_QWORD **)(v6 + 24);
              goto LABEL_13;
            }
          }
          v10 = *(_DWORD *)(v6 + 32);
        }
      }
      v16 = HIDWORD(v140);
      v17 = (unsigned int)(HIDWORD(v140) - (_DWORD)v141);
      if ( (_DWORD)v17 == 1 )
      {
        v54 = (__int64 *)v139.m128i_i64[1];
        if ( !BYTE4(v141) )
          v16 = (unsigned int)v140;
        v55 = v139.m128i_i64[1] + 8 * v16;
        v56 = *(_QWORD *)v139.m128i_i64[1];
        if ( v139.m128i_i64[1] != v55 )
        {
          do
          {
            v56 = *v54;
            v57 = v54;
            if ( (unsigned __int64)*v54 < 0xFFFFFFFFFFFFFFFELL )
              goto LABEL_92;
            ++v54;
          }
          while ( (__int64 *)v55 != v54 );
          v56 = v57[1];
        }
LABEL_92:
        sub_2BF1250(v6 + 72, v56);
        sub_2C19E60(v94);
        goto LABEL_45;
      }
      if ( (v10 & 1) != 0 )
        goto LABEL_45;
      v18 = (unsigned int)(v10 + 1) >> 1;
      if ( !((unsigned int)(v10 + 1) >> 1) )
      {
LABEL_57:
        v129.m128i_i64[0] = (__int64)&v130;
        v129.m128i_i64[1] = 0x400000000LL;
        goto LABEL_58;
      }
      v19 = 0;
      while ( 1 )
      {
        while ( !v19 )
        {
          v20 = *(_QWORD *)(*(_QWORD *)(v6 + 24) + 8LL);
          if ( *(_DWORD *)(v20 + 24) == 1 )
            goto LABEL_23;
LABEL_20:
          ++v19;
          v18 = (unsigned int)(v10 + 1) >> 1;
          if ( (_DWORD)v18 == v19 )
            goto LABEL_57;
        }
        v17 = (unsigned int)((v10 & 1) == 0) + 2 * v19;
        v20 = *(_QWORD *)(*(_QWORD *)(v6 + 24) + 8 * v17);
        if ( *(_DWORD *)(v20 + 24) != 1 )
          goto LABEL_20;
LABEL_23:
        v129.m128i_i32[2] = 64;
        v129.m128i_i64[0] = 0;
        if ( !sub_2C23EC0((__int64)&v129, v20) )
          break;
        sub_969240(v129.m128i_i64);
        v10 = *(_DWORD *)(v6 + 32);
        ++v19;
        v18 = (unsigned int)(v10 + 1) >> 1;
        if ( (_DWORD)v18 == v19 )
          goto LABEL_57;
      }
      sub_969240(v129.m128i_i64);
      v129.m128i_i64[0] = (__int64)&v130;
      v129.m128i_i64[1] = 0x400000000LL;
      if ( v19 )
      {
        v18 = *(_QWORD *)(v6 + 24);
        v21 = *(_QWORD *)(v18 + 8LL * (2 * v19 - (*(_DWORD *)(v6 + 32) & 1u)));
        goto LABEL_26;
      }
LABEL_58:
      v19 = 0;
      v21 = **(_QWORD **)(v6 + 24);
LABEL_26:
      v22 = 0;
      sub_2AB9420((__int64)&v129, v21, v18, v17, v8, v9);
      v26 = *(_DWORD *)(v6 + 32);
      v27 = (unsigned int)(v26 + 1) >> 1;
      if ( (unsigned int)(v26 + 1) >> 1 )
      {
        do
        {
          if ( v22 != v19 )
          {
            if ( v22 )
            {
              sub_2AB9420(
                (__int64)&v129,
                *(_QWORD *)(*(_QWORD *)(v6 + 24) + 8LL * (2 * v22 - (v26 & 1u))),
                v27,
                2 * v22 - (v26 & 1u),
                v24,
                v25);
              v28 = (unsigned int)((*(_DWORD *)(v6 + 32) & 1) == 0) + 2 * v22;
              v32 = *(_QWORD *)(*(_QWORD *)(v6 + 24) + 8 * v28);
            }
            else
            {
              sub_2AB9420((__int64)&v129, **(_QWORD **)(v6 + 24), v27, v23, v24, v25);
              v32 = *(_QWORD *)(*(_QWORD *)(v6 + 24) + 8LL);
            }
            sub_2AB9420((__int64)&v129, v32, v28, v29, v30, v31);
            v26 = *(_DWORD *)(v6 + 32);
          }
          ++v22;
          v27 = (unsigned int)(v26 + 1) >> 1;
        }
        while ( (_DWORD)v27 != v22 );
      }
      v82 = *(_QWORD *)(v6 + 112);
      v84 = v129.m128i_i64[0];
      v88 = v129.m128i_u32[2];
      v33 = sub_22077B0(0x98u);
      v34 = v88;
      v35 = (__int64 *)v84;
      v36 = v82;
      v37 = v33;
      if ( v33 )
      {
        v89 = (_QWORD *)v33;
        v125.m128i_i64[0] = *(_QWORD *)(v82 + 48);
        if ( v125.m128i_i64[0] )
        {
          v80 = v33;
          v81 = v34;
          v83 = (__int64 *)v84;
          v85 = v36;
          sub_2C25AB0(v125.m128i_i64);
          v37 = v80;
          v34 = v81;
          v35 = v83;
          v36 = v85;
        }
        v86 = (_QWORD *)v37;
        sub_2ABB100(v37, 25, v35, v34, v36, v125.m128i_i64);
        sub_9C6650(&v125);
        v37 = (__int64)v86;
        *v86 = &unk_4A243A0;
        v86[5] = &unk_4A243E0;
        v86[12] = &unk_4A24418;
      }
      else
      {
        v89 = 0;
      }
      v87 = v37;
      sub_2C19D60(v89, (__int64)v94);
      if ( v19 )
        v38 = *(_QWORD *)(*(_QWORD *)(v6 + 24) + 8LL * ((unsigned int)((*(_DWORD *)(v6 + 32) & 1) == 0) + 2 * v19));
      else
        v38 = *(_QWORD *)(*(_QWORD *)(v6 + 24) + 8LL);
      if ( v87 )
        v39 = v87 + 96;
      else
        v39 = 0;
      sub_2BF1250(v6 + 72, v39);
      sub_2C19E60(v94);
      sub_2C26230(v38, v39, v40, v41, v42, v43);
      if ( *(_DWORD *)(v87 + 56) == 3 )
      {
        v125.m128i_i64[0] = (__int64)&v121;
        if ( (unsigned __int8)sub_2C2BCF0(&v125, *(_QWORD *)(*(_QWORD *)(v87 + 48) + 16LL)) )
        {
          v59 = *(__int64 **)(v87 + 48);
          v60 = *v59;
          v61 = v59[2];
          sub_2AAED30(v87 + 40, 0, v59[1]);
          sub_2AAED30(v87 + 40, 1u, v60);
          sub_2AAED30(v87 + 40, 2u, v121.m128i_i64[0]);
          if ( !*(_DWORD *)(v61 + 24) )
            sub_2C19E60((__int64 *)(v61 - 96));
        }
      }
      if ( (__int64 **)v129.m128i_i64[0] != &v130 )
        _libc_free(v129.m128i_u64[0]);
LABEL_45:
      if ( !BYTE4(v141) )
        _libc_free(v139.m128i_u64[1]);
    }
    while ( v90 != v98 );
LABEL_48:
    v79 -= 8;
    if ( v78 != v79 )
    {
      v44 = v79;
      do
      {
        if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v44 - 8) + 8LL) - 1 <= 1 )
          break;
        v44 -= 8;
      }
      while ( v78 != v44 );
      v79 = v44;
    }
  }
  while ( v77 != v79 );
LABEL_53:
  sub_C7D6A0(v112, 16LL * (unsigned int)v114, 8);
  if ( v134 != (__int64 *)v136 )
    _libc_free((unsigned __int64)v134);
}
