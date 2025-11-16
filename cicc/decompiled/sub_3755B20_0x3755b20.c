// Function: sub_3755B20
// Address: 0x3755b20
//
void __fastcall sub_3755B20(__int64 *a1, __int64 a2, char a3, unsigned __int8 a4, __m128i *a5, __int64 a6)
{
  __int64 *v6; // r15
  __int64 v7; // r13
  int v8; // eax
  __int64 v9; // r12
  _QWORD *v10; // rbx
  __int64 v11; // rsi
  __int64 v12; // r12
  __int64 v13; // r13
  __int64 *v14; // r15
  _QWORD *v15; // r14
  _QWORD *v16; // rax
  __int64 v17; // r12
  __int64 v18; // rdx
  __int64 v19; // rax
  unsigned __int8 *v20; // rsi
  _QWORD *v21; // rax
  unsigned __int64 v22; // rsi
  __int32 v23; // ebx
  int v24; // edx
  __int32 v25; // r12d
  __int64 v26; // rsi
  __int64 v27; // rax
  __int64 v28; // rcx
  __int64 v29; // rdi
  __int64 *v30; // rsi
  _QWORD *v31; // rax
  __int64 v32; // rdx
  _QWORD *v33; // r14
  unsigned int v34; // eax
  __int64 v35; // rsi
  __int64 v36; // rax
  __int64 v37; // rbx
  __int64 *v38; // r15
  _QWORD *v39; // r12
  _QWORD *v40; // rax
  __int64 v41; // r13
  __int64 v42; // rdx
  __int64 v43; // rax
  unsigned __int8 *v44; // rsi
  __int64 v45; // r12
  __int64 v46; // r12
  __int64 *v47; // rsi
  __int64 v48; // rdi
  int v49; // edi
  __int64 v50; // r12
  int v51; // ecx
  unsigned int *v52; // rdx
  __int64 v53; // rsi
  __int64 v54; // r12
  _QWORD *v55; // rbx
  _QWORD *v56; // rax
  __int64 v57; // r12
  _QWORD *v58; // rax
  __int64 v59; // rax
  bool v60; // cc
  _QWORD *v61; // rax
  __int64 v62; // r8
  __int64 v63; // r9
  __int64 v64; // rax
  __int64 v65; // rax
  _QWORD *v66; // rbx
  __int64 v67; // rax
  unsigned __int64 v68; // rdx
  int v69; // r12d
  unsigned int v70; // r14d
  unsigned int v71; // ebx
  __int64 v72; // rax
  int v73; // eax
  unsigned int v74; // r14d
  int v75; // r12d
  int v76; // ebx
  unsigned int v77; // edx
  unsigned int v78; // esi
  __int64 (*v79)(void); // rax
  unsigned int *v80; // r12
  __int64 v81; // rbx
  unsigned int v82; // r14d
  __int64 v83; // r13
  unsigned int v84; // eax
  __int64 v85; // rax
  __int64 v86; // rbx
  __int64 *v87; // r12
  __int64 v88; // rdx
  __int64 v89; // rax
  unsigned int v90; // r12d
  __int32 v91; // r15d
  __int64 v92; // rax
  unsigned __int64 v93; // rdx
  unsigned int *v94; // rax
  unsigned __int64 *v95; // rax
  unsigned __int64 v96; // r12
  int v97; // edx
  __int64 v98; // rdi
  __int64 (*v99)(); // rcx
  unsigned __int8 v100; // dl
  unsigned int v101; // r12d
  __int32 v102; // edx
  __int64 v103; // r12
  __int64 v104; // rax
  __int64 v105; // rsi
  __int64 v106; // rbx
  __int64 v107; // r12
  __int64 v108; // r13
  __int64 *v109; // r15
  _QWORD *v110; // rax
  __int64 v111; // rdx
  __int64 v112; // rax
  int v113; // eax
  __int64 v114; // rax
  __int64 v115; // rdx
  unsigned __int16 *v116; // rbx
  unsigned __int16 *v117; // r12
  __int32 v118; // eax
  int v119; // [rsp+28h] [rbp-F8h]
  int v122; // [rsp+34h] [rbp-ECh]
  _QWORD *v123; // [rsp+38h] [rbp-E8h]
  int v124; // [rsp+38h] [rbp-E8h]
  __int64 *v125; // [rsp+38h] [rbp-E8h]
  unsigned int v127; // [rsp+40h] [rbp-E0h]
  __int64 v128; // [rsp+48h] [rbp-D8h]
  __int64 v129; // [rsp+48h] [rbp-D8h]
  unsigned int v130; // [rsp+48h] [rbp-D8h]
  __int64 v131; // [rsp+48h] [rbp-D8h]
  _QWORD *v132; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v133; // [rsp+58h] [rbp-C8h]
  __m128i v134; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v135; // [rsp+70h] [rbp-B0h]
  __int64 v136; // [rsp+78h] [rbp-A8h]
  __int64 v137; // [rsp+80h] [rbp-A0h]
  unsigned __int8 *v138; // [rsp+90h] [rbp-90h] BYREF
  __int64 v139; // [rsp+98h] [rbp-88h]
  __int64 v140[4]; // [rsp+A0h] [rbp-80h] BYREF
  __m128i v141; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v142; // [rsp+D0h] [rbp-50h] BYREF
  unsigned __int64 v143; // [rsp+D8h] [rbp-48h]
  __int64 v144; // [rsp+E0h] [rbp-40h]

  v6 = a1;
  v7 = a2;
  v8 = *(_DWORD *)(a2 + 24);
  if ( v8 > 308 )
  {
    if ( v8 <= 367 )
    {
      if ( v8 <= 365 )
      {
        if ( (unsigned int)(v8 - 309) > 1 )
          goto LABEL_170;
        v9 = -160;
        v10 = *(_QWORD **)(a2 + 96);
        if ( v8 != 309 )
          v9 = -240;
        v11 = *(_QWORD *)(a2 + 80);
        v12 = *(_QWORD *)(a1[2] + 8) + v9;
        v134.m128i_i64[0] = v11;
        if ( v11 )
        {
          sub_B96E90((__int64)&v134, v11, 1);
          v138 = (unsigned __int8 *)v134.m128i_i64[0];
          if ( v134.m128i_i64[0] )
          {
            sub_B976B0((__int64)&v134, (unsigned __int8 *)v134.m128i_i64[0], (__int64)&v138);
            v13 = a1[5];
            v134.m128i_i64[0] = 0;
            v14 = (__int64 *)a1[6];
            v139 = 0;
            v140[0] = 0;
            v15 = *(_QWORD **)(v13 + 32);
            v141.m128i_i64[0] = (__int64)v138;
            if ( v138 )
              sub_B96E90((__int64)&v141, (__int64)v138, 1);
            v16 = sub_2E7B380(v15, v12, (unsigned __int8 **)&v141, 0);
            goto LABEL_17;
          }
        }
        else
        {
          v138 = 0;
        }
        v13 = a1[5];
        v14 = (__int64 *)a1[6];
        v139 = 0;
        v140[0] = 0;
        v15 = *(_QWORD **)(v13 + 32);
        v141.m128i_i64[0] = 0;
        v16 = sub_2E7B380(v15, v12, (unsigned __int8 **)&v141, 0);
LABEL_17:
        v17 = (__int64)v16;
        if ( v141.m128i_i64[0] )
          sub_B91220((__int64)&v141, v141.m128i_i64[0]);
        sub_2E31040((__int64 *)(v13 + 40), v17);
        v18 = *v14;
        v19 = *(_QWORD *)v17;
        *(_QWORD *)(v17 + 8) = v14;
        v18 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)v17 = v18 | v19 & 7;
        *(_QWORD *)(v18 + 8) = v17;
        *v14 = v17 | *v14 & 7;
        if ( v139 )
          sub_2E882B0(v17, (__int64)v15, v139);
        if ( v140[0] )
          sub_2E88680(v17, (__int64)v15, v140[0]);
        v141.m128i_i8[0] = 15;
        v141.m128i_i32[0] &= 0xFFF000FF;
        v142 = 0;
        v143 = (unsigned __int64)v10;
        v141.m128i_i32[2] = 0;
        LODWORD(v144) = 0;
        goto LABEL_24;
      }
      v103 = -880;
      if ( v8 != 366 )
        v103 = -920;
      v104 = *(_QWORD *)(a2 + 40);
      v105 = *(_QWORD *)(a2 + 80);
      v106 = *(_QWORD *)(v104 + 40);
      v107 = *(_QWORD *)(a1[2] + 8) + v103;
      v134.m128i_i64[0] = v105;
      if ( v105 )
      {
        sub_B96E90((__int64)&v134, v105, 1);
        v138 = (unsigned __int8 *)v134.m128i_i64[0];
        if ( v134.m128i_i64[0] )
        {
          sub_B976B0((__int64)&v134, (unsigned __int8 *)v134.m128i_i64[0], (__int64)&v138);
          v108 = a1[5];
          v134.m128i_i64[0] = 0;
          v109 = (__int64 *)a1[6];
          v139 = 0;
          v140[0] = 0;
          v15 = *(_QWORD **)(v108 + 32);
          v141.m128i_i64[0] = (__int64)v138;
          if ( v138 )
            sub_B96E90((__int64)&v141, (__int64)v138, 1);
          v110 = sub_2E7B380(v15, v107, (unsigned __int8 **)&v141, 0);
          goto LABEL_147;
        }
      }
      else
      {
        v138 = 0;
      }
      v108 = a1[5];
      v109 = (__int64 *)a1[6];
      v139 = 0;
      v140[0] = 0;
      v15 = *(_QWORD **)(v108 + 32);
      v141.m128i_i64[0] = 0;
      v110 = sub_2E7B380(v15, v107, (unsigned __int8 **)&v141, 0);
LABEL_147:
      v17 = (__int64)v110;
      if ( v141.m128i_i64[0] )
        sub_B91220((__int64)&v141, v141.m128i_i64[0]);
      sub_2E31040((__int64 *)(v108 + 40), v17);
      v111 = *v109;
      v112 = *(_QWORD *)v17;
      *(_QWORD *)(v17 + 8) = v109;
      v111 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v17 = v111 | v112 & 7;
      *(_QWORD *)(v111 + 8) = v17;
      *v109 = v17 | *v109 & 7;
      if ( v139 )
        sub_2E882B0(v17, (__int64)v15, v139);
      if ( v140[0] )
        sub_2E88680(v17, (__int64)v15, v140[0]);
      v113 = *(_DWORD *)(v106 + 96);
      v141.m128i_i64[0] = 5;
      v142 = 0;
      LODWORD(v143) = v113;
LABEL_24:
      sub_2E8EAD0(v17, (__int64)v15, &v141);
      v20 = v138;
      if ( !v138 )
        goto LABEL_51;
LABEL_50:
      sub_B91220((__int64)&v138, (__int64)v20);
      goto LABEL_51;
    }
    if ( v8 != 372 )
      goto LABEL_170;
    v33 = *(_QWORD **)(a2 + 104);
    v123 = *(_QWORD **)(a2 + 96);
    v34 = *(_DWORD *)(a2 + 112);
    v35 = *(_QWORD *)(a2 + 80);
    v127 = v34;
    v36 = *(_QWORD *)(a1[2] + 8);
    v134.m128i_i64[0] = v35;
    v129 = v36 - 960;
    if ( v35 )
    {
      sub_B96E90((__int64)&v134, v35, 1);
      v138 = (unsigned __int8 *)v134.m128i_i64[0];
      if ( v134.m128i_i64[0] )
      {
        sub_B976B0((__int64)&v134, (unsigned __int8 *)v134.m128i_i64[0], (__int64)&v138);
        v37 = a1[5];
        v134.m128i_i64[0] = 0;
        v38 = (__int64 *)a1[6];
        v139 = 0;
        v140[0] = 0;
        v39 = *(_QWORD **)(v37 + 32);
        v141.m128i_i64[0] = (__int64)v138;
        if ( v138 )
          sub_B96E90((__int64)&v141, (__int64)v138, 1);
        v40 = sub_2E7B380(v39, v129, (unsigned __int8 **)&v141, 0);
LABEL_43:
        v41 = (__int64)v40;
        if ( v141.m128i_i64[0] )
          sub_B91220((__int64)&v141, v141.m128i_i64[0]);
        sub_2E31040((__int64 *)(v37 + 40), v41);
        v42 = *v38;
        v43 = *(_QWORD *)v41;
        *(_QWORD *)(v41 + 8) = v38;
        v42 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)v41 = v42 | v43 & 7;
        *(_QWORD *)(v42 + 8) = v41;
        *v38 = v41 | *v38 & 7;
        if ( v139 )
          sub_2E882B0(v41, (__int64)v39, v139);
        if ( v140[0] )
          sub_2E88680(v41, (__int64)v39, v140[0]);
        v141.m128i_i64[0] = 1;
        v143 = (unsigned __int64)v123;
        v142 = 0;
        sub_2E8EAD0(v41, (__int64)v39, &v141);
        v141.m128i_i64[0] = 1;
        v142 = 0;
        v143 = (unsigned __int64)v33;
        sub_2E8EAD0(v41, (__int64)v39, &v141);
        v141.m128i_i64[0] = 1;
        v142 = 0;
        v143 = 0;
        sub_2E8EAD0(v41, (__int64)v39, &v141);
        v141.m128i_i64[0] = 1;
        v142 = 0;
        v143 = v127;
        sub_2E8EAD0(v41, (__int64)v39, &v141);
        v20 = v138;
        if ( !v138 )
          goto LABEL_51;
        goto LABEL_50;
      }
    }
    else
    {
      v138 = 0;
    }
    v37 = a1[5];
    v38 = (__int64 *)a1[6];
    v139 = 0;
    v140[0] = 0;
    v39 = *(_QWORD **)(v37 + 32);
    v141.m128i_i64[0] = 0;
    v40 = sub_2E7B380(v39, v129, (unsigned __int8 **)&v141, 0);
    goto LABEL_43;
  }
  if ( v8 > 306 )
  {
    v49 = *(_DWORD *)(a2 + 64);
    v50 = -80;
    v51 = v49 - 1;
    v52 = (unsigned int *)(*(_QWORD *)(a2 + 40) + 40LL * (unsigned int)(v49 - 1));
    v53 = *(_QWORD *)(a2 + 80);
    if ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)v52 + 48LL) + 16LL * v52[2]) != 262 )
      v51 = v49;
    if ( v8 != 308 )
      v50 = -40;
    v119 = v51;
    v54 = *(_QWORD *)(v6[2] + 8) + v50;
    v134.m128i_i64[0] = v53;
    if ( v53 )
    {
      sub_B96E90((__int64)&v134, v53, 1);
      v141.m128i_i64[0] = v134.m128i_i64[0];
      if ( v134.m128i_i64[0] )
      {
        sub_B976B0((__int64)&v134, (unsigned __int8 *)v134.m128i_i64[0], (__int64)&v141);
        v55 = (_QWORD *)*v6;
        v134.m128i_i64[0] = 0;
        v141.m128i_i64[1] = 0;
        v142 = 0;
        v138 = (unsigned __int8 *)v141.m128i_i64[0];
        if ( v141.m128i_i64[0] )
          sub_B96E90((__int64)&v138, v141.m128i_i64[0], 1);
        v56 = sub_2E7B380(v55, v54, &v138, 0);
        goto LABEL_76;
      }
    }
    else
    {
      v141.m128i_i64[0] = 0;
    }
    v55 = (_QWORD *)*v6;
    v141.m128i_i64[1] = 0;
    v142 = 0;
    v138 = 0;
    v56 = sub_2E7B380(v55, v54, &v138, 0);
LABEL_76:
    v57 = (__int64)v56;
    if ( v141.m128i_i64[1] )
      sub_2E882B0((__int64)v56, (__int64)v55, v141.m128i_i64[1]);
    if ( v142 )
      sub_2E88680(v57, (__int64)v55, v142);
    if ( v138 )
      sub_B91220((__int64)&v138, (__int64)v138);
    v132 = v55;
    v133 = v57;
    if ( v141.m128i_i64[0] )
      sub_B91220((__int64)&v141, v141.m128i_i64[0]);
    if ( v134.m128i_i64[0] )
      sub_B91220((__int64)&v134, v134.m128i_i64[0]);
    v58 = *(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v7 + 40) + 40LL) + 96LL);
    v141.m128i_i8[0] = 9;
    v142 = 0;
    v141.m128i_i32[0] &= 0xFFF000FF;
    v143 = (unsigned __int64)v58;
    v141.m128i_i32[2] = 0;
    LODWORD(v144) = 0;
    sub_2E8EAD0(v133, (__int64)v132, &v141);
    v59 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v7 + 40) + 120LL) + 96LL);
    v60 = *(_DWORD *)(v59 + 32) <= 0x40u;
    v61 = *(_QWORD **)(v59 + 24);
    if ( !v60 )
      v61 = (_QWORD *)*v61;
    v143 = (unsigned __int64)v61;
    v141.m128i_i64[0] = 1;
    v142 = 0;
    sub_2E8EAD0(v133, (__int64)v132, &v141);
    v130 = 4;
    v138 = (unsigned __int8 *)v140;
    v139 = 0x800000000LL;
    v141.m128i_i64[0] = (__int64)&v142;
    v141.m128i_i64[1] = 0x800000000LL;
    while ( 1 )
    {
      v64 = v130;
      if ( v119 == v130 )
        break;
LABEL_90:
      v65 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v7 + 40) + 40 * v64) + 96LL);
      v66 = *(_QWORD **)(v65 + 24);
      if ( *(_DWORD *)(v65 + 32) > 0x40u )
        v66 = (_QWORD *)*v66;
      v124 = (int)v66;
      v122 = (unsigned __int16)v66 >> 3;
      v67 = (unsigned int)v139;
      v68 = (unsigned int)v139 + 1LL;
      v69 = *(_DWORD *)(v133 + 40) & 0xFFFFFF;
      if ( v68 > HIDWORD(v139) )
      {
        sub_C8D5F0((__int64)&v138, v140, v68, 4u, v62, v63);
        v67 = (unsigned int)v139;
      }
      *(_DWORD *)&v138[4 * v67] = v69;
      v136 = (unsigned int)v66;
      LODWORD(v139) = v139 + 1;
      v134.m128i_i64[0] = 1;
      v135 = 0;
      sub_2E8EAD0(v133, (__int64)v132, &v134);
      v70 = v130 + 1;
      switch ( (unsigned __int8)v66 & 7 )
      {
        case 0:
          goto LABEL_138;
        case 1:
        case 5:
        case 6:
          if ( (unsigned __int16)v66 >> 3 )
          {
            v130 = v70 + v122;
            v71 = v70;
            do
            {
              v72 = 5LL * v71++;
              sub_3752760(
                v6,
                (__int64 *)&v132,
                *(_QWORD *)(*(_QWORD *)(v7 + 40) + 8 * v72),
                *(_QWORD *)(*(_QWORD *)(v7 + 40) + 8 * v72 + 8),
                0,
                0,
                (__int64)a5,
                0,
                a3,
                a4);
            }
            while ( v71 != v130 );
          }
          else
          {
            ++v130;
          }
          if ( (v124 & 7) != 1 )
            continue;
          if ( v124 >= 0 )
            continue;
          v73 = *(_DWORD *)&v138[4 * (HIWORD(v124) & 0x7FFF)];
          v74 = v73 + 1;
          if ( !v122 )
            continue;
          v75 = v73 + v122 + 1;
          v76 = *(_DWORD *)&v138[4 * (unsigned int)v139 - 4] - v73;
          do
          {
            v77 = v76 + v74;
            v78 = v74++;
            sub_2E89ED0(v133, v78, v77);
          }
          while ( v75 != v74 );
          v64 = v130;
          if ( v119 != v130 )
            goto LABEL_90;
          goto LABEL_104;
        case 2:
          if ( !((unsigned __int16)v66 >> 3) )
            goto LABEL_138;
          v130 = v70 + v122;
          v101 = v70 + v122;
          do
          {
            v102 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v7 + 40) + 40LL * v70) + 96LL);
            v134.m128i_i64[0] = 0x10000000;
            v135 = 0;
            v134.m128i_i32[2] = v102;
            v136 = 0;
            ++v70;
            v137 = 0;
            *(__int32 *)((char *)v134.m128i_i32 + 3) = (unsigned __int8)(32 * ((unsigned int)(v102 - 1) <= 0x3FFFFFFE))
                                                     | 0x10;
            *(__int32 *)((char *)v134.m128i_i32 + 2) = v134.m128i_i16[1] & 0xF00F;
            v134.m128i_i32[0] &= 0xFFF000FF;
            sub_2E8EAD0(v133, (__int64)v132, &v134);
          }
          while ( v70 != v101 );
          break;
        case 3:
        case 4:
          if ( !((unsigned __int16)v66 >> 3) )
            goto LABEL_138;
          v125 = v6;
          v130 = v70 + v122;
          v90 = v70 + v122;
          do
          {
            v91 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v7 + 40) + 40LL * v70) + 96LL);
            v134.m128i_i64[0] = 0x410000000LL;
            v135 = 0;
            v134.m128i_i32[2] = v91;
            v136 = 0;
            v137 = 0;
            v134.m128i_i8[3] = (32 * ((unsigned int)(v91 - 1) <= 0x3FFFFFFE)) | 0x10;
            v134.m128i_i16[1] &= 0xF00Fu;
            v134.m128i_i32[0] &= 0xFFF000FF;
            sub_2E8EAD0(v133, (__int64)v132, &v134);
            v92 = v141.m128i_u32[2];
            v93 = v141.m128i_u32[2] + 1LL;
            if ( v93 > v141.m128i_u32[3] )
            {
              sub_C8D5F0((__int64)&v141, &v142, v93, 4u, v62, v63);
              v92 = v141.m128i_u32[2];
            }
            ++v70;
            *(_DWORD *)(v141.m128i_i64[0] + 4 * v92) = v91;
            ++v141.m128i_i32[2];
          }
          while ( v70 != v90 );
          v6 = v125;
          break;
        case 7:
          if ( (unsigned __int16)v66 >> 3 )
          {
            v130 = v70 + v122;
            do
            {
              v95 = (unsigned __int64 *)(*(_QWORD *)(v7 + 40) + 40LL * v70);
              v96 = *v95;
              sub_3752760(v6, (__int64 *)&v132, *v95, v95[1], 0, 0, (__int64)a5, 0, a3, a4);
              v97 = *(_DWORD *)(v96 + 24);
              if ( (unsigned int)(v97 - 37) <= 1 || (unsigned int)(v97 - 13) <= 1 )
              {
                v98 = *(_QWORD *)(*v6 + 16);
                v99 = *(__int64 (**)())(*(_QWORD *)v98 + 464LL);
                v100 = 0;
                if ( v99 != sub_30594B0 )
                  v100 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD))v99)(v98, *(_QWORD *)(v96 + 96), 0);
                v94 = (unsigned int *)(*(_QWORD *)(v133 + 32) + 40LL * ((*(_DWORD *)(v133 + 40) & 0xFFFFFFu) - 1));
                *v94 = (v100 << 8) | *v94 & 0xFFF000FF;
              }
              ++v70;
            }
            while ( v70 != v130 );
          }
          else
          {
LABEL_138:
            ++v130;
          }
          break;
      }
    }
LABEL_104:
    if ( (unsigned __int8)sub_B2D610(*(_QWORD *)*v6, 72) )
    {
      v79 = *(__int64 (**)(void))(*(_QWORD *)v6[4] + 2392LL);
      if ( v79 != sub_302E270 )
      {
        v114 = v79();
        v116 = (unsigned __int16 *)(v114 + 2 * v115);
        v117 = (unsigned __int16 *)v114;
        if ( (unsigned __int16 *)v114 != v116 )
        {
          do
          {
            v118 = *v117++;
            v134.m128i_i64[0] = 805306368;
            v135 = 0;
            v134.m128i_i32[2] = v118;
            v136 = 0;
            v137 = 0;
            sub_2E8EAD0(v133, (__int64)v132, &v134);
          }
          while ( v116 != v117 );
        }
      }
    }
    v80 = (unsigned int *)v141.m128i_i64[0];
    v81 = v141.m128i_i64[0] + 4LL * v141.m128i_u32[2];
    if ( v81 != v141.m128i_i64[0] )
    {
      v131 = v7;
      while ( 1 )
      {
        v82 = *v80;
        if ( (unsigned int)sub_2E89C70(v133, *v80, v6[3], 0) != -1 )
        {
          v83 = v133;
          v84 = sub_2E8E710(v133, v82, v6[3], 0, 0);
          if ( v84 == -1 )
          {
            MEMORY[0] &= ~0x400000000uLL;
            goto LABEL_170;
          }
          *(_BYTE *)(*(_QWORD *)(v83 + 32) + 40LL * v84 + 4) &= ~4u;
        }
        if ( (unsigned int *)v81 == ++v80 )
        {
          v7 = v131;
          break;
        }
      }
    }
    v85 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v7 + 40) + 80LL) + 96LL);
    if ( v85 )
    {
      v134.m128i_i64[0] = 14;
      v135 = 0;
      v136 = v85;
      sub_2E8EAD0(v133, (__int64)v132, &v134);
    }
    v86 = v133;
    v87 = (__int64 *)v6[6];
    sub_2E31040((__int64 *)(v6[5] + 40), v133);
    v88 = *v87;
    v89 = *(_QWORD *)v86;
    *(_QWORD *)(v86 + 8) = v87;
    v88 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v86 = v88 | v89 & 7;
    *(_QWORD *)(v88 + 8) = v86;
    *v87 = *v87 & 7 | v86;
    if ( (__int64 *)v141.m128i_i64[0] != &v142 )
      _libc_free(v141.m128i_u64[0]);
    if ( v138 != (unsigned __int8 *)v140 )
      _libc_free((unsigned __int64)v138);
    return;
  }
  if ( v8 == 50 )
  {
    sub_37553D0(
      (__int64)a1,
      (unsigned __int8 *)a2,
      0,
      a3,
      *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL) + 96LL),
      a5);
    return;
  }
  if ( v8 > 50 )
  {
    if ( v8 == 55 )
      return;
LABEL_170:
    BUG();
  }
  if ( v8 > 2 )
  {
    if ( v8 != 49 )
      goto LABEL_170;
    v21 = *(_QWORD **)(a2 + 40);
    v22 = v21[10];
    v23 = *(_DWORD *)(v21[5] + 96LL);
    v24 = *(_DWORD *)(v22 + 24);
    if ( v23 < 0 && v24 == -11 )
    {
      v44 = *(unsigned __int8 **)(v7 + 80);
      v45 = *(_QWORD *)(a1[2] + 8);
      v138 = v44;
      v46 = v45 - 400;
      if ( v44 )
      {
        sub_B96E90((__int64)&v138, (__int64)v44, 1);
        v141.m128i_i64[0] = (__int64)v138;
        if ( v138 )
        {
          sub_B976B0((__int64)&v138, v138, (__int64)&v141);
          v138 = 0;
        }
      }
      else
      {
        v141.m128i_i64[0] = 0;
      }
      v47 = (__int64 *)a1[6];
      v48 = a1[5];
      v141.m128i_i64[1] = 0;
      v142 = 0;
      sub_2F26260(v48, v47, v141.m128i_i64, v46, v23);
      if ( v141.m128i_i64[0] )
        sub_B91220((__int64)&v141, v141.m128i_i64[0]);
      if ( v138 )
        sub_B91220((__int64)&v138, (__int64)v138);
      return;
    }
    if ( v24 == 9 )
      v25 = *(_DWORD *)(v22 + 96);
    else
      v25 = sub_3752000(a1, v22, v21[11], (__int64)a5, (__int64)a5, a6);
    if ( v23 == v25 )
      return;
    v26 = *(_QWORD *)(v7 + 80);
    v27 = *(_QWORD *)(a1[2] + 8);
    v134.m128i_i64[0] = v26;
    v28 = v27 - 800;
    if ( v26 )
    {
      v128 = v27 - 800;
      sub_B96E90((__int64)&v134, v26, 1);
      v28 = v128;
      v138 = (unsigned __int8 *)v134.m128i_i64[0];
      if ( v134.m128i_i64[0] )
      {
        sub_B976B0((__int64)&v134, (unsigned __int8 *)v134.m128i_i64[0], (__int64)&v138);
        v28 = v128;
        v134.m128i_i64[0] = 0;
      }
    }
    else
    {
      v138 = 0;
    }
    v29 = a1[5];
    v139 = 0;
    v30 = (__int64 *)v6[6];
    v140[0] = 0;
    v31 = sub_2F26260(v29, v30, (__int64 *)&v138, v28, v23);
    v141.m128i_i64[0] = 0;
    v142 = 0;
    v141.m128i_i32[2] = v25;
    v143 = 0;
    v144 = 0;
    sub_2E8EAD0(v32, (__int64)v31, &v141);
    if ( v138 )
      sub_B91220((__int64)&v138, (__int64)v138);
LABEL_51:
    if ( v134.m128i_i64[0] )
      sub_B91220((__int64)&v134, v134.m128i_i64[0]);
    return;
  }
  if ( v8 <= 0 )
    goto LABEL_170;
}
