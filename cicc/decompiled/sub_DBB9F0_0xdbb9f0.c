// Function: sub_DBB9F0
// Address: 0xdbb9f0
//
__int64 __fastcall sub_DBB9F0(__int64 a1, __int64 a2, unsigned int a3, unsigned int a4)
{
  unsigned int v4; // r15d
  __int64 v5; // r13
  __int64 v8; // rsi
  unsigned int v9; // r14d
  __int64 v10; // rax
  __int64 v11; // r8
  __int64 *v12; // rdx
  __int64 v13; // rcx
  __int64 *v14; // r12
  __int64 v16; // rsi
  __int64 v17; // rax
  __int16 v18; // cx
  int v19; // edx
  int v20; // r9d
  __int64 v21; // rax
  unsigned int v22; // r12d
  __int64 v23; // rdx
  __int64 v24; // rax
  __int16 v25; // ax
  char v26; // bl
  char v27; // r14
  unsigned int i; // r15d
  char v29; // r8
  __int32 v30; // eax
  int v31; // eax
  __int64 v32; // rax
  unsigned int v33; // ebx
  __int64 v34; // rax
  __int64 v35; // rax
  unsigned int v36; // ebx
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 *v39; // rdx
  unsigned int v40; // r12d
  __int64 *v41; // r15
  __int64 v42; // rsi
  __int64 *v43; // rax
  __int64 v44; // rax
  int v45; // ebx
  __int64 v46; // rax
  __int64 *v47; // rdx
  __int64 *v48; // r13
  __int64 v49; // rsi
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rax
  unsigned __int8 v54; // al
  __int64 v55; // rsi
  __int64 v56; // rsi
  __int32 v57; // eax
  __int64 v58; // rcx
  __int64 v59; // r8
  __int64 v60; // r9
  __int64 v61; // rax
  __int32 v62; // eax
  int v63; // eax
  unsigned int v64; // eax
  __int64 v65; // rdx
  unsigned __int64 v66; // rax
  unsigned int v67; // eax
  __int32 v68; // eax
  int v69; // eax
  __int64 v70; // rax
  unsigned int v71; // eax
  __int64 v72; // rax
  unsigned __int8 v73; // al
  __int64 *v74; // r8
  int v75; // eax
  unsigned int v76; // ebx
  __int64 *v77; // r12
  __int64 v78; // rsi
  __int64 v79; // rax
  __int64 v80; // rax
  __int64 v81; // rax
  __int64 v82; // rax
  unsigned __int64 v83; // rdx
  unsigned __int64 v84; // rax
  unsigned int v85; // eax
  __int64 v86; // rdx
  __int64 v87; // rcx
  __int64 v88; // r8
  unsigned int v89; // ebx
  unsigned __int64 v90; // rax
  __int64 v91; // rax
  __int64 v92; // rdx
  __int64 v93; // rcx
  __int64 v94; // r8
  __int64 v95; // rax
  _QWORD *v96; // rcx
  _QWORD *v97; // rsi
  _QWORD *v98; // rdx
  __int64 v99; // rax
  __int64 *v100; // rax
  __int64 *v101; // [rsp+10h] [rbp-210h]
  unsigned __int64 v102; // [rsp+10h] [rbp-210h]
  char v103; // [rsp+18h] [rbp-208h]
  unsigned __int64 v104; // [rsp+20h] [rbp-200h]
  __int64 v105; // [rsp+20h] [rbp-200h]
  int v106; // [rsp+30h] [rbp-1F0h]
  unsigned int v107; // [rsp+30h] [rbp-1F0h]
  unsigned int v108; // [rsp+30h] [rbp-1F0h]
  unsigned int v109; // [rsp+38h] [rbp-1E8h]
  __int64 v110; // [rsp+38h] [rbp-1E8h]
  unsigned int v111; // [rsp+40h] [rbp-1E0h]
  unsigned int v112; // [rsp+48h] [rbp-1D8h]
  __int64 v113; // [rsp+48h] [rbp-1D8h]
  int v114; // [rsp+50h] [rbp-1D0h]
  __int64 *v115; // [rsp+50h] [rbp-1D0h]
  bool v116; // [rsp+50h] [rbp-1D0h]
  bool v117; // [rsp+50h] [rbp-1D0h]
  __int64 v118; // [rsp+50h] [rbp-1D0h]
  __int64 v119; // [rsp+58h] [rbp-1C8h]
  unsigned int v120; // [rsp+58h] [rbp-1C8h]
  __int64 v121; // [rsp+58h] [rbp-1C8h]
  unsigned int v122; // [rsp+68h] [rbp-1B8h]
  __int64 *v123; // [rsp+68h] [rbp-1B8h]
  unsigned int v124; // [rsp+78h] [rbp-1A8h]
  char v125; // [rsp+8Eh] [rbp-192h] BYREF
  char v126; // [rsp+8Fh] [rbp-191h] BYREF
  __int64 v127[2]; // [rsp+90h] [rbp-190h] BYREF
  _QWORD *v128; // [rsp+A0h] [rbp-180h] BYREF
  unsigned int v129; // [rsp+A8h] [rbp-178h]
  char *v130; // [rsp+B0h] [rbp-170h] BYREF
  unsigned int v131; // [rsp+B8h] [rbp-168h]
  char *v132; // [rsp+C0h] [rbp-160h] BYREF
  unsigned int v133; // [rsp+C8h] [rbp-158h]
  __int64 v134; // [rsp+D0h] [rbp-150h] BYREF
  __int32 v135; // [rsp+D8h] [rbp-148h]
  __int64 v136; // [rsp+E0h] [rbp-140h] BYREF
  int v137; // [rsp+E8h] [rbp-138h]
  __int64 v138[2]; // [rsp+F0h] [rbp-130h] BYREF
  __int64 v139; // [rsp+100h] [rbp-120h] BYREF
  __int64 v140; // [rsp+110h] [rbp-110h] BYREF
  unsigned int v141; // [rsp+118h] [rbp-108h]
  __int64 v142; // [rsp+120h] [rbp-100h] BYREF
  unsigned int v143; // [rsp+128h] [rbp-F8h]
  char *v144; // [rsp+130h] [rbp-F0h] BYREF
  unsigned int v145; // [rsp+138h] [rbp-E8h]
  __int64 v146[2]; // [rsp+140h] [rbp-E0h] BYREF
  unsigned __int64 v147; // [rsp+150h] [rbp-D0h] BYREF
  unsigned int v148; // [rsp+158h] [rbp-C8h]
  __int64 v149[2]; // [rsp+160h] [rbp-C0h] BYREF
  unsigned __int64 v150; // [rsp+170h] [rbp-B0h] BYREF
  unsigned __int32 v151; // [rsp+178h] [rbp-A8h]
  __int64 v152; // [rsp+180h] [rbp-A0h] BYREF
  unsigned int v153; // [rsp+188h] [rbp-98h]
  char v154; // [rsp+190h] [rbp-90h]
  __m128i v155; // [rsp+1A0h] [rbp-80h] BYREF
  __int64 v156; // [rsp+1B0h] [rbp-70h] BYREF
  __int64 v157; // [rsp+1B8h] [rbp-68h]
  __int64 v158[2]; // [rsp+1C0h] [rbp-60h] BYREF
  __int64 v159[2]; // [rsp+1D0h] [rbp-50h] BYREF
  __int16 v160; // [rsp+1E0h] [rbp-40h]

  v4 = a3;
  v5 = a1;
  if ( a3 )
  {
    v8 = *(_QWORD *)(a1 + 1008);
    v9 = 2;
    v10 = *(unsigned int *)(a1 + 1024);
    if ( !(_DWORD)v10 )
      goto LABEL_8;
  }
  else
  {
    v8 = *(_QWORD *)(a1 + 976);
    v9 = 1;
    v10 = *(unsigned int *)(a1 + 992);
    if ( !(_DWORD)v10 )
      goto LABEL_8;
  }
  v11 = ((_DWORD)v10 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v12 = (__int64 *)(v8 + 40 * v11);
  v13 = *v12;
  if ( *v12 == a2 )
  {
LABEL_4:
    if ( v12 != (__int64 *)(v8 + 40 * v10) )
      return (__int64)(v12 + 1);
  }
  else
  {
    v19 = 1;
    while ( v13 != -4096 )
    {
      v20 = v19 + 1;
      LODWORD(v11) = (v10 - 1) & (v19 + v11);
      v12 = (__int64 *)(v8 + 40LL * (unsigned int)v11);
      v13 = *v12;
      if ( a2 == *v12 )
        goto LABEL_4;
      v19 = v20;
    }
  }
LABEL_8:
  if ( !*(_WORD *)(a2 + 24) )
  {
    v16 = *(_QWORD *)(a2 + 32);
    v151 = *(_DWORD *)(v16 + 32);
    if ( v151 > 0x40 )
      sub_C43780((__int64)&v150, (const void **)(v16 + 24));
    else
      v150 = *(_QWORD *)(v16 + 24);
    sub_AADBC0((__int64)&v155, (__int64 *)&v150);
    v14 = sub_DB0AC0(a1, a2, v4, (__int64)&v155);
    sub_969240(&v156);
    sub_969240(v155.m128i_i64);
    sub_969240((__int64 *)&v150);
    return (__int64)v14;
  }
  if ( a4 <= (unsigned int)qword_4F88EA8 )
  {
    v17 = sub_D95540(a2);
    v122 = sub_D97050(a1, v17);
    sub_AADB10((__int64)&v134, v122, 1);
    if ( v4 )
    {
      v64 = sub_DB55F0(a1, a2);
      if ( v64 )
      {
        v124 = v64;
        sub_9865E0((__int64)&v140, v122);
        sub_9865C0((__int64)&v144, (__int64)&v140);
        sub_D94900((__int64)&v144, v124);
        sub_9865C0((__int64)&v147, (__int64)&v144);
        if ( v148 > 0x40 )
        {
          sub_C47690((__int64 *)&v147, v124);
        }
        else
        {
          v65 = 0;
          if ( v124 != v148 )
            v65 = v147 << v124;
          v66 = 0;
          if ( v148 )
            v66 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v148;
          v147 = v65 & v66;
        }
        sub_C46A40((__int64)&v147, 1);
        v151 = v148;
        v148 = 0;
        v150 = v147;
        sub_986680((__int64)v138, v122);
        sub_AADC30((__int64)&v155, (__int64)v138, (__int64 *)&v150);
        sub_D920D0(&v134, v155.m128i_i64);
        sub_969240(&v156);
        sub_969240(v155.m128i_i64);
        sub_969240(v138);
        sub_969240((__int64 *)&v150);
        sub_969240((__int64 *)&v147);
        sub_969240((__int64 *)&v144);
        sub_969240(&v140);
      }
    }
    else
    {
      sub_DB5510((__int64)&v132, a1, a2);
      sub_9691E0((__int64)&v155, v122, -1, 1u, 0);
      sub_C4B490((__int64)v138, (__int64)&v155, (__int64)&v132);
      sub_969240(v155.m128i_i64);
      if ( !sub_9867B0((__int64)v138) )
      {
        sub_9691E0((__int64)&v144, v122, -1, 1u, 0);
        sub_C46B40((__int64)&v144, v138);
        v148 = v145;
        v145 = 0;
        v147 = (unsigned __int64)v144;
        sub_C46A40((__int64)&v147, 1);
        v67 = v148;
        v148 = 0;
        v151 = v67;
        v150 = v147;
        sub_9691E0((__int64)&v140, v122, 0, 0, 0);
        sub_AADC30((__int64)&v155, (__int64)&v140, (__int64 *)&v150);
        sub_D920D0(&v134, v155.m128i_i64);
        sub_969240(&v156);
        sub_969240(v155.m128i_i64);
        sub_969240(&v140);
        sub_969240((__int64 *)&v150);
        sub_969240((__int64 *)&v147);
        sub_969240((__int64 *)&v144);
      }
      sub_969240(v138);
      sub_969240((__int64 *)&v132);
    }
    v18 = *(_WORD *)(a2 + 24);
    switch ( v18 )
    {
      case 0:
      case 16:
        goto LABEL_169;
      case 1:
        sub_988CD0((__int64)&v155, *(_QWORD *)a1, v122);
        goto LABEL_62;
      case 2:
        v53 = sub_DBB9F0(a1, *(_QWORD *)(a2 + 32), v4, a4 + 1);
        sub_AAF450((__int64)&v147, v53);
        sub_AB4490((__int64)&v150, (__int64)&v147, v122);
        goto LABEL_58;
      case 3:
        v52 = sub_DBB9F0(a1, *(_QWORD *)(a2 + 32), v4, a4 + 1);
        sub_AAF450((__int64)&v147, v52);
        sub_AB3F90((__int64)&v150, (__int64)&v147, v122);
        goto LABEL_58;
      case 4:
        v51 = sub_DBB9F0(a1, *(_QWORD *)(a2 + 32), v4, a4 + 1);
        sub_AAF450((__int64)&v147, v51);
        sub_AB41D0((__int64)&v150, (__int64)&v147, v122);
LABEL_58:
        sub_AB2160((__int64)&v155, (__int64)&v134, (__int64)&v150, v9);
        v14 = sub_DB0AC0(a1, a2, v4, (__int64)&v155);
        sub_969240(&v156);
        sub_969240(v155.m128i_i64);
        sub_969240(&v152);
        sub_969240((__int64 *)&v150);
        sub_969240(v149);
        sub_969240((__int64 *)&v147);
        goto LABEL_29;
      case 5:
        v120 = a4 + 1;
        v44 = sub_DBB9F0(a1, **(_QWORD **)(a2 + 32), v4, a4 + 1);
        sub_AAF450((__int64)&v150, v44);
        v45 = (*(unsigned __int16 *)(a2 + 28) >> 1) & 2;
        if ( (*(_WORD *)(a2 + 28) & 2) != 0 )
          v45 = (*(unsigned __int16 *)(a2 + 28) >> 1) & 2 | 1;
        v46 = sub_D91800(*(_QWORD *)(a2 + 32), *(_QWORD *)(a2 + 40), 1);
        v115 = v47;
        if ( (__int64 *)v46 != v47 )
        {
          v48 = (__int64 *)v46;
          do
          {
            v49 = *v48++;
            v50 = sub_DBB9F0(a1, v49, v4, v120);
            sub_ABA0E0((__int64)&v155, (__int64)&v150, v50, v45, v9);
            sub_D920D0((__int64 *)&v150, v155.m128i_i64);
            sub_969240(&v156);
            sub_969240(v155.m128i_i64);
          }
          while ( v115 != v48 );
          v5 = a1;
        }
        goto LABEL_50;
      case 6:
        v36 = a4 + 1;
        v37 = sub_DBB9F0(a1, **(_QWORD **)(a2 + 32), v4, v36);
        sub_AAF450((__int64)&v150, v37);
        v38 = sub_D91800(*(_QWORD *)(a2 + 32), *(_QWORD *)(a2 + 40), 1);
        v123 = v39;
        if ( v39 != (__int64 *)v38 )
        {
          v113 = a2;
          v40 = v4;
          v41 = (__int64 *)v38;
          do
          {
            v42 = *v41++;
            v43 = (__int64 *)sub_DBB9F0(a1, v42, v40, v36);
            sub_AB5480((__int64)&v155, (__int64)&v150, v43);
            sub_D920D0((__int64 *)&v150, v155.m128i_i64);
            sub_969240(&v156);
            sub_969240(v155.m128i_i64);
          }
          while ( v123 != v41 );
          v4 = v40;
          a2 = v113;
        }
LABEL_50:
        sub_AB2160((__int64)&v155, (__int64)&v134, (__int64)&v150, v9);
        v14 = sub_DB0AC0(v5, a2, v4, (__int64)&v155);
        sub_969240(&v156);
        sub_969240(v155.m128i_i64);
        sub_969240(&v152);
        sub_969240((__int64 *)&v150);
        goto LABEL_29;
      case 7:
        v33 = a4 + 1;
        v34 = sub_DBB9F0(a1, *(_QWORD *)(a2 + 32), v4, v33);
        sub_AAF450((__int64)&v144, v34);
        v35 = sub_DBB9F0(a1, *(_QWORD *)(a2 + 40), v4, v33);
        sub_AAF450((__int64)&v147, v35);
        sub_AB6A50((__int64)&v150, (__int64)&v144, (__int64)&v147);
        sub_AB2160((__int64)&v155, (__int64)&v134, (__int64)&v150, v9);
        v14 = sub_DB0AC0(a1, a2, v4, (__int64)&v155);
        sub_969240(&v156);
        sub_969240(v155.m128i_i64);
        sub_969240(&v152);
        sub_969240((__int64 *)&v150);
        sub_969240(v149);
        sub_969240((__int64 *)&v147);
        sub_969240(v146);
        sub_969240((__int64 *)&v144);
        goto LABEL_29;
      case 8:
        v25 = *(_WORD *)(a2 + 28);
        if ( (v25 & 2) != 0 )
        {
          v70 = sub_DBB9F0(a1, **(_QWORD **)(a2 + 32), 0, 0);
          sub_AB0A00((__int64)&v140, v70);
          if ( !sub_9867B0((__int64)&v140) )
          {
            sub_9691E0((__int64)&v144, v122, 0, 0, 0);
            sub_9865C0((__int64)&v147, (__int64)&v140);
            sub_AADC30((__int64)&v150, (__int64)&v147, (__int64 *)&v144);
            sub_AB2160((__int64)&v155, (__int64)&v134, (__int64)&v150, v9);
            sub_D920D0(&v134, v155.m128i_i64);
            sub_969240(&v156);
            sub_969240(v155.m128i_i64);
            sub_969240(&v152);
            sub_969240((__int64 *)&v150);
            sub_969240((__int64 *)&v147);
            sub_969240((__int64 *)&v144);
          }
          sub_969240(&v140);
          v25 = *(_WORD *)(a2 + 28);
        }
        if ( (v25 & 4) != 0 )
        {
          v26 = 1;
          v114 = *(_QWORD *)(a2 + 40);
          if ( v114 == 1 )
            goto LABEL_133;
          v112 = v9;
          v27 = 1;
          v111 = v4;
          for ( i = 1; i != v114; ++i )
          {
            if ( !(unsigned __int8)sub_DBED40(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL * i)) )
              v26 = 0;
            if ( !(unsigned __int8)sub_DBEC80(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL * i)) )
              v27 = 0;
          }
          v29 = v27;
          v4 = v111;
          v9 = v112;
          if ( v26 )
          {
LABEL_133:
            sub_986680((__int64)&v147, v122);
            v81 = sub_DBB9F0(a1, **(_QWORD **)(a2 + 32), 1, 0);
            sub_AB14C0((__int64)&v144, v81);
            sub_9875E0((__int64)&v150, (__int64 *)&v144, (__int64 *)&v147);
            sub_AB2160((__int64)&v155, (__int64)&v134, (__int64)&v150, v9);
            sub_D920D0(&v134, v155.m128i_i64);
            sub_969240(&v156);
            sub_969240(v155.m128i_i64);
            sub_969240(&v152);
            sub_969240((__int64 *)&v150);
            sub_969240((__int64 *)&v144);
            sub_969240((__int64 *)&v147);
          }
          else if ( v29 )
          {
            v82 = sub_DBB9F0(a1, **(_QWORD **)(a2 + 32), 1, 0);
            sub_AB13A0((__int64)&v144, v82);
            sub_C46A40((__int64)&v144, 1);
            v148 = v145;
            v145 = 0;
            v147 = (unsigned __int64)v144;
            sub_986680((__int64)&v140, v122);
            sub_9875E0((__int64)&v150, &v140, (__int64 *)&v147);
            sub_AB2160((__int64)&v155, (__int64)&v134, (__int64)&v150, v112);
            sub_D920D0(&v134, v155.m128i_i64);
            sub_969240(&v156);
            sub_969240(v155.m128i_i64);
            sub_969240(&v152);
            sub_969240((__int64 *)&v150);
            sub_969240(&v140);
            sub_969240((__int64 *)&v147);
            sub_969240((__int64 *)&v144);
          }
        }
        if ( *(_QWORD *)(a2 + 40) != 2 )
          goto LABEL_43;
        v121 = sub_DCF3A0(a1, *(_QWORD *)(a2 + 48), 1);
        if ( sub_D96A50(v121) )
          goto LABEL_118;
        sub_9865C0((__int64)&v144, *(_QWORD *)(v121 + 32) + 24LL);
        v89 = v145;
        if ( v122 >= v145 )
        {
          if ( v122 <= v145 )
            goto LABEL_151;
          sub_C449B0((__int64)&v155, (const void **)&v144, v122);
        }
        else
        {
          if ( v122 < v89 - (unsigned int)sub_9871A0((__int64)&v144) )
            goto LABEL_145;
          sub_C44740((__int64)&v155, &v144, v122);
        }
        sub_D91810((__int64 *)&v144, v155.m128i_i64);
        sub_969240(v155.m128i_i64);
LABEL_151:
        if ( v122 == v145 )
        {
          v91 = sub_D33D80((_QWORD *)a2, a1, v86, v87, v88);
          sub_DBEFC0(&v147, a1, **(_QWORD **)(a2 + 32), v91, &v144);
          sub_AB2160((__int64)&v155, (__int64)&v134, (__int64)&v147, v9);
          sub_D920D0(&v134, v155.m128i_i64);
          sub_969240(&v156);
          sub_969240(v155.m128i_i64);
          v95 = sub_D33D80((_QWORD *)a2, a1, v92, v93, v94);
          sub_DBF480(&v150, a1, **(_QWORD **)(a2 + 32), v95, &v144);
          sub_AB2160((__int64)&v155, (__int64)&v134, (__int64)&v150, v9);
          sub_D920D0(&v134, v155.m128i_i64);
          sub_969240(&v156);
          sub_969240(v155.m128i_i64);
          sub_969240(&v152);
          sub_969240((__int64 *)&v150);
          sub_969240(v149);
          sub_969240((__int64 *)&v147);
        }
LABEL_145:
        sub_969240((__int64 *)&v144);
LABEL_118:
        if ( (_BYTE)qword_4F88C08 )
        {
          v118 = sub_DCF3A0(a1, *(_QWORD *)(a2 + 48), 2);
          if ( !sub_D96A50(v118) )
          {
            v72 = sub_D95540(v121);
            if ( sub_D97050(a1, v72) <= (unsigned __int64)v122 && (*(_BYTE *)(a2 + 28) & 1) != 0 )
            {
              sub_DE4FD0(&v150, a1, a2, v118, v122, v4);
              sub_AB2160((__int64)&v155, (__int64)&v134, (__int64)&v150, v9);
              sub_D920D0(&v134, v155.m128i_i64);
              sub_969240(&v156);
              sub_969240(v155.m128i_i64);
              sub_969240(&v152);
              sub_969240((__int64 *)&v150);
            }
          }
        }
LABEL_43:
        v30 = v135;
        v135 = 0;
        v155.m128i_i32[2] = v30;
        v155.m128i_i64[0] = v134;
        v31 = v137;
        v137 = 0;
        LODWORD(v157) = v31;
        v156 = v136;
        v14 = sub_DB0AC0(a1, a2, v4, (__int64)&v155);
        sub_969240(&v156);
        sub_969240(v155.m128i_i64);
        goto LABEL_29;
      case 9:
      case 10:
      case 11:
      case 12:
      case 13:
        v109 = dword_3F74E60[(unsigned __int16)(v18 - 9)];
        v21 = sub_DBB9F0(a1, **(_QWORD **)(a2 + 32), v4, a4 + 1);
        sub_AAF450((__int64)&v147, v21);
        v106 = *(_QWORD *)(a2 + 40);
        if ( v106 != 1 )
        {
          v119 = a2;
          v22 = 1;
          do
          {
            sub_AAF450((__int64)&v155, (__int64)&v147);
            v23 = v22++;
            v24 = sub_DBB9F0(a1, *(_QWORD *)(*(_QWORD *)(v119 + 32) + 8 * v23), v4, a4 + 1);
            sub_AAF450((__int64)v158, v24);
            sub_ABD750((__int64)&v150, v109, (__int64)&v155);
            sub_D920D0((__int64 *)&v147, (__int64 *)&v150);
            sub_969240(&v152);
            sub_969240((__int64 *)&v150);
            sub_969240(v159);
            sub_969240(v158);
            sub_969240(&v156);
            sub_969240(v155.m128i_i64);
          }
          while ( v106 != v22 );
          a2 = v119;
        }
        sub_AB2160((__int64)&v155, (__int64)&v134, (__int64)&v147, v9);
        v14 = sub_DB0AC0(a1, a2, v4, (__int64)&v155);
        sub_969240(&v156);
        sub_969240(v155.m128i_i64);
        sub_969240(v149);
        sub_969240((__int64 *)&v147);
        goto LABEL_29;
      case 14:
        v32 = sub_DBB9F0(a1, *(_QWORD *)(a2 + 32), v4, a4 + 1);
        sub_AAF450((__int64)&v150, v32);
        sub_AAF450((__int64)&v155, (__int64)&v150);
        v14 = sub_DB0AC0(a1, a2, v4, (__int64)&v155);
        sub_969240(&v156);
        sub_969240(v155.m128i_i64);
        sub_969240(&v152);
        sub_969240((__int64 *)&v150);
        goto LABEL_29;
      case 15:
        v110 = *(_QWORD *)(a2 - 8);
        v54 = *(_BYTE *)v110;
        if ( *(_BYTE *)v110 <= 0x1Cu )
          goto LABEL_110;
        if ( (*(_BYTE *)(v110 + 7) & 0x20) == 0 )
          goto LABEL_124;
        v55 = sub_B91C10(v110, 4);
        if ( v55 )
        {
          sub_ABEA30((__int64)&v155, v55);
LABEL_67:
          v154 = 1;
          v151 = v155.m128i_u32[2];
          v150 = v155.m128i_i64[0];
          v153 = v157;
          v152 = v156;
          sub_AB2160((__int64)&v155, (__int64)&v134, (__int64)&v150, v9);
          sub_D920D0(&v134, v155.m128i_i64);
          sub_969240(&v156);
          sub_969240(v155.m128i_i64);
          goto LABEL_68;
        }
        v54 = *(_BYTE *)v110;
        if ( *(_BYTE *)v110 <= 0x1Cu )
          goto LABEL_110;
LABEL_124:
        v73 = v54 - 34;
        if ( v73 <= 0x33u && ((0x8000000000041uLL >> v73) & 1) != 0 )
        {
          sub_B492D0((__int64)&v155, v110);
          if ( LOBYTE(v158[0]) )
            goto LABEL_67;
          v54 = *(_BYTE *)v110;
LABEL_110:
          if ( v54 == 22 )
          {
            sub_B2D8F0((__int64)&v155, v110);
            if ( LOBYTE(v158[0]) )
              goto LABEL_67;
          }
        }
        v154 = 0;
LABEL_68:
        sub_DBB110((__int64)v138, (_QWORD *)a1, a2 - 32);
        sub_AB2160((__int64)&v155, (__int64)&v134, (__int64)v138, 0);
        sub_D920D0(&v134, v155.m128i_i64);
        sub_969240(&v156);
        sub_969240(v155.m128i_i64);
        v104 = *(_QWORD *)(a1 + 8);
        sub_9AC3E0((__int64)&v140, v110, v104, 0, *(_QWORD *)(a1 + 32), 0, *(_QWORD *)(a1 + 40), 1);
        if ( v122 != v141 )
        {
          sub_D951F0((__int64)&v155, (__int64)&v140, v122);
          sub_D91810(&v140, v155.m128i_i64);
          sub_D91810(&v142, &v156);
          sub_969240(&v156);
          sub_969240(v155.m128i_i64);
        }
        v107 = sub_9AF8B0(v110, v104, 0, *(_QWORD *)(a1 + 32), 0, *(_QWORD *)(a1 + 40), 1);
        v56 = *(_QWORD *)(*(_QWORD *)(a2 - 8) + 8LL);
        if ( *(_BYTE *)(v56 + 8) == 14 )
        {
          v71 = sub_AE43A0(v104, v56);
          if ( v71 - v122 < v107 && v122 < v71 && (int)(v71 - v122) > 0 )
            v107 = v122 + v107 - v71;
        }
        if ( v107 > 1 )
        {
          sub_C48300((__int64)&v155, (__int64)&v140, v107);
          v116 = sub_9867B0((__int64)&v155);
          sub_969240(v155.m128i_i64);
          if ( !v116 )
            sub_9870B0((__int64)&v140, v141 - v107, v141);
          sub_C48300((__int64)&v155, (__int64)&v142, v107);
          v117 = sub_9867B0((__int64)&v155);
          sub_969240(v155.m128i_i64);
          if ( !v117 )
            sub_9870B0((__int64)&v142, v143 - v107, v143);
        }
        sub_D95160((__int64)&v147, (__int64)&v140);
        sub_C46A40((__int64)&v147, 1);
        v57 = v148;
        v148 = 0;
        v155.m128i_i32[2] = v57;
        v155.m128i_i64[0] = v147;
        sub_9865C0((__int64)&v144, (__int64)&v142);
        v103 = sub_AAD8B0((__int64)&v144, &v155);
        sub_969240((__int64 *)&v144);
        sub_969240(v155.m128i_i64);
        sub_969240((__int64 *)&v147);
        if ( !v103 )
        {
          sub_D95160((__int64)&v132, (__int64)&v140);
          sub_C46A40((__int64)&v132, 1);
          v145 = v133;
          v133 = 0;
          v144 = v132;
          sub_9865C0((__int64)&v130, (__int64)&v142);
          sub_AADC30((__int64)&v147, (__int64)&v130, (__int64 *)&v144);
          sub_AB2160((__int64)&v155, (__int64)&v134, (__int64)&v147, v9);
          sub_D920D0(&v134, v155.m128i_i64);
          sub_969240(&v156);
          sub_969240(v155.m128i_i64);
          sub_969240(v149);
          sub_969240((__int64 *)&v147);
          sub_969240((__int64 *)&v130);
          sub_969240((__int64 *)&v144);
          sub_969240((__int64 *)&v132);
        }
        if ( v107 > 1 )
        {
          sub_9865E0((__int64)&v130, v122);
          v108 = v107 - 1;
          sub_9865C0((__int64)&v132, (__int64)&v130);
          sub_D94900((__int64)&v132, v108);
          sub_C46A40((__int64)&v132, 1);
          v145 = v133;
          v133 = 0;
          v144 = v132;
          sub_986680((__int64)v127, v122);
          sub_9865C0((__int64)&v128, (__int64)v127);
          sub_D94900((__int64)&v128, v108);
          sub_AADC30((__int64)&v147, (__int64)&v128, (__int64 *)&v144);
          sub_AB2160((__int64)&v155, (__int64)&v134, (__int64)&v147, v9);
          sub_D920D0(&v134, v155.m128i_i64);
          sub_969240(&v156);
          sub_969240(v155.m128i_i64);
          sub_969240(v149);
          sub_969240((__int64 *)&v147);
          sub_969240((__int64 *)&v128);
          sub_969240(v127);
          sub_969240((__int64 *)&v144);
          sub_969240((__int64 *)&v132);
          sub_969240((__int64 *)&v130);
        }
        if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a2 - 8) + 8LL) + 8LL) == 14 && !v4 )
        {
          v83 = sub_BD4FF0((unsigned __int8 *)v110, v104, &v125, &v126);
          if ( v83 > 1 )
          {
            if ( v122 > 0x3F || v122 && (v58 = 64 - v122, v83 <= 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v122)) )
            {
              sub_9691E0((__int64)&v155, v122, v83, 0, 0);
              sub_9691E0((__int64)&v147, v122, -1, 1u, 0);
              sub_D949F0((__int64)v127, (__int64 *)&v147, (__int64)&v155);
              sub_969240((__int64 *)&v147);
              sub_969240(v155.m128i_i64);
              v102 = 1LL << sub_BD5420(*(unsigned __int8 **)(a2 - 8), v104);
              v84 = sub_C459C0((__int64)v127, v102);
              sub_9691E0((__int64)&v155, v122, v84, 0, 0);
              sub_C46B40((__int64)v127, v155.m128i_i64);
              sub_969240(v155.m128i_i64);
              sub_9691E0((__int64)&v128, v122, 0, 0, 0);
              v155 = (__m128i)v104;
              v156 = 0;
              v157 = 0;
              v158[0] = 0;
              v158[1] = 0;
              v159[0] = 0;
              v159[1] = 0;
              v160 = 257;
              if ( (unsigned __int8)sub_9B6260(v110, &v155, 0) )
              {
                if ( v129 > 0x40 )
                {
                  *v128 = v102;
                  memset(v128 + 1, 0, 8 * (unsigned int)(((unsigned __int64)v129 + 63) >> 6) - 8);
                }
                else
                {
                  v90 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v129;
                  if ( !v129 )
                    v90 = 0;
                  v128 = (_QWORD *)(v102 & v90);
                }
              }
              sub_9865C0((__int64)&v130, (__int64)v127);
              sub_C46A40((__int64)&v130, 1);
              v85 = v131;
              v131 = 0;
              v133 = v85;
              v132 = v130;
              sub_9865C0((__int64)&v144, (__int64)&v128);
              sub_9875E0((__int64)&v147, (__int64 *)&v144, (__int64 *)&v132);
              sub_AB2160((__int64)&v155, (__int64)&v134, (__int64)&v147, v9);
              sub_D920D0(&v134, v155.m128i_i64);
              sub_969240(&v156);
              sub_969240(v155.m128i_i64);
              sub_969240(v149);
              sub_969240((__int64 *)&v147);
              sub_969240((__int64 *)&v144);
              sub_969240((__int64 *)&v132);
              sub_969240((__int64 *)&v130);
              sub_969240((__int64 *)&v128);
              sub_969240(v127);
            }
          }
        }
        if ( *(_BYTE *)v110 == 84 )
        {
          sub_D9CB80((__int64)&v155, a1 + 320, (__int64 *)v110, v58, v59, v60);
          if ( LOBYTE(v158[0]) )
          {
            sub_AADB10((__int64)&v144, v122, 0);
            if ( (*(_BYTE *)(v110 + 7) & 0x40) != 0 )
            {
              v74 = *(__int64 **)(v110 - 8);
              v75 = *(_DWORD *)(v110 + 4);
            }
            else
            {
              v75 = *(_DWORD *)(v110 + 4);
              v74 = (__int64 *)(v110 - 32LL * (v75 & 0x7FFFFFF));
            }
            v76 = a4 + 1;
            v105 = a2;
            v77 = v74;
            v101 = &v74[4 * (v75 & 0x7FFFFFF)];
            while ( v77 != v101 )
            {
              v78 = *v77;
              v77 += 4;
              v79 = sub_DD8400(a1, v78);
              v80 = sub_DBB9F0(a1, v79, v4, v76);
              sub_AAF450((__int64)&v147, v80);
              sub_AB3510((__int64)&v155, (__int64)&v144, (__int64)&v147, 0);
              sub_D920D0((__int64 *)&v144, v155.m128i_i64);
              sub_969240(&v156);
              sub_969240(v155.m128i_i64);
              if ( sub_AAF760((__int64)&v144) )
              {
                a2 = v105;
                sub_969240(v149);
                sub_969240((__int64 *)&v147);
                goto LABEL_154;
              }
              sub_969240(v149);
              sub_969240((__int64 *)&v147);
            }
            a2 = v105;
LABEL_154:
            sub_AB2160((__int64)&v155, (__int64)&v134, (__int64)&v144, v9);
            sub_D920D0(&v134, v155.m128i_i64);
            sub_969240(&v156);
            sub_969240(v155.m128i_i64);
            if ( *(_BYTE *)(a1 + 348) )
            {
              v96 = *(_QWORD **)(a1 + 328);
              v97 = &v96[*(unsigned int *)(a1 + 340)];
              v98 = v96;
              if ( v96 != v97 )
              {
                while ( v110 != *v98 )
                {
                  if ( v97 == ++v98 )
                    goto LABEL_160;
                }
                v99 = (unsigned int)(*(_DWORD *)(a1 + 340) - 1);
                *(_DWORD *)(a1 + 340) = v99;
                *v98 = v96[v99];
                ++*(_QWORD *)(a1 + 320);
              }
            }
            else
            {
              v100 = sub_C8CA60(a1 + 320, v110);
              if ( v100 )
              {
                *v100 = -2;
                ++*(_DWORD *)(a1 + 344);
                ++*(_QWORD *)(a1 + 320);
              }
            }
LABEL_160:
            sub_969240(v146);
            sub_969240((__int64 *)&v144);
          }
        }
        if ( sub_988010(v110) )
        {
          v61 = *(_QWORD *)(v110 - 32);
          if ( !v61 || *(_BYTE *)v61 || *(_QWORD *)(v61 + 24) != *(_QWORD *)(v110 + 80) )
LABEL_169:
            BUG();
          if ( *(_DWORD *)(v61 + 36) == 493 )
          {
            sub_9691E0((__int64)&v155, v122, 0, 0, 0);
            sub_AADBC0((__int64)&v147, v155.m128i_i64);
            sub_969240(v155.m128i_i64);
            sub_ABB6C0((__int64)&v155, (__int64)&v134, (__int64)&v147);
            sub_D920D0(&v134, v155.m128i_i64);
            sub_969240(&v156);
            sub_969240(v155.m128i_i64);
            sub_969240(v149);
            sub_969240((__int64 *)&v147);
          }
        }
        v62 = v135;
        v135 = 0;
        v155.m128i_i32[2] = v62;
        v155.m128i_i64[0] = v134;
        v63 = v137;
        v137 = 0;
        LODWORD(v157) = v63;
        v156 = v136;
        v14 = sub_DB0AC0(a1, a2, v4, (__int64)&v155);
        sub_969240(&v156);
        sub_969240(v155.m128i_i64);
        sub_969240(&v142);
        sub_969240(&v140);
        sub_969240(&v139);
        sub_969240(v138);
        if ( v154 )
        {
          v154 = 0;
          if ( v153 > 0x40 && v152 )
            j_j___libc_free_0_0(v152);
          if ( v151 > 0x40 && v150 )
            j_j___libc_free_0_0(v150);
        }
LABEL_29:
        sub_969240(&v136);
        sub_969240(&v134);
        return (__int64)v14;
      default:
        v68 = v135;
        v135 = 0;
        v155.m128i_i32[2] = v68;
        v155.m128i_i64[0] = v134;
        v69 = v137;
        v137 = 0;
        LODWORD(v157) = v69;
        v156 = v136;
LABEL_62:
        v14 = sub_DB0AC0(a1, a2, v4, (__int64)&v155);
        sub_969240(&v156);
        sub_969240(v155.m128i_i64);
        goto LABEL_29;
    }
  }
  return sub_DDFBD0(a1, a2, v4);
}
