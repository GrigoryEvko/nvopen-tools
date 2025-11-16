// Function: sub_27DEC50
// Address: 0x27dec50
//
__int64 __fastcall sub_27DEC50(
        __int64 a1,
        unsigned __int8 *a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        __int64 a6,
        unsigned __int8 *a7)
{
  __int64 i; // r14
  unsigned __int8 *v8; // r15
  __int64 v11; // rax
  unsigned __int8 **v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rbx
  unsigned __int8 **v17; // rax
  char v19; // dl
  int v20; // edx
  __int64 v21; // rbx
  unsigned int v22; // ecx
  __int64 v23; // rdx
  unsigned __int64 v24; // rdx
  __int64 v25; // rax
  unsigned __int8 **v26; // rax
  unsigned __int8 **v27; // rax
  __int64 v28; // rbx
  __int64 v29; // rdx
  unsigned __int64 v30; // rsi
  unsigned __int8 *v31; // rax
  __int64 v32; // r9
  int v33; // ecx
  unsigned __int64 v34; // rdx
  unsigned __int64 v35; // rsi
  int v36; // ecx
  unsigned __int8 **v37; // rdx
  unsigned __int8 **v38; // rdx
  unsigned int v39; // ebx
  __int64 v41; // r15
  unsigned __int8 *v42; // r12
  unsigned __int64 v43; // rsi
  unsigned __int64 v44; // rdx
  int v45; // ecx
  unsigned __int8 **v46; // rdx
  unsigned __int8 *v47; // r12
  unsigned __int8 *v48; // rax
  __int64 v49; // r9
  __int64 v50; // rdx
  unsigned __int8 *v51; // rax
  unsigned __int8 *v52; // rax
  unsigned __int64 v53; // rdi
  __int64 v54; // r8
  unsigned __int64 v55; // rdx
  unsigned __int8 **v56; // rdx
  unsigned __int8 *v57; // rax
  __int64 v58; // rsi
  __int64 v59; // r13
  __int64 v60; // rcx
  __int64 v61; // r8
  __int64 v62; // r9
  __int64 *v63; // rdi
  unsigned __int8 **v64; // rdx
  unsigned __int8 **v65; // rbx
  __int64 v66; // rax
  char *v67; // r13
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // rcx
  unsigned __int8 **v71; // rdx
  unsigned __int8 *v72; // rsi
  __int64 v73; // rax
  unsigned __int8 **v74; // rdx
  __int64 v75; // rdx
  __int64 v76; // rax
  __int64 v77; // rcx
  int v78; // r9d
  int v79; // eax
  __int64 v80; // rdx
  __int64 v81; // rsi
  __int64 *v82; // rdi
  __int64 *v83; // r13
  unsigned __int8 *v84; // rax
  unsigned __int8 *v85; // r8
  unsigned __int64 v86; // rax
  unsigned __int64 v87; // rcx
  unsigned __int8 **v88; // rax
  __int64 v89; // r9
  unsigned __int8 **v90; // rax
  unsigned __int8 *v91; // rbx
  unsigned __int8 *v92; // rax
  __int64 v93; // rax
  __int64 v94; // rcx
  __int64 v95; // r8
  __int64 v96; // r9
  __int64 v97; // rdx
  __int64 j; // rbx
  __int64 *v99; // rax
  __int64 *v100; // rdx
  __int64 v101; // r11
  __int64 *v102; // rdi
  unsigned __int8 *v103; // rax
  __int64 v104; // r8
  __int64 v105; // r9
  char v106; // al
  __int64 v107; // rbx
  __int64 v108; // r15
  _BYTE *v109; // r11
  unsigned __int8 *v110; // rdi
  __int64 v111; // rcx
  __int64 v112; // r8
  __int64 v113; // r9
  __int64 *v114; // rdx
  bool v115; // al
  bool v116; // al
  unsigned __int8 v117; // al
  signed __int64 v118; // rax
  unsigned __int8 *v119; // rbx
  _BYTE *v120; // rax
  int v121; // eax
  unsigned int v122; // eax
  __int64 *v123; // rbx
  __int64 v124; // r12
  __int64 v125; // rdi
  __int64 v126; // r8
  __int64 v127; // r9
  __int64 v128; // rcx
  _QWORD *v129; // r15
  bool v130; // zf
  __int64 *v131; // rax
  unsigned __int8 *v132; // r10
  __int64 v133; // r15
  __int64 v134; // r13
  unsigned __int8 *v135; // rax
  unsigned __int8 *v136; // r8
  unsigned __int64 v137; // rax
  unsigned __int64 v138; // rcx
  unsigned __int8 **v139; // rax
  __int64 v140; // r9
  unsigned __int8 **v141; // rax
  __int64 v142; // rax
  __int64 v143; // rax
  __int64 v144; // rdx
  __int64 v145; // rcx
  __int64 v146; // r8
  __int64 v147; // r9
  __int64 *v148; // rsi
  __int32 v149; // eax
  unsigned int v150; // eax
  __int64 v151; // rdi
  __int64 v152; // rdx
  __int64 v153; // rcx
  __int64 v154; // r8
  __int64 v155; // rsi
  __int64 *v156; // rax
  unsigned __int64 v157; // rdx
  unsigned __int64 v158; // rcx
  __int64 v159; // rbx
  __int64 v160; // r9
  __int64 *v161; // r8
  __int64 *v162; // r13
  __int64 *v163; // r15
  unsigned __int64 v164; // rax
  unsigned __int64 v165; // rcx
  __int64 *v166; // rax
  __int64 *v167; // rdi
  __int64 v168; // r8
  __int64 v169; // r15
  __int64 *v170; // rax
  _BYTE *v171; // rdi
  __int64 *v172; // rax
  __int64 v173; // r8
  __int64 v174; // r9
  unsigned __int64 v175; // rax
  __int64 *v176; // rax
  __int64 v177; // r8
  __int64 *v178; // rax
  __int64 v179; // [rsp-8h] [rbp-238h]
  _BYTE *v180; // [rsp+8h] [rbp-228h]
  __int64 v181; // [rsp+10h] [rbp-220h]
  unsigned __int64 v182; // [rsp+18h] [rbp-218h]
  __int64 v183; // [rsp+18h] [rbp-218h]
  int v184; // [rsp+20h] [rbp-210h]
  int v185; // [rsp+20h] [rbp-210h]
  __int64 v186; // [rsp+20h] [rbp-210h]
  __int64 v187; // [rsp+28h] [rbp-208h]
  __int64 v188; // [rsp+28h] [rbp-208h]
  unsigned int v189; // [rsp+30h] [rbp-200h]
  __int64 v190; // [rsp+30h] [rbp-200h]
  _BYTE *v191; // [rsp+38h] [rbp-1F8h]
  __int64 v192; // [rsp+38h] [rbp-1F8h]
  __int64 v193; // [rsp+38h] [rbp-1F8h]
  int v194; // [rsp+38h] [rbp-1F8h]
  __int64 v195; // [rsp+38h] [rbp-1F8h]
  int v196; // [rsp+38h] [rbp-1F8h]
  __int64 v197; // [rsp+40h] [rbp-1F0h]
  const void *v198; // [rsp+40h] [rbp-1F0h]
  unsigned __int8 *v199; // [rsp+40h] [rbp-1F0h]
  int v200; // [rsp+40h] [rbp-1F0h]
  __int64 v201; // [rsp+40h] [rbp-1F0h]
  unsigned __int8 *v203; // [rsp+48h] [rbp-1E8h]
  unsigned __int8 *v204; // [rsp+48h] [rbp-1E8h]
  __int64 *v205; // [rsp+48h] [rbp-1E8h]
  __int64 v206; // [rsp+48h] [rbp-1E8h]
  unsigned __int8 *v207; // [rsp+48h] [rbp-1E8h]
  __int64 v208; // [rsp+48h] [rbp-1E8h]
  __int64 v210; // [rsp+50h] [rbp-1E0h]
  int v211; // [rsp+50h] [rbp-1E0h]
  __int64 v212; // [rsp+50h] [rbp-1E0h]
  __int64 v214; // [rsp+58h] [rbp-1D8h]
  __int64 v215; // [rsp+58h] [rbp-1D8h]
  unsigned __int8 *v216; // [rsp+58h] [rbp-1D8h]
  __int64 v217; // [rsp+58h] [rbp-1D8h]
  unsigned __int8 *v218; // [rsp+58h] [rbp-1D8h]
  __int64 v219; // [rsp+60h] [rbp-1D0h] BYREF
  __int64 v220; // [rsp+68h] [rbp-1C8h] BYREF
  __int64 v221; // [rsp+70h] [rbp-1C0h] BYREF
  unsigned int v222; // [rsp+78h] [rbp-1B8h]
  unsigned __int8 *v223; // [rsp+A0h] [rbp-190h] BYREF
  char *v224; // [rsp+A8h] [rbp-188h]
  unsigned __int64 v225; // [rsp+B0h] [rbp-180h] BYREF
  unsigned int v226; // [rsp+B8h] [rbp-178h]
  char v227; // [rsp+BCh] [rbp-174h]
  char v228; // [rsp+C0h] [rbp-170h] BYREF
  __int64 *k; // [rsp+E0h] [rbp-150h] BYREF
  __int64 v230; // [rsp+E8h] [rbp-148h]
  __int64 v231[16]; // [rsp+F0h] [rbp-140h] BYREF
  __m128i v232; // [rsp+170h] [rbp-C0h] BYREF
  unsigned __int64 v233; // [rsp+180h] [rbp-B0h] BYREF
  __int64 v234; // [rsp+188h] [rbp-A8h]
  __int64 v235; // [rsp+190h] [rbp-A0h]
  __int64 v236; // [rsp+198h] [rbp-98h]
  __int64 v237; // [rsp+1A0h] [rbp-90h]
  __int64 v238; // [rsp+1A8h] [rbp-88h]
  __int16 v239; // [rsp+1B0h] [rbp-80h]

  v8 = a2;
  v11 = sub_AA4E30(a3);
  v15 = a6;
  v16 = v11;
  if ( *(_BYTE *)(a6 + 28) )
  {
    v17 = *(unsigned __int8 ***)(a6 + 8);
    v13 = *(unsigned int *)(a6 + 20);
    v12 = &v17[v13];
    if ( v17 != v12 )
    {
      while ( a2 != *v17 )
      {
        if ( v12 == ++v17 )
          goto LABEL_8;
      }
      goto LABEL_6;
    }
LABEL_8:
    if ( (unsigned int)v13 < *(_DWORD *)(a6 + 16) )
    {
      *(_DWORD *)(a6 + 20) = v13 + 1;
      *v12 = a2;
      ++*(_QWORD *)a6;
      v20 = *a2;
      if ( (unsigned int)(v20 - 12) <= 1 )
        goto LABEL_11;
      goto LABEL_27;
    }
  }
  sub_C8CC70(a6, (__int64)a2, (__int64)v12, v13, v14, a6);
  v15 = a6;
  if ( !v19 )
    goto LABEL_6;
  v20 = *a2;
  if ( (unsigned int)(v20 - 12) > 1 )
  {
LABEL_27:
    if ( a5 == 1 )
    {
      v211 = v15;
      v57 = sub_BD3990(a2, (__int64)a2);
      if ( *v57 == 4 )
      {
        v8 = v57;
        goto LABEL_11;
      }
      LOBYTE(v20) = *a2;
      LODWORD(v15) = v211;
    }
    else if ( (_BYTE)v20 == 17 )
    {
      goto LABEL_11;
    }
    if ( (unsigned __int8)v20 <= 0x1Cu || a3 != *((_QWORD *)a2 + 5) )
    {
      v28 = *(_QWORD *)(a3 + 16);
      if ( v28 )
      {
        while ( 1 )
        {
          v29 = *(_QWORD *)(v28 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v29 - 30) <= 0xAu )
            break;
          v28 = *(_QWORD *)(v28 + 8);
          if ( !v28 )
            goto LABEL_42;
        }
LABEL_35:
        i = *(_QWORD *)(v29 + 40);
        v30 = (unsigned __int64)v8;
        v31 = (unsigned __int8 *)sub_22CF3A0(*(__int64 **)(a1 + 32), (__int64)v8, i, a3, (__int64)a7);
        if ( v31
          || (unsigned __int8)(*v8 - 82) <= 1u
          && (v197 = *((_QWORD *)v8 - 8)) != 0
          && (v191 = (_BYTE *)*((_QWORD *)v8 - 4), *v191 <= 0x15u)
          && (v30 = sub_B53900((__int64)v8),
              (v31 = (unsigned __int8 *)sub_22CF6C0(
                                          *(__int64 **)(a1 + 32),
                                          v30,
                                          v197,
                                          (__int64)v191,
                                          i,
                                          a3,
                                          (__int64)a7)) != 0) )
        {
          v33 = *v31;
          if ( (unsigned int)(v33 - 12) > 1 )
          {
            if ( a5 == 1 )
            {
              v31 = sub_BD3990(v31, v30);
              if ( *v31 != 4 )
                goto LABEL_41;
            }
            else if ( (_BYTE)v33 != 17 )
            {
              goto LABEL_41;
            }
          }
          v34 = *(unsigned int *)(a4 + 8);
          v35 = *(unsigned int *)(a4 + 12);
          v36 = *(_DWORD *)(a4 + 8);
          if ( v34 >= v35 )
          {
            if ( v35 < v34 + 1 )
            {
              v199 = v31;
              sub_C8D5F0(a4, (const void *)(a4 + 16), v34 + 1, 0x10u, v34 + 1, v32);
              v34 = *(unsigned int *)(a4 + 8);
              v31 = v199;
            }
            v38 = (unsigned __int8 **)(*(_QWORD *)a4 + 16 * v34);
            *v38 = v31;
            v38[1] = (unsigned __int8 *)i;
            ++*(_DWORD *)(a4 + 8);
          }
          else
          {
            v37 = (unsigned __int8 **)(*(_QWORD *)a4 + 16 * v34);
            if ( v37 )
            {
              *v37 = v31;
              v37[1] = (unsigned __int8 *)i;
              v36 = *(_DWORD *)(a4 + 8);
            }
            *(_DWORD *)(a4 + 8) = v36 + 1;
          }
        }
LABEL_41:
        while ( 1 )
        {
          v28 = *(_QWORD *)(v28 + 8);
          if ( !v28 )
            break;
          v29 = *(_QWORD *)(v28 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v29 - 30) <= 0xAu )
            goto LABEL_35;
        }
      }
      goto LABEL_42;
    }
    if ( (_BYTE)v20 == 84 )
    {
      LODWORD(i) = 0;
      v210 = 8LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
      v198 = (const void *)(a4 + 16);
      if ( (*((_DWORD *)a2 + 1) & 0x7FFFFFF) == 0 )
        goto LABEL_42;
      v192 = a3;
      v39 = a5;
      v41 = 0;
      i = a4;
      while ( 1 )
      {
        v47 = *(unsigned __int8 **)(*((_QWORD *)a2 - 1) + 4 * v41);
        v48 = sub_27DB860(v47, v39);
        v50 = *((_QWORD *)a2 - 1) + 32LL * *((unsigned int *)a2 + 18);
        if ( v48 )
        {
          v42 = *(unsigned __int8 **)(v50 + v41);
          v43 = *(unsigned int *)(i + 12);
          v44 = *(unsigned int *)(i + 8);
          v45 = *(_DWORD *)(i + 8);
          if ( v44 >= v43 )
          {
            if ( v43 < v44 + 1 )
            {
              v203 = v48;
              sub_C8D5F0(i, v198, v44 + 1, 0x10u, v44 + 1, v49);
              v44 = *(unsigned int *)(i + 8);
              v48 = v203;
            }
            v64 = (unsigned __int8 **)(*(_QWORD *)i + 16 * v44);
            *v64 = v48;
            v64[1] = v42;
            ++*(_DWORD *)(i + 8);
            goto LABEL_65;
          }
          v46 = (unsigned __int8 **)(*(_QWORD *)i + 16 * v44);
          if ( v46 )
          {
            *v46 = v48;
            v46[1] = v42;
            v45 = *(_DWORD *)(i + 8);
          }
        }
        else
        {
          v51 = (unsigned __int8 *)sub_22CF3A0(
                                     *(__int64 **)(a1 + 32),
                                     (__int64)v47,
                                     *(_QWORD *)(v50 + v41),
                                     v192,
                                     (__int64)a7);
          v52 = sub_27DB860(v51, v39);
          if ( !v52 )
            goto LABEL_65;
          v53 = *(unsigned int *)(i + 12);
          v54 = *(_QWORD *)(*((_QWORD *)a2 - 1) + 32LL * *((unsigned int *)a2 + 18) + v41);
          v55 = *(unsigned int *)(i + 8);
          v45 = *(_DWORD *)(i + 8);
          if ( v55 >= v53 )
          {
            if ( v53 < v55 + 1 )
            {
              v190 = *(_QWORD *)(*((_QWORD *)a2 - 1) + 32LL * *((unsigned int *)a2 + 18) + v41);
              v207 = v52;
              sub_C8D5F0(i, v198, v55 + 1, 0x10u, v54, v55 + 1);
              v55 = *(unsigned int *)(i + 8);
              v54 = v190;
              v52 = v207;
            }
            v74 = (unsigned __int8 **)(*(_QWORD *)i + 16 * v55);
            *v74 = v52;
            v74[1] = (unsigned __int8 *)v54;
            ++*(_DWORD *)(i + 8);
            goto LABEL_65;
          }
          v56 = (unsigned __int8 **)(*(_QWORD *)i + 16 * v55);
          if ( v56 )
          {
            *v56 = v52;
            v56[1] = (unsigned __int8 *)v54;
            v45 = *(_DWORD *)(i + 8);
          }
        }
        *(_DWORD *)(i + 8) = v45 + 1;
LABEL_65:
        v41 += 8;
        if ( v210 == v41 )
        {
          a4 = i;
          goto LABEL_42;
        }
      }
    }
    if ( (unsigned int)(unsigned __int8)v20 - 67 <= 0xC )
    {
      v58 = *((_QWORD *)a2 - 4);
      v232.m128i_i64[0] = (__int64)&v233;
      v232.m128i_i64[1] = 0x800000000LL;
      sub_27DEC50(a1, v58, a3, (unsigned int)&v232, a5, v15, (__int64)a7);
      LODWORD(i) = 0;
      if ( !v232.m128i_i32[2] )
        goto LABEL_81;
      v59 = v232.m128i_i64[0] + 16LL * v232.m128i_u32[2];
      i = v232.m128i_i64[0];
      do
      {
        k = (__int64 *)sub_96F480((unsigned int)*v8 - 29, *(_QWORD *)i, *((_QWORD *)v8 + 1), v16);
        if ( k )
          sub_27DB960(a4, (__int64 *)&k, (__int64 *)(i + 8), v60, v61, v62);
        i += 16;
      }
      while ( v59 != i );
LABEL_80:
      LOBYTE(i) = *(_DWORD *)(a4 + 8) != 0;
LABEL_81:
      v63 = (__int64 *)v232.m128i_i64[0];
      if ( (unsigned __int64 *)v232.m128i_i64[0] == &v233 )
        return (unsigned int)i;
      goto LABEL_82;
    }
    if ( (_BYTE)v20 == 96 )
    {
      sub_27DEC50(a1, *((_QWORD *)a2 - 4), a3, a4, a5, v15, (__int64)a7);
      v65 = *(unsigned __int8 ***)a4;
      v66 = 16LL * *(unsigned int *)(a4 + 8);
      v67 = (char *)(*(_QWORD *)a4 + v66);
      v68 = v66 >> 6;
      LODWORD(i) = v68;
      if ( v68 )
      {
        i = (__int64)&v65[8 * v68];
        while ( sub_98ED60(*v65, 0, 0, 0, 0) )
        {
          if ( !sub_98ED60(v65[2], 0, 0, 0, 0) )
          {
            v65 += 2;
            goto LABEL_95;
          }
          if ( !sub_98ED60(v65[4], 0, 0, 0, 0) )
          {
            v65 += 4;
            goto LABEL_95;
          }
          if ( !sub_98ED60(v65[6], 0, 0, 0, 0) )
          {
            v65 += 6;
            goto LABEL_95;
          }
          v65 += 8;
          if ( (unsigned __int8 **)i == v65 )
            goto LABEL_177;
        }
        goto LABEL_95;
      }
LABEL_177:
      v118 = v67 - (char *)v65;
      if ( v67 - (char *)v65 != 32 )
      {
        if ( v118 != 48 )
        {
          if ( v118 != 16 )
          {
LABEL_180:
            v65 = (unsigned __int8 **)v67;
LABEL_100:
            v69 = *(_QWORD *)a4 + 16LL * *(unsigned int *)(a4 + 8) - (_QWORD)v67;
            v70 = v69 >> 4;
            if ( v69 > 0 )
            {
              v71 = v65;
              do
              {
                v72 = *(unsigned __int8 **)v67;
                v71 += 2;
                v67 += 16;
                *(v71 - 2) = v72;
                *(v71 - 1) = (unsigned __int8 *)*((_QWORD *)v67 - 1);
                --v70;
              }
              while ( v70 );
              v65 = (unsigned __int8 **)((char *)v65 + v69);
            }
            v73 = ((__int64)v65 - *(_QWORD *)a4) >> 4;
            *(_DWORD *)(a4 + 8) = v73;
            LOBYTE(i) = (_DWORD)v73 != 0;
            return (unsigned int)i;
          }
LABEL_297:
          if ( sub_98ED60(*v65, 0, 0, 0, 0) )
            goto LABEL_180;
LABEL_95:
          if ( v67 != (char *)v65 )
          {
            for ( i = (__int64)(v65 + 2); v67 != (char *)i; i += 16 )
            {
              if ( sub_98ED60(*(unsigned __int8 **)i, 0, 0, 0, 0) )
              {
                v65 += 2;
                *(v65 - 2) = *(unsigned __int8 **)i;
                *(v65 - 1) = *(unsigned __int8 **)(i + 8);
              }
            }
          }
          goto LABEL_100;
        }
        if ( !sub_98ED60(*v65, 0, 0, 0, 0) )
          goto LABEL_95;
        v65 += 2;
      }
      if ( !sub_98ED60(*v65, 0, 0, 0, 0) )
        goto LABEL_95;
      v65 += 2;
      goto LABEL_297;
    }
    v200 = v15;
    v232.m128i_i64[0] = sub_BCAE30(*((_QWORD *)a2 + 1));
    v232.m128i_i64[1] = v75;
    v76 = sub_CA1930(&v232);
    v78 = v200;
    if ( v76 != 1 )
    {
      v79 = *a2;
      v80 = (unsigned int)(v79 - 42);
      if ( (unsigned int)v80 <= 0x11 )
      {
        if ( !a5 )
        {
          i = *((_QWORD *)a2 - 4);
          if ( *(_BYTE *)i == 17 )
          {
            v81 = *((_QWORD *)a2 - 8);
            v232.m128i_i64[0] = (__int64)&v233;
            v232.m128i_i64[1] = 0x800000000LL;
            sub_27DEC50(a1, v81, a3, (unsigned int)&v232, 0, v200, (__int64)a7);
            v82 = (__int64 *)v232.m128i_i64[0];
            v83 = (__int64 *)v232.m128i_i64[0];
            v214 = v232.m128i_i64[0] + 16LL * v232.m128i_u32[2];
            if ( v214 != v232.m128i_i64[0] )
            {
              do
              {
                v84 = (unsigned __int8 *)sub_96E6C0((unsigned int)*v8 - 29, *v83, (_BYTE *)i, v16);
                v85 = sub_27DB860(v84, 0);
                if ( v85 )
                {
                  v86 = *(unsigned int *)(a4 + 8);
                  v87 = *(unsigned int *)(a4 + 12);
                  if ( v86 >= v87 )
                  {
                    v89 = v83[1];
                    if ( v87 < v86 + 1 )
                    {
                      v193 = v83[1];
                      v204 = v85;
                      sub_C8D5F0(a4, (const void *)(a4 + 16), v86 + 1, 0x10u, (__int64)v85, v89);
                      v86 = *(unsigned int *)(a4 + 8);
                      v89 = v193;
                      v85 = v204;
                    }
                    v90 = (unsigned __int8 **)(*(_QWORD *)a4 + 16 * v86);
                    *v90 = v85;
                    v90[1] = (unsigned __int8 *)v89;
                    ++*(_DWORD *)(a4 + 8);
                  }
                  else
                  {
                    v88 = (unsigned __int8 **)(*(_QWORD *)a4 + 16 * v86);
                    if ( v88 )
                    {
                      *v88 = v85;
                      v88[1] = (unsigned __int8 *)v83[1];
                    }
                    ++*(_DWORD *)(a4 + 8);
                  }
                }
                v83 += 2;
              }
              while ( (__int64 *)v214 != v83 );
              v82 = (__int64 *)v232.m128i_i64[0];
            }
            if ( v82 != (__int64 *)&v233 )
              _libc_free((unsigned __int64)v82);
          }
          goto LABEL_42;
        }
LABEL_6:
        LODWORD(i) = 0;
        return (unsigned int)i;
      }
      if ( (unsigned __int8)(v79 - 82) > 1u )
        goto LABEL_126;
      if ( a5 )
        goto LABEL_6;
      goto LABEL_133;
    }
    if ( a5 )
      goto LABEL_6;
    i = (__int64)&k;
    k = &v219;
    v230 = (__int64)&v220;
    v115 = sub_27DEA50(&k, a2);
    v78 = v200;
    if ( !v115 )
    {
      v232.m128i_i64[0] = (__int64)&v219;
      v232.m128i_i64[1] = (__int64)&v220;
      v116 = sub_27DEB50(&v232, (__int64)a2);
      v78 = v200;
      if ( !v116 )
      {
        v117 = *a2;
        if ( *a2 == 59 )
        {
          if ( (a2[7] & 0x40) != 0 )
            v119 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
          else
            v119 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
          v120 = (_BYTE *)*((_QWORD *)v119 + 4);
          if ( *v120 == 17 )
          {
            LOBYTE(v121) = sub_D94040((__int64)(v120 + 24));
            v78 = v200;
            LODWORD(i) = v121;
            if ( (_BYTE)v121 )
            {
              sub_27DEC50(a1, *(_QWORD *)v119, a3, a4, 0, v200, (__int64)a7);
              v122 = *(_DWORD *)(a4 + 8);
              if ( v122 )
              {
                v123 = *(__int64 **)a4;
                v124 = *(_QWORD *)a4 + 16LL * v122;
                while ( (__int64 *)v124 != v123 )
                {
                  v125 = *v123;
                  v123 += 2;
                  *(v123 - 2) = sub_AD63D0(v125);
                }
                return (unsigned int)i;
              }
              goto LABEL_6;
            }
          }
LABEL_126:
          v194 = v78;
          if ( *a2 == 86 )
          {
            v91 = sub_27DB860(*((unsigned __int8 **)a2 - 8), a5);
            i = (__int64)sub_27DB860(*((unsigned __int8 **)a2 - 4), a5);
            v232.m128i_i64[0] = (__int64)&v233;
            v232.m128i_i64[1] = 0x800000000LL;
            if ( i | (unsigned __int64)v91 )
            {
              if ( (unsigned __int8)sub_27DEC50(a1, *((_QWORD *)a2 - 12), a3, (unsigned int)&v232, 0, v194, (__int64)a7) )
              {
                v128 = v232.m128i_i64[0];
                v129 = (_QWORD *)v232.m128i_i64[0];
                v215 = v232.m128i_i64[0] + 16LL * v232.m128i_u32[2];
                while ( (_QWORD *)v215 != v129 )
                {
                  if ( *(_BYTE *)*v129 == 17 )
                  {
                    v130 = !sub_D94040(*v129 + 24LL);
                    v131 = (__int64 *)i;
                    if ( !v130 )
                      v131 = (__int64 *)v91;
                  }
                  else
                  {
                    v131 = (__int64 *)i;
                    if ( v91 )
                      v131 = (__int64 *)v91;
                  }
                  k = v131;
                  if ( v131 )
                    sub_27DB960(a4, (__int64 *)&k, v129 + 1, v128, v126, v127);
                  v129 += 2;
                }
                goto LABEL_80;
              }
              if ( (unsigned __int64 *)v232.m128i_i64[0] != &v233 )
                _libc_free(v232.m128i_u64[0]);
            }
          }
          v92 = (unsigned __int8 *)sub_22CE0E0(*(__int64 **)(a1 + 32), (__int64)a2, (__int64)a7);
          v223 = sub_27DB860(v92, a5);
          if ( v223 )
          {
            i = (__int64)&k;
            v93 = sub_F92F30(a3);
            v232.m128i_i64[0] = v93;
            for ( j = v97; j != v232.m128i_i64[0]; v93 = v232.m128i_i64[0] )
            {
              k = *(__int64 **)(*(_QWORD *)(v93 + 24) + 40LL);
              sub_27DB960(a4, (__int64 *)&v223, (__int64 *)&k, v94, v95, v96);
              v232.m128i_i64[0] = *(_QWORD *)(v232.m128i_i64[0] + 8);
              sub_D4B000(v232.m128i_i64);
            }
          }
LABEL_42:
          LOBYTE(i) = *(_DWORD *)(a4 + 8) != 0;
          return (unsigned int)i;
        }
        if ( v117 != 82 && v117 != 83 )
          goto LABEL_126;
LABEL_133:
        i = *((_QWORD *)a2 - 8);
        v187 = *((_QWORD *)a2 + 1);
        v195 = *((_QWORD *)a2 - 4);
        v189 = *((_WORD *)a2 + 1) & 0x3F;
        if ( *(_BYTE *)i == 84 )
        {
          if ( a3 != *(_QWORD *)(i + 40) )
            goto LABEL_135;
          v201 = *((_QWORD *)a2 - 8);
        }
        else
        {
          if ( *(_BYTE *)v195 != 84 )
            goto LABEL_135;
          if ( a3 != *(_QWORD *)(v195 + 40) )
            goto LABEL_126;
          v201 = *((_QWORD *)a2 - 4);
        }
        v184 = v78;
        v106 = sub_B19060(a1 + 96, a3, v80, v77);
        v78 = v184;
        if ( !v106 )
        {
          v188 = sub_B43CC0(v201);
          if ( a7 )
            v8 = a7;
          v107 = 0;
          v185 = *(_DWORD *)(v201 + 4) & 0x7FFFFFF;
          v206 = (__int64)v8;
          while ( 1 )
          {
            if ( v185 == (_DWORD)v107 )
              goto LABEL_42;
            v114 = *(__int64 **)(*(_QWORD *)(v201 - 8) + 32LL * *(unsigned int *)(v201 + 72) + 8 * v107);
            k = v114;
            if ( i == v201 )
            {
              v108 = *(_QWORD *)(*(_QWORD *)(i - 8) + 32 * v107);
              v109 = (_BYTE *)sub_BD5BF0(v195, a3, (__int64)v114);
            }
            else
            {
              v108 = sub_BD5BF0(i, a3, (__int64)v114);
              v109 = *(_BYTE **)(*(_QWORD *)(v201 - 8) + 32 * v107);
            }
            v232 = (__m128i)(unsigned __int64)v188;
            v233 = 0;
            v234 = 0;
            v235 = 0;
            v182 = v189 | v182 & 0xFFFFFF0000000000LL;
            v236 = 0;
            v237 = 0;
            v238 = 0;
            v239 = 257;
            v180 = v109;
            v110 = (unsigned __int8 *)sub_10197D0(v182, (_BYTE *)v108, v109, &v232);
            if ( !v110 )
            {
              if ( *v180 > 0x15u || *(_BYTE *)v108 > 0x1Cu && a3 == *(_QWORD *)(v108 + 40) )
                goto LABEL_157;
              v110 = (unsigned __int8 *)sub_22CF6C0(
                                          *(__int64 **)(a1 + 32),
                                          v189,
                                          v108,
                                          (__int64)v180,
                                          (__int64)k,
                                          a3,
                                          v206);
            }
            v232.m128i_i64[0] = (__int64)sub_27DB860(v110, 0);
            if ( v232.m128i_i64[0] )
              sub_27DB960(a4, v232.m128i_i64, (__int64 *)&k, v111, v112, v113);
LABEL_157:
            ++v107;
          }
        }
LABEL_135:
        if ( *(_BYTE *)v195 > 0x15u || (unsigned int)*(unsigned __int8 *)(v187 + 8) - 17 <= 1 )
          goto LABEL_126;
        if ( *(_BYTE *)i <= 0x1Cu || a3 != *(_QWORD *)(i + 40) )
        {
          v99 = (__int64 *)sub_F92F30(a3);
          if ( a7 )
            v8 = a7;
          v205 = v100;
          for ( k = v99; v205 != k; sub_D4B000((__int64 *)&k) )
          {
            v102 = *(__int64 **)(a1 + 32);
            v179 = v101;
            v223 = *(unsigned __int8 **)(k[3] + 40);
            v103 = (unsigned __int8 *)sub_22CF6C0(v102, v189, i, v195, (__int64)v223, a3, (__int64)v8);
            v232.m128i_i64[0] = (__int64)sub_27DB860(v103, 0);
            if ( v232.m128i_i64[0] )
              sub_27DB960(a4, v232.m128i_i64, (__int64 *)&v223, v179, v104, v105);
            k = (__int64 *)k[1];
          }
          goto LABEL_42;
        }
        if ( *(_BYTE *)i != 42
          || *(_BYTE *)v195 != 17
          || (v142 = *(_QWORD *)(i - 64), (v183 = v142) == 0)
          || (v208 = *(_QWORD *)(i - 32), *(_BYTE *)v208 != 17)
          || *(_BYTE *)v142 > 0x1Cu && a3 == *(_QWORD *)(v142 + 40) )
        {
          v232.m128i_i64[0] = (__int64)&v233;
          v232.m128i_i64[1] = 0x800000000LL;
          if ( (a2[7] & 0x40) != 0 )
            v132 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
          else
            v132 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
          LODWORD(i) = a4 + 16;
          sub_27DEC50(a1, *(_QWORD *)v132, a3, (unsigned int)&v232, 0, v78, (__int64)a7);
          v133 = v232.m128i_i64[0];
          v134 = v232.m128i_i64[0] + 16LL * v232.m128i_u32[2];
          while ( v134 != v133 )
          {
            v135 = (unsigned __int8 *)sub_9719A0(v189, *(_BYTE **)v133, v195, v16, 0, 0);
            v136 = sub_27DB860(v135, 0);
            if ( v136 )
            {
              v137 = *(unsigned int *)(a4 + 8);
              v138 = *(unsigned int *)(a4 + 12);
              if ( v137 >= v138 )
              {
                v140 = *(_QWORD *)(v133 + 8);
                if ( v138 < v137 + 1 )
                {
                  v212 = *(_QWORD *)(v133 + 8);
                  v216 = v136;
                  sub_C8D5F0(a4, (const void *)(a4 + 16), v137 + 1, 0x10u, (__int64)v136, v140);
                  v137 = *(unsigned int *)(a4 + 8);
                  v140 = v212;
                  v136 = v216;
                }
                v141 = (unsigned __int8 **)(*(_QWORD *)a4 + 16 * v137);
                *v141 = v136;
                v141[1] = (unsigned __int8 *)v140;
                ++*(_DWORD *)(a4 + 8);
              }
              else
              {
                v139 = (unsigned __int8 **)(*(_QWORD *)a4 + 16 * v137);
                if ( v139 )
                {
                  *v139 = v136;
                  v139[1] = *(unsigned __int8 **)(v133 + 8);
                }
                ++*(_DWORD *)(a4 + 8);
              }
            }
            v133 += 16;
          }
          goto LABEL_80;
        }
        v143 = sub_F92F30(a3);
        if ( a7 )
          i = (__int64)a7;
        v181 = v144;
        v220 = v143;
        v186 = i;
        while ( 1 )
        {
          if ( v181 == v220 )
          {
            LOBYTE(i) = *(_DWORD *)(a4 + 8) != 0;
            return (unsigned int)i;
          }
          v148 = *(__int64 **)(a1 + 32);
          v219 = *(_QWORD *)(*(_QWORD *)(v220 + 24) + 40LL);
          sub_22CF530((__int64)&v223, v148, v183, v219, a3, v186);
          v222 = *(_DWORD *)(v208 + 32);
          i = (__int64)&v221;
          if ( v222 > 0x40 )
            sub_C43780((__int64)&v221, (const void **)(v208 + 24));
          else
            v221 = *(_QWORD *)(v208 + 24);
          sub_AADBC0((__int64)&k, &v221);
          sub_AB4F10((__int64)&v232, (__int64)&v223, (__int64)&k);
          if ( (unsigned int)v224 > 0x40 && v223 )
            j_j___libc_free_0_0((unsigned __int64)v223);
          v223 = (unsigned __int8 *)v232.m128i_i64[0];
          v149 = v232.m128i_i32[2];
          v232.m128i_i32[2] = 0;
          LODWORD(v224) = v149;
          if ( v226 > 0x40 && v225 )
            j_j___libc_free_0_0(v225);
          v225 = v233;
          v150 = v234;
          LODWORD(v234) = 0;
          v226 = v150;
          sub_969240((__int64 *)&v233);
          sub_969240(v232.m128i_i64);
          sub_969240(v231);
          sub_969240((__int64 *)&k);
          sub_969240(&v221);
          sub_AB1A50((__int64)&k, v189, v195 + 24);
          if ( (unsigned __int8)sub_AB1BB0((__int64)&k, (__int64)&v223) )
            break;
          sub_ABB300((__int64)&v232, (__int64)&k);
          LODWORD(i) = sub_AB1BB0((__int64)&v232, (__int64)&v223);
          sub_969240((__int64 *)&v233);
          sub_969240(v232.m128i_i64);
          if ( (_BYTE)i )
          {
            v232.m128i_i64[0] = sub_AD6450(v187);
            goto LABEL_230;
          }
LABEL_231:
          sub_969240(v231);
          sub_969240((__int64 *)&k);
          sub_969240((__int64 *)&v225);
          sub_969240((__int64 *)&v223);
          v220 = *(_QWORD *)(v220 + 8);
          sub_D4B000(&v220);
        }
        v232.m128i_i64[0] = sub_AD6400(v187);
LABEL_230:
        sub_27DB960(a4, v232.m128i_i64, &v219, v145, v146, v147);
        goto LABEL_231;
      }
    }
    v232.m128i_i64[0] = (__int64)&v233;
    k = v231;
    v196 = v78;
    v230 = 0x800000000LL;
    v232.m128i_i64[1] = 0x800000000LL;
    sub_27DEC50(a1, v219, a3, (unsigned int)&k, 0, v78, (__int64)a7);
    sub_27DEC50(a1, v220, a3, (unsigned int)&v232, 0, v196, (__int64)a7);
    if ( !(v232.m128i_i32[2] | (unsigned int)v230) )
    {
      LODWORD(i) = 0;
LABEL_271:
      v167 = (__int64 *)v232.m128i_i64[0];
      goto LABEL_272;
    }
    v151 = *((_QWORD *)a2 + 1);
    if ( (unsigned int)*(unsigned __int8 *)(v151 + 8) - 17 <= 1 )
      v151 = **(_QWORD **)(v151 + 16);
    if ( sub_BCAC40(v151, 1)
      && (*a2 == 58
       || *a2 == 86
       && (v155 = *((_QWORD *)a2 + 1), *(_QWORD *)(*((_QWORD *)v8 - 12) + 8LL) == v155)
       && (v171 = (_BYTE *)*((_QWORD *)v8 - 8), *v171 <= 0x15u)
       && sub_AD7A80(v171, v155, v152, v153, v154)) )
    {
      v172 = (__int64 *)sub_BD5C60((__int64)v8);
      v159 = sub_ACD6D0(v172);
    }
    else
    {
      v156 = (__int64 *)sub_BD5C60((__int64)v8);
      v159 = sub_ACD720(v156);
    }
    v161 = k;
    v223 = 0;
    v224 = &v228;
    v227 = 1;
    v225 = 4;
    v162 = &k[2 * (unsigned int)v230];
    v226 = 0;
    if ( v162 == k )
    {
      v167 = (__int64 *)v232.m128i_i64[0];
      v173 = 16LL * v232.m128i_u32[2];
      v169 = v232.m128i_i64[0] + v173;
      if ( v232.m128i_i64[0] == v232.m128i_i64[0] + v173 )
      {
        LOBYTE(i) = *(_DWORD *)(a4 + 8) != 0;
LABEL_272:
        if ( v167 != (__int64 *)&v233 )
          _libc_free((unsigned __int64)v167);
        v63 = k;
        if ( k == v231 )
          return (unsigned int)i;
LABEL_82:
        _libc_free((unsigned __int64)v63);
        return (unsigned int)i;
      }
    }
    else
    {
      v163 = k;
      i = (__int64)&v221;
      do
      {
        if ( *v163 == v159 || (unsigned int)*(unsigned __int8 *)*v163 - 12 <= 1 )
        {
          v164 = *(unsigned int *)(a4 + 8);
          v165 = *(unsigned int *)(a4 + 12);
          if ( v164 >= v165 )
          {
            v161 = (__int64 *)v163[1];
            if ( v165 < v164 + 1 )
            {
              v218 = (unsigned __int8 *)v163[1];
              sub_C8D5F0(a4, (const void *)(a4 + 16), v164 + 1, 0x10u, (__int64)v161, v160);
              v164 = *(unsigned int *)(a4 + 8);
              v161 = (__int64 *)v218;
            }
            v170 = (__int64 *)(*(_QWORD *)a4 + 16 * v164);
            *v170 = v159;
            v170[1] = (__int64)v161;
            ++*(_DWORD *)(a4 + 8);
          }
          else
          {
            v166 = (__int64 *)(*(_QWORD *)a4 + 16 * v164);
            if ( v166 )
            {
              *v166 = v159;
              v166[1] = v163[1];
            }
            ++*(_DWORD *)(a4 + 8);
          }
          sub_D695C0((__int64)&v221, (__int64)&v223, (__int64 *)v163[1], v165, (__int64)v161, v160);
        }
        v163 += 2;
      }
      while ( v162 != v163 );
      v167 = (__int64 *)v232.m128i_i64[0];
      v168 = 16LL * v232.m128i_u32[2];
      v169 = v232.m128i_i64[0] + v168;
      if ( v232.m128i_i64[0] + v168 == v232.m128i_i64[0] )
      {
LABEL_265:
        LOBYTE(i) = *(_DWORD *)(a4 + 8) != 0;
        if ( !v227 )
        {
          _libc_free((unsigned __int64)v224);
          v167 = (__int64 *)v232.m128i_i64[0];
          goto LABEL_272;
        }
        goto LABEL_271;
      }
    }
    i = (__int64)v167;
    do
    {
      if ( (*(_QWORD *)i == v159 || (unsigned int)**(unsigned __int8 **)i - 12 <= 1)
        && !(unsigned __int8)sub_B19060((__int64)&v223, *(_QWORD *)(i + 8), v157, v158) )
      {
        v175 = *(unsigned int *)(a4 + 8);
        v158 = *(unsigned int *)(a4 + 12);
        if ( v175 >= v158 )
        {
          v157 = v175 + 1;
          v177 = *(_QWORD *)(i + 8);
          if ( v158 < v175 + 1 )
          {
            v217 = *(_QWORD *)(i + 8);
            sub_C8D5F0(a4, (const void *)(a4 + 16), v157, 0x10u, v177, v174);
            v175 = *(unsigned int *)(a4 + 8);
            v177 = v217;
          }
          v178 = (__int64 *)(*(_QWORD *)a4 + 16 * v175);
          *v178 = v159;
          v178[1] = v177;
          ++*(_DWORD *)(a4 + 8);
        }
        else
        {
          v176 = (__int64 *)(*(_QWORD *)a4 + 16 * v175);
          if ( v176 )
          {
            *v176 = v159;
            v157 = *(_QWORD *)(i + 8);
            v176[1] = v157;
          }
          ++*(_DWORD *)(a4 + 8);
        }
      }
      i += 16;
    }
    while ( v169 != i );
    goto LABEL_265;
  }
LABEL_11:
  v21 = *(_QWORD *)(a3 + 16);
  v22 = *(_DWORD *)(a4 + 8);
  if ( v21 )
  {
    while ( 1 )
    {
      v23 = *(_QWORD *)(v21 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v23 - 30) <= 0xAu )
        break;
      v21 = *(_QWORD *)(v21 + 8);
      if ( !v21 )
      {
        LOBYTE(i) = v22 != 0;
        return (unsigned int)i;
      }
    }
LABEL_15:
    i = *(_QWORD *)(v23 + 40);
    v24 = *(unsigned int *)(a4 + 12);
    v25 = v22;
    if ( v22 >= v24 )
    {
      if ( v24 < (unsigned __int64)v22 + 1 )
      {
        sub_C8D5F0(a4, (const void *)(a4 + 16), v22 + 1LL, 0x10u, v22 + 1LL, v15);
        v25 = *(unsigned int *)(a4 + 8);
      }
      v27 = (unsigned __int8 **)(*(_QWORD *)a4 + 16 * v25);
      *v27 = v8;
      v27[1] = (unsigned __int8 *)i;
      v22 = *(_DWORD *)(a4 + 8) + 1;
      *(_DWORD *)(a4 + 8) = v22;
    }
    else
    {
      v26 = (unsigned __int8 **)(*(_QWORD *)a4 + 16LL * v22);
      if ( v26 )
      {
        *v26 = v8;
        v26[1] = (unsigned __int8 *)i;
        v22 = *(_DWORD *)(a4 + 8);
      }
      *(_DWORD *)(a4 + 8) = ++v22;
    }
    while ( 1 )
    {
      v21 = *(_QWORD *)(v21 + 8);
      if ( !v21 )
        break;
      v23 = *(_QWORD *)(v21 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v23 - 30) <= 0xAu )
        goto LABEL_15;
    }
  }
  LOBYTE(i) = v22 != 0;
  return (unsigned int)i;
}
