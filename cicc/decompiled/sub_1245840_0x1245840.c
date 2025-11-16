// Function: sub_1245840
// Address: 0x1245840
//
__int64 __fastcall sub_1245840(_QWORD **a1, __int64 *a2, unsigned int a3, int *a4, __int64 a5)
{
  __m128i *v7; // rsi
  unsigned __int64 v8; // r15
  unsigned int v9; // r15d
  _QWORD *v11; // rax
  int v12; // eax
  const char *v13; // rax
  _QWORD *v14; // rax
  int v15; // eax
  unsigned int v16; // eax
  unsigned int v17; // eax
  int v18; // eax
  __int64 **v19; // rdx
  __int64 *v20; // rdx
  const char *v21; // r12
  const char *v22; // rbx
  const char *v23; // rdi
  __m128i v24; // rax
  __int64 v25; // r8
  __int64 v26; // r9
  unsigned __int64 v27; // rax
  __int64 v28; // rdx
  const char *v29; // rbx
  const void **v30; // rcx
  __int64 v31; // rsi
  unsigned __int64 v32; // r14
  unsigned __int64 v33; // rax
  unsigned int v34; // eax
  __int64 v35; // rax
  unsigned __int64 v36; // r14
  __int64 v37; // r9
  __int64 v38; // rax
  _QWORD *v39; // r14
  __int64 v40; // r8
  __int64 v41; // rax
  __int64 v42; // rdi
  __int64 v43; // r8
  _QWORD *v44; // r9
  __int64 v45; // rax
  __int64 v46; // r9
  __int64 v47; // r11
  bool v48; // zf
  __int64 v49; // rdx
  char v50; // al
  char v51; // si
  __int64 v52; // rcx
  char v53; // al
  __int64 v54; // rdx
  __int64 v55; // rcx
  unsigned __int64 *v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rcx
  __int64 v59; // rax
  __int64 v60; // r12
  unsigned __int8 *v61; // rbx
  const char *v62; // rax
  const char *v63; // rax
  __int64 v64; // rdx
  _DWORD *v65; // rbx
  __int64 v66; // rax
  char v67; // cl
  _BYTE *v68; // rax
  const char *v69; // rsi
  _QWORD **v70; // rsi
  _QWORD *v71; // rax
  _QWORD **v72; // rdi
  _QWORD *v73; // rdx
  __int32 v74; // eax
  __int64 v75; // rcx
  __int64 v76; // r9
  __int64 v77; // rcx
  __int64 v78; // r9
  __int64 v79; // rcx
  __int64 v80; // r9
  __int64 v81; // rcx
  __int64 v82; // r9
  __int64 v83; // rcx
  __int64 v84; // r9
  __int64 v85; // rcx
  __int64 v86; // r9
  __int64 v87; // rax
  __m128i *v88; // [rsp-8h] [rbp-7F8h]
  char s2; // [rsp+8h] [rbp-7E8h]
  _QWORD *na; // [rsp+10h] [rbp-7E0h]
  size_t n; // [rsp+10h] [rbp-7E0h]
  _QWORD **v92; // [rsp+50h] [rbp-7A0h]
  __int64 v93; // [rsp+58h] [rbp-798h]
  unsigned __int64 v94; // [rsp+70h] [rbp-780h]
  unsigned __int64 v95; // [rsp+70h] [rbp-780h]
  __int64 v96; // [rsp+70h] [rbp-780h]
  int v97; // [rsp+70h] [rbp-780h]
  unsigned __int64 v98; // [rsp+78h] [rbp-778h]
  __int64 v99; // [rsp+78h] [rbp-778h]
  __int64 v100; // [rsp+78h] [rbp-778h]
  __int64 v101; // [rsp+78h] [rbp-778h]
  __int64 v102; // [rsp+78h] [rbp-778h]
  __int64 v103; // [rsp+78h] [rbp-778h]
  bool v105; // [rsp+80h] [rbp-770h]
  const char *v106; // [rsp+80h] [rbp-770h]
  unsigned __int64 v107; // [rsp+80h] [rbp-770h]
  __int64 v108; // [rsp+80h] [rbp-770h]
  __int64 v109; // [rsp+80h] [rbp-770h]
  unsigned int v110; // [rsp+80h] [rbp-770h]
  unsigned __int8 v111; // [rsp+88h] [rbp-768h]
  __int64 v114; // [rsp+98h] [rbp-758h]
  unsigned __int8 v115; // [rsp+ABh] [rbp-745h] BYREF
  char v116; // [rsp+ACh] [rbp-744h] BYREF
  unsigned __int8 v117; // [rsp+ADh] [rbp-743h] BYREF
  __int16 v118; // [rsp+AEh] [rbp-742h] BYREF
  int v119; // [rsp+B0h] [rbp-740h] BYREF
  int v120; // [rsp+B4h] [rbp-73Ch] BYREF
  int v121; // [rsp+B8h] [rbp-738h] BYREF
  int v122; // [rsp+BCh] [rbp-734h] BYREF
  int v123; // [rsp+C0h] [rbp-730h] BYREF
  unsigned int v124; // [rsp+C4h] [rbp-72Ch] BYREF
  __int64 *v125; // [rsp+C8h] [rbp-728h] BYREF
  __m128i *v126; // [rsp+D0h] [rbp-720h] BYREF
  __int64 v127; // [rsp+D8h] [rbp-718h] BYREF
  __int64 v128; // [rsp+E0h] [rbp-710h] BYREF
  __int64 v129; // [rsp+E8h] [rbp-708h] BYREF
  __int64 v130; // [rsp+F0h] [rbp-700h] BYREF
  unsigned __int64 v131; // [rsp+F8h] [rbp-6F8h] BYREF
  char *v132[2]; // [rsp+100h] [rbp-6F0h] BYREF
  __int64 v133; // [rsp+110h] [rbp-6E0h]
  const void *v134; // [rsp+120h] [rbp-6D0h] BYREF
  _BYTE *v135; // [rsp+128h] [rbp-6C8h]
  _BYTE *v136; // [rsp+130h] [rbp-6C0h]
  __m128i *v137; // [rsp+140h] [rbp-6B0h] BYREF
  size_t v138; // [rsp+148h] [rbp-6A8h]
  _QWORD v139[2]; // [rsp+150h] [rbp-6A0h] BYREF
  _QWORD *v140; // [rsp+160h] [rbp-690h] BYREF
  __int64 v141; // [rsp+168h] [rbp-688h]
  _QWORD v142[2]; // [rsp+170h] [rbp-680h] BYREF
  _QWORD *v143; // [rsp+180h] [rbp-670h] BYREF
  __int64 v144; // [rsp+188h] [rbp-668h]
  _QWORD v145[2]; // [rsp+190h] [rbp-660h] BYREF
  _QWORD *v146; // [rsp+1A0h] [rbp-650h] BYREF
  __int64 v147; // [rsp+1A8h] [rbp-648h]
  _QWORD v148[2]; // [rsp+1B0h] [rbp-640h] BYREF
  __int64 v149[4]; // [rsp+1C0h] [rbp-630h] BYREF
  __int64 v150[4]; // [rsp+1E0h] [rbp-610h] BYREF
  __m128i v151[2]; // [rsp+200h] [rbp-5F0h] BYREF
  __int16 v152; // [rsp+220h] [rbp-5D0h]
  __m128i v153[2]; // [rsp+230h] [rbp-5C0h] BYREF
  char v154; // [rsp+250h] [rbp-5A0h]
  char v155; // [rsp+251h] [rbp-59Fh]
  __m128i v156[3]; // [rsp+260h] [rbp-590h] BYREF
  __m128i v157[2]; // [rsp+290h] [rbp-560h] BYREF
  char v158; // [rsp+2B0h] [rbp-540h]
  char v159; // [rsp+2B1h] [rbp-53Fh]
  __m128i v160[3]; // [rsp+2C0h] [rbp-530h] BYREF
  __m128i v161[2]; // [rsp+2F0h] [rbp-500h] BYREF
  __int16 v162; // [rsp+310h] [rbp-4E0h]
  __m128i v163[3]; // [rsp+320h] [rbp-4D0h] BYREF
  __m128i v164[2]; // [rsp+350h] [rbp-4A0h] BYREF
  char v165; // [rsp+370h] [rbp-480h]
  char v166; // [rsp+371h] [rbp-47Fh]
  __m128i v167[3]; // [rsp+380h] [rbp-470h] BYREF
  __m128i v168[2]; // [rsp+3B0h] [rbp-440h] BYREF
  __int16 v169; // [rsp+3D0h] [rbp-420h]
  __m128i v170[3]; // [rsp+3E0h] [rbp-410h] BYREF
  __m128i v171[2]; // [rsp+410h] [rbp-3E0h] BYREF
  char v172; // [rsp+430h] [rbp-3C0h]
  char v173; // [rsp+431h] [rbp-3BFh]
  _BYTE *v174; // [rsp+440h] [rbp-3B0h] BYREF
  __int64 v175; // [rsp+448h] [rbp-3A8h]
  _BYTE v176[64]; // [rsp+450h] [rbp-3A0h] BYREF
  __int64 *v177; // [rsp+490h] [rbp-360h] BYREF
  _BYTE *v178; // [rsp+498h] [rbp-358h]
  __int64 v179; // [rsp+4A0h] [rbp-350h]
  _BYTE v180[72]; // [rsp+4A8h] [rbp-348h] BYREF
  __int64 *v181; // [rsp+4F0h] [rbp-300h] BYREF
  _BYTE *v182; // [rsp+4F8h] [rbp-2F8h]
  __int64 v183; // [rsp+500h] [rbp-2F0h]
  _BYTE v184[72]; // [rsp+508h] [rbp-2E8h] BYREF
  __m128i v185; // [rsp+550h] [rbp-2A0h] BYREF
  int v186; // [rsp+560h] [rbp-290h]
  __int64 v187; // [rsp+568h] [rbp-288h]
  _QWORD v188[2]; // [rsp+570h] [rbp-280h] BYREF
  char v189; // [rsp+580h] [rbp-270h] BYREF
  char *v190; // [rsp+590h] [rbp-260h]
  __int64 v191; // [rsp+598h] [rbp-258h]
  char v192; // [rsp+5A0h] [rbp-250h] BYREF
  __int64 v193; // [rsp+5B0h] [rbp-240h]
  int v194; // [rsp+5B8h] [rbp-238h]
  char v195; // [rsp+5BCh] [rbp-234h]
  _QWORD v196[5]; // [rsp+5C0h] [rbp-230h] BYREF
  char v197; // [rsp+5E8h] [rbp-208h]
  const char *v198; // [rsp+5F0h] [rbp-200h] BYREF
  __int64 v199; // [rsp+5F8h] [rbp-1F8h]
  _BYTE v200[16]; // [rsp+600h] [rbp-1F0h] BYREF
  char v201; // [rsp+610h] [rbp-1E0h]
  char v202; // [rsp+611h] [rbp-1DFh]

  v7 = (__m128i *)&v119;
  v8 = (unsigned __int64)a1[29];
  v111 = a3;
  v177 = (__int64 *)*a1[43];
  v178 = v180;
  v179 = 0x800000000LL;
  v125 = 0;
  if ( (unsigned __int8)sub_120C500((__int64)a1, (__int64)&v119, &v116, &v120, &v121, &v115) )
    goto LABEL_2;
  v7 = (__m128i *)&v122;
  if ( (unsigned __int8)sub_120C5E0((__int64)a1, &v122)
    || (v7 = (__m128i *)&v177, (unsigned __int8)sub_1218580((__int64)a1, &v177, 0))
    || (v11 = a1[29],
        v7 = (__m128i *)&v125,
        v202 = 1,
        v94 = (unsigned __int64)v11,
        v198 = "expected type",
        v201 = 3,
        (unsigned __int8)sub_12190A0((__int64)a1, &v125, (int *)&v198, 1)) )
  {
LABEL_2:
    v9 = 1;
    goto LABEL_3;
  }
  switch ( v119 )
  {
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 7:
    case 8:
      if ( !(_BYTE)a3 )
      {
        v202 = 1;
        v13 = "invalid linkage for function declaration";
        goto LABEL_15;
      }
      v12 = v120;
      if ( v119 != 7 )
        goto LABEL_19;
      if ( v120 )
        goto LABEL_27;
      goto LABEL_13;
    case 6:
    case 10:
      v202 = 1;
      v13 = "invalid function linkage type";
      goto LABEL_15;
    case 9:
      if ( !(_BYTE)a3 )
        goto LABEL_18;
      v7 = (__m128i *)v8;
      v202 = 1;
      v9 = a3;
      v198 = "invalid linkage for function definition";
      v201 = 3;
      sub_11FD800((__int64)(a1 + 22), (unsigned __int64)v7, (__int64)&v198, 1);
      goto LABEL_3;
    default:
LABEL_18:
      v12 = v120;
LABEL_19:
      if ( v119 != 8 )
        goto LABEL_20;
      if ( v12 )
      {
LABEL_27:
        v202 = 1;
        v13 = "symbol with local linkage must have default visibility";
        goto LABEL_15;
      }
LABEL_13:
      if ( v121 )
      {
        v202 = 1;
        v13 = "symbol with local linkage cannot have a DLL storage class";
LABEL_15:
        v7 = (__m128i *)v8;
        v198 = v13;
        v201 = 3;
        sub_11FD800((__int64)(a1 + 22), v8, (__int64)&v198, 1);
        goto LABEL_2;
      }
LABEL_20:
      v9 = sub_BCB3E0((__int64)v125);
      if ( !(_BYTE)v9 )
      {
        v7 = (__m128i *)v94;
        v202 = 1;
        v9 = 1;
        v198 = "invalid function return type";
        v201 = 3;
        sub_11FD800((__int64)(a1 + 22), v94, (__int64)&v198, 1);
        goto LABEL_3;
      }
      v14 = a1[29];
      LOBYTE(v139[0]) = 0;
      v138 = 0;
      v98 = (unsigned __int64)v14;
      v137 = (__m128i *)v139;
      v15 = *((_DWORD *)a1 + 60);
      if ( v15 == 508 )
      {
        sub_2240AE0(&v137, a1 + 31);
      }
      else
      {
        if ( v15 != 503 )
        {
          v7 = (__m128i *)v98;
          v202 = 1;
          v198 = "expected function name";
          v201 = 3;
          sub_11FD800((__int64)(a1 + 22), v98, (__int64)&v198, 1);
          goto LABEL_24;
        }
        v16 = *((_DWORD *)a1 + 70);
        *a4 = v16;
        v17 = sub_120EA00((__int64)a1, v98, (__int64)"function", 8, (__int64)"@", 1, *((_DWORD *)a1 + 306), v16);
        v7 = v88;
        if ( (_BYTE)v17 )
        {
          v9 = v17;
          goto LABEL_24;
        }
      }
      v93 = (__int64)(a1 + 22);
      v18 = sub_1205200((__int64)(a1 + 22));
      *((_DWORD *)a1 + 60) = v18;
      if ( v18 != 12 )
      {
        v7 = (__m128i *)a1[29];
        v202 = 1;
        v198 = "expected '(' in function argument list";
        v201 = 3;
        sub_11FD800(v93, (unsigned __int64)v7, (__int64)&v198, 1);
        goto LABEL_24;
      }
      v7 = (__m128i *)&v198;
      v19 = (__int64 **)a1[43];
      v198 = v200;
      v199 = 0x800000000LL;
      v20 = *v19;
      v183 = 0x800000000LL;
      v140 = v142;
      v181 = v20;
      v143 = v145;
      v182 = v184;
      v118 = 0;
      v132[0] = 0;
      v132[1] = 0;
      v133 = 0;
      v126 = 0;
      v141 = 0;
      LOBYTE(v142[0]) = 0;
      v144 = 0;
      LOBYTE(v145[0]) = 0;
      v146 = v148;
      v147 = 0;
      LOBYTE(v148[0]) = 0;
      v123 = 0;
      v124 = 0;
      v127 = 0;
      v128 = 0;
      v129 = 0;
      if ( (unsigned __int8)sub_12186F0((__int64)a1, (__int64)&v198, a5, &v117) )
        goto LABEL_49;
      v7 = (__m128i *)&v123;
      if ( (unsigned __int8)sub_120A6C0((__int64)a1, &v123) )
        goto LABEL_49;
      v7 = (__m128i *)&v124;
      if ( (unsigned __int8)sub_1212650((__int64)a1, &v124, *((_DWORD *)a1[43] + 80)) )
        goto LABEL_49;
      v7 = (__m128i *)&v181;
      if ( (unsigned __int8)sub_1218010((__int64)a1, &v181, (__int64)v132, 0, (unsigned __int64 *)&v126) )
        goto LABEL_49;
      if ( *((_DWORD *)a1 + 60) == 95 )
      {
        if ( (unsigned __int8)sub_1205540((__int64)a1) )
        {
          v7 = (__m128i *)&v140;
          if ( (unsigned __int8)sub_120B3D0((__int64)a1, (__int64)&v140) )
            goto LABEL_49;
        }
      }
      if ( *((_DWORD *)a1 + 60) == 96 )
      {
        if ( (unsigned __int8)sub_1205540((__int64)a1) )
        {
          v7 = (__m128i *)&v143;
          if ( (unsigned __int8)sub_120B3D0((__int64)a1, (__int64)&v143) )
            goto LABEL_49;
        }
      }
      v7 = v137;
      if ( (unsigned __int8)sub_121D100((__int64)a1, v137, v138, &v130) )
        goto LABEL_49;
      v7 = (__m128i *)&v118;
      if ( (unsigned __int8)sub_120CD10((__int64)a1, &v118, 0) )
        goto LABEL_49;
      if ( *((_DWORD *)a1 + 60) == 104 )
      {
        if ( (unsigned __int8)sub_1205540((__int64)a1) )
        {
          v7 = (__m128i *)&v146;
          if ( (unsigned __int8)sub_120B3D0((__int64)a1, (__int64)&v146) )
            goto LABEL_49;
        }
      }
      if ( *((_DWORD *)a1 + 60) == 105 )
      {
        if ( (unsigned __int8)sub_1205540((__int64)a1) )
        {
          v7 = (__m128i *)&v127;
          if ( (unsigned __int8)sub_1224A40(a1, &v127) )
            goto LABEL_49;
        }
      }
      if ( *((_DWORD *)a1 + 60) == 106 )
      {
        if ( (unsigned __int8)sub_1205540((__int64)a1) )
        {
          v7 = (__m128i *)&v128;
          if ( (unsigned __int8)sub_1224A40(a1, &v128) )
            goto LABEL_49;
        }
      }
      if ( *((_DWORD *)a1 + 60) == 371 )
      {
        if ( (unsigned __int8)sub_1205540((__int64)a1) )
        {
          v7 = (__m128i *)&v129;
          if ( (unsigned __int8)sub_1224A40(a1, &v129) )
            goto LABEL_49;
        }
      }
      v105 = sub_A75040((__int64)&v181, 4);
      if ( v105 )
      {
        v7 = v126;
        v185.m128i_i64[0] = (__int64)"'builtin' attribute not valid on function";
        LOWORD(v188[0]) = 259;
        sub_11FD800(v93, (unsigned __int64)v126, (__int64)&v185, 1);
        v9 = v105;
        goto LABEL_49;
      }
      v24.m128i_i64[0] = sub_A74DF0((__int64)&v181, 86);
      v185 = v24;
      if ( v24.m128i_i8[8] && v185.m128i_i64[0] )
      {
        _BitScanReverse64(&v27, v185.m128i_u64[0]);
        HIBYTE(v118) = 1;
        LOBYTE(v118) = 63 - (v27 ^ 0x3F);
        sub_A77390((__int64)&v181, 86);
      }
      s2 = a3;
      v174 = v176;
      v28 = 56LL * (unsigned int)v199;
      v134 = 0;
      v175 = 0x800000000LL;
      v135 = 0;
      v29 = v198;
      v136 = 0;
      v106 = &v198[v28];
      v30 = &v134;
      while ( v29 != v106 )
      {
        if ( v135 == v136 )
        {
          sub_918210((__int64)&v134, v135, (_QWORD *)v29 + 1);
        }
        else
        {
          if ( v135 )
          {
            v28 = *((_QWORD *)v29 + 1);
            *(_QWORD *)v135 = v28;
          }
          v135 += 8;
        }
        v31 = *((_QWORD *)v29 + 2);
        v29 += 56;
        sub_1212C70((__int64)&v174, v31, v28, (__int64)v30, v25, v26);
      }
      na = v174;
      v107 = (unsigned int)v175;
      v32 = sub_A7A280(*a1, (__int64)&v177);
      v33 = sub_A7A280(*a1, (__int64)&v181);
      v131 = sub_A78180(*a1, v33, v32, na, v107);
      v34 = sub_A74710(&v131, 1, 85);
      if ( (_BYTE)v34 && *((_BYTE *)v125 + 8) != 7 )
      {
        v7 = (__m128i *)v94;
        v9 = v34;
        v185.m128i_i64[0] = (__int64)"functions with 'sret' argument must return void";
        LOWORD(v188[0]) = 259;
        sub_11FD800(v93, v94, (__int64)&v185, 1);
        goto LABEL_90;
      }
      v95 = sub_BCF480(v125, v134, (v135 - (_BYTE *)v134) >> 3, v117);
      v35 = sub_BCE3C0(*a1, v124);
      v36 = v138;
      v37 = v35;
      *a2 = 0;
      if ( v36 )
      {
        v108 = v35;
        v38 = sub_1212F00((__int64)(a1 + 137), (__int64)&v137);
        if ( (_QWORD **)v38 != a1 + 138 )
        {
          v39 = *(_QWORD **)(v38 + 64);
          v40 = v39[1];
          if ( v108 != v40 )
          {
            v96 = v38;
            sub_1207630(v168[0].m128i_i64, v40);
            sub_1207630(v163[0].m128i_i64, v108);
            sub_8FD6D0((__int64)v160, "invalid forward reference to function '", &v137);
            sub_94F930(v161, (__int64)v160, "' with wrong type: expected '");
            sub_8FD5D0(v164, (__int64)v161, v163);
            sub_94F930(v167, (__int64)v164, "' but was '");
            sub_8FD5D0(v170, (__int64)v167, v168);
            sub_94F930(v171, (__int64)v170, "'");
            LOWORD(v188[0]) = 260;
            v185.m128i_i64[0] = (__int64)v171;
            v7 = *(__m128i **)(v96 + 72);
            sub_11FD800(v93, (unsigned __int64)v7, (__int64)&v185, 1);
            sub_2240A30(v171);
            sub_2240A30(v170);
            sub_2240A30(v167);
            sub_2240A30(v164);
            sub_2240A30(v161);
            sub_2240A30(v160);
            sub_2240A30(v163);
            sub_2240A30(v168);
            goto LABEL_90;
          }
          v41 = sub_220F330(v38, a1 + 138);
          v42 = *(_QWORD *)(v41 + 32);
          v43 = v41;
          if ( v42 != v41 + 48 )
          {
            v109 = v41;
            j_j___libc_free_0(v42, *(_QWORD *)(v41 + 48) + 1LL);
            v43 = v109;
          }
          j_j___libc_free_0(v43, 80);
          a1[142] = (_QWORD *)((char *)a1[142] - 1);
          goto LABEL_104;
        }
        v68 = sub_BA8CB0((__int64)a1[43], (__int64)v137, v36);
        *a2 = (__int64)v68;
        if ( v68 )
        {
          v69 = "invalid redefinition of function '";
          goto LABEL_149;
        }
        if ( sub_BA8B30((__int64)a1[43], (__int64)v137, v138) )
        {
          v69 = "redefinition of function '@";
LABEL_149:
          sub_8FD6D0((__int64)v170, v69, &v137);
          sub_94F930(v171, (__int64)v170, "'");
          v185.m128i_i64[0] = (__int64)v171;
          v7 = (__m128i *)v98;
          LOWORD(v188[0]) = 260;
LABEL_126:
          sub_11FD800(v93, (unsigned __int64)v7, (__int64)&v185, 1);
          sub_2240A30(v171);
          sub_2240A30(v170);
          goto LABEL_90;
        }
LABEL_159:
        v39 = 0;
        goto LABEL_104;
      }
      if ( *a4 == -1 )
        *a4 = *((_DWORD *)a1 + 306);
      v70 = a1 + 144;
      v71 = a1[145];
      v72 = a1 + 144;
      while ( v71 )
      {
        v73 = (_QWORD *)v71[3];
        if ( *((_DWORD *)v71 + 8) >= (unsigned int)*a4 )
        {
          v73 = (_QWORD *)v71[2];
          v72 = (_QWORD **)v71;
        }
        v71 = v73;
      }
      if ( v70 == v72 || (unsigned int)*a4 < *((_DWORD *)v72 + 8) )
        goto LABEL_159;
      v39 = v72[5];
      if ( v37 != v39[1] )
      {
        v173 = 1;
        v171[0].m128i_i64[0] = (__int64)"'";
        v172 = 3;
        v114 = v37;
        sub_1207630(v150, v39[1]);
        v164[0].m128i_i64[0] = (__int64)"' but was '";
        v168[0].m128i_i64[0] = (__int64)v150;
        v169 = 260;
        v166 = 1;
        v165 = 3;
        sub_1207630(v149, v114);
        v162 = 260;
        v157[0].m128i_i64[0] = (__int64)"' disagree: expected '";
        v74 = *a4;
        v161[0].m128i_i64[0] = (__int64)v149;
        v159 = 1;
        v151[0].m128i_i32[0] = v74;
        v153[0].m128i_i64[0] = (__int64)"type of definition and forward reference of '@";
        v158 = 3;
        v152 = 265;
        v155 = 1;
        v154 = 3;
        sub_9C6370(v156, v153, v151, v75, (__int64)v156, v76);
        sub_9C6370(v160, v156, v157, v77, (__int64)v156, v78);
        sub_9C6370(v163, v160, v161, v79, (__int64)v163, v80);
        sub_9C6370(v167, v163, v164, v81, (__int64)v163, v82);
        sub_9C6370(v170, v167, v168, v83, (__int64)v170, v84);
        sub_9C6370(&v185, v170, v171, v85, (__int64)v170, v86);
        v7 = (__m128i *)v98;
        sub_11FD800(v93, v98, (__int64)&v185, 1);
        sub_2240A30(v149);
        sub_2240A30(v150);
        goto LABEL_90;
      }
      v87 = sub_220F330(v72, v70);
      j_j___libc_free_0(v87, 56);
      a1[148] = (_QWORD *)((char *)a1[148] - 1);
LABEL_104:
      v44 = a1[43];
      LOWORD(v188[0]) = 260;
      v99 = (__int64)v44;
      v110 = v124;
      v185.m128i_i64[0] = (__int64)&v137;
      v45 = sub_BD2DA0(136);
      v46 = v99;
      v47 = v45;
      if ( v45 )
      {
        v100 = v45;
        sub_B2C3B0(v45, v95, 0, v110, (__int64)&v185, v46);
        v47 = v100;
      }
      v48 = v138 == 0;
      *a2 = v47;
      if ( v48 )
        sub_1243C70((__int64)(a1 + 149), *a4, v47);
      v49 = *a2;
      v50 = v119;
      v51 = v119 & 0xF;
      if ( (unsigned int)(v119 - 7) > 1 )
      {
        v67 = v51 | *(_BYTE *)(v49 + 32) & 0xF0;
        *(_BYTE *)(v49 + 32) = v67;
        if ( (v50 & 0xFu) - 7 > 1 && ((v67 & 0x30) == 0 || v51 == 9) )
          goto LABEL_111;
      }
      else
      {
        *(_WORD *)(v49 + 32) = *(_WORD *)(v49 + 32) & 0xFCC0 | v119 & 0xF;
      }
      *(_BYTE *)(v49 + 33) |= 0x40u;
LABEL_111:
      if ( v115 )
        *(_BYTE *)(*a2 + 33) |= 0x40u;
      v52 = *a2;
      v53 = (16 * (v120 & 3)) | *(_BYTE *)(*a2 + 32) & 0xCF;
      *(_BYTE *)(*a2 + 32) = v53;
      if ( (v53 & 0xFu) - 7 <= 1 || (v53 & 0x30) != 0 && (v53 & 0xF) != 9 )
        *(_BYTE *)(v52 + 33) |= 0x40u;
      *(_BYTE *)(*a2 + 33) = v121 & 3 | *(_BYTE *)(*a2 + 33) & 0xFC;
      *(_WORD *)(*a2 + 2) = (16 * v122) | *(_WORD *)(*a2 + 2) & 0xC00F;
      *(_QWORD *)(*a2 + 120) = v131;
      v48 = HIBYTE(v118) == 0;
      *(_BYTE *)(*a2 + 32) = ((_BYTE)v123 << 6) | *(_BYTE *)(*a2 + 32) & 0x3F;
      if ( !v48 )
        sub_B2F770(*a2, v118);
      sub_B31A00(*a2, (__int64)v140, v141);
      sub_B30D10(*a2, (__int64)v143, v144);
      sub_B2F990(*a2, v130, v54, v55);
      sub_B2E8C0(*a2, v129);
      if ( v147 )
      {
        v103 = *a2;
        sub_2241BD0(&v185, &v146);
        sub_B2EBE0(v103, &v185);
        sub_2240A30(&v185);
      }
      sub_B2E9C0(*a2, v127);
      sub_B2EAD0(*a2, v128);
      v185.m128i_i64[0] = *a2;
      v56 = sub_121BCE0(a1 + 179, (unsigned __int64 *)&v185);
      v7 = (__m128i *)v132;
      sub_1205F70((__int64)v56, v132);
      v59 = *a2;
      if ( (*(_BYTE *)(*a2 + 2) & 1) != 0 )
      {
        v102 = *a2;
        sub_B2C6D0(*a2, (__int64)v132, v57, v58);
        v59 = v102;
      }
      v92 = a1;
      v60 = 0;
      v97 = v199;
      v61 = *(unsigned __int8 **)(v59 + 96);
      while ( v97 != (_DWORD)v60 )
      {
        v101 = 56 * v60;
        v62 = &v198[56 * v60];
        if ( *((_QWORD *)v62 + 4) )
        {
          LOWORD(v188[0]) = 260;
          v185.m128i_i64[0] = (__int64)(v62 + 24);
          sub_BD6B50(v61, (const char **)&v185);
          v7 = *(__m128i **)&v198[v101 + 24];
          n = *(_QWORD *)&v198[v101 + 32];
          v63 = sub_BD5D20((__int64)v61);
          if ( n != v64 || n && memcmp(v63, v7, n) )
          {
            sub_8FD6D0((__int64)v170, "redefinition of argument '%", &v198[v101 + 24]);
            sub_94F930(v171, (__int64)v170, "'");
            LOWORD(v188[0]) = 260;
            v185.m128i_i64[0] = (__int64)v171;
            v7 = *(__m128i **)&v198[56 * v60];
            goto LABEL_126;
          }
        }
        v61 += 40;
        ++v60;
      }
      if ( v39 )
      {
        v7 = (__m128i *)*a2;
        sub_BD84D0((__int64)v39, *a2);
        sub_B30810(v39);
      }
      if ( s2 )
      {
        v9 = 0;
      }
      else
      {
        v185.m128i_i32[0] = 0;
        v188[0] = &v189;
        v185.m128i_i64[1] = 0;
        v187 = 0;
        v188[1] = 0;
        v189 = 0;
        v190 = &v192;
        v191 = 0;
        v192 = 0;
        v194 = 1;
        v193 = 0;
        v195 = 0;
        v65 = sub_C33320();
        sub_C3B1B0((__int64)v171, 0.0);
        sub_C407B0(v196, v171[0].m128i_i64, v65);
        sub_C338F0((__int64)v171);
        v196[4] = 0;
        v197 = 0;
        if ( v138 )
        {
          v185.m128i_i32[0] = 3;
          sub_2240AE0(v188, &v137);
        }
        else
        {
          v185.m128i_i32[0] = 1;
          v186 = *a4;
        }
        v7 = &v185;
        v66 = sub_1213420((__int64)(v92 + 160), (__int64)&v185);
        if ( (_QWORD **)v66 != v92 + 161 )
        {
          v173 = 1;
          v171[0].m128i_i64[0] = (__int64)"cannot take blockaddress inside a declaration";
          v172 = 3;
          v7 = *(__m128i **)(v66 + 40);
          sub_11FD800(v93, (unsigned __int64)v7, (__int64)v171, 1);
          v111 = v9;
        }
        sub_120A740((__int64)&v185);
        v9 = v111;
      }
LABEL_90:
      if ( v174 != v176 )
        _libc_free(v174, v7);
      if ( v134 )
      {
        v7 = (__m128i *)(v136 - (_BYTE *)v134);
        j_j___libc_free_0(v134, v136 - (_BYTE *)v134);
      }
LABEL_49:
      if ( v146 != v148 )
      {
        v7 = (__m128i *)(v148[0] + 1LL);
        j_j___libc_free_0(v146, v148[0] + 1LL);
      }
      if ( v143 != v145 )
      {
        v7 = (__m128i *)(v145[0] + 1LL);
        j_j___libc_free_0(v143, v145[0] + 1LL);
      }
      if ( v140 != v142 )
      {
        v7 = (__m128i *)(v142[0] + 1LL);
        j_j___libc_free_0(v140, v142[0] + 1LL);
      }
      if ( v132[0] )
      {
        v7 = (__m128i *)(v133 - (unsigned __int64)v132[0]);
        j_j___libc_free_0(v132[0], v133 - (unsigned __int64)v132[0]);
      }
      if ( v182 != v184 )
        _libc_free(v182, v7);
      v21 = v198;
      v22 = &v198[56 * (unsigned int)v199];
      if ( v198 != v22 )
      {
        do
        {
          v22 -= 56;
          v23 = (const char *)*((_QWORD *)v22 + 3);
          if ( v23 != v22 + 40 )
          {
            v7 = (__m128i *)(*((_QWORD *)v22 + 5) + 1LL);
            j_j___libc_free_0(v23, v7);
          }
        }
        while ( v21 != v22 );
        v21 = v198;
      }
      if ( v21 != v200 )
        _libc_free(v21, v7);
LABEL_24:
      if ( v137 != (__m128i *)v139 )
      {
        v7 = (__m128i *)(v139[0] + 1LL);
        j_j___libc_free_0(v137, v139[0] + 1LL);
      }
LABEL_3:
      if ( v178 != v180 )
        _libc_free(v178, v7);
      return v9;
  }
}
