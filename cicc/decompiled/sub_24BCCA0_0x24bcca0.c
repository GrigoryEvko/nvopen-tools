// Function: sub_24BCCA0
// Address: 0x24bcca0
//
__int64 __fastcall sub_24BCCA0(__int64 *a1, __int64 a2)
{
  bool v3; // zf
  __int64 *v4; // r12
  _QWORD *v5; // r13
  __int64 v6; // rsi
  __int64 v7; // rsi
  unsigned int v8; // r14d
  __int64 v9; // r8
  unsigned __int64 v10; // rbx
  unsigned __int64 v11; // r9
  unsigned __int64 *v12; // rax
  __m128i *v13; // r12
  __int64 v14; // r13
  unsigned __int64 v15; // r14
  const char *v16; // r15
  unsigned __int64 v17; // r13
  __int64 v18; // rax
  __m128i v19; // xmm0
  unsigned __int64 v20; // rdx
  _BYTE *v21; // r15
  const char *v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  __int64 v27; // rax
  unsigned __int64 v28; // rdx
  unsigned __int64 v29; // rax
  __int64 v30; // rsi
  __int64 v31; // rdx
  __int64 v32; // rdi
  __int64 *v33; // rdi
  unsigned __int64 v34; // rdx
  int v35; // ecx
  unsigned __int64 v36; // rdx
  unsigned int v37; // edx
  unsigned __int64 v38; // rax
  __int64 v39; // rdx
  unsigned __int64 v40; // rdx
  bool v41; // cf
  __int64 *v42; // r14
  unsigned __int64 v43; // rcx
  __int64 v44; // rbx
  __int64 v45; // rax
  __int64 v46; // rcx
  __int64 v47; // r12
  __int64 v48; // rax
  __int64 v49; // rcx
  unsigned __int8 *v50; // rdi
  __int64 v51; // rax
  unsigned __int8 *v52; // rax
  __int64 v53; // rax
  __int64 v54; // rdi
  unsigned __int64 v55; // rax
  int v56; // edx
  _QWORD *v57; // rdi
  _QWORD *v58; // rax
  int v59; // edx
  __int64 v60; // r12
  int v61; // r13d
  __int64 v62; // rax
  unsigned __int64 v63; // r12
  _BYTE *v64; // rbx
  __int64 v65; // rdx
  unsigned int v66; // esi
  __int64 v67; // r8
  __int64 v68; // r9
  __int64 v69; // rdx
  unsigned __int64 v70; // r12
  __m128i *v71; // rax
  const __m128i *v72; // rdi
  __m128i *v73; // rcx
  const __m128i *v74; // rbx
  const __m128i *i; // rdx
  __int64 v76; // rcx
  __int64 v77; // rax
  __int64 v78; // r12
  __int64 v79; // rax
  __int64 v80; // r13
  __int64 v81; // r14
  __int64 v82; // rdi
  __int64 v83; // rax
  int v84; // eax
  __int64 v85; // rax
  __int64 v86; // rdx
  unsigned __int64 v87; // rbx
  __int64 v88; // rdx
  __int64 v89; // rax
  _QWORD *v90; // rax
  __int64 v91; // r12
  const char *v92; // r15
  const char *v93; // rbx
  __int64 v94; // rdx
  unsigned int v95; // esi
  int v96; // eax
  int v97; // eax
  int v98; // edx
  __int64 v99; // rax
  __int64 v100; // rdx
  __int64 v101; // rdx
  __m128i *v102; // rsi
  unsigned __int64 v103; // r13
  const __m128i *v104; // r8
  __m128i *v105; // rsi
  __m128i *v106; // rsi
  __int64 *v107; // r12
  __int64 v108; // r13
  __int64 v109; // rax
  __int64 *v110; // rbx
  __int64 v111; // rsi
  const char *v112; // rdi
  char *v113; // r14
  size_t v114; // rax
  __int64 v115; // r13
  __int64 v116; // r13
  __int64 v117; // r13
  char *v118; // rbx
  __int64 v119; // rdx
  __int64 v120; // rcx
  __int64 v121; // r9
  __m128i v122; // xmm4
  __m128i v123; // xmm6
  __int64 v124; // r8
  unsigned __int64 *v125; // rbx
  unsigned __int64 *v126; // r15
  unsigned __int64 v127; // rdi
  unsigned __int64 *v128; // rbx
  unsigned __int64 *v129; // r12
  unsigned __int64 v130; // rdi
  __int64 v131; // rdx
  __int64 v132; // rcx
  __int64 v133; // r8
  __int64 v134; // r9
  __int64 v135; // rdx
  __int64 v136; // rcx
  __int64 v137; // r8
  __int64 v138; // r9
  _QWORD *v139; // rbx
  _QWORD *v140; // r14
  void (__fastcall *v141)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v142; // rax
  __int64 v143; // rcx
  __int64 v144; // rdx
  unsigned int v145; // ecx
  __int64 v146; // rax
  unsigned __int64 v147; // rax
  unsigned __int64 v148; // rcx
  __int64 v149; // rax
  unsigned __int64 v150; // rax
  int v151; // esi
  __int64 v152; // rax
  char v153; // dh
  __int16 v154; // cx
  __int64 v155; // r9
  char v156; // al
  unsigned int v157; // eax
  int v158; // eax
  unsigned int v159; // edx
  __int64 v160; // rax
  __int64 v161; // rdx
  __int64 v162; // rdx
  __int64 v163; // rax
  __int64 v164; // rax
  __int64 v165; // rax
  __int64 v166; // rsi
  __m128i *v167; // r15
  __int64 v168; // r14
  __int64 v169; // rbx
  int v170; // r13d
  __int64 v171; // rax
  unsigned int v172; // eax
  __int64 v173; // r12
  char *v174; // rax
  unsigned __int8 *v175; // rax
  size_t v176; // rdx
  void *v177; // rdi
  size_t v178; // r13
  char *v179; // rax
  __int64 v180; // [rsp-10h] [rbp-D00h]
  __int64 v181; // [rsp-8h] [rbp-CF8h]
  unsigned __int64 v182; // [rsp+8h] [rbp-CE8h]
  unsigned __int64 v183; // [rsp+30h] [rbp-CC0h]
  __int64 v184; // [rsp+58h] [rbp-C98h]
  unsigned __int64 v185; // [rsp+60h] [rbp-C90h]
  unsigned int v186; // [rsp+6Ch] [rbp-C84h]
  char *v187; // [rsp+78h] [rbp-C78h]
  __int64 v188; // [rsp+80h] [rbp-C70h]
  __int64 v189; // [rsp+88h] [rbp-C68h]
  __int64 v190; // [rsp+A0h] [rbp-C50h]
  __int64 v191; // [rsp+A8h] [rbp-C48h]
  __int64 v192; // [rsp+B0h] [rbp-C40h]
  unsigned __int64 v193; // [rsp+B8h] [rbp-C38h]
  __int64 v194; // [rsp+C0h] [rbp-C30h]
  __m128i v195; // [rsp+D0h] [rbp-C20h] BYREF
  void **v196; // [rsp+E0h] [rbp-C10h]
  _BYTE *v197; // [rsp+E8h] [rbp-C08h]
  _BYTE *v198; // [rsp+F0h] [rbp-C00h]
  unsigned __int64 v199; // [rsp+F8h] [rbp-BF8h]
  __m128i *v200; // [rsp+100h] [rbp-BF0h]
  __int64 *v201; // [rsp+108h] [rbp-BE8h]
  unsigned __int64 v202; // [rsp+118h] [rbp-BD8h] BYREF
  __m128i *v203; // [rsp+120h] [rbp-BD0h] BYREF
  __m128i *v204; // [rsp+128h] [rbp-BC8h]
  const __m128i *v205; // [rsp+130h] [rbp-BC0h]
  __m128i *v206; // [rsp+140h] [rbp-BB0h] BYREF
  unsigned int v207; // [rsp+148h] [rbp-BA8h]
  char v208; // [rsp+150h] [rbp-BA0h] BYREF
  __int64 v209[2]; // [rsp+190h] [rbp-B60h] BYREF
  __int64 v210; // [rsp+1A0h] [rbp-B50h] BYREF
  __int64 *v211; // [rsp+1B0h] [rbp-B40h]
  __int64 v212; // [rsp+1C0h] [rbp-B30h] BYREF
  __int64 v213[2]; // [rsp+1E0h] [rbp-B10h] BYREF
  __int64 v214; // [rsp+1F0h] [rbp-B00h] BYREF
  __int64 *v215; // [rsp+200h] [rbp-AF0h]
  __int64 v216; // [rsp+210h] [rbp-AE0h] BYREF
  __int64 v217[2]; // [rsp+230h] [rbp-AC0h] BYREF
  __int64 v218; // [rsp+240h] [rbp-AB0h] BYREF
  __int64 *v219; // [rsp+250h] [rbp-AA0h]
  __int64 v220; // [rsp+260h] [rbp-A90h] BYREF
  __int64 v221[2]; // [rsp+280h] [rbp-A70h] BYREF
  __int64 v222; // [rsp+290h] [rbp-A60h] BYREF
  __int64 *v223; // [rsp+2A0h] [rbp-A50h]
  __int64 v224; // [rsp+2B0h] [rbp-A40h] BYREF
  char *v225; // [rsp+2D0h] [rbp-A20h] BYREF
  __int64 v226; // [rsp+2D8h] [rbp-A18h]
  _BYTE v227[128]; // [rsp+2E0h] [rbp-A10h] BYREF
  unsigned __int64 *v228; // [rsp+360h] [rbp-990h] BYREF
  unsigned int v229; // [rsp+368h] [rbp-988h]
  unsigned int v230; // [rsp+36Ch] [rbp-984h]
  _QWORD v231[16]; // [rsp+370h] [rbp-980h] BYREF
  __int64 v232; // [rsp+3F0h] [rbp-900h] BYREF
  __int64 v233; // [rsp+3F8h] [rbp-8F8h]
  __int64 v234; // [rsp+400h] [rbp-8F0h] BYREF
  unsigned int v235; // [rsp+408h] [rbp-8E8h]
  _BYTE *v236; // [rsp+480h] [rbp-870h] BYREF
  __int64 v237; // [rsp+488h] [rbp-868h]
  _BYTE v238[32]; // [rsp+490h] [rbp-860h] BYREF
  __int64 v239; // [rsp+4B0h] [rbp-840h]
  __int64 v240; // [rsp+4B8h] [rbp-838h]
  __int64 v241; // [rsp+4C0h] [rbp-830h]
  __int64 v242; // [rsp+4C8h] [rbp-828h]
  void **v243; // [rsp+4D0h] [rbp-820h]
  void **v244; // [rsp+4D8h] [rbp-818h]
  __int64 v245; // [rsp+4E0h] [rbp-810h]
  int v246; // [rsp+4E8h] [rbp-808h]
  __int16 v247; // [rsp+4ECh] [rbp-804h]
  char v248; // [rsp+4EEh] [rbp-802h]
  __int64 v249; // [rsp+4F0h] [rbp-800h]
  __int64 v250; // [rsp+4F8h] [rbp-7F8h]
  void *v251; // [rsp+500h] [rbp-7F0h] BYREF
  void *v252; // [rsp+508h] [rbp-7E8h] BYREF
  __int64 *v253; // [rsp+510h] [rbp-7E0h] BYREF
  __int64 v254; // [rsp+518h] [rbp-7D8h]
  _BYTE v255[384]; // [rsp+520h] [rbp-7D0h] BYREF
  __m128i v256; // [rsp+6A0h] [rbp-650h] BYREF
  __int64 v257; // [rsp+6B0h] [rbp-640h]
  __m128i v258; // [rsp+6B8h] [rbp-638h]
  __int64 v259; // [rsp+6C8h] [rbp-628h]
  __m128i v260; // [rsp+6D0h] [rbp-620h]
  __m128i v261; // [rsp+6E0h] [rbp-610h]
  __m128i *v262; // [rsp+6F0h] [rbp-600h] BYREF
  __int64 v263; // [rsp+6F8h] [rbp-5F8h]
  _BYTE v264[324]; // [rsp+700h] [rbp-5F0h] BYREF
  int v265; // [rsp+844h] [rbp-4ACh]
  __int64 v266; // [rsp+848h] [rbp-4A8h]
  const char *v267; // [rsp+850h] [rbp-4A0h] BYREF
  __int64 v268; // [rsp+858h] [rbp-498h]
  _QWORD v269[2]; // [rsp+860h] [rbp-490h] BYREF
  __int16 v270; // [rsp+870h] [rbp-480h]
  __int64 v271; // [rsp+880h] [rbp-470h]
  __int64 v272; // [rsp+888h] [rbp-468h]
  __int64 v273; // [rsp+890h] [rbp-460h]
  __int64 v274; // [rsp+898h] [rbp-458h]
  unsigned __int64 *v275; // [rsp+8A0h] [rbp-450h]
  void **v276; // [rsp+8A8h] [rbp-448h]
  __int64 v277; // [rsp+8B0h] [rbp-440h] BYREF
  int v278; // [rsp+8B8h] [rbp-438h]
  __int16 v279; // [rsp+8BCh] [rbp-434h]
  char v280; // [rsp+8BEh] [rbp-432h]
  __int64 v281; // [rsp+8C0h] [rbp-430h]
  __int64 v282; // [rsp+8C8h] [rbp-428h]
  void *v283; // [rsp+8D0h] [rbp-420h] BYREF
  void *v284; // [rsp+8D8h] [rbp-418h] BYREF
  const char *v285; // [rsp+A00h] [rbp-2F0h] BYREF
  __int64 v286; // [rsp+A08h] [rbp-2E8h]
  _BYTE v287[16]; // [rsp+A10h] [rbp-2E0h] BYREF
  __int16 v288; // [rsp+A20h] [rbp-2D0h]
  __int64 v289; // [rsp+C10h] [rbp-E0h]
  __int64 v290; // [rsp+C18h] [rbp-D8h]
  __int64 v291; // [rsp+C20h] [rbp-D0h]
  __int64 v292; // [rsp+C28h] [rbp-C8h]
  char v293; // [rsp+C30h] [rbp-C0h]
  __int64 v294; // [rsp+C38h] [rbp-B8h]
  char *v295; // [rsp+C40h] [rbp-B0h]
  __int64 v296; // [rsp+C48h] [rbp-A8h]
  int v297; // [rsp+C50h] [rbp-A0h]
  char v298; // [rsp+C54h] [rbp-9Ch]
  char v299; // [rsp+C58h] [rbp-98h] BYREF
  __int16 v300; // [rsp+C98h] [rbp-58h]
  _QWORD *v301; // [rsp+CA0h] [rbp-50h]
  _QWORD *v302; // [rsp+CA8h] [rbp-48h]
  __int64 v303; // [rsp+CB0h] [rbp-40h]

  v3 = *(_BYTE *)a2 == 85;
  v201 = a1;
  if ( !v3 )
  {
    if ( LOBYTE(qword_4FEC628[8]) )
      goto LABEL_22;
    v4 = (__int64 *)v201[4];
LABEL_4:
    v5 = (_QWORD *)(a2 + 72);
    if ( !(unsigned __int8)sub_A73ED0((_QWORD *)(a2 + 72), 23) && !(unsigned __int8)sub_B49560(a2, 23)
      || (unsigned __int8)sub_A73ED0((_QWORD *)(a2 + 72), 4)
      || (unsigned __int8)sub_B49560(a2, 4) )
    {
      v6 = *(_QWORD *)(a2 - 32);
      if ( v6 )
      {
        if ( !*(_BYTE *)v6
          && *(_QWORD *)(v6 + 24) == *(_QWORD *)(a2 + 80)
          && sub_981210(*v4, v6, (unsigned int *)&v285)
          && (_DWORD)v285 == 357 )
        {
          return 0;
        }
      }
    }
    v4 = (__int64 *)v201[4];
    if ( *(_BYTE *)a2 != 85 )
    {
LABEL_10:
      if ( !(unsigned __int8)sub_A73ED0(v5, 23) && !(unsigned __int8)sub_B49560(a2, 23)
        || (unsigned __int8)sub_A73ED0(v5, 4)
        || (unsigned __int8)sub_B49560(a2, 4) )
      {
        v7 = *(_QWORD *)(a2 - 32);
        if ( v7 )
        {
          if ( !*(_BYTE *)v7
            && *(_QWORD *)(v7 + 24) == *(_QWORD *)(a2 + 80)
            && sub_981210(*v4, v7, (unsigned int *)&v285)
            && (_DWORD)v285 == 186 )
          {
            return 0;
          }
        }
      }
      goto LABEL_22;
    }
    v32 = *(_QWORD *)(a2 - 32);
    if ( !v32 )
    {
      v5 = (_QWORD *)(a2 + 72);
      goto LABEL_10;
    }
LABEL_256:
    if ( !*(_BYTE *)v32
      && *(_QWORD *)(v32 + 24) == *(_QWORD *)(a2 + 80)
      && (*(_BYTE *)(v32 + 33) & 0x20) != 0
      && (unsigned int)(*(_DWORD *)(v32 + 36) - 238) <= 7
      && ((1LL << (*(_BYTE *)(v32 + 36) + 18)) & 0xAD) != 0 )
    {
      goto LABEL_22;
    }
    v5 = (_QWORD *)(a2 + 72);
    goto LABEL_10;
  }
  v31 = *(_QWORD *)(a2 - 32);
  v32 = v31;
  if ( !v31
    || *(_BYTE *)v31
    || *(_QWORD *)(v31 + 24) != *(_QWORD *)(a2 + 80)
    || (*(_BYTE *)(v31 + 33) & 0x20) == 0
    || (v151 = *(_DWORD *)(v31 + 36), (unsigned int)(v151 - 238) > 7)
    || ((1LL << ((unsigned __int8)v151 + 18)) & 0xAD) == 0 )
  {
    if ( LOBYTE(qword_4FEC628[8]) )
      goto LABEL_22;
    v32 = v31;
    v4 = (__int64 *)v201[4];
    if ( !v31 )
      goto LABEL_4;
LABEL_251:
    if ( !*(_BYTE *)v31
      && *(_QWORD *)(v31 + 24) == *(_QWORD *)(a2 + 80)
      && (*(_BYTE *)(v31 + 33) & 0x20) != 0
      && (unsigned int)(*(_DWORD *)(v31 + 36) - 238) <= 7
      && ((1LL << (*(_BYTE *)(v31 + 36) + 18)) & 0xAD) != 0 )
    {
      goto LABEL_256;
    }
    goto LABEL_4;
  }
  if ( v151 == 241 )
    return 0;
  if ( !LOBYTE(qword_4FEC628[8]) )
  {
    v4 = (__int64 *)v201[4];
    goto LABEL_251;
  }
LABEL_22:
  sub_ED2710((__int64)&v206, a2, 1, 0x16u, &v202, 0);
  if ( !v207 )
    goto LABEL_38;
  v10 = v202;
  v11 = v202;
  if ( !(_BYTE)qword_4FEC748
    || (v23 = (const char *)sub_FDD2C0((__int64 *)v201[1], *(_QWORD *)(a2 + 40), 0),
        v286 = v24,
        v8 = (unsigned __int8)v24,
        v285 = v23,
        v11 = (unsigned __int64)v23,
        (_BYTE)v24) )
  {
    if ( (unsigned int)qword_4FECAC8 <= v11 && v202 )
    {
      v202 = v11;
      v225 = v227;
      v230 = 16;
      v232 = 0;
      v233 = 1;
      v226 = 0x1000000000LL;
      v228 = v231;
      v12 = (unsigned __int64 *)&v234;
      do
        *v12++ = -1;
      while ( v12 != (unsigned __int64 *)&v236 );
      v13 = v206;
      v231[0] = 0;
      v253 = (__int64 *)v255;
      v14 = v207;
      v254 = 0x1800000000LL;
      v229 = 1;
      v200 = &v206[v14];
      if ( v206 == &v206[v14] )
      {
        v8 = 0;
        goto LABEL_71;
      }
      v198 = (_BYTE *)v10;
      v15 = v11;
      v185 = v11;
      v186 = 0;
      v199 = 0;
      v197 = (_BYTE *)a2;
      while ( 1 )
      {
        v16 = (const char *)v13->m128i_i64[0];
        v17 = v13->m128i_u64[1];
        if ( (_BYTE)qword_4FEC748 )
        {
          if ( v17
            && (_BitScanReverse64(&v34, v17), v35 = 63 - (v34 ^ 0x3F), v15)
            && (_BitScanReverse64(&v36, v15), v37 = 63 - (v36 ^ 0x3F) + v35, v37 > 0x3E) )
          {
            v38 = -1;
            if ( v37 == 63 )
            {
              v39 = v15 * (v17 >> 1);
              if ( v39 >= 0 )
              {
                v38 = 2 * v39;
                if ( (v17 & 1) != 0 )
                {
                  v40 = v15 + v38;
                  if ( v15 >= v38 )
                    v38 = v15;
                  v41 = v40 < v38;
                  v38 = -1;
                  if ( !v41 )
                    v38 = v40;
                }
              }
            }
          }
          else
          {
            v38 = v15 * v17;
          }
          v17 = v38 / v10;
        }
        if ( (unsigned __int64)v16 > 8 && (unsigned int)sub_39FAC40(v13->m128i_i64[0]) != 1
          || (unsigned int)qword_4FEC588 < (__int64)v16 )
        {
          v18 = (unsigned int)v254;
          v19 = _mm_loadu_si128(v13);
          v20 = (unsigned int)v254 + 1LL;
          if ( v20 > HIDWORD(v254) )
          {
            v195 = v19;
            sub_C8D5F0((__int64)&v253, v255, v20, 0x10u, v9, v11);
            v18 = (unsigned int)v254;
            v19 = _mm_load_si128(&v195);
          }
          *(__m128i *)&v253[2 * v18] = v19;
          LODWORD(v254) = v254 + 1;
          goto LABEL_36;
        }
        if ( v17 < (unsigned int)qword_4FECAC8 || v17 < v185 * (unsigned int)qword_4FEC908 / 0x64 )
          break;
        v267 = v16;
        sub_24BC900((__int64)&v285, (__int64)&v232, (__int64 *)&v267);
        if ( !(_BYTE)v288 )
        {
          v8 = 0;
          v173 = (__int64)sub_CB72A0();
          v174 = *(char **)(v173 + 32);
          if ( *(_QWORD *)(v173 + 24) - (_QWORD)v174 <= 0x29u )
          {
            v173 = sub_CB6200(v173, "warning: Invalid Profile Data in Function ", 0x2Au);
          }
          else
          {
            qmemcpy(v174, "warning: Invalid Profile Data in Function ", 0x2Au);
            *(_QWORD *)(v173 + 32) += 42LL;
          }
          v175 = (unsigned __int8 *)sub_BD5D20(*v201);
          v177 = *(void **)(v173 + 32);
          v178 = v176;
          if ( *(_QWORD *)(v173 + 24) - (_QWORD)v177 < v176 )
          {
            v173 = sub_CB6200(v173, v175, v176);
          }
          else if ( v176 )
          {
            memcpy(v177, v175, v176);
            *(_QWORD *)(v173 + 32) += v178;
          }
          v179 = *(char **)(v173 + 32);
          if ( *(_QWORD *)(v173 + 24) - (_QWORD)v179 <= 0x2Du )
          {
            sub_CB6200(v173, ": Two identical values in MemOp value counts.\n", 0x2Eu);
          }
          else
          {
            qmemcpy(v179, ": Two identical values in MemOp value counts.\n", 0x2Eu);
            *(_QWORD *)(v173 + 32) += 46LL;
          }
          v33 = v253;
LABEL_69:
          if ( v33 != (__int64 *)v255 )
            _libc_free((unsigned __int64)v33);
LABEL_71:
          if ( (v233 & 1) == 0 )
            sub_C7D6A0(v234, 8LL * v235, 8);
          if ( v228 != v231 )
            _libc_free((unsigned __int64)v228);
          if ( v225 != v227 )
            _libc_free((unsigned __int64)v225);
          goto LABEL_39;
        }
        v25 = (unsigned int)v226;
        v26 = (unsigned int)v226 + 1LL;
        if ( v26 > HIDWORD(v226) )
        {
          sub_C8D5F0((__int64)&v225, v227, v26, 8u, v9, v11);
          v25 = (unsigned int)v226;
        }
        *(_QWORD *)&v225[8 * v25] = v16;
        v27 = v229;
        LODWORD(v226) = v226 + 1;
        v28 = v229 + 1LL;
        if ( v28 > v230 )
        {
          sub_C8D5F0((__int64)&v228, v231, v28, 8u, v9, v11);
          v27 = v229;
        }
        v228[v27] = v17;
        v29 = v199;
        ++v229;
        v30 = v13->m128i_i64[1];
        if ( v199 < v17 )
          v29 = v17;
        ++v186;
        v185 -= v17;
        v199 = v29;
        v198 -= v30;
        if ( v186 >= (unsigned int)qword_4FEC828 && (_DWORD)qword_4FEC828 )
        {
          v21 = v197;
          sub_24BC0C0((__int64)&v253, (char *)&v253[2 * (unsigned int)v254], v13[1].m128i_i8, v200->m128i_i8);
          goto LABEL_67;
        }
LABEL_36:
        if ( ++v13 == v200 )
        {
          v21 = v197;
          goto LABEL_67;
        }
      }
      v21 = v197;
      sub_24BC0C0((__int64)&v253, (char *)&v253[2 * (unsigned int)v254], v13->m128i_i8, v200->m128i_i8);
LABEL_67:
      if ( !v186 )
      {
        v33 = v253;
        v8 = 0;
        goto LABEL_69;
      }
      v42 = v201;
      *v228 = v185;
      v43 = v199;
      v44 = *((_QWORD *)v21 + 5);
      if ( v185 >= v199 )
        v43 = v185;
      v184 = *((_QWORD *)v21 + 5);
      v183 = v43;
      v182 = v202;
      v45 = sub_FDD860((__int64 *)v42[1], v184);
      v46 = v42[3];
      v288 = 257;
      v47 = v45;
      v48 = sub_F36960(v44, (__int64 *)v21 + 3, 0, v46, 0, 0, (void **)&v285, 0);
      v49 = v42[3];
      v50 = (unsigned __int8 *)v48;
      v51 = *((_QWORD *)v21 + 4);
      v288 = 257;
      if ( v51 )
        v51 -= 24;
      v191 = (__int64)v50;
      v52 = (unsigned __int8 *)sub_F36960((__int64)v50, (__int64 *)(v51 + 24), 0, v49, 0, 0, (void **)&v285, 0);
      v285 = "MemOP.Merge";
      v193 = (unsigned __int64)v52;
      v288 = 259;
      sub_BD6B50(v52, &v285);
      sub_FE1040((__int64 *)v42[1], v193, v47);
      v285 = "MemOP.Default";
      v288 = 259;
      sub_BD6B50(v50, &v285);
      v53 = v42[3];
      v54 = *v42;
      v300 = 0;
      v285 = v287;
      v291 = v53;
      v286 = 0x1000000000LL;
      v289 = 0;
      v290 = 0;
      v292 = 0;
      v293 = 0;
      v294 = 0;
      v295 = &v299;
      v296 = 8;
      v297 = 0;
      v298 = 1;
      v301 = 0;
      v302 = 0;
      v303 = 0;
      v188 = sub_B2BE50(v54);
      v242 = sub_AA48A0(v44);
      v243 = &v251;
      v244 = &v252;
      v239 = v44;
      LOWORD(v241) = 0;
      v251 = &unk_49DA100;
      v245 = 0;
      v247 = 512;
      v246 = 0;
      v248 = 7;
      v249 = 0;
      v250 = 0;
      v252 = &unk_49DA0B0;
      v240 = v44 + 48;
      v200 = *(__m128i **)(v44 + 48);
      v55 = (unsigned __int64)v200 & 0xFFFFFFFFFFFFFFF8LL;
      v236 = v238;
      v237 = 0x200000000LL;
      if ( ((unsigned __int64)v200 & 0xFFFFFFFFFFFFFFF8LL) == v44 + 48 )
      {
        v57 = 0;
      }
      else
      {
        if ( !v55 )
          BUG();
        v56 = *(unsigned __int8 *)(v55 - 24);
        v57 = 0;
        v58 = (_QWORD *)(v55 - 24);
        if ( (unsigned int)(v56 - 30) < 0xB )
          v57 = v58;
      }
      sub_B43D60(v57);
      if ( *v21 != 85 )
      {
        v59 = *((_DWORD *)v21 + 1);
        goto LABEL_100;
      }
      v59 = *((_DWORD *)v21 + 1);
      v165 = *((_QWORD *)v21 - 4);
      v60 = *(_QWORD *)&v21[32 * (2LL - (v59 & 0x7FFFFFF))];
      if ( !v165
        || *(_BYTE *)v165
        || *(_QWORD *)(v165 + 24) != *((_QWORD *)v21 + 10)
        || (*(_BYTE *)(v165 + 33) & 0x20) == 0
        || (unsigned int)(*(_DWORD *)(v165 + 36) - 238) > 7
        || ((1LL << (*(_BYTE *)(v165 + 36) + 18)) & 0xAD) == 0 )
      {
LABEL_100:
        v60 = *(_QWORD *)&v21[32 * (2LL - (v59 & 0x7FFFFFF))];
      }
      v61 = v226;
      v270 = 257;
      v62 = sub_BD2DA0(80);
      v190 = v62;
      if ( v62 )
        sub_B53A60(v62, v60, v191, v61, 0, 0);
      (*((void (__fastcall **)(void **, __int64, const char **, __int64, __int64))*v244 + 2))(
        v244,
        v190,
        &v267,
        v240,
        v241);
      v63 = (unsigned __int64)v236;
      v64 = &v236[16 * (unsigned int)v237];
      if ( v236 != v64 )
      {
        do
        {
          v65 = *(_QWORD *)(v63 + 8);
          v66 = *(_DWORD *)v63;
          v63 += 16LL;
          sub_B99FD0(v190, v66, v65);
        }
        while ( v64 != (_BYTE *)v63 );
      }
      v189 = *((_QWORD *)v21 + 1);
      if ( *(_BYTE *)(v189 + 8) == 7 )
      {
        v192 = 0;
      }
      else
      {
        v152 = sub_AA4FF0(v193);
        LOBYTE(v154) = 1;
        v155 = v152;
        v156 = 0;
        if ( v155 )
          v156 = v153;
        HIBYTE(v154) = v156;
        sub_2412230((__int64)&v267, v193, v155, v154, 0, v155, 0, 0);
        v256.m128i_i64[0] = (__int64)"MemOP.RVMerge";
        v258.m128i_i16[4] = 259;
        v192 = sub_D5C860((__int64 *)&v267, v189, (int)v226 + 1, (__int64)&v256);
        sub_BD84D0((__int64)v21, v192);
        LODWORD(v200) = *(_DWORD *)(v192 + 4);
        v157 = (unsigned int)v200 & 0x7FFFFFF;
        if ( ((unsigned int)v200 & 0x7FFFFFF) == *(_DWORD *)(v192 + 72) )
        {
          sub_B48D90(v192);
          LODWORD(v200) = *(_DWORD *)(v192 + 4);
          v157 = (unsigned int)v200 & 0x7FFFFFF;
        }
        v158 = (v157 + 1) & 0x7FFFFFF;
        LODWORD(v200) = *(_DWORD *)(v192 + 4);
        v159 = v158 | (unsigned int)v200 & 0xF8000000;
        v160 = *(_QWORD *)(v192 - 8) + 32LL * (unsigned int)(v158 - 1);
        *(_DWORD *)(v192 + 4) = v159;
        if ( *(_QWORD *)v160 )
        {
          v161 = *(_QWORD *)(v160 + 8);
          **(_QWORD **)(v160 + 16) = v161;
          if ( v161 )
            *(_QWORD *)(v161 + 16) = *(_QWORD *)(v160 + 16);
        }
        *(_QWORD *)v160 = v21;
        v162 = *((_QWORD *)v21 + 2);
        *(_QWORD *)(v160 + 8) = v162;
        if ( v162 )
          *(_QWORD *)(v162 + 16) = v160 + 8;
        *(_QWORD *)(v160 + 16) = v21 + 16;
        *((_QWORD *)v21 + 2) = v160;
        LODWORD(v200) = *(_DWORD *)(v192 + 4);
        *(_QWORD *)(*(_QWORD *)(v192 - 8)
                  + 32LL * *(unsigned int *)(v192 + 72)
                  + 8LL * (((unsigned int)v200 & 0x7FFFFFF) - 1)) = v191;
        nullsub_61();
        v283 = &unk_49DA100;
        nullsub_63();
        if ( v267 != (const char *)v269 )
          _libc_free((unsigned __int64)v267);
      }
      sub_B99FD0((__int64)v21, 2u, 0);
      if ( v198 || v207 != v186 )
      {
        sub_ED2230(*(__int64 ***)(*v201 + 40), (__int64)v21, v253, (unsigned int)v254, (__int64)v198, 1u, v207);
        v67 = v180;
        v68 = v181;
      }
      v203 = 0;
      v204 = 0;
      v69 = (unsigned int)v226;
      v205 = 0;
      if ( v201[3] && (_DWORD)v226 )
      {
        v70 = 2LL * (unsigned int)v226;
        v71 = (__m128i *)sub_22077B0(v70 * 16);
        v72 = v203;
        v73 = v204;
        v74 = v71;
        for ( i = v203; v73 != i; ++v71 )
        {
          if ( v71 )
            *v71 = _mm_loadu_si128(i);
          ++i;
        }
        if ( v72 )
          j_j___libc_free_0((unsigned __int64)v72);
        v203 = (__m128i *)v74;
        v69 = (unsigned int)v226;
        v204 = (__m128i *)v74;
        v205 = &v74[v70];
      }
      v76 = (__int64)&v225[8 * v69];
      v187 = (char *)v76;
      if ( (char *)v76 != v225 )
      {
        v200 = (__m128i *)v225;
        v196 = &v283;
        v195.m128i_i64[0] = (__int64)v269;
        v198 = v21;
        v197 = &unk_49DA100;
        while ( 1 )
        {
          v77 = v200->m128i_i64[0];
          v270 = 2819;
          v221[0] = v77;
          v78 = *v201;
          v267 = "MemOP.Case.";
          v269[0] = v221;
          v79 = sub_22077B0(0x50u);
          v80 = v79;
          if ( v79 )
            sub_AA4D50(v79, v188, (__int64)&v267, v78, v191);
          v81 = sub_B47F80(v198);
          v82 = *(_QWORD *)(*(_QWORD *)(v81 + 32 * (2LL - (*(_DWORD *)(v81 + 4) & 0x7FFFFFF))) + 8LL);
          if ( *(_BYTE *)(v82 + 8) != 12 )
            v82 = 0;
          v83 = sub_ACD640(v82, v221[0], 0);
          v3 = *(_BYTE *)v81 == 85;
          v199 = v83;
          if ( !v3 )
            break;
          v84 = *(_DWORD *)(v81 + 4);
          v143 = *(_QWORD *)(v81 - 32);
          v144 = v81 + 32 * (2LL - (v84 & 0x7FFFFFF));
          if ( !v143 )
            goto LABEL_127;
          if ( *(_BYTE *)v143 )
            goto LABEL_127;
          if ( *(_QWORD *)(v143 + 24) != *(_QWORD *)(v81 + 80) )
            goto LABEL_127;
          if ( (*(_BYTE *)(v143 + 33) & 0x20) == 0 )
            goto LABEL_127;
          v145 = *(_DWORD *)(v143 + 36) - 238;
          if ( v145 > 7 || ((1LL << v145) & 0xAD) == 0 )
            goto LABEL_127;
          if ( *(_QWORD *)v144 )
          {
            v146 = *(_QWORD *)(v144 + 8);
            **(_QWORD **)(v144 + 16) = v146;
            if ( v146 )
              *(_QWORD *)(v146 + 16) = *(_QWORD *)(v144 + 16);
          }
          v147 = v199;
          *(_QWORD *)v144 = v199;
          if ( v147 )
          {
            v148 = v147 + 16;
            v149 = *(_QWORD *)(v147 + 16);
            *(_QWORD *)(v144 + 8) = v149;
            if ( v149 )
              *(_QWORD *)(v149 + 16) = v144 + 8;
            v150 = v199;
            *(_QWORD *)(v144 + 16) = v148;
            *(_QWORD *)(v150 + 16) = v144;
          }
LABEL_134:
          v89 = v194;
          LOWORD(v89) = 0;
          v194 = v89;
          sub_B44240((_QWORD *)v81, v80, (unsigned __int64 *)(v80 + 48), v89);
          v274 = sub_AA48A0(v80);
          v267 = (const char *)v195.m128i_i64[0];
          v275 = (unsigned __int64 *)v196;
          v268 = 0x200000000LL;
          v276 = &v284;
          v258.m128i_i16[4] = 257;
          v283 = v197;
          LOWORD(v273) = 0;
          v279 = 512;
          v277 = 0;
          v278 = 0;
          v280 = 7;
          v281 = 0;
          v282 = 0;
          v284 = &unk_49DA0B0;
          v271 = v80;
          v272 = v80 + 48;
          v90 = sub_BD2C40(72, 1u);
          v91 = (__int64)v90;
          if ( v90 )
            sub_B4C8F0((__int64)v90, v193, 1u, 0, 0);
          (*((void (__fastcall **)(void **, __int64, __m128i *, __int64, __int64))*v276 + 2))(
            v276,
            v91,
            &v256,
            v272,
            v273);
          v92 = v267;
          v93 = &v267[16 * (unsigned int)v268];
          if ( v267 != v93 )
          {
            do
            {
              v94 = *((_QWORD *)v92 + 1);
              v95 = *(_DWORD *)v92;
              v92 += 16;
              sub_B99FD0(v91, v95, v94);
            }
            while ( v93 != v92 );
          }
          sub_B53E30(v190, v199, v80);
          if ( *(_BYTE *)(v189 + 8) != 7 )
          {
            LODWORD(v199) = *(_DWORD *)(v192 + 4);
            v96 = v199 & 0x7FFFFFF;
            if ( (v199 & 0x7FFFFFF) == *(_DWORD *)(v192 + 72) )
            {
              sub_B48D90(v192);
              LODWORD(v199) = *(_DWORD *)(v192 + 4);
              v96 = v199 & 0x7FFFFFF;
            }
            v97 = (v96 + 1) & 0x7FFFFFF;
            LODWORD(v199) = *(_DWORD *)(v192 + 4);
            v98 = v97 | v199 & 0xF8000000;
            v99 = *(_QWORD *)(v192 - 8) + 32LL * (unsigned int)(v97 - 1);
            *(_DWORD *)(v192 + 4) = v98;
            if ( *(_QWORD *)v99 )
            {
              v100 = *(_QWORD *)(v99 + 8);
              **(_QWORD **)(v99 + 16) = v100;
              if ( v100 )
                *(_QWORD *)(v100 + 16) = *(_QWORD *)(v99 + 16);
            }
            *(_QWORD *)v99 = v81;
            v101 = *(_QWORD *)(v81 + 16);
            *(_QWORD *)(v99 + 8) = v101;
            if ( v101 )
              *(_QWORD *)(v101 + 16) = v99 + 8;
            *(_QWORD *)(v99 + 16) = v81 + 16;
            *(_QWORD *)(v81 + 16) = v99;
            LODWORD(v199) = *(_DWORD *)(v192 + 4);
            *(_QWORD *)(*(_QWORD *)(v192 - 8)
                      + 32LL * *(unsigned int *)(v192 + 72)
                      + 8LL * (((unsigned int)v199 & 0x7FFFFFF) - 1)) = v80;
          }
          if ( !v201[3] )
            goto LABEL_154;
          v102 = v204;
          v256.m128i_i64[0] = v80;
          v103 = v80 & 0xFFFFFFFFFFFFFFFBLL;
          v104 = v205;
          v256.m128i_i64[1] = v193 & 0xFFFFFFFFFFFFFFFBLL;
          if ( v204 != v205 )
          {
            if ( v204 )
            {
              *v204 = _mm_load_si128(&v256);
              v102 = v204;
              v104 = v205;
            }
            v105 = v102 + 1;
            v256.m128i_i64[1] = v103;
            v204 = v105;
            v256.m128i_i64[0] = v184;
            if ( v105 == v104 )
              goto LABEL_263;
LABEL_152:
            *v105 = _mm_load_si128(&v256);
            v105 = v204;
            goto LABEL_153;
          }
          sub_F38BA0((const __m128i **)&v203, v204, &v256);
          v256.m128i_i64[1] = v103;
          v105 = v204;
          v256.m128i_i64[0] = v184;
          if ( v205 == v204 )
          {
            v104 = v204;
LABEL_263:
            sub_F38BA0((const __m128i **)&v203, v104, &v256);
            goto LABEL_154;
          }
          if ( v204 )
            goto LABEL_152;
LABEL_153:
          v204 = v105 + 1;
LABEL_154:
          nullsub_61();
          v283 = v197;
          nullsub_63();
          if ( v267 != (const char *)v195.m128i_i64[0] )
            _libc_free((unsigned __int64)v267);
          v200 = (__m128i *)((char *)v200 + 8);
          if ( v187 == (char *)v200 )
          {
            v21 = v198;
            goto LABEL_158;
          }
        }
        v84 = *(_DWORD *)(v81 + 4);
LABEL_127:
        v85 = v81 + 32 * (2LL - (v84 & 0x7FFFFFF));
        if ( *(_QWORD *)v85 )
        {
          v86 = *(_QWORD *)(v85 + 8);
          **(_QWORD **)(v85 + 16) = v86;
          if ( v86 )
            *(_QWORD *)(v86 + 16) = *(_QWORD *)(v85 + 16);
        }
        v87 = v199;
        *(_QWORD *)v85 = v199;
        if ( v87 )
        {
          v88 = *(_QWORD *)(v87 + 16);
          *(_QWORD *)(v85 + 8) = v88;
          if ( v88 )
            *(_QWORD *)(v88 + 16) = v85 + 8;
          *(_QWORD *)(v85 + 16) = v87 + 16;
          *(_QWORD *)(v199 + 16) = v85;
        }
        goto LABEL_134;
      }
LABEL_158:
      v106 = v203;
      sub_FFB3D0((__int64)&v285, (unsigned __int64 *)v203, v204 - v203, v76, v67, v68);
      if ( v203 != v204 )
        v204 = v203;
      if ( v183 )
      {
        v106 = (__m128i *)v190;
        sub_24AD4F0(*(_QWORD *)(*v201 + 40), v190, v228, v229, v183);
      }
      v107 = (__int64 *)v201[2];
      v108 = *v107;
      v109 = sub_B2BE50(*v107);
      if ( !sub_B6EA50(v109) )
      {
        v163 = sub_B2BE50(v108);
        v164 = sub_B6F970(v163);
        if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v164 + 48LL))(v164) )
        {
LABEL_209:
          if ( v203 )
          {
            v106 = (__m128i *)((char *)v205 - (char *)v203);
            j_j___libc_free_0((unsigned __int64)v203);
          }
          nullsub_61();
          v251 = &unk_49DA100;
          nullsub_63();
          if ( v236 != v238 )
            _libc_free((unsigned __int64)v236);
          sub_FFCE90((__int64)&v285, (__int64)v106, v131, v132, v133, v134);
          sub_FFD870((__int64)&v285, (__int64)v106, v135, v136, v137, v138);
          sub_FFBC40((__int64)&v285, (__int64)v106);
          v139 = v302;
          v140 = v301;
          if ( v302 != v301 )
          {
            do
            {
              v141 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v140[7];
              *v140 = &unk_49E5048;
              if ( v141 )
                v141(v140 + 5, v140 + 5, 3);
              *v140 = &unk_49DB368;
              v142 = v140[3];
              if ( v142 != 0 && v142 != -4096 && v142 != -8192 )
                sub_BD60C0(v140 + 1);
              v140 += 9;
            }
            while ( v139 != v140 );
            v140 = v301;
          }
          if ( v140 )
            j_j___libc_free_0((unsigned __int64)v140);
          if ( !v298 )
            _libc_free((unsigned __int64)v295);
          if ( v285 != v287 )
            _libc_free((unsigned __int64)v285);
          v33 = v253;
          v8 = 1;
          goto LABEL_69;
        }
      }
      sub_B174A0((__int64)&v267, (__int64)"pgo-memop-opt", (__int64)"memopt-opt", 10, (__int64)v21);
      sub_B18290((__int64)&v267, "optimized ", 0xAu);
      if ( *v21 == 85
        && (v171 = *((_QWORD *)v21 - 4)) != 0
        && !*(_BYTE *)v171
        && *(_QWORD *)(v171 + 24) == *((_QWORD *)v21 + 10)
        && (*(_BYTE *)(v171 + 33) & 0x20) != 0
        && (v172 = *(_DWORD *)(v171 + 36), v172 - 238 <= 7)
        && ((1LL << ((unsigned __int8)v172 + 18)) & 0xAD) != 0 )
      {
        if ( v172 == 243 )
        {
          v112 = "memset";
        }
        else if ( v172 > 0xF3 )
        {
          v112 = "unknown";
        }
        else
        {
          if ( v172 == 238 )
          {
            v113 = "memcpy";
            v112 = "memcpy";
LABEL_174:
            v114 = strlen(v112);
            sub_B16430((__int64)v221, "Memop", 5u, v113, v114);
            v115 = sub_23FD640((__int64)&v267, (__int64)v221);
            sub_B18290(v115, " with count ", 0xCu);
            sub_B16B10(v217, "Count", 5, v182 - v185);
            v116 = sub_23FD640(v115, (__int64)v217);
            sub_B18290(v116, " out of ", 8u);
            sub_B16B10(v213, "Total", 5, v202);
            v117 = sub_23FD640(v116, (__int64)v213);
            sub_B18290(v117, " for ", 5u);
            sub_B169E0(v209, "Versions", 8, v186);
            v118 = (char *)sub_23FD640(v117, (__int64)v209);
            sub_B18290((__int64)v118, " versions", 9u);
            v256.m128i_i32[2] = *((_DWORD *)v118 + 2);
            v256.m128i_i8[12] = v118[12];
            v257 = *((_QWORD *)v118 + 2);
            v122 = _mm_loadu_si128((const __m128i *)(v118 + 24));
            v256.m128i_i64[0] = (__int64)&unk_49D9D40;
            v258 = v122;
            v259 = *((_QWORD *)v118 + 5);
            v260 = _mm_loadu_si128((const __m128i *)v118 + 3);
            v123 = _mm_loadu_si128((const __m128i *)v118 + 4);
            v262 = (__m128i *)v264;
            v263 = 0x400000000LL;
            v261 = v123;
            v124 = *((unsigned int *)v118 + 22);
            if ( (_DWORD)v124 && &v262 != (__m128i **)(v118 + 80) )
            {
              v166 = (unsigned int)v124;
              v167 = (__m128i *)v264;
              if ( (unsigned int)v124 > 4 )
              {
                LODWORD(v201) = *((_DWORD *)v118 + 22);
                sub_11F02D0((__int64)&v262, (unsigned int)v124, v119, v120, v124, v121);
                v167 = v262;
                v166 = *((unsigned int *)v118 + 22);
                LODWORD(v124) = (_DWORD)v201;
              }
              v168 = *((_QWORD *)v118 + 10);
              if ( v168 != v168 + 80 * v166 )
              {
                v201 = v107;
                v200 = (__m128i *)v118;
                v169 = v168;
                v170 = v124;
                do
                {
                  if ( v167 )
                  {
                    v167->m128i_i64[0] = (__int64)v167[1].m128i_i64;
                    sub_24BC010(v167->m128i_i64, *(_BYTE **)v169, *(_QWORD *)v169 + *(_QWORD *)(v169 + 8));
                    v167[2].m128i_i64[0] = (__int64)v167[3].m128i_i64;
                    sub_24BC010(
                      v167[2].m128i_i64,
                      *(_BYTE **)(v169 + 32),
                      *(_QWORD *)(v169 + 32) + *(_QWORD *)(v169 + 40));
                    v167[4] = _mm_loadu_si128((const __m128i *)(v169 + 64));
                  }
                  v169 += 80;
                  v167 += 5;
                }
                while ( v168 + 80 * v166 != v169 );
                v107 = v201;
                v118 = (char *)v200;
                LODWORD(v124) = v170;
              }
              LODWORD(v263) = v124;
            }
            v264[320] = v118[416];
            v265 = *((_DWORD *)v118 + 105);
            v266 = *((_QWORD *)v118 + 53);
            v256.m128i_i64[0] = (__int64)&unk_49D9D78;
            if ( v211 != &v212 )
              j_j___libc_free_0((unsigned __int64)v211);
            if ( (__int64 *)v209[0] != &v210 )
              j_j___libc_free_0(v209[0]);
            if ( v215 != &v216 )
              j_j___libc_free_0((unsigned __int64)v215);
            if ( (__int64 *)v213[0] != &v214 )
              j_j___libc_free_0(v213[0]);
            if ( v219 != &v220 )
              j_j___libc_free_0((unsigned __int64)v219);
            if ( (__int64 *)v217[0] != &v218 )
              j_j___libc_free_0(v217[0]);
            if ( v223 != &v224 )
              j_j___libc_free_0((unsigned __int64)v223);
            if ( (__int64 *)v221[0] != &v222 )
              j_j___libc_free_0(v221[0]);
            v125 = v275;
            v267 = (const char *)&unk_49D9D40;
            v126 = &v275[10 * (unsigned int)v276];
            if ( v275 != v126 )
            {
              do
              {
                v126 -= 10;
                v127 = v126[4];
                if ( (unsigned __int64 *)v127 != v126 + 6 )
                  j_j___libc_free_0(v127);
                if ( (unsigned __int64 *)*v126 != v126 + 2 )
                  j_j___libc_free_0(*v126);
              }
              while ( v125 != v126 );
              v126 = v275;
            }
            if ( v126 != (unsigned __int64 *)&v277 )
              _libc_free((unsigned __int64)v126);
            v106 = &v256;
            sub_1049740(v107, (__int64)&v256);
            v128 = (unsigned __int64 *)v262;
            v256.m128i_i64[0] = (__int64)&unk_49D9D40;
            v129 = (unsigned __int64 *)&v262[5 * (unsigned int)v263];
            if ( v262 != (__m128i *)v129 )
            {
              do
              {
                v129 -= 10;
                v130 = v129[4];
                if ( (unsigned __int64 *)v130 != v129 + 6 )
                {
                  v106 = (__m128i *)(v129[6] + 1);
                  j_j___libc_free_0(v130);
                }
                if ( (unsigned __int64 *)*v129 != v129 + 2 )
                {
                  v106 = (__m128i *)(v129[2] + 1);
                  j_j___libc_free_0(*v129);
                }
              }
              while ( v128 != v129 );
              v129 = (unsigned __int64 *)v262;
            }
            if ( v129 != (unsigned __int64 *)v264 )
              _libc_free((unsigned __int64)v129);
            goto LABEL_209;
          }
          v112 = "unknown";
          if ( v172 == 241 )
            v112 = "memmove";
        }
      }
      else
      {
        v110 = (__int64 *)v201[4];
        if ( ((unsigned __int8)sub_A73ED0((_QWORD *)v21 + 9, 23) || (unsigned __int8)sub_B49560((__int64)v21, 23))
          && !(unsigned __int8)sub_A73ED0((_QWORD *)v21 + 9, 4)
          && !(unsigned __int8)sub_B49560((__int64)v21, 4) )
        {
          goto LABEL_329;
        }
        v111 = *((_QWORD *)v21 - 4);
        if ( !v111
          || *(_BYTE *)v111
          || *(_QWORD *)(v111 + 24) != *((_QWORD *)v21 + 10)
          || !sub_981210(*v110, v111, (unsigned int *)&v256) )
        {
          goto LABEL_329;
        }
        if ( v256.m128i_i32[0] == 357 )
        {
          v112 = "memcmp";
          goto LABEL_173;
        }
        if ( v256.m128i_i32[0] != 186 )
LABEL_329:
          BUG();
        v112 = "bcmp";
      }
LABEL_173:
      v113 = (char *)v112;
      goto LABEL_174;
    }
LABEL_38:
    v8 = 0;
  }
LABEL_39:
  if ( v206 != (__m128i *)&v208 )
    _libc_free((unsigned __int64)v206);
  return v8;
}
