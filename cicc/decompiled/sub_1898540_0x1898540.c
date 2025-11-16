// Function: sub_1898540
// Address: 0x1898540
//
__int64 *__fastcall sub_1898540(__int64 *a1, __int64 a2, __int64 *a3, __int64 *a4)
{
  __int64 v5; // r12
  __int64 v6; // rbx
  __int64 *v7; // r13
  __int64 *v8; // r12
  __int64 v9; // r15
  __int64 *v10; // rbx
  __int64 *v11; // r14
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rax
  void *v15; // rdi
  unsigned int v16; // eax
  __int64 v17; // rdx
  unsigned __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rdi
  unsigned __int64 *v22; // rbx
  unsigned __int64 *v23; // r12
  unsigned __int64 v24; // rdi
  unsigned __int64 *v25; // r12
  _QWORD *v26; // rbx
  _QWORD *v27; // r12
  __int64 v28; // r13
  __int64 v29; // rdi
  __int64 *v31; // r15
  int v32; // ebx
  __int64 v33; // rdi
  int v34; // eax
  float v35; // xmm0_4
  int v36; // esi
  __int64 v37; // rcx
  unsigned int v38; // edx
  __int64 v39; // rax
  __int64 v40; // rdi
  __int64 v41; // r13
  __int64 v42; // rdi
  unsigned __int64 v43; // rax
  unsigned __int64 i; // rdi
  unsigned int v45; // ecx
  __int64 v46; // rdx
  __int64 v47; // r8
  int v48; // eax
  __int64 v49; // rax
  int v50; // esi
  __int64 v51; // rdi
  int v52; // ecx
  __int64 v53; // rax
  int v54; // esi
  __int64 v55; // rdx
  unsigned int v56; // ecx
  __int64 v57; // rax
  __int64 v58; // rdi
  __int64 v59; // rax
  _BYTE *v60; // rsi
  __int64 v61; // rax
  unsigned int v62; // eax
  __int64 v63; // rax
  int v64; // r8d
  unsigned int v65; // ecx
  __int64 *v66; // rdx
  __int64 v67; // r9
  unsigned int v68; // ebx
  __int64 *v69; // rdi
  __int64 k; // rax
  __int64 *v71; // r12
  __int64 v72; // rbx
  __int64 v73; // r14
  __int64 v74; // rax
  __int64 v75; // rdx
  __int64 v76; // r12
  unsigned __int64 v77; // rbx
  __int64 *v78; // rcx
  __int64 v79; // rax
  __int64 v80; // rax
  __int64 v81; // rax
  __int64 v82; // r8
  char *v83; // rax
  __int64 v84; // rdx
  __int64 v85; // rbx
  _QWORD *v86; // rbx
  _QWORD *v87; // r13
  _QWORD *v88; // rdi
  _QWORD *v89; // rbx
  _QWORD *v90; // r12
  _QWORD *v91; // rdi
  __int64 v92; // r13
  __int64 v93; // r12
  int v94; // r14d
  __int64 v95; // rax
  unsigned int v96; // r12d
  unsigned __int64 v97; // r14
  unsigned __int64 n; // rdi
  __int64 *v99; // rsi
  int v100; // eax
  unsigned __int64 *v101; // rdx
  unsigned __int64 *v102; // r12
  unsigned __int64 *v103; // rbx
  unsigned __int64 v104; // rdi
  unsigned __int64 *v105; // rbx
  unsigned __int64 v106; // rdi
  _QWORD *v107; // rbx
  _QWORD *v108; // r14
  __int64 v109; // rax
  __int64 v110; // rdi
  unsigned __int64 *v111; // rbx
  unsigned __int64 *v112; // r12
  __int64 *v113; // rax
  __int64 *v114; // rbx
  int v115; // r11d
  int v116; // ebx
  __int64 v117; // r11
  int v118; // edi
  __int64 v119; // rax
  __int64 v120; // rax
  int j; // edx
  int v122; // r10d
  __int64 *v123; // rbx
  __int64 *v124; // r12
  int v125; // r13d
  __int64 v126; // rdi
  __int64 *v127; // rbx
  __int64 *v128; // r14
  unsigned int v129; // ecx
  __int64 v130; // rax
  __int64 v131; // rdi
  __int64 v132; // rdx
  int v133; // esi
  int v134; // edi
  __int64 v135; // rax
  __int64 v136; // rdx
  int v137; // r8d
  int v138; // r9d
  __int64 v139; // rcx
  __int64 v140; // rbx
  __int64 v141; // rax
  int v142; // r8d
  int v143; // r9d
  unsigned int v144; // eax
  _QWORD *v145; // rbx
  __int64 *v146; // rdi
  __int64 v147; // rax
  __int64 v148; // rax
  __int64 v149; // r8
  __int64 v150; // r12
  __int64 v151; // rbx
  _QWORD *v152; // rbx
  _QWORD *v153; // r13
  _QWORD *v154; // rdi
  _QWORD *v155; // rbx
  _QWORD *v156; // rdi
  int v157; // r11d
  __int64 v158; // r10
  int v159; // ebx
  __int64 v160; // r11
  int v161; // edx
  __int64 v162; // rax
  __int64 v163; // rax
  int v164; // [rsp+14h] [rbp-87Ch]
  __int64 *v165; // [rsp+20h] [rbp-870h]
  __int64 *v166; // [rsp+28h] [rbp-868h]
  __int64 *v167; // [rsp+40h] [rbp-850h]
  __int64 v168; // [rsp+48h] [rbp-848h]
  __int64 *v172; // [rsp+78h] [rbp-818h]
  __int64 *m; // [rsp+78h] [rbp-818h]
  __int64 v174; // [rsp+78h] [rbp-818h]
  __int64 v176; // [rsp+90h] [rbp-800h]
  unsigned __int64 v177; // [rsp+98h] [rbp-7F8h]
  char v178; // [rsp+A2h] [rbp-7EEh]
  char v179; // [rsp+A3h] [rbp-7EDh]
  unsigned int v180; // [rsp+A4h] [rbp-7ECh]
  unsigned int v181; // [rsp+A8h] [rbp-7E8h]
  __int64 v182; // [rsp+A8h] [rbp-7E8h]
  unsigned int v183; // [rsp+B4h] [rbp-7DCh] BYREF
  __int64 v184; // [rsp+B8h] [rbp-7D8h] BYREF
  _BYTE *v185; // [rsp+C0h] [rbp-7D0h] BYREF
  _BYTE *v186; // [rsp+C8h] [rbp-7C8h]
  _BYTE *v187; // [rsp+D0h] [rbp-7C0h]
  __int64 v188; // [rsp+E0h] [rbp-7B0h] BYREF
  __int64 v189; // [rsp+E8h] [rbp-7A8h]
  __int64 v190; // [rsp+F0h] [rbp-7A0h]
  unsigned int v191; // [rsp+F8h] [rbp-798h]
  unsigned __int64 v192[2]; // [rsp+100h] [rbp-790h] BYREF
  char v193; // [rsp+110h] [rbp-780h] BYREF
  __int64 v194; // [rsp+118h] [rbp-778h]
  _QWORD *v195; // [rsp+120h] [rbp-770h]
  __int64 v196; // [rsp+128h] [rbp-768h]
  unsigned int v197; // [rsp+130h] [rbp-760h]
  __int64 *v198; // [rsp+140h] [rbp-750h]
  char v199; // [rsp+148h] [rbp-748h]
  int v200; // [rsp+14Ch] [rbp-744h]
  __int64 *v201; // [rsp+150h] [rbp-740h] BYREF
  __int64 v202; // [rsp+158h] [rbp-738h]
  _BYTE v203[64]; // [rsp+160h] [rbp-730h] BYREF
  _QWORD v204[2]; // [rsp+1A0h] [rbp-6F0h] BYREF
  __int64 v205; // [rsp+1B0h] [rbp-6E0h] BYREF
  __int64 *v206; // [rsp+1C0h] [rbp-6D0h]
  __int64 v207; // [rsp+1D0h] [rbp-6C0h] BYREF
  _QWORD v208[2]; // [rsp+200h] [rbp-690h] BYREF
  _QWORD v209[2]; // [rsp+210h] [rbp-680h] BYREF
  _QWORD *v210; // [rsp+220h] [rbp-670h]
  _QWORD v211[6]; // [rsp+230h] [rbp-660h] BYREF
  char v212[8]; // [rsp+260h] [rbp-630h] BYREF
  __int64 v213; // [rsp+268h] [rbp-628h]
  __int64 *v214; // [rsp+280h] [rbp-610h]
  __int64 *v215; // [rsp+288h] [rbp-608h]
  __int64 v216; // [rsp+290h] [rbp-600h]
  unsigned __int64 v217; // [rsp+298h] [rbp-5F8h]
  unsigned __int64 v218; // [rsp+2A0h] [rbp-5F0h]
  unsigned __int64 *v219; // [rsp+2A8h] [rbp-5E8h]
  unsigned int v220; // [rsp+2B0h] [rbp-5E0h]
  char v221; // [rsp+2B8h] [rbp-5D8h] BYREF
  unsigned __int64 *v222; // [rsp+2D8h] [rbp-5B8h]
  unsigned int v223; // [rsp+2E0h] [rbp-5B0h]
  __int64 v224; // [rsp+2E8h] [rbp-5A8h] BYREF
  __int64 v225; // [rsp+300h] [rbp-590h] BYREF
  _QWORD *v226; // [rsp+308h] [rbp-588h]
  __int64 v227; // [rsp+310h] [rbp-580h]
  __int64 v228; // [rsp+318h] [rbp-578h]
  __int64 v229; // [rsp+320h] [rbp-570h]
  __int64 v230; // [rsp+328h] [rbp-568h]
  __int64 v231; // [rsp+330h] [rbp-560h]
  __int64 v232; // [rsp+338h] [rbp-558h]
  __int64 v233; // [rsp+348h] [rbp-548h]
  _BYTE *v234; // [rsp+350h] [rbp-540h]
  _BYTE *v235; // [rsp+358h] [rbp-538h]
  __int64 v236; // [rsp+360h] [rbp-530h]
  int v237; // [rsp+368h] [rbp-528h]
  _BYTE v238[128]; // [rsp+370h] [rbp-520h] BYREF
  __int64 v239; // [rsp+3F0h] [rbp-4A0h]
  _BYTE *v240; // [rsp+3F8h] [rbp-498h]
  _BYTE *v241; // [rsp+400h] [rbp-490h]
  __int64 v242; // [rsp+408h] [rbp-488h]
  int v243; // [rsp+410h] [rbp-480h]
  _BYTE v244[136]; // [rsp+418h] [rbp-478h] BYREF
  __int64 *v245; // [rsp+4A0h] [rbp-3F0h] BYREF
  __int64 v246; // [rsp+4A8h] [rbp-3E8h] BYREF
  __int64 v247; // [rsp+4B0h] [rbp-3E0h] BYREF
  __m128i v248; // [rsp+4B8h] [rbp-3D8h]
  __int64 v249; // [rsp+4C8h] [rbp-3C8h]
  __int64 v250; // [rsp+4D0h] [rbp-3C0h]
  __m128i v251; // [rsp+4D8h] [rbp-3B8h]
  __int64 v252; // [rsp+4E8h] [rbp-3A8h]
  char v253; // [rsp+4F0h] [rbp-3A0h]
  _BYTE *v254; // [rsp+4F8h] [rbp-398h] BYREF
  __int64 v255; // [rsp+500h] [rbp-390h]
  _BYTE v256[352]; // [rsp+508h] [rbp-388h] BYREF
  char v257; // [rsp+668h] [rbp-228h]
  int v258; // [rsp+66Ch] [rbp-224h]
  __int64 v259; // [rsp+670h] [rbp-220h]
  __int64 *v260; // [rsp+680h] [rbp-210h] BYREF
  __int64 v261; // [rsp+688h] [rbp-208h] BYREF
  __int64 v262; // [rsp+690h] [rbp-200h] BYREF
  __int64 v263; // [rsp+698h] [rbp-1F8h]
  __int64 v264; // [rsp+6A0h] [rbp-1F0h]
  __int64 v265; // [rsp+6D0h] [rbp-1C0h]
  _QWORD *v266; // [rsp+6D8h] [rbp-1B8h]
  __int64 v267; // [rsp+6E0h] [rbp-1B0h]
  _BYTE v268[424]; // [rsp+6E8h] [rbp-1A8h] BYREF

  v5 = a3[10];
  if ( v5 )
    v5 -= 24;
  v198 = a3;
  v192[0] = (unsigned __int64)&v193;
  v192[1] = 0x100000000LL;
  v194 = 0;
  v195 = 0;
  v196 = 0;
  v197 = 0;
  v199 = 0;
  v200 = 0;
  sub_15D3930((__int64)v192);
  sub_14019E0((__int64)v212, (__int64)v192);
  v225 = 0;
  v234 = v238;
  v235 = v238;
  v226 = 0;
  v227 = 0;
  v228 = 0;
  v229 = 0;
  v230 = 0;
  v231 = 0;
  v232 = 0;
  v233 = 0;
  v236 = 16;
  v237 = 0;
  v239 = 0;
  v240 = v244;
  v241 = v244;
  v242 = 16;
  v243 = 0;
  sub_137CAE0((__int64)&v225, a3, (__int64)v212, 0);
  if ( *(_BYTE *)(a2 + 40) )
  {
    v167 = 0;
    v166 = (__int64 *)(*(__int64 (__fastcall **)(_QWORD, __int64 *))(a2 + 24))(*(_QWORD *)(a2 + 32), a3);
  }
  else
  {
    v113 = (__int64 *)sub_22077B0(8);
    v166 = v113;
    v114 = v113;
    if ( v113 )
    {
      sub_13702A0(v113, a3, (__int64)&v225, (__int64)v212);
      v167 = v114;
    }
    else
    {
      v167 = 0;
    }
  }
  v6 = *(_QWORD *)(a2 + 48);
  v179 = sub_1441AE0((_QWORD *)v6);
  if ( !v179 || **(_DWORD **)(v6 + 8) )
  {
    *a1 = 0;
    goto LABEL_8;
  }
  v168 = sub_22077B0(432);
  if ( v168 )
  {
    *(_QWORD *)v168 = v168 + 16;
    *(_QWORD *)(v168 + 8) = 0x400000000LL;
  }
  v31 = (__int64 *)a3[10];
  if ( a3 + 9 == v31 )
  {
    v35 = 0.0;
  }
  else
  {
    v32 = 0;
    do
    {
      v33 = (__int64)(v31 - 3);
      if ( !v31 )
        v33 = 0;
      v34 = sub_1897310(v33);
      v31 = (__int64 *)v31[1];
      v32 += v34;
    }
    while ( a3 + 9 != v31 );
    v35 = (float)v32;
  }
  v164 = (int)(float)(v35 * *(float *)&dword_4FACD80);
  sub_16AF710(&v183, (int)(float)((float)dword_4FACCA0 * *(float *)&dword_4FACBC0), dword_4FACCA0);
  v184 = v5;
  v185 = 0;
  v186 = 0;
  v187 = 0;
  v188 = 0;
  v189 = 0;
  v190 = 0;
  v191 = 0;
  sub_1292090((__int64)&v185, 0, &v184);
  v36 = v191;
  if ( !v191 )
  {
    ++v188;
    goto LABEL_317;
  }
  v37 = v184;
  v38 = (v191 - 1) & (((unsigned int)v184 >> 9) ^ ((unsigned int)v184 >> 4));
  v39 = v189 + 16LL * v38;
  v40 = *(_QWORD *)v39;
  if ( v184 != *(_QWORD *)v39 )
  {
    v159 = 1;
    v160 = 0;
    while ( v40 != -8 )
    {
      if ( v40 == -16 && !v160 )
        v160 = v39;
      v38 = (v191 - 1) & (v159 + v38);
      v39 = v189 + 16LL * v38;
      v40 = *(_QWORD *)v39;
      if ( v184 == *(_QWORD *)v39 )
        goto LABEL_71;
      ++v159;
    }
    if ( v160 )
      v39 = v160;
    ++v188;
    v161 = v190 + 1;
    if ( 4 * ((int)v190 + 1) < 3 * v191 )
    {
      if ( v191 - HIDWORD(v190) - v161 > v191 >> 3 )
      {
LABEL_305:
        LODWORD(v190) = v161;
        if ( *(_QWORD *)v39 != -8 )
          --HIDWORD(v190);
        *(_QWORD *)v39 = v37;
        *(_BYTE *)(v39 + 8) = 0;
        goto LABEL_71;
      }
LABEL_318:
      sub_1898380((__int64)&v188, v36);
      sub_1898170((__int64)&v188, &v184, &v260);
      v39 = (__int64)v260;
      v37 = v184;
      v161 = v190 + 1;
      goto LABEL_305;
    }
LABEL_317:
    v36 = 2 * v191;
    goto LABEL_318;
  }
LABEL_71:
  *(_BYTE *)(v39 + 8) = 1;
  v178 = 0;
LABEL_72:
  while ( v186 != v185 )
  {
    v41 = *((_QWORD *)v186 - 1);
    v42 = *(_QWORD *)(a2 + 48);
    v186 -= 8;
    if ( !sub_1442060(v42, v41, v166) )
    {
      sub_1368C40((__int64)&v260, v166, v41);
      v43 = 0;
      if ( (_BYTE)v261 )
      {
        sub_1368C40((__int64)&v245, v166, v41);
        v43 = (unsigned __int64)v245;
      }
      if ( (unsigned int)dword_4FACCA0 <= v43 )
      {
        v176 = v41;
        v180 = 0;
        v177 = sub_157EBA0(v41);
        for ( i = v177; ; i = sub_157EBA0(v176) )
        {
          v48 = 0;
          if ( i )
            v48 = sub_15F4D60(i);
          if ( v48 == v180 )
            goto LABEL_72;
          v49 = sub_15F4DF0(v177, v180);
          v50 = v191;
          v245 = (__int64 *)v49;
          if ( !v191 )
            break;
          v45 = (v191 - 1) & (((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4));
          v46 = v189 + 16LL * v45;
          v47 = *(_QWORD *)v46;
          if ( v49 != *(_QWORD *)v46 )
          {
            v115 = 1;
            v51 = 0;
            while ( v47 != -8 )
            {
              if ( v47 == -16 && !v51 )
                v51 = v46;
              v45 = (v191 - 1) & (v115 + v45);
              v46 = v189 + 16LL * v45;
              v47 = *(_QWORD *)v46;
              if ( v49 == *(_QWORD *)v46 )
                goto LABEL_79;
              ++v115;
            }
            if ( !v51 )
              v51 = v46;
            ++v188;
            v52 = v190 + 1;
            if ( 4 * ((int)v190 + 1) < 3 * v191 )
            {
              if ( v191 - HIDWORD(v190) - v52 <= v191 >> 3 )
              {
LABEL_87:
                sub_1898380((__int64)&v188, v50);
                sub_1898170((__int64)&v188, (__int64 *)&v245, &v260);
                v51 = (__int64)v260;
                v49 = (__int64)v245;
                v52 = v190 + 1;
              }
              LODWORD(v190) = v52;
              if ( *(_QWORD *)v51 != -8 )
                --HIDWORD(v190);
              *(_QWORD *)v51 = v49;
              *(_BYTE *)(v51 + 8) = 0;
              goto LABEL_91;
            }
LABEL_86:
            v50 = 2 * v191;
            goto LABEL_87;
          }
LABEL_79:
          if ( *(_BYTE *)(v46 + 8) )
            goto LABEL_80;
LABEL_91:
          v53 = sub_15F4DF0(v177, v180);
          v54 = v191;
          v245 = (__int64 *)v53;
          v55 = v53;
          if ( !v191 )
          {
            ++v188;
LABEL_219:
            v54 = 2 * v191;
LABEL_220:
            sub_1898380((__int64)&v188, v54);
            sub_1898170((__int64)&v188, (__int64 *)&v245, &v260);
            v57 = (__int64)v260;
            v55 = (__int64)v245;
            v118 = v190 + 1;
            goto LABEL_215;
          }
          v56 = (v191 - 1) & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
          v57 = v189 + 16LL * v56;
          v58 = *(_QWORD *)v57;
          if ( v55 == *(_QWORD *)v57 )
            goto LABEL_93;
          v116 = 1;
          v117 = 0;
          while ( v58 != -8 )
          {
            if ( v117 || v58 != -16 )
              v57 = v117;
            v56 = (v191 - 1) & (v116 + v56);
            v58 = *(_QWORD *)(v189 + 16LL * v56);
            if ( v55 == v58 )
            {
              v57 = v189 + 16LL * v56;
              goto LABEL_93;
            }
            ++v116;
            v117 = v57;
            v57 = v189 + 16LL * v56;
          }
          if ( v117 )
            v57 = v117;
          ++v188;
          v118 = v190 + 1;
          if ( 4 * ((int)v190 + 1) >= 3 * v191 )
            goto LABEL_219;
          if ( v191 - HIDWORD(v190) - v118 <= v191 >> 3 )
            goto LABEL_220;
LABEL_215:
          LODWORD(v190) = v118;
          if ( *(_QWORD *)v57 != -8 )
            --HIDWORD(v190);
          *(_QWORD *)v57 = v55;
          *(_BYTE *)(v57 + 8) = 0;
LABEL_93:
          *(_BYTE *)(v57 + 8) = 1;
          v59 = sub_15F4DF0(v177, v180);
          v60 = v186;
          v260 = (__int64 *)v59;
          if ( v186 == v187 )
          {
            sub_15D0700((__int64)&v185, v186, &v260);
          }
          else
          {
            if ( v186 )
            {
              *(_QWORD *)v186 = v59;
              v60 = v186;
            }
            v186 = v60 + 8;
          }
          v61 = sub_15F4DF0(v177, v180);
          v62 = sub_13774B0((__int64)&v225, v176, v61);
          if ( v62 <= v183 )
          {
            v201 = (__int64 *)v203;
            v202 = 0x800000000LL;
            v63 = sub_15F4DF0(v177, v180);
            LODWORD(v202) = 0;
            if ( !v197 )
              goto LABEL_142;
            v64 = v197 - 1;
            v65 = (v197 - 1) & (((unsigned int)v63 >> 9) ^ ((unsigned int)v63 >> 4));
            v66 = &v195[2 * v65];
            v67 = *v66;
            if ( v63 != *v66 )
            {
              for ( j = 1; ; j = v122 )
              {
                if ( v67 == -8 )
                  goto LABEL_142;
                v122 = j + 1;
                v65 = v64 & (j + v65);
                v66 = &v195[2 * v65];
                v67 = *v66;
                if ( v63 == *v66 )
                  break;
              }
            }
            if ( v66 == &v195[2 * v197] || !v66[1] )
              goto LABEL_142;
            v68 = 1;
            v262 = v66[1];
            v69 = &v262;
            v260 = &v262;
            v261 = 0x800000001LL;
            for ( k = 0; ; k = (unsigned int)v202 )
            {
              v71 = (__int64 *)v69[v68 - 1];
              LODWORD(v261) = v68 - 1;
              v72 = *v71;
              if ( (unsigned int)k >= HIDWORD(v202) )
              {
                sub_16CD150((__int64)&v201, v203, 0, 8, v64, v67);
                k = (unsigned int)v202;
              }
              v201[k] = v72;
              v73 = v71[3];
              v74 = v71[4];
              v75 = (unsigned int)v261;
              LODWORD(v202) = v202 + 1;
              v76 = v74 - v73;
              v77 = (v74 - v73) >> 3;
              if ( v77 > HIDWORD(v261) - (unsigned __int64)(unsigned int)v261 )
              {
                sub_16CD150((__int64)&v260, &v262, v77 + (unsigned int)v261, 8, v64, v67);
                v75 = (unsigned int)v261;
              }
              v69 = v260;
              v78 = &v260[v75];
              if ( v76 > 0 )
              {
                v79 = 0;
                do
                {
                  v78[v79] = *(_QWORD *)(v73 + 8 * v79);
                  ++v79;
                }
                while ( (__int64)(v77 - v79) > 0 );
                v69 = v260;
                LODWORD(v75) = v261;
              }
              v68 = v77 + v75;
              LODWORD(v261) = v68;
              if ( !v68 )
                break;
            }
            if ( v69 != &v262 )
              _libc_free((unsigned __int64)v69);
            v172 = v201;
            v181 = v202;
            if ( (unsigned int)v202 > 1 && (v92 = *v201, (v93 = *(_QWORD *)(*v201 + 8)) != 0) )
            {
              while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v93) + 16) - 25) > 9u )
              {
                v93 = *(_QWORD *)(v93 + 8);
                if ( !v93 )
                  goto LABEL_143;
              }
              v94 = 0;
              while ( 1 )
              {
                v93 = *(_QWORD *)(v93 + 8);
                if ( !v93 )
                  break;
                while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v93) + 16) - 25) <= 9u )
                {
                  v93 = *(_QWORD *)(v93 + 8);
                  ++v94;
                  if ( !v93 )
                    goto LABEL_154;
                }
              }
LABEL_154:
              if ( v94 )
                goto LABEL_143;
              v95 = v181;
              v182 = 0;
              v165 = &v172[v95];
              for ( m = v172 + 1; ; ++m )
              {
                v96 = v68;
                v97 = sub_157EBA0(v92);
                for ( n = v97; ; n = sub_157EBA0(v92) )
                {
                  v100 = v68;
                  if ( n )
                    v100 = sub_15F4D60(n);
                  if ( v96 == v100 )
                    break;
                  v260 = (__int64 *)sub_15F4DF0(v97, v96);
                  v99 = &v201[(unsigned int)v202];
                  if ( v99 == sub_18970F0(v201, (__int64)v99, (__int64 *)&v260) )
                  {
                    if ( v182 )
                    {
                      v80 = sub_15E0530(*a4);
                      if ( !sub_1602790(v80) )
                      {
                        v119 = sub_15E0530(*a4);
                        v120 = sub_16033E0(v119);
                        if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v120 + 48LL))(v120) )
                          goto LABEL_142;
                      }
                      v81 = *(_QWORD *)(sub_15F4DF0(v97, v96) + 48);
                      v82 = v81 - 24;
                      if ( !v81 )
                        v82 = 0;
                      sub_15CA5C0((__int64)&v260, (__int64)"partial-inlining", (__int64)"MultiExitRegion", 15, v82);
                      sub_15CAB20((__int64)&v260, "Region dominated by ", 0x14u);
                      v83 = (char *)sub_1649960(*v201);
                      sub_15C9800((__int64)v208, "Block", 5, v83, v84);
                      v85 = sub_17C21B0((__int64)&v260, (__int64)v208);
                      sub_15CAB20(v85, " has more than one region exit edge.", 0x24u);
                      LODWORD(v246) = *(_DWORD *)(v85 + 8);
                      BYTE4(v246) = *(_BYTE *)(v85 + 12);
                      v247 = *(_QWORD *)(v85 + 16);
                      v248 = _mm_loadu_si128((const __m128i *)(v85 + 24));
                      v249 = *(_QWORD *)(v85 + 40);
                      v245 = (__int64 *)&unk_49ECF68;
                      v250 = *(_QWORD *)(v85 + 48);
                      v251 = _mm_loadu_si128((const __m128i *)(v85 + 56));
                      v253 = *(_BYTE *)(v85 + 80);
                      if ( v253 )
                        v252 = *(_QWORD *)(v85 + 72);
                      v254 = v256;
                      v255 = 0x400000000LL;
                      if ( *(_DWORD *)(v85 + 96) )
                        sub_1897E20((__int64)&v254, v85 + 88);
                      v257 = *(_BYTE *)(v85 + 456);
                      v258 = *(_DWORD *)(v85 + 460);
                      v259 = *(_QWORD *)(v85 + 464);
                      v245 = (__int64 *)&unk_49ECFC8;
                      if ( v210 != v211 )
                        j_j___libc_free_0(v210, v211[0] + 1LL);
                      if ( (_QWORD *)v208[0] != v209 )
                        j_j___libc_free_0(v208[0], v209[0] + 1LL);
                      v86 = v266;
                      v260 = (__int64 *)&unk_49ECF68;
                      v87 = &v266[11 * (unsigned int)v267];
                      if ( v266 != v87 )
                      {
                        do
                        {
                          v87 -= 11;
                          v88 = (_QWORD *)v87[4];
                          if ( v88 != v87 + 6 )
                            j_j___libc_free_0(v88, v87[6] + 1LL);
                          if ( (_QWORD *)*v87 != v87 + 2 )
                            j_j___libc_free_0(*v87, v87[2] + 1LL);
                        }
                        while ( v86 != v87 );
                        v87 = v266;
                      }
                      if ( v87 != (_QWORD *)v268 )
                        _libc_free((unsigned __int64)v87);
                      sub_143AA50(a4, (__int64)&v245);
                      v89 = v254;
                      v245 = (__int64 *)&unk_49ECF68;
                      v90 = &v254[88 * (unsigned int)v255];
                      if ( v254 != (_BYTE *)v90 )
                      {
                        do
                        {
                          v90 -= 11;
                          v91 = (_QWORD *)v90[4];
                          if ( v91 != v90 + 6 )
                            j_j___libc_free_0(v91, v90[6] + 1LL);
                          if ( (_QWORD *)*v90 != v90 + 2 )
                            j_j___libc_free_0(*v90, v90[2] + 1LL);
                        }
                        while ( v89 != v90 );
LABEL_139:
                        v90 = v254;
                      }
LABEL_140:
                      if ( v90 != (_QWORD *)v256 )
                        _libc_free((unsigned __int64)v90);
LABEL_142:
                      v172 = v201;
                      goto LABEL_143;
                    }
                    v182 = v92;
                  }
                  ++v96;
                }
                if ( m == v165 )
                  break;
                v92 = *m;
              }
              v172 = v201;
              if ( !v182 )
                goto LABEL_143;
              v123 = &v201[(unsigned int)v202];
              if ( v123 == v201 )
              {
                if ( v164 > 0 )
                  goto LABEL_259;
              }
              else
              {
                v124 = v201;
                v125 = 0;
                do
                {
                  v126 = *v124++;
                  v125 += sub_1897310(v126);
                }
                while ( v123 != v124 );
                if ( v164 > v125 )
                {
LABEL_259:
                  v147 = sub_15E0530(*a4);
                  if ( !sub_1602790(v147) )
                  {
                    v162 = sub_15E0530(*a4);
                    v163 = sub_16033E0(v162);
                    if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v163 + 48LL))(v163) )
                      goto LABEL_142;
                  }
                  v148 = *(_QWORD *)(sub_15F4DF0(v177, v180) + 48);
                  v149 = v148 - 24;
                  if ( !v148 )
                    v149 = 0;
                  sub_15CA700((__int64)&v260, (__int64)"partial-inlining", (__int64)"TooCostly", 9, v149);
                  sub_15C9340((__int64)v208, "Callee", 6u, (__int64)a3);
                  v150 = sub_1897C30((__int64)&v260, (__int64)v208);
                  sub_15CAB20(v150, " inline cost-savings smaller than ", 0x22u);
                  sub_15C9890((__int64)v204, "Cost", 4, (unsigned int)v164);
                  v151 = sub_160FEC0(v150, (__int64)v204);
                  LODWORD(v246) = *(_DWORD *)(v151 + 8);
                  BYTE4(v246) = *(_BYTE *)(v151 + 12);
                  v247 = *(_QWORD *)(v151 + 16);
                  v248 = _mm_loadu_si128((const __m128i *)(v151 + 24));
                  v249 = *(_QWORD *)(v151 + 40);
                  v245 = (__int64 *)&unk_49ECF68;
                  v250 = *(_QWORD *)(v151 + 48);
                  v251 = _mm_loadu_si128((const __m128i *)(v151 + 56));
                  v253 = *(_BYTE *)(v151 + 80);
                  if ( v253 )
                    v252 = *(_QWORD *)(v151 + 72);
                  v254 = v256;
                  v255 = 0x400000000LL;
                  if ( *(_DWORD *)(v151 + 96) )
                    sub_1897E20((__int64)&v254, v151 + 88);
                  v257 = *(_BYTE *)(v151 + 456);
                  v258 = *(_DWORD *)(v151 + 460);
                  v259 = *(_QWORD *)(v151 + 464);
                  v245 = (__int64 *)&unk_49ECFF8;
                  if ( v206 != &v207 )
                    j_j___libc_free_0(v206, v207 + 1);
                  if ( (__int64 *)v204[0] != &v205 )
                    j_j___libc_free_0(v204[0], v205 + 1);
                  if ( v210 != v211 )
                    j_j___libc_free_0(v210, v211[0] + 1LL);
                  if ( (_QWORD *)v208[0] != v209 )
                    j_j___libc_free_0(v208[0], v209[0] + 1LL);
                  v152 = v266;
                  v260 = (__int64 *)&unk_49ECF68;
                  v153 = &v266[11 * (unsigned int)v267];
                  if ( v266 != v153 )
                  {
                    do
                    {
                      v153 -= 11;
                      v154 = (_QWORD *)v153[4];
                      if ( v154 != v153 + 6 )
                        j_j___libc_free_0(v154, v153[6] + 1LL);
                      if ( (_QWORD *)*v153 != v153 + 2 )
                        j_j___libc_free_0(*v153, v153[2] + 1LL);
                    }
                    while ( v152 != v153 );
                    v153 = v266;
                  }
                  if ( v153 != (_QWORD *)v268 )
                    _libc_free((unsigned __int64)v153);
                  sub_143AA50(a4, (__int64)&v245);
                  v155 = v254;
                  v245 = (__int64 *)&unk_49ECF68;
                  v90 = &v254[88 * (unsigned int)v255];
                  if ( v254 != (_BYTE *)v90 )
                  {
                    do
                    {
                      v90 -= 11;
                      v156 = (_QWORD *)v90[4];
                      if ( v156 != v90 + 6 )
                        j_j___libc_free_0(v156, v90[6] + 1LL);
                      if ( (_QWORD *)*v90 != v90 + 2 )
                        j_j___libc_free_0(*v90, v90[2] + 1LL);
                    }
                    while ( v155 != v90 );
                    goto LABEL_139;
                  }
                  goto LABEL_140;
                }
                v127 = v201;
                v128 = &v201[(unsigned int)v202];
                if ( v201 != v128 )
                {
                  while ( 1 )
                  {
                    v132 = *v127;
                    v133 = v191;
                    v245 = (__int64 *)*v127;
                    if ( !v191 )
                      break;
                    v129 = (v191 - 1) & (((unsigned int)v132 >> 9) ^ ((unsigned int)v132 >> 4));
                    v130 = v189 + 16LL * v129;
                    v131 = *(_QWORD *)v130;
                    if ( v132 == *(_QWORD *)v130 )
                    {
LABEL_235:
                      ++v127;
                      *(_BYTE *)(v130 + 8) = 1;
                      if ( v128 == v127 )
                        goto LABEL_243;
                    }
                    else
                    {
                      v157 = 1;
                      v158 = 0;
                      while ( v131 != -8 )
                      {
                        if ( !v158 && v131 == -16 )
                          v158 = v130;
                        v129 = (v191 - 1) & (v157 + v129);
                        v130 = v189 + 16LL * v129;
                        v131 = *(_QWORD *)v130;
                        if ( v132 == *(_QWORD *)v130 )
                          goto LABEL_235;
                        ++v157;
                      }
                      if ( v158 )
                        v130 = v158;
                      ++v188;
                      v134 = v190 + 1;
                      if ( 4 * ((int)v190 + 1) < 3 * v191 )
                      {
                        if ( v191 - HIDWORD(v190) - v134 > v191 >> 3 )
                          goto LABEL_240;
                        goto LABEL_239;
                      }
LABEL_238:
                      v133 = 2 * v191;
LABEL_239:
                      sub_1898380((__int64)&v188, v133);
                      sub_1898170((__int64)&v188, (__int64 *)&v245, &v260);
                      v130 = (__int64)v260;
                      v132 = (__int64)v245;
                      v134 = v190 + 1;
LABEL_240:
                      LODWORD(v190) = v134;
                      if ( *(_QWORD *)v130 != -8 )
                        --HIDWORD(v190);
                      ++v127;
                      *(_BYTE *)(v130 + 8) = 0;
                      *(_QWORD *)v130 = v132;
                      *(_BYTE *)(v130 + 8) = 1;
                      if ( v128 == v127 )
                        goto LABEL_243;
                    }
                  }
                  ++v188;
                  goto LABEL_238;
                }
              }
LABEL_243:
              v135 = sub_157F1C0(v182);
              v139 = 0x800000000LL;
              v140 = v135;
              v141 = *v201;
              v245 = &v247;
              v246 = 0x800000000LL;
              if ( (_DWORD)v202 )
              {
                v174 = v141;
                sub_18971B0((__int64)&v245, (__int64)&v201, v136, 0x800000000LL, v137, v138);
                v139 = (unsigned int)v246;
                v260 = &v262;
                v261 = 0x800000000LL;
                v141 = v174;
                if ( (_DWORD)v246 )
                {
                  sub_18971B0((__int64)&v260, (__int64)&v245, v136, (unsigned int)v246, v137, v138);
                  v141 = v174;
                }
              }
              else
              {
                v261 = 0x800000000LL;
                v260 = &v262;
              }
              v265 = v141;
              v267 = v140;
              v266 = (_QWORD *)v182;
              if ( v245 != &v247 )
                _libc_free((unsigned __int64)v245);
              sub_18971B0((__int64)&v260, (__int64)&v201, v136, v139, v137, v138);
              v144 = *(_DWORD *)(v168 + 8);
              if ( v144 >= *(_DWORD *)(v168 + 12) )
              {
                sub_18976A0(v168);
                v144 = *(_DWORD *)(v168 + 8);
              }
              v145 = (_QWORD *)(*(_QWORD *)v168 + 104LL * v144);
              if ( v145 )
              {
                *v145 = v145 + 2;
                v145[1] = 0x800000000LL;
                if ( (_DWORD)v261 )
                  sub_18971B0((__int64)v145, (__int64)&v260, (unsigned int)v261, 13LL * v144, v142, v143);
                v145[10] = v265;
                v145[11] = v266;
                v145[12] = v267;
                v144 = *(_DWORD *)(v168 + 8);
              }
              v146 = v260;
              *(_DWORD *)(v168 + 8) = v144 + 1;
              if ( v146 != &v262 )
                _libc_free((unsigned __int64)v146);
              if ( v201 != (__int64 *)v203 )
                _libc_free((unsigned __int64)v201);
              v178 = v179;
            }
            else
            {
LABEL_143:
              if ( v172 != (__int64 *)v203 )
                _libc_free((unsigned __int64)v172);
            }
          }
LABEL_80:
          ++v180;
        }
        ++v188;
        goto LABEL_86;
      }
    }
  }
  v110 = v189;
  if ( v178 )
  {
    *a1 = v168;
    j___libc_free_0(v110);
    if ( v185 )
      j_j___libc_free_0(v185, v187 - v185);
  }
  else
  {
    *a1 = 0;
    j___libc_free_0(v110);
    if ( v185 )
      j_j___libc_free_0(v185, v187 - v185);
    if ( v168 )
    {
      v111 = *(unsigned __int64 **)v168;
      v112 = (unsigned __int64 *)(*(_QWORD *)v168 + 104LL * *(unsigned int *)(v168 + 8));
      if ( *(unsigned __int64 **)v168 != v112 )
      {
        do
        {
          v112 -= 13;
          if ( (unsigned __int64 *)*v112 != v112 + 2 )
            _libc_free(*v112);
        }
        while ( v111 != v112 );
        v112 = *(unsigned __int64 **)v168;
      }
      if ( v112 != (unsigned __int64 *)(v168 + 16) )
        _libc_free((unsigned __int64)v112);
      j_j___libc_free_0(v168, 432);
    }
  }
LABEL_8:
  if ( v167 )
  {
    sub_1368A00(v167);
    j_j___libc_free_0(v167, 8);
  }
  if ( v241 != v240 )
    _libc_free((unsigned __int64)v241);
  if ( v235 != v234 )
    _libc_free((unsigned __int64)v235);
  j___libc_free_0(v230);
  if ( (_DWORD)v228 )
  {
    v107 = v226;
    v246 = 2;
    v247 = 0;
    v108 = &v226[5 * (unsigned int)v228];
    v248 = (__m128i)0xFFFFFFFFFFFFFFF8LL;
    v245 = (__int64 *)&unk_49E8A80;
    v261 = 2;
    v262 = 0;
    v263 = -16;
    v260 = (__int64 *)&unk_49E8A80;
    v264 = 0;
    do
    {
      v109 = v107[3];
      *v107 = &unk_49EE2B0;
      if ( v109 != 0 && v109 != -8 && v109 != -16 )
        sub_1649B30(v107 + 1);
      v107 += 5;
    }
    while ( v108 != v107 );
    v260 = (__int64 *)&unk_49EE2B0;
    if ( v263 != 0 && v263 != -8 && v263 != -16 )
      sub_1649B30(&v261);
    v245 = (__int64 *)&unk_49EE2B0;
    if ( v248.m128i_i64[0] != 0 && v248.m128i_i64[0] != -8 && v248.m128i_i64[0] != -16 )
      sub_1649B30(&v246);
  }
  j___libc_free_0(v226);
  sub_142D890((__int64)v212);
  v7 = v215;
  v8 = v214;
  if ( v214 != v215 )
  {
    do
    {
      v9 = *v8;
      v10 = *(__int64 **)(*v8 + 8);
      v11 = *(__int64 **)(*v8 + 16);
      if ( v10 == v11 )
      {
        *(_BYTE *)(v9 + 160) = 1;
      }
      else
      {
        do
        {
          v12 = *v10++;
          sub_13FACC0(v12);
        }
        while ( v11 != v10 );
        *(_BYTE *)(v9 + 160) = 1;
        v13 = *(_QWORD *)(v9 + 8);
        if ( *(_QWORD *)(v9 + 16) != v13 )
          *(_QWORD *)(v9 + 16) = v13;
      }
      v14 = *(_QWORD *)(v9 + 32);
      if ( v14 != *(_QWORD *)(v9 + 40) )
        *(_QWORD *)(v9 + 40) = v14;
      ++*(_QWORD *)(v9 + 56);
      v15 = *(void **)(v9 + 72);
      if ( v15 == *(void **)(v9 + 64) )
      {
        *(_QWORD *)v9 = 0;
      }
      else
      {
        v16 = 4 * (*(_DWORD *)(v9 + 84) - *(_DWORD *)(v9 + 88));
        v17 = *(unsigned int *)(v9 + 80);
        if ( v16 < 0x20 )
          v16 = 32;
        if ( v16 < (unsigned int)v17 )
          sub_16CC920(v9 + 56);
        else
          memset(v15, -1, 8 * v17);
        v18 = *(_QWORD *)(v9 + 72);
        v19 = *(_QWORD *)(v9 + 64);
        *(_QWORD *)v9 = 0;
        if ( v18 != v19 )
          _libc_free(v18);
      }
      v20 = *(_QWORD *)(v9 + 32);
      if ( v20 )
        j_j___libc_free_0(v20, *(_QWORD *)(v9 + 48) - v20);
      v21 = *(_QWORD *)(v9 + 8);
      if ( v21 )
        j_j___libc_free_0(v21, *(_QWORD *)(v9 + 24) - v21);
      ++v8;
    }
    while ( v7 != v8 );
    if ( v214 != v215 )
      v215 = v214;
  }
  v22 = v222;
  v23 = &v222[2 * v223];
  if ( v222 != v23 )
  {
    do
    {
      v24 = *v22;
      v22 += 2;
      _libc_free(v24);
    }
    while ( v23 != v22 );
  }
  v223 = 0;
  if ( !v220 )
    goto LABEL_39;
  v101 = v219;
  v224 = 0;
  v102 = &v219[v220];
  v103 = v219 + 1;
  v217 = *v219;
  v218 = v217 + 4096;
  if ( v102 != v219 + 1 )
  {
    do
    {
      v104 = *v103++;
      _libc_free(v104);
    }
    while ( v102 != v103 );
    v101 = v219;
  }
  v220 = 1;
  _libc_free(*v101);
  v105 = v222;
  v25 = &v222[2 * v223];
  if ( v222 != v25 )
  {
    do
    {
      v106 = *v105;
      v105 += 2;
      _libc_free(v106);
    }
    while ( v105 != v25 );
LABEL_39:
    v25 = v222;
  }
  if ( v25 != (unsigned __int64 *)&v224 )
    _libc_free((unsigned __int64)v25);
  if ( v219 != (unsigned __int64 *)&v221 )
    _libc_free((unsigned __int64)v219);
  if ( v214 )
    j_j___libc_free_0(v214, v216 - (_QWORD)v214);
  j___libc_free_0(v213);
  if ( v197 )
  {
    v26 = v195;
    v27 = &v195[2 * v197];
    do
    {
      if ( *v26 != -16 && *v26 != -8 )
      {
        v28 = v26[1];
        if ( v28 )
        {
          v29 = *(_QWORD *)(v28 + 24);
          if ( v29 )
            j_j___libc_free_0(v29, *(_QWORD *)(v28 + 40) - v29);
          j_j___libc_free_0(v28, 56);
        }
      }
      v26 += 2;
    }
    while ( v27 != v26 );
  }
  j___libc_free_0(v195);
  if ( (char *)v192[0] != &v193 )
    _libc_free(v192[0]);
  return a1;
}
