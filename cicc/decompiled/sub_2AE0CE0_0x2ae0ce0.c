// Function: sub_2AE0CE0
// Address: 0x2ae0ce0
//
void __fastcall sub_2AE0CE0(__int64 a1, __int64 *a2)
{
  __int64 v2; // rdx
  __int64 *v3; // rax
  __int64 v4; // rdx
  __int8 v5; // r12
  __int32 v6; // r13d
  _QWORD *v7; // r15
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rsi
  _QWORD *v11; // rdx
  int v12; // ecx
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int8 **v30; // rdi
  _BYTE *v31; // rsi
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // rcx
  unsigned __int64 v35; // rdx
  unsigned __int64 v36; // r15
  __int64 v37; // rax
  __int64 v38; // rcx
  __int64 v39; // rsi
  __int64 v40; // rsi
  __int64 v41; // r8
  __int64 v42; // r9
  __int64 v43; // rcx
  unsigned __int64 v44; // r15
  __int64 v45; // rax
  __int64 v46; // rcx
  __int64 v47; // rsi
  __int64 v48; // rsi
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 v51; // rcx
  unsigned __int64 v52; // rbx
  __int64 v53; // rax
  __int64 v54; // rcx
  __int64 v55; // rsi
  __int64 v56; // rsi
  __int64 v57; // r8
  __int64 v58; // r9
  __int64 v59; // rcx
  unsigned __int64 v60; // rbx
  __int64 v61; // rax
  __int64 v62; // rcx
  __int64 v63; // rsi
  __int64 v64; // rsi
  unsigned __int64 v65; // rdx
  __int64 v66; // rax
  __int64 v67; // rsi
  __int64 v68; // rcx
  __m128i *v69; // rbx
  __m128i *v70; // r15
  __int64 v71; // rbx
  __int64 v72; // rax
  int v73; // edx
  __m128i *v74; // rax
  __int64 v75; // rdx
  unsigned __int64 v76; // rcx
  char v77; // di
  __m128i v78; // xmm0
  __m128i *v79; // rdi
  __int64 *v80; // r13
  unsigned int v81; // eax
  int v82; // r14d
  __m128i *v83; // rbx
  unsigned __int64 v84; // rdi
  unsigned int v85; // r8d
  unsigned int v86; // eax
  __int64 v87; // rcx
  __int64 v88; // rsi
  __int64 *v89; // rsi
  __m128i *v90; // r12
  __int64 v91; // rax
  unsigned __int64 v92; // rbx
  __m128i *v93; // r14
  unsigned __int64 v94; // rsi
  __m128i *v95; // rbx
  const __m128i *v96; // rdi
  __m128i *v97; // r15
  __int64 v98; // rax
  __int64 v99; // rbx
  int v100; // eax
  int v101; // r12d
  char v102; // dl
  __int64 v103; // rcx
  __m128i *v104; // rdx
  __int64 v105; // rsi
  __m128i *k; // rax
  __m128i *v107; // rax
  __m128i si128; // xmm0
  int v109; // eax
  __int64 v110; // r12
  __m128i *v111; // rbx
  int v112; // r14d
  void **v113; // rdi
  __int8 v114; // al
  unsigned __int64 v115; // rsi
  int v116; // edx
  _BYTE *v117; // rax
  size_t v118; // rdx
  unsigned __int8 *v119; // r15
  size_t v120; // r12
  __int64 v121; // rax
  void *v122; // rdi
  __int64 v123; // r14
  __int64 v124; // r14
  char *v125; // rax
  __m128i v127; // [rsp+10h] [rbp-7D0h] BYREF
  __int64 *v128; // [rsp+20h] [rbp-7C0h]
  __m128i *v129; // [rsp+28h] [rbp-7B8h]
  __int64 *v130; // [rsp+30h] [rbp-7B0h]
  __m128i *i; // [rsp+38h] [rbp-7A8h]
  unsigned __int64 v132; // [rsp+40h] [rbp-7A0h]
  __m128i *v133; // [rsp+48h] [rbp-798h]
  __int64 v134; // [rsp+50h] [rbp-790h]
  __m128i *v135; // [rsp+58h] [rbp-788h]
  __int64 v136; // [rsp+68h] [rbp-778h]
  __m128i v137; // [rsp+70h] [rbp-770h] BYREF
  __m128i *v138; // [rsp+80h] [rbp-760h] BYREF
  __int64 v139; // [rsp+88h] [rbp-758h]
  _BYTE v140[48]; // [rsp+90h] [rbp-750h] BYREF
  _QWORD v141[3]; // [rsp+C0h] [rbp-720h] BYREF
  __int64 v142; // [rsp+D8h] [rbp-708h]
  __int64 v143; // [rsp+E0h] [rbp-700h]
  __int64 v144; // [rsp+E8h] [rbp-6F8h]
  _QWORD *v145; // [rsp+F0h] [rbp-6F0h]
  __int64 v146; // [rsp+F8h] [rbp-6E8h]
  __int64 v147; // [rsp+100h] [rbp-6E0h]
  __int64 v148; // [rsp+108h] [rbp-6D8h]
  __int64 v149; // [rsp+110h] [rbp-6D0h]
  char *v150; // [rsp+118h] [rbp-6C8h]
  __int64 v151; // [rsp+120h] [rbp-6C0h]
  int v152; // [rsp+128h] [rbp-6B8h]
  char v153; // [rsp+12Ch] [rbp-6B4h]
  char v154; // [rsp+130h] [rbp-6B0h] BYREF
  int v155; // [rsp+170h] [rbp-670h]
  unsigned __int8 *v156[3]; // [rsp+180h] [rbp-660h] BYREF
  char v157; // [rsp+19Ch] [rbp-644h]
  unsigned __int64 v158; // [rsp+1E0h] [rbp-600h]
  unsigned __int64 v159; // [rsp+200h] [rbp-5E0h]
  char v160; // [rsp+214h] [rbp-5CCh]
  unsigned __int64 v161; // [rsp+258h] [rbp-588h]
  __int64 v162; // [rsp+270h] [rbp-570h] BYREF
  unsigned __int64 v163; // [rsp+278h] [rbp-568h]
  __int64 v164; // [rsp+280h] [rbp-560h]
  unsigned int j; // [rsp+288h] [rbp-558h]
  char v166; // [rsp+28Ch] [rbp-554h]
  _BYTE v167[64]; // [rsp+290h] [rbp-550h] BYREF
  unsigned __int64 v168; // [rsp+2D0h] [rbp-510h]
  __int64 v169; // [rsp+2D8h] [rbp-508h]
  unsigned __int64 v170; // [rsp+2E0h] [rbp-500h]
  __int16 v171; // [rsp+2E8h] [rbp-4F8h]
  char v172[8]; // [rsp+2F0h] [rbp-4F0h] BYREF
  unsigned __int64 v173; // [rsp+2F8h] [rbp-4E8h]
  char v174; // [rsp+30Ch] [rbp-4D4h]
  _BYTE v175[64]; // [rsp+310h] [rbp-4D0h] BYREF
  __int64 v176; // [rsp+350h] [rbp-490h]
  __int64 v177; // [rsp+358h] [rbp-488h]
  unsigned __int64 v178; // [rsp+360h] [rbp-480h]
  __int16 v179; // [rsp+368h] [rbp-478h]
  __int16 v180; // [rsp+378h] [rbp-468h]
  __int8 *v181; // [rsp+380h] [rbp-460h] BYREF
  unsigned __int64 v182; // [rsp+388h] [rbp-458h]
  char v183; // [rsp+390h] [rbp-450h] BYREF
  char v184; // [rsp+39Ch] [rbp-444h]
  _BYTE v185[64]; // [rsp+3A0h] [rbp-440h] BYREF
  __int64 v186; // [rsp+3E0h] [rbp-400h]
  __int64 v187; // [rsp+3E8h] [rbp-3F8h]
  unsigned __int64 v188; // [rsp+3F0h] [rbp-3F0h]
  __int16 v189; // [rsp+3F8h] [rbp-3E8h]
  char v190[8]; // [rsp+400h] [rbp-3E0h] BYREF
  unsigned __int64 v191; // [rsp+408h] [rbp-3D8h]
  char v192; // [rsp+41Ch] [rbp-3C4h]
  _BYTE v193[64]; // [rsp+420h] [rbp-3C0h] BYREF
  unsigned __int64 v194; // [rsp+460h] [rbp-380h]
  __int64 v195; // [rsp+468h] [rbp-378h]
  unsigned __int64 v196; // [rsp+470h] [rbp-370h]
  __int16 v197; // [rsp+478h] [rbp-368h]
  __int16 v198; // [rsp+488h] [rbp-358h]
  unsigned __int64 v199; // [rsp+490h] [rbp-350h]
  char v200; // [rsp+4A4h] [rbp-33Ch]
  unsigned __int64 v201; // [rsp+4E8h] [rbp-2F8h]
  unsigned __int64 v202; // [rsp+510h] [rbp-2D0h]
  char v203; // [rsp+524h] [rbp-2BCh]
  unsigned __int64 v204; // [rsp+568h] [rbp-278h]
  void *v205; // [rsp+590h] [rbp-250h] BYREF
  unsigned __int64 v206; // [rsp+598h] [rbp-248h]
  __int64 v207; // [rsp+5A0h] [rbp-240h]
  __int64 v208; // [rsp+5A8h] [rbp-238h]
  __m128i *v209; // [rsp+5B0h] [rbp-230h]
  __int64 v210; // [rsp+5B8h] [rbp-228h]
  __int8 **v211; // [rsp+5C0h] [rbp-220h]
  unsigned __int64 v212; // [rsp+5F0h] [rbp-1F0h]
  __int64 v213; // [rsp+5F8h] [rbp-1E8h]
  __int16 v214; // [rsp+608h] [rbp-1D8h]
  char v215[8]; // [rsp+610h] [rbp-1D0h] BYREF
  unsigned __int64 v216; // [rsp+618h] [rbp-1C8h]
  char v217; // [rsp+62Ch] [rbp-1B4h]
  unsigned __int64 v218; // [rsp+670h] [rbp-170h]
  __int64 v219; // [rsp+678h] [rbp-168h]
  __int16 v220; // [rsp+688h] [rbp-158h]
  __int16 v221; // [rsp+698h] [rbp-148h]
  char v222[8]; // [rsp+6A0h] [rbp-140h] BYREF
  unsigned __int64 v223; // [rsp+6A8h] [rbp-138h]
  char v224; // [rsp+6BCh] [rbp-124h]
  unsigned __int64 v225; // [rsp+700h] [rbp-E0h]
  __int64 v226; // [rsp+708h] [rbp-D8h]
  __int16 v227; // [rsp+718h] [rbp-C8h]
  char v228[8]; // [rsp+720h] [rbp-C0h] BYREF
  unsigned __int64 v229; // [rsp+728h] [rbp-B8h]
  char v230; // [rsp+73Ch] [rbp-A4h]
  unsigned __int64 v231; // [rsp+780h] [rbp-60h]
  __int64 v232; // [rsp+788h] [rbp-58h]
  __int16 v233; // [rsp+798h] [rbp-48h]
  __int16 v234; // [rsp+7A8h] [rbp-38h]

  v129 = (__m128i *)v140;
  v2 = *(unsigned int *)(a1 + 96);
  v138 = (__m128i *)v140;
  v139 = 0x300000000LL;
  v3 = *(__int64 **)(a1 + 88);
  v130 = (__int64 *)a1;
  v128 = &v3[v2];
  if ( v3 != v128 )
  {
    v132 = (unsigned __int64)v3;
    do
    {
      v4 = *(unsigned int *)(*(_QWORD *)v132 + 88LL);
      v133 = *(__m128i **)(*(_QWORD *)v132 + 80LL);
      for ( i = (__m128i *)((char *)v133 + 8 * v4); i != v133; v133 = (__m128i *)((char *)v133 + 8) )
      {
        v5 = v133->m128i_i8[4];
        v6 = v133->m128i_i32[0];
        v136 = v133->m128i_i64[0];
        if ( v5 || v6 != 1 )
        {
          v7 = (_QWORD *)v132;
          v141[2] = 0;
          v142 = 0;
          v8 = v130[6];
          v9 = v130[5];
          v143 = 0;
          v144 = 0;
          v10 = *(_QWORD *)(v8 + 456);
          v11 = *(_QWORD **)(v9 + 336);
          v12 = *(_DWORD *)(v8 + 992);
          v141[0] = *(_QWORD *)(v8 + 448);
          v141[1] = v10;
          v145 = v11;
          v13 = *v11;
          v148 = v8;
          v146 = v13;
          v147 = v13;
          v155 = v12;
          v149 = 0;
          v150 = &v154;
          v151 = 8;
          v152 = 0;
          v153 = 1;
          sub_2ADF280(v130, *(_QWORD *)v132, v136, (__int64)v141);
          v14 = sub_2BF3F10(*v7);
          sub_2AC67F0(v156, *(_QWORD *)(v14 + 112));
          v134 = (__int64)&v162;
          sub_2AB1B90((__int64)&v162, (__int64)v156, v15, v16, v17, v18);
          sub_2AD74E0((__int64)&v181, (__int64)&v162, v19, v20, v21, v22);
          sub_2ABCD20((__int64)&v205, &v181, v23, v24, v25, v26);
          if ( v204 )
            j_j___libc_free_0(v204);
          if ( !v203 )
            _libc_free(v202);
          if ( v201 )
            j_j___libc_free_0(v201);
          if ( !v200 )
            _libc_free(v199);
          if ( v194 )
            j_j___libc_free_0(v194);
          if ( !v192 )
            _libc_free(v191);
          if ( v186 )
            j_j___libc_free_0(v186);
          if ( !v184 )
            _libc_free(v182);
          if ( v176 )
            j_j___libc_free_0(v176);
          if ( !v174 )
            _libc_free(v173);
          if ( v168 )
            j_j___libc_free_0(v168);
          if ( !v166 )
            _libc_free(v163);
          v30 = (__int8 **)v134;
          v31 = v167;
          sub_C8CD80(v134, (__int64)v167, (__int64)&v205, v27, v28, v29);
          v34 = v213;
          v35 = v212;
          v168 = 0;
          v169 = 0;
          v170 = 0;
          v36 = v213 - v212;
          if ( v213 == v212 )
          {
            v37 = 0;
          }
          else
          {
            if ( v36 > 0x7FFFFFFFFFFFFFE0LL )
              goto LABEL_253;
            v37 = sub_22077B0(v213 - v212);
            v34 = v213;
            v35 = v212;
          }
          v168 = v37;
          v169 = v37;
          v170 = v37 + v36;
          if ( v35 == v34 )
          {
            v38 = v37;
          }
          else
          {
            v38 = v37 + v34 - v35;
            do
            {
              if ( v37 )
              {
                v39 = *(_QWORD *)v35;
                *(_BYTE *)(v37 + 24) = 0;
                *(_QWORD *)v37 = v39;
                if ( *(_BYTE *)(v35 + 24) )
                {
                  *(_QWORD *)(v37 + 8) = *(_QWORD *)(v35 + 8);
                  v40 = *(_QWORD *)(v35 + 16);
                  *(_BYTE *)(v37 + 24) = 1;
                  *(_QWORD *)(v37 + 16) = v40;
                }
              }
              v37 += 32;
              v35 += 32LL;
            }
            while ( v37 != v38 );
          }
          v31 = v175;
          v169 = v38;
          v30 = (__int8 **)v172;
          v171 = v214;
          sub_C8CD80((__int64)v172, (__int64)v175, (__int64)v215, v38, v32, v33);
          v43 = v219;
          v35 = v218;
          v176 = 0;
          v177 = 0;
          v178 = 0;
          v44 = v219 - v218;
          if ( v219 == v218 )
          {
            v45 = 0;
          }
          else
          {
            if ( v44 > 0x7FFFFFFFFFFFFFE0LL )
              goto LABEL_253;
            v45 = sub_22077B0(v219 - v218);
            v43 = v219;
            v35 = v218;
          }
          v176 = v45;
          v177 = v45;
          v178 = v45 + v44;
          if ( v35 == v43 )
          {
            v46 = v45;
          }
          else
          {
            v46 = v45 + v43 - v35;
            do
            {
              if ( v45 )
              {
                v47 = *(_QWORD *)v35;
                *(_BYTE *)(v45 + 24) = 0;
                *(_QWORD *)v45 = v47;
                if ( *(_BYTE *)(v35 + 24) )
                {
                  *(_QWORD *)(v45 + 8) = *(_QWORD *)(v35 + 8);
                  v48 = *(_QWORD *)(v35 + 16);
                  *(_BYTE *)(v45 + 24) = 1;
                  *(_QWORD *)(v45 + 16) = v48;
                }
              }
              v45 += 32;
              v35 += 32LL;
            }
            while ( v45 != v46 );
          }
          v30 = &v181;
          v31 = v185;
          v177 = v46;
          v179 = v220;
          v180 = v221;
          sub_C8CD80((__int64)&v181, (__int64)v185, (__int64)v222, v46, v41, v42);
          v51 = v226;
          v35 = v225;
          v186 = 0;
          v187 = 0;
          v188 = 0;
          v52 = v226 - v225;
          if ( v226 == v225 )
          {
            v53 = 0;
          }
          else
          {
            if ( v52 > 0x7FFFFFFFFFFFFFE0LL )
              goto LABEL_253;
            v53 = sub_22077B0(v226 - v225);
            v51 = v226;
            v35 = v225;
          }
          v186 = v53;
          v187 = v53;
          v188 = v53 + v52;
          if ( v35 == v51 )
          {
            v54 = v53;
          }
          else
          {
            v54 = v53 + v51 - v35;
            do
            {
              if ( v53 )
              {
                v55 = *(_QWORD *)v35;
                *(_BYTE *)(v53 + 24) = 0;
                *(_QWORD *)v53 = v55;
                if ( *(_BYTE *)(v35 + 24) )
                {
                  *(_QWORD *)(v53 + 8) = *(_QWORD *)(v35 + 8);
                  v56 = *(_QWORD *)(v35 + 16);
                  *(_BYTE *)(v53 + 24) = 1;
                  *(_QWORD *)(v53 + 16) = v56;
                }
              }
              v53 += 32;
              v35 += 32LL;
            }
            while ( v53 != v54 );
          }
          v31 = v193;
          v187 = v54;
          v30 = (__int8 **)v190;
          v189 = v227;
          sub_C8CD80((__int64)v190, (__int64)v193, (__int64)v228, v54, v49, v50);
          v59 = v232;
          v35 = v231;
          v194 = 0;
          v195 = 0;
          v196 = 0;
          v60 = v232 - v231;
          if ( v232 == v231 )
          {
            v61 = 0;
          }
          else
          {
            if ( v60 > 0x7FFFFFFFFFFFFFE0LL )
LABEL_253:
              sub_4261EA(v30, v31, v35);
            v61 = sub_22077B0(v232 - v231);
            v59 = v232;
            v35 = v231;
          }
          v194 = v61;
          v195 = v61;
          v196 = v61 + v60;
          if ( v59 == v35 )
          {
            v62 = v61;
          }
          else
          {
            v62 = v61 + v59 - v35;
            do
            {
              if ( v61 )
              {
                v63 = *(_QWORD *)v35;
                *(_BYTE *)(v61 + 24) = 0;
                *(_QWORD *)v61 = v63;
                if ( *(_BYTE *)(v35 + 24) )
                {
                  *(_QWORD *)(v61 + 8) = *(_QWORD *)(v35 + 8);
                  v64 = *(_QWORD *)(v35 + 16);
                  *(_BYTE *)(v61 + 24) = 1;
                  *(_QWORD *)(v61 + 16) = v64;
                }
              }
              v61 += 32;
              v35 += 32LL;
            }
            while ( v61 != v62 );
          }
          v195 = v62;
          v65 = v168;
          v197 = v233;
          v198 = v234;
          v66 = v169;
LABEL_67:
          v67 = v186;
          v68 = v187 - v186;
          if ( v66 - v65 != v187 - v186 )
          {
LABEL_68:
            v69 = *(__m128i **)(v66 - 32);
            v70 = (__m128i *)v69[7].m128i_i64[1];
            v135 = v69 + 7;
            if ( &v69[7] == v70 )
              goto LABEL_78;
            while ( 1 )
            {
LABEL_71:
              v71 = 0;
              if ( v70 )
                v71 = (__int64)&v70[-2].m128i_i64[1];
              LODWORD(v136) = v6;
              BYTE4(v136) = v5;
              v67 = v136;
              sub_2C19F20(v71, v136, v141);
              if ( !(_DWORD)v65 )
                goto LABEL_70;
              v72 = (unsigned int)v139;
              v67 = HIDWORD(v139);
              v73 = v139;
              if ( (unsigned int)v139 >= (unsigned __int64)HIDWORD(v139) )
                break;
              v74 = &v138[(unsigned int)v139];
              if ( v74 )
              {
                v75 = v136;
                v74->m128i_i64[0] = v71;
                v74->m128i_i64[1] = v75;
                v73 = v139;
              }
              v65 = (unsigned int)(v73 + 1);
              LODWORD(v139) = v65;
              v70 = (__m128i *)v70->m128i_i64[1];
              if ( v135 == v70 )
              {
                while ( 1 )
                {
LABEL_78:
                  sub_2AD7320(v134, v67, v65, v68, v57, v58);
                  v66 = v169;
                  v65 = v168;
                  v67 = v176;
                  if ( v169 - v168 == v177 - v176 )
                  {
                    if ( v168 == v169 )
                      goto LABEL_67;
                    v76 = v168;
                    while ( *(_QWORD *)v76 == *(_QWORD *)v67 )
                    {
                      v77 = *(_BYTE *)(v76 + 24);
                      if ( v77 != *(_BYTE *)(v67 + 24)
                        || v77
                        && (*(_QWORD *)(v76 + 8) != *(_QWORD *)(v67 + 8)
                         || *(_QWORD *)(v76 + 16) != *(_QWORD *)(v67 + 16)) )
                      {
                        break;
                      }
                      v76 += 32LL;
                      v67 += 32;
                      if ( v169 == v76 )
                        goto LABEL_67;
                    }
                  }
                  v68 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v169 - 32) + 8LL) - 1;
                  if ( (unsigned int)v68 <= 1 )
                    goto LABEL_67;
                }
              }
            }
            v65 = (unsigned int)v139 + 1LL;
            v137.m128i_i64[0] = v71;
            v137.m128i_i32[2] = v6;
            v137.m128i_i8[12] = v5;
            v78 = _mm_load_si128(&v137);
            if ( HIDWORD(v139) < v65 )
            {
              v67 = (__int64)v129;
              v127 = v78;
              sub_C8D5F0((__int64)&v138, v129, v65, 0x10u, v57, v58);
              v72 = (unsigned int)v139;
              v78 = _mm_load_si128(&v127);
            }
            v138[v72] = v78;
            LODWORD(v139) = v139 + 1;
LABEL_70:
            v70 = (__m128i *)v70->m128i_i64[1];
            if ( v135 == v70 )
              goto LABEL_78;
            goto LABEL_71;
          }
          while ( v66 != v65 )
          {
            if ( *(_QWORD *)v65 != *(_QWORD *)v67 )
              goto LABEL_68;
            v68 = *(unsigned __int8 *)(v65 + 24);
            if ( (_BYTE)v68 != *(_BYTE *)(v67 + 24)
              || (_BYTE)v68
              && (*(_QWORD *)(v65 + 8) != *(_QWORD *)(v67 + 8) || *(_QWORD *)(v65 + 16) != *(_QWORD *)(v67 + 16)) )
            {
              goto LABEL_68;
            }
            v65 += 32LL;
            v67 += 32;
          }
          if ( v194 )
            j_j___libc_free_0(v194);
          if ( !v192 )
            _libc_free(v191);
          if ( v186 )
            j_j___libc_free_0(v186);
          if ( !v184 )
            _libc_free(v182);
          if ( v176 )
            j_j___libc_free_0(v176);
          if ( !v174 )
            _libc_free(v173);
          if ( v168 )
            j_j___libc_free_0(v168);
          if ( !v166 )
            _libc_free(v163);
          if ( v231 )
            j_j___libc_free_0(v231);
          if ( !v230 )
            _libc_free(v229);
          if ( v225 )
            j_j___libc_free_0(v225);
          if ( !v224 )
            _libc_free(v223);
          if ( v218 )
            j_j___libc_free_0(v218);
          if ( !v217 )
            _libc_free(v216);
          if ( v212 )
            j_j___libc_free_0(v212);
          if ( !BYTE4(v208) )
            _libc_free(v206);
          if ( v161 )
            j_j___libc_free_0(v161);
          if ( !v160 )
            _libc_free(v159);
          if ( v158 )
            j_j___libc_free_0(v158);
          if ( !v157 )
            _libc_free((unsigned __int64)v156[1]);
          if ( !v153 )
            _libc_free((unsigned __int64)v150);
          sub_C7D6A0(v142, 16LL * (unsigned int)v144, 8);
        }
      }
      v132 += 8LL;
    }
    while ( v128 != (__int64 *)v132 );
    v79 = v138;
    if ( (_DWORD)v139 )
    {
      v80 = (__int64 *)v138;
      v81 = 0;
      v82 = 0;
      v162 = 0;
      v163 = 0;
      v83 = &v138[(unsigned int)v139];
      v84 = 0;
      v164 = 0;
      for ( j = 0; ; v81 = j )
      {
        v88 = *v80;
        if ( v81 )
        {
          v85 = v81 - 1;
          v86 = (v81 - 1) & (((unsigned int)v88 >> 9) ^ ((unsigned int)v88 >> 4));
          v87 = *(_QWORD *)(v84 + 16LL * v86);
          if ( v88 == v87 )
          {
LABEL_149:
            v80 += 2;
            if ( v83 == (__m128i *)v80 )
              goto LABEL_153;
            goto LABEL_150;
          }
          v116 = 1;
          while ( v87 != -4096 )
          {
            v86 = v85 & (v116 + v86);
            v87 = *(_QWORD *)(v84 + 16LL * v86);
            if ( v88 == v87 )
              goto LABEL_149;
            ++v116;
          }
        }
        v89 = v80;
        v80 += 2;
        *(_DWORD *)sub_2ACCA00((__int64)&v162, v89) = v82++;
        if ( v83 == (__m128i *)v80 )
        {
LABEL_153:
          v90 = v138;
          v132 = (unsigned int)v139;
          v91 = 16LL * (unsigned int)v139;
          v92 = v91;
          v135 = (__m128i *)((char *)v138 + v91);
          v93 = (__m128i *)((char *)v138 + v91);
          if ( v138 != (__m128i *)&v138->m128i_i8[v91] )
          {
            v94 = (unsigned __int64)v138->m128i_u64 + v91;
            _BitScanReverse64((unsigned __int64 *)&v91, v91 >> 4);
            sub_2ACDB80((__int64)v138, v94, 2LL * (int)(63 - (v91 ^ 0x3F)), (__int64)&v162);
            if ( v92 <= 0x100 )
            {
              sub_2ACD0B0(v90, v135, (__int64)&v162);
            }
            else
            {
              v95 = v90 + 16;
              sub_2ACD0B0(v90, v90 + 16, (__int64)&v162);
              if ( v93 != &v90[16] )
              {
                do
                {
                  v96 = v95++;
                  sub_2ACCB40(v96, (__int64)&v162);
                }
                while ( v135 != v95 );
              }
            }
            v135 = v138;
            v132 = (unsigned int)v139;
          }
          v134 = 0;
          v97 = 0;
          while ( 1 )
          {
            if ( !v134 )
            {
              v97 = v135;
              v134 = v132 != 0;
            }
            v99 = v97->m128i_i64[0];
            v100 = *(unsigned __int8 *)(v97->m128i_i64[0] + 8);
            if ( (unsigned int)(v100 - 29) <= 7 )
            {
              v102 = 1;
            }
            else
            {
              if ( (_BYTE)v100 == 24 )
              {
                v102 = 1;
                v101 = 57;
LABEL_222:
                if ( (_BYTE)v100 == 4 && !v102 )
                {
                  v101 = *(unsigned __int8 *)(v99 + 160);
                  goto LABEL_171;
                }
LABEL_168:
                if ( (_BYTE)v100 == 23 && !v102 )
                {
LABEL_170:
                  v101 = *(_DWORD *)(v99 + 160);
                  goto LABEL_171;
                }
LABEL_235:
                if ( (_BYTE)v100 == 9 && !v102 )
                {
                  v101 = **(unsigned __int8 **)(v99 + 136) - 29;
                  goto LABEL_171;
                }
LABEL_240:
                if ( (_BYTE)v100 == 16 && !v102 )
                  goto LABEL_170;
                goto LABEL_230;
              }
              v101 = 33;
              v102 = 1;
              switch ( (_BYTE)v100 )
              {
                case 0x16:
                  goto LABEL_168;
                case 0x14:
                  v101 = 32;
                  goto LABEL_235;
                case 0xE:
                  v101 = 56;
                  goto LABEL_240;
              }
              v102 = 0;
            }
            if ( (_BYTE)v100 != 18 || v102 )
            {
              v101 = 55;
              goto LABEL_222;
            }
            v101 = 56;
            v102 = 1;
LABEL_230:
            if ( (_BYTE)v100 == 5 && !v102 )
              v101 = (*(_DWORD *)(v99 + 56) != 1 - ((*(_BYTE *)(v99 + 104) == 0) - 1)) + 32;
LABEL_171:
            v103 = v134;
            if ( v132 == v134 )
            {
              v133 = &v97[v103];
              if ( v97 == &v97[v103] )
              {
LABEL_251:
                v135 = (__m128i *)((char *)v135 + v103 * 16);
                goto LABEL_177;
              }
              v104 = v135;
              v105 = v99;
              for ( k = v97;
                    v105 == v104->m128i_i64[0]
                 && k->m128i_i32[2] == v104->m128i_i32[2]
                 && k->m128i_i8[12] == v104->m128i_i8[12];
                    v105 = k->m128i_i64[0] )
              {
                ++k;
                ++v104;
                if ( v133 == k )
                  goto LABEL_251;
              }
            }
            if ( v99 == v135[v134].m128i_i64[0] )
            {
              v97 = v135;
              v98 = v134 + 1;
              if ( v134 + 1 > v132 )
                v98 = v132;
              v134 = v98;
              goto LABEL_162;
            }
            v135 = (__m128i *)((char *)v135 + v103 * 16);
            v133 = &v97[v103];
LABEL_177:
            v181 = &v183;
            v210 = 0x100000000LL;
            i = (__m128i *)&v181;
            v211 = &v181;
            v205 = &unk_49DD210;
            v182 = 0;
            v183 = 0;
            v206 = 0;
            v207 = 0;
            v208 = 0;
            v209 = 0;
            sub_CB5980((__int64)&v205, 0, 0, 0);
            v107 = v209;
            if ( (unsigned __int64)(v208 - (_QWORD)v209) <= 0x38 )
            {
              sub_CB6200((__int64)&v205, "Recipe with invalid costs prevented vectorization at VF=(", 0x39u);
            }
            else
            {
              si128 = _mm_load_si128((const __m128i *)&xmmword_439F1B0);
              v209[3].m128i_i8[8] = 40;
              v107[3].m128i_i64[0] = 0x3D4656207461206ELL;
              *v107 = si128;
              v107[1] = _mm_load_si128((const __m128i *)&xmmword_439F1C0);
              v107[2] = _mm_load_si128((const __m128i *)&xmmword_439F1D0);
              v209 = (__m128i *)((char *)v209 + 57);
            }
            if ( v97 != v133 )
            {
              v109 = v101;
              v110 = v99;
              v111 = v97;
              v112 = v109;
              do
              {
                if ( v111->m128i_i32[2] != v97->m128i_i32[2]
                  || (v114 = v97->m128i_i8[12], v113 = &v205, v111->m128i_i8[12] != v114) )
                {
                  if ( (unsigned __int64)(v208 - (_QWORD)v209) > 1 )
                  {
                    v113 = &v205;
                    v209->m128i_i16[0] = 8236;
                    v209 = (__m128i *)((char *)v209 + 2);
                  }
                  else
                  {
                    v113 = (void **)sub_CB6200((__int64)&v205, (unsigned __int8 *)", ", 2u);
                  }
                  v114 = v111->m128i_i8[12];
                }
                if ( v114 )
                {
                  v117 = v113[4];
                  if ( (unsigned __int64)((_BYTE *)v113[3] - v117) <= 8 )
                  {
                    v127.m128i_i64[0] = (__int64)v113;
                    sub_CB6200((__int64)v113, "vscale x ", 9u);
                    v113 = (void **)v127.m128i_i64[0];
                  }
                  else
                  {
                    v117[8] = 32;
                    *(_QWORD *)v117 = 0x7820656C61637376LL;
                    v113[4] = (char *)v113[4] + 9;
                  }
                }
                v115 = v111->m128i_u32[2];
                ++v111;
                sub_CB59D0((__int64)v113, v115);
              }
              while ( v111 != v133 );
              v99 = v110;
              v101 = v112;
            }
            if ( (unsigned __int64)(v208 - (_QWORD)v209) <= 1 )
            {
              sub_CB6200((__int64)&v205, "):", 2u);
            }
            else
            {
              v209->m128i_i16[0] = 14889;
              v209 = (__m128i *)((char *)v209 + 2);
            }
            if ( v101 == 56 )
            {
              if ( *(_BYTE *)(v99 + 8) == 18 )
                v119 = (unsigned __int8 *)sub_2C1AC70(v99);
              else
                v119 = (unsigned __int8 *)sub_BD5D20(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v99 + 48)
                                                                           + 8LL
                                                                           * (unsigned int)(*(_DWORD *)(v99 + 56) - 1))
                                                               + 40LL));
              v120 = v118;
              v121 = sub_904010((__int64)&v205, " call to ");
              v122 = *(void **)(v121 + 32);
              v123 = v121;
              if ( v120 > *(_QWORD *)(v121 + 24) - (_QWORD)v122 )
              {
                sub_CB6200(v121, v119, v120);
              }
              else if ( v120 )
              {
                memcpy(v122, v119, v120);
                *(_QWORD *)(v123 + 32) += v120;
              }
            }
            else
            {
              v124 = sub_904010((__int64)&v205, " ");
              v125 = sub_B458E0(v101);
              sub_904010(v124, v125);
            }
            v156[0] = *(unsigned __int8 **)(v99 + 88);
            if ( v156[0] )
              sub_2AAAFA0((__int64 *)v156);
            v97 = 0;
            sub_2AB8CE0(v181, v182, (__int64)"InvalidCost", 11, a2, *v130, v156);
            sub_9C6650(v156);
            v132 -= v134;
            v205 = &unk_49DD210;
            sub_CB5840((__int64)&v205);
            sub_2240A30((unsigned __int64 *)i);
            v134 = 0;
LABEL_162:
            if ( !v132 )
            {
              sub_C7D6A0(v163, 16LL * j, 8);
              v79 = v138;
              goto LABEL_190;
            }
          }
        }
LABEL_150:
        v84 = v163;
      }
    }
LABEL_190:
    if ( v79 != v129 )
      _libc_free((unsigned __int64)v79);
  }
}
