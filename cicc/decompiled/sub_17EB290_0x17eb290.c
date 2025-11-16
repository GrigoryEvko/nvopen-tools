// Function: sub_17EB290
// Address: 0x17eb290
//
__int64 __fastcall sub_17EB290(__int64 *a1, __int64 *a2, __int64 a3, __int64 *a4, _QWORD *a5)
{
  char *v8; // rax
  __int64 v9; // rbx
  __int64 v10; // rax
  unsigned __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // r15
  unsigned __int64 v14; // r14
  unsigned __int64 v15; // rbx
  unsigned int i; // r12d
  __int64 v17; // r13
  unsigned __int64 v18; // r15
  __int64 v19; // r9
  __int64 v20; // r15
  unsigned __int64 v21; // rax
  __int64 v22; // rax
  unsigned __int64 v23; // rdi
  int v24; // eax
  unsigned __int64 v25; // rdi
  char *v26; // rbx
  char *v27; // r15
  __int64 v28; // r14
  __int64 v29; // r13
  __int64 *v30; // rax
  __int64 *v31; // r12
  __int64 v32; // rcx
  __int64 *v33; // r14
  __int64 *v34; // rax
  __int64 v35; // rdx
  __int64 *v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rdi
  __int64 *v39; // rbx
  __int64 **v40; // r12
  __int64 **v41; // rbx
  __int64 *v42; // rax
  __int64 v43; // rdi
  __int64 **v44; // r12
  char *v45; // rbx
  __int64 *v46; // rax
  __int64 v47; // rsi
  _BYTE *v48; // rdi
  __int64 *v49; // rsi
  __int64 v50; // rdi
  unsigned int v51; // r13d
  unsigned __int64 v52; // r12
  int v53; // ebx
  __int64 v54; // rsi
  unsigned int v55; // ecx
  __int64 *v56; // rdx
  __int64 v57; // r9
  __int64 v58; // rax
  unsigned int v59; // r15d
  int j; // ecx
  size_t v61; // rsi
  size_t v62; // rsi
  size_t v63; // rdx
  __int64 ***v64; // rax
  int v65; // ecx
  char *k; // rax
  __int64 v67; // rax
  char *v68; // rbx
  unsigned __int64 *v69; // rsi
  __int64 v70; // rcx
  __int64 v71; // r12
  __int64 v72; // r15
  unsigned int v73; // eax
  __int64 v74; // r15
  _QWORD *v75; // rax
  __int64 *v76; // r15
  __int64 *v77; // r14
  __int64 *v78; // rax
  __int64 *v79; // r12
  __int64 *v80; // r15
  __int64 *v81; // rsi
  __int64 v82; // rsi
  __int64 *v83; // rbx
  unsigned __int64 v84; // rax
  __int64 *v85; // rax
  __int64 v86; // r12
  _QWORD *v87; // rax
  unsigned __int64 *v88; // rsi
  __int64 v89; // rax
  __int64 v90; // rax
  __int64 **v91; // rdx
  __int64 v92; // rax
  __int64 v93; // rax
  __int64 v94; // rax
  __int64 v95; // rdx
  unsigned __int8 v96; // cl
  int v97; // edx
  int v98; // r10d
  unsigned __int64 v99; // rcx
  __int64 v100; // rsi
  __int64 v101; // rax
  __int64 v102; // rsi
  __int64 v103; // rdx
  unsigned __int8 *v104; // rsi
  __int64 v105; // r14
  __int64 v106; // rax
  __int64 v107; // rbx
  __int64 v108; // r15
  __int64 v109; // rax
  __int64 v110; // rdx
  int v111; // edx
  _QWORD *v112; // rbx
  _QWORD *v113; // r12
  __int64 v114; // rdi
  __int64 result; // rax
  char *v116; // rbx
  char *v117; // r12
  char *v118; // rbx
  char *v119; // r12
  __int64 v120; // rdx
  char *v121; // rsi
  __m128i v122; // rax
  char v123; // al
  void **v124; // rdx
  size_t v125; // rdx
  _QWORD *v126; // rbx
  _QWORD *v127; // r12
  __int64 v128; // rdi
  char *v129; // rbx
  char *v130; // r12
  char *v131; // rbx
  __int64 v132; // rcx
  char v133; // al
  __m128i *v134; // rdx
  _BYTE *v135; // rdi
  __int64 *v136; // rsi
  __int64 v137; // r15
  _QWORD *v138; // rdi
  __m128i v139; // rax
  char v140; // al
  __m128i *v141; // rdx
  __int64 v142; // r15
  _QWORD *v143; // rdx
  _QWORD *v144; // r13
  _QWORD *v145; // r12
  __int64 v146; // r14
  char v147; // al
  __int64 v148; // rdx
  char *v149; // rsi
  __int64 v150; // rdx
  char v151; // al
  __m128i *v152; // rdx
  size_t v153; // rdx
  __int64 v154; // rax
  __int64 *v155; // rdx
  __int64 v156; // rcx
  __int64 v157; // [rsp+8h] [rbp-358h]
  __int64 v159; // [rsp+20h] [rbp-340h]
  unsigned __int64 v160; // [rsp+28h] [rbp-338h]
  __int64 v162; // [rsp+38h] [rbp-328h]
  unsigned __int64 v163; // [rsp+40h] [rbp-320h]
  __int64 *v164; // [rsp+48h] [rbp-318h]
  __int64 *v166; // [rsp+60h] [rbp-300h]
  __int64 v167; // [rsp+68h] [rbp-2F8h]
  bool v168; // [rsp+77h] [rbp-2E9h]
  unsigned __int64 v169; // [rsp+78h] [rbp-2E8h]
  int v170; // [rsp+88h] [rbp-2D8h]
  int v171; // [rsp+88h] [rbp-2D8h]
  __int64 v172; // [rsp+90h] [rbp-2D0h]
  __int64 *v173; // [rsp+90h] [rbp-2D0h]
  unsigned __int64 src; // [rsp+98h] [rbp-2C8h]
  void *srca; // [rsp+98h] [rbp-2C8h]
  __int64 *srcb; // [rsp+98h] [rbp-2C8h]
  __int64 v177; // [rsp+A0h] [rbp-2C0h]
  __int64 *v178; // [rsp+A0h] [rbp-2C0h]
  __int64 **v179; // [rsp+A0h] [rbp-2C0h]
  unsigned __int64 v180; // [rsp+A8h] [rbp-2B8h]
  __int64 *v181; // [rsp+A8h] [rbp-2B8h]
  __int64 v182; // [rsp+A8h] [rbp-2B8h]
  __int64 v183; // [rsp+A8h] [rbp-2B8h]
  __int64 v184; // [rsp+A8h] [rbp-2B8h]
  __int64 v185; // [rsp+B0h] [rbp-2B0h]
  __int64 v186; // [rsp+B0h] [rbp-2B0h]
  __int64 *v187; // [rsp+B0h] [rbp-2B0h]
  char v188; // [rsp+B8h] [rbp-2A8h]
  __int64 v189; // [rsp+B8h] [rbp-2A8h]
  char *v190; // [rsp+B8h] [rbp-2A8h]
  unsigned int m; // [rsp+B8h] [rbp-2A8h]
  int v192; // [rsp+B8h] [rbp-2A8h]
  __int64 v193; // [rsp+C8h] [rbp-298h] BYREF
  _QWORD v194[2]; // [rsp+D0h] [rbp-290h] BYREF
  __m128i v195; // [rsp+E0h] [rbp-280h] BYREF
  __int64 *v196; // [rsp+F0h] [rbp-270h]
  __m128i v197; // [rsp+100h] [rbp-260h] BYREF
  __int64 *v198; // [rsp+110h] [rbp-250h]
  __m128i v199; // [rsp+120h] [rbp-240h] BYREF
  __int64 *v200; // [rsp+130h] [rbp-230h]
  __int64 v201[2]; // [rsp+140h] [rbp-220h] BYREF
  _QWORD v202[2]; // [rsp+150h] [rbp-210h] BYREF
  void *v203[2]; // [rsp+160h] [rbp-200h] BYREF
  __int64 v204; // [rsp+170h] [rbp-1F0h] BYREF
  unsigned __int64 *v205; // [rsp+180h] [rbp-1E0h] BYREF
  size_t v206; // [rsp+188h] [rbp-1D8h]
  _QWORD v207[4]; // [rsp+190h] [rbp-1D0h] BYREF
  size_t n[2]; // [rsp+1B0h] [rbp-1B0h] BYREF
  __int64 *v209; // [rsp+1C0h] [rbp-1A0h] BYREF
  _QWORD *v210; // [rsp+1C8h] [rbp-198h]
  __int64 v211; // [rsp+1D0h] [rbp-190h]
  int v212; // [rsp+1D8h] [rbp-188h]
  __int64 v213; // [rsp+1E0h] [rbp-180h]
  __int64 v214; // [rsp+1E8h] [rbp-178h]
  __int64 *v215; // [rsp+200h] [rbp-160h]
  _QWORD *v216; // [rsp+208h] [rbp-158h]
  char *v217; // [rsp+210h] [rbp-150h]
  char *v218; // [rsp+218h] [rbp-148h]
  char *v219; // [rsp+220h] [rbp-140h]
  __int64 *v220; // [rsp+228h] [rbp-138h] BYREF
  __int64 v221; // [rsp+230h] [rbp-130h]
  __m128i *v222; // [rsp+238h] [rbp-128h]
  int v223; // [rsp+240h] [rbp-120h]
  __int64 ***v224; // [rsp+248h] [rbp-118h]
  unsigned __int64 v225; // [rsp+250h] [rbp-110h]
  __int64 v226; // [rsp+258h] [rbp-108h]
  __int64 *v227; // [rsp+260h] [rbp-100h] BYREF
  __int64 v228; // [rsp+268h] [rbp-F8h]
  __int64 v229; // [rsp+270h] [rbp-F0h]
  __int64 ***v230; // [rsp+278h] [rbp-E8h]
  unsigned __int64 v231; // [rsp+280h] [rbp-E0h]
  __int64 v232; // [rsp+288h] [rbp-D8h]
  __int64 v233; // [rsp+290h] [rbp-D0h]
  __int64 v234; // [rsp+298h] [rbp-C8h]
  __int64 v235; // [rsp+2A0h] [rbp-C0h]
  void *dest; // [rsp+2A8h] [rbp-B8h] BYREF
  size_t v237; // [rsp+2B0h] [rbp-B0h]
  _QWORD v238[2]; // [rsp+2B8h] [rbp-A8h] BYREF
  __int64 ***v239; // [rsp+2C8h] [rbp-98h]
  unsigned __int64 v240; // [rsp+2D0h] [rbp-90h] BYREF
  __int64 *v241; // [rsp+2D8h] [rbp-88h] BYREF
  char *v242; // [rsp+2E0h] [rbp-80h]
  char *v243; // [rsp+2E8h] [rbp-78h]
  __int64 v244; // [rsp+2F0h] [rbp-70h]
  __int64 v245; // [rsp+2F8h] [rbp-68h]
  _QWORD *v246; // [rsp+300h] [rbp-60h]
  __int64 v247; // [rsp+308h] [rbp-58h]
  unsigned int v248; // [rsp+310h] [rbp-50h]
  char v249; // [rsp+318h] [rbp-48h]
  __int64 v250; // [rsp+320h] [rbp-40h]
  __int64 *v251; // [rsp+328h] [rbp-38h]

  sub_1AAD1B0(a1, a3, a4);
  v215 = a1;
  v216 = a5;
  v218 = 0;
  v8 = (char *)sub_22077B0(48);
  v217 = v8;
  v219 = v8 + 48;
  if ( v8 )
  {
    *(_QWORD *)v8 = 0;
    *((_QWORD *)v8 + 1) = 0;
    *((_QWORD *)v8 + 2) = 0;
  }
  *((_QWORD *)v8 + 3) = 0;
  *((_QWORD *)v8 + 4) = 0;
  *((_QWORD *)v8 + 5) = 0;
  v218 = v8 + 48;
  v220 = a1;
  v227 = a1;
  dest = v238;
  v221 = 0;
  v241 = v215;
  v222 = 0;
  v223 = 0;
  v224 = 0;
  v225 = 0;
  v226 = 0;
  v228 = 0;
  v229 = 0;
  v230 = 0;
  v231 = 0;
  v232 = 0;
  v233 = 0;
  v234 = 0;
  v235 = 0;
  v237 = 0;
  LOBYTE(v238[0]) = 0;
  v240 = 0;
  v242 = 0;
  v243 = 0;
  v244 = 0;
  v245 = 0;
  v246 = 0;
  v247 = 0;
  v248 = 0;
  v249 = 0;
  v250 = a3;
  v251 = a4;
  v9 = v215[10];
  v160 = 2;
  v10 = v9 - 24;
  if ( !v9 )
    v10 = 0;
  v162 = v10;
  if ( a4 )
    v160 = sub_1368DC0((__int64)a4);
  v157 = sub_17E4440((__int64)&v241, 0, v162, v160);
  v11 = sub_157EBA0(v162);
  if ( v11 && (unsigned int)sub_15F4D60(v11) )
  {
    v164 = v241 + 9;
    v166 = (__int64 *)v241[10];
    if ( v166 == v241 + 9 )
    {
      v167 = 0;
      v177 = 0;
      v180 = 0;
      v169 = 0;
    }
    else
    {
      v169 = 0;
      v180 = 0;
      v167 = 0;
      v159 = 0;
      v177 = 0;
      v163 = 0;
      do
      {
        v13 = (__int64)(v166 - 3);
        if ( !v166 )
          v13 = 0;
        src = 2;
        v14 = sub_157EBA0(v13);
        if ( v251 )
          src = sub_1368AA0(v251, v13);
        v170 = sub_15F4D60(v14);
        if ( v170 )
        {
          v172 = v13;
          v15 = 2;
          v168 = v162 == v13;
          for ( i = 0; i != v170; ++i )
          {
            v17 = sub_15F4DF0(v14, i);
            v188 = sub_137E040(v14, i, 0);
            v18 = src;
            if ( v188 )
            {
              v19 = -1;
              if ( src <= 0x4189374BC6A7EELL )
                v19 = 1000 * src;
              v18 = v19;
            }
            if ( v250 )
            {
              LODWORD(n[0]) = sub_13774B0(v250, v172, v17);
              v15 = sub_16AF780((unsigned int *)n, v18);
            }
            v20 = sub_17E4440((__int64)&v241, v172, v17, v15);
            *(_BYTE *)(v20 + 26) = v188;
            v21 = v180;
            if ( v15 > v180 )
            {
              if ( v168 )
                v21 = v15;
              v180 = v21;
              v22 = v177;
              if ( v168 )
                v22 = v20;
              v177 = v22;
            }
            v23 = sub_157EBA0(v17);
            if ( v23 )
            {
              v24 = sub_15F4D60(v23);
              v25 = v169;
              if ( v15 > v169 )
              {
                if ( v24 )
                  v20 = v167;
                else
                  v25 = v15;
                v167 = v20;
                v169 = v25;
              }
            }
          }
        }
        else
        {
          v249 = 1;
          v12 = sub_17E4440((__int64)&v241, v13, 0, src);
          if ( src > v163 )
          {
            v163 = src;
            v159 = v12;
          }
        }
        v166 = (__int64 *)v166[1];
      }
      while ( v164 != v166 );
      if ( v160 < v163 || 2 * v160 >= 3 * v163 )
      {
        if ( v180 < v169 )
          goto LABEL_42;
      }
      else
      {
        *(_QWORD *)(v157 + 16) = v163;
        *(_QWORD *)(v159 + 16) = v160 + 1;
        if ( v180 < v169 )
          goto LABEL_42;
      }
    }
    if ( 2 * v180 < 3 * v169 )
    {
      *(_QWORD *)(v177 + 16) = v169;
      *(_QWORD *)(v167 + 16) = v180 + 1;
    }
  }
  else
  {
    sub_17E4440((__int64)&v241, v162, 0, v160);
  }
LABEL_42:
  v26 = v243;
  v27 = v242;
  if ( v243 - v242 <= 0 )
  {
LABEL_226:
    v29 = 0;
    v31 = 0;
    sub_17E3F90(v27, v26);
  }
  else
  {
    v28 = (v243 - v242) >> 3;
    while ( 1 )
    {
      v29 = v28;
      v30 = (__int64 *)sub_2207800(8 * v28, &unk_435FF63);
      v31 = v30;
      if ( v30 )
        break;
      v28 >>= 1;
      if ( !v28 )
        goto LABEL_226;
    }
    v32 = v28;
    v33 = &v30[v29];
    *v30 = *(_QWORD *)v27;
    v34 = v30 + 1;
    *(_QWORD *)v27 = 0;
    if ( v33 == v31 + 1 )
    {
      v36 = v31;
    }
    else
    {
      do
      {
        v35 = *(v34 - 1);
        *(v34++ - 1) = 0;
        *(v34 - 1) = v35;
      }
      while ( v33 != v34 );
      v36 = &v31[v29 - 1];
    }
    v37 = *v36;
    *v36 = 0;
    v38 = *(_QWORD *)v27;
    *(_QWORD *)v27 = v37;
    if ( v38 )
    {
      v189 = v32;
      j_j___libc_free_0(v38, 32);
      sub_17E66F0(v27, v26, v31, v189);
    }
    else
    {
      sub_17E66F0(v27, v26, v31, v32);
    }
    v39 = v31;
    do
    {
      if ( *v39 )
        j_j___libc_free_0(*v39, 32);
      ++v39;
    }
    while ( v33 != v39 );
  }
  j_j___libc_free_0(v31, v29 * 8);
  v40 = (__int64 **)v243;
  v41 = (__int64 **)v242;
  if ( v242 != v243 )
  {
    do
    {
      v42 = *v41;
      if ( !*((_BYTE *)*v41 + 25) )
      {
        if ( *((_BYTE *)v42 + 26) )
        {
          v43 = v42[1];
          if ( v43 )
          {
            if ( sub_157F790(v43) && (unsigned __int8)sub_17E41E0((__int64)&v241, **v41, (*v41)[1]) )
              *((_BYTE *)*v41 + 24) = 1;
          }
        }
      }
      ++v41;
    }
    while ( v40 != v41 );
    v44 = (__int64 **)v242;
    v45 = v243;
    if ( v242 != v243 )
    {
      do
      {
        v46 = *v44;
        if ( !*((_BYTE *)*v44 + 25) )
        {
          v47 = *v46;
          if ( v249 || v47 )
          {
            if ( (unsigned __int8)sub_17E41E0((__int64)&v241, v47, v46[1]) )
              *((_BYTE *)*v44 + 24) = 1;
          }
        }
        ++v44;
      }
      while ( v45 != (char *)v44 );
    }
  }
  sub_1695660((__int64)n, (__int64)v215, 0);
  v48 = dest;
  if ( (__int64 **)n[0] == &v209 )
  {
    v125 = n[1];
    if ( n[1] )
    {
      if ( n[1] == 1 )
        *(_BYTE *)dest = (_BYTE)v209;
      else
        memcpy(dest, &v209, n[1]);
      v125 = n[1];
      v48 = dest;
    }
    v237 = v125;
    v48[v125] = 0;
    v48 = (_BYTE *)n[0];
  }
  else
  {
    if ( dest == v238 )
    {
      dest = (void *)n[0];
      v237 = n[1];
      v238[0] = v209;
    }
    else
    {
      v49 = (__int64 *)v238[0];
      dest = (void *)n[0];
      v237 = n[1];
      v238[0] = v209;
      if ( v48 )
      {
        n[0] = (size_t)v48;
        v209 = v49;
        goto LABEL_71;
      }
    }
    n[0] = (size_t)&v209;
    v48 = &v209;
  }
LABEL_71:
  n[1] = 0;
  *v48 = 0;
  if ( (__int64 **)n[0] != &v209 )
    j_j___libc_free_0(n[0], (char *)v209 + 1);
  n[0] = 0;
  n[1] = 0;
  v209 = 0;
  LODWORD(v205) = -1;
  v178 = v215 + 9;
  v181 = (__int64 *)v215[10];
  if ( v181 == v215 + 9 )
  {
    v63 = 0;
    v62 = 0;
  }
  else
  {
    do
    {
      v50 = (__int64)(v181 - 3);
      if ( !v181 )
        v50 = 0;
      v51 = 0;
      v52 = sub_157EBA0(v50);
      v53 = sub_15F4D60(v52);
      if ( v53 )
      {
        do
        {
          v54 = sub_15F4DF0(v52, v51);
          if ( v248 )
          {
            v55 = (v248 - 1) & (((unsigned int)v54 >> 9) ^ ((unsigned int)v54 >> 4));
            v56 = &v246[2 * v55];
            v57 = *v56;
            if ( v54 == *v56 )
            {
LABEL_79:
              if ( v56 != &v246[2 * v248] )
              {
                v58 = v56[1];
                if ( v58 )
                {
                  v59 = *(_DWORD *)(v58 + 8);
                  for ( j = 0; j != 32; j += 8 )
                  {
                    v61 = n[1];
                    LOBYTE(v203[0]) = v59 >> j;
                    if ( (__int64 *)n[1] == v209 )
                    {
                      v192 = j;
                      sub_17EB120((__int64)n, (const void *)n[1], (char *)v203);
                      j = v192;
                    }
                    else
                    {
                      if ( n[1] )
                      {
                        *(_BYTE *)n[1] = v59 >> j;
                        v61 = n[1];
                      }
                      n[1] = v61 + 1;
                    }
                  }
                }
              }
            }
            else
            {
              v97 = 1;
              while ( v57 != -8 )
              {
                v98 = v97 + 1;
                v55 = (v248 - 1) & (v97 + v55);
                v56 = &v246[2 * v55];
                v57 = *v56;
                if ( v54 == *v56 )
                  goto LABEL_79;
                v97 = v98;
              }
            }
          }
          ++v51;
        }
        while ( v53 != v51 );
      }
      v181 = (__int64 *)v181[1];
    }
    while ( v178 != v181 );
    v62 = n[0];
    v63 = n[1] - n[0];
  }
  sub_3946250(&v205, v62, v63);
  v240 = ((__int64)(*((_QWORD *)v217 + 1) - *(_QWORD *)v217) >> 3 << 48)
       | (unsigned int)v205
       | ((unsigned __int64)(unsigned int)v221 << 56)
       | ((v243 - v242) >> 3 << 32);
  if ( n[0] )
    j_j___libc_free_0(n[0], (char *)v209 - n[0]);
  if ( a5[3] && (unsigned __int8)sub_17E85A0((__int64)v215, v216) )
  {
    v121 = (char *)sub_1649960((__int64)v215);
    if ( v121 )
    {
      v201[0] = (__int64)v202;
      sub_17E2210(v201, v121, (__int64)&v121[v120]);
    }
    else
    {
      LOBYTE(v202[0]) = 0;
      v201[0] = (__int64)v202;
      v201[1] = 0;
    }
    LOWORD(v207[0]) = 267;
    v205 = &v240;
    v122.m128i_i64[0] = (__int64)sub_1649960((__int64)v215);
    v197 = v122;
    v199.m128i_i64[0] = (__int64)&v197;
    v199.m128i_i64[1] = (__int64)".";
    v123 = v207[0];
    LOWORD(v200) = 773;
    if ( LOBYTE(v207[0]) )
    {
      if ( LOBYTE(v207[0]) == 1 )
      {
        *(__m128i *)n = _mm_loadu_si128(&v199);
        v209 = v200;
      }
      else
      {
        v124 = (void **)v205;
        if ( BYTE1(v207[0]) != 1 )
        {
          v124 = (void **)&v205;
          v123 = 2;
        }
        n[1] = (size_t)v124;
        n[0] = (size_t)&v199;
        LOBYTE(v209) = 2;
        BYTE1(v209) = v123;
      }
    }
    else
    {
      LOWORD(v209) = 256;
    }
    sub_16E2FC0((__int64 *)v203, (__int64)n);
    LOWORD(v209) = 260;
    n[0] = (size_t)v203;
    sub_164B780((__int64)v215, (__int64 *)n);
    LOWORD(v209) = 260;
    n[0] = (size_t)v201;
    sub_15E5880(4, (__int64)n, v215);
    LOWORD(v198) = 267;
    v197.m128i_i64[0] = (__int64)&v240;
    v205 = v207;
    sub_17E2330((__int64 *)&v205, dest, (__int64)dest + v237);
    if ( v206 == 0x3FFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"basic_string::append");
    sub_2241490(&v205, ".", 1, v132);
    v133 = (char)v198;
    if ( (_BYTE)v198 )
    {
      if ( (_BYTE)v198 == 1 )
      {
        v199.m128i_i64[0] = (__int64)&v205;
        LOWORD(v200) = 260;
      }
      else
      {
        v134 = (__m128i *)v197.m128i_i64[0];
        if ( BYTE1(v198) != 1 )
        {
          v134 = &v197;
          v133 = 2;
        }
        v199.m128i_i64[0] = (__int64)&v205;
        v199.m128i_i64[1] = (__int64)v134;
        LOBYTE(v200) = 4;
        BYTE1(v200) = v133;
      }
    }
    else
    {
      LOWORD(v200) = 256;
    }
    sub_16E2FC0((__int64 *)n, (__int64)&v199);
    v135 = dest;
    if ( (__int64 **)n[0] == &v209 )
    {
      v153 = n[1];
      if ( n[1] )
      {
        if ( n[1] == 1 )
          *(_BYTE *)dest = (_BYTE)v209;
        else
          memcpy(dest, &v209, n[1]);
        v153 = n[1];
        v135 = dest;
      }
      v237 = v153;
      v135[v153] = 0;
      v135 = (_BYTE *)n[0];
      goto LABEL_264;
    }
    if ( dest == v238 )
    {
      dest = (void *)n[0];
      v237 = n[1];
      v238[0] = v209;
    }
    else
    {
      v136 = (__int64 *)v238[0];
      dest = (void *)n[0];
      v237 = n[1];
      v238[0] = v209;
      if ( v135 )
      {
        n[0] = (size_t)v135;
        v209 = v136;
LABEL_264:
        n[1] = 0;
        *v135 = 0;
        if ( (__int64 **)n[0] != &v209 )
          j_j___libc_free_0(n[0], (char *)v209 + 1);
        if ( v205 != v207 )
          j_j___libc_free_0(v205, v207[0] + 1LL);
        v137 = v215[6];
        if ( v137 )
        {
          v138 = (_QWORD *)v215[6];
          v183 = v215[5];
          LOWORD(v200) = 267;
          v193 = v137;
          v199.m128i_i64[0] = (__int64)&v240;
          v139.m128i_i64[0] = sub_1580C70(v138);
          v195 = v139;
          LOWORD(v198) = 773;
          v197.m128i_i64[0] = (__int64)&v195;
          v197.m128i_i64[1] = (__int64)".";
          v140 = (char)v200;
          if ( (_BYTE)v200 )
          {
            if ( (_BYTE)v200 == 1 )
            {
              *(__m128i *)n = _mm_loadu_si128(&v197);
              v209 = v198;
            }
            else
            {
              v141 = (__m128i *)v199.m128i_i64[0];
              if ( BYTE1(v200) != 1 )
              {
                v141 = &v199;
                v140 = 2;
              }
              n[1] = (size_t)v141;
              LOBYTE(v209) = 2;
              n[0] = (size_t)&v197;
              BYTE1(v209) = v140;
            }
          }
          else
          {
            LOWORD(v209) = 256;
          }
          sub_16E2FC0((__int64 *)&v205, (__int64)n);
          v184 = sub_1633B90(v183, v205, v206);
          *(_DWORD *)(v184 + 8) = *(_DWORD *)(v137 + 8);
          v142 = sub_17E8510(v216, (unsigned __int64 *)&v193);
          v144 = v143;
          if ( (_QWORD *)v142 != v143 )
          {
            v145 = (_QWORD *)v142;
            do
            {
              v146 = v145[2];
              v147 = *(_BYTE *)(v146 + 16);
              if ( v147 == 1 )
              {
                v149 = (char *)sub_1649960(v145[2]);
                n[0] = (size_t)&v209;
                if ( v149 )
                {
                  sub_17E2210((__int64 *)n, v149, (__int64)&v149[v148]);
                }
                else
                {
                  n[1] = 0;
                  LOBYTE(v209) = 0;
                }
                LOWORD(v198) = 267;
                v197.m128i_i64[0] = (__int64)&v240;
                v194[0] = sub_1649960(v146);
                LOWORD(v196) = 773;
                v194[1] = v150;
                v195.m128i_i64[0] = (__int64)v194;
                v195.m128i_i64[1] = (__int64)".";
                v151 = (char)v198;
                if ( (_BYTE)v198 )
                {
                  if ( (_BYTE)v198 == 1 )
                  {
                    v199 = _mm_loadu_si128(&v195);
                    v200 = v196;
                  }
                  else
                  {
                    v152 = (__m128i *)v197.m128i_i64[0];
                    if ( BYTE1(v198) != 1 )
                    {
                      v152 = &v197;
                      v151 = 2;
                    }
                    v199.m128i_i64[1] = (__int64)v152;
                    LOBYTE(v200) = 2;
                    v199.m128i_i64[0] = (__int64)&v195;
                    BYTE1(v200) = v151;
                  }
                }
                else
                {
                  LOWORD(v200) = 256;
                }
                sub_164B780(v146, v199.m128i_i64);
                LOWORD(v200) = 260;
                v199.m128i_i64[0] = (__int64)n;
                sub_15E5880(4, (__int64)&v199, (__int64 *)v146);
                if ( (__int64 **)n[0] != &v209 )
                  j_j___libc_free_0(n[0], (char *)v209 + 1);
              }
              else
              {
                if ( v147 )
                {
                  MEMORY[0x30] = v184;
                  BUG();
                }
                *(_QWORD *)(v146 + 48) = v184;
              }
              v145 = (_QWORD *)*v145;
            }
            while ( v144 != v145 );
          }
          if ( v205 != v207 )
            j_j___libc_free_0(v205, v207[0] + 1LL);
          if ( v203[0] != &v204 )
            j_j___libc_free_0(v203[0], v204 + 1);
          if ( (_QWORD *)v201[0] != v202 )
            j_j___libc_free_0(v201[0], v202[0] + 1LL);
        }
        else
        {
          v154 = sub_1633B90(v215[5], v203[0], (size_t)v203[1]);
          v155 = v215;
          v156 = v154;
          LOBYTE(v154) = v215[4] & 0xF0 | 3;
          *((_BYTE *)v215 + 32) = v154;
          if ( (v154 & 0x30) != 0 )
            *((_BYTE *)v155 + 33) |= 0x40u;
          v155[6] = v156;
          sub_2240A30(v203);
          sub_2240A30(v201);
        }
        goto LABEL_93;
      }
    }
    n[0] = (size_t)&v209;
    v135 = &v209;
    goto LABEL_264;
  }
LABEL_93:
  v64 = (__int64 ***)sub_1694C10((__int64)v215, (char *)dest, v237);
  v65 = 0;
  v239 = v64;
  for ( k = v242; v243 != k; k += 8 )
  {
    if ( !*(_BYTE *)(*(_QWORD *)k + 24LL) )
      v65 += *(_BYTE *)(*(_QWORD *)k + 25LL) == 0;
  }
  v195.m128i_i32[0] = 0;
  v171 = v221 + v65;
  v67 = sub_16471D0((_QWORD *)*a2, 0);
  v68 = v242;
  v179 = (__int64 **)v67;
  v190 = v243;
  if ( v243 != v242 )
  {
    while ( 1 )
    {
      v70 = *(_QWORD *)v68;
      if ( *(_BYTE *)(*(_QWORD *)v68 + 24LL) || *(_BYTE *)(v70 + 25) )
        goto LABEL_106;
      v71 = *(_QWORD *)v70;
      v72 = *(_QWORD *)(v70 + 8);
      v182 = *(_QWORD *)v68;
      if ( *(_QWORD *)v70 )
      {
        if ( !v72 )
          goto LABEL_115;
        srca = (void *)sub_157EBA0(v71);
        if ( (unsigned int)sub_15F4D60((__int64)srca) <= 1 )
          goto LABEL_115;
        if ( !*(_BYTE *)(v182 + 26) )
        {
          v71 = v72;
          goto LABEL_115;
        }
        v73 = sub_137DFF0(v71, v72);
        n[0] = 0;
        n[1] = 0;
        LODWORD(v209) = (_DWORD)&loc_1000000;
        v71 = sub_1AAC5F0(srca, v73, n);
        *(_BYTE *)(v182 + 25) = 1;
      }
      else
      {
        v71 = *(_QWORD *)(v70 + 8);
      }
      if ( !v71 )
      {
LABEL_106:
        v68 += 8;
        if ( v190 == v68 )
          break;
      }
      else
      {
LABEL_115:
        v74 = sub_157EE30(v71);
        v75 = (_QWORD *)sub_157E9C0(v71);
        n[1] = v71;
        n[0] = 0;
        v210 = v75;
        v211 = 0;
        v212 = 0;
        v213 = 0;
        v214 = 0;
        v209 = (__int64 *)v74;
        if ( v74 != v71 + 40 )
        {
          if ( !v74 )
            BUG();
          v69 = *(unsigned __int64 **)(v74 + 24);
          v205 = v69;
          if ( v69 )
          {
            sub_1623A60((__int64)&v205, (__int64)v69, 2);
            if ( n[0] )
              sub_161E7C0((__int64)n, n[0]);
            n[0] = (size_t)v205;
            if ( v205 )
            {
              sub_1623210((__int64)&v205, (unsigned __int8 *)v205, (__int64)n);
              ++v195.m128i_i32[0];
              if ( n[0] )
                sub_161E7C0((__int64)n, n[0]);
              goto LABEL_106;
            }
          }
        }
        ++v195.m128i_i32[0];
        v68 += 8;
        if ( v190 == v68 )
          break;
      }
    }
  }
  HIDWORD(v221) = 1;
  v222 = &v195;
  v76 = (__int64 *)a1[10];
  v223 = v171;
  v225 = v240;
  v224 = v239;
  v173 = a1 + 9;
  if ( a1 + 9 != v76 )
  {
    v77 = v76;
    do
    {
      v78 = v77;
      v77 = (__int64 *)v77[1];
      v79 = (__int64 *)v78[3];
      v80 = v78 + 2;
LABEL_120:
      while ( v80 != v79 )
      {
        while ( 1 )
        {
          v81 = v79;
          v79 = (__int64 *)v79[1];
          if ( *((_BYTE *)v81 - 8) != 79 || !byte_4FA52E0 || *(_BYTE *)(*(_QWORD *)*(v81 - 12) + 8LL) == 16 )
            break;
          v82 = (__int64)(v81 - 3);
          if ( HIDWORD(v221) == 1 )
          {
            sub_17E2B70((__int64)&v220, v82);
            goto LABEL_120;
          }
          if ( HIDWORD(v221) == 2 )
          {
            sub_17EA5C0(&v220, v82);
            goto LABEL_120;
          }
          LODWORD(v221) = v221 + 1;
          if ( v80 == v79 )
            goto LABEL_127;
        }
      }
LABEL_127:
      ;
    }
    while ( v173 != v77 );
  }
  if ( byte_4FA5900 )
  {
    if ( v248 )
    {
      v126 = v246;
      v127 = &v246[2 * v248];
      do
      {
        if ( *v126 != -8 && *v126 != -16 )
        {
          v128 = v126[1];
          if ( v128 )
            j_j___libc_free_0(v128, 16);
        }
        v126 += 2;
      }
      while ( v127 != v126 );
    }
    j___libc_free_0(v246);
    v129 = v243;
    v130 = v242;
    if ( v243 != v242 )
    {
      do
      {
        if ( *(_QWORD *)v130 )
          j_j___libc_free_0(*(_QWORD *)v130, 32);
        v130 += 8;
      }
      while ( v129 != v130 );
      v130 = v242;
    }
    if ( v130 )
      j_j___libc_free_0(v130, v244 - (_QWORD)v130);
    result = sub_2240A30(&dest);
    if ( v233 )
      result = j_j___libc_free_0(v233, v235 - v233);
    v131 = v218;
    v119 = v217;
    if ( v218 != v217 )
    {
      do
      {
        if ( *(_QWORD *)v119 )
          result = j_j___libc_free_0(*(_QWORD *)v119, *((_QWORD *)v119 + 2) - *(_QWORD *)v119);
        v119 += 24;
      }
      while ( v131 != v119 );
      goto LABEL_206;
    }
  }
  else
  {
    v83 = *(__int64 **)v217;
    srcb = (__int64 *)*((_QWORD *)v217 + 1);
    if ( srcb != *(__int64 **)v217 )
    {
      for ( m = 0; ; ++m )
      {
        v95 = *v83;
        v96 = *(_BYTE *)(*v83 + 16);
        if ( v96 <= 0x17u )
          break;
        if ( v96 == 78 )
        {
          v99 = v95 | 4;
        }
        else
        {
          v84 = 0;
          if ( v96 != 29 )
            goto LABEL_133;
          v99 = v95 & 0xFFFFFFFFFFFFFFFBLL;
        }
        v84 = v99 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v99 & 4) == 0 )
          goto LABEL_133;
        v85 = (__int64 *)(v84 - 24);
LABEL_134:
        v185 = *v83;
        v86 = *v85;
        v87 = (_QWORD *)sub_16498A0(*v83);
        n[0] = 0;
        v210 = v87;
        v211 = 0;
        v212 = 0;
        v213 = 0;
        v214 = 0;
        n[1] = *(_QWORD *)(v185 + 40);
        v209 = (__int64 *)(v185 + 24);
        v88 = *(unsigned __int64 **)(v185 + 48);
        v205 = v88;
        if ( v88 )
        {
          sub_1623A60((__int64)&v205, (__int64)v88, 2);
          if ( n[0] )
            sub_161E7C0((__int64)n, n[0]);
          n[0] = (size_t)v205;
          if ( v205 )
            sub_1623210((__int64)&v205, (unsigned __int8 *)v205, (__int64)n);
        }
        LOWORD(v202[0]) = 257;
        v205 = (unsigned __int64 *)sub_15A4510(v239, v179, 0);
        v186 = v240;
        v89 = sub_1643360(v210);
        v90 = sub_159C470(v89, v186, 0);
        LOWORD(v200) = 257;
        v206 = v90;
        v91 = (__int64 **)sub_1643360(v210);
        if ( v91 != *(__int64 ***)v86 )
        {
          if ( *(_BYTE *)(v86 + 16) > 0x10u )
          {
            LOWORD(v204) = 257;
            v86 = sub_15FDBD0(45, v86, (__int64)v91, (__int64)v203, 0);
            if ( n[1] )
            {
              v187 = v209;
              sub_157E9D0(n[1] + 40, v86);
              v100 = *v187;
              v101 = *(_QWORD *)(v86 + 24) & 7LL;
              *(_QWORD *)(v86 + 32) = v187;
              v100 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v86 + 24) = v100 | v101;
              *(_QWORD *)(v100 + 8) = v86 + 24;
              *v187 = *v187 & 7 | (v86 + 24);
            }
            sub_164B780(v86, v199.m128i_i64);
            if ( n[0] )
            {
              v197.m128i_i64[0] = n[0];
              sub_1623A60((__int64)&v197, n[0], 2);
              v102 = *(_QWORD *)(v86 + 48);
              v103 = v86 + 48;
              if ( v102 )
              {
                sub_161E7C0(v86 + 48, v102);
                v103 = v86 + 48;
              }
              v104 = (unsigned __int8 *)v197.m128i_i64[0];
              *(_QWORD *)(v86 + 48) = v197.m128i_i64[0];
              if ( v104 )
                sub_1623210((__int64)&v197, v104, v103);
            }
          }
          else
          {
            v86 = sub_15A46C0(45, (__int64 ***)v86, v91, 0);
          }
        }
        v207[0] = v86;
        v92 = sub_1643350(v210);
        v207[1] = sub_159C470(v92, 0, 0);
        v93 = sub_1643350(v210);
        v207[2] = sub_159C470(v93, m, 0);
        v94 = sub_15E26F0(a2, 112, 0, 0);
        sub_17E28C0((__int64)n, *(_QWORD *)(v94 + 24), v94, (__int64 *)&v205, 5, v201, 0);
        if ( n[0] )
          sub_161E7C0((__int64)n, n[0]);
        if ( srcb == ++v83 )
          goto LABEL_174;
      }
      v84 = 0;
LABEL_133:
      v85 = (__int64 *)(v84 - 72);
      goto LABEL_134;
    }
LABEL_174:
    HIDWORD(v228) = 1;
    HIDWORD(v229) = v171;
    v231 = v240;
    v230 = v239;
    v105 = a1[10];
    while ( v173 != (__int64 *)v105 )
    {
      v106 = v105;
      v105 = *(_QWORD *)(v105 + 8);
      v107 = *(_QWORD *)(v106 + 24);
      v108 = v106 + 16;
      while ( v108 != v107 )
      {
        while ( 1 )
        {
          v109 = v107;
          v107 = *(_QWORD *)(v107 + 8);
          if ( *(_BYTE *)(v109 - 8) != 78 )
            break;
          v110 = *(_QWORD *)(v109 - 48);
          if ( *(_BYTE *)(v110 + 16) )
            break;
          v111 = *(_DWORD *)(v110 + 36);
          if ( v111 != 135 && v111 != 137 && v111 != 133 )
            break;
          sub_17E8630((__int64)&v227, (unsigned __int8 *)(v109 - 24));
          if ( v108 == v107 )
            goto LABEL_183;
        }
      }
LABEL_183:
      ;
    }
    if ( v248 )
    {
      v112 = v246;
      v113 = &v246[2 * v248];
      do
      {
        if ( *v112 != -16 && *v112 != -8 )
        {
          v114 = v112[1];
          if ( v114 )
            j_j___libc_free_0(v114, 16);
        }
        v112 += 2;
      }
      while ( v113 != v112 );
    }
    result = j___libc_free_0(v246);
    v116 = v243;
    v117 = v242;
    if ( v243 != v242 )
    {
      do
      {
        if ( *(_QWORD *)v117 )
          result = j_j___libc_free_0(*(_QWORD *)v117, 32);
        v117 += 8;
      }
      while ( v116 != v117 );
      v117 = v242;
    }
    if ( v117 )
      result = j_j___libc_free_0(v117, v244 - (_QWORD)v117);
    if ( dest != v238 )
      result = j_j___libc_free_0(dest, v238[0] + 1LL);
    if ( v233 )
      result = j_j___libc_free_0(v233, v235 - v233);
    v118 = v218;
    v119 = v217;
    if ( v218 != v217 )
    {
      do
      {
        if ( *(_QWORD *)v119 )
          result = j_j___libc_free_0(*(_QWORD *)v119, *((_QWORD *)v119 + 2) - *(_QWORD *)v119);
        v119 += 24;
      }
      while ( v118 != v119 );
LABEL_206:
      v119 = v217;
    }
  }
  if ( v119 )
    return j_j___libc_free_0(v119, v219 - v119);
  return result;
}
