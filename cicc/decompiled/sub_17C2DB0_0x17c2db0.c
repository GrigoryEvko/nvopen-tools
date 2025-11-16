// Function: sub_17C2DB0
// Address: 0x17c2db0
//
__int64 __fastcall sub_17C2DB0(__int64 a1, _QWORD *a2, char a3, unsigned __int8 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // rax
  void **v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rdi
  unsigned __int64 v10; // r8
  __int64 v11; // r12
  __int64 v12; // rbx
  unsigned __int64 v13; // rdi
  __int64 *v15; // rbx
  __int64 v16; // rsi
  __int64 v17; // rax
  void **p_src; // rdi
  __int64 v19; // rdx
  __int64 *v20; // rax
  __int64 v21; // rbx
  __int64 v22; // rax
  __int64 v23; // r15
  __int64 v24; // r13
  __int64 v25; // rax
  __int64 v26; // rax
  _BYTE *v27; // r8
  _BYTE *v28; // rax
  _BYTE *v29; // r12
  __int64 v30; // rax
  __int64 v31; // rbx
  __int64 v32; // r13
  size_t v33; // rdx
  __int64 *v34; // r12
  unsigned __int64 v35; // rsi
  __int64 v36; // rax
  __int64 v37; // rcx
  __int64 v38; // r8
  int v39; // r13d
  __int64 v40; // rdx
  __int64 v41; // rbx
  __m128i *v42; // rdx
  __m128i *v43; // r13
  __int64 v44; // r15
  __int64 *v45; // rbx
  __int64 v46; // rax
  __m128i v47; // xmm1
  __m128i v48; // xmm2
  _QWORD *v49; // r13
  _QWORD *v50; // rbx
  _QWORD *v51; // r13
  _QWORD *v52; // rdi
  int v53; // r8d
  unsigned __int64 v54; // rcx
  __int64 v55; // rbx
  __int64 *v56; // r13
  unsigned __int64 v57; // r15
  __int64 v58; // rsi
  __int64 v59; // r12
  __int64 v60; // rax
  __m128i v61; // xmm3
  __m128i v62; // xmm4
  _QWORD *v63; // rbx
  _QWORD *v64; // rbx
  _QWORD *v65; // rdi
  char *v66; // rdi
  __int64 v67; // rax
  __int64 v68; // rcx
  unsigned __int64 *v69; // rdx
  unsigned __int64 v70; // r9
  unsigned __int8 v71; // al
  unsigned __int64 v72; // rdi
  char v73; // al
  unsigned __int64 v74; // r9
  __int64 v75; // rax
  __m128i v76; // xmm7
  __m128i v77; // xmm5
  _QWORD *v78; // rbx
  _QWORD *v79; // rbx
  _QWORD *v80; // rdi
  _QWORD *v81; // rbx
  _QWORD *v82; // rdi
  _QWORD *v83; // r13
  _QWORD *v84; // rdi
  signed __int64 v85; // rdx
  __int64 v86; // rax
  bool v87; // cf
  unsigned __int64 v88; // rax
  __int64 v89; // r10
  __int64 v90; // r10
  __int64 v91; // rax
  __m128i *v92; // r11
  __m128i *v93; // r10
  __int64 v94; // rax
  unsigned __int64 *v95; // rdx
  const __m128i *v96; // rax
  __m128i *v97; // rax
  __m128i *v98; // rdi
  __int64 v99; // rax
  __int64 v100; // rax
  __int64 v101; // rax
  __int64 v102; // rax
  __int64 **v103; // rax
  __int64 v104; // rax
  __int64 v105; // rax
  __int64 v106; // r9
  __int64 v107; // r13
  __int64 v108; // rbx
  char *v109; // r13
  size_t v110; // rdx
  _QWORD *v111; // rbx
  _QWORD *v112; // r13
  _QWORD *v113; // rdi
  _QWORD *v114; // rbx
  _QWORD *v115; // rdi
  __int64 v116; // rax
  __int64 v117; // rax
  char v118; // al
  _QWORD *v119; // r13
  _QWORD *v120; // rdi
  __int64 v121; // rax
  __m128i v122; // xmm5
  __m128i v123; // xmm6
  _QWORD *v124; // rbx
  _QWORD *v125; // rbx
  _QWORD *v126; // rdi
  _QWORD *v127; // r13
  _QWORD *v128; // rdi
  __int64 v129; // r12
  __int64 v130; // rax
  __int64 v131; // rax
  __int64 v132; // rax
  __int64 v133; // rax
  unsigned __int64 v134; // [rsp+10h] [rbp-640h]
  unsigned __int64 v137; // [rsp+30h] [rbp-620h]
  __int64 v138; // [rsp+38h] [rbp-618h]
  __int64 v139; // [rsp+40h] [rbp-610h]
  __int64 *dest; // [rsp+48h] [rbp-608h]
  unsigned __int8 v141; // [rsp+5Dh] [rbp-5F3h]
  char v143; // [rsp+5Fh] [rbp-5F1h]
  __int64 v144; // [rsp+60h] [rbp-5F0h]
  unsigned __int64 v145; // [rsp+68h] [rbp-5E8h]
  unsigned __int64 v146; // [rsp+70h] [rbp-5E0h]
  __int64 v148; // [rsp+A0h] [rbp-5B0h]
  __int64 v149; // [rsp+A8h] [rbp-5A8h]
  __int64 *v150; // [rsp+B0h] [rbp-5A0h]
  __int64 *v151; // [rsp+B8h] [rbp-598h]
  __m128i *v152; // [rsp+C0h] [rbp-590h]
  signed __int64 v153; // [rsp+C0h] [rbp-590h]
  __m128i *v154; // [rsp+C8h] [rbp-588h]
  __m128i *v155; // [rsp+C8h] [rbp-588h]
  __int64 *v156; // [rsp+D0h] [rbp-580h]
  unsigned int v157; // [rsp+D8h] [rbp-578h]
  __int64 v158; // [rsp+D8h] [rbp-578h]
  __int64 v159; // [rsp+E0h] [rbp-570h]
  __m128i *v160; // [rsp+E0h] [rbp-570h]
  unsigned __int64 v161; // [rsp+E0h] [rbp-570h]
  unsigned __int64 v162; // [rsp+E0h] [rbp-570h]
  __int64 v163; // [rsp+E0h] [rbp-570h]
  __m128i *v164; // [rsp+E0h] [rbp-570h]
  __int64 *v165; // [rsp+E8h] [rbp-568h]
  unsigned int v166; // [rsp+E8h] [rbp-568h]
  int v167; // [rsp+E8h] [rbp-568h]
  __int64 v168; // [rsp+E8h] [rbp-568h]
  __int64 v169; // [rsp+E8h] [rbp-568h]
  unsigned int v170; // [rsp+F0h] [rbp-560h] BYREF
  int v171; // [rsp+F4h] [rbp-55Ch] BYREF
  __int64 v172; // [rsp+F8h] [rbp-558h] BYREF
  unsigned __int64 v173; // [rsp+100h] [rbp-550h] BYREF
  char *s; // [rsp+108h] [rbp-548h] BYREF
  _QWORD v175[2]; // [rsp+110h] [rbp-540h] BYREF
  __int64 v176; // [rsp+120h] [rbp-530h] BYREF
  __int64 *v177; // [rsp+130h] [rbp-520h]
  __int64 v178; // [rsp+140h] [rbp-510h] BYREF
  unsigned __int64 v179[2]; // [rsp+170h] [rbp-4E0h] BYREF
  __int64 v180; // [rsp+180h] [rbp-4D0h] BYREF
  __int64 *v181; // [rsp+190h] [rbp-4C0h]
  __int64 v182; // [rsp+1A0h] [rbp-4B0h] BYREF
  _QWORD v183[3]; // [rsp+1D0h] [rbp-480h] BYREF
  unsigned __int64 v184; // [rsp+1E8h] [rbp-468h]
  __int64 v185; // [rsp+1F0h] [rbp-460h]
  __int64 v186; // [rsp+1F8h] [rbp-458h]
  __int64 v187; // [rsp+208h] [rbp-448h]
  __int64 v188; // [rsp+210h] [rbp-440h]
  __int64 v189; // [rsp+218h] [rbp-438h]
  char *v190; // [rsp+220h] [rbp-430h]
  char *v191; // [rsp+228h] [rbp-428h]
  __int64 v192; // [rsp+230h] [rbp-420h]
  __int64 v193; // [rsp+238h] [rbp-418h]
  __int64 v194; // [rsp+240h] [rbp-410h]
  __int64 v195; // [rsp+248h] [rbp-408h]
  char v196; // [rsp+250h] [rbp-400h]
  __int64 v197; // [rsp+260h] [rbp-3F0h] BYREF
  int v198; // [rsp+268h] [rbp-3E8h]
  char v199; // [rsp+26Ch] [rbp-3E4h]
  _BYTE *v200; // [rsp+270h] [rbp-3E0h]
  __m128i v201; // [rsp+278h] [rbp-3D8h]
  __int64 v202; // [rsp+288h] [rbp-3C8h]
  __int64 v203; // [rsp+290h] [rbp-3C0h]
  __m128i v204; // [rsp+298h] [rbp-3B8h]
  __int64 v205; // [rsp+2A8h] [rbp-3A8h]
  char v206; // [rsp+2B0h] [rbp-3A0h]
  _BYTE *v207; // [rsp+2B8h] [rbp-398h] BYREF
  __int64 v208; // [rsp+2C0h] [rbp-390h]
  _BYTE v209[352]; // [rsp+2C8h] [rbp-388h] BYREF
  char v210; // [rsp+428h] [rbp-228h]
  int v211; // [rsp+42Ch] [rbp-224h]
  __int64 v212; // [rsp+430h] [rbp-220h]
  void *src; // [rsp+440h] [rbp-210h] BYREF
  _BYTE *v214; // [rsp+448h] [rbp-208h]
  _BYTE *v215; // [rsp+450h] [rbp-200h] BYREF
  __m128i v216; // [rsp+458h] [rbp-1F8h] BYREF
  __int64 v217; // [rsp+468h] [rbp-1E8h]
  __int64 v218; // [rsp+470h] [rbp-1E0h]
  __m128i v219; // [rsp+478h] [rbp-1D8h] BYREF
  __int64 v220; // [rsp+488h] [rbp-1C8h]
  char v221; // [rsp+490h] [rbp-1C0h]
  _QWORD *v222; // [rsp+498h] [rbp-1B8h] BYREF
  unsigned int v223; // [rsp+4A0h] [rbp-1B0h]
  _BYTE v224[352]; // [rsp+4A8h] [rbp-1A8h] BYREF
  char v225; // [rsp+608h] [rbp-48h]
  int v226; // [rsp+60Ch] [rbp-44h]
  __int64 v227; // [rsp+610h] [rbp-40h]

  v186 = 0x1000000000LL;
  memset(v183, 0, sizeof(v183));
  v184 = 0;
  v185 = 0;
  v187 = 0;
  v188 = 0;
  v189 = 0;
  v190 = 0;
  v191 = 0;
  v192 = 0;
  v193 = 0;
  v194 = 0;
  v195 = 0;
  v196 = 0;
  sub_1697B70(v179, (__int64)v183, a1, a3, a5, a6);
  if ( (v179[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    v6 = v179[0] & 0xFFFFFFFFFFFFFFFELL | 1;
    v179[0] = 0;
    v7 = (void **)&v197;
    v197 = v6;
    sub_12BF440((__int64)&src, &v197);
    if ( (v197 & 1) != 0 || (v197 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_16BCAE0(&v197, (__int64)&v197, v8);
    if ( src != &v215 )
    {
      v7 = (void **)(v215 + 1);
      j_j___libc_free_0(src, v215 + 1);
    }
    if ( (v179[0] & 1) != 0 || (v141 = 0, (v179[0] & 0xFFFFFFFFFFFFFFFELL) != 0) )
      sub_16BCAE0(v179, (__int64)v7, v8);
    goto LABEL_8;
  }
  v141 = 0;
  v139 = a1 + 24;
  v148 = *(_QWORD *)(a1 + 32);
  if ( v148 == a1 + 24 )
    goto LABEL_8;
  do
  {
    v15 = (__int64 *)(v148 - 56);
    if ( !v148 )
      v15 = 0;
    if ( !sub_15E4F60((__int64)v15) )
    {
      v16 = 35;
      v143 = sub_1560180((__int64)(v15 + 14), 35);
      if ( !v143 )
      {
        if ( a5 )
        {
          v17 = sub_16374B0(a5, (__int64)&unk_4F9EE60, a1);
          v16 = (__int64)&unk_4F99CB8;
          v138 = 0;
          v159 = sub_1639800(*(_QWORD *)(v17 + 8), (__int64)&unk_4F99CB8, (__int64)v15) + 8;
        }
        else
        {
          v103 = (__int64 **)sub_22077B0(24);
          v159 = (__int64)v103;
          if ( v103 )
          {
            v16 = (__int64)v15;
            sub_143A950(v103, v15);
          }
          v138 = v159;
        }
        p_src = (void **)&v172;
        sub_14DE650(&v172);
        v20 = v15 + 9;
        src = 0;
        v214 = 0;
        v215 = 0;
        v21 = v15[10];
        v165 = v20;
        if ( v20 != (__int64 *)v21 )
        {
          do
          {
            v22 = v21;
            v21 = *(_QWORD *)(v21 + 8);
            v23 = *(_QWORD *)(v22 + 24);
            v24 = v22 + 16;
            while ( v24 != v23 )
            {
LABEL_32:
              v25 = v23;
              v23 = *(_QWORD *)(v23 + 8);
              v16 = v25 - 24;
              switch ( *(_BYTE *)(v25 - 8) )
              {
                case 0x18:
                case 0x19:
                case 0x1A:
                case 0x1B:
                case 0x1C:
                case 0x1E:
                case 0x1F:
                case 0x20:
                case 0x21:
                case 0x22:
                case 0x23:
                case 0x24:
                case 0x25:
                case 0x26:
                case 0x27:
                case 0x28:
                case 0x29:
                case 0x2A:
                case 0x2B:
                case 0x2C:
                case 0x2D:
                case 0x2E:
                case 0x2F:
                case 0x30:
                case 0x31:
                case 0x32:
                case 0x33:
                case 0x34:
                case 0x35:
                case 0x36:
                case 0x37:
                case 0x38:
                case 0x39:
                case 0x3A:
                case 0x3B:
                case 0x3C:
                case 0x3D:
                case 0x3E:
                case 0x3F:
                case 0x40:
                case 0x41:
                case 0x42:
                case 0x43:
                case 0x44:
                case 0x45:
                case 0x46:
                case 0x47:
                case 0x48:
                case 0x49:
                case 0x4A:
                case 0x4B:
                case 0x4C:
                case 0x4D:
                case 0x4F:
                case 0x50:
                case 0x51:
                case 0x52:
                case 0x53:
                case 0x54:
                case 0x55:
                case 0x56:
                case 0x57:
                case 0x58:
                  continue;
                case 0x1D:
                  v16 &= 0xFFFFFFFFFFFFFFF8LL;
                  v26 = *(_QWORD *)(v16 - 72);
                  if ( !v26
                    || *(_BYTE *)(v26 + 16) <= 0x10u
                    || *(_BYTE *)(v16 + 16) == 78 && *(_BYTE *)(*(_QWORD *)(v16 - 24) + 16LL) == 20 )
                  {
                    continue;
                  }
                  v197 = v16;
                  v27 = v214;
                  if ( v214 != v215 )
                  {
                    if ( v214 )
                    {
                      *(_QWORD *)v214 = v16;
                      v27 = v214;
                    }
                    v214 = v27 + 8;
                    if ( v24 == v23 )
                      goto LABEL_41;
                    goto LABEL_32;
                  }
                  v16 = (__int64)v214;
                  p_src = &src;
                  sub_17C2330((__int64)&src, v214, &v197);
                  break;
                case 0x4E:
                  p_src = &src;
                  sub_17C2D30((__int64)&src, v16);
                  continue;
              }
            }
LABEL_41:
            ;
          }
          while ( v165 != (__int64 *)v21 );
          v28 = v214;
          v29 = src;
          v137 = v214 - (_BYTE *)src;
          if ( v214 == src )
          {
            v33 = 0;
            v150 = 0;
            dest = 0;
            v32 = v215 - (_BYTE *)src;
          }
          else
          {
            if ( v137 > 0x7FFFFFFFFFFFFFF8LL )
              sub_4261EA(p_src, v16, v19);
            v30 = sub_22077B0(v137);
            v29 = src;
            v31 = v30;
            dest = (__int64 *)v30;
            v28 = v214;
            v32 = v215 - (_BYTE *)src;
            v33 = v214 - (_BYTE *)src;
            v150 = (__int64 *)(v214 - (_BYTE *)src + v31);
          }
          if ( v29 != v28 )
          {
            memmove(dest, v29, v33);
            goto LABEL_47;
          }
          if ( v29 )
LABEL_47:
            j_j___libc_free_0(v29, v32);
          if ( dest != v150 )
          {
            v156 = dest;
            v34 = (__int64 *)v159;
            while ( 1 )
            {
              v35 = *v156;
              v36 = sub_14DE7C0(&v172, *v156, &v170, (__int64 *)&v173, &v171);
              v39 = v171;
              v151 = (__int64 *)v36;
              v149 = v40;
              if ( !v171 )
                goto LABEL_83;
              if ( a2 )
              {
                if ( (unsigned __int8)sub_1441AE0(a2) )
                {
                  v35 = v173;
                  if ( !sub_1441CD0((__int64)a2, v173) )
                    goto LABEL_83;
                }
                v39 = v171;
                v41 = *v156;
                if ( dword_4FA30C0 )
                {
                  if ( dword_4CD48D8 <= (unsigned int)dword_4FA30C0 )
                    goto LABEL_83;
                }
                if ( !v171 )
                  goto LABEL_83;
              }
              else
              {
                v41 = *v156;
                if ( dword_4FA30C0 && dword_4CD48D8 <= (unsigned int)dword_4FA30C0 )
                  goto LABEL_83;
              }
              v42 = 0;
              v152 = 0;
              v157 = v39;
              v43 = 0;
              v145 = v41 & 0xFFFFFFFFFFFFFFFBLL;
              v154 = 0;
              v166 = 0;
              v144 = v41 | 4;
              v44 = v41;
              v45 = v151;
              while ( 1 )
              {
                if ( byte_4FA2D40 && *(_BYTE *)(v44 + 16) == 78 )
                {
                  v160 = v43;
                  v46 = sub_15E0530(*v34);
                  if ( !sub_1602790(v46) )
                  {
                    v99 = sub_15E0530(*v34);
                    v100 = sub_16033E0(v99);
                    if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v100 + 48LL))(v100) )
                      goto LABEL_74;
                  }
                  sub_15CA5C0((__int64)&src, (__int64)"pgo-icall-prom", (__int64)"UserOptions", 11, v44);
                  sub_15CAB20((__int64)&src, " Not promote: User options", 0x1Au);
                  v47 = _mm_loadu_si128(&v216);
                  v48 = _mm_loadu_si128(&v219);
                  v198 = (int)v214;
                  v201 = v47;
                  v199 = BYTE4(v214);
                  v204 = v48;
                  v200 = v215;
                  v202 = v217;
                  v197 = (__int64)&unk_49ECF68;
                  v203 = v218;
                  v206 = v221;
                  if ( v221 )
                    v205 = v220;
                  v208 = 0x400000000LL;
                  v207 = v209;
                  if ( v223 )
                  {
                    sub_17C24C0((__int64)&v207, (__int64)&v222);
                    v81 = v222;
                    v210 = v225;
                    v211 = v226;
                    v212 = v227;
                    v197 = (__int64)&unk_49ECFC8;
                    src = &unk_49ECF68;
                    v49 = &v222[11 * v223];
                    if ( v222 != v49 )
                    {
                      do
                      {
                        v49 -= 11;
                        v82 = (_QWORD *)v49[4];
                        if ( v82 != v49 + 6 )
                          j_j___libc_free_0(v82, v49[6] + 1LL);
                        if ( (_QWORD *)*v49 != v49 + 2 )
                          j_j___libc_free_0(*v49, v49[2] + 1LL);
                      }
                      while ( v81 != v49 );
                      v49 = v222;
                    }
                  }
                  else
                  {
                    v49 = v222;
                    v210 = v225;
                    v211 = v226;
                    v212 = v227;
                    v197 = (__int64)&unk_49ECFC8;
                  }
                  if ( v49 != (_QWORD *)v224 )
                    _libc_free((unsigned __int64)v49);
                  sub_143AA50(v34, (__int64)&v197);
                  v50 = v207;
                  v197 = (__int64)&unk_49ECF68;
                  v51 = &v207[88 * (unsigned int)v208];
                  if ( v207 == (_BYTE *)v51 )
                    goto LABEL_72;
                  do
                  {
                    v51 -= 11;
                    v52 = (_QWORD *)v51[4];
                    if ( v52 != v51 + 6 )
                      j_j___libc_free_0(v52, v51[6] + 1LL);
                    if ( (_QWORD *)*v51 != v51 + 2 )
                      j_j___libc_free_0(*v51, v51[2] + 1LL);
                  }
                  while ( v50 != v51 );
LABEL_71:
                  v51 = v207;
                  goto LABEL_72;
                }
                if ( byte_4FA2E20 && *(_BYTE *)(v44 + 16) == 29 )
                {
                  v160 = v43;
                  v60 = sub_15E0530(*v34);
                  if ( !sub_1602790(v60) )
                  {
                    v101 = sub_15E0530(*v34);
                    v102 = sub_16033E0(v101);
                    if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v102 + 48LL))(v102) )
                      goto LABEL_74;
                  }
                  sub_15CA5C0((__int64)&src, (__int64)"pgo-icall-prom", (__int64)"UserOptions", 11, v44);
                  sub_15CAB20((__int64)&src, " Not promote: User options", 0x1Au);
                  v61 = _mm_loadu_si128(&v216);
                  v62 = _mm_loadu_si128(&v219);
                  v198 = (int)v214;
                  v201 = v61;
                  v199 = BYTE4(v214);
                  v204 = v62;
                  v200 = v215;
                  v202 = v217;
                  v197 = (__int64)&unk_49ECF68;
                  v203 = v218;
                  v206 = v221;
                  if ( v221 )
                    v205 = v220;
                  v207 = v209;
                  v208 = 0x400000000LL;
                  if ( v223 )
                  {
                    sub_17C24C0((__int64)&v207, (__int64)&v222);
                    v83 = v222;
                    v210 = v225;
                    v211 = v226;
                    v212 = v227;
                    v197 = (__int64)&unk_49ECFC8;
                    src = &unk_49ECF68;
                    v63 = &v222[11 * v223];
                    if ( v222 != v63 )
                    {
                      do
                      {
                        v63 -= 11;
                        v84 = (_QWORD *)v63[4];
                        if ( v84 != v63 + 6 )
                          j_j___libc_free_0(v84, v63[6] + 1LL);
                        if ( (_QWORD *)*v63 != v63 + 2 )
                          j_j___libc_free_0(*v63, v63[2] + 1LL);
                      }
                      while ( v83 != v63 );
                      v63 = v222;
                    }
                  }
                  else
                  {
                    v63 = v222;
                    v210 = v225;
                    v211 = v226;
                    v212 = v227;
                    v197 = (__int64)&unk_49ECFC8;
                  }
                  if ( v63 != (_QWORD *)v224 )
                    _libc_free((unsigned __int64)v63);
                  sub_143AA50(v34, (__int64)&v197);
                  v64 = v207;
                  v197 = (__int64)&unk_49ECF68;
                  v51 = &v207[88 * (unsigned int)v208];
                  if ( v207 == (_BYTE *)v51 )
                    goto LABEL_72;
                  do
                  {
                    v51 -= 11;
                    v65 = (_QWORD *)v51[4];
                    if ( v65 != v51 + 6 )
                      j_j___libc_free_0(v65, v51[6] + 1LL);
                    if ( (_QWORD *)*v51 != v51 + 2 )
                      j_j___libc_free_0(*v51, v51[2] + 1LL);
                  }
                  while ( v64 != v51 );
                  goto LABEL_71;
                }
                v146 = v45[1];
                if ( dword_4FA31A0 && dword_4CD48F8 >= (unsigned int)dword_4FA31A0 )
                {
                  v160 = v43;
                  v121 = sub_15E0530(*v34);
                  if ( !sub_1602790(v121) )
                  {
                    v132 = sub_15E0530(*v34);
                    v133 = sub_16033E0(v132);
                    if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v133 + 48LL))(v133) )
                      goto LABEL_74;
                  }
                  sub_15CA5C0((__int64)&src, (__int64)"pgo-icall-prom", (__int64)"CutOffReached", 13, v44);
                  sub_15CAB20((__int64)&src, " Not promote: Cutoff reached", 0x1Cu);
                  v122 = _mm_loadu_si128(&v216);
                  v123 = _mm_loadu_si128(&v219);
                  v198 = (int)v214;
                  v201 = v122;
                  v199 = BYTE4(v214);
                  v204 = v123;
                  v200 = v215;
                  v202 = v217;
                  v197 = (__int64)&unk_49ECF68;
                  v203 = v218;
                  v206 = v221;
                  if ( v221 )
                    v205 = v220;
                  v207 = v209;
                  v208 = 0x400000000LL;
                  if ( v223 )
                  {
                    sub_17C24C0((__int64)&v207, (__int64)&v222);
                    v127 = v222;
                    v210 = v225;
                    v211 = v226;
                    v212 = v227;
                    v197 = (__int64)&unk_49ECFC8;
                    src = &unk_49ECF68;
                    v124 = &v222[11 * v223];
                    if ( v222 != v124 )
                    {
                      do
                      {
                        v124 -= 11;
                        v128 = (_QWORD *)v124[4];
                        if ( v128 != v124 + 6 )
                          j_j___libc_free_0(v128, v124[6] + 1LL);
                        if ( (_QWORD *)*v124 != v124 + 2 )
                          j_j___libc_free_0(*v124, v124[2] + 1LL);
                      }
                      while ( v127 != v124 );
                      v124 = v222;
                    }
                  }
                  else
                  {
                    v124 = v222;
                    v210 = v225;
                    v211 = v226;
                    v212 = v227;
                    v197 = (__int64)&unk_49ECFC8;
                  }
                  if ( v124 != (_QWORD *)v224 )
                    _libc_free((unsigned __int64)v124);
                  sub_143AA50(v34, (__int64)&v197);
                  v125 = v207;
                  v197 = (__int64)&unk_49ECF68;
                  v51 = &v207[88 * (unsigned int)v208];
                  if ( v207 != (_BYTE *)v51 )
                  {
                    do
                    {
                      v51 -= 11;
                      v126 = (_QWORD *)v51[4];
                      if ( v126 != v51 + 6 )
                        j_j___libc_free_0(v126, v51[6] + 1LL);
                      if ( (_QWORD *)*v51 != v51 + 2 )
                        j_j___libc_free_0(*v51, v51[2] + 1LL);
                    }
                    while ( v125 != v51 );
                    goto LABEL_71;
                  }
                  goto LABEL_72;
                }
                v161 = *v45;
                sub_16977B0((__int64)v183, v35, (__int64)v42, v37, v38, *v45);
                v66 = v190;
                v67 = (v191 - v190) >> 4;
                if ( v191 - v190 > 0 )
                {
                  do
                  {
                    while ( 1 )
                    {
                      v68 = v67 >> 1;
                      v69 = (unsigned __int64 *)&v66[16 * (v67 >> 1)];
                      if ( v161 <= *v69 )
                        break;
                      v66 = (char *)(v69 + 2);
                      v67 = v67 - v68 - 1;
                      if ( v67 <= 0 )
                        goto LABEL_122;
                    }
                    v67 >>= 1;
                  }
                  while ( v68 > 0 );
                }
LABEL_122:
                if ( v191 == v66 )
                  break;
                if ( v161 != *(_QWORD *)v66 )
                  break;
                v70 = *((_QWORD *)v66 + 1);
                if ( !v70 )
                  break;
                v71 = *(_BYTE *)(v44 + 16);
                v72 = 0;
                s = 0;
                if ( v71 > 0x17u )
                {
                  if ( v71 == 78 )
                  {
                    v72 = v144;
                  }
                  else if ( v71 == 29 )
                  {
                    v72 = v145;
                  }
                }
                v35 = v70;
                v162 = v70;
                v73 = sub_1AB3AB0(v72, v70, &s);
                v74 = v162;
                if ( !v73 )
                {
                  v168 = v162;
                  v160 = v43;
                  v104 = sub_15E0530(*v34);
                  v105 = sub_1602790(v104);
                  v106 = v168;
                  if ( !v105 )
                  {
                    v116 = sub_15E0530(*v34);
                    v117 = sub_16033E0(v116);
                    v118 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v117 + 48LL))(v117);
                    v106 = v168;
                    if ( !v118 )
                      goto LABEL_74;
                  }
                  v169 = v106;
                  sub_15CA5C0((__int64)&src, (__int64)"pgo-icall-prom", (__int64)"UnableToPromote", 15, v44);
                  sub_15CAB20((__int64)&src, "Cannot promote indirect call to ", 0x20u);
                  sub_15C9340((__int64)v179, "TargetFunction", 0xEu, v169);
                  v107 = sub_17C21B0((__int64)&src, (__int64)v179);
                  sub_15CAB20(v107, " with count of ", 0xFu);
                  sub_15C9D40((__int64)v175, "Count", 5, v146);
                  v108 = sub_17C21B0(v107, (__int64)v175);
                  sub_15CAB20(v108, ": ", 2u);
                  v109 = s;
                  v110 = 0;
                  if ( s )
                    v110 = strlen(s);
                  sub_15CAB20(v108, v109, v110);
                  v198 = *(_DWORD *)(v108 + 8);
                  v199 = *(_BYTE *)(v108 + 12);
                  v200 = *(_BYTE **)(v108 + 16);
                  v201 = _mm_loadu_si128((const __m128i *)(v108 + 24));
                  v202 = *(_QWORD *)(v108 + 40);
                  v197 = (__int64)&unk_49ECF68;
                  v203 = *(_QWORD *)(v108 + 48);
                  v204 = _mm_loadu_si128((const __m128i *)(v108 + 56));
                  v206 = *(_BYTE *)(v108 + 80);
                  if ( v206 )
                    v205 = *(_QWORD *)(v108 + 72);
                  v207 = v209;
                  v208 = 0x400000000LL;
                  if ( *(_DWORD *)(v108 + 96) )
                    sub_17C24C0((__int64)&v207, v108 + 88);
                  v210 = *(_BYTE *)(v108 + 456);
                  v211 = *(_DWORD *)(v108 + 460);
                  v212 = *(_QWORD *)(v108 + 464);
                  v197 = (__int64)&unk_49ECFC8;
                  if ( v177 != &v178 )
                    j_j___libc_free_0(v177, v178 + 1);
                  if ( (__int64 *)v175[0] != &v176 )
                    j_j___libc_free_0(v175[0], v176 + 1);
                  if ( v181 != &v182 )
                    j_j___libc_free_0(v181, v182 + 1);
                  if ( (__int64 *)v179[0] != &v180 )
                    j_j___libc_free_0(v179[0], v180 + 1);
                  v111 = v222;
                  src = &unk_49ECF68;
                  v112 = &v222[11 * v223];
                  if ( v222 != v112 )
                  {
                    do
                    {
                      v112 -= 11;
                      v113 = (_QWORD *)v112[4];
                      if ( v113 != v112 + 6 )
                        j_j___libc_free_0(v113, v112[6] + 1LL);
                      if ( (_QWORD *)*v112 != v112 + 2 )
                        j_j___libc_free_0(*v112, v112[2] + 1LL);
                    }
                    while ( v111 != v112 );
                    v112 = v222;
                  }
                  if ( v112 != (_QWORD *)v224 )
                    _libc_free((unsigned __int64)v112);
                  sub_143AA50(v34, (__int64)&v197);
                  v114 = v207;
                  v197 = (__int64)&unk_49ECF68;
                  v51 = &v207[88 * (unsigned int)v208];
                  if ( v207 == (_BYTE *)v51 )
                    goto LABEL_72;
                  do
                  {
                    v51 -= 11;
                    v115 = (_QWORD *)v51[4];
                    if ( v115 != v51 + 6 )
                      j_j___libc_free_0(v115, v51[6] + 1LL);
                    if ( (_QWORD *)*v51 != v51 + 2 )
                      j_j___libc_free_0(*v51, v51[2] + 1LL);
                  }
                  while ( v114 != v51 );
                  goto LABEL_71;
                }
                if ( v152 == v43 )
                {
                  v85 = (char *)v152 - (char *)v154;
                  v86 = v152 - v154;
                  if ( v86 == 0x7FFFFFFFFFFFFFFLL )
                    sub_4262D8((__int64)"vector::_M_realloc_insert");
                  v35 = 1;
                  if ( v86 )
                    v35 = v152 - v154;
                  v87 = __CFADD__(v35, v86);
                  v88 = v35 + v86;
                  if ( !v87 )
                  {
                    if ( v88 )
                    {
                      v35 = 0x7FFFFFFFFFFFFFFLL;
                      v89 = 0x7FFFFFFFFFFFFFFLL;
                      if ( v88 <= 0x7FFFFFFFFFFFFFFLL )
                        v89 = v88;
                      v90 = 16 * v89;
LABEL_184:
                      v134 = v162;
                      v163 = v90;
                      v91 = sub_22077B0(v90);
                      v85 = (char *)v152 - (char *)v154;
                      v74 = v134;
                      v92 = (__m128i *)v91;
                      v93 = (__m128i *)(v91 + v163);
                      v94 = v91 + 16;
                    }
                    else
                    {
                      v94 = 16;
                      v93 = 0;
                      v92 = 0;
                    }
                    v95 = (unsigned __int64 *)((char *)v92->m128i_u64 + v85);
                    if ( v95 )
                    {
                      v35 = v146;
                      *v95 = v74;
                      v95[1] = v146;
                    }
                    v42 = v154;
                    if ( v43 == v154 )
                    {
                      v43 = (__m128i *)v94;
                    }
                    else
                    {
                      v96 = v154;
                      v35 = (char *)v43 - (char *)v154;
                      v42 = v92;
                      do
                      {
                        if ( v42 )
                          *v42 = _mm_loadu_si128(v96);
                        ++v96;
                        ++v42;
                      }
                      while ( v96 != v43 );
                      v43 = (__m128i *)((char *)v92 + v35 + 16);
                    }
                    v97 = v154;
                    if ( v154 )
                    {
                      v98 = v154;
                      v155 = v92;
                      v164 = v93;
                      v35 = (char *)v152 - (char *)v97;
                      j_j___libc_free_0(v98, (char *)v152 - (char *)v97);
                      v92 = v155;
                      v93 = v164;
                    }
                    v152 = v93;
                    v154 = v92;
                    goto LABEL_134;
                  }
                  v90 = 0x7FFFFFFFFFFFFFF0LL;
                  goto LABEL_184;
                }
                if ( v43 )
                {
                  v43->m128i_i64[0] = v162;
                  v43->m128i_i64[1] = v146;
                }
                ++v43;
LABEL_134:
                ++v166;
                v45 += 2;
                if ( v166 >= v157 )
                {
                  v160 = v43;
                  goto LABEL_74;
                }
              }
              v160 = v43;
              v75 = sub_15E0530(*v34);
              if ( !sub_1602790(v75) )
              {
                v130 = sub_15E0530(*v34);
                v131 = sub_16033E0(v130);
                if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v131 + 48LL))(v131) )
                  goto LABEL_74;
              }
              sub_15CA5C0((__int64)&src, (__int64)"pgo-icall-prom", (__int64)"UnableToFindTarget", 18, v44);
              sub_15CAB20((__int64)&src, "Cannot promote indirect call: target not found", 0x2Eu);
              v76 = _mm_loadu_si128(&v216);
              v77 = _mm_loadu_si128(&v219);
              v198 = (int)v214;
              v201 = v76;
              v199 = BYTE4(v214);
              v204 = v77;
              v200 = v215;
              v202 = v217;
              v197 = (__int64)&unk_49ECF68;
              v203 = v218;
              v206 = v221;
              if ( v221 )
                v205 = v220;
              v207 = v209;
              v208 = 0x400000000LL;
              if ( v223 )
              {
                sub_17C24C0((__int64)&v207, (__int64)&v222);
                v119 = v222;
                v210 = v225;
                v211 = v226;
                v212 = v227;
                v197 = (__int64)&unk_49ECFC8;
                src = &unk_49ECF68;
                v78 = &v222[11 * v223];
                if ( v222 != v78 )
                {
                  do
                  {
                    v78 -= 11;
                    v120 = (_QWORD *)v78[4];
                    if ( v120 != v78 + 6 )
                      j_j___libc_free_0(v120, v78[6] + 1LL);
                    if ( (_QWORD *)*v78 != v78 + 2 )
                      j_j___libc_free_0(*v78, v78[2] + 1LL);
                  }
                  while ( v119 != v78 );
                  v78 = v222;
                }
              }
              else
              {
                v78 = v222;
                v210 = v225;
                v211 = v226;
                v212 = v227;
                v197 = (__int64)&unk_49ECFC8;
              }
              if ( v78 != (_QWORD *)v224 )
                _libc_free((unsigned __int64)v78);
              sub_143AA50(v34, (__int64)&v197);
              v79 = v207;
              v197 = (__int64)&unk_49ECF68;
              v51 = &v207[88 * (unsigned int)v208];
              if ( v207 != (_BYTE *)v51 )
              {
                do
                {
                  v51 -= 11;
                  v80 = (_QWORD *)v51[4];
                  if ( v80 != v51 + 6 )
                    j_j___libc_free_0(v80, v51[6] + 1LL);
                  if ( (_QWORD *)*v51 != v51 + 2 )
                    j_j___libc_free_0(*v51, v51[2] + 1LL);
                }
                while ( v79 != v51 );
                goto LABEL_71;
              }
LABEL_72:
              if ( v51 != (_QWORD *)v209 )
                _libc_free((unsigned __int64)v51);
LABEL_74:
              v158 = *v156;
              v153 = (char *)v152 - (char *)v154;
              if ( v160 != v154 )
              {
                v53 = a4;
                v54 = v173;
                LODWORD(v55) = 0;
                v56 = (__int64 *)v154;
                do
                {
                  v57 = v56[1];
                  v58 = *v56;
                  v55 = (unsigned int)(v55 + 1);
                  v167 = v53;
                  v56 += 2;
                  sub_17C2750(v158, v58, v57, v54, v53, v34);
                  v53 = v167;
                  v54 = v173 - v57;
                  v173 -= v57;
                }
                while ( v160 != (__m128i *)v56 );
                if ( (_DWORD)v55 )
                {
                  sub_1625C10(*v156, 2, 0);
                  if ( v173 && v170 != (_DWORD)v55 )
                  {
                    sub_1694FA0((__int64 **)a1, *v156, &v151[2 * v55], v149 - v55, v173, 0, v171);
                    if ( v154 )
                      j_j___libc_free_0(v154, v153);
                    v143 = 1;
                    goto LABEL_83;
                  }
                  v143 = 1;
                }
              }
              if ( v154 )
                j_j___libc_free_0(v154, v153);
LABEL_83:
              if ( v150 == ++v156 )
              {
                v141 |= v143;
                break;
              }
            }
          }
          if ( dest )
            j_j___libc_free_0(dest, v137);
        }
        if ( v172 )
          j_j___libc_free_0_0(v172);
        if ( dword_4FA31A0 && dword_4CD48F8 >= (unsigned int)dword_4FA31A0 )
        {
          if ( v138 )
          {
            v129 = *(_QWORD *)(v138 + 16);
            if ( v129 )
            {
              sub_1368A00(*(__int64 **)(v138 + 16));
              j_j___libc_free_0(v129, 8);
            }
            j_j___libc_free_0(v138, 24);
          }
LABEL_8:
          v9 = v193;
          if ( !v193 )
            goto LABEL_10;
          goto LABEL_9;
        }
        if ( v138 )
        {
          v59 = *(_QWORD *)(v138 + 16);
          if ( v59 )
          {
            sub_1368A00(*(__int64 **)(v138 + 16));
            j_j___libc_free_0(v59, 8);
          }
          j_j___libc_free_0(v138, 24);
        }
      }
    }
    v148 = *(_QWORD *)(v148 + 8);
  }
  while ( v139 != v148 );
  v9 = v193;
  if ( v193 )
LABEL_9:
    j_j___libc_free_0(v9, v195 - v9);
LABEL_10:
  if ( v190 )
    j_j___libc_free_0(v190, v192 - (_QWORD)v190);
  if ( v187 )
    j_j___libc_free_0(v187, v189 - v187);
  v10 = v184;
  if ( HIDWORD(v185) && (_DWORD)v185 )
  {
    v11 = 8LL * (unsigned int)v185;
    v12 = 0;
    do
    {
      v13 = *(_QWORD *)(v10 + v12);
      if ( v13 != -8 && v13 )
      {
        _libc_free(v13);
        v10 = v184;
      }
      v12 += 8;
    }
    while ( v11 != v12 );
  }
  _libc_free(v10);
  return v141;
}
