// Function: sub_E5CCC0
// Address: 0xe5ccc0
//
unsigned __int64 __fastcall sub_E5CCC0(__int64 *a1, _QWORD *a2, __int64 *a3)
{
  unsigned __int64 result; // rax
  __int64 v6; // r13
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // r8
  __int64 v10; // rdx
  __int64 v11; // rax
  __m128i v12; // rax
  char v13; // al
  __m128i *v14; // rcx
  char v15; // al
  __m128i *v16; // rsi
  char v17; // cl
  __m128i *v18; // rdx
  __int64 v19; // r8
  __int64 v20; // rdx
  __int64 v21; // rax
  __m128i v22; // rax
  char v23; // al
  __m128i *v24; // rcx
  char v25; // al
  __m128i *v26; // rsi
  char v27; // cl
  __m128i *v28; // rdx
  __int64 v30; // r14
  _QWORD *v31; // rsi
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  int v36; // ebx
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // kr00_8
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  unsigned int v46; // eax
  unsigned __int32 v47; // edx
  size_t v48; // rdx
  unsigned __int64 v49; // r12
  __int64 v50; // rbx
  char v51; // si
  __int64 v52; // rdi
  __int64 v53; // rbx
  __int64 v54; // rcx
  __int64 v55; // r12
  __int64 v56; // rdx
  __int64 v57; // rdi
  __int64 v58; // rcx
  __int64 v59; // r8
  __int64 v60; // r9
  __int64 v61; // rcx
  __int64 v62; // r8
  __int64 v63; // r9
  unsigned int v64; // esi
  unsigned __int64 v65; // rdi
  unsigned __int8 v66; // dl
  __int64 v67; // rax
  char v68; // cl
  __int64 v69; // rax
  unsigned __int64 v70; // rcx
  unsigned __int64 v71; // r12
  __int64 v72; // rbx
  size_t v73; // r14
  size_t v74; // r12
  _QWORD *v75; // rdx
  unsigned __int64 v76; // rsi
  unsigned int v77; // ecx
  __int64 v78; // rcx
  __int64 v79; // r8
  __int64 v80; // r9
  __int64 v81; // rcx
  __int64 v82; // r8
  __int64 v83; // r9
  __int64 v84; // r12
  __int64 v85; // rdx
  __int64 v86; // rax
  __int16 v87; // dx
  unsigned __int64 v88; // rax
  unsigned __int64 v89; // rdx
  __int64 v90; // rax
  unsigned __int32 v91; // edx
  __m128i v92; // xmm0
  __m128i v93; // xmm1
  __m128i v94; // xmm2
  __m128i v95; // xmm3
  unsigned __int64 v96; // r8
  unsigned __int64 v97; // rdx
  unsigned int v98; // r12d
  unsigned __int64 v99; // rsi
  unsigned int v100; // eax
  __int64 v101; // rsi
  _BYTE *v102; // rdi
  unsigned __int64 v103; // rcx
  unsigned __int64 v104; // r8
  __int64 v105; // rdx
  unsigned __int64 v106; // rax
  char v107; // r10
  __int64 v108; // r9
  unsigned __int64 v109; // rdx
  unsigned int v110; // r12d
  unsigned __int64 v111; // rsi
  unsigned int v112; // eax
  __int64 v113; // rsi
  _BYTE *v114; // r8
  unsigned __int64 v115; // rcx
  __int64 v116; // rdx
  unsigned __int64 v117; // rax
  char v118; // r10
  __int64 v119; // r9
  __m128i *v120; // rax
  __int64 v121; // rcx
  __m128i *v122; // rax
  unsigned __int64 v123; // rax
  unsigned __int64 v124; // rdi
  unsigned __int64 v125; // rcx
  __m128i *v126; // rax
  __int64 v127; // rcx
  __m128i *v128; // rdx
  __int64 v129; // rcx
  __m128i *v130; // rax
  __m128i v131; // xmm7
  __m128i v132; // xmm5
  __int64 v133; // rcx
  __int64 v134; // r8
  __int64 v135; // r9
  __int64 v136; // rcx
  __int64 v137; // r8
  __int64 v138; // r9
  __int64 v139; // rcx
  __int64 v140; // r8
  __int64 v141; // r9
  __int64 v142; // [rsp+8h] [rbp-228h]
  unsigned __int64 v143; // [rsp+8h] [rbp-228h]
  unsigned __int64 v144; // [rsp+8h] [rbp-228h]
  char v145; // [rsp+8h] [rbp-228h]
  __int64 v146; // [rsp+10h] [rbp-220h]
  __int64 v147; // [rsp+18h] [rbp-218h]
  unsigned __int64 v148; // [rsp+18h] [rbp-218h]
  unsigned __int64 v149; // [rsp+18h] [rbp-218h]
  int v150; // [rsp+18h] [rbp-218h]
  __int64 v151; // [rsp+20h] [rbp-210h]
  unsigned __int64 v152; // [rsp+20h] [rbp-210h]
  unsigned __int64 v153; // [rsp+20h] [rbp-210h]
  __int64 v154; // [rsp+28h] [rbp-208h]
  __int64 v155; // [rsp+28h] [rbp-208h]
  __int64 v156; // [rsp+30h] [rbp-200h]
  __int64 v157; // [rsp+30h] [rbp-200h]
  __int64 v158; // [rsp+30h] [rbp-200h]
  __int64 v159; // [rsp+38h] [rbp-1F8h]
  __int64 v160; // [rsp+38h] [rbp-1F8h]
  unsigned __int64 v161; // [rsp+38h] [rbp-1F8h]
  unsigned __int64 v162; // [rsp+40h] [rbp-1F0h] BYREF
  unsigned __int64 v163; // [rsp+48h] [rbp-1E8h] BYREF
  __m128i v164[2]; // [rsp+50h] [rbp-1E0h] BYREF
  char v165; // [rsp+70h] [rbp-1C0h]
  char v166; // [rsp+71h] [rbp-1BFh]
  __m128i v167[2]; // [rsp+80h] [rbp-1B0h] BYREF
  __int16 v168; // [rsp+A0h] [rbp-190h]
  __m128i v169; // [rsp+B0h] [rbp-180h] BYREF
  _QWORD v170[4]; // [rsp+C0h] [rbp-170h] BYREF
  __m128i v171; // [rsp+E0h] [rbp-150h] BYREF
  __m128i v172; // [rsp+F0h] [rbp-140h] BYREF
  char v173; // [rsp+100h] [rbp-130h]
  char v174; // [rsp+101h] [rbp-12Fh]
  __m128i v175; // [rsp+110h] [rbp-120h] BYREF
  __m128i v176; // [rsp+120h] [rbp-110h] BYREF
  __int64 v177; // [rsp+130h] [rbp-100h]
  __m128i v178; // [rsp+140h] [rbp-F0h] BYREF
  _QWORD v179[2]; // [rsp+150h] [rbp-E0h] BYREF
  __int16 v180; // [rsp+160h] [rbp-D0h]
  __m128i v181; // [rsp+170h] [rbp-C0h] BYREF
  __m128i v182; // [rsp+180h] [rbp-B0h] BYREF
  __int64 v183; // [rsp+190h] [rbp-A0h]
  __m128i v184; // [rsp+1A0h] [rbp-90h] BYREF
  __m128i v185; // [rsp+1B0h] [rbp-80h] BYREF
  char v186; // [rsp+1C0h] [rbp-70h]
  char v187; // [rsp+1C1h] [rbp-6Fh]
  _BYTE v188[8]; // [rsp+1C8h] [rbp-68h]
  __m128i v189; // [rsp+1D0h] [rbp-60h] BYREF
  __m128i v190; // [rsp+1E0h] [rbp-50h]
  __int64 v191; // [rsp+1F0h] [rbp-40h]

  if ( (a3[6] & 0x20) != 0 )
  {
    result = a3[1];
    v6 = *(_QWORD *)result;
    if ( !*(_QWORD *)result )
      return result;
    while ( 1 )
    {
      while ( 1 )
      {
        result = *(unsigned __int8 *)(v6 + 28);
        if ( (_BYTE)result == 2 )
          goto LABEL_5;
        if ( (unsigned __int8)result <= 2u )
          break;
        if ( (_BYTE)result != 5 )
          goto LABEL_225;
        v6 = *(_QWORD *)v6;
        if ( !v6 )
          return result;
      }
      if ( (_BYTE)result )
        break;
LABEL_5:
      v6 = *(_QWORD *)v6;
      if ( !v6 )
        return result;
    }
    if ( !*(_DWORD *)(v6 + 104) )
    {
LABEL_12:
      v7 = *(_QWORD *)(v6 + 48);
      if ( (_DWORD)v7 )
      {
        result = *(_QWORD *)(v6 + 40);
        v8 = result + (unsigned int)(v7 - 1) + 1;
        while ( !*(_BYTE *)result )
        {
          if ( v8 == ++result )
            goto LABEL_5;
        }
        v9 = *a1;
        v187 = 1;
        v184.m128i_i64[0] = (__int64)"' cannot have non-zero initializers";
        v10 = a3[16];
        v159 = v9;
        v178.m128i_i64[1] = a3[17];
        v11 = *a3;
        v180 = 261;
        v186 = 3;
        v178.m128i_i64[0] = v10;
        v12.m128i_i64[0] = (*(__int64 (__fastcall **)(__int64 *))(v11 + 16))(a3);
        v176.m128i_i64[0] = (__int64)" section '";
        v175 = v12;
        v13 = v180;
        LOWORD(v177) = 773;
        if ( (_BYTE)v180 )
        {
          if ( (_BYTE)v180 == 1 )
          {
            v92 = _mm_load_si128(&v175);
            v93 = _mm_load_si128(&v176);
            v183 = v177;
            v15 = v186;
            v181 = v92;
            v182 = v93;
            if ( v186 )
            {
              if ( v186 == 1 )
                goto LABEL_198;
              if ( BYTE1(v183) == 1 )
              {
                v17 = 5;
                v146 = v181.m128i_i64[1];
                v16 = (__m128i *)v181.m128i_i64[0];
              }
              else
              {
LABEL_22:
                v16 = &v181;
                v17 = 2;
              }
              if ( v187 == 1 )
              {
                v142 = v184.m128i_i64[1];
                v18 = (__m128i *)v184.m128i_i64[0];
              }
              else
              {
                v18 = &v184;
                v15 = 2;
              }
              v189.m128i_i64[0] = (__int64)v16;
              v190.m128i_i64[0] = (__int64)v18;
              v189.m128i_i64[1] = v146;
              LOBYTE(v191) = v17;
              v190.m128i_i64[1] = v142;
              BYTE1(v191) = v15;
LABEL_41:
              result = sub_E66880(v159, 0, &v189);
              goto LABEL_5;
            }
          }
          else
          {
            if ( HIBYTE(v180) == 1 )
            {
              v154 = v178.m128i_i64[1];
              v14 = (__m128i *)v178.m128i_i64[0];
            }
            else
            {
              v14 = &v178;
              v13 = 2;
            }
            BYTE1(v183) = v13;
            v15 = v186;
            v181.m128i_i64[0] = (__int64)&v175;
            v182.m128i_i64[0] = (__int64)v14;
            v182.m128i_i64[1] = v154;
            LOBYTE(v183) = 2;
            if ( v186 )
            {
              if ( v186 != 1 )
                goto LABEL_22;
LABEL_198:
              v131 = _mm_load_si128(&v182);
              v189 = _mm_load_si128(&v181);
              v191 = v183;
              v190 = v131;
              goto LABEL_41;
            }
          }
        }
        else
        {
          LOWORD(v183) = 256;
        }
        LOWORD(v191) = 256;
        goto LABEL_41;
      }
      goto LABEL_5;
    }
    v19 = *a1;
    v187 = 1;
    v20 = a3[16];
    v186 = 3;
    v184.m128i_i64[0] = (__int64)"' cannot have fixups";
    v160 = v19;
    v178.m128i_i64[1] = a3[17];
    v21 = *a3;
    v180 = 261;
    v178.m128i_i64[0] = v20;
    v22.m128i_i64[0] = (*(__int64 (__fastcall **)(__int64 *))(v21 + 16))(a3);
    v176.m128i_i64[0] = (__int64)" section '";
    v175 = v22;
    v23 = v180;
    LOWORD(v177) = 773;
    if ( (_BYTE)v180 )
    {
      if ( (_BYTE)v180 == 1 )
      {
        v94 = _mm_load_si128(&v175);
        v95 = _mm_load_si128(&v176);
        v183 = v177;
        v25 = v186;
        v181 = v94;
        v182 = v95;
        if ( v186 )
        {
          if ( v186 != 1 )
          {
            if ( BYTE1(v183) == 1 )
            {
              v27 = 5;
              v151 = v181.m128i_i64[1];
              v26 = (__m128i *)v181.m128i_i64[0];
              goto LABEL_33;
            }
LABEL_32:
            v26 = &v181;
            v27 = 2;
LABEL_33:
            if ( v187 == 1 )
            {
              v147 = v184.m128i_i64[1];
              v28 = (__m128i *)v184.m128i_i64[0];
            }
            else
            {
              v28 = &v184;
              v25 = 2;
            }
            v189.m128i_i64[0] = (__int64)v26;
            v190.m128i_i64[0] = (__int64)v28;
            v189.m128i_i64[1] = v151;
            LOBYTE(v191) = v27;
            v190.m128i_i64[1] = v147;
            BYTE1(v191) = v25;
            goto LABEL_38;
          }
          goto LABEL_199;
        }
      }
      else
      {
        if ( HIBYTE(v180) == 1 )
        {
          v156 = v178.m128i_i64[1];
          v24 = (__m128i *)v178.m128i_i64[0];
        }
        else
        {
          v24 = &v178;
          v23 = 2;
        }
        BYTE1(v183) = v23;
        v25 = v186;
        v181.m128i_i64[0] = (__int64)&v175;
        v182.m128i_i64[0] = (__int64)v24;
        v182.m128i_i64[1] = v156;
        LOBYTE(v183) = 2;
        if ( v186 )
        {
          if ( v186 != 1 )
            goto LABEL_32;
LABEL_199:
          v132 = _mm_load_si128(&v182);
          v189 = _mm_load_si128(&v181);
          v191 = v183;
          v190 = v132;
          goto LABEL_38;
        }
      }
    }
    else
    {
      LOWORD(v183) = 256;
    }
    LOWORD(v191) = 256;
LABEL_38:
    result = sub_E66880(v160, 0, &v189);
    goto LABEL_12;
  }
  (*(void (__fastcall **)(_QWORD *))(*a2 + 80LL))(a2);
  result = a3[1];
  v30 = *(_QWORD *)result;
  if ( *(_QWORD *)result )
  {
    while ( 1 )
    {
      v31 = (_QWORD *)v30;
      v32 = sub_E5BD20(a1, v30);
      v34 = *(unsigned __int8 *)(v30 + 28);
      v162 = v32;
      v35 = v32;
      v36 = *(_DWORD *)(a1[1] + 8);
      if ( (unsigned __int8)v34 <= 0xDu && ((1LL << v34) & 0x20D2) != 0 )
      {
        v31 = a2;
        sub_E5CBA0((__int64)a1, (__int64)a2, v30, v32);
      }
      v39 = (*(__int64 (__fastcall **)(_QWORD *, _QWORD *, __int64, __int64, __int64))(*a2 + 80LL))(
              a2,
              v31,
              v33,
              v34,
              v35);
      result = *(unsigned __int8 *)(v30 + 28);
      switch ( *(_BYTE *)(v30 + 28) )
      {
        case 0:
          v76 = *(unsigned int *)(v30 + 40);
          v77 = v76;
          result = v162 / v76;
          v163 = v162 / v76;
          if ( v162 != v162 / v76 * v76 )
          {
            v167[0].m128i_i32[0] = v76;
            v184.m128i_i64[0] = (__int64)"'";
            v178.m128i_i64[0] = (__int64)&v162;
            v171.m128i_i64[0] = (__int64)"' is not a divisor of padding size '";
            v164[0].m128i_i64[0] = (__int64)"undefined .align directive, value size '";
            v187 = 1;
            v186 = 3;
            v180 = 267;
            v174 = 1;
            v173 = 3;
            v168 = 265;
            v166 = 1;
            v165 = 3;
            sub_9C6370(&v169, v164, v167, v76, v37, v38);
            sub_9C6370(&v175, &v169, &v171, v133, v134, v135);
            sub_9C6370(&v181, &v175, &v178, v136, v137, v138);
            sub_9C6370(&v189, &v181, &v184, v139, v140, v141);
            sub_C64D30((__int64)&v189, 1u);
          }
          if ( (*(_BYTE *)(v30 + 31) & 1) != 0 )
          {
            result = (*(__int64 (__fastcall **)(__int64, _QWORD *, unsigned __int64, _QWORD))(*(_QWORD *)a1[1] + 192LL))(
                       a1[1],
                       a2,
                       result,
                       *(_QWORD *)(v30 + 48));
            if ( !(_BYTE)result )
            {
              v187 = 1;
              v184.m128i_i64[0] = (__int64)" bytes";
              v178.m128i_i64[0] = (__int64)&v163;
              v186 = 3;
              v175.m128i_i64[0] = (__int64)"unable to write nop sequence of ";
              v180 = 267;
              LOWORD(v177) = 259;
              sub_9C6370(&v181, &v175, &v178, v78, v79, v80);
              sub_9C6370(&v189, &v181, &v184, v81, v82, v83);
              sub_C64D30((__int64)&v189, 1u);
            }
          }
          else if ( v162 >= v76 )
          {
            v84 = 0;
            while ( 1 )
            {
              if ( v77 == 4 )
              {
                v90 = *(_QWORD *)(v30 + 32);
                v91 = _byteswap_ulong(v90);
                if ( v36 != 1 )
                  LODWORD(v90) = v91;
                v189.m128i_i32[0] = v90;
                result = sub_CB6200((__int64)a2, (unsigned __int8 *)&v189, 4u);
              }
              else if ( v77 > 4 )
              {
                if ( v77 != 8 )
LABEL_225:
                  BUG();
                v88 = *(_QWORD *)(v30 + 32);
                v89 = _byteswap_uint64(v88);
                if ( v36 != 1 )
                  v88 = v89;
                v189.m128i_i64[0] = v88;
                result = sub_CB6200((__int64)a2, (unsigned __int8 *)&v189, 8u);
              }
              else if ( v77 == 1 )
              {
                v85 = *(_QWORD *)(v30 + 32);
                result = a2[4];
                if ( result >= a2[3] )
                {
                  result = sub_CB5D20((__int64)a2, v85);
                }
                else
                {
                  a2[4] = result + 1;
                  *(_BYTE *)result = v85;
                }
              }
              else
              {
                if ( v77 != 2 )
                  goto LABEL_225;
                v86 = *(_QWORD *)(v30 + 32);
                v87 = __ROL2__(v86, 8);
                if ( v36 != 1 )
                  LOWORD(v86) = v87;
                v189.m128i_i16[0] = v86;
                result = sub_CB6200((__int64)a2, (unsigned __int8 *)&v189, 2u);
              }
              if ( v163 == ++v84 )
                goto LABEL_48;
              v77 = *(_DWORD *)(v30 + 40);
            }
          }
          goto LABEL_48;
        case 1:
        case 4:
        case 6:
        case 7:
        case 8:
        case 0xC:
        case 0xD:
          result = sub_CB6200((__int64)a2, *(unsigned __int8 **)(v30 + 40), *(_QWORD *)(v30 + 48));
          goto LABEL_48;
        case 2:
          v64 = *(unsigned __int8 *)(v30 + 30);
          v65 = *(_QWORD *)(v30 + 32);
          v66 = *(_BYTE *)(v30 + 30);
          if ( !v66 )
            goto LABEL_81;
          v67 = 0;
          do
          {
            v68 = v64 - 1 - v67;
            if ( v36 == 1 )
              v68 = v67;
            v189.m128i_i8[v67++] = v65 >> (8 * v68);
          }
          while ( v64 != (_DWORD)v67 );
          if ( v64 != 16 )
          {
LABEL_81:
            v69 = v66;
            do
            {
              v189.m128i_i8[v69] = v189.m128i_i8[v69 - v66];
              ++v69;
            }
            while ( (unsigned int)v69 <= 0xF );
          }
          v70 = v162;
          v71 = v64 * (0x10 / v64);
          if ( v71 > v162 )
            goto LABEL_90;
          v157 = v30;
          v72 = 0;
          v73 = v64 * (0x10 / v64);
          v74 = v162 / v73;
          do
          {
            while ( 1 )
            {
              v75 = (_QWORD *)a2[4];
              if ( v73 <= a2[3] - (_QWORD)v75 )
                break;
              ++v72;
              sub_CB6200((__int64)a2, (unsigned __int8 *)&v189, v73);
              if ( v74 == v72 )
                goto LABEL_89;
            }
            if ( v64 * (0x10 / v64) )
            {
              if ( (unsigned int)v73 >= 8 )
              {
                *v75 = v189.m128i_i64[0];
                *(_QWORD *)((char *)v75 + v73 - 8) = *(_QWORD *)&v188[v73];
                qmemcpy(
                  (void *)((unsigned __int64)(v75 + 1) & 0xFFFFFFFFFFFFFFF8LL),
                  (const void *)((char *)&v189 - ((char *)v75 - ((unsigned __int64)(v75 + 1) & 0xFFFFFFFFFFFFFFF8LL))),
                  8LL * (((unsigned int)v73 + (_DWORD)v75 - (((_DWORD)v75 + 8) & 0xFFFFFFF8)) >> 3));
              }
              else if ( (((_BYTE)v64 * (0x10 / v64)) & 4) != 0 )
              {
                *(_DWORD *)v75 = v189.m128i_i32[0];
                *(_DWORD *)((char *)v75 + v73 - 4) = *(_DWORD *)&v188[v73 + 4];
              }
              else
              {
                *(_BYTE *)v75 = v189.m128i_i8[0];
                if ( (v73 & 2) != 0 )
                  *(_WORD *)((char *)v75 + v73 - 2) = *(_WORD *)&v188[v73 + 6];
              }
              a2[4] += v73;
            }
            ++v72;
          }
          while ( v74 != v72 );
LABEL_89:
          v71 = v64 * (0x10 / v64);
          v30 = v157;
          v70 = v162;
LABEL_90:
          result = v70 / v71;
          v48 = v70 % v71;
          if ( v70 % v71 )
LABEL_55:
            result = sub_CB6200((__int64)a2, (unsigned __int8 *)&v189, v48);
          goto LABEL_48;
        case 3:
          v52 = a1[1];
          v53 = *(_QWORD *)(v30 + 32);
          v54 = *(_QWORD *)(v30 + 40);
          result = *(_QWORD *)(*(_QWORD *)v52 + 184LL);
          if ( (__int64 (*)())result == sub_E5B890 )
          {
            if ( v54 <= 0 )
            {
              v161 = 0;
              goto LABEL_66;
            }
            v96 = 0;
            v161 = 0;
            v155 = *a1;
LABEL_215:
            v150 = v96;
            v153 = v54;
            v178.m128i_i64[0] = (__int64)v179;
            sub_2240A50(&v178, 1, 45, v54, v96);
            v102 = (_BYTE *)v178.m128i_i64[0];
            v103 = v153;
            LODWORD(v104) = v150;
LABEL_146:
            v104 = (unsigned int)(v104 + 48);
            *v102 = v104;
            goto LABEL_147;
          }
          v158 = *(_QWORD *)(v30 + 40);
          LODWORD(result) = ((__int64 (__fastcall *)(__int64, _QWORD))result)(v52, *(_QWORD *)(v30 + 56));
          v54 = v158;
          result = (unsigned int)result;
          v161 = result;
          if ( (unsigned int)result >= v158 )
            goto LABEL_66;
          v96 = (unsigned int)result;
          v155 = *a1;
          if ( (unsigned int)result <= 9uLL )
            goto LABEL_215;
          if ( (unsigned int)result <= 0x63uLL )
          {
            v149 = (unsigned int)result;
            v178.m128i_i64[0] = (__int64)v179;
            sub_2240A50(&v178, 2, 45, v158, (unsigned int)result);
            v102 = (_BYTE *)v178.m128i_i64[0];
            v103 = v158;
            v104 = v149;
          }
          else
          {
            if ( (unsigned int)result <= 0x3E7uLL )
            {
              v98 = 2;
              v101 = 3;
            }
            else if ( (unsigned int)result <= 0x270FuLL )
            {
              v98 = 3;
              v101 = 4;
            }
            else
            {
              v97 = (unsigned int)result;
              v98 = 1;
              do
              {
                v99 = v97;
                v100 = v98;
                v98 += 4;
                v97 /= 0x2710u;
                if ( v99 <= 0x1869F )
                {
                  v101 = v98;
                  v98 = v100 + 3;
                  goto LABEL_143;
                }
                if ( v99 <= 0xF423F )
                {
                  v178.m128i_i64[0] = (__int64)v179;
                  v152 = v96;
                  sub_2240A50(&v178, v100 + 5, 45, v158, v96);
                  v102 = (_BYTE *)v178.m128i_i64[0];
                  v104 = v152;
                  v103 = v158;
                  goto LABEL_144;
                }
                if ( v99 <= (unsigned __int64)&loc_98967F )
                {
                  v101 = v100 + 6;
                  v98 = v100 + 5;
                  goto LABEL_143;
                }
              }
              while ( v99 > 0x5F5E0FF );
              v101 = v100 + 7;
              v98 = v100 + 6;
            }
LABEL_143:
            v148 = v96;
            v178.m128i_i64[0] = (__int64)v179;
            sub_2240A50(&v178, v101, 45, v158, v96);
            v102 = (_BYTE *)v178.m128i_i64[0];
            v103 = v158;
            v104 = v148;
            do
            {
LABEL_144:
              v105 = v104
                   - 20
                   * (v104 / 0x64
                    + ((((v104 >> 2) * (unsigned __int128)0x28F5C28F5C28F5C3uLL) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
              v106 = v104;
              v104 /= 0x64u;
              v107 = a00010203040506_0[2 * v105 + 1];
              LOBYTE(v105) = a00010203040506_0[2 * v105];
              v102[v98] = v107;
              v108 = v98 - 1;
              v98 -= 2;
              v102[v108] = v105;
            }
            while ( v106 > 0x270F );
            if ( v106 <= 0x3E7 )
              goto LABEL_146;
          }
          v102[1] = a00010203040506_0[2 * v104 + 1];
          *v102 = a00010203040506_0[2 * v104];
LABEL_147:
          if ( v103 <= 9 )
          {
            v145 = v103;
            v169.m128i_i64[0] = (__int64)v170;
            sub_2240A50(&v169, 1, 45, v103, v104);
            v114 = (_BYTE *)v169.m128i_i64[0];
            LOBYTE(v115) = v145;
LABEL_161:
            *v114 = v115 + 48;
            goto LABEL_162;
          }
          if ( v103 <= 0x63 )
          {
            v144 = v103;
            v169.m128i_i64[0] = (__int64)v170;
            sub_2240A50(&v169, 2, 45, v103, v104);
            v114 = (_BYTE *)v169.m128i_i64[0];
            v115 = v144;
          }
          else
          {
            if ( v103 <= 0x3E7 )
            {
              v110 = 2;
              v113 = 3;
            }
            else if ( v103 <= 0x270F )
            {
              v110 = 3;
              v113 = 4;
            }
            else
            {
              v109 = v103;
              v110 = 1;
              while ( 1 )
              {
                v111 = v109;
                v112 = v110;
                v110 += 4;
                v109 /= 0x2710u;
                if ( v111 <= 0x1869F )
                {
                  v113 = v110;
                  v110 = v112 + 3;
                  goto LABEL_157;
                }
                if ( v111 <= 0xF423F )
                {
                  v143 = v103;
                  v169.m128i_i64[0] = (__int64)v170;
                  v113 = v112 + 5;
                  goto LABEL_158;
                }
                if ( v111 <= (unsigned __int64)&loc_98967F )
                  break;
                if ( v111 <= 0x5F5E0FF )
                {
                  v113 = v112 + 7;
                  v110 = v112 + 6;
                  goto LABEL_157;
                }
              }
              v113 = v112 + 6;
              v110 = v112 + 5;
            }
LABEL_157:
            v143 = v103;
            v169.m128i_i64[0] = (__int64)v170;
LABEL_158:
            sub_2240A50(&v169, v113, 45, v103, v104);
            v114 = (_BYTE *)v169.m128i_i64[0];
            v115 = v143;
            do
            {
              v116 = v115
                   - 20
                   * (v115 / 0x64
                    + ((((v115 >> 2) * (unsigned __int128)0x28F5C28F5C28F5C3uLL) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
              v117 = v115;
              v115 /= 0x64u;
              v118 = a00010203040506_0[2 * v116 + 1];
              LOBYTE(v116) = a00010203040506_0[2 * v116];
              v114[v110] = v118;
              v119 = v110 - 1;
              v110 -= 2;
              v114[v119] = v116;
            }
            while ( v117 > 0x270F );
            if ( v117 <= 0x3E7 )
              goto LABEL_161;
          }
          v114[1] = a00010203040506_0[2 * v115 + 1];
          *v114 = a00010203040506_0[2 * v115];
LABEL_162:
          v120 = (__m128i *)sub_2241130(&v169, 0, 0, "illegal NOP size ", 17);
          v171.m128i_i64[0] = (__int64)&v172;
          if ( (__m128i *)v120->m128i_i64[0] == &v120[1] )
          {
            v172 = _mm_loadu_si128(v120 + 1);
          }
          else
          {
            v171.m128i_i64[0] = v120->m128i_i64[0];
            v172.m128i_i64[0] = v120[1].m128i_i64[0];
          }
          v121 = v120->m128i_i64[1];
          v171.m128i_i64[1] = v121;
          v120->m128i_i64[0] = (__int64)v120[1].m128i_i64;
          v120->m128i_i64[1] = 0;
          v120[1].m128i_i8[0] = 0;
          if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v171.m128i_i64[1]) <= 0x16 )
LABEL_222:
            sub_4262D8((__int64)"basic_string::append");
          v122 = (__m128i *)sub_2241490(&v171, ". (expected within [0, ", 23, v121);
          v175.m128i_i64[0] = (__int64)&v176;
          if ( (__m128i *)v122->m128i_i64[0] == &v122[1] )
          {
            v176 = _mm_loadu_si128(v122 + 1);
          }
          else
          {
            v175.m128i_i64[0] = v122->m128i_i64[0];
            v176.m128i_i64[0] = v122[1].m128i_i64[0];
          }
          v175.m128i_i64[1] = v122->m128i_i64[1];
          v122->m128i_i64[0] = (__int64)v122[1].m128i_i64;
          v122->m128i_i64[1] = 0;
          v122[1].m128i_i8[0] = 0;
          v123 = 15;
          v124 = 15;
          if ( (__m128i *)v175.m128i_i64[0] != &v176 )
            v124 = v176.m128i_i64[0];
          v125 = v175.m128i_i64[1] + v178.m128i_i64[1];
          if ( v175.m128i_i64[1] + v178.m128i_i64[1] <= v124 )
            goto LABEL_173;
          if ( (_QWORD *)v178.m128i_i64[0] != v179 )
            v123 = v179[0];
          if ( v125 <= v123 )
          {
            v126 = (__m128i *)sub_2241130(&v178, 0, 0, v175.m128i_i64[0], v175.m128i_i64[1]);
            v181.m128i_i64[0] = (__int64)&v182;
            v127 = v126->m128i_i64[0];
            v128 = v126 + 1;
            if ( (__m128i *)v126->m128i_i64[0] != &v126[1] )
            {
LABEL_174:
              v181.m128i_i64[0] = v127;
              v182.m128i_i64[0] = v126[1].m128i_i64[0];
              goto LABEL_175;
            }
          }
          else
          {
LABEL_173:
            v126 = (__m128i *)sub_2241490(&v175, v178.m128i_i64[0], v178.m128i_i64[1], v125);
            v181.m128i_i64[0] = (__int64)&v182;
            v127 = v126->m128i_i64[0];
            v128 = v126 + 1;
            if ( (__m128i *)v126->m128i_i64[0] != &v126[1] )
              goto LABEL_174;
          }
          v182 = _mm_loadu_si128(v126 + 1);
LABEL_175:
          v129 = v126->m128i_i64[1];
          v181.m128i_i64[1] = v129;
          v126->m128i_i64[0] = (__int64)v128;
          v126->m128i_i64[1] = 0;
          v126[1].m128i_i8[0] = 0;
          if ( v181.m128i_i64[1] == 0x3FFFFFFFFFFFFFFFLL || v181.m128i_i64[1] == 4611686018427387902LL )
            goto LABEL_222;
          v130 = (__m128i *)sub_2241490(&v181, "])", 2, v129);
          v184.m128i_i64[0] = (__int64)&v185;
          if ( (__m128i *)v130->m128i_i64[0] == &v130[1] )
          {
            v185 = _mm_loadu_si128(v130 + 1);
          }
          else
          {
            v184.m128i_i64[0] = v130->m128i_i64[0];
            v185.m128i_i64[0] = v130[1].m128i_i64[0];
          }
          v184.m128i_i64[1] = v130->m128i_i64[1];
          v130->m128i_i64[0] = (__int64)v130[1].m128i_i64;
          v130->m128i_i64[1] = 0;
          v130[1].m128i_i8[0] = 0;
          LOWORD(v191) = 260;
          v189.m128i_i64[0] = (__int64)&v184;
          result = sub_E66880(v155, *(_QWORD *)(v30 + 48), &v189);
          if ( (__m128i *)v184.m128i_i64[0] != &v185 )
            result = j_j___libc_free_0(v184.m128i_i64[0], v185.m128i_i64[0] + 1);
          if ( (__m128i *)v181.m128i_i64[0] != &v182 )
            result = j_j___libc_free_0(v181.m128i_i64[0], v182.m128i_i64[0] + 1);
          if ( (__m128i *)v175.m128i_i64[0] != &v176 )
            result = j_j___libc_free_0(v175.m128i_i64[0], v176.m128i_i64[0] + 1);
          if ( (__m128i *)v171.m128i_i64[0] != &v172 )
            result = j_j___libc_free_0(v171.m128i_i64[0], v172.m128i_i64[0] + 1);
          if ( (_QWORD *)v169.m128i_i64[0] != v170 )
            result = j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
          if ( (_QWORD *)v178.m128i_i64[0] != v179 )
            result = j_j___libc_free_0(v178.m128i_i64[0], v179[0] + 1LL);
          v54 = v161;
LABEL_66:
          if ( !v54 )
            v54 = v161;
          if ( v53 )
          {
            v55 = v54;
            do
            {
              v56 = v53;
              v57 = a1[1];
              if ( v55 <= v53 )
                v56 = v55;
              v171.m128i_i64[0] = v56;
              result = (*(__int64 (__fastcall **)(__int64, _QWORD *, __int64, _QWORD))(*(_QWORD *)v57 + 192LL))(
                         v57,
                         a2,
                         v56,
                         *(_QWORD *)(v30 + 56));
              if ( !(_BYTE)result )
              {
                v187 = 1;
                v184.m128i_i64[0] = (__int64)" bytes";
                v180 = 267;
                v178.m128i_i64[0] = (__int64)&v171;
                v175.m128i_i64[0] = (__int64)"unable to write nop sequence of the remaining ";
                v186 = 3;
                LOWORD(v177) = 259;
                sub_9C6370(&v181, &v175, &v178, v58, v59, v60);
                sub_9C6370(&v189, &v181, &v184, v61, v62, v63);
                sub_C64D30((__int64)&v189, 1u);
              }
              v53 -= v171.m128i_i64[0];
            }
            while ( v53 );
          }
LABEL_48:
          v30 = *(_QWORD *)v30;
          if ( !v30 )
            return result;
          break;
        case 5:
          v49 = v162;
          if ( v162 )
          {
            v50 = 0;
            do
            {
              v51 = *(_BYTE *)(v30 + 30);
              result = a2[4];
              if ( result < a2[3] )
              {
                a2[4] = result + 1;
                *(_BYTE *)result = v51;
              }
              else
              {
                result = sub_CB5D20((__int64)a2, v51);
              }
              ++v50;
            }
            while ( v49 != v50 );
          }
          goto LABEL_48;
        case 9:
          result = (*(__int64 (__fastcall **)(__int64, _QWORD *, unsigned __int64, _QWORD))(*(_QWORD *)a1[1] + 192LL))(
                     a1[1],
                     a2,
                     v162,
                     *(_QWORD *)(v30 + 48));
          if ( !(_BYTE)result )
          {
            v187 = 1;
            v184.m128i_i64[0] = (__int64)" bytes";
            v178.m128i_i64[0] = (__int64)&v162;
            v186 = 3;
            v175.m128i_i64[0] = (__int64)"unable to write nop sequence of ";
            v180 = 267;
            LOWORD(v177) = 259;
            sub_9C6370(&v181, &v175, &v178, v40, v41, v42);
            sub_9C6370(&v189, &v181, &v184, v43, v44, v45);
            sub_C64D30((__int64)&v189, 1u);
          }
          goto LABEL_48;
        case 0xA:
          v46 = *(_DWORD *)(*(_QWORD *)(v30 + 32) + 16LL);
          v47 = _byteswap_ulong(v46);
          if ( v36 != 1 )
            v46 = v47;
          v48 = 4;
          v189.m128i_i32[0] = v46;
          goto LABEL_55;
        case 0xB:
          result = sub_CB6200((__int64)a2, *(unsigned __int8 **)(v30 + 64), *(_QWORD *)(v30 + 72));
          goto LABEL_48;
        case 0xE:
          goto LABEL_225;
        default:
          result = v39;
          goto LABEL_48;
      }
    }
  }
  return result;
}
