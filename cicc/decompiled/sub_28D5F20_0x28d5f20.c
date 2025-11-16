// Function: sub_28D5F20
// Address: 0x28d5f20
//
unsigned __int64 __fastcall sub_28D5F20(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // rbx
  bool v8; // zf
  _QWORD *v9; // rax
  __int64 *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rbx
  __m128i v13; // rax
  __int64 *v14; // rsi
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rdx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rcx
  const __m128i *v22; // rdi
  unsigned __int64 v23; // rbx
  __int64 v24; // rax
  __int64 v25; // rsi
  __m128i *v26; // rdx
  const __m128i *v27; // rax
  __int64 v28; // r8
  __int64 v29; // r9
  unsigned __int64 v30; // rbx
  __int64 v31; // rax
  __m128i *v32; // rdx
  __m128i *v33; // rcx
  const __m128i *v34; // rax
  __int64 v35; // rax
  __int64 v36; // rsi
  __int64 v37; // rcx
  __int64 v38; // rbx
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rsi
  __int64 *v42; // rax
  __int64 v43; // r14
  __int64 v44; // r12
  unsigned int v45; // esi
  __int64 v46; // rdi
  __int64 v47; // rax
  __int64 v48; // rbx
  unsigned int v49; // esi
  __int64 v50; // r13
  __int64 v51; // rdi
  int v52; // r10d
  __int64 *v53; // rax
  __int64 v54; // r12
  int v55; // eax
  __int64 v56; // r13
  __int64 v57; // rbx
  _QWORD *v58; // rax
  unsigned int v59; // esi
  int v60; // r11d
  _QWORD *v61; // rax
  __int64 v62; // rdi
  _QWORD *v63; // rax
  __int64 v64; // r12
  __int64 v65; // r14
  __int64 *v66; // rax
  __int64 v67; // rcx
  __int64 *v68; // rdx
  __int64 v69; // rbx
  _QWORD *v70; // rax
  char v71; // dl
  __int64 v72; // rbx
  unsigned __int8 *v73; // rsi
  int v74; // edi
  __int64 v75; // rax
  unsigned __int64 v76; // rdi
  __int64 v77; // rax
  __m128i v78; // rax
  __int64 v79; // rcx
  int v80; // eax
  int v81; // r10d
  int v82; // edi
  int v83; // r8d
  __int64 v84; // r11
  int v85; // edi
  _QWORD *v86; // rsi
  int v87; // r8d
  __int64 v88; // r14
  int v89; // esi
  __int64 v90; // rdi
  int v91; // edi
  int v92; // edi
  __int64 v93; // r10
  unsigned int v94; // esi
  int v95; // esi
  int v96; // r10d
  int v97; // edi
  int v98; // edx
  int v99; // eax
  int v100; // eax
  int v101; // esi
  int v102; // edx
  unsigned int v103; // esi
  __int64 v104; // rdx
  int v105; // edi
  int v106; // eax
  __int64 v107; // rdx
  __int64 v108; // rcx
  _BYTE *v109; // rbx
  unsigned __int64 result; // rax
  _BYTE *v111; // r14
  __int64 v112; // rcx
  __int64 v113; // r8
  __int64 *v114; // r10
  int v115; // r11d
  unsigned int v116; // edx
  __int64 *v117; // rax
  __int64 v118; // rdi
  __int64 v119; // r12
  unsigned int v120; // esi
  int v121; // edx
  __int64 v122; // rdx
  __int64 v123; // rcx
  int v124; // edi
  __int64 v125; // r14
  __int64 v127; // [rsp+18h] [rbp-278h]
  __int64 v128; // [rsp+38h] [rbp-258h]
  __int64 v129; // [rsp+38h] [rbp-258h]
  __int64 v130; // [rsp+48h] [rbp-248h] BYREF
  __m128i v131; // [rsp+50h] [rbp-240h] BYREF
  char v132; // [rsp+60h] [rbp-230h]
  __int64 v133[16]; // [rsp+70h] [rbp-220h] BYREF
  _QWORD *v134; // [rsp+F0h] [rbp-1A0h] BYREF
  _BYTE *v135; // [rsp+F8h] [rbp-198h]
  __int64 v136; // [rsp+100h] [rbp-190h]
  int v137; // [rsp+108h] [rbp-188h]
  char v138; // [rsp+10Ch] [rbp-184h]
  _BYTE v139[64]; // [rsp+110h] [rbp-180h] BYREF
  __m128i *v140; // [rsp+150h] [rbp-140h] BYREF
  __m128i *v141; // [rsp+158h] [rbp-138h]
  __int8 *v142; // [rsp+160h] [rbp-130h]
  __m128i v143; // [rsp+170h] [rbp-120h] BYREF
  char v144; // [rsp+180h] [rbp-110h]
  const __m128i *v145; // [rsp+1D0h] [rbp-C0h]
  __int64 v146; // [rsp+1D8h] [rbp-B8h]
  _BYTE v147[96]; // [rsp+1E8h] [rbp-A8h] BYREF
  const __m128i *v148; // [rsp+248h] [rbp-48h]
  __int64 *v149; // [rsp+250h] [rbp-40h]

  *(_DWORD *)(a1 + 1424) = 0;
  v3 = sub_28CC470(a1, 0, 0);
  v4 = *(_QWORD *)(a1 + 32);
  *(_QWORD *)(a1 + 1392) = v3;
  *(_QWORD *)(v3 + 48) = *(_QWORD *)(v4 + 128);
  v5 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 128LL);
  v6 = sub_28CC470(a1, 0, 0);
  *(_QWORD *)(v6 + 48) = v5;
  v7 = v6;
  v127 = a1 + 1920;
  v133[0] = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 128LL);
  v8 = (unsigned __int8)sub_28C75F0(a1 + 1920, v133, &v134) == 0;
  v9 = v134;
  if ( !v8 )
  {
    v10 = v134 + 1;
    goto LABEL_3;
  }
  v101 = *(_DWORD *)(a1 + 1936);
  ++*(_QWORD *)(a1 + 1920);
  v143.m128i_i64[0] = (__int64)v9;
  v102 = v101 + 1;
  v103 = *(_DWORD *)(a1 + 1944);
  if ( 4 * v102 >= 3 * v103 )
  {
    v125 = a1 + 1920;
    sub_28C9B10(v127, 2 * v103);
    goto LABEL_214;
  }
  if ( v103 - *(_DWORD *)(a1 + 1940) - v102 <= v103 >> 3 )
  {
    v125 = a1 + 1920;
    sub_28C9B10(v127, v103);
LABEL_214:
    sub_28C75F0(v125, v133, &v143);
    v102 = *(_DWORD *)(a1 + 1936) + 1;
    v9 = (_QWORD *)v143.m128i_i64[0];
  }
  *(_DWORD *)(a1 + 1936) = v102;
  if ( *v9 != -4096 )
    --*(_DWORD *)(a1 + 1940);
  v104 = v133[0];
  v10 = v9 + 1;
  *v10 = 0;
  *(v10 - 1) = v104;
LABEL_3:
  *v10 = v7;
  v11 = *(_QWORD *)(a1 + 8);
  memset(v133, 0, 0x78u);
  v133[1] = (__int64)&v133[4];
  LODWORD(v133[2]) = 8;
  BYTE4(v133[3]) = 1;
  v12 = *(_QWORD *)(v11 + 96);
  v134 = 0;
  v135 = v139;
  v136 = 8;
  v137 = 0;
  v138 = 1;
  v140 = 0;
  v141 = 0;
  v142 = 0;
  v13.m128i_i64[0] = (__int64)sub_AE6EC0((__int64)&v134, v12);
  if ( v138 )
    v13.m128i_i64[1] = (__int64)&v135[8 * HIDWORD(v136)];
  else
    v13.m128i_i64[1] = (__int64)&v135[8 * (unsigned int)v136];
  v143 = v13;
  sub_254BBF0((__int64)&v143);
  v143.m128i_i64[0] = v12;
  v144 = 0;
  sub_28D5EE0((__int64)&v140, &v143);
  sub_28CD1F0(&v143, &v134, v133);
  sub_28CB010((__int64)&v134);
  sub_28CB010((__int64)v133);
  v14 = &v133[4];
  sub_C8CD80((__int64)v133, (__int64)&v133[4], (__int64)&v143, v15, v16, v17);
  v21 = v146;
  v22 = v145;
  memset(&v133[12], 0, 24);
  v23 = v146 - (_QWORD)v145;
  if ( (const __m128i *)v146 == v145 )
  {
    v23 = 0;
    v25 = 0;
  }
  else
  {
    if ( v23 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_215;
    v24 = sub_22077B0(v146 - (_QWORD)v145);
    v21 = v146;
    v22 = v145;
    v25 = v24;
  }
  v133[12] = v25;
  v133[13] = v25;
  v133[14] = v25 + v23;
  if ( v22 != (const __m128i *)v21 )
  {
    v26 = (__m128i *)v25;
    v27 = v22;
    do
    {
      if ( v26 )
      {
        *v26 = _mm_loadu_si128(v27);
        v19 = v27[1].m128i_i64[0];
        v26[1].m128i_i64[0] = v19;
      }
      v27 = (const __m128i *)((char *)v27 + 24);
      v26 = (__m128i *)((char *)v26 + 24);
    }
    while ( (const __m128i *)v21 != v27 );
    v25 += 8 * ((unsigned __int64)(v21 - 24 - (_QWORD)v22) >> 3) + 24;
  }
  v133[13] = v25;
  sub_C8CD80((__int64)&v134, (__int64)v139, (__int64)v147, v21, v19, v20);
  v14 = v149;
  v22 = v148;
  v140 = 0;
  v141 = 0;
  v142 = 0;
  v30 = (char *)v149 - (char *)v148;
  if ( v149 == (__int64 *)v148 )
  {
    v32 = 0;
    goto LABEL_17;
  }
  if ( v30 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_215:
    sub_4261EA(v22, v14, v18);
  v31 = sub_22077B0((char *)v149 - (char *)v148);
  v14 = v149;
  v22 = v148;
  v32 = (__m128i *)v31;
LABEL_17:
  v140 = v32;
  v33 = v32;
  v141 = v32;
  v142 = &v32->m128i_i8[v30];
  if ( v22 != (const __m128i *)v14 )
  {
    v34 = v22;
    do
    {
      if ( v33 )
      {
        *v33 = _mm_loadu_si128(v34);
        v28 = v34[1].m128i_i64[0];
        v33[1].m128i_i64[0] = v28;
      }
      v34 = (const __m128i *)((char *)v34 + 24);
      v33 = (__m128i *)((char *)v33 + 24);
    }
    while ( v34 != (const __m128i *)v14 );
    v33 = (__m128i *)((char *)v32 + 8 * ((unsigned __int64)((char *)&v34[-2].m128i_u64[1] - (char *)v22) >> 3) + 24);
  }
  v141 = v33;
  v35 = v133[12];
  v36 = v133[13];
  while ( 1 )
  {
    v37 = (char *)v33 - (char *)v32;
    if ( v36 - v35 == v37 )
      break;
LABEL_25:
    v38 = **(_QWORD **)(v36 - 24);
    v39 = *(_QWORD *)(a1 + 32);
    v40 = *(unsigned int *)(v39 + 120);
    v41 = *(_QWORD *)(v39 + 104);
    v128 = v38;
    if ( (_DWORD)v40 )
    {
      v37 = ((_DWORD)v40 - 1) & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
      v42 = (__int64 *)(v41 + 16 * v37);
      v28 = *v42;
      if ( v38 != *v42 )
      {
        v99 = 1;
        while ( v28 != -4096 )
        {
          v29 = (unsigned int)(v99 + 1);
          v37 = ((_DWORD)v40 - 1) & (unsigned int)(v99 + v37);
          v42 = (__int64 *)(v41 + 16LL * (unsigned int)v37);
          v28 = *v42;
          if ( v38 == *v42 )
            goto LABEL_27;
          v99 = v29;
        }
        goto LABEL_41;
      }
LABEL_27:
      v40 = v41 + 16 * v40;
      if ( v42 != (__int64 *)v40 )
      {
        v43 = v42[1];
        if ( v43 )
        {
          v44 = *(_QWORD *)(v43 + 8);
          if ( v44 != v43 )
          {
            while ( 1 )
            {
              v48 = v44 - 48;
              v49 = *(_DWORD *)(a1 + 1944);
              if ( !v44 )
                v48 = 0;
              v50 = *(_QWORD *)(a1 + 1392);
              v130 = v48;
              if ( !v49 )
                break;
              v28 = v49 - 1;
              v51 = *(_QWORD *)(a1 + 1928);
              v52 = 1;
              v29 = 0;
              v40 = (unsigned int)v28 & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
              v53 = (__int64 *)(v51 + 16 * v40);
              v37 = *v53;
              if ( v48 != *v53 )
              {
                while ( v37 != -4096 )
                {
                  if ( v37 == -8192 && !v29 )
                    v29 = (__int64)v53;
                  v40 = (unsigned int)v28 & (v52 + (_DWORD)v40);
                  v53 = (__int64 *)(v51 + 16LL * (unsigned int)v40);
                  v37 = *v53;
                  if ( v48 == *v53 )
                    goto LABEL_38;
                  ++v52;
                }
                v91 = *(_DWORD *)(a1 + 1936);
                if ( v29 )
                  v53 = (__int64 *)v29;
                ++*(_QWORD *)(a1 + 1920);
                v92 = v91 + 1;
                v131.m128i_i64[0] = (__int64)v53;
                if ( 4 * v92 < 3 * v49 )
                {
                  v40 = v48;
                  v37 = v49 - *(_DWORD *)(a1 + 1940) - v92;
                  v28 = v49 >> 3;
                  if ( (unsigned int)v37 <= (unsigned int)v28 )
                  {
                    sub_28C9B10(v127, v49);
                    sub_28C75F0(v127, &v130, &v131);
                    v40 = v130;
                    v92 = *(_DWORD *)(a1 + 1936) + 1;
                    v53 = (__int64 *)v131.m128i_i64[0];
                  }
LABEL_139:
                  *(_DWORD *)(a1 + 1936) = v92;
                  if ( *v53 != -4096 )
                    --*(_DWORD *)(a1 + 1940);
                  *v53 = v40;
                  v53[1] = 0;
                  goto LABEL_38;
                }
LABEL_143:
                sub_28C9B10(v127, 2 * v49);
                v28 = *(unsigned int *)(a1 + 1944);
                if ( (_DWORD)v28 )
                {
                  v40 = v130;
                  v28 = (unsigned int)(v28 - 1);
                  v93 = *(_QWORD *)(a1 + 1928);
                  v94 = v28 & (((unsigned int)v130 >> 9) ^ ((unsigned int)v130 >> 4));
                  v53 = (__int64 *)(v93 + 16LL * v94);
                  v29 = *v53;
                  if ( v130 == *v53 )
                  {
LABEL_145:
                    v95 = *(_DWORD *)(a1 + 1936);
                    v131.m128i_i64[0] = (__int64)v53;
                    v92 = v95 + 1;
                  }
                  else
                  {
                    v105 = 1;
                    v37 = 0;
                    while ( v29 != -4096 )
                    {
                      if ( v29 == -8192 && !v37 )
                        v37 = (__int64)v53;
                      v94 = v28 & (v105 + v94);
                      v53 = (__int64 *)(v93 + 16LL * v94);
                      v29 = *v53;
                      if ( v130 == *v53 )
                        goto LABEL_145;
                      ++v105;
                    }
                    if ( !v37 )
                      v37 = (__int64)v53;
                    v106 = *(_DWORD *)(a1 + 1936);
                    v131.m128i_i64[0] = v37;
                    v92 = v106 + 1;
                    v53 = (__int64 *)v37;
                  }
                }
                else
                {
                  v100 = *(_DWORD *)(a1 + 1936);
                  v40 = v130;
                  v131.m128i_i64[0] = 0;
                  v92 = v100 + 1;
                  v53 = 0;
                }
                goto LABEL_139;
              }
LABEL_38:
              v53[1] = v50;
              if ( *(_BYTE *)v48 == 27 )
              {
                if ( **(_BYTE **)(v48 + 72) != 62 )
                  goto LABEL_33;
                ++*(_DWORD *)(*(_QWORD *)(a1 + 1392) + 176LL);
                v44 = *(_QWORD *)(v44 + 8);
                if ( v43 == v44 )
                  goto LABEL_41;
              }
              else
              {
                sub_AE6EC0(*(_QWORD *)(a1 + 1392) + 128LL, v48);
                v45 = *(_DWORD *)(a1 + 1976);
                v131.m128i_i64[0] = v48;
                v131.m128i_i32[2] = 1;
                if ( !v45 )
                {
                  ++*(_QWORD *)(a1 + 1952);
                  v130 = 0;
                  goto LABEL_156;
                }
                v28 = v45 - 1;
                v46 = *(_QWORD *)(a1 + 1960);
                v40 = (unsigned int)v28 & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
                v47 = v46 + 16 * v40;
                v37 = *(_QWORD *)v47;
                if ( v48 != *(_QWORD *)v47 )
                {
                  v96 = 1;
                  v29 = 0;
                  while ( v37 != -4096 )
                  {
                    if ( v37 == -8192 && !v29 )
                      v29 = v47;
                    v40 = (unsigned int)v28 & (v96 + (_DWORD)v40);
                    v47 = v46 + 16LL * (unsigned int)v40;
                    v37 = *(_QWORD *)v47;
                    if ( v48 == *(_QWORD *)v47 )
                      goto LABEL_33;
                    ++v96;
                  }
                  v97 = *(_DWORD *)(a1 + 1968);
                  if ( v29 )
                    v47 = v29;
                  ++*(_QWORD *)(a1 + 1952);
                  v98 = v97 + 1;
                  v130 = v47;
                  if ( 4 * (v97 + 1) < 3 * v45 )
                  {
                    v37 = v45 - *(_DWORD *)(a1 + 1972) - v98;
                    if ( (unsigned int)v37 > v45 >> 3 )
                    {
LABEL_152:
                      *(_DWORD *)(a1 + 1968) = v98;
                      if ( *(_QWORD *)v47 != -4096 )
                        --*(_DWORD *)(a1 + 1972);
                      *(_QWORD *)v47 = v48;
                      v40 = v131.m128i_u32[2];
                      *(_DWORD *)(v47 + 8) = v131.m128i_i32[2];
                      goto LABEL_33;
                    }
LABEL_157:
                    sub_28C9E70(a1 + 1952, v45);
                    sub_28C7770(a1 + 1952, v131.m128i_i64, &v130);
                    v48 = v131.m128i_i64[0];
                    v98 = *(_DWORD *)(a1 + 1968) + 1;
                    v47 = v130;
                    goto LABEL_152;
                  }
LABEL_156:
                  v45 *= 2;
                  goto LABEL_157;
                }
LABEL_33:
                v44 = *(_QWORD *)(v44 + 8);
                if ( v43 == v44 )
                  goto LABEL_41;
              }
            }
            ++*(_QWORD *)(a1 + 1920);
            v131.m128i_i64[0] = 0;
            goto LABEL_143;
          }
        }
      }
    }
LABEL_41:
    v54 = *(_QWORD *)(v128 + 56);
    v129 = v128 + 48;
    while ( v129 != v54 )
    {
      if ( !v54 )
        BUG();
      v55 = *(unsigned __int8 *)(v54 - 24);
      if ( (_BYTE)v55 == 84 )
      {
        v72 = *(_QWORD *)(v54 - 8);
        if ( !v72 )
          goto LABEL_46;
        do
        {
          v73 = *(unsigned __int8 **)(v72 + 24);
          v74 = *v73;
          if ( (unsigned __int8)v74 > 0x1Cu )
          {
            v37 = *(unsigned int *)(a1 + 2440);
            v28 = *(_QWORD *)(a1 + 2424);
            if ( (_DWORD)v37 )
            {
              v37 = (unsigned int)(v37 - 1);
              v40 = (unsigned int)v37 & (((unsigned int)v73 >> 9) ^ ((unsigned int)v73 >> 4));
              v75 = v28 + 16 * v40;
              v29 = *(_QWORD *)v75;
              if ( v73 == *(unsigned __int8 **)v75 )
              {
LABEL_85:
                if ( *(_DWORD *)(v75 + 8) )
                {
                  if ( (_BYTE)qword_5004548 )
                  {
                    v76 = (unsigned int)(v74 - 42);
                    if ( (unsigned __int8)v76 <= 0x2Cu )
                    {
                      v77 = 0x1300000BFFFFLL;
                      if ( _bittest64(&v77, v76) )
                      {
                        v78.m128i_i64[0] = (__int64)sub_AE6EC0(a1 + 1496, (__int64)v73);
                        if ( *(_BYTE *)(a1 + 1524) )
                          v79 = *(unsigned int *)(a1 + 1516);
                        else
                          v79 = *(unsigned int *)(a1 + 1512);
                        v78.m128i_i64[1] = *(_QWORD *)(a1 + 1504) + 8 * v79;
                        v131 = v78;
                        sub_254BBF0((__int64)&v131);
                      }
                    }
                  }
                }
              }
              else
              {
                v80 = 1;
                while ( v29 != -4096 )
                {
                  v81 = v80 + 1;
                  v40 = (unsigned int)v37 & (v80 + (_DWORD)v40);
                  v75 = v28 + 16LL * (unsigned int)v40;
                  v29 = *(_QWORD *)v75;
                  if ( v73 == *(unsigned __int8 **)v75 )
                    goto LABEL_85;
                  v80 = v81;
                }
              }
            }
          }
          v72 = *(_QWORD *)(v72 + 8);
        }
        while ( v72 );
        v55 = *(unsigned __int8 *)(v54 - 24);
      }
      if ( (unsigned int)(v55 - 30) <= 0xA && *(_BYTE *)(*(_QWORD *)(v54 - 16) + 8LL) == 7 )
        goto LABEL_55;
LABEL_46:
      v56 = *(_QWORD *)(a1 + 1392);
      v57 = v54 - 24;
      if ( !*(_BYTE *)(v56 + 92) )
        goto LABEL_78;
      v58 = *(_QWORD **)(v56 + 72);
      v37 = *(unsigned int *)(v56 + 84);
      v40 = (__int64)&v58[v37];
      if ( v58 == (_QWORD *)v40 )
      {
LABEL_79:
        if ( (unsigned int)v37 >= *(_DWORD *)(v56 + 80) )
        {
LABEL_78:
          sub_C8CC70(v56 + 64, v54 - 24, v40, v37, v28, v29);
          v56 = *(_QWORD *)(a1 + 1392);
          goto LABEL_51;
        }
        *(_DWORD *)(v56 + 84) = v37 + 1;
        *(_QWORD *)v40 = v57;
        ++*(_QWORD *)(v56 + 64);
        v56 = *(_QWORD *)(a1 + 1392);
      }
      else
      {
        while ( v57 != *v58 )
        {
          if ( (_QWORD *)v40 == ++v58 )
            goto LABEL_79;
        }
      }
LABEL_51:
      v59 = *(_DWORD *)(a1 + 1456);
      if ( !v59 )
      {
        ++*(_QWORD *)(a1 + 1432);
        goto LABEL_115;
      }
      v29 = v59 - 1;
      v28 = *(_QWORD *)(a1 + 1440);
      v60 = 1;
      v61 = 0;
      v37 = (unsigned int)v29 & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
      v40 = v28 + 16 * v37;
      v62 = *(_QWORD *)v40;
      if ( v57 != *(_QWORD *)v40 )
      {
        while ( v62 != -4096 )
        {
          if ( !v61 && v62 == -8192 )
            v61 = (_QWORD *)v40;
          v37 = (unsigned int)v29 & (v60 + (_DWORD)v37);
          v40 = v28 + 16LL * (unsigned int)v37;
          v62 = *(_QWORD *)v40;
          if ( v57 == *(_QWORD *)v40 )
            goto LABEL_53;
          ++v60;
        }
        v82 = *(_DWORD *)(a1 + 1448);
        if ( !v61 )
          v61 = (_QWORD *)v40;
        ++*(_QWORD *)(a1 + 1432);
        v37 = (unsigned int)(v82 + 1);
        if ( 4 * (int)v37 < 3 * v59 )
        {
          v40 = v59 - *(_DWORD *)(a1 + 1452) - (unsigned int)v37;
          if ( (unsigned int)v40 <= v59 >> 3 )
          {
            sub_28C9810(a1 + 1432, v59);
            v87 = *(_DWORD *)(a1 + 1456);
            if ( !v87 )
            {
LABEL_227:
              ++*(_DWORD *)(a1 + 1448);
              BUG();
            }
            v28 = (unsigned int)(v87 - 1);
            v29 = *(_QWORD *)(a1 + 1440);
            v40 = 0;
            LODWORD(v88) = v28 & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
            v37 = (unsigned int)(*(_DWORD *)(a1 + 1448) + 1);
            v89 = 1;
            v61 = (_QWORD *)(v29 + 16LL * (unsigned int)v88);
            v90 = *v61;
            if ( v57 != *v61 )
            {
              while ( v90 != -4096 )
              {
                if ( !v40 && v90 == -8192 )
                  v40 = (__int64)v61;
                v88 = (unsigned int)v28 & ((_DWORD)v88 + v89);
                v61 = (_QWORD *)(v29 + 16 * v88);
                v90 = *v61;
                if ( v57 == *v61 )
                  goto LABEL_110;
                ++v89;
              }
              if ( v40 )
                v61 = (_QWORD *)v40;
            }
          }
          goto LABEL_110;
        }
LABEL_115:
        sub_28C9810(a1 + 1432, 2 * v59);
        v83 = *(_DWORD *)(a1 + 1456);
        if ( !v83 )
          goto LABEL_227;
        v28 = (unsigned int)(v83 - 1);
        v29 = *(_QWORD *)(a1 + 1440);
        v40 = (unsigned int)v28 & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
        v37 = (unsigned int)(*(_DWORD *)(a1 + 1448) + 1);
        v61 = (_QWORD *)(v29 + 16 * v40);
        v84 = *v61;
        if ( v57 != *v61 )
        {
          v85 = 1;
          v86 = 0;
          while ( v84 != -4096 )
          {
            if ( v84 == -8192 && !v86 )
              v86 = v61;
            v40 = (unsigned int)v28 & ((_DWORD)v40 + v85);
            v61 = (_QWORD *)(v29 + 16 * v40);
            v84 = *v61;
            if ( v57 == *v61 )
              goto LABEL_110;
            ++v85;
          }
          if ( v86 )
            v61 = v86;
        }
LABEL_110:
        *(_DWORD *)(a1 + 1448) = v37;
        if ( *v61 != -4096 )
          --*(_DWORD *)(a1 + 1452);
        *v61 = v57;
        v63 = v61 + 1;
        *v63 = 0;
        goto LABEL_54;
      }
LABEL_53:
      v63 = (_QWORD *)(v40 + 8);
LABEL_54:
      *v63 = v56;
LABEL_55:
      v54 = *(_QWORD *)(v54 + 8);
    }
    v64 = v133[13];
    do
    {
      v65 = *(_QWORD *)(v64 - 24);
      if ( !*(_BYTE *)(v64 - 8) )
      {
        v66 = *(__int64 **)(v65 + 24);
        *(_BYTE *)(v64 - 8) = 1;
        *(_QWORD *)(v64 - 16) = v66;
        goto LABEL_59;
      }
      while ( 1 )
      {
        v66 = *(__int64 **)(v64 - 16);
LABEL_59:
        v67 = *(unsigned int *)(v65 + 32);
        if ( v66 == (__int64 *)(*(_QWORD *)(v65 + 24) + 8 * v67) )
          break;
        v68 = v66 + 1;
        *(_QWORD *)(v64 - 16) = v66 + 1;
        v69 = *v66;
        if ( !BYTE4(v133[3]) )
          goto LABEL_73;
        v70 = (_QWORD *)v133[1];
        v67 = HIDWORD(v133[2]);
        v68 = (__int64 *)(v133[1] + 8LL * HIDWORD(v133[2]));
        if ( (__int64 *)v133[1] == v68 )
        {
LABEL_76:
          if ( HIDWORD(v133[2]) < LODWORD(v133[2]) )
          {
            ++HIDWORD(v133[2]);
            *v68 = v69;
            ++v133[0];
LABEL_74:
            v131.m128i_i64[0] = v69;
            v132 = 0;
            sub_28D5EE0((__int64)&v133[12], &v131);
            v35 = v133[12];
            v36 = v133[13];
            goto LABEL_75;
          }
LABEL_73:
          sub_C8CC70((__int64)v133, v69, (__int64)v68, v67, v28, v29);
          if ( v71 )
            goto LABEL_74;
        }
        else
        {
          while ( v69 != *v70 )
          {
            if ( v68 == ++v70 )
              goto LABEL_76;
          }
        }
      }
      v133[13] -= 24;
      v35 = v133[12];
      v64 = v133[13];
    }
    while ( v133[13] != v133[12] );
    v36 = v133[12];
LABEL_75:
    v32 = v140;
    v33 = v141;
  }
  while ( v36 != v35 )
  {
    if ( *(_QWORD *)v35 != v32->m128i_i64[0] )
      goto LABEL_25;
    v37 = *(unsigned __int8 *)(v35 + 16);
    if ( (_BYTE)v37 != v32[1].m128i_i8[0] || (_BYTE)v37 && *(_QWORD *)(v35 + 8) != v32->m128i_i64[1] )
      goto LABEL_25;
    v35 += 24;
    v32 = (__m128i *)((char *)v32 + 24);
  }
  sub_28CB010((__int64)&v134);
  sub_28CB010((__int64)v133);
  sub_28CB010((__int64)v147);
  sub_28CB010((__int64)&v143);
  if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
  {
    sub_B2C6D0(a2, v36, v107, v108);
    v109 = *(_BYTE **)(a2 + 96);
    result = 5LL * *(_QWORD *)(a2 + 104);
    v111 = &v109[40 * *(_QWORD *)(a2 + 104)];
    if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
    {
      result = sub_B2C6D0(a2, v36, v122, v123);
      v109 = *(_BYTE **)(a2 + 96);
    }
  }
  else
  {
    v109 = *(_BYTE **)(a2 + 96);
    result = 5LL * *(_QWORD *)(a2 + 104);
    v111 = &v109[40 * *(_QWORD *)(a2 + 104)];
  }
  if ( v111 != v109 )
  {
    while ( 1 )
    {
      v134 = v109;
      v119 = sub_28CC470(a1, v109, 0);
      sub_AE6EC0(v119 + 64, (__int64)v134);
      v120 = *(_DWORD *)(a1 + 1456);
      if ( !v120 )
        break;
      v112 = (__int64)v134;
      v113 = *(_QWORD *)(a1 + 1440);
      v114 = 0;
      v115 = 1;
      v116 = (v120 - 1) & (((unsigned int)v134 >> 9) ^ ((unsigned int)v134 >> 4));
      v117 = (__int64 *)(v113 + 16LL * v116);
      v118 = *v117;
      if ( (_QWORD *)*v117 == v134 )
      {
LABEL_186:
        v109 += 40;
        result = (unsigned __int64)(v117 + 1);
        *(_QWORD *)result = v119;
        if ( v111 == v109 )
          return result;
      }
      else
      {
        while ( v118 != -4096 )
        {
          if ( !v114 && v118 == -8192 )
            v114 = v117;
          v116 = (v120 - 1) & (v115 + v116);
          v117 = (__int64 *)(v113 + 16LL * v116);
          v118 = *v117;
          if ( v134 == (_QWORD *)*v117 )
            goto LABEL_186;
          ++v115;
        }
        v124 = *(_DWORD *)(a1 + 1448);
        if ( v114 )
          v117 = v114;
        ++*(_QWORD *)(a1 + 1432);
        v121 = v124 + 1;
        v143.m128i_i64[0] = (__int64)v117;
        if ( 4 * (v124 + 1) < 3 * v120 )
        {
          if ( v120 - *(_DWORD *)(a1 + 1452) - v121 > v120 >> 3 )
            goto LABEL_191;
          goto LABEL_190;
        }
LABEL_189:
        v120 *= 2;
LABEL_190:
        sub_28C9810(a1 + 1432, v120);
        sub_28C74C0(a1 + 1432, (__int64 *)&v134, &v143);
        v112 = (__int64)v134;
        v121 = *(_DWORD *)(a1 + 1448) + 1;
        v117 = (__int64 *)v143.m128i_i64[0];
LABEL_191:
        *(_DWORD *)(a1 + 1448) = v121;
        if ( *v117 != -4096 )
          --*(_DWORD *)(a1 + 1452);
        *v117 = v112;
        v109 += 40;
        result = (unsigned __int64)(v117 + 1);
        *(_QWORD *)result = 0;
        *(_QWORD *)result = v119;
        if ( v111 == v109 )
          return result;
      }
    }
    ++*(_QWORD *)(a1 + 1432);
    v143.m128i_i64[0] = 0;
    goto LABEL_189;
  }
  return result;
}
