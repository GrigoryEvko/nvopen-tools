// Function: sub_387CB50
// Address: 0x387cb50
//
__int64 __fastcall sub_387CB50(
        __int64 *a1,
        __int64 a2,
        __m128 a3,
        __m128i a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10,
        __int64 a11,
        __int64 a12,
        __int64 a13)
{
  __int64 *v13; // r15
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r8
  int v17; // r9d
  __int64 v18; // rbx
  __int64 v19; // rax
  __int64 v20; // r12
  __int64 v21; // rdx
  __int64 **v22; // r13
  __int64 v23; // rbx
  __int64 **v24; // r12
  unsigned __int64 v25; // rax
  __int64 **v26; // r15
  __int64 *v27; // r12
  __int64 **i; // r13
  __int64 *v29; // r14
  char v30; // al
  unsigned int v31; // ebx
  __int64 *v32; // r12
  __int64 v33; // rbx
  __int64 v34; // rcx
  __int64 *v35; // rdx
  __int64 v36; // rdi
  __int64 v37; // rax
  _QWORD *v38; // r13
  double v39; // xmm4_8
  double v40; // xmm5_8
  __int64 v41; // rax
  _QWORD *v42; // rax
  unsigned int v43; // eax
  _QWORD *v44; // rdi
  unsigned __int64 v45; // rdi
  char v47; // al
  __int64 v48; // rax
  __int64 v49; // rcx
  double v50; // xmm4_8
  double v51; // xmm5_8
  __int64 v52; // r9
  __int64 v53; // rax
  char v54; // r10
  unsigned int v55; // edi
  __int64 v56; // rdx
  __int64 v57; // rax
  __int64 v58; // rsi
  __int64 v59; // rsi
  __int64 v60; // rsi
  __int64 v61; // rax
  char v62; // r10
  __int64 v63; // rax
  unsigned int v64; // edi
  __int64 v65; // rdx
  __int64 v66; // rax
  __int64 v67; // rsi
  __int64 v68; // rcx
  __int64 v69; // rax
  __int64 v70; // rsi
  _QWORD *v71; // rax
  __int64 v72; // r13
  __int64 v73; // rax
  unsigned __int8 *v74; // rsi
  __int64 v75; // rsi
  _BYTE *v76; // rax
  __int64 **v77; // rsi
  unsigned int v78; // eax
  _QWORD *v79; // rdi
  int v80; // esi
  int v81; // eax
  _QWORD *v82; // r13
  __int64 v83; // rax
  char v84; // al
  unsigned __int8 **v85; // rdx
  int v86; // esi
  int v87; // eax
  unsigned __int8 *v88; // rax
  _QWORD *v89; // rdi
  __int64 v90; // r9
  __int64 *v91; // r13
  __int64 v92; // rcx
  __int64 v93; // rax
  __int64 v94; // rax
  __int64 v95; // rax
  __int64 v96; // rdi
  __int64 v97; // rax
  __int64 v98; // rax
  unsigned __int8 *v99; // rsi
  __int64 v100; // rsi
  unsigned __int8 *v101; // rax
  _QWORD *v102; // rdi
  __int64 v103; // rax
  __int64 v104; // [rsp+0h] [rbp-1A0h]
  __int64 v105; // [rsp+8h] [rbp-198h]
  __int64 v106; // [rsp+8h] [rbp-198h]
  _QWORD *v107; // [rsp+10h] [rbp-190h]
  __int64 v108; // [rsp+10h] [rbp-190h]
  __int64 v109; // [rsp+10h] [rbp-190h]
  __int64 v110; // [rsp+18h] [rbp-188h]
  __int64 v111; // [rsp+20h] [rbp-180h]
  __int64 v112; // [rsp+28h] [rbp-178h]
  _QWORD *v115; // [rsp+48h] [rbp-158h]
  __int64 v116; // [rsp+48h] [rbp-158h]
  __int64 v117; // [rsp+48h] [rbp-158h]
  __int64 v118; // [rsp+48h] [rbp-158h]
  __int64 v119; // [rsp+48h] [rbp-158h]
  __int64 **v120; // [rsp+58h] [rbp-148h]
  unsigned int v121; // [rsp+58h] [rbp-148h]
  __int64 *v123; // [rsp+68h] [rbp-138h]
  __int64 v124; // [rsp+68h] [rbp-138h]
  __int64 v125[2]; // [rsp+70h] [rbp-130h] BYREF
  __int16 v126; // [rsp+80h] [rbp-120h]
  unsigned __int8 *v127[2]; // [rsp+90h] [rbp-110h] BYREF
  __int16 v128; // [rsp+A0h] [rbp-100h]
  __int64 v129; // [rsp+B0h] [rbp-F0h] BYREF
  unsigned __int64 v130; // [rsp+B8h] [rbp-E8h]
  __int64 v131; // [rsp+C0h] [rbp-E0h]
  unsigned int v132; // [rsp+C8h] [rbp-D8h]
  __m128i v133; // [rsp+D0h] [rbp-D0h] BYREF
  __int64 *v134; // [rsp+E0h] [rbp-C0h]
  __int64 v135; // [rsp+E8h] [rbp-B8h]
  __int64 v136; // [rsp+F0h] [rbp-B0h]
  int v137; // [rsp+F8h] [rbp-A8h]
  __int64 v138; // [rsp+100h] [rbp-A0h]
  __int64 v139; // [rsp+108h] [rbp-98h]
  void *src; // [rsp+120h] [rbp-80h] BYREF
  __int64 v141; // [rsp+128h] [rbp-78h]
  _BYTE v142[112]; // [rsp+130h] [rbp-70h] BYREF

  v13 = a1;
  src = v142;
  v141 = 0x800000000LL;
  v14 = sub_157F280(**(_QWORD **)(a2 + 32));
  if ( v15 == v14 )
  {
    v19 = (unsigned int)v141;
  }
  else
  {
    v18 = v14;
    v19 = (unsigned int)v141;
    v20 = v15;
    do
    {
      if ( (unsigned int)v19 >= HIDWORD(v141) )
      {
        sub_16CD150((__int64)&src, v142, 0, 8, v16, v17);
        v19 = (unsigned int)v141;
      }
      *((_QWORD *)src + v19) = v18;
      v19 = (unsigned int)(v141 + 1);
      LODWORD(v141) = v141 + 1;
      if ( !v18 )
        BUG();
      v21 = *(_QWORD *)(v18 + 32);
      if ( !v21 )
        BUG();
      v18 = 0;
      if ( *(_BYTE *)(v21 - 8) == 77 )
        v18 = v21 - 24;
    }
    while ( v20 != v18 );
  }
  v22 = (__int64 **)src;
  v23 = 8 * v19;
  if ( !a13 )
  {
    v123 = (__int64 *)((char *)src + v23);
    goto LABEL_23;
  }
  v24 = (__int64 **)((char *)src + v23);
  v123 = (__int64 *)src;
  if ( (char *)src + v23 == src )
    goto LABEL_23;
  _BitScanReverse64(&v25, v23 >> 3);
  sub_386FE40((__int64)src, (__int64 **)((char *)src + v23), 2LL * (int)(63 - (v25 ^ 0x3F)));
  if ( (unsigned __int64)v23 <= 0x80 )
  {
    sub_386F0B0(v22, &v22[(unsigned __int64)v23 / 8]);
    goto LABEL_118;
  }
  sub_386F0B0(v22, v22 + 16);
  if ( v24 == v22 + 16 )
    goto LABEL_118;
  v26 = v22 + 16;
  v120 = v24;
  while ( 2 )
  {
    v27 = *v26;
    for ( i = v26; ; --i )
    {
      v29 = *(i - 1);
      v16 = *v27;
      v30 = *(_BYTE *)(*v29 + 8);
      if ( *(_BYTE *)(*v27 + 8) == 11 )
        break;
      if ( v30 != 11 )
        goto LABEL_16;
LABEL_21:
      *i = v29;
    }
    if ( v30 == 11 )
    {
      v124 = *v27;
      v31 = sub_1643030(*v29);
      if ( v31 < (unsigned int)sub_1643030(v124) )
        goto LABEL_21;
    }
LABEL_16:
    *i = v27;
    if ( v120 != ++v26 )
      continue;
    break;
  }
  v13 = a1;
LABEL_118:
  v22 = (__int64 **)src;
  v123 = (__int64 *)((char *)src + 8 * (unsigned int)v141);
LABEL_23:
  v129 = 0;
  v130 = 0;
  v131 = 0;
  v132 = 0;
  if ( v22 != (__int64 **)v123 )
  {
    v121 = 0;
    v32 = (__int64 *)v22;
    while ( 1 )
    {
      v33 = *v32;
      v34 = *(_QWORD *)(*v13 + 40);
      v35 = *(__int64 **)(*v13 + 56);
      v36 = *v32;
      v37 = *(_QWORD *)(*v13 + 48);
      v133.m128i_i64[0] = v13[1];
      v133.m128i_i64[1] = v34;
      v134 = v35;
      v135 = v37;
      v136 = 0;
      v38 = (_QWORD *)sub_13E3350(v36, &v133, 0, 1, v16);
      if ( v38 )
      {
        if ( *(_QWORD *)v33 != *v38 )
          goto LABEL_26;
LABEL_32:
        sub_164D160(v33, (__int64)v38, a3, *(double *)a4.m128i_i64, a5, a6, v39, v40, a9, a10);
        v43 = *(_DWORD *)(a12 + 8);
        if ( v43 >= *(_DWORD *)(a12 + 12) )
        {
          sub_170B450(a12, 0);
          v43 = *(_DWORD *)(a12 + 8);
        }
        v44 = (_QWORD *)(*(_QWORD *)a12 + 24LL * v43);
        if ( v44 )
        {
          *v44 = 6;
          v44[1] = 0;
          v44[2] = v33;
          if ( v33 != -8 && v33 != -16 )
            sub_164C220((__int64)v44);
          v43 = *(_DWORD *)(a12 + 8);
        }
        ++v121;
        ++v32;
        *(_DWORD *)(a12 + 8) = v43 + 1;
        if ( v123 == v32 )
        {
LABEL_36:
          v45 = v130;
          goto LABEL_37;
        }
      }
      else
      {
        if ( sub_1456C80(*v13, *(_QWORD *)v33)
          && (v41 = sub_146F1B0(*v13, v33), !*(_WORD *)(v41 + 24))
          && (v42 = *(_QWORD **)(v41 + 32)) != 0 )
        {
          v38 = v42;
          if ( *(_QWORD *)v33 == *v42 )
            goto LABEL_32;
        }
        else if ( sub_1456C80(*v13, *(_QWORD *)v33) )
        {
          v127[0] = (unsigned __int8 *)sub_146F1B0(*v13, v33);
          v47 = sub_3873070((__int64)&v129, (__int64 *)v127, &v133);
          v115 = (_QWORD *)v133.m128i_i64[0];
          if ( !v47 )
          {
            v80 = v132;
            ++v129;
            v81 = v131 + 1;
            if ( 4 * ((int)v131 + 1) >= 3 * v132 )
            {
              v80 = 2 * v132;
            }
            else if ( v132 - HIDWORD(v131) - v81 > v132 >> 3 )
            {
LABEL_100:
              LODWORD(v131) = v81;
              if ( *v115 != -8 )
                --HIDWORD(v131);
              *v115 = v127[0];
              v115[1] = 0;
LABEL_103:
              v115[1] = v33;
              if ( *(_BYTE *)(*(_QWORD *)v33 + 8LL) != 11 || !a13 || !(unsigned __int8)sub_14A2CF0(a13) )
                goto LABEL_26;
              v82 = (_QWORD *)*v13;
              v117 = **((_QWORD **)src + (unsigned int)v141 - 1);
              v83 = sub_146F1B0(*v13, v33);
              v127[0] = (unsigned __int8 *)sub_14835F0(v82, v83, v117, 0, (__m128i)a3, a4);
              v84 = sub_3873070((__int64)&v129, (__int64 *)v127, &v133);
              v85 = (unsigned __int8 **)v133.m128i_i64[0];
              if ( !v84 )
              {
                v86 = v132;
                ++v129;
                v87 = v131 + 1;
                if ( 4 * ((int)v131 + 1) >= 3 * v132 )
                {
                  v86 = 2 * v132;
                }
                else if ( v132 - HIDWORD(v131) - v87 > v132 >> 3 )
                {
                  goto LABEL_109;
                }
                sub_387C990((__int64)&v129, v86);
                sub_3873070((__int64)&v129, (__int64 *)v127, &v133);
                v85 = (unsigned __int8 **)v133.m128i_i64[0];
                v87 = v131 + 1;
LABEL_109:
                LODWORD(v131) = v87;
                if ( *v85 != (unsigned __int8 *)-8LL )
                  --HIDWORD(v131);
                v88 = v127[0];
                v85[1] = 0;
                *v85 = v88;
              }
              v85[1] = (unsigned __int8 *)v33;
              goto LABEL_26;
            }
            sub_387C990((__int64)&v129, v80);
            sub_3873070((__int64)&v129, (__int64 *)v127, &v133);
            v115 = (_QWORD *)v133.m128i_i64[0];
            v81 = v131 + 1;
            goto LABEL_100;
          }
          v48 = *(_QWORD *)(v133.m128i_i64[0] + 8);
          if ( !v48 )
            goto LABEL_103;
          if ( (*(_BYTE *)(*(_QWORD *)v33 + 8LL) == 15) == (*(_BYTE *)(*(_QWORD *)v48 + 8LL) == 15) )
          {
            v110 = v33;
            v49 = sub_13FCB50(a2);
            v52 = v115[1];
            if ( v49 )
            {
              v53 = 0x17FFFFFFE8LL;
              v54 = *(_BYTE *)(v52 + 23) & 0x40;
              v55 = *(_DWORD *)(v52 + 20) & 0xFFFFFFF;
              if ( v55 )
              {
                v56 = 24LL * *(unsigned int *)(v52 + 56) + 8;
                v57 = 0;
                do
                {
                  v58 = v52 - 24LL * v55;
                  if ( v54 )
                    v58 = *(_QWORD *)(v52 - 8);
                  if ( v49 == *(_QWORD *)(v58 + v56) )
                  {
                    v53 = 24 * v57;
                    goto LABEL_52;
                  }
                  ++v57;
                  v56 += 8;
                }
                while ( v55 != (_DWORD)v57 );
                v53 = 0x17FFFFFFE8LL;
              }
LABEL_52:
              if ( v54 )
                v59 = *(_QWORD *)(v52 - 8);
              else
                v59 = v52 - 24LL * v55;
              v60 = *(_QWORD *)(v59 + v53);
              v61 = 0;
              if ( *(_BYTE *)(v60 + 16) > 0x17u )
                v61 = v60;
              v62 = *(_BYTE *)(v33 + 23) & 0x40;
              v111 = v61;
              v63 = 0x17FFFFFFE8LL;
              v64 = *(_DWORD *)(v33 + 20) & 0xFFFFFFF;
              if ( v64 )
              {
                v65 = 24LL * *(unsigned int *)(v33 + 56) + 8;
                v66 = 0;
                do
                {
                  v67 = v33 - 24LL * v64;
                  if ( v62 )
                    v67 = *(_QWORD *)(v33 - 8);
                  if ( v49 == *(_QWORD *)(v67 + v65) )
                  {
                    v63 = 24 * v66;
                    goto LABEL_63;
                  }
                  ++v66;
                  v65 += 8;
                }
                while ( v64 != (_DWORD)v66 );
                v63 = 0x17FFFFFFE8LL;
              }
LABEL_63:
              if ( v62 )
                v68 = *(_QWORD *)(v33 - 8);
              else
                v68 = v33 - 24LL * v64;
              v112 = *(_QWORD *)(v68 + v63);
              if ( !v112 )
                BUG();
              if ( *(_BYTE *)(v112 + 16) > 0x17u && v111 )
              {
                if ( *(_QWORD *)v52 == *(_QWORD *)v33 )
                {
                  v125[0] = v33;
                  if ( !(unsigned __int8)sub_3872F10((__int64)(v13 + 28), v125, &v133)
                    && !(unsigned __int8)sub_3870860((__int64)v13, v115[1], (__int64 *)v111, a2) )
                  {
                    v127[0] = (unsigned __int8 *)v33;
                    if ( (unsigned __int8)sub_3872F10((__int64)(v13 + 28), (__int64 *)v127, &v133)
                      || (unsigned __int8)sub_3870860((__int64)v13, v33, (__int64 *)v112, a2) )
                    {
                      v94 = v115[1];
                      v115[1] = v33;
                      v110 = v94;
                      v33 = v94;
                      v95 = v111;
                      v111 = v112;
                      v112 = v95;
                    }
                  }
                }
                v107 = (_QWORD *)*v13;
                v105 = *(_QWORD *)v112;
                v69 = sub_146F1B0(*v13, v111);
                v108 = sub_1483C80(v107, v69, v105, (__m128i)a3, a4);
                if ( v111 != v112 && v108 == sub_146F1B0(*v13, v112) )
                {
                  if ( *(_BYTE *)(v111 + 16) <= 0x17u
                    || (v70 = *(_QWORD *)(v111 + 40), v104 = *(_QWORD *)(v112 + 40), v70 == v104)
                    || (v106 = *(_QWORD *)(*v13 + 64), (v109 = sub_13AE450(v106, v70)) == 0)
                    || (v71 = (_QWORD *)sub_13AE450(v106, v104), (_QWORD *)v109 == v71) )
                  {
LABEL_145:
                    if ( (unsigned __int8)sub_3871F10(v13, v111, v112) )
                    {
                      if ( *(_QWORD *)v112 != *(_QWORD *)v111 )
                      {
                        v96 = *(_QWORD *)(v111 + 40);
                        if ( *(_BYTE *)(v111 + 16) == 77 )
                        {
                          v97 = sub_157EE30(v96);
                          v38 = (_QWORD *)v97;
                          if ( v97 )
                            v38 = (_QWORD *)(v97 - 24);
                        }
                        else
                        {
                          v103 = *(_QWORD *)(v111 + 32);
                          if ( v103 != v96 + 40 && v103 )
                            v38 = (_QWORD *)(v103 - 24);
                        }
                        v98 = sub_16498A0((__int64)v38);
                        v133 = 0u;
                        v135 = v98;
                        v134 = 0;
                        v136 = 0;
                        v137 = 0;
                        v138 = 0;
                        v139 = 0;
                        sub_17050D0(v133.m128i_i64, (__int64)v38);
                        v99 = *(unsigned __int8 **)(v112 + 48);
                        v127[0] = v99;
                        if ( v99 )
                        {
                          sub_1623A60((__int64)v127, (__int64)v99, 2);
                          v100 = v133.m128i_i64[0];
                          if ( v133.m128i_i64[0] )
                            goto LABEL_152;
LABEL_153:
                          v133.m128i_i64[0] = (__int64)v127[0];
                          if ( v127[0] )
                          {
                            sub_1623210((__int64)v127, v127[0], (__int64)&v133);
                            v127[0] = 0;
                          }
                        }
                        else
                        {
                          v100 = v133.m128i_i64[0];
                          if ( v133.m128i_i64[0] )
                          {
LABEL_152:
                            sub_161E7C0((__int64)&v133, v100);
                            goto LABEL_153;
                          }
                        }
                        sub_17CD270((__int64 *)v127);
                        v101 = (unsigned __int8 *)v13[2];
                        v128 = 257;
                        if ( *v101 )
                        {
                          v127[0] = v101;
                          LOBYTE(v128) = 3;
                        }
                        v111 = sub_3871DE0(v133.m128i_i64, v111, *(__int64 ***)v112, (__int64 *)v127);
                        sub_17CD270(v133.m128i_i64);
                      }
                      sub_164D160(v112, v111, a3, *(double *)a4.m128i_i64, a5, a6, v50, v51, a9, a10);
                      if ( *(_DWORD *)(a12 + 8) >= *(_DWORD *)(a12 + 12) )
                        sub_170B450(a12, 0);
                      v102 = (_QWORD *)(*(_QWORD *)a12 + 24LL * *(unsigned int *)(a12 + 8));
                      if ( v102 )
                      {
                        *v102 = 6;
                        v102[1] = 0;
                        v102[2] = v112;
                        if ( v112 != -16 && v112 != -8 )
                          sub_164C220((__int64)v102);
                      }
                      ++*(_DWORD *)(a12 + 8);
                    }
                  }
                  else
                  {
                    while ( v71 )
                    {
                      v71 = (_QWORD *)*v71;
                      if ( (_QWORD *)v109 == v71 )
                        goto LABEL_145;
                    }
                  }
                }
                v52 = v115[1];
              }
            }
            ++v121;
            if ( *(_QWORD *)v33 != *(_QWORD *)v52 )
            {
              v72 = sub_157EE30(**(_QWORD **)(a2 + 32));
              if ( v72 )
                v72 -= 24;
              v73 = sub_16498A0(v72);
              v133 = 0u;
              v134 = 0;
              v135 = v73;
              v136 = 0;
              v137 = 0;
              v138 = 0;
              v139 = 0;
              sub_17050D0(v133.m128i_i64, v72);
              v74 = *(unsigned __int8 **)(v33 + 48);
              v127[0] = v74;
              if ( v74 )
              {
                sub_1623A60((__int64)v127, (__int64)v74, 2);
                v75 = v133.m128i_i64[0];
                if ( v133.m128i_i64[0] )
                  goto LABEL_84;
LABEL_85:
                v133.m128i_i64[0] = (__int64)v127[0];
                if ( v127[0] )
                  sub_1623210((__int64)v127, v127[0], (__int64)&v133);
              }
              else
              {
                v75 = v133.m128i_i64[0];
                if ( v133.m128i_i64[0] )
                {
LABEL_84:
                  sub_161E7C0((__int64)&v133, v75);
                  goto LABEL_85;
                }
              }
              v76 = (_BYTE *)v13[2];
              v126 = 257;
              if ( *v76 )
              {
                v125[0] = (__int64)v76;
                LOBYTE(v126) = 3;
              }
              v77 = *(__int64 ***)v33;
              v52 = v115[1];
              if ( *(_QWORD *)v33 != *(_QWORD *)v52 )
              {
                if ( *(_BYTE *)(v52 + 16) > 0x10u )
                {
                  v89 = (_QWORD *)v115[1];
                  v128 = 257;
                  v90 = sub_15FDF30(v89, (__int64)v77, (__int64)v127, 0);
                  if ( v133.m128i_i64[1] )
                  {
                    v91 = v134;
                    v118 = v90;
                    sub_157E9D0(v133.m128i_i64[1] + 40, v90);
                    v90 = v118;
                    v92 = *v91;
                    v93 = *(_QWORD *)(v118 + 24);
                    *(_QWORD *)(v118 + 32) = v91;
                    v92 &= 0xFFFFFFFFFFFFFFF8LL;
                    *(_QWORD *)(v118 + 24) = v92 | v93 & 7;
                    *(_QWORD *)(v92 + 8) = v118 + 24;
                    *v91 = *v91 & 7 | (v118 + 24);
                  }
                  v119 = v90;
                  sub_164B780(v90, v125);
                  sub_12A86E0(v133.m128i_i64, v119);
                  v52 = v119;
                }
                else
                {
                  v52 = sub_15A4670((__int64 ***)v115[1], v77);
                }
              }
              if ( v133.m128i_i64[0] )
              {
                v116 = v52;
                sub_161E7C0((__int64)&v133, v133.m128i_i64[0]);
                v52 = v116;
              }
            }
            sub_164D160(v110, v52, a3, *(double *)a4.m128i_i64, a5, a6, v50, v51, a9, a10);
            v78 = *(_DWORD *)(a12 + 8);
            if ( v78 >= *(_DWORD *)(a12 + 12) )
            {
              sub_170B450(a12, 0);
              v78 = *(_DWORD *)(a12 + 8);
            }
            v79 = (_QWORD *)(*(_QWORD *)a12 + 24LL * v78);
            if ( v79 )
            {
              *v79 = 6;
              v79[1] = 0;
              v79[2] = v33;
              if ( v33 != -16 && v33 != -8 )
                sub_164C220((__int64)v79);
              v78 = *(_DWORD *)(a12 + 8);
            }
            *(_DWORD *)(a12 + 8) = v78 + 1;
          }
        }
LABEL_26:
        if ( v123 == ++v32 )
          goto LABEL_36;
      }
    }
  }
  v121 = 0;
  v45 = 0;
LABEL_37:
  j___libc_free_0(v45);
  if ( src != v142 )
    _libc_free((unsigned __int64)src);
  return v121;
}
