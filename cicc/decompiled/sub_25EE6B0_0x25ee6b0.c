// Function: sub_25EE6B0
// Address: 0x25ee6b0
//
__int64 __fastcall sub_25EE6B0(__int64 a1)
{
  unsigned int v1; // r13d
  __int64 v2; // r15
  __int64 v3; // rax
  __int64 v4; // rsi
  __int64 v5; // r15
  __int64 v6; // rbx
  _BYTE *v7; // rbx
  unsigned __int64 v8; // r12
  unsigned __int64 v9; // rdi
  char v11; // al
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rsi
  unsigned int v14; // r14d
  __int64 v15; // rax
  __int64 v16; // rdx
  unsigned __int64 v17; // rax
  __int64 v18; // rax
  _QWORD *v19; // r15
  _QWORD *v20; // rax
  unsigned __int64 v21; // r14
  __int64 v22; // r9
  unsigned int v23; // r14d
  unsigned __int64 **v24; // r13
  _BYTE *v25; // rax
  int v26; // eax
  unsigned int v27; // edx
  _QWORD *v28; // rax
  unsigned __int64 v29; // rbx
  unsigned int v30; // eax
  unsigned int i; // r14d
  __int64 v32; // r12
  __int8 *v33; // r13
  size_t v34; // r8
  unsigned __int64 *v35; // rax
  __m128i v36; // rax
  char v37; // al
  const void **v38; // rdx
  _QWORD *v39; // rax
  __int64 v40; // r13
  __int64 v41; // rbx
  __int8 v42; // al
  unsigned int v43; // ebx
  __int8 v44; // al
  __int64 *v45; // r14
  __int64 *v46; // r12
  __int64 v47; // rdx
  __int64 v48; // rax
  _QWORD *v49; // rsi
  unsigned __int64 v50; // rax
  __int64 v51; // rsi
  __int64 v52; // rax
  unsigned __int8 v53; // al
  __int64 v54; // r8
  __int64 *v55; // rax
  __int64 v56; // rax
  unsigned __int8 v57; // al
  int v58; // eax
  int v59; // eax
  unsigned __int64 v60; // rdi
  __int8 v61; // al
  __int64 v62; // r12
  __int64 v63; // rcx
  __int64 v64; // rax
  __int64 v65; // rcx
  _BYTE *v66; // rdx
  _BYTE *v67; // r13
  __int64 v68; // rcx
  __int64 v69; // rax
  _BYTE *v70; // rbx
  unsigned __int64 v71; // rdi
  int v72; // ebx
  unsigned int v73; // eax
  unsigned __int64 v74; // rcx
  unsigned __int64 v75; // rax
  char *v76; // rsi
  __int64 v77; // rax
  unsigned __int64 ***v78; // rdi
  unsigned int *v79; // rbx
  __int64 v80; // r13
  __int64 v81; // r12
  __int64 v82; // rdi
  __int64 *v83; // rax
  __int64 v84; // r12
  unsigned __int8 *v85; // r14
  _QWORD *v86; // rax
  __int64 v87; // rax
  unsigned __int8 v88; // r8
  __int64 v89; // r11
  __int64 v90; // rax
  __m128i v91; // xmm3
  __int64 v92; // [rsp+8h] [rbp-1C8h]
  __int64 v93; // [rsp+10h] [rbp-1C0h]
  __int64 v95; // [rsp+28h] [rbp-1A8h]
  __int64 v96; // [rsp+30h] [rbp-1A0h]
  __int64 v97; // [rsp+38h] [rbp-198h]
  __int64 v98; // [rsp+38h] [rbp-198h]
  unsigned __int64 v99; // [rsp+40h] [rbp-190h]
  int v100; // [rsp+40h] [rbp-190h]
  void *v101; // [rsp+40h] [rbp-190h]
  size_t n; // [rsp+48h] [rbp-188h]
  unsigned int na; // [rsp+48h] [rbp-188h]
  unsigned int v104; // [rsp+50h] [rbp-180h]
  unsigned __int8 v105; // [rsp+58h] [rbp-178h]
  unsigned __int8 v106; // [rsp+58h] [rbp-178h]
  int v107; // [rsp+60h] [rbp-170h]
  unsigned int v108; // [rsp+60h] [rbp-170h]
  __int64 v109; // [rsp+60h] [rbp-170h]
  unsigned int v110; // [rsp+60h] [rbp-170h]
  __int64 v111; // [rsp+68h] [rbp-168h]
  __int64 v112; // [rsp+70h] [rbp-160h]
  _QWORD *v113; // [rsp+70h] [rbp-160h]
  __int64 v114; // [rsp+70h] [rbp-160h]
  unsigned int *v115; // [rsp+70h] [rbp-160h]
  char v116; // [rsp+78h] [rbp-158h]
  __int64 v117; // [rsp+78h] [rbp-158h]
  int v118; // [rsp+78h] [rbp-158h]
  __int64 v119; // [rsp+78h] [rbp-158h]
  __int64 v120; // [rsp+78h] [rbp-158h]
  unsigned __int64 v121; // [rsp+88h] [rbp-148h] BYREF
  unsigned __int64 **v122; // [rsp+90h] [rbp-140h] BYREF
  unsigned int v123; // [rsp+98h] [rbp-138h]
  __m128i v124; // [rsp+A0h] [rbp-130h] BYREF
  _BYTE v125[16]; // [rsp+B0h] [rbp-120h] BYREF
  unsigned __int64 *v126; // [rsp+C0h] [rbp-110h] BYREF
  size_t v127; // [rsp+C8h] [rbp-108h]
  unsigned __int64 **v128; // [rsp+D0h] [rbp-100h] BYREF
  unsigned int v129; // [rsp+D8h] [rbp-F8h]
  __m128i v130; // [rsp+E0h] [rbp-F0h] BYREF
  __m128i v131; // [rsp+F0h] [rbp-E0h] BYREF
  __int64 v132; // [rsp+100h] [rbp-D0h]
  unsigned __int64 **v133; // [rsp+110h] [rbp-C0h] BYREF
  __int64 v134; // [rsp+118h] [rbp-B8h]
  unsigned __int64 v135; // [rsp+120h] [rbp-B0h] BYREF
  unsigned int v136; // [rsp+128h] [rbp-A8h]
  __int16 v137; // [rsp+130h] [rbp-A0h]
  __m128i v138; // [rsp+140h] [rbp-90h] BYREF
  __m128i v139; // [rsp+150h] [rbp-80h] BYREF
  __int64 v140; // [rsp+160h] [rbp-70h]
  _BYTE *v141; // [rsp+170h] [rbp-60h] BYREF
  __int64 v142; // [rsp+178h] [rbp-58h]
  _BYTE v143[80]; // [rsp+180h] [rbp-50h] BYREF

  v1 = 0;
  if ( (*(_BYTE *)(a1 + 32) & 0xFu) - 7 <= 1 )
  {
    v2 = *(_QWORD *)(a1 - 32);
    v93 = v2;
    if ( v2 )
    {
      if ( *(_BYTE *)v2 == 10 )
      {
        v3 = sub_B2F730(a1);
        v4 = *(_QWORD *)(v2 + 8);
        v5 = v3;
        v97 = v3;
        v111 = sub_AE4AC0(v3, v4);
        n = v111 + 24;
        v96 = *(_DWORD *)(v111 + 20) & 0x7FFFFFFF;
        v104 = sub_AE43F0(v5, *(_QWORD *)(a1 + 8));
        v141 = v143;
        v142 = 0x100000000LL;
        v112 = *(_QWORD *)(a1 + 16);
        if ( v112 )
        {
          while ( 1 )
          {
            v6 = *(_QWORD *)(v112 + 24);
            if ( *(_BYTE *)v6 != 63 && (*(_BYTE *)v6 != 5 || *(_WORD *)(v6 + 2) != 34)
              || (sub_BB52D0((__int64)&v138, *(_QWORD *)(v112 + 24)), !(_BYTE)v140) )
            {
LABEL_7:
              v1 = 0;
              goto LABEL_8;
            }
            v123 = v104;
            if ( v104 > 0x40 )
              sub_C43690((__int64)&v122, 0, 0);
            else
              v122 = 0;
            if ( !(unsigned __int8)sub_BB6360(v6, v97, (__int64)&v122, 0, 0) )
              goto LABEL_49;
            sub_AB4E00((__int64)&v130, (__int64)&v138, v104);
            v124.m128i_i32[2] = v123;
            if ( v123 > 0x40 )
              sub_C43780((__int64)&v124, (const void **)&v122);
            else
              v124.m128i_i64[0] = (__int64)v122;
            sub_AADBC0((__int64)&v133, v124.m128i_i64);
            sub_AB4F10((__int64)&v126, (__int64)&v130, (__int64)&v133);
            if ( v136 > 0x40 && v135 )
              j_j___libc_free_0_0(v135);
            if ( (unsigned int)v134 > 0x40 && v133 )
              j_j___libc_free_0_0((unsigned __int64)v133);
            if ( v124.m128i_i32[2] > 0x40u && v124.m128i_i64[0] )
              j_j___libc_free_0_0(v124.m128i_u64[0]);
            if ( v131.m128i_i32[2] > 0x40u && v131.m128i_i64[0] )
              j_j___libc_free_0_0(v131.m128i_u64[0]);
            if ( v130.m128i_i32[2] > 0x40u && v130.m128i_i64[0] )
              j_j___libc_free_0_0(v130.m128i_u64[0]);
            if ( !sub_AB1B10((__int64)&v126, (__int64)&v122) )
            {
              if ( v129 <= 0x40 )
              {
                if ( v128 != v122 )
                  goto LABEL_46;
              }
              else if ( !sub_C43C50((__int64)&v128, (const void **)&v122) )
              {
                goto LABEL_203;
              }
            }
            v11 = *(_BYTE *)(v111 + 8);
            v133 = *(unsigned __int64 ***)v111;
            LOBYTE(v134) = v11;
            v12 = sub_CA1930(&v133);
            v107 = v127;
            if ( (unsigned int)v127 > 0x40 )
            {
              v99 = v12;
              if ( v107 - (unsigned int)sub_C444A0((__int64)&v126) > 0x40 || v99 <= *v126 )
              {
LABEL_44:
                v14 = v129;
LABEL_45:
                if ( v14 > 0x40 )
                {
LABEL_203:
                  v60 = (unsigned __int64)v128;
LABEL_125:
                  if ( v60 )
                    j_j___libc_free_0_0(v60);
                }
LABEL_46:
                if ( (unsigned int)v127 > 0x40 && v126 )
                  j_j___libc_free_0_0((unsigned __int64)v126);
LABEL_49:
                if ( v123 > 0x40 && v122 )
                  j_j___libc_free_0_0((unsigned __int64)v122);
                if ( (_BYTE)v140 )
                {
                  LOBYTE(v140) = 0;
                  if ( v139.m128i_i32[2] > 0x40u && v139.m128i_i64[0] )
                    j_j___libc_free_0_0(v139.m128i_u64[0]);
                  if ( v138.m128i_i32[2] > 0x40u && v138.m128i_i64[0] )
                    j_j___libc_free_0_0(v138.m128i_u64[0]);
                }
                goto LABEL_7;
              }
              v13 = *v126;
            }
            else
            {
              v13 = (unsigned __int64)v126;
              if ( v12 <= (unsigned __int64)v126 )
                goto LABEL_44;
            }
            v108 = sub_AE1C80(v111, v13);
            v124 = _mm_loadu_si128((const __m128i *)(n + 16LL * v108));
            if ( v108 == v96 - 1 )
            {
              v61 = *(_BYTE *)(v111 + 8);
              v130.m128i_i64[0] = *(_QWORD *)v111;
              v130.m128i_i8[8] = v61;
            }
            else
            {
              v130 = _mm_loadu_si128((const __m128i *)(n + 16LL * (v108 + 1)));
            }
            v15 = sub_CA1930(&v124);
            v16 = v15;
            v100 = v127;
            if ( (unsigned int)v127 > 0x40 )
            {
              v95 = v15;
              v58 = sub_C444A0((__int64)&v126);
              v16 = v95;
              if ( (unsigned int)(v100 - v58) > 0x40 )
                goto LABEL_44;
              v17 = *v126;
            }
            else
            {
              v17 = (unsigned __int64)v126;
            }
            if ( v16 != v17 )
              goto LABEL_44;
            v18 = sub_CA1930(&v130);
            v14 = v129;
            v19 = (_QWORD *)v18;
            if ( v129 > 0x40 )
            {
              v59 = sub_C444A0((__int64)&v128);
              v60 = (unsigned __int64)v128;
              if ( v14 - v59 > 0x40 )
                goto LABEL_125;
              v20 = *v128;
            }
            else
            {
              v20 = v128;
            }
            if ( v19 != v20 )
              goto LABEL_45;
            v21 = sub_CA1930(&v124);
            LODWORD(v134) = v123;
            if ( v123 > 0x40 )
              sub_C43780((__int64)&v133, (const void **)&v122);
            else
              v133 = v122;
            sub_C46F20((__int64)&v133, v21);
            v23 = v134;
            v24 = v133;
            LODWORD(v134) = 0;
            if ( (unsigned int)v142 >= HIDWORD(v142) )
              break;
            v25 = &v141[32 * (unsigned int)v142];
            if ( v25 )
            {
              *(_QWORD *)v25 = v6;
              *((_DWORD *)v25 + 6) = v23;
              *((_DWORD *)v25 + 2) = v108;
              *((_QWORD *)v25 + 2) = v24;
              v26 = v142;
              v27 = v134;
              goto LABEL_75;
            }
            if ( v23 > 0x40 && v133 )
            {
              j_j___libc_free_0_0((unsigned __int64)v133);
              v26 = v142;
              v27 = v134;
LABEL_75:
              LODWORD(v142) = v26 + 1;
LABEL_76:
              if ( v27 > 0x40 && v133 )
                j_j___libc_free_0_0((unsigned __int64)v133);
              goto LABEL_79;
            }
            LODWORD(v142) = v142 + 1;
LABEL_79:
            if ( v129 > 0x40 && v128 )
              j_j___libc_free_0_0((unsigned __int64)v128);
            if ( (unsigned int)v127 > 0x40 && v126 )
              j_j___libc_free_0_0((unsigned __int64)v126);
            if ( v123 > 0x40 && v122 )
              j_j___libc_free_0_0((unsigned __int64)v122);
            if ( (_BYTE)v140 )
            {
              LOBYTE(v140) = 0;
              if ( v139.m128i_i32[2] > 0x40u && v139.m128i_i64[0] )
                j_j___libc_free_0_0(v139.m128i_u64[0]);
              if ( v138.m128i_i32[2] > 0x40u && v138.m128i_i64[0] )
                j_j___libc_free_0_0(v138.m128i_u64[0]);
            }
            v112 = *(_QWORD *)(v112 + 8);
            if ( !v112 )
              goto LABEL_90;
          }
          v62 = sub_C8D7D0((__int64)&v141, (__int64)v143, 0, 0x20u, &v121, v22);
          v63 = 32LL * (unsigned int)v142;
          v64 = v63 + v62;
          if ( v63 + v62 )
          {
            *(_QWORD *)v64 = v6;
            *(_DWORD *)(v64 + 24) = v23;
            *(_DWORD *)(v64 + 8) = v108;
            v65 = (unsigned int)v142;
            *(_QWORD *)(v64 + 16) = v24;
            v63 = 32 * v65;
          }
          else if ( v23 > 0x40 && v24 )
          {
            j_j___libc_free_0_0((unsigned __int64)v24);
            v63 = 32LL * (unsigned int)v142;
          }
          v66 = v141;
          v67 = &v141[v63];
          if ( v141 != &v141[v63] )
          {
            v68 = v62 + v63;
            v69 = v62;
            do
            {
              if ( v69 )
              {
                *(_QWORD *)v69 = *(_QWORD *)v66;
                *(_DWORD *)(v69 + 8) = *((_DWORD *)v66 + 2);
                *(_DWORD *)(v69 + 24) = *((_DWORD *)v66 + 6);
                *(_QWORD *)(v69 + 16) = *((_QWORD *)v66 + 2);
                *((_DWORD *)v66 + 6) = 0;
              }
              v69 += 32;
              v66 += 32;
            }
            while ( v69 != v68 );
            v67 = v141;
            v70 = &v141[32 * (unsigned int)v142];
            if ( v70 != v141 )
            {
              do
              {
                v70 -= 32;
                if ( *((_DWORD *)v70 + 6) > 0x40u )
                {
                  v71 = *((_QWORD *)v70 + 2);
                  if ( v71 )
                    j_j___libc_free_0_0(v71);
                }
              }
              while ( v70 != v67 );
              v67 = v141;
            }
          }
          v72 = v121;
          if ( v67 != v143 )
            _libc_free((unsigned __int64)v67);
          LODWORD(v142) = v142 + 1;
          v27 = v134;
          v141 = (_BYTE *)v62;
          HIDWORD(v142) = v72;
          goto LABEL_76;
        }
LABEL_90:
        v124.m128i_i64[0] = (__int64)v125;
        v124.m128i_i64[1] = 0x200000000LL;
        sub_B91D10(a1, 19, (__int64)&v124);
        v28 = (_QWORD *)sub_BD5C60(a1);
        v98 = sub_BCB2D0(v28);
        if ( (*(_DWORD *)(v93 + 4) & 0x7FFFFFF) != 0 )
        {
          v29 = 8LL * (*(_DWORD *)(v93 + 4) & 0x7FFFFFF);
          v101 = (void *)sub_22077B0(v29);
          memset(v101, 0, v29);
          v30 = *(_DWORD *)(v93 + 4) & 0x7FFFFFF;
          if ( v30 )
          {
            for ( i = 0; ; i = na )
            {
              v109 = *(_QWORD *)(a1 + 40);
              v32 = *(_QWORD *)(v93 + 32 * (i - (unsigned __int64)v30));
              v113 = *(_QWORD **)(v32 + 8);
              v116 = *(_BYTE *)(a1 + 80) & 1;
              if ( !i )
              {
                v139.m128i_i8[4] = 48;
                v33 = &v139.m128i_i8[4];
                v126 = (unsigned __int64 *)&v128;
LABEL_95:
                v34 = 1;
                LOBYTE(v128) = *v33;
                v35 = (unsigned __int64 *)&v128;
                goto LABEL_96;
              }
              v74 = i;
              v33 = &v139.m128i_i8[5];
              do
              {
                *--v33 = v74 % 0xA + 48;
                v75 = v74;
                v74 /= 0xAu;
              }
              while ( v75 > 9 );
              v76 = (char *)(&v139.m128i_u8[5] - (unsigned __int8 *)v33);
              v133 = (unsigned __int64 **)(&v139.m128i_u8[5] - (unsigned __int8 *)v33);
              v34 = &v139.m128i_u8[5] - (unsigned __int8 *)v33;
              v126 = (unsigned __int64 *)&v128;
              if ( (unsigned __int64)(&v139.m128i_u8[5] - (unsigned __int8 *)v33) > 0xF )
              {
                v77 = sub_22409D0((__int64)&v126, (unsigned __int64 *)&v133, 0);
                v34 = &v139.m128i_u8[5] - (unsigned __int8 *)v33;
                v126 = (unsigned __int64 *)v77;
                v78 = (unsigned __int64 ***)v77;
                v128 = v133;
              }
              else
              {
                if ( v76 == (char *)1 )
                  goto LABEL_95;
                if ( !v76 )
                {
                  v35 = (unsigned __int64 *)&v128;
                  goto LABEL_96;
                }
                v78 = &v128;
              }
              memcpy(v78, v33, v34);
              v34 = (size_t)v133;
              v35 = v126;
LABEL_96:
              v127 = v34;
              *((_BYTE *)v35 + v34) = 0;
              v137 = 260;
              v133 = &v126;
              v36.m128i_i64[0] = (__int64)sub_BD5D20(a1);
              v130 = v36;
              v131.m128i_i64[0] = (__int64)".";
              v37 = v137;
              LOWORD(v132) = 773;
              if ( (_BYTE)v137 )
              {
                if ( (_BYTE)v137 == 1 )
                {
                  v91 = _mm_loadu_si128(&v131);
                  v138 = _mm_loadu_si128(&v130);
                  v140 = v132;
                  v139 = v91;
                }
                else
                {
                  if ( HIBYTE(v137) == 1 )
                  {
                    v38 = (const void **)v133;
                    v92 = v134;
                  }
                  else
                  {
                    v38 = (const void **)&v133;
                    v37 = 2;
                  }
                  v139.m128i_i64[0] = (__int64)v38;
                  v138.m128i_i64[0] = (__int64)&v130;
                  v139.m128i_i64[1] = v92;
                  LOBYTE(v140) = 2;
                  BYTE1(v140) = v37;
                }
              }
              else
              {
                LOWORD(v140) = 256;
              }
              BYTE4(v122) = 0;
              v39 = sub_BD2C40(88, unk_3F0FAE8);
              v40 = (__int64)v39;
              if ( v39 )
                sub_B30000((__int64)v39, v109, v113, v116, 8, v32, (__int64)&v138, 0, 0, (__int64)v122, 0);
              if ( v126 != (unsigned __int64 *)&v128 )
                j_j___libc_free_0((unsigned __int64)v126);
              *((_QWORD *)v101 + i) = v40;
              v41 = 16LL * i;
              v42 = *(_BYTE *)(v111 + v41 + 32);
              v138.m128i_i64[0] = *(_QWORD *)(v111 + v41 + 24);
              v138.m128i_i8[8] = v42;
              v43 = sub_CA1930(&v138);
              na = i + 1;
              if ( (*(_DWORD *)(v93 + 4) & 0x7FFFFFF) - 1 == i )
              {
                v44 = *(_BYTE *)(v111 + 8);
                v138.m128i_i64[0] = *(_QWORD *)v111;
              }
              else
              {
                v44 = *(_BYTE *)(v111 + 16LL * na + 32);
                v138.m128i_i64[0] = *(_QWORD *)(v111 + 16LL * na + 24);
              }
              v138.m128i_i8[8] = v44;
              v110 = sub_CA1930(&v138);
              v45 = (__int64 *)v124.m128i_i64[0];
              v46 = (__int64 *)(v124.m128i_i64[0] + 8LL * v124.m128i_u32[2]);
              if ( v46 != (__int64 *)v124.m128i_i64[0] )
              {
                do
                {
                  v57 = *(_BYTE *)(*v45 - 16);
                  if ( (v57 & 2) != 0 )
                    v47 = *(_QWORD *)(*v45 - 32);
                  else
                    v47 = *v45 - 16 - 8LL * ((v57 >> 2) & 0xF);
                  v48 = *(_QWORD *)(*(_QWORD *)v47 + 136LL);
                  v49 = *(_QWORD **)(v48 + 24);
                  if ( *(_DWORD *)(v48 + 32) > 0x40u )
                    v49 = (_QWORD *)*v49;
                  v117 = *v45;
                  v50 = (unsigned __int64)v49 - 1;
                  if ( !v49 )
                    v50 = 0;
                  v114 = *v45 - 16;
                  if ( v50 >= v43 && v110 > v50 )
                  {
                    v51 = (__int64)v49 - v43;
                    v52 = sub_ACD640(v98, v51, 0);
                    v138.m128i_i64[0] = (__int64)sub_B98A20(v52, v51);
                    v53 = *(_BYTE *)(v117 - 16);
                    if ( (v53 & 2) != 0 )
                      v54 = *(_QWORD *)(v117 - 32);
                    else
                      v54 = v114 - 8LL * ((v53 >> 2) & 0xF);
                    v138.m128i_i64[1] = *(_QWORD *)(v54 + 8);
                    v55 = (__int64 *)sub_BD5C60(a1);
                    v56 = sub_B9C770(v55, v138.m128i_i64, (__int64 *)2, 0, 1);
                    sub_B994D0(v40, 19, v56);
                  }
                  ++v45;
                }
                while ( v46 != v45 );
              }
              if ( (*(_BYTE *)(a1 + 7) & 0x20) != 0 && sub_B91C10(a1, 28) )
              {
                v73 = sub_B92110(a1);
                sub_B9D920(v40, v73);
              }
              v30 = *(_DWORD *)(v93 + 4) & 0x7FFFFFF;
              if ( v30 == na )
                break;
            }
          }
        }
        else
        {
          v101 = 0;
        }
        v79 = (unsigned int *)v141;
        v80 = 32LL * (unsigned int)v142;
        v115 = (unsigned int *)&v141[v80];
        if ( &v141[v80] != v141 )
        {
          do
          {
            LOBYTE(v137) = 0;
            v118 = (*(_BYTE *)(*(_QWORD *)v79 + 1LL) >> 1 << 31 >> 31) & 3;
            v83 = (__int64 *)sub_BD5C60(a1);
            v84 = sub_ACCFD0(v83, (__int64)(v79 + 4));
            v85 = (unsigned __int8 *)*((_QWORD *)v101 + v79[2]);
            v86 = (_QWORD *)sub_BD5C60(a1);
            v87 = sub_BCB2B0(v86);
            LOBYTE(v140) = 0;
            v88 = v118;
            v89 = v87;
            if ( (_BYTE)v137 )
            {
              v138.m128i_i32[2] = v134;
              if ( (unsigned int)v134 > 0x40 )
              {
                v106 = v118;
                v120 = v87;
                sub_C43780((__int64)&v138, (const void **)&v133);
                v88 = v106;
                v89 = v120;
              }
              else
              {
                v138.m128i_i64[0] = (__int64)v133;
              }
              v139.m128i_i32[2] = v136;
              if ( v136 > 0x40 )
              {
                v105 = v88;
                v119 = v89;
                sub_C43780((__int64)&v139, (const void **)&v135);
                v88 = v105;
                v89 = v119;
              }
              else
              {
                v139.m128i_i64[0] = v135;
              }
              LOBYTE(v140) = 1;
            }
            v130.m128i_i64[0] = v84;
            v81 = sub_AD9FD0(v89, v85, v130.m128i_i64, 1, v88, (__int64)&v138, 0);
            if ( (_BYTE)v140 )
            {
              LOBYTE(v140) = 0;
              if ( v139.m128i_i32[2] > 0x40u && v139.m128i_i64[0] )
                j_j___libc_free_0_0(v139.m128i_u64[0]);
              if ( v138.m128i_i32[2] > 0x40u && v138.m128i_i64[0] )
                j_j___libc_free_0_0(v138.m128i_u64[0]);
            }
            if ( (_BYTE)v137 )
            {
              LOBYTE(v137) = 0;
              if ( v136 > 0x40 && v135 )
                j_j___libc_free_0_0(v135);
              if ( (unsigned int)v134 > 0x40 && v133 )
                j_j___libc_free_0_0((unsigned __int64)v133);
            }
            v82 = *(_QWORD *)v79;
            v79 += 8;
            sub_BD84D0(v82, v81);
          }
          while ( v115 != v79 );
        }
        if ( *(_QWORD *)(a1 + 16) )
        {
          v90 = sub_ACADE0(*(__int64 ***)(a1 + 8));
          sub_BD84D0(a1, v90);
        }
        sub_B30290(a1);
        if ( v101 )
          j_j___libc_free_0((unsigned __int64)v101);
        if ( (_BYTE *)v124.m128i_i64[0] != v125 )
          _libc_free(v124.m128i_u64[0]);
        v1 = 1;
LABEL_8:
        v7 = v141;
        v8 = (unsigned __int64)&v141[32 * (unsigned int)v142];
        if ( v141 != (_BYTE *)v8 )
        {
          do
          {
            v8 -= 32LL;
            if ( *(_DWORD *)(v8 + 24) > 0x40u )
            {
              v9 = *(_QWORD *)(v8 + 16);
              if ( v9 )
                j_j___libc_free_0_0(v9);
            }
          }
          while ( v7 != (_BYTE *)v8 );
          v8 = (unsigned __int64)v141;
        }
        if ( (_BYTE *)v8 != v143 )
          _libc_free(v8);
      }
    }
  }
  return v1;
}
