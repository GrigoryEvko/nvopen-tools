// Function: sub_24A7CA0
// Address: 0x24a7ca0
//
void __fastcall sub_24A7CA0(__int64 a1, __int64 a2, void **a3)
{
  __int64 v3; // r15
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 **v11; // r12
  unsigned int i; // r13d
  _BYTE *v13; // rdx
  char *v14; // rcx
  char *v15; // rsi
  bool v16; // zf
  char *v17; // rax
  __int64 v18; // rdi
  _BYTE *v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rdi
  __int64 v22; // rax
  _WORD *v23; // rdx
  __int64 v24; // rdi
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // rsi
  unsigned int v28; // edx
  __int64 *v29; // rax
  __int64 v30; // r11
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rdi
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // rsi
  unsigned int v37; // edx
  __int64 *v38; // rax
  __int64 v39; // r11
  __int64 v40; // r8
  __int64 v41; // r13
  __int64 *v42; // rax
  char *v43; // rdx
  char *v44; // rdx
  char *v45; // rax
  char v46; // al
  char v47; // dl
  __m128i *v48; // rcx
  __m128i *v49; // rsi
  char v50; // dl
  __m128i *v51; // rsi
  __m128i *v52; // rcx
  char v53; // dl
  __m128i *v54; // rsi
  __m128i *v55; // rcx
  char v56; // dl
  __m128i *v57; // rax
  _QWORD *v58; // rcx
  char v59; // dl
  char v60; // si
  _QWORD *v61; // rcx
  __m128i v62; // xmm3
  int v63; // eax
  int v64; // eax
  __m128i v65; // xmm6
  __m128i v66; // xmm7
  __m128i v67; // xmm2
  __m128i v68; // xmm3
  __m128i v69; // xmm0
  __m128i v70; // xmm1
  __m128i v71; // xmm5
  __int64 *v72; // rax
  __int64 *v73; // r12
  __int64 v74; // r8
  __int64 *v75; // r15
  __int64 v76; // rdx
  __int64 v77; // r13
  const char *v78; // rax
  size_t v79; // rdx
  _WORD *v80; // rdi
  unsigned __int8 *v81; // rsi
  unsigned __int64 v82; // rax
  __int64 v83; // rax
  __int32 v84; // eax
  __int64 v85; // rdi
  _BYTE *v86; // rax
  __int64 v87; // rcx
  __int64 v88; // r8
  __int64 v89; // r9
  char v90; // al
  __m128i *v91; // rsi
  unsigned __int64 v92; // rax
  __int64 v93; // rax
  __int64 v94; // rax
  __m128i v95; // xmm1
  __m128i v96; // xmm6
  __m128i v97; // xmm7
  int v98; // r10d
  int v99; // r10d
  __int64 v100; // [rsp+0h] [rbp-2D0h]
  __int64 v101; // [rsp+8h] [rbp-2C8h]
  __int64 v102; // [rsp+10h] [rbp-2C0h]
  __int64 v103; // [rsp+18h] [rbp-2B8h]
  __int64 v104; // [rsp+20h] [rbp-2B0h]
  __int64 v105; // [rsp+28h] [rbp-2A8h]
  __int64 v106; // [rsp+30h] [rbp-2A0h]
  __int64 v107; // [rsp+38h] [rbp-298h]
  __int64 v108; // [rsp+40h] [rbp-290h]
  __int64 v109; // [rsp+48h] [rbp-288h]
  __int64 **v111; // [rsp+58h] [rbp-278h]
  size_t v112; // [rsp+58h] [rbp-278h]
  __int64 v113; // [rsp+58h] [rbp-278h]
  unsigned int v114; // [rsp+68h] [rbp-268h]
  __int64 v115[2]; // [rsp+70h] [rbp-260h] BYREF
  __int64 v116; // [rsp+80h] [rbp-250h] BYREF
  _QWORD v117[4]; // [rsp+90h] [rbp-240h] BYREF
  char v118; // [rsp+B0h] [rbp-220h]
  char v119; // [rsp+B1h] [rbp-21Fh]
  _QWORD v120[4]; // [rsp+C0h] [rbp-210h] BYREF
  __int16 v121; // [rsp+E0h] [rbp-1F0h]
  __m128i v122[2]; // [rsp+F0h] [rbp-1E0h] BYREF
  char v123; // [rsp+110h] [rbp-1C0h]
  char v124; // [rsp+111h] [rbp-1BFh]
  __m128i v125[2]; // [rsp+120h] [rbp-1B0h] BYREF
  char v126; // [rsp+140h] [rbp-190h]
  char v127; // [rsp+141h] [rbp-18Fh]
  __m128i v128; // [rsp+150h] [rbp-180h] BYREF
  __m128i v129; // [rsp+160h] [rbp-170h] BYREF
  __int64 v130; // [rsp+170h] [rbp-160h]
  __m128i v131; // [rsp+180h] [rbp-150h] BYREF
  __m128i v132; // [rsp+190h] [rbp-140h] BYREF
  __int64 v133; // [rsp+1A0h] [rbp-130h]
  __m128i v134; // [rsp+1B0h] [rbp-120h] BYREF
  __m128i v135; // [rsp+1C0h] [rbp-110h] BYREF
  __int64 v136; // [rsp+1D0h] [rbp-100h]
  __m128i v137; // [rsp+1E0h] [rbp-F0h] BYREF
  __m128i v138; // [rsp+1F0h] [rbp-E0h] BYREF
  __int64 v139; // [rsp+200h] [rbp-D0h]
  __m128i v140; // [rsp+210h] [rbp-C0h] BYREF
  __m128i v141; // [rsp+220h] [rbp-B0h] BYREF
  __int64 v142; // [rsp+230h] [rbp-A0h]
  __m128i v143; // [rsp+240h] [rbp-90h] BYREF
  __m128i v144; // [rsp+250h] [rbp-80h] BYREF
  __int64 v145; // [rsp+260h] [rbp-70h]
  __m128i v146; // [rsp+270h] [rbp-60h] BYREF
  __m128i v147; // [rsp+280h] [rbp-50h] BYREF
  __int64 v148; // [rsp+290h] [rbp-40h]

  v3 = a1;
  sub_CA0F50(v146.m128i_i64, a3);
  v6 = v146.m128i_i64[1];
  if ( (__m128i *)v146.m128i_i64[0] != &v147 )
    j_j___libc_free_0(v146.m128i_u64[0]);
  if ( v6 )
  {
    sub_CA0E80((__int64)a3, a2);
    sub_904010(a2, "\n");
  }
  v7 = sub_904010(a2, "  Number of Basic Blocks: ");
  v8 = sub_CB59D0(v7, *(unsigned int *)(a1 + 48));
  sub_904010(v8, "\n");
  if ( *(_DWORD *)(a1 + 48) )
  {
    v72 = *(__int64 **)(a1 + 40);
    v73 = &v72[2 * *(unsigned int *)(a1 + 56)];
    if ( v72 != v73 )
    {
      while ( 1 )
      {
        v74 = *v72;
        if ( *v72 != -4096 && v74 != -8192 )
          break;
        v72 += 2;
        if ( v73 == v72 )
          goto LABEL_6;
      }
      if ( v73 != v72 )
      {
        v75 = v72;
        while ( 1 )
        {
          v76 = *(_QWORD *)(a2 + 32);
          if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v76) <= 5 )
          {
            v113 = v74;
            v94 = sub_CB6200(a2, "  BB: ", 6u);
            v74 = v113;
            v77 = v94;
          }
          else
          {
            *(_DWORD *)v76 = 1111629856;
            *(_WORD *)(v76 + 4) = 8250;
            v77 = a2;
            *(_QWORD *)(a2 + 32) += 6LL;
          }
          if ( v74 )
          {
            v78 = sub_BD5D20(v74);
            v80 = *(_WORD **)(v77 + 32);
            v81 = (unsigned __int8 *)v78;
            v82 = *(_QWORD *)(v77 + 24) - (_QWORD)v80;
            if ( v82 < v79 )
              goto LABEL_136;
            if ( !v79 )
              goto LABEL_109;
          }
          else
          {
            v80 = *(_WORD **)(v77 + 32);
            v79 = 8;
            v81 = "FakeNode";
            if ( *(_QWORD *)(v77 + 24) - (_QWORD)v80 <= 7u )
            {
LABEL_136:
              v93 = sub_CB6200(v77, v81, v79);
              v80 = *(_WORD **)(v93 + 32);
              v77 = v93;
              v82 = *(_QWORD *)(v93 + 24) - (_QWORD)v80;
LABEL_109:
              if ( v82 > 1 )
                goto LABEL_110;
              goto LABEL_133;
            }
          }
          v112 = v79;
          memcpy(v80, v81, v79);
          v80 = (_WORD *)(v112 + *(_QWORD *)(v77 + 32));
          v92 = *(_QWORD *)(v77 + 24) - (_QWORD)v80;
          *(_QWORD *)(v77 + 32) = v80;
          if ( v92 > 1 )
          {
LABEL_110:
            *v80 = 8224;
            *(_QWORD *)(v77 + 32) += 2LL;
            goto LABEL_111;
          }
LABEL_133:
          v77 = sub_CB6200(v77, (unsigned __int8 *)"  ", 2u);
LABEL_111:
          v83 = v75[1];
          if ( *(_BYTE *)(v83 + 24) )
          {
            v143.m128i_i64[0] = v83 + 16;
            v137.m128i_i64[0] = (__int64)"  Count=";
            LOWORD(v145) = 267;
            LOWORD(v139) = 259;
            v147.m128i_i32[0] = *(_DWORD *)(v83 + 8);
            v146.m128i_i64[0] = (__int64)"Index=";
            LOWORD(v148) = 2307;
            sub_CA0F50(v134.m128i_i64, (void **)&v146);
            v90 = v139;
            if ( (_BYTE)v139 )
            {
              if ( (_BYTE)v139 == 1 )
              {
                v140.m128i_i64[0] = (__int64)&v134;
                LOWORD(v142) = 260;
              }
              else
              {
                if ( BYTE1(v139) == 1 )
                {
                  v109 = v137.m128i_i64[1];
                  v91 = (__m128i *)v137.m128i_i64[0];
                }
                else
                {
                  v91 = &v137;
                  v90 = 2;
                }
                v87 = v109;
                v140.m128i_i64[0] = (__int64)&v134;
                v141.m128i_i64[0] = (__int64)v91;
                v141.m128i_i64[1] = v109;
                LOBYTE(v142) = 4;
                BYTE1(v142) = v90;
              }
            }
            else
            {
              LOWORD(v142) = 256;
            }
            sub_9C6370(&v146, &v140, &v143, v87, v88, v89);
            sub_CA0F50(v131.m128i_i64, (void **)&v146);
            if ( (__m128i *)v134.m128i_i64[0] != &v135 )
              j_j___libc_free_0(v134.m128i_u64[0]);
          }
          else
          {
            v84 = *(_DWORD *)(v83 + 8);
            v146.m128i_i64[0] = (__int64)"Index=";
            v147.m128i_i32[0] = v84;
            LOWORD(v148) = 2307;
            sub_CA0F50(v131.m128i_i64, (void **)&v146);
          }
          v85 = sub_CB6200(v77, (unsigned __int8 *)v131.m128i_i64[0], v131.m128i_u64[1]);
          v86 = *(_BYTE **)(v85 + 32);
          if ( *(_BYTE **)(v85 + 24) == v86 )
          {
            sub_CB6200(v85, (unsigned __int8 *)"\n", 1u);
          }
          else
          {
            *v86 = 10;
            ++*(_QWORD *)(v85 + 32);
          }
          if ( (__m128i *)v131.m128i_i64[0] != &v132 )
            j_j___libc_free_0(v131.m128i_u64[0]);
          v75 += 2;
          if ( v75 != v73 )
          {
            while ( 1 )
            {
              v74 = *v75;
              if ( *v75 != -8192 && v74 != -4096 )
                break;
              v75 += 2;
              if ( v73 == v75 )
                goto LABEL_121;
            }
            if ( v73 != v75 )
              continue;
          }
LABEL_121:
          v3 = a1;
          break;
        }
      }
    }
  }
LABEL_6:
  v9 = sub_904010(a2, "  Number of Edges: ");
  v10 = sub_CB59D0(v9, (__int64)(*(_QWORD *)(v3 + 16) - *(_QWORD *)(v3 + 8)) >> 3);
  sub_904010(v10, " (*: Instrument, C: CriticalEdge, -: Removed)\n");
  v11 = *(__int64 ***)(v3 + 8);
  v111 = *(__int64 ***)(v3 + 16);
  if ( v111 != v11 )
  {
    for ( i = 0; ; i = v114 )
    {
      v20 = *(_QWORD *)(a2 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v20) <= 6 )
      {
        v21 = sub_CB6200(a2, "  Edge ", 7u);
      }
      else
      {
        *(_DWORD *)v20 = 1682251808;
        v21 = a2;
        *(_WORD *)(v20 + 4) = 25959;
        *(_BYTE *)(v20 + 6) = 32;
        *(_QWORD *)(a2 + 32) += 7LL;
      }
      v114 = i + 1;
      v22 = sub_CB59D0(v21, i);
      v23 = *(_WORD **)(v22 + 32);
      v24 = v22;
      if ( *(_QWORD *)(v22 + 24) - (_QWORD)v23 <= 1u )
      {
        v24 = sub_CB6200(v22, (unsigned __int8 *)": ", 2u);
      }
      else
      {
        *v23 = 8250;
        *(_QWORD *)(v22 + 32) += 2LL;
      }
      v25 = *(unsigned int *)(v3 + 56);
      v26 = *(_QWORD *)(v3 + 40);
      v27 = **v11;
      if ( (_DWORD)v25 )
      {
        v28 = (v25 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
        v29 = (__int64 *)(v26 + 16LL * v28);
        v30 = *v29;
        if ( v27 == *v29 )
          goto LABEL_27;
        v64 = 1;
        while ( v30 != -4096 )
        {
          v98 = v64 + 1;
          v28 = (v25 - 1) & (v64 + v28);
          v29 = (__int64 *)(v26 + 16LL * v28);
          v30 = *v29;
          if ( v27 == *v29 )
            goto LABEL_27;
          v64 = v98;
        }
      }
      v29 = (__int64 *)(v26 + 16 * v25);
LABEL_27:
      v31 = sub_CB59D0(v24, *(unsigned int *)(v29[1] + 8));
      v32 = *(_QWORD *)(v31 + 32);
      v33 = v31;
      if ( (unsigned __int64)(*(_QWORD *)(v31 + 24) - v32) <= 2 )
      {
        v33 = sub_CB6200(v31, "-->", 3u);
      }
      else
      {
        *(_BYTE *)(v32 + 2) = 62;
        *(_WORD *)v32 = 11565;
        *(_QWORD *)(v31 + 32) += 3LL;
      }
      v34 = *(unsigned int *)(v3 + 56);
      v35 = *(_QWORD *)(v3 + 40);
      v36 = (*v11)[1];
      if ( (_DWORD)v34 )
      {
        v37 = (v34 - 1) & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
        v38 = (__int64 *)(v35 + 16LL * v37);
        v39 = *v38;
        if ( v36 == *v38 )
          goto LABEL_31;
        v63 = 1;
        while ( v39 != -4096 )
        {
          v99 = v63 + 1;
          v37 = (v34 - 1) & (v63 + v37);
          v38 = (__int64 *)(v35 + 16LL * v37);
          v39 = *v38;
          if ( v36 == *v38 )
            goto LABEL_31;
          v63 = v99;
        }
      }
      v38 = (__int64 *)(v35 + 16 * v34);
LABEL_31:
      v41 = sub_CB59D0(v33, *(unsigned int *)(v38[1] + 8));
      v42 = *v11;
      if ( *((_BYTE *)*v11 + 40) )
      {
        v120[0] = v42 + 4;
        v117[0] = "  Count=";
        v143.m128i_i64[0] = (__int64)(v42 + 2);
        v43 = " ";
        v137.m128i_i64[0] = (__int64)"  W=";
        v121 = 267;
        v119 = 1;
        v118 = 3;
        LOWORD(v145) = 267;
        LOWORD(v139) = 259;
        if ( *((_BYTE *)v42 + 26) )
          v43 = (char *)"c";
        LOWORD(v133) = 259;
        v131.m128i_i64[0] = (__int64)v43;
        v44 = "*";
        if ( *((_BYTE *)v42 + 24) )
          v44 = " ";
        v127 = 1;
        v126 = 3;
        v16 = *((_BYTE *)v42 + 25) == 0;
        v45 = "-";
        if ( v16 )
          v45 = " ";
        v125[0].m128i_i64[0] = (__int64)v44;
        v124 = 1;
        v122[0].m128i_i64[0] = (__int64)v45;
        v123 = 3;
        sub_9C6370(&v128, v122, v125, (__int64)&v128, v40, 267);
        v46 = v130;
        if ( !(_BYTE)v130 || (v47 = v133, v48 = &v128, !(_BYTE)v133) )
        {
          LOWORD(v136) = 256;
          goto LABEL_62;
        }
        if ( (_BYTE)v130 == 1 )
        {
          v69 = _mm_loadu_si128(&v131);
          v70 = _mm_loadu_si128(&v132);
          v136 = v133;
          v46 = v133;
          v134 = v69;
          v135 = v70;
        }
        else
        {
          if ( (_BYTE)v133 != 1 )
          {
            if ( BYTE1(v130) == 1 )
            {
              v107 = v128.m128i_i64[1];
              v48 = (__m128i *)v128.m128i_i64[0];
            }
            else
            {
              v46 = 2;
            }
            if ( BYTE1(v133) == 1 )
            {
              v106 = v131.m128i_i64[1];
              v49 = (__m128i *)v131.m128i_i64[0];
            }
            else
            {
              v49 = &v131;
              v47 = 2;
            }
            v134.m128i_i64[0] = (__int64)v48;
            BYTE1(v136) = v47;
            v50 = v139;
            v134.m128i_i64[1] = v107;
            v135.m128i_i64[0] = (__int64)v49;
            v135.m128i_i64[1] = v106;
            LOBYTE(v136) = v46;
            if ( (_BYTE)v139 )
            {
LABEL_47:
              if ( v50 != 1 )
              {
                if ( BYTE1(v136) == 1 )
                {
                  v101 = v134.m128i_i64[1];
                  v51 = (__m128i *)v134.m128i_i64[0];
                }
                else
                {
                  v51 = &v134;
                  v46 = 2;
                }
                if ( BYTE1(v139) == 1 )
                {
                  v100 = v137.m128i_i64[1];
                  v52 = (__m128i *)v137.m128i_i64[0];
                }
                else
                {
                  v52 = &v137;
                  v50 = 2;
                }
                v140.m128i_i64[0] = (__int64)v51;
                v141.m128i_i64[0] = (__int64)v52;
                v140.m128i_i64[1] = v101;
                v141.m128i_i64[1] = v100;
                LOBYTE(v142) = v46;
                BYTE1(v142) = v50;
                goto LABEL_53;
              }
              v46 = v136;
              v96 = _mm_loadu_si128(&v134);
              v97 = _mm_loadu_si128(&v135);
              v142 = v136;
              v140 = v96;
              v141 = v97;
              if ( (_BYTE)v136 )
                goto LABEL_53;
LABEL_63:
              LOWORD(v148) = 256;
              goto LABEL_64;
            }
LABEL_62:
            LOWORD(v142) = 256;
            goto LABEL_63;
          }
          v65 = _mm_loadu_si128(&v128);
          v66 = _mm_loadu_si128(&v129);
          v136 = v130;
          v134 = v65;
          v135 = v66;
        }
        v50 = v139;
        if ( !(_BYTE)v139 )
          goto LABEL_62;
        if ( v46 != 1 )
          goto LABEL_47;
        v67 = _mm_loadu_si128(&v137);
        v68 = _mm_loadu_si128(&v138);
        v142 = v139;
        v46 = v139;
        v140 = v67;
        v141 = v68;
LABEL_53:
        v53 = v145;
        if ( !(_BYTE)v145 )
          goto LABEL_63;
        if ( v46 == 1 )
        {
          v71 = _mm_loadu_si128(&v144);
          v146 = _mm_loadu_si128(&v143);
          v148 = v145;
          v147 = v71;
        }
        else if ( (_BYTE)v145 == 1 )
        {
          v95 = _mm_loadu_si128(&v141);
          v146 = _mm_loadu_si128(&v140);
          v148 = v142;
          v147 = v95;
        }
        else
        {
          if ( BYTE1(v142) == 1 )
          {
            v105 = v140.m128i_i64[1];
            v54 = (__m128i *)v140.m128i_i64[0];
          }
          else
          {
            v54 = &v140;
            v46 = 2;
          }
          if ( BYTE1(v145) == 1 )
          {
            v104 = v143.m128i_i64[1];
            v55 = (__m128i *)v143.m128i_i64[0];
          }
          else
          {
            v55 = &v143;
            v53 = 2;
          }
          v146.m128i_i64[0] = (__int64)v54;
          v147.m128i_i64[0] = (__int64)v55;
          v146.m128i_i64[1] = v105;
          v147.m128i_i64[1] = v104;
          LOBYTE(v148) = v46;
          BYTE1(v148) = v53;
        }
LABEL_64:
        sub_CA0F50(v115, (void **)&v146);
        v56 = v118;
        v57 = (__m128i *)v115;
        if ( v118 )
        {
          if ( v118 == 1 )
          {
            v143.m128i_i64[0] = (__int64)v115;
            LOWORD(v145) = 260;
            v59 = v121;
            if ( (_BYTE)v121 )
            {
              if ( (_BYTE)v121 == 1 )
                goto LABEL_75;
              v60 = 4;
              v103 = v143.m128i_i64[1];
              if ( HIBYTE(v121) == 1 )
                goto LABEL_143;
LABEL_71:
              v61 = v120;
              v59 = 2;
LABEL_72:
              v146.m128i_i64[0] = (__int64)v57;
              v147.m128i_i64[0] = (__int64)v61;
              v146.m128i_i64[1] = v103;
              LOBYTE(v148) = v60;
              v147.m128i_i64[1] = v102;
              BYTE1(v148) = v59;
LABEL_78:
              sub_CA0F50(v134.m128i_i64, (void **)&v146);
              if ( (__int64 *)v115[0] != &v116 )
                j_j___libc_free_0(v115[0]);
              goto LABEL_15;
            }
          }
          else
          {
            if ( v119 == 1 )
            {
              v108 = v117[1];
              v58 = (_QWORD *)v117[0];
            }
            else
            {
              v58 = v117;
              v56 = 2;
            }
            v143.m128i_i64[0] = (__int64)v115;
            BYTE1(v145) = v56;
            v59 = v121;
            v144.m128i_i64[0] = (__int64)v58;
            v144.m128i_i64[1] = v108;
            LOBYTE(v145) = 4;
            if ( (_BYTE)v121 )
            {
              if ( (_BYTE)v121 != 1 )
              {
                v57 = &v143;
                v60 = 2;
                if ( HIBYTE(v121) != 1 )
                  goto LABEL_71;
LABEL_143:
                v102 = v120[1];
                v61 = (_QWORD *)v120[0];
                goto LABEL_72;
              }
LABEL_75:
              v62 = _mm_loadu_si128(&v144);
              v146 = _mm_loadu_si128(&v143);
              v148 = v145;
              v147 = v62;
              goto LABEL_78;
            }
          }
        }
        else
        {
          LOWORD(v145) = 256;
        }
        LOWORD(v148) = 256;
        goto LABEL_78;
      }
      v13 = v42 + 2;
      v14 = " ";
      if ( *((_BYTE *)v42 + 26) )
        v14 = (char *)"c";
      v15 = "*";
      if ( *((_BYTE *)v42 + 24) )
        v15 = " ";
      v16 = *((_BYTE *)v42 + 25) == 0;
      LOWORD(v139) = 771;
      v17 = "-";
      v147.m128i_i64[0] = (__int64)v13;
      if ( v16 )
        v17 = " ";
      v138.m128i_i64[0] = (__int64)v15;
      v137.m128i_i64[0] = (__int64)v17;
      v140 = (__m128i)(unsigned __int64)&v137;
      LOWORD(v142) = 770;
      v143 = (__m128i)(unsigned __int64)&v140;
      v144.m128i_i64[0] = (__int64)"  W=";
      LOWORD(v145) = 770;
      v146 = (__m128i)(unsigned __int64)&v143;
      v141.m128i_i64[0] = (__int64)v14;
      LOWORD(v148) = 2818;
      sub_CA0F50(v134.m128i_i64, (void **)&v146);
LABEL_15:
      v18 = sub_CB6200(v41, (unsigned __int8 *)v134.m128i_i64[0], v134.m128i_u64[1]);
      v19 = *(_BYTE **)(v18 + 32);
      if ( *(_BYTE **)(v18 + 24) == v19 )
      {
        sub_CB6200(v18, (unsigned __int8 *)"\n", 1u);
      }
      else
      {
        *v19 = 10;
        ++*(_QWORD *)(v18 + 32);
      }
      if ( (__m128i *)v134.m128i_i64[0] != &v135 )
        j_j___libc_free_0(v134.m128i_u64[0]);
      if ( v111 == ++v11 )
        return;
    }
  }
}
