// Function: sub_F16D60
// Address: 0xf16d60
//
unsigned __int8 *__fastcall sub_F16D60(const __m128i *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 v5; // rcx
  __int64 v6; // rax
  _BYTE **v7; // rax
  __int64 v8; // r8
  unsigned __int8 *result; // rax
  __int64 v10; // r15
  unsigned __int8 **v11; // r15
  __int64 v12; // rbx
  signed __int64 v13; // rax
  unsigned __int64 v14; // rdi
  __int64 v15; // r14
  __int64 v16; // rdx
  unsigned __int8 v17; // al
  __int64 v18; // r15
  __int64 v19; // r14
  char *v20; // r14
  __int64 v21; // rbx
  unsigned __int8 **v22; // r15
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rax
  signed __int64 v27; // rbx
  int v28; // r12d
  _BYTE **v29; // r14
  unsigned __int64 v30; // rdi
  unsigned __int64 v31; // r10
  __int64 v32; // rax
  __int64 v33; // rax
  unsigned __int64 v34; // rdx
  bool v35; // zf
  char *v36; // rsi
  __int64 v37; // rsi
  char v38; // r8
  __int64 v39; // rsi
  bool v40; // al
  unsigned __int8 v41; // r15
  __m128i *v42; // rbx
  bool v43; // al
  unsigned __int8 v44; // al
  unsigned int v45; // ebx
  int v46; // eax
  __int64 v47; // r15
  __int64 v48; // r14
  __int64 v49; // r15
  __int64 v50; // rcx
  __int64 v51; // rax
  unsigned int v52; // r10d
  __int64 v53; // rsi
  __int64 *v54; // rax
  __int64 v55; // rax
  __int64 v56; // r8
  __int64 v57; // rdx
  unsigned __int64 v58; // r9
  unsigned int v59; // edx
  unsigned int v60; // esi
  __m128i *v61; // r12
  unsigned __int8 *v62; // r13
  _BYTE *v63; // rdi
  __int64 v64; // rbx
  unsigned int **v65; // r15
  unsigned __int8 v66; // al
  int v67; // edx
  __int64 v68; // rax
  __int64 v69; // rax
  char *v70; // r15
  unsigned int **v71; // r14
  __int64 v72; // rax
  __int64 v73; // rax
  _QWORD *v74; // rax
  __int64 v75; // rax
  __int64 v76; // r8
  __int64 v77; // r9
  __int64 v78; // r15
  __int64 v79; // rax
  unsigned __int64 v80; // rdx
  __int64 v81; // r8
  __int64 v82; // r9
  __int64 v83; // rax
  __int64 v84; // rdi
  __m128i v85; // xmm6
  __m128i v86; // xmm2
  unsigned __int64 v87; // xmm4_8
  __int64 v88; // rax
  __int64 v89; // r8
  __int64 v90; // r9
  _QWORD *v91; // r15
  __int64 v92; // rcx
  __int64 v93; // rdx
  _QWORD *v94; // rbx
  __int64 v95; // r14
  unsigned __int64 v96; // rdx
  char *v97; // rdx
  __int64 v98; // rdx
  unsigned __int64 v99; // r8
  __int64 v100; // rdx
  int v101; // eax
  _QWORD *v102; // rbx
  __int64 v103; // r15
  char *v104; // rax
  __int64 v105; // rdx
  _QWORD *v106; // r15
  __int64 v107; // r14
  char *v108; // rax
  int v109; // eax
  unsigned int v110; // r14d
  _QWORD *v111; // rbx
  unsigned __int64 v112; // r8
  char *v113; // rax
  __int64 v114; // rdx
  __int64 v115; // rax
  unsigned int **v116; // r11
  unsigned int *v117; // rbx
  __int64 v118; // r14
  __int64 v119; // rdx
  __int64 v120; // [rsp+8h] [rbp-148h]
  unsigned int v121; // [rsp+8h] [rbp-148h]
  __int64 v123; // [rsp+18h] [rbp-138h]
  unsigned int v124; // [rsp+18h] [rbp-138h]
  unsigned int v125; // [rsp+18h] [rbp-138h]
  __int64 v126; // [rsp+18h] [rbp-138h]
  unsigned int **v127; // [rsp+18h] [rbp-138h]
  char v128; // [rsp+20h] [rbp-130h]
  unsigned __int8 *v129; // [rsp+20h] [rbp-130h]
  int v130; // [rsp+20h] [rbp-130h]
  unsigned __int8 *v131; // [rsp+20h] [rbp-130h]
  unsigned __int8 *v132; // [rsp+20h] [rbp-130h]
  __int64 v133; // [rsp+20h] [rbp-130h]
  __int64 v134; // [rsp+20h] [rbp-130h]
  unsigned __int8 *v135; // [rsp+20h] [rbp-130h]
  __int64 v136; // [rsp+20h] [rbp-130h]
  unsigned __int8 *v137; // [rsp+20h] [rbp-130h]
  __int64 v138; // [rsp+20h] [rbp-130h]
  __int64 v139; // [rsp+20h] [rbp-130h]
  unsigned __int8 *v140; // [rsp+20h] [rbp-130h]
  __int64 v141; // [rsp+38h] [rbp-118h] BYREF
  __int64 v142; // [rsp+40h] [rbp-110h] BYREF
  unsigned int v143; // [rsp+48h] [rbp-108h]
  _QWORD v144[4]; // [rsp+50h] [rbp-100h] BYREF
  __int16 v145; // [rsp+70h] [rbp-E0h]
  __m128i v146; // [rsp+80h] [rbp-D0h] BYREF
  __m128i v147; // [rsp+90h] [rbp-C0h] BYREF
  unsigned __int64 v148; // [rsp+A0h] [rbp-B0h]
  __int64 v149; // [rsp+A8h] [rbp-A8h]
  __m128i v150; // [rsp+B0h] [rbp-A0h]
  __int64 v151; // [rsp+C0h] [rbp-90h]
  char *v152; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v153; // [rsp+D8h] [rbp-78h]
  char v154[112]; // [rsp+E0h] [rbp-70h] BYREF

  v4 = a2;
  v5 = *(_QWORD *)(a3 + 8);
  v120 = v5;
  if ( (unsigned int)*(unsigned __int8 *)(v5 + 8) - 17 <= 1 )
    v120 = **(_QWORD **)(v5 + 16);
  v128 = sub_B4DD90(a2);
  if ( v128 )
  {
    v6 = *(_QWORD *)(a3 + 16);
    if ( v6 && !*(_QWORD *)(v6 + 8)
      || (v7 = (_BYTE **)(a3 + 32 * (1LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF))), (_BYTE **)a3 == v7) )
    {
LABEL_31:
      if ( (*(_BYTE *)(a3 + 7) & 0x40) != 0 )
        v21 = *(_QWORD *)(a3 - 8);
      else
        v21 = a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
      v22 = (unsigned __int8 **)(v21 + 32);
      v23 = sub_BB5290(a3);
      v24 = v23 & 0xFFFFFFFFFFFFFFF9LL | 4;
      v25 = v23 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v25 )
      {
        v123 = v24;
        v25 = sub_BCBAE0(0, *(unsigned __int8 **)(v21 + 32), v24);
        v24 = v123;
      }
      v141 = v25;
      v26 = *(_DWORD *)(a3 + 4) & 0x7FFFFFF;
      if ( a3 != a3 + 32 * (1 - v26) )
      {
        v27 = v24;
        v124 = 0;
        v28 = 1;
        v29 = (_BYTE **)(a3 + 32 * (1 - v26));
        while ( 1 )
        {
          v30 = v27 & 0xFFFFFFFFFFFFFFF8LL;
          v31 = v27 & 0xFFFFFFFFFFFFFFF8LL;
          if ( **v29 == 17 )
            goto LABEL_44;
          if ( !v27 )
            goto LABEL_60;
          v32 = (v27 >> 1) & 3;
          if ( v32 != 2 )
            break;
          v33 = v27 & 0xFFFFFFFFFFFFFFF8LL;
          if ( !v30 )
            goto LABEL_60;
LABEL_43:
          v141 = v33;
          v124 = v28;
          v128 = 0;
LABEL_44:
          if ( !v27 )
            goto LABEL_52;
          v24 = (v27 >> 1) & 3;
          if ( v24 != 2 )
          {
            if ( v24 == 1 && v30 )
            {
              v31 = *(_QWORD *)(v30 + 24);
              goto LABEL_47;
            }
LABEL_52:
            v31 = sub_BCBAE0(v30, *v22, v24);
            goto LABEL_47;
          }
          if ( !v30 )
            goto LABEL_52;
LABEL_47:
          v24 = *(unsigned __int8 *)(v31 + 8);
          if ( (_BYTE)v24 == 16 )
          {
            v27 = *(_QWORD *)(v31 + 24) & 0xFFFFFFFFFFFFFFF9LL | 4;
          }
          else if ( (unsigned int)(unsigned __int8)v24 - 17 > 1 )
          {
            v27 = 0;
            if ( (_BYTE)v24 == 15 )
              v27 = v31 & 0xFFFFFFFFFFFFFFF9LL;
          }
          else
          {
            v27 = v31 & 0xFFFFFFFFFFFFFFF9LL | 2;
          }
          v29 += 4;
          ++v28;
          v22 += 4;
          if ( (_BYTE **)a3 == v29 )
          {
            v4 = a2;
            goto LABEL_71;
          }
        }
        if ( v32 == 1 && v30 )
        {
          v33 = *(_QWORD *)(v30 + 24);
          goto LABEL_43;
        }
LABEL_60:
        v33 = sub_BCBAE0(v30, *v22, v24);
        v31 = v27 & 0xFFFFFFFFFFFFFFF8LL;
        v30 = v27 & 0xFFFFFFFFFFFFFFF8LL;
        goto LABEL_43;
      }
      v124 = 0;
LABEL_71:
      v143 = sub_AE43F0(a1[5].m128i_i64[1], v120);
      if ( v143 > 0x40 )
        sub_C43690((__int64)&v142, 0, 0);
      else
        v142 = 0;
      if ( (*(_DWORD *)(a3 + 4) & 0x7FFFFFF) - 1 != v124 )
      {
        if ( sub_BCEA30(v141) )
        {
          result = 0;
LABEL_112:
          if ( v143 > 0x40 )
          {
            if ( v142 )
            {
              v132 = result;
              j_j___libc_free_0_0(v142);
              return v132;
            }
          }
          return result;
        }
        v36 = v154;
        v152 = v154;
        v153 = 0x600000000LL;
        if ( !v128 )
        {
          v74 = (_QWORD *)sub_BD5C60(v4);
          v75 = sub_BCB2D0(v74);
          v78 = sub_AD6530(v75, (__int64)v154);
          v79 = (unsigned int)v153;
          v80 = (unsigned int)v153 + 1LL;
          if ( v80 > HIDWORD(v153) )
          {
            sub_C8D5F0((__int64)&v152, v154, v80, 8u, v76, v77);
            v79 = (unsigned int)v153;
          }
          *(_QWORD *)&v152[8 * v79] = v78;
          LODWORD(v153) = v153 + 1;
          v36 = &v152[8 * (unsigned int)v153];
        }
        sub_F091F0(
          (__int64)&v152,
          v36,
          (char *)(a3 + 32 * (v124 + 1LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF))),
          (_QWORD *)a3);
        v37 = sub_AE54E0(a1[5].m128i_i64[1], v141, v152, (unsigned int)v153);
        sub_C46A40((__int64)&v142, v37);
        if ( v152 != v154 )
          _libc_free(v152, v37);
      }
      v38 = sub_B4DE60(v4, a1[5].m128i_i64[1], (__int64)&v142);
      result = 0;
      if ( v38 )
      {
        v39 = a1[5].m128i_i64[1];
        sub_AE5990((unsigned int *)&v146, v39, &v141, (__int64)&v142);
        if ( v143 <= 0x40 )
        {
          v40 = v142 == 0;
        }
        else
        {
          v121 = v143;
          v40 = v121 == (unsigned int)sub_C444A0((__int64)&v142);
        }
        if ( v40 )
        {
          v41 = v128 ^ 1;
          if ( v128
            || ((v42 = (__m128i *)v146.m128i_i64[0], *(_DWORD *)(v146.m128i_i64[0] + 8) <= 0x40u)
              ? (v43 = *(_QWORD *)v146.m128i_i64[0] == 0)
              : (v130 = *(_DWORD *)(v146.m128i_i64[0] + 8), v43 = v130 == (unsigned int)sub_C444A0(v146.m128i_i64[0])),
                v43) )
          {
            v44 = (unsigned __int8)(*(_BYTE *)(a3 + 1) & *(_BYTE *)(v4 + 1)) >> 1;
            v45 = v44;
            if ( (*(_BYTE *)(a3 + 1) & *(_BYTE *)(v4 + 1) & 2) == 0 && (v44 & 2) != 0 )
              v45 = v44 & 0xFC;
            v153 = 0x600000000LL;
            v46 = *(_DWORD *)(a3 + 4);
            v47 = 16LL * v41;
            v152 = v154;
            sub_F091F0(
              (__int64)&v152,
              v154,
              (char *)(a3 + 32 * (1LL - (v46 & 0x7FFFFFF))),
              (_QWORD *)(a3 - 32LL * ((v46 & 0x7FFFFFF) - 1 - v124)));
            v48 = v146.m128i_i64[0] + v47;
            if ( v146.m128i_i64[0] + v47 == v146.m128i_i64[0] + 16LL * v146.m128i_u32[2] )
            {
              v59 = v153;
            }
            else
            {
              v49 = v146.m128i_i64[0] + 16LL * v146.m128i_u32[2];
              do
              {
                v54 = (__int64 *)sub_BD5C60(v4);
                v55 = sub_ACCFD0(v54, v48);
                v57 = (unsigned int)v153;
                v58 = (unsigned int)v153 + 1LL;
                if ( v58 > HIDWORD(v153) )
                {
                  v136 = v55;
                  sub_C8D5F0((__int64)&v152, v154, (unsigned int)v153 + 1LL, 8u, v56, v58);
                  v57 = (unsigned int)v153;
                  v55 = v136;
                }
                *(_QWORD *)&v152[8 * v57] = v55;
                v59 = v153 + 1;
                LODWORD(v153) = v153 + 1;
                v60 = *(_DWORD *)(v48 + 8);
                if ( v60 <= 0x40 )
                  v50 = *(_QWORD *)v48;
                else
                  v50 = *(_QWORD *)(*(_QWORD *)v48 + 8LL * ((v60 - 1) >> 6));
                v51 = v50 & (1LL << ((unsigned __int8)v60 - 1));
                v52 = *(_DWORD *)(v146.m128i_i64[0] + 8);
                v53 = *(_QWORD *)v146.m128i_i64[0];
                if ( v52 > 0x40 )
                  v53 = *(_QWORD *)(v53 + 8LL * ((v52 - 1) >> 6));
                if ( ((v53 & (1LL << ((unsigned __int8)v52 - 1))) == 0) != (v51 == 0) )
                  v45 &= 0xFFFFFFFC;
                if ( v51 )
                  v45 &= ~4u;
                v48 += 16;
              }
              while ( v49 != v48 );
            }
            v70 = v152;
            v145 = 257;
            v126 = v59;
            v71 = (unsigned int **)a1[2].m128i_i64[0];
            v134 = *(_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
            v72 = sub_BB5290(a3);
            v73 = sub_921130(v71, v72, v134, (_BYTE **)v70, v126, (__int64)v144, v45);
            v39 = v4;
            result = sub_F162A0((__int64)a1, v4, v73);
            if ( v152 != v154 )
            {
              v135 = result;
              _libc_free(v152, v4);
              result = v135;
            }
            v42 = (__m128i *)v146.m128i_i64[0];
          }
          else
          {
            result = 0;
          }
        }
        else
        {
          v42 = (__m128i *)v146.m128i_i64[0];
          result = 0;
        }
        v61 = &v42[v146.m128i_u32[2]];
        if ( v61 != v42 )
        {
          v62 = result;
          do
          {
            --v61;
            if ( v61->m128i_i32[2] > 0x40u && v61->m128i_i64[0] )
              j_j___libc_free_0_0(v61->m128i_i64[0]);
          }
          while ( v61 != v42 );
          v42 = (__m128i *)v146.m128i_i64[0];
          result = v62;
        }
        if ( v42 != &v147 )
        {
          v131 = result;
          _libc_free(v42, v39);
          result = v131;
        }
      }
      goto LABEL_112;
    }
    while ( **v7 == 17 )
    {
      v7 += 4;
      if ( (_BYTE **)a3 == v7 )
        goto LABEL_31;
    }
  }
  v8 = sub_BB52B0(a3);
  result = 0;
  if ( *(_QWORD *)(a2 + 72) != v8 )
    return result;
  v152 = v154;
  v153 = 0x800000000LL;
  if ( (*(_BYTE *)(a3 + 7) & 0x40) != 0 )
    v10 = *(_QWORD *)(a3 - 8);
  else
    v10 = a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
  v11 = (unsigned __int8 **)(v10 + 32);
  v12 = a3;
  v13 = sub_BB5290(a3) & 0xFFFFFFFFFFFFFFF9LL | 4;
  if ( (*(_BYTE *)(a3 + 7) & 0x40) != 0 )
    v12 = *(_QWORD *)(a3 - 8) + 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
  if ( v11 == (unsigned __int8 **)v12 )
  {
LABEL_115:
    v63 = *(_BYTE **)(v4 + 32 * (1LL - (*(_DWORD *)(v4 + 4) & 0x7FFFFFF)));
    if ( *v63 <= 0x15u && sub_AC30F0((__int64)v63) )
    {
      v64 = (unsigned int)v153;
      v83 = *(_DWORD *)(a3 + 4) & 0x7FFFFFF;
      if ( (_DWORD)v83 != 1 )
      {
        v106 = (_QWORD *)(a3 + 32 * (1 - v83));
        v107 = (-32 * (1 - v83)) >> 5;
        if ( v107 + (unsigned __int64)(unsigned int)v153 > HIDWORD(v153) )
        {
          a2 = (__int64)v154;
          sub_C8D5F0((__int64)&v152, v154, v107 + (unsigned int)v153, 8u, v81, v82);
          v64 = (unsigned int)v153;
        }
        v108 = &v152[8 * v64];
        if ( (_QWORD *)a3 != v106 )
        {
          do
          {
            if ( v108 )
              *(_QWORD *)v108 = *v106;
            v106 += 4;
            v108 += 8;
          }
          while ( (_QWORD *)a3 != v106 );
          LODWORD(v64) = v153;
        }
        v109 = *(_DWORD *)(v4 + 4);
        v110 = v64 + v107;
        LODWORD(v153) = v110;
        v111 = (_QWORD *)(v4 + 32 * (2LL - (v109 & 0x7FFFFFF)));
        v103 = (-32 * (2LL - (v109 & 0x7FFFFFF))) >> 5;
        v100 = v110;
        v112 = v103 + v110;
        if ( v112 > HIDWORD(v153) )
        {
          a2 = (__int64)v154;
          sub_C8D5F0((__int64)&v152, v154, v103 + v110, 8u, v112, v82);
          v100 = (unsigned int)v153;
        }
        v20 = v152;
        v113 = &v152[8 * v100];
        if ( (_QWORD *)v4 != v111 )
        {
          do
          {
            if ( v113 )
              *(_QWORD *)v113 = *v111;
            v111 += 4;
            v113 += 8;
          }
          while ( (_QWORD *)v4 != v111 );
          goto LABEL_154;
        }
        goto LABEL_155;
      }
      v20 = v152;
    }
    else
    {
      v64 = (unsigned int)v153;
      v20 = v152;
    }
LABEL_117:
    result = 0;
    if ( (_DWORD)v64 )
    {
      v65 = (unsigned int **)a1[2].m128i_i64[0];
      v66 = (unsigned __int8)(*(_BYTE *)(a3 + 1) & *(_BYTE *)(v4 + 1)) >> 1;
      v67 = v66;
      if ( (*(_BYTE *)(a3 + 1) & *(_BYTE *)(v4 + 1) & 2) == 0 && (v66 & 2) != 0 )
        v67 = v66 & 0xFC;
      v125 = v67;
      LOWORD(v148) = 257;
      v133 = *(_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
      v68 = sub_BB5290(a3);
      v69 = sub_921130(v65, v68, v133, (_BYTE **)v20, v64, (__int64)&v146, v125);
      a2 = v4;
      result = sub_F162A0((__int64)a1, v4, v69);
      v20 = v152;
    }
    goto LABEL_27;
  }
  do
  {
    while ( 1 )
    {
      v14 = v13 & 0xFFFFFFFFFFFFFFF8LL;
      v15 = (v13 >> 1) & 3;
      v16 = v13 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v13 )
        goto LABEL_53;
      if ( v15 == 2 )
      {
        if ( v14 )
          goto LABEL_21;
LABEL_53:
        a2 = (__int64)*v11;
        v16 = sub_BCBAE0(v14, *v11, v16);
        goto LABEL_21;
      }
      if ( v15 != 1 || !v14 )
        goto LABEL_53;
      v16 = *(_QWORD *)(v14 + 24);
LABEL_21:
      v17 = *(_BYTE *)(v16 + 8);
      if ( v17 != 16 )
        break;
      v13 = *(_QWORD *)(v16 + 24) & 0xFFFFFFFFFFFFFFF9LL | 4;
LABEL_17:
      v11 += 4;
      if ( (unsigned __int8 **)v12 == v11 )
        goto LABEL_24;
    }
    if ( (unsigned int)v17 - 17 > 1 )
    {
      v34 = v16 & 0xFFFFFFFFFFFFFFF9LL;
      v35 = v17 == 15;
      v13 = 0;
      if ( v35 )
        v13 = v34;
      goto LABEL_17;
    }
    v11 += 4;
    v13 = v16 & 0xFFFFFFFFFFFFFFF9LL | 2;
  }
  while ( (unsigned __int8 **)v12 != v11 );
LABEL_24:
  if ( !v15 )
    goto LABEL_115;
  v18 = *(_QWORD *)(v4 + 32 * (1LL - (*(_DWORD *)(v4 + 4) & 0x7FFFFFF)));
  v19 = *(_QWORD *)(a3
                  + 32 * ((*(_DWORD *)(a3 + 4) & 0x7FFFFFFu) - 1 - (unsigned __int64)(*(_DWORD *)(a3 + 4) & 0x7FFFFFF)));
  if ( *(_QWORD *)(v19 + 8) != *(_QWORD *)(v18 + 8) )
  {
    v20 = v152;
    result = 0;
    goto LABEL_27;
  }
  a2 = *(_QWORD *)(a3
                 + 32 * ((*(_DWORD *)(a3 + 4) & 0x7FFFFFFu) - 1 - (unsigned __int64)(*(_DWORD *)(a3 + 4) & 0x7FFFFFF)));
  v84 = *(_QWORD *)(v4 + 32 * (1LL - (*(_DWORD *)(v4 + 4) & 0x7FFFFFF)));
  v85 = _mm_loadu_si128(a1 + 9);
  v86 = _mm_loadu_si128(a1 + 7);
  v87 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
  v88 = a1[10].m128i_i64[0];
  v146 = _mm_loadu_si128(a1 + 6);
  v148 = v87;
  v151 = v88;
  v149 = v4;
  v147 = v86;
  v150 = v85;
  result = (unsigned __int8 *)sub_101BE10(v84, a2, 0, 0, &v146);
  if ( result )
    goto LABEL_139;
  v105 = *(_QWORD *)(a3 + 16);
  if ( !v105 || *(_QWORD *)(v105 + 8) || !(_BYTE)qword_4F8B548 )
  {
    v20 = v152;
    goto LABEL_27;
  }
  v138 = a1[2].m128i_i64[0];
  v144[0] = sub_BD5D20(a3);
  v144[1] = v114;
  v145 = 773;
  v144[2] = ".sum";
  result = (unsigned __int8 *)(*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(v138 + 80) + 32LL))(
                                *(_QWORD *)(v138 + 80),
                                13,
                                v19,
                                v18,
                                0,
                                0);
  if ( result )
    goto LABEL_139;
  LOWORD(v148) = 257;
  v115 = sub_B504D0(13, v19, v18, (__int64)&v146, 0, 0);
  v116 = (unsigned int **)v138;
  v139 = v115;
  a2 = v115;
  v127 = v116;
  (*(void (__fastcall **)(unsigned int *, __int64, _QWORD *, unsigned int *, unsigned int *))(*(_QWORD *)v116[11] + 16LL))(
    v116[11],
    v115,
    v144,
    v116[7],
    v116[8]);
  result = (unsigned __int8 *)v139;
  v117 = *v127;
  v118 = (__int64)&(*v127)[4 * *((unsigned int *)v127 + 2)];
  if ( *v127 != (unsigned int *)v118 )
  {
    do
    {
      v119 = *((_QWORD *)v117 + 1);
      a2 = *v117;
      v117 += 4;
      sub_B99FD0(v139, a2, v119);
    }
    while ( (unsigned int *)v118 != v117 );
    result = (unsigned __int8 *)v139;
  }
  v20 = v152;
  if ( result )
  {
LABEL_139:
    a2 = HIDWORD(v153);
    v91 = (_QWORD *)(a3 - 32);
    v92 = (unsigned int)v153;
    v93 = 32 * (1LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
    v94 = (_QWORD *)(a3 + v93);
    v95 = (-32 - v93) >> 5;
    v96 = v95 + (unsigned int)v153;
    if ( v96 > HIDWORD(v153) )
    {
      a2 = (__int64)v154;
      v137 = result;
      sub_C8D5F0((__int64)&v152, v154, v96, 8u, v89, v90);
      v92 = (unsigned int)v153;
      result = v137;
    }
    v97 = &v152[8 * v92];
    if ( v91 != v94 )
    {
      do
      {
        if ( v97 )
          *(_QWORD *)v97 = *v94;
        v94 += 4;
        v97 += 8;
      }
      while ( v91 != v94 );
      LODWORD(v92) = v153;
    }
    LODWORD(v153) = v95 + v92;
    v98 = (unsigned int)(v95 + v92);
    v99 = (unsigned int)v98 + 1LL;
    if ( v99 > HIDWORD(v153) )
    {
      a2 = (__int64)v154;
      v140 = result;
      sub_C8D5F0((__int64)&v152, v154, (unsigned int)v98 + 1LL, 8u, v99, v90);
      v98 = (unsigned int)v153;
      result = v140;
    }
    *(_QWORD *)&v152[8 * v98] = result;
    v101 = *(_DWORD *)(v4 + 4);
    LODWORD(v153) = v153 + 1;
    v100 = (unsigned int)v153;
    v102 = (_QWORD *)(v4 + 64 - 32LL * (v101 & 0x7FFFFFF));
    v103 = (32LL * (v101 & 0x7FFFFFF) - 64) >> 5;
    if ( v103 + (unsigned __int64)(unsigned int)v153 > HIDWORD(v153) )
    {
      a2 = (__int64)v154;
      sub_C8D5F0((__int64)&v152, v154, v103 + (unsigned int)v153, 8u, v103 + (unsigned int)v153, v90);
      v100 = (unsigned int)v153;
    }
    v20 = v152;
    v104 = &v152[8 * v100];
    if ( (_QWORD *)v4 != v102 )
    {
      do
      {
        if ( v104 )
          *(_QWORD *)v104 = *v102;
        v102 += 4;
        v104 += 8;
      }
      while ( (_QWORD *)v4 != v102 );
LABEL_154:
      LODWORD(v100) = v153;
      v20 = v152;
    }
LABEL_155:
    LODWORD(v153) = v103 + v100;
    v64 = (unsigned int)(v103 + v100);
    goto LABEL_117;
  }
LABEL_27:
  if ( v20 != v154 )
  {
    v129 = result;
    _libc_free(v20, a2);
    return v129;
  }
  return result;
}
