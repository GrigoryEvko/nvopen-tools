// Function: sub_1C32740
// Address: 0x1c32740
//
__int64 __fastcall sub_1C32740(__int64 a1, __int64 a2)
{
  unsigned __int8 v4; // bl
  unsigned __int8 v5; // r14
  char v6; // al
  const char *v7; // rsi
  char v8; // r15
  char v9; // al
  _DWORD *v10; // r15
  _DWORD *v11; // rdx
  _DWORD *v12; // rbx
  int v13; // esi
  __int64 *v14; // rax
  __m128i *v15; // rdx
  __int64 v16; // rdi
  __m128i v17; // xmm0
  void *v18; // rdx
  int v19; // eax
  __int64 *v20; // rax
  __m128i *v21; // rdx
  __int64 v22; // rdi
  __m128i v23; // xmm0
  __int64 v24; // rdx
  __m128i v25; // xmm0
  __int64 *v26; // rax
  __m128i *v27; // rdx
  __m128i v28; // xmm0
  __int64 *v29; // rax
  __m128i *v30; // rdx
  __m128i v31; // xmm0
  __int64 *v32; // rax
  __m128i *v33; // rdx
  __m128i v34; // xmm0
  int v35; // ecx
  __int16 v36; // ax
  __int64 result; // rax
  __int64 *v38; // rax
  __m128i *v39; // rdx
  __m128i v40; // xmm0
  __int64 v41; // rbx
  __int64 v42; // rdi
  __int64 v43; // rdx
  __int64 v44; // rdi
  __m128i *v45; // rdx
  __m128i v46; // xmm0
  __m128i v47; // xmm0
  __int64 *v48; // rax
  __m128i *v49; // rdx
  __int64 *v50; // rax
  __m128i *v51; // rdx
  __m128i v52; // xmm0
  __int64 *v53; // rax
  __m128i *v54; // rdx
  __int64 *v55; // rax
  __m128i *v56; // rdx
  __m128i v57; // xmm0
  __int64 *v58; // rax
  __m128i *v59; // rdx
  __m128i v60; // xmm0
  __int64 *v61; // rax
  __m128i *v62; // rdx
  __m128i v63; // xmm0
  __int64 *v64; // rax
  __m128i *v65; // rdx
  __m128i v66; // xmm0
  __int64 *v67; // rax
  __m128i *v68; // rdx
  __m128i v69; // xmm0
  __int64 *v70; // rax
  __m128i *v71; // rdx
  __m128i v72; // xmm0
  __int64 *v73; // rax
  __m128i *v74; // rdx
  __m128i v75; // xmm0
  __int64 *v76; // rax
  __m128i *v77; // rdx
  __m128i v78; // xmm0
  __int64 *v79; // rax
  __m128i *v80; // rdx
  __m128i v81; // xmm0
  __int64 *v82; // rax
  __m128i *v83; // rdx
  __m128i v84; // xmm0
  __int64 *v85; // rax
  __m128i *v86; // rdx
  __m128i v87; // xmm0
  __int64 *v88; // rax
  __m128i *v89; // rdx
  __m128i v90; // xmm0
  __int64 *v91; // rax
  __m128i *v92; // rdx
  __m128i v93; // xmm0
  __int64 *v94; // rax
  __m128i *v95; // rdx
  __m128i v96; // xmm0
  __int64 *v97; // rax
  __m128i *v98; // rdx
  __m128i v99; // xmm0
  __int64 *v100; // rax
  __m128i *v101; // rdx
  __int64 *v102; // rax
  __m128i *v103; // rdx
  __m128i v104; // xmm0
  __int64 *v105; // rax
  __m128i *v106; // rdx
  __m128i v107; // xmm0
  __int64 *v108; // rax
  __m128i *v109; // rdx
  __m128i v110; // xmm0
  __int64 *v111; // rax
  __m128i *v112; // rdx
  __m128i v113; // xmm0
  __int64 *v114; // rax
  __m128i *v115; // rdx
  __m128i v116; // xmm0
  __int64 *v117; // rax
  __m128i *v118; // rdx
  __int64 v119; // r15
  __m128i v120; // xmm0
  __m128i *v121; // rdi
  __int64 *v122; // rax
  __m128i *v123; // rdx
  __m128i v124; // xmm0
  __int64 *v125; // rax
  __int64 v126; // rax
  __int64 *v127; // rax
  __m128i *v128; // rdx
  __m128i si128; // xmm0
  __int64 v130; // rdi
  __m128i *v131; // rdx
  void *v132; // rax
  size_t v133; // rdx
  __int64 v134; // rax
  __int64 *v135; // rax
  __m128i *v136; // rdx
  __m128i v137; // xmm0
  __m128i *v138; // rdx
  __int64 v139; // rax
  size_t v140; // [rsp+0h] [rbp-60h]
  char v141; // [rsp+8h] [rbp-58h]
  __int64 v142; // [rsp+8h] [rbp-58h]
  int v143; // [rsp+10h] [rbp-50h] BYREF
  int v144; // [rsp+14h] [rbp-4Ch] BYREF
  unsigned int v145; // [rsp+18h] [rbp-48h] BYREF
  int v146; // [rsp+1Ch] [rbp-44h] BYREF
  __int64 v147; // [rsp+20h] [rbp-40h] BYREF
  _QWORD v148[7]; // [rsp+28h] [rbp-38h] BYREF

  v4 = sub_1C2EEF0(a2, &v143);
  v5 = sub_1C2EF10(a2, &v144);
  v6 = sub_1C2EF30(a2, &v145);
  v7 = (const char *)&v146;
  v8 = v6;
  v9 = sub_1C2EF50(a2, &v146);
  v141 = v9;
  if ( !(v5 | v4) && !v8 && !v9 )
  {
    sub_1C2F070(a2);
    goto LABEL_5;
  }
  v19 = *(_DWORD *)(a1 + 8);
  if ( v19 && v19 <= 899 )
  {
    v7 = (const char *)a2;
    v127 = sub_1C31B60(a1, a2, 2u);
    v128 = (__m128i *)v127[3];
    if ( (unsigned __int64)(v127[2] - (_QWORD)v128) <= 0x5B )
    {
      v7 = "Cluster dimensions and cluster maximum blocks are not supported on pre-Hopper Architectures\n";
      sub_16E7EE0(
        (__int64)v127,
        "Cluster dimensions and cluster maximum blocks are not supported on pre-Hopper Architectures\n",
        0x5Cu);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_42D0540);
      qmemcpy(&v128[5], "chitectures\n", 12);
      *v128 = si128;
      v128[1] = _mm_load_si128((const __m128i *)&xmmword_42D0550);
      v128[2] = _mm_load_si128((const __m128i *)&xmmword_42D0560);
      v128[3] = _mm_load_si128((const __m128i *)&xmmword_42D0570);
      v128[4] = _mm_load_si128((const __m128i *)&xmmword_42D0580);
      v127[3] += 92;
    }
  }
  if ( !(unsigned __int8)sub_1C2F070(a2) )
  {
    if ( !(v5 | v4) && !v8 && !v141 )
      goto LABEL_5;
    v7 = (const char *)a2;
    v29 = sub_1C31B60(a1, a2, 2u);
    v30 = (__m128i *)v29[3];
    if ( (unsigned __int64)(v29[2] - (_QWORD)v30) <= 0x53 )
    {
      v7 = "Cluster dimensions and cluster maximum blocks are only allowed for kernel functions\n";
      sub_16E7EE0(
        (__int64)v29,
        "Cluster dimensions and cluster maximum blocks are only allowed for kernel functions\n",
        0x54u);
    }
    else
    {
      v31 = _mm_load_si128((const __m128i *)&xmmword_42D0540);
      v30[5].m128i_i32[0] = 175337071;
      *v30 = v31;
      v30[1] = _mm_load_si128((const __m128i *)&xmmword_42D0550);
      v30[2] = _mm_load_si128((const __m128i *)&xmmword_42D0560);
      v30[3] = _mm_load_si128((const __m128i *)&xmmword_42D0590);
      v30[4] = _mm_load_si128((const __m128i *)&xmmword_42D05A0);
      v29[3] += 84;
    }
  }
  if ( v4 && v143 )
    v4 = 0;
  if ( v5 && !v144 )
  {
    if ( v8 && !v145 && v4 )
      goto LABEL_39;
    goto LABEL_32;
  }
  if ( v8 && (v7 = (const char *)v145, !v145) || v4 )
  {
LABEL_32:
    v7 = (const char *)a2;
    v20 = sub_1C31B60(a1, a2, 2u);
    v21 = (__m128i *)v20[3];
    v22 = (__int64)v20;
    if ( (unsigned __int64)(v20[2] - (_QWORD)v21) <= 0x2A )
    {
      v7 = "If any cluster dimension is specified as 0 ";
      v134 = sub_16E7EE0((__int64)v20, "If any cluster dimension is specified as 0 ", 0x2Bu);
      v24 = *(_QWORD *)(v134 + 24);
      v22 = v134;
    }
    else
    {
      v23 = _mm_load_si128((const __m128i *)&xmmword_42D05B0);
      qmemcpy(&v21[2], "ified as 0 ", 11);
      *v21 = v23;
      v21[1] = _mm_load_si128((const __m128i *)&xmmword_42D05C0);
      v24 = v20[3] + 43;
      v20[3] = v24;
    }
    if ( (unsigned __int64)(*(_QWORD *)(v22 + 16) - v24) <= 0x30 )
    {
      v7 = "then all other dimensions must be specified as 0\n";
      sub_16E7EE0(v22, "then all other dimensions must be specified as 0\n", 0x31u);
    }
    else
    {
      v25 = _mm_load_si128((const __m128i *)&xmmword_42D05D0);
      *(_BYTE *)(v24 + 48) = 10;
      *(__m128i *)v24 = v25;
      *(__m128i *)(v24 + 16) = _mm_load_si128((const __m128i *)&xmmword_42D05E0);
      *(__m128i *)(v24 + 32) = _mm_load_si128((const __m128i *)&xmmword_42D05F0);
      *(_QWORD *)(v22 + 24) += 49LL;
    }
  }
LABEL_39:
  if ( v141 && !v146 )
  {
    v7 = (const char *)a2;
    v26 = sub_1C31B60(a1, a2, 2u);
    v27 = (__m128i *)v26[3];
    if ( (unsigned __int64)(v26[2] - (_QWORD)v27) <= 0x27 )
    {
      v7 = "Cluster maximum blocks must be non-zero\n";
      sub_16E7EE0((__int64)v26, "Cluster maximum blocks must be non-zero\n", 0x28u);
    }
    else
    {
      v28 = _mm_load_si128((const __m128i *)&xmmword_42D0600);
      v27[2].m128i_i64[0] = 0xA6F72657A2D6E6FLL;
      *v27 = v28;
      v27[1] = _mm_load_si128((const __m128i *)&xmmword_42D0610);
      v26[3] += 40;
    }
  }
LABEL_5:
  v142 = **(_QWORD **)(*(_QWORD *)(a2 + 24) + 16LL);
  if ( (*(_BYTE *)(a2 + 33) & 0x20) != 0 )
    goto LABEL_51;
  if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
  {
    sub_15E08E0(a2, (__int64)v7);
    v10 = *(_DWORD **)(a2 + 88);
    if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
      sub_15E08E0(a2, (__int64)v7);
    v11 = *(_DWORD **)(a2 + 88);
  }
  else
  {
    v10 = *(_DWORD **)(a2 + 88);
    v11 = v10;
  }
  v12 = &v11[10 * *(_QWORD *)(a2 + 96)];
  if ( v12 != v10 )
  {
    while ( 1 )
    {
      if ( *(_BYTE *)(*(_QWORD *)v10 + 8LL) == 11 && (unsigned int)sub_1643030(*(_QWORD *)v10) <= 0x1F )
      {
        v13 = v10[8] + 1;
        v147 = *(_QWORD *)(a2 + 112);
        if ( !(unsigned __int8)sub_1560260(&v147, v13, 40) )
        {
          v148[0] = *(_QWORD *)(a2 + 112);
          if ( !(unsigned __int8)sub_1560260(v148, v10[8] + 1, 58) )
          {
            v14 = sub_1C31B60(a1, a2, 2u);
            v15 = (__m128i *)v14[3];
            v16 = (__int64)v14;
            if ( (unsigned __int64)(v14[2] - (_QWORD)v15) <= 0x2B )
            {
              v139 = sub_16E7EE0((__int64)v14, "Integer parameter less than 32-bits without ", 0x2Cu);
              v18 = *(void **)(v139 + 24);
              v16 = v139;
            }
            else
            {
              v17 = _mm_load_si128((const __m128i *)&xmmword_42D0620);
              qmemcpy(&v15[2], "its without ", 12);
              *v15 = v17;
              v15[1] = _mm_load_si128((const __m128i *)&xmmword_42D0630);
              v18 = (void *)(v14[3] + 44);
              v14[3] = (__int64)v18;
            }
            if ( *(_QWORD *)(v16 + 16) - (_QWORD)v18 <= 0xEu )
            {
              sub_16E7EE0(v16, "sext/zext flag\n", 0xFu);
            }
            else
            {
              qmemcpy(v18, "sext/zext flag\n", 15);
              *(_QWORD *)(v16 + 24) += 15LL;
            }
          }
        }
      }
      v148[0] = *(_QWORD *)(a2 + 112);
      if ( (unsigned __int8)sub_1560260(v148, v10[8] + 1, 12) )
      {
        v38 = sub_1C31B60(a1, a2, 1u);
        v39 = (__m128i *)v38[3];
        if ( (unsigned __int64)(v38[2] - (_QWORD)v39) <= 0x2C )
        {
          sub_16E7EE0((__int64)v38, "InReg attribute on parameter will be ignored\n", 0x2Du);
        }
        else
        {
          v40 = _mm_load_si128((const __m128i *)&xmmword_42D0640);
          qmemcpy(&v39[2], "l be ignored\n", 13);
          *v39 = v40;
          v39[1] = _mm_load_si128((const __m128i *)&xmmword_42D0650);
          v38[3] += 45;
        }
      }
      v148[0] = *(_QWORD *)(a2 + 112);
      if ( !(unsigned __int8)sub_1560260(v148, v10[8] + 1, 19) )
        goto LABEL_12;
      v32 = sub_1C31B60(a1, a2, 1u);
      v33 = (__m128i *)v32[3];
      if ( (unsigned __int64)(v32[2] - (_QWORD)v33) <= 0x2B )
      {
        sub_16E7EE0((__int64)v32, "Nest attribute on parameter will be ignored\n", 0x2Cu);
LABEL_12:
        v10 += 10;
        if ( v10 == v12 )
          break;
      }
      else
      {
        v34 = _mm_load_si128((const __m128i *)&xmmword_42D0660);
        v10 += 10;
        qmemcpy(&v33[2], " be ignored\n", 12);
        *v33 = v34;
        v33[1] = _mm_load_si128((const __m128i *)&xmmword_42D0670);
        v32[3] += 44;
        if ( v10 == v12 )
          break;
      }
    }
  }
  if ( *(_BYTE *)(v142 + 8) == 11 && (unsigned int)sub_1643030(v142) <= 0x1F )
  {
    v147 = *(_QWORD *)(a2 + 112);
    if ( !(unsigned __int8)sub_1560260(&v147, 0, 40) )
    {
      v148[0] = *(_QWORD *)(a2 + 112);
      if ( !(unsigned __int8)sub_1560260(v148, 0, 58) )
      {
        v125 = sub_1C31B60(a1, a2, 2u);
        v126 = sub_1263B40((__int64)v125, "Integer return less than 32-bits without ");
        sub_1263B40(v126, "sext/zext flag\n");
      }
    }
  }
LABEL_51:
  v35 = *(_DWORD *)(a2 + 32) >> 15;
  if ( (*(_DWORD *)(a2 + 32) & 0x200000) == 0 )
    goto LABEL_52;
  v117 = sub_1C31B60(a1, a2, 0);
  v118 = (__m128i *)v117[3];
  v119 = (__int64)v117;
  if ( (unsigned __int64)(v117[2] - (_QWORD)v118) <= 0x17 )
  {
    v119 = sub_16E7EE0((__int64)v117, "Explicit section marker ", 0x18u);
  }
  else
  {
    v120 = _mm_load_si128((const __m128i *)&xmmword_42D0680);
    v118[1].m128i_i64[0] = 0x2072656B72616D20LL;
    *v118 = v120;
    v117[3] += 24;
  }
  if ( (*(_BYTE *)(a2 + 34) & 0x20) != 0 )
  {
    v132 = (void *)sub_15E61A0(a2);
    v121 = *(__m128i **)(v119 + 24);
    if ( *(_QWORD *)(v119 + 16) - (_QWORD)v121 >= v133 )
    {
      if ( v133 )
      {
        v140 = v133;
        memcpy(v121, v132, v133);
        v138 = (__m128i *)(*(_QWORD *)(v119 + 24) + v140);
        *(_QWORD *)(v119 + 24) = v138;
        v121 = v138;
      }
      goto LABEL_169;
    }
    v119 = sub_16E7EE0(v119, (char *)v132, v133);
  }
  v121 = *(__m128i **)(v119 + 24);
LABEL_169:
  if ( *(_QWORD *)(v119 + 16) - (_QWORD)v121 <= 0xFu )
  {
    sub_16E7EE0(v119, "is not allowed.\n", 0x10u);
  }
  else
  {
    *v121 = _mm_load_si128((const __m128i *)&xmmword_42D0690);
    *(_QWORD *)(v119 + 24) += 16LL;
  }
  sub_1C31880(a1);
  v35 = *(_DWORD *)(a2 + 32) >> 15;
LABEL_52:
  if ( (unsigned int)(1 << v35) >> 1 )
  {
    v114 = sub_1C31B60(a1, a2, 0);
    v115 = (__m128i *)v114[3];
    if ( (unsigned __int64)(v114[2] - (_QWORD)v115) <= 0x22 )
    {
      sub_16E7EE0((__int64)v114, "Explicit alignment is not allowed.\n", 0x23u);
    }
    else
    {
      v116 = _mm_load_si128((const __m128i *)&xmmword_42D06A0);
      v115[2].m128i_i8[2] = 10;
      v115[2].m128i_i16[0] = 11876;
      *v115 = v116;
      v115[1] = _mm_load_si128((const __m128i *)&xmmword_42D06B0);
      v114[3] += 35;
    }
    sub_1C31880(a1);
  }
  v36 = *(_WORD *)(a2 + 18);
  if ( (v36 & 2) != 0 )
  {
    v111 = sub_1C31B60(a1, a2, 0);
    v112 = (__m128i *)v111[3];
    if ( (unsigned __int64)(v111[2] - (_QWORD)v112) <= 0x1B )
    {
      sub_16E7EE0((__int64)v111, "Prefix data is not allowed.\n", 0x1Cu);
    }
    else
    {
      v113 = _mm_load_si128((const __m128i *)&xmmword_42D06C0);
      qmemcpy(&v112[1], "ot allowed.\n", 12);
      *v112 = v113;
      v111[3] += 28;
    }
    sub_1C31880(a1);
    v36 = *(_WORD *)(a2 + 18);
  }
  if ( (v36 & 4) != 0 )
  {
    v108 = sub_1C31B60(a1, a2, 0);
    v109 = (__m128i *)v108[3];
    if ( (unsigned __int64)(v108[2] - (_QWORD)v109) <= 0x1D )
    {
      sub_16E7EE0((__int64)v108, "Prologue data is not allowed.\n", 0x1Eu);
    }
    else
    {
      v110 = _mm_load_si128((const __m128i *)&xmmword_42D06D0);
      qmemcpy(&v109[1], " not allowed.\n", 14);
      *v109 = v110;
      v108[3] += 30;
    }
    sub_1C31880(a1);
    v36 = *(_WORD *)(a2 + 18);
  }
  if ( (v36 & 8) != 0 )
  {
    v105 = sub_1C31B60(a1, a2, 0);
    v106 = (__m128i *)v105[3];
    if ( (unsigned __int64)(v105[2] - (_QWORD)v106) <= 0x24 )
    {
      sub_16E7EE0((__int64)v105, "Personality function is not allowed.\n", 0x25u);
    }
    else
    {
      v107 = _mm_load_si128((const __m128i *)&xmmword_42D06E0);
      v106[2].m128i_i32[0] = 778331511;
      v106[2].m128i_i8[4] = 10;
      *v106 = v107;
      v106[1] = _mm_load_si128((const __m128i *)&xmmword_42D06F0);
      v105[3] += 37;
    }
    sub_1C31880(a1);
    v36 = *(_WORD *)(a2 + 18);
  }
  if ( (v36 & 0x4000) != 0 )
  {
    v102 = sub_1C31B60(a1, a2, 0);
    v103 = (__m128i *)v102[3];
    if ( (unsigned __int64)(v102[2] - (_QWORD)v103) <= 0x1B )
    {
      sub_16E7EE0((__int64)v102, "GC names are not supported.\n", 0x1Cu);
    }
    else
    {
      v104 = _mm_load_si128((const __m128i *)&xmmword_42D0700);
      qmemcpy(&v103[1], " supported.\n", 12);
      *v103 = v104;
      v102[3] += 28;
    }
    sub_1C31880(a1);
  }
  v148[0] = *(_QWORD *)(a2 + 112);
  if ( (unsigned __int8)sub_1560180((__int64)v148, 1) )
  {
    v100 = sub_1C31B60(a1, a2, 0);
    v101 = (__m128i *)v100[3];
    if ( (unsigned __int64)(v100[2] - (_QWORD)v101) <= 0x2F )
    {
      sub_16E7EE0((__int64)v100, "alignstack function attribute is not supported.\n", 0x30u);
    }
    else
    {
      *v101 = _mm_load_si128((const __m128i *)&xmmword_42D0710);
      v101[1] = _mm_load_si128((const __m128i *)&xmmword_42D0720);
      v101[2] = _mm_load_si128((const __m128i *)&xmmword_42D0730);
      v100[3] += 48;
    }
    sub_1C31880(a1);
  }
  v148[0] = *(_QWORD *)(a2 + 112);
  if ( (unsigned __int8)sub_1560180((__int64)v148, 31) )
  {
    v97 = sub_1C31B60(a1, a2, 0);
    v98 = (__m128i *)v97[3];
    if ( (unsigned __int64)(v97[2] - (_QWORD)v98) <= 0x30 )
    {
      sub_16E7EE0((__int64)v97, "nonlazybind function attribute is not supported.\n", 0x31u);
    }
    else
    {
      v99 = _mm_load_si128((const __m128i *)&xmmword_42D0740);
      v98[3].m128i_i8[0] = 10;
      *v98 = v99;
      v98[1] = _mm_load_si128((const __m128i *)&xmmword_42D0750);
      v98[2] = _mm_load_si128((const __m128i *)&xmmword_42D0760);
      v97[3] += 49;
    }
    sub_1C31880(a1);
  }
  v148[0] = *(_QWORD *)(a2 + 112);
  if ( (unsigned __int8)sub_1560180((__int64)v148, 18) )
  {
    v94 = sub_1C31B60(a1, a2, 0);
    v95 = (__m128i *)v94[3];
    if ( (unsigned __int64)(v94[2] - (_QWORD)v95) <= 0x2A )
    {
      sub_16E7EE0((__int64)v94, "naked function attribute is not supported.\n", 0x2Bu);
    }
    else
    {
      v96 = _mm_load_si128((const __m128i *)&xmmword_42D0770);
      qmemcpy(&v95[2], "supported.\n", 11);
      *v95 = v96;
      v95[1] = _mm_load_si128((const __m128i *)&xmmword_42D0780);
      v94[3] += 43;
    }
    sub_1C31880(a1);
  }
  v148[0] = *(_QWORD *)(a2 + 112);
  if ( (unsigned __int8)sub_1560180((__int64)v148, 25) )
  {
    v91 = sub_1C31B60(a1, a2, 0);
    v92 = (__m128i *)v91[3];
    if ( (unsigned __int64)(v91[2] - (_QWORD)v92) <= 0x34 )
    {
      sub_16E7EE0((__int64)v91, "noimplicitfloat function attribute is not supported.\n", 0x35u);
    }
    else
    {
      v93 = _mm_load_si128((const __m128i *)&xmmword_42D0790);
      v92[3].m128i_i32[0] = 778331508;
      v92[3].m128i_i8[4] = 10;
      *v92 = v93;
      v92[1] = _mm_load_si128((const __m128i *)&xmmword_42D07A0);
      v92[2] = _mm_load_si128((const __m128i *)&xmmword_42D07B0);
      v91[3] += 53;
    }
    sub_1C31880(a1);
  }
  v148[0] = *(_QWORD *)(a2 + 112);
  if ( (unsigned __int8)sub_1560180((__int64)v148, 28) )
  {
    v88 = sub_1C31B60(a1, a2, 0);
    v89 = (__m128i *)v88[3];
    if ( (unsigned __int64)(v88[2] - (_QWORD)v89) <= 0x2E )
    {
      sub_16E7EE0((__int64)v88, "noredzone function attribute is not supported.\n", 0x2Fu);
    }
    else
    {
      v90 = _mm_load_si128((const __m128i *)&xmmword_42D07C0);
      qmemcpy(&v89[2], "not supported.\n", 15);
      *v89 = v90;
      v89[1] = _mm_load_si128((const __m128i *)&xmmword_42D07D0);
      v88[3] += 47;
    }
    sub_1C31880(a1);
  }
  v148[0] = *(_QWORD *)(a2 + 112);
  if ( (unsigned __int8)sub_1560180((__int64)v148, 39) )
  {
    v85 = sub_1C31B60(a1, a2, 0);
    v86 = (__m128i *)v85[3];
    if ( (unsigned __int64)(v85[2] - (_QWORD)v86) <= 0x32 )
    {
      sub_16E7EE0((__int64)v85, "returns_twice function attribute is not supported.\n", 0x33u);
    }
    else
    {
      v87 = _mm_load_si128((const __m128i *)&xmmword_42D07E0);
      v86[3].m128i_i8[2] = 10;
      v86[3].m128i_i16[0] = 11876;
      *v86 = v87;
      v86[1] = _mm_load_si128((const __m128i *)&xmmword_42D07F0);
      v86[2] = _mm_load_si128((const __m128i *)&xmmword_42D0800);
      v85[3] += 51;
    }
    sub_1C31880(a1);
  }
  v148[0] = *(_QWORD *)(a2 + 112);
  if ( (unsigned __int8)sub_1560180((__int64)v148, 49) )
  {
    v82 = sub_1C31B60(a1, a2, 0);
    v83 = (__m128i *)v82[3];
    if ( (unsigned __int64)(v82[2] - (_QWORD)v83) <= 0x28 )
    {
      sub_16E7EE0((__int64)v82, "ssp function attribute is not supported.\n", 0x29u);
    }
    else
    {
      v84 = _mm_load_si128((const __m128i *)&xmmword_42D0810);
      v83[2].m128i_i8[8] = 10;
      v83[2].m128i_i64[0] = 0x2E646574726F7070LL;
      *v83 = v84;
      v83[1] = _mm_load_si128((const __m128i *)&xmmword_42D0820);
      v82[3] += 41;
    }
    sub_1C31880(a1);
  }
  v148[0] = *(_QWORD *)(a2 + 112);
  if ( (unsigned __int8)sub_1560180((__int64)v148, 50) )
  {
    v79 = sub_1C31B60(a1, a2, 0);
    v80 = (__m128i *)v79[3];
    if ( (unsigned __int64)(v79[2] - (_QWORD)v80) <= 0x2B )
    {
      sub_16E7EE0((__int64)v79, "sspreq function attribute is not supported.\n", 0x2Cu);
    }
    else
    {
      v81 = _mm_load_si128((const __m128i *)&xmmword_42D0830);
      qmemcpy(&v80[2], " supported.\n", 12);
      *v80 = v81;
      v80[1] = _mm_load_si128((const __m128i *)&xmmword_42D0840);
      v79[3] += 44;
    }
    sub_1C31880(a1);
  }
  v148[0] = *(_QWORD *)(a2 + 112);
  if ( (unsigned __int8)sub_1560180((__int64)v148, 51) )
  {
    v76 = sub_1C31B60(a1, a2, 0);
    v77 = (__m128i *)v76[3];
    if ( (unsigned __int64)(v76[2] - (_QWORD)v77) <= 0x2E )
    {
      sub_16E7EE0((__int64)v76, "sspstrong function attribute is not supported.\n", 0x2Fu);
    }
    else
    {
      v78 = _mm_load_si128((const __m128i *)&xmmword_42D0850);
      qmemcpy(&v77[2], "not supported.\n", 15);
      *v77 = v78;
      v77[1] = _mm_load_si128((const __m128i *)&xmmword_42D07D0);
      v76[3] += 47;
    }
    sub_1C31880(a1);
  }
  v148[0] = *(_QWORD *)(a2 + 112);
  if ( (unsigned __int8)sub_1560180((__int64)v148, 56) )
  {
    v73 = sub_1C31B60(a1, a2, 0);
    v74 = (__m128i *)v73[3];
    if ( (unsigned __int64)(v73[2] - (_QWORD)v74) <= 0x2C )
    {
      sub_16E7EE0((__int64)v73, "uwtable function attribute is not supported.\n", 0x2Du);
    }
    else
    {
      v75 = _mm_load_si128((const __m128i *)&xmmword_42D0860);
      qmemcpy(&v74[2], "t supported.\n", 13);
      *v74 = v75;
      v74[1] = _mm_load_si128((const __m128i *)&xmmword_42D0870);
      v73[3] += 45;
    }
    sub_1C31880(a1);
  }
  v148[0] = *(_QWORD *)(a2 + 112);
  if ( (unsigned __int8)sub_1560180((__int64)v148, 16) )
  {
    v70 = sub_1C31B60(a1, a2, 0);
    v71 = (__m128i *)v70[3];
    if ( (unsigned __int64)(v70[2] - (_QWORD)v71) <= 0x2E )
    {
      sub_16E7EE0((__int64)v70, "jumptable function attribute is not supported.\n", 0x2Fu);
    }
    else
    {
      v72 = _mm_load_si128((const __m128i *)&xmmword_42D0880);
      qmemcpy(&v71[2], "not supported.\n", 15);
      *v71 = v72;
      v71[1] = _mm_load_si128((const __m128i *)&xmmword_42D07D0);
      v70[3] += 47;
    }
    sub_1C31880(a1);
  }
  v148[0] = *(_QWORD *)(a2 + 112);
  if ( (unsigned __int8)sub_1560180((__int64)v148, 5) )
  {
    v67 = sub_1C31B60(a1, a2, 0);
    v68 = (__m128i *)v67[3];
    if ( (unsigned __int64)(v67[2] - (_QWORD)v68) <= 0x2C )
    {
      sub_16E7EE0((__int64)v67, "builtin function attribute is not supported.\n", 0x2Du);
    }
    else
    {
      v69 = _mm_load_si128((const __m128i *)&xmmword_42D0890);
      qmemcpy(&v68[2], "t supported.\n", 13);
      *v68 = v69;
      v68[1] = _mm_load_si128((const __m128i *)&xmmword_42D0870);
      v67[3] += 45;
    }
    sub_1C31880(a1);
  }
  v148[0] = *(_QWORD *)(a2 + 112);
  if ( (unsigned __int8)sub_1560180((__int64)v148, 21) )
  {
    v64 = sub_1C31B60(a1, a2, 0);
    v65 = (__m128i *)v64[3];
    if ( (unsigned __int64)(v64[2] - (_QWORD)v65) <= 0x2E )
    {
      sub_16E7EE0((__int64)v64, "nobuiltin function attribute is not supported.\n", 0x2Fu);
    }
    else
    {
      v66 = _mm_load_si128((const __m128i *)&xmmword_42D08A0);
      qmemcpy(&v65[2], "not supported.\n", 15);
      *v65 = v66;
      v65[1] = _mm_load_si128((const __m128i *)&xmmword_42D07D0);
      v64[3] += 47;
    }
    sub_1C31880(a1);
  }
  v148[0] = *(_QWORD *)(a2 + 112);
  if ( (unsigned __int8)sub_1560180((__int64)v148, 42) )
  {
    v61 = sub_1C31B60(a1, a2, 0);
    v62 = (__m128i *)v61[3];
    if ( (unsigned __int64)(v61[2] - (_QWORD)v62) <= 0x35 )
    {
      sub_16E7EE0((__int64)v61, "sanitize_address function attribute is not supported.\n", 0x36u);
    }
    else
    {
      v63 = _mm_load_si128((const __m128i *)&xmmword_4293160);
      v62[3].m128i_i32[0] = 1684370546;
      v62[3].m128i_i16[2] = 2606;
      *v62 = v63;
      v62[1] = _mm_load_si128((const __m128i *)&xmmword_42D08B0);
      v62[2] = _mm_load_si128((const __m128i *)&xmmword_42D08C0);
      v61[3] += 54;
    }
    sub_1C31880(a1);
  }
  v148[0] = *(_QWORD *)(a2 + 112);
  if ( (unsigned __int8)sub_1560180((__int64)v148, 44) )
  {
    v58 = sub_1C31B60(a1, a2, 0);
    v59 = (__m128i *)v58[3];
    if ( (unsigned __int64)(v58[2] - (_QWORD)v59) <= 0x34 )
    {
      sub_16E7EE0((__int64)v58, "sanitize_memory function attribute is not supported.\n", 0x35u);
    }
    else
    {
      v60 = _mm_load_si128((const __m128i *)&xmmword_42D08D0);
      v59[3].m128i_i32[0] = 778331508;
      v59[3].m128i_i8[4] = 10;
      *v59 = v60;
      v59[1] = _mm_load_si128((const __m128i *)&xmmword_42D07A0);
      v59[2] = _mm_load_si128((const __m128i *)&xmmword_42D07B0);
      v58[3] += 53;
    }
    sub_1C31880(a1);
  }
  v148[0] = *(_QWORD *)(a2 + 112);
  if ( (unsigned __int8)sub_1560180((__int64)v148, 45) )
  {
    v55 = sub_1C31B60(a1, a2, 0);
    v56 = (__m128i *)v55[3];
    if ( (unsigned __int64)(v55[2] - (_QWORD)v56) <= 0x34 )
    {
      sub_16E7EE0((__int64)v55, "sanitize_thread function attribute is not supported.\n", 0x35u);
    }
    else
    {
      v57 = _mm_load_si128((const __m128i *)&xmmword_42D08E0);
      v56[3].m128i_i32[0] = 778331508;
      v56[3].m128i_i8[4] = 10;
      *v56 = v57;
      v56[1] = _mm_load_si128((const __m128i *)&xmmword_42D07A0);
      v56[2] = _mm_load_si128((const __m128i *)&xmmword_42D07B0);
      v55[3] += 53;
    }
    sub_1C31880(a1);
  }
  v148[0] = *(_QWORD *)(a2 + 112);
  if ( (unsigned __int8)sub_1560180((__int64)v148, 48) )
  {
    v53 = sub_1C31B60(a1, a2, 0);
    v54 = (__m128i *)v53[3];
    if ( (unsigned __int64)(v53[2] - (_QWORD)v54) <= 0x2F )
    {
      sub_16E7EE0((__int64)v53, "alignstack function attribute is not supported.\n", 0x30u);
    }
    else
    {
      *v54 = _mm_load_si128((const __m128i *)&xmmword_42D0710);
      v54[1] = _mm_load_si128((const __m128i *)&xmmword_42D0720);
      v54[2] = _mm_load_si128((const __m128i *)&xmmword_42D0730);
      v53[3] += 48;
    }
    sub_1C31880(a1);
  }
  v148[0] = *(_QWORD *)(a2 + 112);
  if ( (unsigned __int8)sub_1560180((__int64)v148, 41) )
  {
    v50 = sub_1C31B60(a1, a2, 0);
    v51 = (__m128i *)v50[3];
    if ( (unsigned __int64)(v50[2] - (_QWORD)v51) <= 0x2E )
    {
      sub_16E7EE0((__int64)v50, "safestack function attribute is not supported.\n", 0x2Fu);
    }
    else
    {
      v52 = _mm_load_si128((const __m128i *)&xmmword_42D08F0);
      qmemcpy(&v51[2], "not supported.\n", 15);
      *v51 = v52;
      v51[1] = _mm_load_si128((const __m128i *)&xmmword_42D07D0);
      v50[3] += 47;
    }
    sub_1C31880(a1);
  }
  if ( (unsigned __int8)sub_1C2F070(a2) )
  {
    if ( !*(_BYTE *)(v142 + 8) )
      goto LABEL_77;
    v122 = sub_1C31B60(a1, a2, 0);
    v123 = (__m128i *)v122[3];
    if ( (unsigned __int64)(v122[2] - (_QWORD)v123) <= 0x18 )
    {
      sub_16E7EE0((__int64)v122, "non-void entry function.\n", 0x19u);
    }
    else
    {
      v124 = _mm_load_si128((const __m128i *)&xmmword_42D0900);
      v123[1].m128i_i8[8] = 10;
      v123[1].m128i_i64[0] = 0x2E6E6F6974636E75LL;
      *v123 = v124;
      v122[3] += 25;
    }
LABEL_184:
    sub_1C31880(a1);
    goto LABEL_77;
  }
  if ( (unsigned __int8)sub_1C2FA80(a2)
    || (unsigned __int8)sub_1C2FAB0(a2)
    || (unsigned __int8)sub_1C2FAE0(a2)
    || (unsigned __int8)sub_1C2FB10(a2)
    || (unsigned __int8)sub_1C2FB40(a2) )
  {
    if ( *(_BYTE *)(v142 + 8) )
    {
      v135 = sub_1C31B60(a1, a2, 0);
      v136 = (__m128i *)v135[3];
      if ( (unsigned __int64)(v135[2] - (_QWORD)v136) <= 0x18 )
      {
        sub_16E7EE0((__int64)v135, "non-void entry function.\n", 0x19u);
      }
      else
      {
        v137 = _mm_load_si128((const __m128i *)&xmmword_42D0900);
        v136[1].m128i_i8[8] = 10;
        v136[1].m128i_i64[0] = 0x2E6E6F6974636E75LL;
        *v136 = v137;
        v135[3] += 25;
      }
      sub_1C31880(a1);
    }
    if ( *(_DWORD *)(*(_QWORD *)(a2 + 24) + 12LL) != 1 )
    {
      v48 = sub_1C31B60(a1, a2, 0);
      v49 = (__m128i *)v48[3];
      if ( (unsigned __int64)(v48[2] - (_QWORD)v49) <= 0x1F )
      {
        sub_16E7EE0((__int64)v48, "entry function with parameters.\n", 0x20u);
      }
      else
      {
        *v49 = _mm_load_si128((const __m128i *)&xmmword_42D0910);
        v49[1] = _mm_load_si128((const __m128i *)&xmmword_42D0920);
        v48[3] += 32;
      }
      goto LABEL_184;
    }
  }
LABEL_77:
  result = sub_1C2FBD0(a2);
  if ( (_BYTE)result )
  {
    result = sub_1C2FC50(a2);
    v41 = result;
    if ( result )
    {
      result = *(_QWORD *)(result + 24);
      if ( *(_BYTE *)(**(_QWORD **)(result + 16) + 8LL) )
      {
        sub_1263B40(*(_QWORD *)(a1 + 24), "Error: ");
        v130 = *(_QWORD *)(a1 + 24);
        v131 = *(__m128i **)(v130 + 24);
        if ( *(_QWORD *)(v130 + 16) - (_QWORD)v131 <= 0x1Fu )
        {
          sub_16E7EE0(v130, "non-void exit handler function.\n", 0x20u);
        }
        else
        {
          *v131 = _mm_load_si128((const __m128i *)&xmmword_42D0930);
          v131[1] = _mm_load_si128((const __m128i *)&xmmword_42D0940);
          *(_QWORD *)(v130 + 24) += 32LL;
        }
        sub_1C31880(a1);
        result = *(_QWORD *)(v41 + 24);
      }
      if ( *(_DWORD *)(result + 12) != 1 )
      {
        v42 = *(_QWORD *)(a1 + 24);
        v43 = *(_QWORD *)(v42 + 24);
        if ( (unsigned __int64)(*(_QWORD *)(v42 + 16) - v43) <= 6 )
        {
          sub_16E7EE0(v42, "Error: ", 7u);
        }
        else
        {
          *(_DWORD *)v43 = 1869771333;
          *(_WORD *)(v43 + 4) = 14962;
          *(_BYTE *)(v43 + 6) = 32;
          *(_QWORD *)(v42 + 24) += 7LL;
        }
        v44 = *(_QWORD *)(a1 + 24);
        v45 = *(__m128i **)(v44 + 24);
        if ( *(_QWORD *)(v44 + 16) - (_QWORD)v45 <= 0x26u )
        {
          sub_16E7EE0(v44, "exit handler function with parameters.\n", 0x27u);
        }
        else
        {
          v46 = _mm_load_si128((const __m128i *)&xmmword_42D0950);
          v45[2].m128i_i32[0] = 1919251557;
          v45[2].m128i_i16[2] = 11891;
          *v45 = v46;
          v47 = _mm_load_si128((const __m128i *)&xmmword_42D0960);
          v45[2].m128i_i8[6] = 10;
          v45[1] = v47;
          *(_QWORD *)(v44 + 24) += 39LL;
        }
        return sub_1C31880(a1);
      }
    }
  }
  return result;
}
