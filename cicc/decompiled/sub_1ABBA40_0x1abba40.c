// Function: sub_1ABBA40
// Address: 0x1abba40
//
__int64 __fastcall sub_1ABBA40(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        _QWORD **a8)
{
  __int64 v8; // r15
  __int64 v9; // r13
  unsigned int v10; // eax
  _QWORD *v11; // rax
  __int64 **v12; // rbx
  __int64 **v13; // r12
  _BYTE *v14; // rdx
  _BYTE *v15; // rsi
  __int64 **v16; // r14
  __int64 **v17; // r12
  __int64 v18; // rax
  __int64 ***v19; // rbx
  char v20; // al
  __int64 ***v21; // r12
  _BYTE *v22; // rsi
  __int64 *v23; // rdi
  __int64 v24; // rax
  _BYTE *v25; // rdx
  unsigned __int8 v26; // cl
  __int64 v27; // r12
  __int64 v28; // rdx
  char v29; // al
  __int64 v30; // rdx
  const char **v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rbx
  __int64 v34; // r12
  __int64 *v35; // r13
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rsi
  __int64 v39; // rcx
  __int64 v40; // rax
  __int64 v41; // r14
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // r15
  __int64 v45; // rax
  __int64 v46; // r12
  __int64 v47; // r12
  __int64 v48; // rbx
  __int64 *v49; // r15
  __int64 v50; // rdi
  int v51; // eax
  __int64 v52; // rcx
  int v53; // eax
  __int64 v54; // r9
  int v55; // r10d
  unsigned int v56; // edx
  __int64 v57; // rbx
  __int64 v58; // rax
  __int64 v59; // r14
  __int64 v60; // r12
  __int64 v61; // r13
  __int64 v62; // r14
  __int64 v63; // rdi
  __int64 v64; // rdx
  __int64 v65; // rax
  __int64 v66; // rdx
  __int64 v67; // r14
  __int64 v68; // r12
  const char *v69; // rax
  __int64 v70; // rdi
  __int64 v71; // rdx
  __int64 v72; // rbx
  __int64 v73; // rax
  __int64 v74; // r12
  __int64 v75; // r12
  __int64 v76; // r15
  __int64 v77; // r14
  __int64 v78; // r14
  __int64 *v79; // rbx
  __int64 v80; // r14
  int v81; // eax
  __int64 v82; // rsi
  unsigned int v83; // ecx
  __int64 v84; // r8
  __int64 v85; // rdi
  int v86; // eax
  __int64 v87; // rdx
  _QWORD *v89; // rax
  __int64 **v90; // rax
  __int64 v91; // rdx
  __int64 v92; // rcx
  _QWORD *v93; // rax
  __int64 v94; // rax
  __int64 v95; // rdi
  __int64 v96; // rdx
  __int64 v97; // r15
  _QWORD *v98; // rax
  __int64 v99; // r12
  _QWORD *v100; // rcx
  __int64 v101; // rax
  __int64 *v102; // rax
  __int64 v103; // rax
  __int64 v104; // rcx
  __int64 *v105; // r10
  __int64 v106; // rax
  __int64 v107; // rdx
  __int64 *v108; // rax
  __int64 v109; // rax
  _QWORD *v110; // rax
  _QWORD *v111; // rax
  __int64 v112; // rax
  _BYTE *v113; // rsi
  __m128i v114; // xmm0
  int v115; // r9d
  __int64 *v117; // [rsp+8h] [rbp-138h]
  __int64 v120; // [rsp+20h] [rbp-120h]
  __int64 v121; // [rsp+28h] [rbp-118h]
  __int64 v122; // [rsp+30h] [rbp-110h]
  __int64 v123; // [rsp+30h] [rbp-110h]
  unsigned __int64 v124; // [rsp+40h] [rbp-100h]
  _QWORD *v125; // [rsp+48h] [rbp-F8h]
  int v126; // [rsp+48h] [rbp-F8h]
  __int64 v127; // [rsp+58h] [rbp-E8h]
  __int64 v128; // [rsp+58h] [rbp-E8h]
  __int64 v129; // [rsp+60h] [rbp-E0h]
  __int64 v130; // [rsp+60h] [rbp-E0h]
  unsigned int v132; // [rsp+68h] [rbp-D8h]
  _QWORD v133[2]; // [rsp+70h] [rbp-D0h] BYREF
  _QWORD v134[2]; // [rsp+80h] [rbp-C0h] BYREF
  _QWORD *v135; // [rsp+90h] [rbp-B0h] BYREF
  _BYTE *v136; // [rsp+98h] [rbp-A8h]
  _BYTE *v137; // [rsp+A0h] [rbp-A0h]
  __m128i v138; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v139; // [rsp+C0h] [rbp-80h]
  const char *v140; // [rsp+D0h] [rbp-70h] BYREF
  __int64 v141; // [rsp+D8h] [rbp-68h]
  __int16 v142; // [rsp+E0h] [rbp-60h]
  __m128i v143; // [rsp+F0h] [rbp-50h] BYREF
  __int64 v144; // [rsp+100h] [rbp-40h]

  v8 = a2;
  v9 = a1;
  v10 = *(_DWORD *)(a1 + 96);
  if ( v10 <= 1 )
  {
    v111 = (_QWORD *)sub_157E9C0(a4);
    *(_QWORD *)(a1 + 104) = sub_1643270(v111);
  }
  else if ( v10 == 2 )
  {
    v11 = (_QWORD *)sub_157E9C0(a4);
    *(_QWORD *)(a1 + 104) = sub_1643320(v11);
  }
  else
  {
    v110 = (_QWORD *)sub_157E9C0(a4);
    *(_QWORD *)(a1 + 104) = sub_1643340(v110);
  }
  v12 = *(__int64 ***)(a2 + 32);
  v13 = *(__int64 ***)(a2 + 40);
  v135 = 0;
  v136 = 0;
  v137 = 0;
  if ( v12 != v13 )
  {
    v14 = 0;
    v15 = 0;
    v16 = v13;
    v17 = v12;
    while ( 1 )
    {
      v18 = **v17;
      v143.m128i_i64[0] = v18;
      if ( v15 == v14 )
      {
        ++v17;
        sub_1278040((__int64)&v135, v15, &v143);
        if ( v16 == v17 )
          break;
      }
      else
      {
        if ( v15 )
        {
          *(_QWORD *)v15 = v18;
          v15 = v136;
        }
        ++v17;
        v136 = v15 + 8;
        if ( v16 == v17 )
          break;
      }
      v15 = v136;
      v14 = v137;
    }
  }
  v19 = *(__int64 ****)(a3 + 40);
  v20 = *(_BYTE *)(a1 + 8);
  if ( *(__int64 ****)(a3 + 32) != v19 )
  {
    v21 = *(__int64 ****)(a3 + 32);
    while ( 1 )
    {
      v23 = **v21;
      if ( v20 )
      {
        v143.m128i_i64[0] = (__int64)**v21;
        v22 = v136;
        if ( v136 == v137 )
          goto LABEL_33;
        if ( v136 )
        {
          *(_QWORD *)v136 = v23;
          v22 = v136;
          v20 = *(_BYTE *)(v9 + 8);
        }
        ++v21;
        v136 = v22 + 8;
        if ( v19 == v21 )
          break;
      }
      else
      {
        v24 = sub_1646BA0(v23, 0);
        v22 = v136;
        v143.m128i_i64[0] = v24;
        if ( v136 != v137 )
        {
          if ( v136 )
          {
            *(_QWORD *)v136 = v24;
            v22 = v136;
          }
          v136 = v22 + 8;
          goto LABEL_23;
        }
LABEL_33:
        sub_1278040((__int64)&v135, v22, &v143);
LABEL_23:
        ++v21;
        v20 = *(_BYTE *)(v9 + 8);
        if ( v19 == v21 )
          break;
      }
    }
  }
  v25 = v136;
  if ( v20
    && ((__int64)(*(_QWORD *)(v8 + 40) - *(_QWORD *)(v8 + 32)) >> 3)
     + ((__int64)(*(_QWORD *)(a3 + 40) - *(_QWORD *)(a3 + 32)) >> 3) )
  {
    v117 = (__int64 *)sub_1645600(*a8, v135, (v136 - (_BYTE *)v135) >> 3, 0);
    if ( v135 != (_QWORD *)v136 )
      v136 = v135;
    v112 = sub_1646BA0(v117, 0);
    v113 = v136;
    v143.m128i_i64[0] = v112;
    if ( v136 == v137 )
    {
      sub_1278040((__int64)&v135, v136, &v143);
      v25 = v136;
    }
    else
    {
      if ( v136 )
      {
        *(_QWORD *)v136 = v112;
        v113 = v136;
      }
      v25 = v113 + 8;
      v136 = v113 + 8;
    }
  }
  v26 = 0;
  if ( *(_BYTE *)(v9 + 32) )
    v26 = *(_DWORD *)(*(_QWORD *)(a7 + 24) + 8LL) >> 8 != 0;
  v27 = sub_1644EA0(*(__int64 **)(v9 + 104), v135, (v25 - (_BYTE *)v135) >> 3, v26);
  v134[0] = sub_1649960(a4);
  v134[1] = v28;
  v142 = 261;
  v140 = (const char *)v134;
  v133[0] = sub_1649960(a7);
  v138.m128i_i64[0] = (__int64)v133;
  v138.m128i_i64[1] = (__int64)"_";
  v29 = v142;
  v133[1] = v30;
  LOWORD(v139) = 773;
  if ( (_BYTE)v142 )
  {
    if ( (_BYTE)v142 == 1 )
    {
      v114 = _mm_loadu_si128(&v138);
      v144 = v139;
      v143 = v114;
    }
    else
    {
      if ( HIBYTE(v142) == 1 )
      {
        v31 = (const char **)v140;
      }
      else
      {
        v31 = &v140;
        v29 = 2;
      }
      v143.m128i_i64[1] = (__int64)v31;
      v143.m128i_i64[0] = (__int64)&v138;
      LOBYTE(v144) = 2;
      BYTE1(v144) = v29;
    }
  }
  else
  {
    LOWORD(v144) = 256;
  }
  v32 = sub_1648B60(120);
  v121 = v32;
  if ( v32 )
    sub_15E2490(v32, v27, 7, (__int64)&v143, (__int64)a8);
  if ( (unsigned __int8)sub_1560180(a7 + 112, 30) )
    sub_15E0D50(v121, -1, 30);
  if ( (unsigned __int8)sub_1560180(a7 + 112, 56) )
    sub_15E0D50(v121, -1, 56);
  v140 = *(const char **)(a7 + 112);
  v143.m128i_i64[0] = sub_1560250(&v140);
  v33 = sub_155EE30(v143.m128i_i64);
  v34 = sub_155EE40(v143.m128i_i64);
  if ( v33 != v34 )
  {
    v129 = v9;
    v35 = (__int64 *)v33;
    while ( sub_155D3E0((__int64)v35) )
    {
      v36 = sub_155D7D0(v35);
      if ( v37 == 5 && *(_DWORD *)v36 == 1853188212 && *(_BYTE *)(v36 + 4) == 107 )
      {
        if ( (__int64 *)v34 == ++v35 )
        {
LABEL_51:
          v9 = v129;
          goto LABEL_52;
        }
      }
      else
      {
LABEL_44:
        sub_15E0DA0(v121, -1, *v35);
LABEL_45:
        if ( (__int64 *)v34 == ++v35 )
          goto LABEL_51;
      }
    }
    switch ( (unsigned int)sub_155D410(v35) )
    {
      case 0u:
      case 1u:
      case 2u:
      case 4u:
      case 5u:
      case 6u:
      case 8u:
      case 9u:
      case 0xAu:
      case 0xBu:
      case 0xCu:
      case 0xDu:
      case 0xEu:
      case 0x10u:
      case 0x12u:
      case 0x13u:
      case 0x14u:
      case 0x15u:
      case 0x16u:
      case 0x1Du:
      case 0x20u:
      case 0x24u:
      case 0x25u:
      case 0x26u:
      case 0x27u:
      case 0x28u:
      case 0x2Fu:
      case 0x30u:
      case 0x35u:
      case 0x36u:
      case 0x37u:
      case 0x39u:
      case 0x3Au:
      case 0x3Bu:
        goto LABEL_45;
      default:
        goto LABEL_44;
    }
    goto LABEL_45;
  }
LABEL_52:
  v38 = a5;
  sub_15E01D0(v121 + 72, a5);
  v39 = *(_QWORD *)(v121 + 72);
  v40 = *(_QWORD *)(a5 + 24);
  *(_QWORD *)(a5 + 32) = v121 + 72;
  v39 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)(a5 + 24) = v39 | v40 & 7;
  *(_QWORD *)(v39 + 8) = a5 + 24;
  *(_QWORD *)(v121 + 72) = *(_QWORD *)(v121 + 72) & 7LL | (a5 + 24);
  if ( (*(_BYTE *)(v121 + 18) & 1) != 0 )
    sub_15E08E0(v121, a5);
  v41 = v8;
  v130 = 0;
  v127 = *(_QWORD *)(v121 + 88);
  v42 = (__int64)(*(_QWORD *)(v8 + 40) - *(_QWORD *)(v8 + 32)) >> 3;
  v120 = (unsigned int)v42;
  if ( (_DWORD)v42 )
  {
    do
    {
      if ( *(_BYTE *)(v9 + 8) )
      {
        v89 = (_QWORD *)sub_157E9C0(a4);
        v90 = (__int64 **)sub_1643350(v89);
        v138.m128i_i64[0] = sub_15A06D0(v90, v38, v91, v92);
        v93 = (_QWORD *)sub_157E9C0(a4);
        v94 = sub_1643350(v93);
        v138.m128i_i64[1] = sub_159C470(v94, v130, 0);
        v95 = *(_QWORD *)(v121 + 80);
        if ( v95 )
          v95 -= 24;
        v124 = sub_157EBA0(v95);
        v140 = sub_1649960(*(_QWORD *)(*(_QWORD *)(v41 + 32) + 8 * v130));
        v143.m128i_i64[0] = (__int64)"gep_";
        v141 = v96;
        v143.m128i_i64[1] = (__int64)&v140;
        LOWORD(v144) = 1283;
        v97 = (__int64)v117;
        if ( !v117 )
        {
          v109 = *(_QWORD *)v127;
          if ( *(_BYTE *)(*(_QWORD *)v127 + 8LL) == 16 )
            v109 = **(_QWORD **)(v109 + 16);
          v97 = *(_QWORD *)(v109 + 24);
        }
        v98 = sub_1648A60(72, 3u);
        v99 = (__int64)v98;
        if ( v98 )
        {
          v100 = v98 - 9;
          v101 = *(_QWORD *)v127;
          if ( *(_BYTE *)(*(_QWORD *)v127 + 8LL) == 16 )
            v101 = **(_QWORD **)(v101 + 16);
          v123 = (__int64)v100;
          v126 = *(_DWORD *)(v101 + 8) >> 8;
          v102 = (__int64 *)sub_15F9F50(v97, (__int64)&v138, 2);
          v103 = sub_1646BA0(v102, v126);
          v104 = v123;
          v105 = (__int64 *)v103;
          v106 = *(_QWORD *)v127;
          if ( *(_BYTE *)(*(_QWORD *)v127 + 8LL) == 16
            || (v106 = *(_QWORD *)v138.m128i_i64[0], *(_BYTE *)(*(_QWORD *)v138.m128i_i64[0] + 8LL) == 16)
            || (v106 = *(_QWORD *)v138.m128i_i64[1], *(_BYTE *)(*(_QWORD *)v138.m128i_i64[1] + 8LL) == 16) )
          {
            v108 = sub_16463B0(v105, *(_QWORD *)(v106 + 32));
            v104 = v123;
            v105 = v108;
          }
          sub_15F1EA0(v99, (__int64)v105, 32, v104, 3, v124);
          *(_QWORD *)(v99 + 56) = v97;
          *(_QWORD *)(v99 + 64) = sub_15F9F50(v97, (__int64)&v138, 2);
          sub_15F9CE0(v99, v127, v138.m128i_i64, 2, (__int64)&v143);
        }
        v38 = 1;
        v140 = sub_1649960(*(_QWORD *)(*(_QWORD *)(v41 + 32) + 8 * v130));
        v143.m128i_i64[0] = (__int64)"loadgep_";
        v141 = v107;
        LOWORD(v144) = 1283;
        v143.m128i_i64[1] = (__int64)&v140;
        v125 = sub_1648A60(64, 1u);
        if ( v125 )
        {
          v38 = v99;
          sub_15F90E0((__int64)v125, v99, (__int64)&v143, v124);
        }
      }
      else
      {
        v125 = (_QWORD *)v127;
        v127 += 40;
      }
      v43 = *(_QWORD *)(*(_QWORD *)(v41 + 32) + 8 * v130);
      v44 = *(_QWORD *)(v43 + 8);
      if ( v44 )
      {
        v45 = *(_QWORD *)(v43 + 8);
        v46 = 0;
        do
        {
          v45 = *(_QWORD *)(v45 + 8);
          ++v46;
        }
        while ( v45 );
        if ( v46 > 0xFFFFFFFFFFFFFFFLL )
          goto LABEL_141;
        v122 = 8 * v46;
        v47 = sub_22077B0(8 * v46);
        v48 = v47;
        do
        {
          v48 += 8;
          *(_QWORD *)(v48 - 8) = sub_1648700(v44);
          v44 = *(_QWORD *)(v44 + 8);
        }
        while ( v44 );
        if ( v48 == v47 )
          goto LABEL_71;
        v49 = (__int64 *)v47;
        do
        {
          v50 = *v49;
          if ( *(_BYTE *)(*v49 + 16) > 0x17u )
          {
            v51 = *(_DWORD *)(v9 + 64);
            if ( v51 )
            {
              v52 = *(_QWORD *)(v50 + 40);
              v53 = v51 - 1;
              v54 = *(_QWORD *)(v9 + 48);
              v55 = 1;
              v56 = v53 & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
              v38 = *(_QWORD *)(v54 + 8LL * v56);
              if ( v52 == v38 )
              {
LABEL_68:
                v38 = *(_QWORD *)(*(_QWORD *)(v41 + 32) + 8 * v130);
                sub_1648780(v50, v38, (__int64)v125);
              }
              else
              {
                while ( v38 != -8 )
                {
                  v56 = v53 & (v55 + v56);
                  v38 = *(_QWORD *)(v54 + 8LL * v56);
                  if ( v52 == v38 )
                    goto LABEL_68;
                  ++v55;
                }
              }
            }
          }
          ++v49;
        }
        while ( (__int64 *)v48 != v49 );
        if ( v47 )
        {
LABEL_71:
          v38 = v122;
          j_j___libc_free_0(v47, v122);
        }
      }
      ++v130;
    }
    while ( v120 != v130 );
    v8 = v41;
  }
  if ( !*(_BYTE *)(v9 + 8) )
  {
    if ( (*(_BYTE *)(v121 + 18) & 1) != 0 )
      sub_15E08E0(v121, v38);
    v57 = *(_QWORD *)(v121 + 88);
    v58 = *(_QWORD *)(v8 + 32);
    v59 = (*(_QWORD *)(v8 + 40) - v58) >> 3;
    v132 = v59;
    if ( (_DWORD)v59 )
    {
      v60 = *(_QWORD *)(v121 + 88);
      v128 = v9;
      v61 = 0;
      v62 = 8LL * (unsigned int)(v59 - 1);
      while ( 1 )
      {
        v63 = v60;
        v60 += 40;
        v140 = sub_1649960(*(_QWORD *)(v58 + v61));
        v141 = v64;
        LOWORD(v144) = 261;
        v143.m128i_i64[0] = (__int64)&v140;
        sub_164B780(v63, v143.m128i_i64);
        if ( v62 == v61 )
          break;
        v58 = *(_QWORD *)(v8 + 32);
        v61 += 8;
      }
      v9 = v128;
      v57 += 40LL * v132;
    }
    v65 = *(_QWORD *)(a3 + 32);
    v66 = (*(_QWORD *)(a3 + 40) - v65) >> 3;
    if ( (_DWORD)v66 )
    {
      v67 = 0;
      v68 = 8LL * (unsigned int)(v66 - 1);
      while ( 1 )
      {
        v69 = sub_1649960(*(_QWORD *)(v65 + v67));
        v70 = v57;
        v143.m128i_i64[1] = (__int64)".out";
        v140 = v69;
        v57 += 40;
        LOWORD(v144) = 773;
        v141 = v71;
        v143.m128i_i64[0] = (__int64)&v140;
        sub_164B780(v70, v143.m128i_i64);
        if ( v67 == v68 )
          break;
        v67 += 8;
        v65 = *(_QWORD *)(a3 + 32);
      }
    }
  }
  v72 = *(_QWORD *)(a4 + 8);
  if ( v72 )
  {
    v73 = *(_QWORD *)(a4 + 8);
    v74 = 0;
    do
    {
      v73 = *(_QWORD *)(v73 + 8);
      ++v74;
    }
    while ( v73 );
    if ( v74 > 0xFFFFFFFFFFFFFFFLL )
LABEL_141:
      sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
    v75 = 8 * v74;
    v76 = sub_22077B0(v75);
    v77 = v76;
    do
    {
      v77 += 8;
      *(_QWORD *)(v77 - 8) = sub_1648700(v72);
      v72 = *(_QWORD *)(v72 + 8);
    }
    while ( v72 );
    v78 = (v77 - v76) >> 3;
    if ( (_DWORD)v78 )
    {
      v79 = (__int64 *)v76;
      v80 = v76 + 8LL * (unsigned int)(v78 - 1) + 8;
      while ( 1 )
      {
        v85 = *v79;
        if ( (unsigned __int8)(*(_BYTE *)(*v79 + 16) - 25) > 9u )
          goto LABEL_95;
        v86 = *(_DWORD *)(v9 + 64);
        v87 = *(_QWORD *)(v85 + 40);
        if ( !v86 )
          goto LABEL_98;
        v81 = v86 - 1;
        v82 = *(_QWORD *)(v9 + 48);
        v83 = v81 & (((unsigned int)v87 >> 9) ^ ((unsigned int)v87 >> 4));
        v84 = *(_QWORD *)(v82 + 8LL * v83);
        if ( v87 == v84 )
        {
LABEL_95:
          if ( (__int64 *)v80 == ++v79 )
            break;
        }
        else
        {
          v115 = 1;
          while ( v84 != -8 )
          {
            v83 = v81 & (v115 + v83);
            v84 = *(_QWORD *)(v82 + 8LL * v83);
            if ( v87 == v84 )
              goto LABEL_95;
            ++v115;
          }
LABEL_98:
          if ( a7 != *(_QWORD *)(v87 + 56) )
            goto LABEL_95;
          ++v79;
          sub_1648780(v85, a4, a6);
          if ( (__int64 *)v80 == v79 )
            break;
        }
      }
    }
    if ( v76 )
      j_j___libc_free_0(v76, v75);
  }
  if ( v135 )
    j_j___libc_free_0(v135, v137 - (_BYTE *)v135);
  return v121;
}
