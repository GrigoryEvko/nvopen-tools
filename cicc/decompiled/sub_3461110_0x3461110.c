// Function: sub_3461110
// Address: 0x3461110
//
__m128i *__fastcall sub_3461110(__m128i a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rsi
  __int64 v6; // rax
  __m128i v7; // xmm3
  unsigned int v8; // r14d
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rax
  unsigned __int16 v11; // bx
  __int64 v12; // rax
  unsigned __int16 v13; // r13
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  unsigned __int64 v18; // rdx
  unsigned __int16 v19; // ax
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int8 v23; // al
  unsigned int v24; // eax
  int v25; // eax
  __int64 v26; // rdx
  __int64 v27; // rbx
  __int64 v28; // rdx
  __int8 v29; // cl
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // rdx
  __int64 v35; // rdx
  __int64 v36; // r9
  __int64 v37; // rbx
  int v38; // edx
  __int64 v39; // rcx
  __int64 v40; // r8
  __m128i *v41; // r14
  __int64 v42; // rax
  __m128i *v43; // rdx
  __m128i *v44; // r15
  unsigned __int64 v45; // rdx
  __m128i **v46; // rax
  __int64 v47; // rax
  __int128 v48; // rax
  __int64 v49; // r9
  unsigned __int8 *v50; // rax
  unsigned __int64 v51; // rdx
  unsigned __int64 v52; // r15
  unsigned __int64 v53; // r14
  unsigned __int8 *v54; // r10
  unsigned __int64 v55; // rdx
  unsigned __int64 v56; // r11
  __int64 v57; // rax
  __int16 v58; // di
  unsigned __int8 v59; // si
  unsigned __int64 v60; // r8
  __int64 v61; // rcx
  char v62; // r9
  __int64 v63; // rdx
  int v64; // eax
  _OWORD *v65; // rdx
  __m128i *v66; // r14
  unsigned int v68; // edi
  __int128 v69; // rax
  __int128 v70; // kr00_16
  __int64 v71; // r15
  __int128 v72; // rax
  unsigned int v73; // ecx
  __int64 v74; // rax
  __int128 v75; // rax
  __int64 v76; // r9
  __int128 v77; // rax
  __int64 v78; // r9
  unsigned int v79; // edx
  __int128 v80; // rax
  __int64 v81; // r9
  int v82; // r9d
  int v83; // r9d
  __int64 v84; // rdx
  _BYTE *v85; // rax
  __int64 v86; // rax
  __int64 v87; // rax
  unsigned __int16 v88; // ax
  __int64 v89; // rdx
  __int128 v90; // [rsp-10h] [rbp-250h]
  __m128i v91; // [rsp+10h] [rbp-230h]
  unsigned __int64 v92; // [rsp+20h] [rbp-220h]
  unsigned int v93; // [rsp+28h] [rbp-218h]
  __int64 v94; // [rsp+30h] [rbp-210h]
  __int64 v95; // [rsp+38h] [rbp-208h]
  unsigned int v96; // [rsp+38h] [rbp-208h]
  unsigned __int64 v97; // [rsp+40h] [rbp-200h]
  int v98; // [rsp+50h] [rbp-1F0h]
  __int128 v99; // [rsp+50h] [rbp-1F0h]
  __int64 v101; // [rsp+68h] [rbp-1D8h]
  __int16 v102; // [rsp+7Ah] [rbp-1C6h]
  __int64 v103; // [rsp+80h] [rbp-1C0h]
  unsigned __int64 v104; // [rsp+88h] [rbp-1B8h]
  unsigned int v105; // [rsp+90h] [rbp-1B0h]
  unsigned int v106; // [rsp+90h] [rbp-1B0h]
  __int128 v107; // [rsp+90h] [rbp-1B0h]
  unsigned __int64 v108; // [rsp+A8h] [rbp-198h]
  __int64 v109; // [rsp+C0h] [rbp-180h] BYREF
  int v110; // [rsp+C8h] [rbp-178h]
  unsigned __int16 v111; // [rsp+D0h] [rbp-170h] BYREF
  unsigned __int64 v112; // [rsp+D8h] [rbp-168h]
  unsigned __int16 v113; // [rsp+E0h] [rbp-160h] BYREF
  __int64 v114; // [rsp+E8h] [rbp-158h]
  __int64 v115; // [rsp+F0h] [rbp-150h] BYREF
  unsigned __int64 v116; // [rsp+F8h] [rbp-148h]
  __int64 v117; // [rsp+100h] [rbp-140h]
  __int64 v118; // [rsp+108h] [rbp-138h]
  __int64 v119; // [rsp+110h] [rbp-130h]
  __int64 v120; // [rsp+118h] [rbp-128h]
  __int64 v121; // [rsp+120h] [rbp-120h]
  __int64 v122; // [rsp+128h] [rbp-118h]
  __int64 v123; // [rsp+130h] [rbp-110h]
  __int64 v124; // [rsp+138h] [rbp-108h]
  __int128 v125; // [rsp+140h] [rbp-100h]
  __int64 v126; // [rsp+150h] [rbp-F0h]
  _OWORD v127[2]; // [rsp+160h] [rbp-E0h] BYREF
  __m128i v128; // [rsp+180h] [rbp-C0h] BYREF
  _OWORD v129[11]; // [rsp+190h] [rbp-B0h] BYREF

  v5 = *(_QWORD *)(a3 + 80);
  v109 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v109, v5, 1);
  v110 = *(_DWORD *)(a3 + 72);
  v6 = *(_QWORD *)(a3 + 40);
  v7 = _mm_loadu_si128((const __m128i *)(v6 + 40));
  v8 = *(_DWORD *)(v6 + 48);
  v92 = *(_QWORD *)v6;
  v91 = _mm_loadu_si128((const __m128i *)(v6 + 80));
  v97 = *(_QWORD *)(v6 + 8);
  v9 = *(_QWORD *)(v6 + 40);
  v10 = *(_QWORD *)(a3 + 104);
  v11 = *(_WORD *)(a3 + 96);
  v108 = v7.m128i_u64[1];
  v104 = v9;
  v111 = v11;
  v112 = v10;
  if ( v11 )
  {
    if ( (unsigned __int16)(v11 - 176) > 0x34u )
      goto LABEL_5;
LABEL_27:
    sub_C64ED0("Cannot scalarize scalable vector stores", 1u);
  }
  if ( sub_3007100((__int64)&v111) )
    goto LABEL_27;
LABEL_5:
  v101 = v8;
  v12 = *(_QWORD *)(v9 + 48) + 16LL * v8;
  v13 = *(_WORD *)v12;
  v14 = *(_QWORD *)(v12 + 8);
  v113 = v13;
  v103 = v14;
  v114 = v14;
  if ( v13 )
  {
    if ( (unsigned __int16)(v13 - 17) <= 0xD3u )
    {
      v103 = 0;
      v13 = word_4456580[v13 - 1];
    }
  }
  else if ( sub_30070B0((__int64)&v113) )
  {
    v88 = sub_3009970((__int64)&v113, v5, v15, v16, v17);
    v11 = v111;
    v103 = v89;
    v13 = v88;
  }
  if ( v11 )
  {
    if ( (unsigned __int16)(v11 - 17) <= 0xD3u )
    {
      v116 = 0;
      LOWORD(v115) = word_4456580[v11 - 1];
      v19 = v11;
      goto LABEL_31;
    }
    goto LABEL_10;
  }
  if ( !sub_30070B0((__int64)&v111) )
  {
LABEL_10:
    v18 = v112;
    v19 = v11;
    goto LABEL_11;
  }
  v11 = sub_3009970((__int64)&v111, v5, v31, v32, v33);
  v19 = v111;
LABEL_11:
  LOWORD(v115) = v11;
  v116 = v18;
  if ( !v19 )
  {
    if ( !sub_3007100((__int64)&v111) )
      goto LABEL_13;
    goto LABEL_40;
  }
LABEL_31:
  if ( (unsigned __int16)(v19 - 176) > 0x34u )
    goto LABEL_32;
LABEL_40:
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( v111 )
  {
    if ( (unsigned __int16)(v111 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
LABEL_32:
    v11 = v115;
    v105 = word_4456340[v111 - 1];
    if ( !(_WORD)v115 )
      goto LABEL_14;
    goto LABEL_33;
  }
  v11 = v115;
LABEL_13:
  v105 = sub_3007130((__int64)&v111, v5);
  if ( !v11 )
  {
LABEL_14:
    v117 = sub_3007260((__int64)&v115);
    v118 = v20;
    if ( v117 )
    {
      v119 = sub_3007260((__int64)&v115);
      v120 = v34;
      if ( (v119 & 7) == 0 )
      {
        v30 = sub_3007260((__int64)&v115);
        v123 = v30;
        v124 = v35;
        v29 = v35;
        goto LABEL_44;
      }
    }
    goto LABEL_15;
  }
LABEL_33:
  if ( v11 == 1 || (unsigned __int16)(v11 - 504) <= 7u )
    goto LABEL_89;
  v28 = *(_QWORD *)&byte_444C4A0[16 * v11 - 16];
  v29 = byte_444C4A0[16 * v11 - 8];
  if ( v28 && (v28 & 7) == 0 )
  {
    v30 = *(_QWORD *)&byte_444C4A0[16 * v11 - 16];
LABEL_44:
    v128.m128i_i64[0] = v30;
    v128.m128i_i8[8] = v29;
    v98 = (unsigned __int64)sub_CA1930(&v128) >> 3;
    v128.m128i_i64[0] = (__int64)v129;
    v128.m128i_i64[1] = 0x800000000LL;
    if ( v105 )
    {
      v95 = v105;
      v37 = 0;
      v106 = 0;
      do
      {
        *(_QWORD *)&v48 = sub_3400EE0(a4, v37, (__int64)&v109, 0, a1);
        v108 = v101 | v108 & 0xFFFFFFFF00000000LL;
        v50 = sub_3406EB0((_QWORD *)a4, 0x9Eu, (__int64)&v109, v13, v103, v49, __PAIR128__(v108, v104), v48);
        v52 = v51;
        v53 = (unsigned __int64)v50;
        BYTE8(v127[0]) = 0;
        *(_QWORD *)&v127[0] = v106;
        v54 = sub_3409320((_QWORD *)a4, v91.m128i_i64[0], v91.m128i_i64[1], v106, 0, (__int64)&v109, a1, 1);
        v56 = v55;
        v57 = *(_QWORD *)(a3 + 112);
        a1 = _mm_loadu_si128((const __m128i *)(v57 + 40));
        v127[0] = a1;
        v127[1] = _mm_loadu_si128((const __m128i *)(v57 + 56));
        v58 = *(_WORD *)(v57 + 32);
        v59 = *(_BYTE *)(v57 + 34);
        v60 = *(_QWORD *)v57 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v60 )
        {
          v61 = *(_QWORD *)(v57 + 8) + v106;
          v62 = *(_BYTE *)(v57 + 20);
          if ( (*(_QWORD *)v57 & 4) != 0 )
          {
            *((_QWORD *)&v125 + 1) = *(_QWORD *)(v57 + 8) + v106;
            BYTE4(v126) = v62;
            *(_QWORD *)&v125 = v60 | 4;
            LODWORD(v126) = *(_DWORD *)(v60 + 12);
          }
          else
          {
            v63 = *(_QWORD *)(v60 + 8);
            *(_QWORD *)&v125 = *(_QWORD *)v57 & 0xFFFFFFFFFFFFFFF8LL;
            *((_QWORD *)&v125 + 1) = v61;
            v64 = *(unsigned __int8 *)(v63 + 8);
            BYTE4(v126) = v62;
            if ( (unsigned int)(v64 - 17) <= 1 )
              v63 = **(_QWORD **)(v63 + 16);
            LODWORD(v126) = *(_DWORD *)(v63 + 8) >> 8;
          }
        }
        else
        {
          v38 = *(_DWORD *)(v57 + 16);
          v39 = *(_QWORD *)(v57 + 8) + v106;
          *(_QWORD *)&v125 = 0;
          *((_QWORD *)&v125 + 1) = v39;
          LODWORD(v126) = v38;
          BYTE4(v126) = 0;
        }
        v41 = sub_33F5040(
                (_QWORD *)a4,
                v92,
                v97,
                (__int64)&v109,
                v53,
                v52,
                (unsigned __int64)v54,
                v56,
                v125,
                v126,
                v115,
                v116,
                v59,
                v58,
                (__int64)v127);
        v42 = v128.m128i_u32[2];
        v44 = v43;
        v45 = v128.m128i_u32[2] + 1LL;
        if ( v45 > v128.m128i_u32[3] )
        {
          sub_C8D5F0((__int64)&v128, v129, v45, 0x10u, v40, v36);
          v42 = v128.m128i_u32[2];
        }
        v46 = (__m128i **)(v128.m128i_i64[0] + 16 * v42);
        ++v37;
        *v46 = v41;
        v46[1] = v44;
        v106 += v98;
        v47 = (unsigned int)++v128.m128i_i32[2];
      }
      while ( v37 != v95 );
      v65 = (_OWORD *)v128.m128i_i64[0];
    }
    else
    {
      v65 = v129;
      v47 = 0;
    }
    *((_QWORD *)&v90 + 1) = v47;
    *(_QWORD *)&v90 = v65;
    v66 = (__m128i *)sub_33FC220((_QWORD *)a4, 2, (__int64)&v109, 1, 0, v36, v90);
    if ( (_OWORD *)v128.m128i_i64[0] != v129 )
      _libc_free(v128.m128i_u64[0]);
    goto LABEL_59;
  }
LABEL_15:
  if ( !v111 )
  {
    v121 = sub_3007260((__int64)&v111);
    v122 = v21;
    v22 = v121;
    v23 = v122;
    goto LABEL_17;
  }
  if ( v111 == 1 || (unsigned __int16)(v111 - 504) <= 7u )
LABEL_89:
    BUG();
  v87 = 16LL * (v111 - 1);
  v22 = *(_QWORD *)&byte_444C4A0[v87];
  v23 = byte_444C4A0[v87 + 8];
LABEL_17:
  v128.m128i_i8[8] = v23;
  v128.m128i_i64[0] = v22;
  v24 = sub_CA1930(&v128);
  switch ( v24 )
  {
    case 1u:
      LOWORD(v25) = 2;
      goto LABEL_63;
    case 2u:
      LOWORD(v25) = 3;
      goto LABEL_63;
    case 4u:
      LOWORD(v25) = 4;
LABEL_63:
      v27 = 0;
      goto LABEL_64;
    case 8u:
      LOWORD(v25) = 5;
      goto LABEL_63;
    case 0x10u:
      LOWORD(v25) = 6;
      goto LABEL_63;
    case 0x20u:
      LOWORD(v25) = 7;
      goto LABEL_63;
    case 0x40u:
      LOWORD(v25) = 8;
      goto LABEL_63;
    case 0x80u:
      LOWORD(v25) = 9;
      goto LABEL_63;
  }
  v25 = sub_3007020(*(_QWORD **)(a4 + 64), v24);
  v102 = HIWORD(v25);
  v27 = v26;
LABEL_64:
  HIWORD(v68) = v102;
  LOWORD(v68) = v25;
  *(_QWORD *)&v69 = sub_3400BD0(a4, 0, (__int64)&v109, v68, v27, 0, a1, 0);
  v70 = v69;
  v94 = v105;
  v93 = v105 - 1;
  if ( v105 )
  {
    v107 = v69;
    v71 = 0;
    do
    {
      *(_QWORD *)&v80 = sub_3400EE0(a4, v71, (__int64)&v109, 0, a1);
      v108 = v8 | v108 & 0xFFFFFFFF00000000LL;
      sub_3406EB0((_QWORD *)a4, 0x9Eu, (__int64)&v109, v13, v103, v81, __PAIR128__(v108, v104), v80);
      sub_33FAF80(a4, 216, (__int64)&v109, (unsigned int)v115, v116, v82, a1);
      *(_QWORD *)&v99 = sub_33FAF80(a4, 214, (__int64)&v109, v68, v27, v83, a1);
      *((_QWORD *)&v99 + 1) = v84;
      v85 = (_BYTE *)sub_2E79000(*(__int64 **)(a4 + 40));
      v73 = v93 - v71;
      if ( !*v85 )
        v73 = v71;
      if ( (_WORD)v115 )
      {
        if ( (_WORD)v115 == 1 || (unsigned __int16)(v115 - 504) <= 7u )
          goto LABEL_89;
        *(_QWORD *)&v72 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v115 - 16];
        BYTE8(v72) = byte_444C4A0[16 * (unsigned __int16)v115 - 8];
      }
      else
      {
        v96 = v73;
        *(_QWORD *)&v72 = sub_3007260((__int64)&v115);
        v73 = v96;
        v127[0] = v72;
      }
      v128.m128i_i8[8] = BYTE8(v72);
      ++v71;
      v128.m128i_i64[0] = v73 * (_QWORD)v72;
      v74 = sub_CA1930(&v128);
      *(_QWORD *)&v75 = sub_3400BD0(a4, v74, (__int64)&v109, v68, v27, 0, a1, 0);
      *(_QWORD *)&v77 = sub_3406EB0((_QWORD *)a4, 0xBEu, (__int64)&v109, v68, v27, v76, v99, v75);
      *(_QWORD *)&v107 = sub_3406EB0((_QWORD *)a4, 0xBBu, (__int64)&v109, v68, v27, v78, v107, v77);
      *((_QWORD *)&v107 + 1) = v79 | *((_QWORD *)&v107 + 1) & 0xFFFFFFFF00000000LL;
    }
    while ( v94 != v71 );
    v70 = v107;
  }
  v86 = *(_QWORD *)(a3 + 112);
  v128 = _mm_loadu_si128((const __m128i *)(v86 + 40));
  v129[0] = _mm_loadu_si128((const __m128i *)(v86 + 56));
  v66 = sub_33F4560(
          (_QWORD *)a4,
          v92,
          v97,
          (__int64)&v109,
          v70,
          *((unsigned __int64 *)&v70 + 1),
          v91.m128i_u64[0],
          v91.m128i_u64[1],
          *(_OWORD *)v86,
          *(_QWORD *)(v86 + 16),
          *(_BYTE *)(v86 + 34),
          *(_WORD *)(v86 + 32),
          (__int64)&v128);
LABEL_59:
  if ( v109 )
    sub_B91220((__int64)&v109, v109);
  return v66;
}
