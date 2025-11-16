// Function: sub_304D410
// Address: 0x304d410
//
__int64 __fastcall sub_304D410(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  unsigned __int16 *v5; // rdx
  int v6; // eax
  __int64 v7; // r12
  __int64 v8; // rax
  unsigned __int64 v9; // rdx
  char v10; // al
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // rdx
  unsigned __int16 v15; // r12
  __int64 v16; // rdx
  _QWORD *v17; // rax
  __int64 v18; // r9
  const __m128i *v19; // r12
  unsigned __int16 v20; // r15
  __m128i v21; // xmm1
  __int64 v22; // r13
  __int64 v23; // r12
  __int64 v24; // rdx
  __int16 v25; // r15
  unsigned __int16 v26; // ax
  __int64 v27; // rdx
  unsigned __int64 v28; // r8
  __int64 *v29; // rdx
  __int64 v30; // rsi
  __int64 v31; // rdi
  int v32; // eax
  __int64 v33; // r8
  __int64 v34; // r9
  const __m128i *v35; // rax
  __int32 v36; // edx
  __m128i v37; // xmm2
  __int64 v38; // rax
  unsigned __int64 v39; // rdx
  __m128i *v40; // rax
  __int64 v41; // rax
  __int64 v42; // r15
  __int64 v43; // r13
  __int64 v44; // r12
  int v45; // eax
  int v46; // edx
  __int64 v47; // rsi
  int v48; // ecx
  int v49; // r8d
  __int64 v50; // r12
  __int64 v52; // rdx
  bool v53; // al
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 v57; // rsi
  __int64 v58; // rdx
  unsigned int v59; // eax
  __int64 v60; // rsi
  __int64 v61; // rax
  __int64 v62; // rdx
  __int64 v63; // r8
  int v64; // r9d
  __int64 v65; // rcx
  __int64 v66; // r11
  __int64 v67; // r10
  int v68; // r8d
  __int16 v69; // ax
  __int64 v70; // rsi
  __int16 v71; // r13
  __int64 v72; // rax
  __int64 v73; // rdx
  __int64 v74; // r9
  unsigned __int16 v75; // dx
  __int64 v76; // r8
  __int64 v77; // rax
  __int64 v78; // rdx
  __int64 *v79; // rdx
  __int64 v80; // rsi
  __int64 v81; // rsi
  __int64 v82; // rax
  __int64 v83; // rax
  __int64 v84; // rdx
  bool v85; // al
  __int64 v86; // rcx
  unsigned __int16 v87; // ax
  __int64 v88; // rdx
  __int64 v89; // r10
  int v90; // edx
  int v91; // eax
  int v92; // eax
  __int128 v93; // [rsp-10h] [rbp-1D0h]
  __int128 v94; // [rsp-10h] [rbp-1D0h]
  __int128 v95; // [rsp-10h] [rbp-1D0h]
  __m128i v96; // [rsp+8h] [rbp-1B8h]
  int v97; // [rsp+1Ch] [rbp-1A4h]
  __int64 v98; // [rsp+20h] [rbp-1A0h]
  __int64 v99; // [rsp+30h] [rbp-190h]
  __int64 v100; // [rsp+38h] [rbp-188h]
  __int64 v101; // [rsp+40h] [rbp-180h]
  __int64 v102; // [rsp+40h] [rbp-180h]
  __int64 v103; // [rsp+40h] [rbp-180h]
  __int64 v104; // [rsp+48h] [rbp-178h]
  __int64 v105; // [rsp+48h] [rbp-178h]
  int v106; // [rsp+50h] [rbp-170h]
  __int64 v107; // [rsp+50h] [rbp-170h]
  __int64 v108; // [rsp+50h] [rbp-170h]
  __int64 v109; // [rsp+50h] [rbp-170h]
  __int64 v110; // [rsp+50h] [rbp-170h]
  __int64 v111; // [rsp+58h] [rbp-168h]
  __int64 v112; // [rsp+58h] [rbp-168h]
  __int64 v113; // [rsp+58h] [rbp-168h]
  __int64 v114; // [rsp+58h] [rbp-168h]
  unsigned __int16 v115; // [rsp+70h] [rbp-150h]
  int v116; // [rsp+70h] [rbp-150h]
  int v117; // [rsp+78h] [rbp-148h]
  __int64 v118; // [rsp+80h] [rbp-140h]
  unsigned int i; // [rsp+80h] [rbp-140h]
  __int64 v120; // [rsp+88h] [rbp-138h]
  unsigned __int16 v121; // [rsp+90h] [rbp-130h] BYREF
  __int64 v122; // [rsp+98h] [rbp-128h]
  __int64 v123; // [rsp+A0h] [rbp-120h] BYREF
  int v124; // [rsp+A8h] [rbp-118h]
  __int64 v125; // [rsp+B0h] [rbp-110h] BYREF
  __int64 v126; // [rsp+B8h] [rbp-108h]
  __int64 v127; // [rsp+C0h] [rbp-100h]
  __int64 v128; // [rsp+C8h] [rbp-F8h]
  __m128i v129; // [rsp+D0h] [rbp-F0h] BYREF
  __m128i v130; // [rsp+E0h] [rbp-E0h] BYREF
  __m128i v131; // [rsp+F0h] [rbp-D0h] BYREF
  _OWORD *v132; // [rsp+100h] [rbp-C0h] BYREF
  __int64 v133; // [rsp+108h] [rbp-B8h]
  _OWORD v134[11]; // [rsp+110h] [rbp-B0h] BYREF

  v4 = a2;
  v5 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 120LL) + 48LL)
                          + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 128LL));
  v6 = *v5;
  v7 = *((_QWORD *)v5 + 1);
  v121 = v6;
  v122 = v7;
  if ( (_WORD)v6 )
  {
    if ( (unsigned __int16)(v6 - 17) > 0xD3u )
    {
      LOWORD(v125) = v6;
      v126 = v7;
      goto LABEL_4;
    }
    LOWORD(v6) = word_4456580[v6 - 1];
    v52 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v121) )
    {
      v126 = v7;
      LOWORD(v125) = 0;
LABEL_9:
      v127 = sub_3007260((__int64)&v125);
      v128 = v14;
      v9 = v127;
      v10 = v128;
      goto LABEL_10;
    }
    LOWORD(v6) = sub_3009970((__int64)&v121, a2, v11, v12, v13);
  }
  LOWORD(v125) = v6;
  v126 = v52;
  if ( !(_WORD)v6 )
    goto LABEL_9;
LABEL_4:
  if ( (_WORD)v6 == 1 || (unsigned __int16)(v6 - 504) <= 7u )
    goto LABEL_120;
  v8 = 16LL * ((unsigned __int16)v6 - 1);
  v9 = *(_QWORD *)&byte_444C4A0[v8];
  v10 = byte_444C4A0[v8 + 8];
LABEL_10:
  LOBYTE(v133) = v10;
  v132 = (_OWORD *)v9;
  if ( (unsigned __int64)sub_CA1930(&v132) <= 0xF )
  {
    v98 = 0;
    v115 = 6;
  }
  else
  {
    v115 = v125;
    v98 = v126;
  }
  v15 = v121;
  v99 = v115;
  v16 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL) + 96LL);
  v17 = *(_QWORD **)(v16 + 24);
  if ( *(_DWORD *)(v16 + 32) > 0x40u )
    v17 = (_QWORD *)*v17;
  if ( v17 != (_QWORD *)9553 )
  {
    LOWORD(v132) = v121;
    v133 = v122;
    if ( v121 )
    {
      if ( (unsigned __int16)(v121 - 17) > 0xD3u )
      {
LABEL_17:
        v97 = 559;
        goto LABEL_18;
      }
      if ( (unsigned __int16)(v121 - 176) <= 0x34u )
      {
        sub_CA17B0(
          "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use E"
          "VT::getVectorElementCount() instead");
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
      }
      v91 = word_4456340[v15 - 1];
      if ( v91 == 2 )
        goto LABEL_105;
    }
    else
    {
      if ( !sub_30070B0((__int64)&v132) )
        goto LABEL_17;
      if ( sub_3007100((__int64)&v132) )
        sub_CA17B0(
          "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use E"
          "VT::getVectorElementCount() instead");
      v91 = sub_3007130((__int64)&v132, a2);
      if ( v91 == 2 )
      {
LABEL_105:
        v97 = 560;
        goto LABEL_18;
      }
    }
    if ( v91 == 4 )
    {
      v97 = 561;
      goto LABEL_18;
    }
LABEL_120:
    BUG();
  }
  v129.m128i_i16[0] = v121;
  v129.m128i_i64[1] = v122;
  if ( v121 )
  {
    if ( (unsigned __int16)(v121 - 17) > 0xD3u )
    {
      v97 = 565;
      goto LABEL_18;
    }
    if ( (unsigned __int16)(v121 - 176) <= 0x34u )
    {
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    }
    v92 = word_4456340[v15 - 1];
  }
  else
  {
    if ( !sub_30070B0((__int64)&v129) )
    {
      v97 = 565;
      goto LABEL_18;
    }
    if ( sub_3007100((__int64)&v129) )
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
    v92 = sub_3007130((__int64)&v129, a2);
  }
  if ( v92 == 2 )
  {
    v97 = 566;
  }
  else
  {
    if ( v92 != 4 )
      goto LABEL_120;
    v97 = 567;
  }
LABEL_18:
  sub_3030880((__int64)&v132, a2, a3);
  v19 = *(const __m128i **)(a2 + 40);
  v20 = v121;
  v96.m128i_i64[0] = *(_QWORD *)&v134[0];
  v96.m128i_i64[1] = DWORD2(v134[0]);
  v129 = _mm_loadu_si128(v19);
  v130.m128i_i64[0] = (__int64)v132;
  v134[0] = v129;
  v130.m128i_i64[1] = (unsigned int)v133;
  v21 = _mm_load_si128(&v130);
  v132 = v134;
  v133 = 0x800000002LL;
  v134[1] = v21;
  if ( v121 )
  {
    if ( (unsigned __int16)(v121 - 17) <= 0xD3u )
    {
      if ( v121 == 37 )
      {
        v22 = v19[8].m128i_i64[0];
        v23 = v19[7].m128i_i64[1];
        v24 = 0;
        v25 = 5;
        goto LABEL_22;
      }
      goto LABEL_55;
    }
    v22 = v19[8].m128i_i64[0];
    v23 = v19[7].m128i_i64[1];
  }
  else
  {
    if ( sub_30070B0((__int64)&v121) )
    {
LABEL_55:
      for ( i = 0; ; ++i )
      {
        if ( v20 )
        {
          if ( (unsigned __int16)(v20 - 176) > 0x34u )
            goto LABEL_57;
        }
        else if ( !sub_3007100((__int64)&v121) )
        {
          goto LABEL_79;
        }
        sub_CA17B0(
          "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use E"
          "VT::getVectorElementCount() instead");
        if ( v121 )
        {
          if ( (unsigned __int16)(v121 - 176) <= 0x34u )
            sub_CA17B0(
              "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se MVT::getVectorElementCount() instead");
LABEL_57:
          v59 = word_4456340[v121 - 1];
          goto LABEL_58;
        }
LABEL_79:
        v59 = sub_3007130((__int64)&v121, a2);
LABEL_58:
        v60 = *(_QWORD *)(v4 + 80);
        if ( i >= v59 )
          goto LABEL_27;
        v125 = *(_QWORD *)(v4 + 80);
        if ( v60 )
          sub_B96E90((__int64)&v125, v60, 1);
        LODWORD(v126) = *(_DWORD *)(v4 + 72);
        v61 = sub_3400D50(a3, i, &v125, 0);
        v65 = *(_QWORD *)(v4 + 40);
        v66 = v62;
        v67 = v61;
        if ( v121 )
        {
          v68 = 0;
          v69 = word_4456580[v121 - 1];
        }
        else
        {
          v103 = *(_QWORD *)(v4 + 40);
          v109 = v61;
          v113 = v62;
          v69 = sub_3009970((__int64)&v121, i, 0, v65, v63);
          v65 = v103;
          v67 = v109;
          v66 = v113;
          v68 = v90;
        }
        v70 = *(_QWORD *)(v4 + 80);
        v71 = v69;
        v123 = v70;
        if ( v70 )
        {
          v100 = v65;
          v101 = v67;
          v104 = v66;
          v106 = v68;
          sub_B96E90((__int64)&v123, v70, 1);
          v65 = v100;
          v67 = v101;
          v66 = v104;
          v68 = v106;
        }
        *((_QWORD *)&v94 + 1) = v66;
        *(_QWORD *)&v94 = v67;
        v124 = *(_DWORD *)(v4 + 72);
        v72 = sub_3406EB0(a3, 158, (unsigned int)&v123, v71, v68, v64, *(_OWORD *)(v65 + 120), v94);
        v74 = v73;
        v75 = v121;
        v76 = v72;
        if ( v121 )
        {
          if ( (unsigned __int16)(v121 - 17) > 0xD3u )
            goto LABEL_67;
          v75 = word_4456580[v121 - 1];
          v77 = 0;
        }
        else
        {
          v102 = v72;
          v105 = v74;
          v85 = sub_30070B0((__int64)&v121);
          v75 = 0;
          v76 = v102;
          v74 = v105;
          if ( !v85 )
          {
LABEL_67:
            v77 = v122;
            goto LABEL_68;
          }
          v87 = sub_3009970((__int64)&v121, 158, 0, v86, v102);
          v76 = v102;
          v74 = v105;
          v89 = v88;
          v75 = v87;
          v77 = v89;
        }
LABEL_68:
        if ( (v115 != v75 || !v115 && v98 != v77) && v121 != 37 )
        {
          v81 = *(_QWORD *)(v4 + 80);
          v129.m128i_i64[0] = v81;
          if ( v81 )
          {
            v107 = v76;
            v111 = v74;
            sub_B96E90((__int64)&v129, v81, 1);
            v76 = v107;
            v74 = v111;
          }
          *((_QWORD *)&v95 + 1) = v74;
          *(_QWORD *)&v95 = v76;
          v129.m128i_i32[2] = *(_DWORD *)(v4 + 72);
          v82 = v99;
          LOWORD(v82) = v115;
          v99 = v82;
          v83 = sub_33FAF80(a3, 215, (unsigned int)&v129, v82, v98, v74, v95);
          v76 = v83;
          v74 = v84;
          if ( v129.m128i_i64[0] )
          {
            v108 = v83;
            v112 = v84;
            sub_B91220((__int64)&v129, v129.m128i_i64[0]);
            v76 = v108;
            v74 = v112;
          }
        }
        v78 = (unsigned int)v133;
        if ( (unsigned __int64)(unsigned int)v133 + 1 > HIDWORD(v133) )
        {
          v110 = v76;
          v114 = v74;
          sub_C8D5F0((__int64)&v132, v134, (unsigned int)v133 + 1LL, 0x10u, v76, v74);
          v78 = (unsigned int)v133;
          v76 = v110;
          v74 = v114;
        }
        v79 = (__int64 *)&v132[v78];
        *v79 = v76;
        v80 = v123;
        v79[1] = v74;
        LODWORD(v133) = v133 + 1;
        if ( v80 )
          sub_B91220((__int64)&v123, v80);
        a2 = v125;
        if ( v125 )
          sub_B91220((__int64)&v125, v125);
        v20 = v121;
      }
    }
    v22 = v19[8].m128i_i64[0];
    v53 = sub_30070B0((__int64)&v121);
    v23 = v19[7].m128i_i64[1];
    if ( v53 )
    {
      v25 = sub_3009970((__int64)&v121, a2, v54, v55, v56);
LABEL_22:
      v26 = v115;
      if ( v115 != v25 )
        goto LABEL_46;
      goto LABEL_23;
    }
  }
  v26 = v115;
  v24 = v122;
  if ( v115 != v20 )
    goto LABEL_46;
LABEL_23:
  if ( !v26 && v98 != v24 )
  {
LABEL_46:
    if ( v121 != 37 )
    {
      v57 = *(_QWORD *)(a2 + 80);
      v129.m128i_i64[0] = v57;
      if ( v57 )
        sub_B96E90((__int64)&v129, v57, 1);
      *((_QWORD *)&v93 + 1) = v22;
      *(_QWORD *)&v93 = v23;
      v129.m128i_i32[2] = *(_DWORD *)(v4 + 72);
      v23 = sub_33FAF80(a3, 215, (unsigned int)&v129, v115, v98, v18, v93);
      v22 = v58;
      if ( v129.m128i_i64[0] )
        sub_B91220((__int64)&v129, v129.m128i_i64[0]);
    }
  }
  v27 = (unsigned int)v133;
  v28 = (unsigned int)v133 + 1LL;
  if ( v28 > HIDWORD(v133) )
  {
    sub_C8D5F0((__int64)&v132, v134, (unsigned int)v133 + 1LL, 0x10u, v28, v18);
    v27 = (unsigned int)v133;
  }
  v29 = (__int64 *)&v132[v27];
  *v29 = v23;
  v29[1] = v22;
  LODWORD(v133) = v133 + 1;
LABEL_27:
  v30 = *(_QWORD *)(v4 + 80);
  v125 = v30;
  if ( v30 )
    sub_B96E90((__int64)&v125, v30, 1);
  v31 = *(_QWORD *)(v4 + 112);
  LODWORD(v126) = *(_DWORD *)(v4 + 72);
  v32 = sub_2EAC1E0(v31);
  v129.m128i_i64[0] = sub_3400BD0(a3, v32, (unsigned int)&v125, 5, 0, 1, 0);
  v35 = *(const __m128i **)(v4 + 40);
  v129.m128i_i32[2] = v36;
  v37 = _mm_loadu_si128(v35 + 10);
  v131 = v96;
  v130 = v37;
  v38 = (unsigned int)v133;
  v39 = (unsigned int)v133 + 3LL;
  if ( v39 > HIDWORD(v133) )
  {
    sub_C8D5F0((__int64)&v132, v134, v39, 0x10u, v33, v34);
    v38 = (unsigned int)v133;
  }
  v40 = (__m128i *)&v132[v38];
  *v40 = _mm_load_si128(&v129);
  v40[1] = _mm_load_si128(&v130);
  v40[2] = _mm_load_si128(&v131);
  v41 = (unsigned int)(v133 + 3);
  LODWORD(v133) = v133 + 3;
  if ( v125 )
  {
    sub_B91220((__int64)&v125, v125);
    v41 = (unsigned int)v133;
  }
  v42 = *(_QWORD *)(v4 + 112);
  v120 = v41;
  v43 = *(_QWORD *)(v4 + 104);
  v118 = (__int64)v132;
  v44 = *(unsigned __int16 *)(v4 + 96);
  v45 = sub_33ED250(a3, 1, 0, v132);
  v47 = *(_QWORD *)(v4 + 80);
  v48 = v45;
  v49 = v46;
  v129.m128i_i64[0] = v47;
  if ( v47 )
  {
    v116 = v46;
    v117 = v45;
    sub_B96E90((__int64)&v129, v47, 1);
    v49 = v116;
    v48 = v117;
  }
  v129.m128i_i32[2] = *(_DWORD *)(v4 + 72);
  v50 = sub_33EA9D0(a3, v97, (unsigned int)&v129, v48, v49, v42, v118, v120, v44, v43);
  if ( v129.m128i_i64[0] )
    sub_B91220((__int64)&v129, v129.m128i_i64[0]);
  if ( v132 != v134 )
    _libc_free((unsigned __int64)v132);
  return v50;
}
