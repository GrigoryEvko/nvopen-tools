// Function: sub_3767A40
// Address: 0x3767a40
//
unsigned __int8 *__fastcall sub_3767A40(__int64 *a1, __int64 a2)
{
  __int64 v4; // rsi
  __int16 *v5; // rdx
  __int16 v6; // ax
  __int64 v7; // rdx
  const __m128i *v8; // rax
  __int64 v9; // r15
  __int64 v10; // r14
  __int128 v11; // xmm0
  __int64 v12; // rax
  __int16 v13; // dx
  __int64 v14; // rax
  unsigned __int16 v15; // r12
  unsigned __int16 v16; // r13
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rdx
  int v20; // eax
  unsigned __int64 v21; // rcx
  __int64 v22; // rax
  __int64 v23; // rdi
  __int64 v24; // rdx
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  unsigned __int8 *v27; // rax
  char *v28; // rcx
  __int64 v29; // r8
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // r9
  int v33; // edx
  __int64 i; // rdx
  _BYTE *v35; // rax
  int v36; // edx
  int v37; // eax
  __int64 v38; // rdx
  __int64 v39; // rbx
  int v40; // r9d
  unsigned __int8 *v41; // r14
  __int64 v43; // rax
  unsigned __int16 v44; // r13
  unsigned __int64 v45; // r12
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // rdx
  unsigned __int64 v50; // rcx
  __int64 v51; // rdx
  unsigned __int16 v52; // r15
  unsigned __int64 v53; // r12
  __int64 *v54; // r14
  __int16 v55; // ax
  __int64 v56; // rcx
  _QWORD *v57; // r13
  unsigned __int8 *v58; // rax
  _QWORD *v59; // rdi
  __int64 v60; // rdx
  __int64 v61; // r15
  unsigned __int8 *v62; // r14
  __int128 v63; // rax
  __int64 v64; // r9
  unsigned int v65; // edx
  __int64 v66; // rdx
  __int64 v67; // rdx
  __int64 v68; // rcx
  __int64 v69; // r8
  unsigned __int16 v70; // ax
  __int64 v71; // rdx
  __int64 v72; // rdx
  __int128 v73; // [rsp-10h] [rbp-150h]
  __int64 v74; // [rsp+0h] [rbp-140h]
  __int64 v75; // [rsp+8h] [rbp-138h]
  __int64 v76; // [rsp+10h] [rbp-130h]
  __int64 v77; // [rsp+10h] [rbp-130h]
  __int128 v78; // [rsp+10h] [rbp-130h]
  __int64 v79; // [rsp+18h] [rbp-128h]
  unsigned __int64 v80; // [rsp+20h] [rbp-120h]
  int v81; // [rsp+28h] [rbp-118h]
  int v82; // [rsp+2Ch] [rbp-114h]
  __int64 v83; // [rsp+40h] [rbp-100h] BYREF
  int v84; // [rsp+48h] [rbp-F8h]
  unsigned int v85; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v86; // [rsp+58h] [rbp-E8h]
  __int64 v87; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v88; // [rsp+68h] [rbp-D8h]
  __int64 v89; // [rsp+70h] [rbp-D0h] BYREF
  char v90; // [rsp+78h] [rbp-C8h]
  __int64 v91; // [rsp+80h] [rbp-C0h]
  __int64 v92; // [rsp+88h] [rbp-B8h]
  __int64 v93; // [rsp+90h] [rbp-B0h]
  __int64 v94; // [rsp+98h] [rbp-A8h]
  __int64 v95; // [rsp+A0h] [rbp-A0h]
  __int64 v96; // [rsp+A8h] [rbp-98h]
  __int64 v97; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v98; // [rsp+B8h] [rbp-88h]
  char *v99; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v100; // [rsp+C8h] [rbp-78h]
  _BYTE v101[112]; // [rsp+D0h] [rbp-70h] BYREF

  v4 = *(_QWORD *)(a2 + 80);
  v83 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v83, v4, 1);
  v5 = *(__int16 **)(a2 + 48);
  v84 = *(_DWORD *)(a2 + 72);
  v6 = *v5;
  v7 = *((_QWORD *)v5 + 1);
  LOWORD(v85) = v6;
  v86 = v7;
  if ( v6 )
  {
    if ( (unsigned __int16)(v6 - 176) > 0x34u )
    {
LABEL_5:
      v81 = word_4456340[(unsigned __int16)v85 - 1];
      goto LABEL_8;
    }
  }
  else if ( !sub_3007100((__int64)&v85) )
  {
    goto LABEL_7;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( (_WORD)v85 )
  {
    if ( (unsigned __int16)(v85 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    goto LABEL_5;
  }
LABEL_7:
  v81 = sub_3007130((__int64)&v85, v4);
LABEL_8:
  v8 = *(const __m128i **)(a2 + 40);
  v9 = v8->m128i_u32[2];
  v10 = v8->m128i_i64[0];
  v11 = (__int128)_mm_loadu_si128(v8);
  v12 = *(_QWORD *)(v8->m128i_i64[0] + 48) + 16 * v9;
  v13 = *(_WORD *)v12;
  v14 = *(_QWORD *)(v12 + 8);
  LOWORD(v87) = v13;
  v88 = v14;
  if ( v13 )
  {
    if ( (unsigned __int16)(v13 - 176) > 0x34u )
      goto LABEL_10;
  }
  else if ( !sub_3007100((__int64)&v87) )
  {
LABEL_16:
    v20 = sub_3007130((__int64)&v87, v4);
    v15 = v85;
    v82 = v20;
    v17 = v86;
    if ( !(_WORD)v85 )
      goto LABEL_67;
    LOWORD(v97) = v85;
    v16 = 0;
    v98 = v86;
LABEL_18:
    if ( v15 == 1 || (unsigned __int16)(v15 - 504) <= 7u )
      goto LABEL_80;
    v21 = *(_QWORD *)&byte_444C4A0[16 * v15 - 16];
    v4 = (unsigned __int8)byte_444C4A0[16 * v15 - 8];
    if ( v16 )
      goto LABEL_62;
    goto LABEL_21;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( !(_WORD)v87 )
    goto LABEL_16;
  if ( (unsigned __int16)(v87 - 176) <= 0x34u )
    sub_CA17B0(
      "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT::"
      "getVectorElementCount() instead");
LABEL_10:
  v15 = v85;
  v16 = v87;
  v82 = word_4456340[(unsigned __int16)v87 - 1];
  v17 = v86;
  if ( (_WORD)v85 != (_WORD)v87 )
  {
    LOWORD(v97) = v85;
    v98 = v86;
    if ( !(_WORD)v85 )
      goto LABEL_61;
    goto LABEL_18;
  }
  if ( !(_WORD)v85 )
  {
LABEL_67:
    if ( v88 == v17 )
      goto LABEL_46;
    v98 = v17;
    v16 = 0;
    LOWORD(v97) = 0;
LABEL_61:
    v93 = sub_3007260((__int64)&v97);
    v21 = v93;
    v94 = v66;
    v4 = (unsigned __int8)v66;
    if ( v16 )
    {
LABEL_62:
      if ( v16 == 1 || (unsigned __int16)(v16 - 504) <= 7u )
        goto LABEL_80;
      v26 = *(_QWORD *)&byte_444C4A0[16 * v16 - 16];
      LOBYTE(v25) = byte_444C4A0[16 * v16 - 8];
LABEL_22:
      if ( (_BYTE)v25 && !(_BYTE)v4 || v21 < v26 )
        goto LABEL_24;
      if ( v15 )
        goto LABEL_12;
LABEL_46:
      v18 = sub_3007260((__int64)&v85);
      v95 = v18;
      v96 = v19;
      goto LABEL_47;
    }
LABEL_21:
    v80 = v21;
    v22 = sub_3007260((__int64)&v87);
    v4 = (unsigned __int8)v4;
    v21 = v80;
    v23 = v22;
    v25 = v24;
    v91 = v23;
    v26 = v23;
    v92 = v25;
    goto LABEL_22;
  }
LABEL_12:
  if ( v15 == 1 || (unsigned __int16)(v15 - 504) <= 7u )
    goto LABEL_80;
  v19 = 16LL * (v15 - 1);
  v18 = *(_QWORD *)&byte_444C4A0[v19];
  LOBYTE(v19) = byte_444C4A0[v19 + 8];
LABEL_47:
  v89 = v18;
  v90 = v19;
  v43 = sub_CA1930(&v89);
  v44 = v87;
  v45 = v43;
  if ( (_WORD)v87 )
  {
    if ( (unsigned __int16)(v87 - 17) > 0xD3u )
      goto LABEL_49;
    v44 = word_4456580[(unsigned __int16)v87 - 1];
    v49 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v87) )
    {
LABEL_49:
      v49 = v88;
      goto LABEL_50;
    }
    v44 = sub_3009970((__int64)&v87, v4, v46, v47, v48);
  }
LABEL_50:
  LOWORD(v99) = v44;
  v100 = v49;
  if ( !v44 )
  {
    v97 = sub_3007260((__int64)&v99);
    v50 = v97;
    v98 = v51;
    goto LABEL_52;
  }
  if ( v44 == 1 || (unsigned __int16)(v44 - 504) <= 7u )
LABEL_80:
    BUG();
  v50 = *(_QWORD *)&byte_444C4A0[16 * v44 - 16];
LABEL_52:
  v52 = v87;
  v82 = v45 / v50;
  v53 = v45 / v50;
  if ( (_WORD)v87 )
  {
    if ( (unsigned __int16)(v87 - 17) <= 0xD3u )
    {
      v77 = 0;
      v52 = word_4456580[(unsigned __int16)v87 - 1];
      goto LABEL_55;
    }
  }
  else if ( sub_30070B0((__int64)&v87) )
  {
    v70 = sub_3009970((__int64)&v87, v4, v67, v68, v69);
    v77 = v71;
    v52 = v70;
    goto LABEL_55;
  }
  v77 = v88;
LABEL_55:
  v54 = *(__int64 **)(*a1 + 64);
  v55 = sub_2D43050(v52, v53);
  v56 = 0;
  if ( !v55 )
  {
    v55 = sub_3009400(v54, v52, v77, (unsigned int)v53, 0);
    v56 = v72;
  }
  v57 = (_QWORD *)*a1;
  v88 = v56;
  LOWORD(v87) = v55;
  v58 = sub_3400EE0((__int64)v57, 0, (__int64)&v83, 0, (__m128i)v11);
  v59 = (_QWORD *)*a1;
  v61 = v60;
  v62 = v58;
  v99 = 0;
  LODWORD(v100) = 0;
  *(_QWORD *)&v63 = sub_33F17F0(v59, 51, (__int64)&v99, v87, v88);
  if ( v99 )
  {
    v78 = v63;
    sub_B91220((__int64)&v99, (__int64)v99);
    v63 = v78;
  }
  *((_QWORD *)&v73 + 1) = v61;
  *(_QWORD *)&v73 = v62;
  v10 = sub_340F900(v57, 0xA0u, (__int64)&v83, v87, v88, v64, v63, v11, v73);
  v9 = v65;
LABEL_24:
  v27 = sub_3400BD0(*a1, 0, (__int64)&v83, (unsigned int)v87, v88, 0, (__m128i)v11, 0);
  v99 = v101;
  v28 = v101;
  v29 = (__int64)v27;
  v30 = v82;
  v32 = v31;
  v33 = 0;
  v100 = 0x1000000000LL;
  if ( (unsigned __int64)v82 > 0x10 )
  {
    v74 = v29;
    v75 = v32;
    sub_C8D5F0((__int64)&v99, v101, v82, 4u, v29, v32);
    v33 = v100;
    v29 = v74;
    v32 = v75;
    v30 = v82;
    v28 = &v99[4 * (unsigned int)v100];
  }
  if ( v30 > 0 )
  {
    for ( i = 0; i != v30; ++i )
      *(_DWORD *)&v28[4 * i] = i;
    v33 = v100;
  }
  v76 = v29;
  LODWORD(v100) = v33 + v30;
  v79 = v32;
  v35 = (_BYTE *)sub_2E79000(*(__int64 **)(*a1 + 40));
  v36 = v82 / v81 - 1;
  if ( !*v35 )
    v36 = 0;
  if ( v81 > 0 )
  {
    v37 = v82;
    v38 = 4LL * v36;
    do
    {
      *(_DWORD *)&v99[v38] = v37++;
      v38 += 4LL * (v82 / v81);
    }
    while ( v37 != v82 + v81 );
  }
  v39 = *a1;
  sub_33FCE10(
    v39,
    (unsigned int)v87,
    v88,
    (__int64)&v83,
    v76,
    v79,
    (__m128i)v11,
    v10,
    v9 | *((_QWORD *)&v11 + 1) & 0xFFFFFFFF00000000LL,
    v99,
    (unsigned int)v100);
  v41 = sub_33FAF80(v39, 234, (__int64)&v83, v85, v86, v40, (__m128i)v11);
  if ( v99 != v101 )
    _libc_free((unsigned __int64)v99);
  if ( v83 )
    sub_B91220((__int64)&v83, v83);
  return v41;
}
