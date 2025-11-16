// Function: sub_3768330
// Address: 0x3768330
//
unsigned __int8 *__fastcall sub_3768330(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rsi
  __int16 *v9; // rdx
  __int16 v10; // ax
  __int64 v11; // rdx
  int v12; // ebx
  const __m128i *v13; // roff
  __int128 v14; // xmm0
  __int64 v15; // r14
  __int64 v16; // rax
  __int16 v17; // dx
  __int64 v18; // rax
  unsigned __int16 v19; // r13
  unsigned __int16 v20; // r15
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  char v24; // al
  int v25; // eax
  unsigned __int64 v26; // rcx
  __int64 v27; // rax
  __int64 v28; // rdi
  __int64 v29; // rdx
  __int64 v30; // rax
  unsigned __int64 v31; // rdx
  __int64 v32; // r8
  _BYTE *v33; // rdi
  int v34; // eax
  _BYTE *v35; // rax
  int v36; // edx
  int v37; // eax
  __int64 v38; // rdx
  _QWORD *v39; // r12
  _QWORD *v40; // rax
  __int64 v41; // rdx
  int v42; // r9d
  unsigned __int8 *result; // rax
  __int64 v44; // rdx
  __int64 v45; // rax
  unsigned __int16 v46; // r14
  unsigned __int64 v47; // r13
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // rdx
  unsigned __int64 v52; // rcx
  __int64 v53; // rdx
  unsigned __int64 v54; // rax
  unsigned __int16 v55; // cx
  unsigned int v56; // r13d
  unsigned int v57; // r14d
  __int64 *v58; // r15
  __int16 v59; // ax
  __int64 v60; // rcx
  __int64 v61; // r9
  unsigned __int8 *v62; // rax
  _QWORD *v63; // rdi
  __int64 v64; // rdx
  __int64 v65; // r15
  unsigned __int8 *v66; // r14
  __int128 v67; // rax
  _QWORD *v68; // r9
  unsigned int v69; // edx
  __int64 v70; // rdx
  bool v71; // al
  __int64 v72; // rdx
  __int64 v73; // r8
  unsigned __int16 v74; // ax
  __int64 v75; // rdx
  __int64 v76; // rdx
  __int128 v77; // [rsp-10h] [rbp-170h]
  __int128 v78; // [rsp+0h] [rbp-160h]
  unsigned __int64 v79; // [rsp+10h] [rbp-150h]
  int v80; // [rsp+10h] [rbp-150h]
  _QWORD *v81; // [rsp+10h] [rbp-150h]
  __int64 v82; // [rsp+18h] [rbp-148h]
  __int64 v83; // [rsp+28h] [rbp-138h]
  __int64 v84; // [rsp+28h] [rbp-138h]
  _QWORD *v85; // [rsp+28h] [rbp-138h]
  int v86; // [rsp+30h] [rbp-130h]
  int v87; // [rsp+30h] [rbp-130h]
  const void *v88; // [rsp+30h] [rbp-130h]
  __int64 v89; // [rsp+38h] [rbp-128h]
  unsigned __int8 *v90; // [rsp+40h] [rbp-120h]
  unsigned __int8 *v91; // [rsp+40h] [rbp-120h]
  __int64 v92; // [rsp+50h] [rbp-110h] BYREF
  int v93; // [rsp+58h] [rbp-108h]
  unsigned int v94; // [rsp+60h] [rbp-100h] BYREF
  __int64 v95; // [rsp+68h] [rbp-F8h]
  __int64 v96; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v97; // [rsp+78h] [rbp-E8h]
  __int64 v98; // [rsp+80h] [rbp-E0h] BYREF
  char v99; // [rsp+88h] [rbp-D8h]
  __int64 v100; // [rsp+90h] [rbp-D0h]
  __int64 v101; // [rsp+98h] [rbp-C8h]
  __int64 v102; // [rsp+A0h] [rbp-C0h]
  __int64 v103; // [rsp+A8h] [rbp-B8h]
  __int64 v104; // [rsp+B0h] [rbp-B0h]
  __int64 v105; // [rsp+B8h] [rbp-A8h]
  __int64 v106; // [rsp+C0h] [rbp-A0h]
  __int64 v107; // [rsp+C8h] [rbp-98h]
  __int64 v108; // [rsp+D0h] [rbp-90h] BYREF
  __int64 v109; // [rsp+D8h] [rbp-88h]
  _BYTE *v110; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v111; // [rsp+E8h] [rbp-78h]
  _BYTE s[112]; // [rsp+F0h] [rbp-70h] BYREF

  v8 = *(_QWORD *)(a2 + 80);
  v92 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v92, v8, 1);
  v9 = *(__int16 **)(a2 + 48);
  v93 = *(_DWORD *)(a2 + 72);
  v10 = *v9;
  v11 = *((_QWORD *)v9 + 1);
  LOWORD(v94) = v10;
  v95 = v11;
  if ( v10 )
  {
    if ( (unsigned __int16)(v10 - 176) > 0x34u )
    {
LABEL_5:
      v12 = word_4456340[(unsigned __int16)v94 - 1];
      goto LABEL_8;
    }
  }
  else if ( !sub_3007100((__int64)&v94) )
  {
    goto LABEL_7;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( (_WORD)v94 )
  {
    if ( (unsigned __int16)(v94 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    goto LABEL_5;
  }
LABEL_7:
  v12 = sub_3007130((__int64)&v94, v8);
LABEL_8:
  v13 = *(const __m128i **)(a2 + 40);
  v14 = (__int128)_mm_loadu_si128(v13);
  v15 = v13->m128i_i64[0];
  v83 = v13->m128i_u32[2];
  v16 = *(_QWORD *)(v13->m128i_i64[0] + 48) + 16 * v83;
  v17 = *(_WORD *)v16;
  v18 = *(_QWORD *)(v16 + 8);
  LOWORD(v96) = v17;
  v97 = v18;
  if ( v17 )
  {
    if ( (unsigned __int16)(v17 - 176) > 0x34u )
      goto LABEL_10;
  }
  else if ( !sub_3007100((__int64)&v96) )
  {
LABEL_16:
    v25 = sub_3007130((__int64)&v96, v8);
    v19 = v94;
    v86 = v25;
    v21 = v95;
    if ( !(_WORD)v94 )
      goto LABEL_69;
    LOWORD(v108) = v94;
    v20 = 0;
    v109 = v95;
LABEL_18:
    if ( v19 == 1 || (unsigned __int16)(v19 - 504) <= 7u )
      goto LABEL_82;
    v26 = *(_QWORD *)&byte_444C4A0[16 * v19 - 16];
    v8 = (unsigned __int8)byte_444C4A0[16 * v19 - 8];
    if ( v20 )
      goto LABEL_64;
    goto LABEL_21;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( !(_WORD)v96 )
    goto LABEL_16;
  if ( (unsigned __int16)(v96 - 176) <= 0x34u )
    sub_CA17B0(
      "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT::"
      "getVectorElementCount() instead");
LABEL_10:
  v19 = v94;
  v20 = v96;
  v86 = word_4456340[(unsigned __int16)v96 - 1];
  v21 = v95;
  if ( (_WORD)v96 != (_WORD)v94 )
  {
    LOWORD(v108) = v94;
    v109 = v95;
    if ( !(_WORD)v94 )
      goto LABEL_63;
    goto LABEL_18;
  }
  if ( !(_WORD)v94 )
  {
LABEL_69:
    if ( v97 == v21 )
      goto LABEL_48;
    v109 = v21;
    v20 = 0;
    LOWORD(v108) = 0;
LABEL_63:
    v102 = sub_3007260((__int64)&v108);
    v26 = v102;
    v103 = v70;
    v8 = (unsigned __int8)v70;
    if ( v20 )
    {
LABEL_64:
      if ( v20 == 1 || (unsigned __int16)(v20 - 504) <= 7u )
        goto LABEL_82;
      v31 = *(_QWORD *)&byte_444C4A0[16 * v20 - 16];
      LOBYTE(v30) = byte_444C4A0[16 * v20 - 8];
LABEL_22:
      if ( (_BYTE)v30 && !(_BYTE)v8 || v26 < v31 )
        goto LABEL_24;
      if ( v19 )
        goto LABEL_12;
LABEL_48:
      v104 = sub_3007260((__int64)&v94);
      v105 = v44;
      v23 = v104;
      v24 = v105;
      goto LABEL_49;
    }
LABEL_21:
    v79 = v26;
    v27 = sub_3007260((__int64)&v96);
    v8 = (unsigned __int8)v8;
    v26 = v79;
    v28 = v27;
    v30 = v29;
    v100 = v28;
    v31 = v28;
    v101 = v30;
    goto LABEL_22;
  }
LABEL_12:
  if ( v19 == 1 || (unsigned __int16)(v19 - 504) <= 7u )
    goto LABEL_82;
  v22 = 16LL * (v19 - 1);
  v23 = *(_QWORD *)&byte_444C4A0[v22];
  v24 = byte_444C4A0[v22 + 8];
LABEL_49:
  v98 = v23;
  v99 = v24;
  v45 = sub_CA1930(&v98);
  v46 = v96;
  v47 = v45;
  if ( (_WORD)v96 )
  {
    if ( (unsigned __int16)(v96 - 17) > 0xD3u )
      goto LABEL_51;
    v46 = word_4456580[(unsigned __int16)v96 - 1];
    v51 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v96) )
    {
LABEL_51:
      v51 = v97;
      goto LABEL_52;
    }
    v46 = sub_3009970((__int64)&v96, v8, v48, v49, v50);
  }
LABEL_52:
  LOWORD(v110) = v46;
  v111 = v51;
  if ( !v46 )
  {
    v106 = sub_3007260((__int64)&v110);
    v52 = v106;
    v107 = v53;
    goto LABEL_54;
  }
  if ( v46 == 1 || (unsigned __int16)(v46 - 504) <= 7u )
LABEL_82:
    BUG();
  v52 = *(_QWORD *)&byte_444C4A0[16 * v46 - 16];
LABEL_54:
  v54 = v47 / v52;
  v55 = v96;
  v86 = v54;
  v56 = v54;
  if ( (_WORD)v96 )
  {
    if ( (unsigned __int16)(v96 - 17) <= 0xD3u )
    {
      v84 = 0;
      v55 = word_4456580[(unsigned __int16)v96 - 1];
      goto LABEL_57;
    }
  }
  else
  {
    v71 = sub_30070B0((__int64)&v96);
    v55 = 0;
    if ( v71 )
    {
      v74 = sub_3009970((__int64)&v96, v8, v72, 0, v73);
      v84 = v75;
      v55 = v74;
      goto LABEL_57;
    }
  }
  v84 = v97;
LABEL_57:
  v57 = v55;
  v58 = *(__int64 **)(*a1 + 64);
  v59 = sub_2D43050(v55, v56);
  v60 = 0;
  if ( !v59 )
  {
    v59 = sub_3009400(v58, v57, v84, v56, 0);
    v60 = v76;
  }
  v61 = *a1;
  v97 = v60;
  LOWORD(v96) = v59;
  v85 = (_QWORD *)v61;
  v62 = sub_3400EE0(v61, 0, (__int64)&v92, 0, (__m128i)v14);
  v63 = (_QWORD *)*a1;
  v65 = v64;
  v66 = v62;
  v110 = 0;
  LODWORD(v111) = 0;
  *(_QWORD *)&v67 = sub_33F17F0(v63, 51, (__int64)&v110, v96, v97);
  v68 = v85;
  if ( v110 )
  {
    v78 = v67;
    sub_B91220((__int64)&v110, (__int64)v110);
    v67 = v78;
    v68 = v85;
  }
  *((_QWORD *)&v77 + 1) = v65;
  *(_QWORD *)&v77 = v66;
  v15 = sub_340F900(v68, 0xA0u, (__int64)&v92, v96, v97, (__int64)v68, v67, v14, v77);
  v83 = v69;
LABEL_24:
  v32 = v86;
  v110 = s;
  v111 = 0x1000000000LL;
  if ( v86 )
  {
    v33 = s;
    v34 = 0;
    if ( (unsigned __int64)v86 > 0x10 )
    {
      sub_C8D5F0((__int64)&v110, s, v86, 4u, v86, a6);
      v34 = v111;
      v32 = v86;
      v33 = &v110[4 * (unsigned int)v111];
    }
    if ( 4 * v32 )
    {
      v80 = v32;
      memset(v33, 255, 4 * v32);
      v34 = v111;
      LODWORD(v32) = v80;
    }
    LODWORD(v111) = v34 + v32;
  }
  v87 = v86 / v12;
  v35 = (_BYTE *)sub_2E79000(*(__int64 **)(*a1 + 40));
  v36 = v87 - 1;
  if ( !*v35 )
    v36 = 0;
  if ( v12 > 0 )
  {
    v37 = 0;
    v38 = 4LL * v36;
    do
    {
      *(_DWORD *)&v110[v38] = v37++;
      v38 += 4LL * v87;
    }
    while ( v12 != v37 );
  }
  v39 = (_QWORD *)*a1;
  v108 = 0;
  v88 = v110;
  LODWORD(v109) = 0;
  v89 = (unsigned int)v111;
  v40 = sub_33F17F0(v39, 51, (__int64)&v108, v96, v97);
  if ( v108 )
  {
    v81 = v40;
    v82 = v41;
    sub_B91220((__int64)&v108, v108);
    v40 = v81;
    v41 = v82;
  }
  sub_33FCE10(
    (__int64)v39,
    (unsigned int)v96,
    v97,
    (__int64)&v92,
    v15,
    v83 | *((_QWORD *)&v14 + 1) & 0xFFFFFFFF00000000LL,
    (__m128i)v14,
    (__int64)v40,
    v41,
    v88,
    v89);
  result = sub_33FAF80((__int64)v39, 234, (__int64)&v92, v94, v95, v42, (__m128i)v14);
  if ( v110 != s )
  {
    v90 = result;
    _libc_free((unsigned __int64)v110);
    result = v90;
  }
  if ( v92 )
  {
    v91 = result;
    sub_B91220((__int64)&v92, v92);
    return v91;
  }
  return result;
}
