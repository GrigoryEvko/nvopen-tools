// Function: sub_303D690
// Address: 0x303d690
//
__int64 __fastcall sub_303D690(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // r14
  unsigned int v10; // ecx
  __int64 v11; // rdi
  __int64 v12; // rax
  __int16 v13; // dx
  __int64 v14; // rax
  int v15; // eax
  char v16; // cl
  __int64 v17; // r13
  __int64 v19; // rax
  __int64 *v20; // rdi
  __int64 v21; // r13
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rsi
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // r8
  __int64 v31; // r9
  unsigned __int32 v32; // r13d
  __m128i v33; // xmm3
  unsigned __int32 v34; // eax
  unsigned int v35; // edx
  unsigned int v36; // r13d
  unsigned __int64 v37; // r15
  __int64 v38; // rax
  unsigned __int64 v39; // rdx
  __int64 *v40; // rax
  __int128 v41; // rax
  int v42; // r9d
  __int128 v43; // rax
  unsigned int v44; // edx
  const __m128i *v45; // r13
  __int64 v46; // rsi
  __int64 v47; // r14
  const __m128i *v48; // rax
  const __m128i *v49; // r13
  unsigned __int64 v50; // r14
  unsigned __int64 v51; // rdx
  unsigned __int64 v52; // rdx
  __m128i *v53; // rcx
  unsigned int v54; // eax
  __int64 v55; // r14
  __int64 v56; // r13
  int v57; // eax
  int v58; // edx
  __int64 v59; // r10
  unsigned __int64 v60; // r11
  int v61; // r13d
  __int64 v62; // r14
  unsigned int v63; // ebx
  int v64; // r9d
  __int64 v65; // rax
  __int64 v66; // rdx
  unsigned __int64 v67; // rdx
  __int64 *v68; // rax
  _BYTE *v69; // rdi
  __int128 v70; // [rsp-20h] [rbp-230h]
  __int128 v71; // [rsp-20h] [rbp-230h]
  __int64 v72; // [rsp+0h] [rbp-210h]
  __int64 v73; // [rsp+0h] [rbp-210h]
  __int64 v74; // [rsp+8h] [rbp-208h]
  __int64 v75; // [rsp+10h] [rbp-200h]
  int v76; // [rsp+1Ch] [rbp-1F4h]
  int v77; // [rsp+28h] [rbp-1E8h]
  unsigned __int8 v78; // [rsp+60h] [rbp-1B0h]
  unsigned __int64 v79; // [rsp+60h] [rbp-1B0h]
  int v80; // [rsp+60h] [rbp-1B0h]
  unsigned int v81; // [rsp+68h] [rbp-1A8h]
  __int64 v82; // [rsp+68h] [rbp-1A8h]
  __int64 v83; // [rsp+68h] [rbp-1A8h]
  __m128i v84; // [rsp+70h] [rbp-1A0h]
  __int64 v85; // [rsp+70h] [rbp-1A0h]
  __int64 v86; // [rsp+70h] [rbp-1A0h]
  const __m128i *v87; // [rsp+70h] [rbp-1A0h]
  __int64 v88; // [rsp+78h] [rbp-198h]
  __int64 v89; // [rsp+78h] [rbp-198h]
  unsigned __int64 v90; // [rsp+78h] [rbp-198h]
  __int64 v91; // [rsp+90h] [rbp-180h] BYREF
  int v92; // [rsp+98h] [rbp-178h]
  int v93; // [rsp+A0h] [rbp-170h] BYREF
  __int64 v94; // [rsp+A8h] [rbp-168h]
  __int64 v95; // [rsp+B0h] [rbp-160h]
  __int64 v96; // [rsp+B8h] [rbp-158h]
  __m128i v97; // [rsp+C0h] [rbp-150h] BYREF
  __int64 v98; // [rsp+D0h] [rbp-140h]
  __m128i v99; // [rsp+E0h] [rbp-130h] BYREF
  __int64 v100; // [rsp+F0h] [rbp-120h]
  char v101; // [rsp+F8h] [rbp-118h]
  _BYTE *v102; // [rsp+100h] [rbp-110h] BYREF
  __int64 v103; // [rsp+108h] [rbp-108h]
  _BYTE v104[64]; // [rsp+110h] [rbp-100h] BYREF
  _OWORD *v105; // [rsp+150h] [rbp-C0h] BYREF
  __int64 v106; // [rsp+158h] [rbp-B8h]
  _OWORD v107[11]; // [rsp+160h] [rbp-B0h] BYREF

  v6 = a2;
  v7 = *(_QWORD *)(a2 + 40);
  v8 = *(_QWORD *)(a2 + 80);
  v91 = v8;
  v9 = *(_QWORD *)(v7 + 40);
  v10 = *(_DWORD *)(v7 + 48);
  v84 = _mm_loadu_si128((const __m128i *)(v7 + 40));
  if ( v8 )
  {
    v81 = *(_DWORD *)(v7 + 48);
    sub_B96E90((__int64)&v91, v8, 1);
    v10 = v81;
  }
  v11 = *(_QWORD *)(v6 + 112);
  v92 = *(_DWORD *)(v6 + 72);
  v82 = v10;
  v12 = *(_QWORD *)(v9 + 48) + 16LL * v10;
  v13 = *(_WORD *)v12;
  v14 = *(_QWORD *)(v12 + 8);
  LOWORD(v93) = v13;
  v94 = v14;
  v15 = sub_2EAC1E0(v11);
  v16 = 0;
  if ( v15 == 1 )
  {
    v19 = *(_QWORD *)(a1 + 537016);
    if ( *(_DWORD *)(v19 + 344) > 0x63u )
      v16 = *(_DWORD *)(v19 + 336) > 0x57u;
  }
  sub_3035DC0((__int64)&v99, v93, v94, v16);
  if ( !v101 )
    goto LABEL_5;
  v20 = *(__int64 **)(a4 + 40);
  v97 = _mm_load_si128(&v99);
  v98 = v100;
  v21 = sub_2E79000(v20);
  v78 = sub_2EAC4F0(*(_QWORD *)(v6 + 112));
  v26 = sub_3007410((__int64)&v93, *(__int64 **)(a4 + 64), v22, v23, v24, v25);
  if ( v78 < (unsigned __int8)sub_AE5260(v21, v26) )
    goto LABEL_5;
  if ( v97.m128i_i16[4] )
  {
    if ( v97.m128i_i16[4] == 1 || (unsigned __int16)(v97.m128i_i16[4] - 504) <= 7u )
      BUG();
    v28 = 16LL * (v97.m128i_u16[4] - 1);
    v27 = *(_QWORD *)&byte_444C4A0[v28];
    LOBYTE(v28) = byte_444C4A0[v28 + 8];
  }
  else
  {
    v27 = sub_3007260((__int64)&v97.m128i_i64[1]);
    v95 = v27;
    v96 = v28;
  }
  v105 = (_OWORD *)v27;
  LOBYTE(v106) = v28;
  v29 = sub_CA1930(&v105);
  v32 = v97.m128i_i32[0];
  v79 = v29;
  switch ( v97.m128i_i32[0] )
  {
    case 4:
      v76 = 554;
      break;
    case 8:
      v76 = 555;
      break;
    case 2:
      v76 = 553;
      break;
    default:
LABEL_5:
      v17 = 0;
      goto LABEL_6;
  }
  v105 = v107;
  v106 = 0x800000000LL;
  v33 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v6 + 40));
  LODWORD(v106) = 1;
  v107[0] = v33;
  if ( (_WORD)v93 )
  {
    if ( (unsigned __int16)(v93 - 176) > 0x34u )
    {
LABEL_20:
      v34 = word_4456340[(unsigned __int16)v93 - 1];
      goto LABEL_21;
    }
  }
  else if ( !sub_3007100((__int64)&v93) )
  {
    goto LABEL_39;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( (_WORD)v93 )
  {
    if ( (unsigned __int16)(v93 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    goto LABEL_20;
  }
LABEL_39:
  v34 = sub_3007130((__int64)&v93, v26);
LABEL_21:
  if ( v32 < v34 )
  {
    if ( v97.m128i_i16[4] )
    {
      if ( (unsigned __int16)(v97.m128i_i16[4] - 176) > 0x34u )
      {
LABEL_50:
        v80 = word_4456340[v97.m128i_u16[4] - 1];
        goto LABEL_51;
      }
    }
    else if ( !sub_3007100((__int64)&v97.m128i_i64[1]) )
    {
      goto LABEL_60;
    }
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( v97.m128i_i16[4] )
    {
      if ( (unsigned __int16)(v97.m128i_i16[4] - 176) <= 0x34u )
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
      goto LABEL_50;
    }
LABEL_60:
    v80 = sub_3007130((__int64)&v97.m128i_i64[1], v26);
LABEL_51:
    if ( v97.m128i_i32[0] )
    {
      v61 = 0;
      v77 = v9;
      v62 = v72;
      v75 = v6;
      v63 = 0;
      do
      {
        LOWORD(v62) = 0;
        v102 = v104;
        v103 = 0x400000000LL;
        v84.m128i_i64[1] = v82 | v84.m128i_i64[1] & 0xFFFFFFFF00000000LL;
        sub_3408690(a4, v77, v82, (unsigned int)&v102, v61, v80, v62, 0);
        *((_QWORD *)&v71 + 1) = (unsigned int)v103;
        *(_QWORD *)&v71 = v102;
        v30 = sub_33FC220(a4, 156, (unsigned int)&v91, v97.m128i_i32[2], v98, v64, v71);
        v65 = (unsigned int)v106;
        v31 = v66;
        v67 = (unsigned int)v106 + 1LL;
        if ( v67 > HIDWORD(v106) )
        {
          v73 = v30;
          v74 = v31;
          sub_C8D5F0((__int64)&v105, v107, v67, 0x10u, v30, v31);
          v65 = (unsigned int)v106;
          v30 = v73;
          v31 = v74;
        }
        v68 = (__int64 *)&v105[v65];
        *v68 = v30;
        v69 = v102;
        v68[1] = v31;
        LODWORD(v106) = v106 + 1;
        if ( v69 != v104 )
          _libc_free((unsigned __int64)v69);
        ++v63;
        v61 += v80;
      }
      while ( v97.m128i_i32[0] > v63 );
      v6 = v75;
      v35 = v106;
    }
    else
    {
      v35 = v106;
    }
    goto LABEL_29;
  }
  v35 = v106;
  v36 = 0;
  if ( v97.m128i_i32[0] )
  {
    v37 = v84.m128i_u64[1];
    do
    {
      *(_QWORD *)&v41 = sub_3400D50(a4, v36, &v91, 0);
      v37 = v82 | v37 & 0xFFFFFFFF00000000LL;
      *((_QWORD *)&v70 + 1) = v37;
      *(_QWORD *)&v70 = v9;
      *(_QWORD *)&v43 = sub_3406EB0(a4, 158, (unsigned int)&v91, v97.m128i_i32[2], v98, v42, v70, v41);
      v60 = *((_QWORD *)&v43 + 1);
      v59 = v43;
      if ( v79 <= 0xF )
      {
        v88 = *((_QWORD *)&v43 + 1);
        v59 = sub_33FAF80(a4, 215, (unsigned int)&v91, 6, 0, v31, v43);
        v60 = v44 | v88 & 0xFFFFFFFF00000000LL;
      }
      v38 = (unsigned int)v106;
      v39 = (unsigned int)v106 + 1LL;
      if ( v39 > HIDWORD(v106) )
      {
        v86 = v59;
        v90 = v60;
        sub_C8D5F0((__int64)&v105, v107, v39, 0x10u, v30, v31);
        v38 = (unsigned int)v106;
        v60 = v90;
        v59 = v86;
      }
      v40 = (__int64 *)&v105[v38];
      ++v36;
      *v40 = v59;
      v40[1] = v60;
      v35 = v106 + 1;
      LODWORD(v106) = v106 + 1;
    }
    while ( v97.m128i_i32[0] > v36 );
  }
LABEL_29:
  v45 = *(const __m128i **)(v6 + 40);
  v46 = v35;
  v47 = 40LL * *(unsigned int *)(v6 + 64);
  v48 = (const __m128i *)((char *)v45 + v47);
  v49 = v45 + 5;
  v50 = 0xCCCCCCCCCCCCCCCDLL * ((v47 - 80) >> 3);
  v51 = v50 + v35;
  if ( v51 > HIDWORD(v106) )
  {
    v87 = v48;
    sub_C8D5F0((__int64)&v105, v107, v51, 0x10u, v30, v31);
    v46 = (unsigned int)v106;
    v48 = v87;
  }
  v52 = (unsigned __int64)v105;
  v53 = (__m128i *)&v105[v46];
  if ( v49 != v48 )
  {
    do
    {
      if ( v53 )
        *v53 = _mm_loadu_si128(v49);
      v49 = (const __m128i *)((char *)v49 + 40);
      ++v53;
    }
    while ( v48 != v49 );
    v52 = (unsigned __int64)v105;
    LODWORD(v46) = v106;
  }
  v54 = v50 + v46;
  v55 = *(_QWORD *)(v6 + 104);
  v85 = v52;
  v56 = *(unsigned __int16 *)(v6 + 96);
  v83 = *(_QWORD *)(v6 + 112);
  v89 = v54;
  LODWORD(v106) = v54;
  v57 = sub_33ED250(a4, 1, 0, v53);
  v17 = sub_33EA9D0(a4, v76, (unsigned int)&v91, v57, v58, v83, v85, v89, v56, v55);
  if ( v105 != v107 )
    _libc_free((unsigned __int64)v105);
LABEL_6:
  if ( v91 )
    sub_B91220((__int64)&v91, v91);
  return v17;
}
