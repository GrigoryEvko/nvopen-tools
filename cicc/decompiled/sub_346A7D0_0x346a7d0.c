// Function: sub_346A7D0
// Address: 0x346a7d0
//
unsigned __int8 *__fastcall sub_346A7D0(__int64 a1, __int64 a2, _QWORD *a3, __m128i a4)
{
  __int64 v5; // rsi
  int v6; // edi
  __int64 v7; // r8
  unsigned __int16 *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  int v11; // eax
  int v12; // ecx
  int v13; // eax
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 *v16; // r15
  __int64 v17; // r13
  unsigned __int16 v18; // cx
  char v19; // al
  unsigned int v20; // ebx
  unsigned __int16 v21; // ax
  __int64 v22; // r9
  __int64 v23; // rbx
  __int64 v24; // r13
  __int64 v25; // rax
  __int16 v26; // dx
  __int64 v27; // rax
  __m128i v28; // xmm1
  unsigned __int8 *v29; // rax
  int v30; // eax
  int v31; // r11d
  __int64 v32; // r15
  int v33; // r9d
  __int64 v34; // rax
  unsigned __int8 *v35; // r12
  unsigned __int64 v36; // r13
  __int64 v37; // r15
  unsigned int v38; // ebx
  unsigned int v39; // edx
  unsigned __int16 *v40; // rax
  __int64 v41; // r8
  __int64 v43; // rdx
  unsigned __int16 v44; // ax
  unsigned __int64 v45; // rax
  int v46; // eax
  bool v47; // al
  unsigned int v48; // eax
  __int128 v49; // [rsp-20h] [rbp-1D0h]
  __int128 v50; // [rsp-20h] [rbp-1D0h]
  __int128 v51; // [rsp-10h] [rbp-1C0h]
  __int128 v52; // [rsp-10h] [rbp-1C0h]
  __int16 v53; // [rsp+2h] [rbp-1AEh]
  unsigned int v55; // [rsp+18h] [rbp-198h]
  __int64 v56; // [rsp+18h] [rbp-198h]
  unsigned __int16 v57; // [rsp+18h] [rbp-198h]
  __int16 v58; // [rsp+1Ah] [rbp-196h]
  __int16 v59; // [rsp+1Ah] [rbp-196h]
  __int16 v60; // [rsp+1Ah] [rbp-196h]
  unsigned int v62; // [rsp+2Ch] [rbp-184h]
  __int64 v63; // [rsp+30h] [rbp-180h]
  unsigned __int64 v64; // [rsp+38h] [rbp-178h]
  __int16 v65; // [rsp+40h] [rbp-170h]
  unsigned __int64 v66; // [rsp+48h] [rbp-168h]
  __int32 v67; // [rsp+78h] [rbp-138h]
  __int64 v68; // [rsp+90h] [rbp-120h] BYREF
  int v69; // [rsp+98h] [rbp-118h]
  __m128i v70; // [rsp+A0h] [rbp-110h] BYREF
  __int64 v71; // [rsp+B0h] [rbp-100h] BYREF
  __int64 v72; // [rsp+B8h] [rbp-F8h]
  __m128i v73; // [rsp+C0h] [rbp-F0h] BYREF
  __m128i v74; // [rsp+D0h] [rbp-E0h] BYREF
  __int64 v75[2]; // [rsp+E0h] [rbp-D0h] BYREF
  __m128i v76; // [rsp+F0h] [rbp-C0h] BYREF
  __m128i v77; // [rsp+100h] [rbp-B0h] BYREF

  v5 = *(_QWORD *)(a2 + 80);
  v68 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v68, v5, 1);
  v6 = *(_DWORD *)(a2 + 24);
  v69 = *(_DWORD *)(a2 + 72);
  v62 = sub_33CB000(v6);
  v70 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  v8 = (unsigned __int16 *)(*(_QWORD *)(v70.m128i_i64[0] + 48) + 16LL * v70.m128i_u32[2]);
  LODWORD(v9) = *v8;
  v10 = *((_QWORD *)v8 + 1);
  LOWORD(v71) = v9;
  v72 = v10;
  if ( (_WORD)v9 )
  {
    if ( (unsigned __int16)(v9 - 176) > 0x34u )
    {
      v11 = (unsigned __int16)v9;
      v12 = word_4456340[(unsigned __int16)v9 - 1];
      v5 = (unsigned int)(v12 - 1);
      if ( (v12 & (unsigned int)v5) == 0 )
        goto LABEL_21;
LABEL_27:
      v30 = v11 - 1;
      v63 = 0;
      v31 = (unsigned __int16)word_4456580[v30];
      v65 = word_4456580[v30];
LABEL_28:
      if ( (unsigned __int16)(v9 - 176) <= 0x34u )
        goto LABEL_58;
LABEL_29:
      v32 = word_4456340[(unsigned __int16)v71 - 1];
      goto LABEL_30;
    }
LABEL_62:
    sub_C64ED0("Expanding reductions for scalable vectors is undefined.", 1u);
  }
  v55 = v9;
  if ( sub_3007100((__int64)&v71) )
    goto LABEL_62;
  v13 = sub_3007240((__int64)&v71);
  v9 = v55;
  v14 = (unsigned int)(v13 - 1);
  if ( (v13 & (unsigned int)v14) != 0 )
    goto LABEL_55;
  while ( 1 )
  {
LABEL_21:
    if ( (_WORD)v9 )
    {
      if ( (unsigned __int16)(v9 - 176) > 0x34u )
        goto LABEL_11;
    }
    else if ( !sub_3007100((__int64)&v71) )
    {
      if ( (unsigned int)sub_3007130((__int64)&v71, v5) <= 1 )
        break;
LABEL_48:
      v16 = (__int64 *)a3[8];
      goto LABEL_49;
    }
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( !(_WORD)v71 )
    {
      if ( (unsigned int)sub_3007130((__int64)&v71, v5) <= 1 )
        break;
      goto LABEL_48;
    }
    if ( (unsigned __int16)(v71 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
LABEL_11:
    v14 = (__int64)word_4456340;
    v9 = (unsigned __int16)v71;
    v15 = (unsigned __int16)v71 - 1;
    v5 = word_4456340[v15];
    if ( (unsigned __int16)v5 <= 1u )
      goto LABEL_25;
    v16 = (__int64 *)a3[8];
    if ( (_WORD)v71 )
    {
      v17 = 0;
      v18 = word_4456580[v15];
LABEL_14:
      v19 = (unsigned __int16)(v9 - 176) <= 0x34u;
      LOBYTE(v9) = v19;
      goto LABEL_15;
    }
LABEL_49:
    v44 = sub_3009970((__int64)&v71, v5, v9, v14, v7);
    v17 = v9;
    LOWORD(v9) = v71;
    v18 = v44;
    if ( (_WORD)v71 )
    {
      LODWORD(v5) = word_4456340[(unsigned __int16)v71 - 1];
      goto LABEL_14;
    }
    v57 = v44;
    v45 = sub_3007240((__int64)&v71);
    v18 = v57;
    LODWORD(v5) = v45;
    v9 = HIDWORD(v45);
    v19 = BYTE4(v45);
LABEL_15:
    v5 = (unsigned int)v5 >> 1;
    v76.m128i_i8[4] = v9;
    v20 = v18;
    v76.m128i_i32[0] = v5;
    if ( v19 )
    {
      v21 = sub_2D43AD0(v18, v5);
      if ( v21 )
        goto LABEL_17;
    }
    else
    {
      v21 = sub_2D43050(v18, v5);
      if ( v21 )
      {
LABEL_17:
        v23 = 0;
        v24 = v21;
        if ( v21 == 1 )
          goto LABEL_18;
        goto LABEL_44;
      }
    }
    v5 = v20;
    v21 = sub_3009450(v16, v20, v17, v76.m128i_i64[0], v7, v22);
    v24 = v21;
    v23 = v43;
    if ( v21 == 1 )
      goto LABEL_18;
    if ( !v21 )
      break;
LABEL_44:
    if ( !*(_QWORD *)(a1 + 8LL * v21 + 112) )
      break;
LABEL_18:
    if ( v62 <= 0x1F3 && (*(_BYTE *)(v62 + a1 + 500LL * v21 + 6414) & 0xFB) != 0 )
      break;
    v73.m128i_i16[0] = 0;
    v73.m128i_i64[1] = 0;
    v25 = *(_QWORD *)(v70.m128i_i64[0] + 48) + 16LL * v70.m128i_u32[2];
    v74.m128i_i16[0] = 0;
    v74.m128i_i64[1] = 0;
    v26 = *(_WORD *)v25;
    v27 = *(_QWORD *)(v25 + 8);
    LOWORD(v75[0]) = v26;
    v75[1] = v27;
    sub_33D0340((__int64)&v76, (__int64)a3, v75);
    a4 = _mm_loadu_si128(&v76);
    v28 = _mm_loadu_si128(&v77);
    v73 = a4;
    v74 = v28;
    sub_3408290(
      (__int64)&v76,
      a3,
      (__int128 *)v70.m128i_i8,
      (__int64)&v68,
      (unsigned int *)&v73,
      (unsigned int *)&v74,
      a4);
    v5 = v62;
    v66 = v76.m128i_u32[2] | v66 & 0xFFFFFFFF00000000LL;
    v64 = v77.m128i_u32[2] | v64 & 0xFFFFFFFF00000000LL;
    *((_QWORD *)&v51 + 1) = v64;
    *(_QWORD *)&v51 = v77.m128i_i64[0];
    *((_QWORD *)&v49 + 1) = v66;
    *(_QWORD *)&v49 = v76.m128i_i64[0];
    v29 = sub_3405C90(a3, v62, (__int64)&v68, (unsigned int)v24, v23, *(_DWORD *)(a2 + 28), a4, v49, v51);
    v71 = v24;
    v67 = v9;
    LOWORD(v9) = v24;
    v70.m128i_i64[0] = (__int64)v29;
    v72 = v23;
    v70.m128i_i32[2] = v67;
  }
  v9 = (unsigned __int16)v71;
LABEL_25:
  if ( (_WORD)v9 )
  {
    v11 = (unsigned __int16)v9;
    goto LABEL_27;
  }
LABEL_55:
  v46 = sub_3009970((__int64)&v71, v5, v9, v14, v7);
  v63 = v9;
  LOWORD(v9) = v71;
  HIWORD(v31) = HIWORD(v46);
  v65 = v46;
  if ( (_WORD)v71 )
    goto LABEL_28;
  v58 = HIWORD(v46);
  v47 = sub_3007100((__int64)&v71);
  HIWORD(v31) = v58;
  if ( !v47 )
    goto LABEL_57;
LABEL_58:
  v60 = HIWORD(v31);
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  HIWORD(v31) = v60;
  if ( (_WORD)v71 )
  {
    if ( (unsigned __int16)(v71 - 176) <= 0x34u )
    {
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
      HIWORD(v31) = v60;
    }
    goto LABEL_29;
  }
LABEL_57:
  v59 = HIWORD(v31);
  v48 = sub_3007130((__int64)&v71, v5);
  HIWORD(v31) = v59;
  v32 = v48;
LABEL_30:
  v76.m128i_i64[0] = (__int64)&v77;
  v76.m128i_i64[1] = 0x800000000LL;
  v53 = HIWORD(v31);
  sub_3408690(a3, v70.m128i_i64[0], v70.m128i_i64[1], (unsigned __int16 *)&v76, 0, v32, a4, 0, 0);
  v34 = v76.m128i_i64[0];
  v35 = *(unsigned __int8 **)v76.m128i_i64[0];
  v36 = *(_QWORD *)(v76.m128i_i64[0] + 8);
  if ( (unsigned int)v32 > 1 )
  {
    v56 = 16 * v32;
    v37 = 16;
    HIWORD(v38) = v53;
    while ( 1 )
    {
      v52 = *(_OWORD *)(v34 + v37);
      LOWORD(v38) = v65;
      v37 += 16;
      *((_QWORD *)&v50 + 1) = v36;
      *(_QWORD *)&v50 = v35;
      v35 = sub_3405C90(a3, v62, (__int64)&v68, v38, v63, *(_DWORD *)(a2 + 28), a4, v50, v52);
      v36 = v39 | v36 & 0xFFFFFFFF00000000LL;
      if ( v56 == v37 )
        break;
      v34 = v76.m128i_i64[0];
    }
  }
  v40 = *(unsigned __int16 **)(a2 + 48);
  v41 = *((_QWORD *)v40 + 1);
  if ( *v40 != v65 || !v65 && v41 != v63 )
    v35 = sub_33FAF80((__int64)a3, 215, (__int64)&v68, *v40, v41, v33, a4);
  if ( (__m128i *)v76.m128i_i64[0] != &v77 )
    _libc_free(v76.m128i_u64[0]);
  if ( v68 )
    sub_B91220((__int64)&v68, v68);
  return v35;
}
