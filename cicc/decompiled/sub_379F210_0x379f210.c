// Function: sub_379F210
// Address: 0x379f210
//
unsigned __int8 *__fastcall sub_379F210(__int64 *a1, __int64 a2)
{
  const __m128i *v4; // rax
  __m128i v5; // xmm0
  __int64 v6; // rsi
  __int64 v7; // r9
  __int64 (__fastcall *v8)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v9; // rax
  unsigned __int16 v10; // si
  __int64 v11; // r8
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rsi
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // rdx
  _WORD *v18; // rcx
  __int64 v19; // r13
  __int64 v20; // rax
  int v21; // r15d
  int v22; // eax
  __int64 v23; // rdx
  unsigned __int16 v24; // ax
  __int64 v25; // rdx
  unsigned __int16 v26; // ax
  unsigned int v27; // r14d
  __int64 v28; // rdx
  __int64 v29; // r9
  _QWORD *v30; // rdi
  unsigned int v31; // eax
  __int64 v32; // r15
  __int64 v33; // r8
  unsigned __int8 *v34; // r10
  unsigned int v35; // edx
  unsigned __int64 v36; // r11
  __int64 v37; // rax
  unsigned __int64 v38; // rdx
  unsigned __int8 **v39; // rax
  __int128 v40; // rax
  __int64 v41; // r9
  __int64 v42; // rdx
  int v43; // r9d
  unsigned __int8 *v44; // rax
  __int64 v45; // r11
  __int64 v46; // rdx
  unsigned __int8 *v47; // rax
  __int64 v48; // rcx
  unsigned int v49; // r12d
  __int128 v50; // rax
  __int64 v51; // r8
  __int64 v52; // rax
  unsigned __int64 v53; // rdx
  __int64 *v54; // rax
  unsigned __int8 *v55; // r14
  unsigned __int32 v57; // edx
  unsigned __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rax
  __int64 v61; // rdx
  __int64 v62; // rdx
  __int64 v63; // rdx
  __int64 v64; // rdx
  __int128 v65; // [rsp-10h] [rbp-230h]
  __int64 v66; // [rsp-8h] [rbp-228h]
  unsigned int v67; // [rsp+4h] [rbp-21Ch]
  __int64 v68; // [rsp+18h] [rbp-208h]
  __int64 v69; // [rsp+20h] [rbp-200h]
  __int64 v70; // [rsp+28h] [rbp-1F8h]
  unsigned int v71; // [rsp+30h] [rbp-1F0h]
  __int64 v72; // [rsp+38h] [rbp-1E8h]
  __int64 v73; // [rsp+40h] [rbp-1E0h]
  __int16 v74; // [rsp+4Ah] [rbp-1D6h]
  unsigned int v75; // [rsp+4Ch] [rbp-1D4h]
  unsigned __int32 v76; // [rsp+50h] [rbp-1D0h]
  unsigned __int8 *v77; // [rsp+50h] [rbp-1D0h]
  __int64 v78; // [rsp+58h] [rbp-1C8h]
  __int64 v79; // [rsp+58h] [rbp-1C8h]
  __int64 v80; // [rsp+58h] [rbp-1C8h]
  unsigned __int64 v81; // [rsp+58h] [rbp-1C8h]
  __int128 v82; // [rsp+60h] [rbp-1C0h]
  __int64 v83; // [rsp+60h] [rbp-1C0h]
  unsigned __int64 v84; // [rsp+68h] [rbp-1B8h]
  __int64 v85; // [rsp+68h] [rbp-1B8h]
  __int64 v86; // [rsp+A0h] [rbp-180h] BYREF
  int v87; // [rsp+A8h] [rbp-178h]
  __int64 v88; // [rsp+B0h] [rbp-170h] BYREF
  __int64 v89; // [rsp+B8h] [rbp-168h]
  unsigned __int16 v90; // [rsp+C0h] [rbp-160h] BYREF
  __int64 v91; // [rsp+C8h] [rbp-158h]
  _QWORD *v92; // [rsp+D0h] [rbp-150h] BYREF
  __int64 v93; // [rsp+D8h] [rbp-148h]
  _QWORD *v94; // [rsp+E0h] [rbp-140h] BYREF
  __int64 v95; // [rsp+E8h] [rbp-138h]
  _QWORD v96[38]; // [rsp+F0h] [rbp-130h] BYREF

  v67 = *(_DWORD *)(a2 + 24);
  v4 = *(const __m128i **)(a2 + 40);
  v5 = _mm_loadu_si128(v4);
  v73 = v4->m128i_i64[0];
  v6 = *(_QWORD *)(a2 + 80);
  v76 = v4->m128i_u32[2];
  v86 = v6;
  v84 = v5.m128i_u64[1];
  if ( v6 )
    sub_B96E90((__int64)&v86, v6, 1);
  v7 = *a1;
  v87 = *(_DWORD *)(a2 + 72);
  v8 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v7 + 592LL);
  v9 = *(__int16 **)(a2 + 48);
  v10 = *v9;
  v11 = *((_QWORD *)v9 + 1);
  v12 = a1[1];
  if ( v8 == sub_2D56A50 )
  {
    v13 = v10;
    v14 = v7;
    sub_2FE6CC0((__int64)&v94, v7, *(_QWORD *)(v12 + 64), v13, v11);
    v17 = (unsigned __int16)v95;
    LOWORD(v88) = v95;
    v89 = v96[0];
  }
  else
  {
    v63 = v10;
    v14 = *(_QWORD *)(v12 + 64);
    LODWORD(v88) = v8(v7, v14, v63, v11);
    v89 = v64;
    v17 = (unsigned int)v88;
  }
  if ( (_WORD)v17 )
  {
    v18 = word_4456580;
    v19 = 0;
    v20 = (unsigned __int16)v17 - 1;
    v21 = (unsigned __int16)word_4456580[v20];
    v74 = word_4456580[v20];
  }
  else
  {
    v22 = sub_3009970((__int64)&v88, v14, v17, v15, v16);
    v19 = v17;
    LOWORD(v17) = v88;
    HIWORD(v21) = HIWORD(v22);
    v74 = v22;
    if ( !(_WORD)v88 )
    {
      if ( !sub_3007100((__int64)&v88) )
        goto LABEL_11;
      goto LABEL_53;
    }
  }
  if ( (unsigned __int16)(v17 - 176) > 0x34u )
  {
LABEL_8:
    v75 = word_4456340[(unsigned __int16)v88 - 1];
    goto LABEL_12;
  }
LABEL_53:
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( (_WORD)v88 )
  {
    if ( (unsigned __int16)(v88 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    goto LABEL_8;
  }
LABEL_11:
  v75 = sub_3007130((__int64)&v88, v14);
LABEL_12:
  v23 = *(_QWORD *)(v73 + 48) + 16LL * v76;
  v24 = *(_WORD *)v23;
  v25 = *(_QWORD *)(v23 + 8);
  v90 = v24;
  v91 = v25;
  if ( v24 )
  {
    v70 = 0;
    v14 = (unsigned __int16)word_4456580[v24 - 1];
    v71 = (unsigned __int16)word_4456580[v24 - 1];
  }
  else
  {
    v71 = sub_3009970((__int64)&v90, v14, v25, (__int64)v18, v16);
    v24 = v90;
    v70 = v28;
    if ( !v90 )
    {
      if ( !sub_3007100((__int64)&v90) )
        goto LABEL_18;
      goto LABEL_51;
    }
  }
  if ( (unsigned __int16)(v24 - 176) <= 0x34u )
  {
LABEL_51:
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( v90 )
    {
      if ( (unsigned __int16)(v90 - 176) <= 0x34u )
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
      goto LABEL_15;
    }
LABEL_18:
    v27 = sub_3007130((__int64)&v90, v14);
    v26 = 0;
    goto LABEL_19;
  }
LABEL_15:
  v26 = v90;
  v27 = word_4456340[v90 - 1];
LABEL_19:
  sub_2FE6CC0((__int64)&v94, *a1, *(_QWORD *)(a1[1] + 64), v26, v91);
  if ( (_BYTE)v94 != 7 )
    goto LABEL_20;
  v73 = sub_379AB60((__int64)a1, v5.m128i_u64[0], v5.m128i_i64[1]);
  v76 = v57;
  v58 = v57 | v5.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  v59 = *(_QWORD *)(v73 + 48) + 16LL * v57;
  v84 = v58;
  v60 = *(_QWORD *)(v59 + 8);
  v90 = *(_WORD *)v59;
  v91 = v60;
  v94 = (_QWORD *)sub_2D5B750((unsigned __int16 *)&v88);
  v95 = v61;
  v92 = (_QWORD *)sub_2D5B750(&v90);
  v93 = v62;
  if ( v92 == v94 )
  {
    v30 = (_QWORD *)a1[1];
    if ( (_BYTE)v93 == (_BYTE)v95 && v67 - 223 <= 2 )
    {
      v55 = sub_33FAF80((__int64)v30, v67, (__int64)&v86, (unsigned int)v88, v89, v29, v5);
      goto LABEL_48;
    }
  }
  else
  {
LABEL_20:
    v30 = (_QWORD *)a1[1];
  }
  v94 = v96;
  v95 = 0x1000000000LL;
  v31 = v27;
  if ( v27 > v75 )
    v31 = v75;
  v68 = v76;
  v69 = v31;
  if ( v31 )
  {
    v72 = v19;
    WORD1(v19) = HIWORD(v21);
    v32 = 0;
    while ( 1 )
    {
      *(_QWORD *)&v40 = sub_3400EE0((__int64)v30, v32, (__int64)&v86, 0, v5);
      v84 = v68 | v84 & 0xFFFFFFFF00000000LL;
      sub_3406EB0(v30, 0x9Eu, (__int64)&v86, v71, v70, v41, __PAIR128__(v84, v73), v40);
      switch ( v67 )
      {
        case 0xE0u:
          v66 = v42;
          LOWORD(v19) = v74;
          v80 = v42;
          v47 = sub_33FAF80(a1[1], 213, (__int64)&v86, (unsigned int)v19, v72, v43, v5);
          v45 = v80;
          v34 = v47;
          v46 = (unsigned int)v46;
          break;
        case 0xE1u:
          LOWORD(v19) = v74;
          v78 = v42;
          v34 = sub_33FAF80(a1[1], 214, (__int64)&v86, (unsigned int)v19, v72, v43, v5);
          v36 = v35 | v78 & 0xFFFFFFFF00000000LL;
          goto LABEL_26;
        case 0xDFu:
          v66 = v42;
          LOWORD(v19) = v74;
          v79 = v42;
          v44 = sub_33FAF80(a1[1], 215, (__int64)&v86, (unsigned int)v19, v72, v43, v5);
          v45 = v79;
          v34 = v44;
          v46 = (unsigned int)v46;
          break;
        default:
          BUG();
      }
      v33 = v66;
      v36 = v46 | v45 & 0xFFFFFFFF00000000LL;
LABEL_26:
      v37 = (unsigned int)v95;
      v38 = (unsigned int)v95 + 1LL;
      if ( v38 > HIDWORD(v95) )
      {
        v77 = v34;
        v81 = v36;
        sub_C8D5F0((__int64)&v94, v96, v38, 0x10u, v33, v29);
        v37 = (unsigned int)v95;
        v34 = v77;
        v36 = v81;
      }
      v39 = (unsigned __int8 **)&v94[2 * v37];
      ++v32;
      *v39 = v34;
      v39[1] = (unsigned __int8 *)v36;
      v31 = v95 + 1;
      LODWORD(v95) = v95 + 1;
      if ( v32 == v69 )
      {
        HIWORD(v21) = WORD1(v19);
        v30 = (_QWORD *)a1[1];
        v19 = v72;
        v48 = v31;
        goto LABEL_37;
      }
      v30 = (_QWORD *)a1[1];
    }
  }
  v48 = 0;
LABEL_37:
  if ( v75 != v31 )
  {
    HIWORD(v49) = HIWORD(v21);
    while ( 1 )
    {
      LOWORD(v49) = v74;
      v92 = 0;
      LODWORD(v93) = 0;
      *(_QWORD *)&v50 = sub_33F17F0(v30, 51, (__int64)&v92, v49, v19);
      v29 = *((_QWORD *)&v50 + 1);
      v51 = v50;
      if ( v92 )
      {
        v82 = v50;
        sub_B91220((__int64)&v92, (__int64)v92);
        v29 = *((_QWORD *)&v82 + 1);
        v51 = v82;
      }
      v52 = (unsigned int)v95;
      v53 = (unsigned int)v95 + 1LL;
      if ( v53 > HIDWORD(v95) )
      {
        v83 = v51;
        v85 = v29;
        sub_C8D5F0((__int64)&v94, v96, v53, 0x10u, v51, v29);
        v52 = (unsigned int)v95;
        v29 = v85;
        v51 = v83;
      }
      v54 = &v94[2 * v52];
      *v54 = v51;
      v54[1] = v29;
      LODWORD(v95) = v95 + 1;
      if ( v75 == (_DWORD)v95 )
        break;
      v30 = (_QWORD *)a1[1];
    }
    v48 = v75;
    v30 = (_QWORD *)a1[1];
  }
  *((_QWORD *)&v65 + 1) = v48;
  *(_QWORD *)&v65 = v94;
  v55 = sub_33FC220(v30, 156, (__int64)&v86, v88, v89, v29, v65);
  if ( v94 != v96 )
    _libc_free((unsigned __int64)v94);
LABEL_48:
  if ( v86 )
    sub_B91220((__int64)&v86, v86);
  return v55;
}
