// Function: sub_326D540
// Address: 0x326d540
//
__int64 __fastcall sub_326D540(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int16 *v7; // rax
  unsigned __int16 v8; // r12
  __int64 v9; // r13
  int v10; // edx
  unsigned int v11; // r14d
  __int16 *v12; // rax
  __int64 v13; // rax
  __m128i v14; // xmm1
  __int64 v15; // rdi
  __m128i v16; // xmm2
  __int64 v17; // rax
  __int64 v18; // rax
  __int16 v19; // dx
  __int64 v20; // rax
  unsigned int v21; // r15d
  unsigned int v22; // r12d
  int v23; // r13d
  unsigned int v24; // ebx
  __int64 v25; // r15
  _DWORD *v26; // rax
  __int64 v27; // r8
  _DWORD *v28; // rcx
  __int64 v29; // rsi
  int v30; // edi
  int v31; // ecx
  __int64 v32; // r13
  __m128i v34; // rax
  __int64 v35; // r8
  __int64 v36; // rax
  unsigned __int64 v37; // rdx
  __int64 *v38; // rax
  __int64 v39; // rsi
  unsigned __int64 v40; // r12
  __int64 v41; // r13
  __int64 v42; // rax
  __int64 v43; // rsi
  __int64 v44; // rax
  int v45; // esi
  unsigned __int64 v46; // r8
  __m128i v47; // xmm0
  _DWORD *v48; // rax
  __int64 v49; // r8
  __int64 v50; // r11
  _DWORD *v51; // rcx
  __int64 v52; // rsi
  __int64 v53; // r12
  __int64 v54; // r13
  _QWORD *v55; // r15
  __int64 v56; // rdx
  int v57; // r9d
  __int64 v58; // r15
  __int64 v59; // rdx
  __int64 v60; // r12
  __int64 v61; // rsi
  __int64 v62; // rax
  __int128 v63; // [rsp-10h] [rbp-140h]
  __int64 v64; // [rsp+8h] [rbp-128h]
  __m128i v65; // [rsp+10h] [rbp-120h] BYREF
  __m128i v66; // [rsp+20h] [rbp-110h]
  __int64 v67; // [rsp+30h] [rbp-100h]
  _BYTE *v68; // [rsp+38h] [rbp-F8h]
  __int64 v69; // [rsp+40h] [rbp-F0h]
  int v70; // [rsp+4Ch] [rbp-E4h]
  __int64 v71; // [rsp+50h] [rbp-E0h]
  __int64 v72; // [rsp+58h] [rbp-D8h]
  __int64 v73; // [rsp+60h] [rbp-D0h]
  __int64 v74; // [rsp+68h] [rbp-C8h]
  __int64 v75; // [rsp+70h] [rbp-C0h]
  __int64 v76; // [rsp+78h] [rbp-B8h]
  int v77; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v78; // [rsp+88h] [rbp-A8h]
  __int64 v79; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v80; // [rsp+98h] [rbp-98h]
  __int64 v81; // [rsp+A0h] [rbp-90h] BYREF
  int v82; // [rsp+A8h] [rbp-88h]
  _BYTE *v83; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v84; // [rsp+B8h] [rbp-78h]
  _BYTE v85[112]; // [rsp+C0h] [rbp-70h] BYREF

  v6 = a1;
  v7 = *(__int16 **)(a1 + 48);
  v69 = a2;
  v8 = *v7;
  v9 = *((_QWORD *)v7 + 1);
  LOWORD(v77) = v8;
  v78 = v9;
  if ( v8 )
  {
    if ( (unsigned __int16)(v8 - 176) <= 0x34u )
    {
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
      v7 = *(__int16 **)(a1 + 48);
    }
    v10 = v8;
    v8 = *v7;
    v9 = *((_QWORD *)v7 + 1);
    v11 = word_4456340[v10 - 1];
  }
  else
  {
    if ( sub_3007100((__int64)&v77) )
    {
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
      v12 = *(__int16 **)(a1 + 48);
      v8 = *v12;
      v9 = *((_QWORD *)v12 + 1);
    }
    v11 = sub_3007130((__int64)&v77, a2);
  }
  v13 = *(_QWORD *)(a1 + 40);
  v14 = _mm_loadu_si128((const __m128i *)v13);
  v15 = *(_QWORD *)v13;
  v16 = _mm_loadu_si128((const __m128i *)(v13 + 40));
  v17 = *(_QWORD *)(v13 + 40);
  LOWORD(v83) = v8;
  v71 = v15;
  v67 = v17;
  v84 = v9;
  v66 = v14;
  v65 = v16;
  if ( v8 )
  {
    if ( (unsigned __int16)(v8 - 176) <= 0x34u )
    {
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    }
  }
  else if ( sub_3007100((__int64)&v83) )
  {
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
  }
  v72 = *(_QWORD *)(v6 + 96);
  v83 = v85;
  v68 = v85;
  v84 = 0x400000000LL;
  v18 = *(_QWORD *)(**(_QWORD **)(v71 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(v71 + 40) + 8LL);
  v19 = *(_WORD *)v18;
  v20 = *(_QWORD *)(v18 + 8);
  LOWORD(v79) = v19;
  v80 = v20;
  if ( v19 )
  {
    if ( (unsigned __int16)(v19 - 176) > 0x34u )
    {
LABEL_13:
      v21 = word_4456340[(unsigned __int16)v79 - 1];
      goto LABEL_18;
    }
  }
  else if ( !sub_3007100((__int64)&v79) )
  {
    goto LABEL_17;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( (_WORD)v79 )
  {
    if ( (unsigned __int16)(v79 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    goto LABEL_13;
  }
LABEL_17:
  v21 = sub_3007130((__int64)&v79, a2);
LABEL_18:
  v70 = v11 / v21;
  if ( 2 * v21 == v11 && *(_DWORD *)(v67 + 24) == 51 )
  {
    v48 = sub_325E120((_DWORD *)(v72 + 4LL * v21), v72 + 4LL * v21 + 4LL * v21);
    if ( v51 == v48 )
    {
      v52 = *(_QWORD *)(v6 + 80);
      v53 = v50;
      v54 = v49;
      v55 = *(_QWORD **)(v71 + 40);
      v81 = v52;
      if ( v52 )
        sub_B96E90((__int64)&v81, v52, 1);
      v82 = *(_DWORD *)(v6 + 72);
      v75 = sub_33FCE10(v69, v79, v80, (unsigned int)&v81, *v55, v55[1], v55[5], v55[6], v53, v54);
      v66.m128i_i64[0] = v75;
      v76 = v56;
      v66.m128i_i64[1] = (unsigned int)v56 | v66.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      if ( v81 )
        sub_B91220((__int64)&v81, v81);
      v81 = 0;
      v82 = 0;
      v58 = sub_33F17F0(v69, 51, &v81, v79, v80);
      v60 = v59;
      if ( v81 )
        sub_B91220((__int64)&v81, v81);
      v74 = v60;
      v73 = v58;
      v61 = *(_QWORD *)(v6 + 80);
      v65.m128i_i64[0] = v58;
      v65.m128i_i64[1] = (unsigned int)v60 | v65.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      v81 = v61;
      if ( v61 )
        sub_B96E90((__int64)&v81, v61, 1);
      v82 = *(_DWORD *)(v6 + 72);
      v62 = sub_3406EB0(v69, 159, (unsigned int)&v81, v77, v78, v57, *(_OWORD *)&v66, *(_OWORD *)&v65);
      v43 = v81;
      v32 = v62;
      if ( v81 )
        goto LABEL_44;
      goto LABEL_31;
    }
  }
  v22 = 0;
  v23 = 0;
  v66.m128i_i64[0] = (__int64)&v83;
  if ( v21 > v11 )
    goto LABEL_41;
  v64 = v6;
  v24 = v21;
  v25 = 4LL * v21;
  do
  {
    v26 = sub_325E120((_DWORD *)(v72 + 4LL * v22), v72 + 4LL * v22 + v25);
    if ( v28 != v26 )
    {
      v29 = 0;
      v30 = -1;
      while ( 1 )
      {
        v31 = *(_DWORD *)(v27 + 4 * v29);
        if ( v31 != -1 )
        {
          if ( v31 % (int)v24 != (_DWORD)v29 || v30 >= 0 && v31 / v24 != v30 )
          {
            v32 = 0;
            goto LABEL_31;
          }
          v30 = v31 / v24;
        }
        if ( v29 == v24 - 1 )
          break;
        ++v29;
      }
      LODWORD(a6) = v71;
      v44 = (unsigned int)v84;
      v45 = *(_DWORD *)(v71 + 64);
      v46 = (unsigned int)v84 + 1LL;
      if ( v45 > v30 )
      {
        v47 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v71 + 40) + 40LL * (unsigned int)v30));
        if ( v46 <= HIDWORD(v84) )
          goto LABEL_47;
      }
      else
      {
        v47 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v67 + 40) + 40LL * (unsigned int)(v30 - v45)));
        if ( v46 <= HIDWORD(v84) )
        {
LABEL_47:
          *(__m128i *)&v83[16 * v44] = v47;
          LODWORD(v84) = v84 + 1;
          goto LABEL_39;
        }
      }
      v65 = v47;
      sub_C8D5F0(v66.m128i_i64[0], v68, (unsigned int)v84 + 1LL, 0x10u, v46, v71);
      v44 = (unsigned int)v84;
      v47 = _mm_load_si128(&v65);
      goto LABEL_47;
    }
    v81 = 0;
    v82 = 0;
    v34.m128i_i64[0] = sub_33F17F0(v69, 51, &v81, v79, v80);
    a6 = v34.m128i_i64[1];
    v35 = v34.m128i_i64[0];
    if ( v81 )
    {
      v65 = v34;
      sub_B91220((__int64)&v81, v81);
      a6 = v65.m128i_i64[1];
      v35 = v65.m128i_i64[0];
    }
    v36 = (unsigned int)v84;
    v37 = (unsigned int)v84 + 1LL;
    if ( v37 > HIDWORD(v84) )
    {
      v65.m128i_i64[0] = v35;
      v65.m128i_i64[1] = a6;
      sub_C8D5F0(v66.m128i_i64[0], v68, v37, 0x10u, v35, a6);
      v36 = (unsigned int)v84;
      a6 = v65.m128i_i64[1];
      v35 = v65.m128i_i64[0];
    }
    v38 = (__int64 *)&v83[16 * v36];
    *v38 = v35;
    v38[1] = a6;
    LODWORD(v84) = v84 + 1;
LABEL_39:
    ++v23;
    v22 += v24;
  }
  while ( v70 != v23 );
  v6 = v64;
LABEL_41:
  v39 = *(_QWORD *)(v6 + 80);
  v40 = (unsigned __int64)v83;
  v41 = (unsigned int)v84;
  v81 = v39;
  if ( v39 )
    sub_B96E90((__int64)&v81, v39, 1);
  *((_QWORD *)&v63 + 1) = v41;
  *(_QWORD *)&v63 = v40;
  v82 = *(_DWORD *)(v6 + 72);
  v42 = sub_33FC220(v69, 159, (unsigned int)&v81, v77, v78, a6, v63);
  v43 = v81;
  v32 = v42;
  if ( v81 )
LABEL_44:
    sub_B91220((__int64)&v81, v43);
LABEL_31:
  if ( v83 != v68 )
    _libc_free((unsigned __int64)v83);
  return v32;
}
