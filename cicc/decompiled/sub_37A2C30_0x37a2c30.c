// Function: sub_37A2C30
// Address: 0x37a2c30
//
__m128i *__fastcall sub_37A2C30(__int64 *a1, unsigned __int64 a2, __m128i a3)
{
  unsigned int v3; // ebx
  __int64 v6; // r9
  __int64 (__fastcall *v7)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v8; // rax
  unsigned __int16 v9; // si
  __int64 v10; // r8
  __int64 v11; // rax
  __int64 v12; // r11
  _QWORD *v13; // rdx
  __int64 v14; // r15
  __int64 v15; // rax
  unsigned __int16 v16; // cx
  __int64 v17; // rax
  unsigned __int64 v18; // rsi
  __m128i v19; // rax
  __int64 v20; // r8
  __int64 *v21; // rdi
  __int64 v22; // rsi
  unsigned __int16 *v23; // rdx
  __int64 v24; // rcx
  int v25; // eax
  unsigned __int16 v26; // ax
  __int64 v27; // rdx
  unsigned int v28; // r14d
  int v29; // eax
  __int64 v30; // r8
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // r14
  __int64 v34; // rbx
  unsigned __int16 *v35; // rax
  unsigned int v36; // ecx
  __int64 v37; // rdx
  __int64 v38; // rax
  __int64 v39; // r8
  __int64 v40; // rcx
  unsigned __int8 *v41; // rax
  __m128i v42; // xmm1
  __int64 v43; // rdx
  __int64 v44; // roff
  __m128i v45; // xmm0
  int v46; // r15d
  __m128i v47; // xmm2
  __int64 v48; // rbx
  unsigned int v49; // r14d
  unsigned __int64 v50; // r15
  __int64 v51; // rax
  __int64 v52; // rbx
  _QWORD *v53; // rdi
  __int64 *v54; // r9
  unsigned __int16 v55; // ax
  unsigned __int8 v56; // r14
  unsigned __int64 v57; // rax
  __int32 v58; // edx
  __m128i *v59; // r14
  __int64 v61; // rdx
  __int64 v62; // rcx
  __int64 v63; // r8
  __int64 v64; // rdx
  bool v65; // al
  __int64 v66; // r8
  unsigned __int64 v67; // rdx
  __int64 v68; // rdx
  __int64 v69; // rdx
  __int64 v70; // rdx
  __int64 v71; // [rsp+0h] [rbp-160h]
  __int64 v72; // [rsp+8h] [rbp-158h]
  __int64 *v73; // [rsp+8h] [rbp-158h]
  __int64 *v74; // [rsp+10h] [rbp-150h]
  __int64 v75; // [rsp+10h] [rbp-150h]
  __int64 v76; // [rsp+10h] [rbp-150h]
  int v77; // [rsp+18h] [rbp-148h]
  unsigned int v78; // [rsp+18h] [rbp-148h]
  unsigned int v79; // [rsp+18h] [rbp-148h]
  unsigned int v80; // [rsp+18h] [rbp-148h]
  __int64 v81; // [rsp+28h] [rbp-138h]
  unsigned __int64 v82; // [rsp+28h] [rbp-138h]
  __m128i v83; // [rsp+30h] [rbp-130h] BYREF
  int v84; // [rsp+44h] [rbp-11Ch]
  __m128i *v85; // [rsp+48h] [rbp-118h]
  __m128i *v86; // [rsp+50h] [rbp-110h]
  __int64 v87; // [rsp+58h] [rbp-108h]
  __int64 v88; // [rsp+60h] [rbp-100h]
  _QWORD *v89; // [rsp+68h] [rbp-F8h]
  unsigned __int8 *v90; // [rsp+70h] [rbp-F0h]
  __int64 v91; // [rsp+78h] [rbp-E8h]
  unsigned __int8 *v92; // [rsp+80h] [rbp-E0h]
  __int64 v93; // [rsp+88h] [rbp-D8h]
  unsigned int v94; // [rsp+90h] [rbp-D0h] BYREF
  __int64 v95; // [rsp+98h] [rbp-C8h]
  unsigned __int16 v96; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v97; // [rsp+A8h] [rbp-B8h]
  __int64 v98; // [rsp+B0h] [rbp-B0h] BYREF
  int v99; // [rsp+B8h] [rbp-A8h]
  __int16 v100; // [rsp+C0h] [rbp-A0h] BYREF
  __int64 v101; // [rsp+C8h] [rbp-98h]
  __m128i v102; // [rsp+D0h] [rbp-90h] BYREF
  __m128i v103; // [rsp+E0h] [rbp-80h]
  unsigned __int8 *v104; // [rsp+F0h] [rbp-70h]
  unsigned __int64 v105; // [rsp+F8h] [rbp-68h]
  __m128i v106; // [rsp+100h] [rbp-60h]
  unsigned __int8 *v107; // [rsp+110h] [rbp-50h]
  unsigned __int64 v108; // [rsp+118h] [rbp-48h]
  __m128i *v109; // [rsp+120h] [rbp-40h]
  int v110; // [rsp+128h] [rbp-38h]

  v6 = *a1;
  v7 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v8 = *(__int16 **)(a2 + 48);
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  v11 = a1[1];
  v12 = *(_QWORD *)(v11 + 64);
  if ( v7 == sub_2D56A50 )
  {
    v86 = &v102;
    sub_2FE6CC0((__int64)&v102, v6, v12, v9, v10);
    LOWORD(v94) = v102.m128i_i16[4];
    v95 = v103.m128i_i64[0];
  }
  else
  {
    v94 = v7(*a1, *(_QWORD *)(v11 + 64), v9, v10);
    v95 = v70;
    v86 = &v102;
  }
  v13 = *(_QWORD **)(a2 + 40);
  v14 = v13[10];
  v81 = v13[11];
  v15 = *(_QWORD *)(v14 + 48) + 16LL * *((unsigned int *)v13 + 22);
  v16 = *(_WORD *)v15;
  v17 = *(_QWORD *)(v15 + 8);
  v96 = v16;
  v97 = v17;
  v18 = v13[5];
  v19.m128i_i64[0] = sub_379AB60((__int64)a1, v18, v13[6]);
  v83 = v19;
  v19.m128i_i64[0] = *(_QWORD *)(a2 + 40);
  v21 = *(__int64 **)(v19.m128i_i64[0] + 200);
  v84 = *(_DWORD *)(v19.m128i_i64[0] + 208);
  v85 = (__m128i *)v21;
  if ( (_WORD)v94 )
  {
    if ( (unsigned __int16)(v94 - 176) > 0x34u )
    {
LABEL_5:
      LODWORD(v89) = word_4456340[(unsigned __int16)v94 - 1];
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
  LODWORD(v89) = sub_3007130((__int64)&v94, v18);
LABEL_8:
  v22 = *(_QWORD *)(a2 + 80);
  v98 = v22;
  if ( v22 )
    sub_B96E90((__int64)&v98, v22, 1);
  v99 = *(_DWORD *)(a2 + 72);
  if ( (_WORD)v94 )
  {
    if ( (unsigned __int16)(v94 - 176) > 0x34u )
      goto LABEL_12;
  }
  else if ( !sub_3007100((__int64)&v94) )
  {
    goto LABEL_15;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( !(_WORD)v94 )
  {
LABEL_15:
    v24 = (unsigned int)sub_3007130((__int64)&v94, v22);
    v25 = v96;
    if ( !v96 )
      goto LABEL_13;
    goto LABEL_16;
  }
  if ( (unsigned __int16)(v94 - 176) <= 0x34u )
    sub_CA17B0(
      "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT::"
      "getVectorElementCount() instead");
LABEL_12:
  v23 = word_4456340;
  v24 = word_4456340[(unsigned __int16)v94 - 1];
  v25 = v96;
  if ( !v96 )
  {
LABEL_13:
    v77 = v24;
    v26 = sub_3009970((__int64)&v96, v22, (__int64)v23, v24, v20);
    LODWORD(v24) = v77;
    v72 = v27;
    goto LABEL_17;
  }
LABEL_16:
  v72 = 0;
  v26 = word_4456580[v25 - 1];
LABEL_17:
  v78 = v24;
  v28 = v26;
  v74 = *(__int64 **)(a1[1] + 64);
  LOWORD(v29) = sub_2D43050(v26, v24);
  v30 = 0;
  if ( !(_WORD)v29 )
  {
    v29 = sub_3009400(v74, v28, v72, v78, 0);
    HIWORD(v3) = HIWORD(v29);
    v30 = v69;
  }
  LOWORD(v3) = v29;
  v92 = sub_3790540((__int64)a1, v14, v81, v3, v30, 1, a3);
  v93 = v31;
  v82 = (unsigned int)v31 | v81 & 0xFFFFFFFF00000000LL;
  v32 = *(_QWORD *)(a2 + 40);
  v33 = *(_QWORD *)(v32 + 160);
  v34 = *(_QWORD *)(v32 + 168);
  v35 = (unsigned __int16 *)(*(_QWORD *)(v33 + 48) + 16LL * *(unsigned int *)(v32 + 168));
  v36 = *v35;
  v37 = *((_QWORD *)v35 + 1);
  v102.m128i_i16[0] = v36;
  v102.m128i_i64[1] = v37;
  if ( (_WORD)v36 )
  {
    if ( (unsigned __int16)(v36 - 17) <= 0xD3u )
    {
      v37 = 0;
      LOWORD(v36) = word_4456580[v36 - 1];
    }
  }
  else
  {
    v76 = v37;
    v80 = v36;
    v65 = sub_30070B0((__int64)v86);
    LOWORD(v36) = v80;
    v37 = v76;
    if ( v65 )
      LOWORD(v36) = sub_3009970((__int64)v86, v14, v76, v80, v66);
  }
  v75 = v37;
  v79 = (unsigned __int16)v36;
  v73 = *(__int64 **)(a1[1] + 64);
  LOWORD(v38) = sub_2D43050(v36, (int)v89);
  v39 = 0;
  if ( !(_WORD)v38 )
  {
    v38 = sub_3009400(v73, v79, v75, (unsigned int)v89, 0);
    v71 = v38;
    v39 = v68;
  }
  v40 = v71;
  LOWORD(v40) = v38;
  v41 = sub_3790540((__int64)a1, v33, v34, v40, v39, 0, a3);
  v42 = _mm_load_si128(&v83);
  v90 = v41;
  v91 = v43;
  v44 = *(_QWORD *)(a2 + 40);
  v45 = _mm_loadu_si128((const __m128i *)v44);
  v104 = v92;
  v105 = v82;
  v46 = *(unsigned __int16 *)(a2 + 96);
  v102 = v45;
  v103 = v42;
  v47 = _mm_loadu_si128((const __m128i *)(v44 + 120));
  v108 = (unsigned int)v43 | v34 & 0xFFFFFFFF00000000LL;
  v48 = *(_QWORD *)(a2 + 104);
  v109 = v85;
  v107 = v41;
  v110 = v84;
  v100 = v46;
  v101 = v48;
  v106 = v47;
  if ( (_WORD)v46 )
  {
    if ( (unsigned __int16)(v46 - 17) <= 0xD3u )
    {
      v48 = 0;
      LOWORD(v46) = word_4456580[v46 - 1];
    }
  }
  else if ( sub_30070B0((__int64)&v100) )
  {
    LOWORD(v46) = sub_3009970((__int64)&v100, v33, v61, v62, v63);
    v48 = v64;
  }
  v49 = (unsigned __int16)v46;
  v50 = 0;
  v85 = *(__m128i **)(a1[1] + 64);
  LOWORD(v51) = sub_2D43050(v49, (int)v89);
  if ( !(_WORD)v51 )
  {
    v51 = sub_3009400(v85->m128i_i64, v49, v48, (unsigned int)v89, 0);
    v88 = v51;
    v50 = v67;
  }
  v52 = v88;
  v53 = (_QWORD *)a1[1];
  v54 = *(__int64 **)(a2 + 112);
  v87 = 6;
  LOWORD(v52) = v51;
  v55 = *(_WORD *)(a2 + 32);
  v56 = *(_BYTE *)(a2 + 33);
  v89 = v53;
  v88 = v52;
  v85 = (__m128i *)v54;
  LOWORD(v52) = v55 >> 7;
  v57 = sub_33E5110(v53, v94, v95, 1, 0);
  v59 = sub_33E8420(v89, v57, v58, v88, v50, (__int64)&v98, (unsigned __int64 *)v86, v87, v85, v52 & 7, (v56 >> 2) & 3);
  sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v59, 1);
  if ( v98 )
    sub_B91220((__int64)&v98, v98);
  return v59;
}
