// Function: sub_3847980
// Address: 0x3847980
//
unsigned __int8 *__fastcall sub_3847980(__int64 *a1, __int64 a2, __m128i a3)
{
  __int16 *v4; // rdx
  __int16 v5; // ax
  __int64 v6; // rdx
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdx
  unsigned __int16 v10; // si
  __int64 v11; // r8
  __int64 (__fastcall *v12)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rsi
  __int64 v16; // rdx
  unsigned __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rbx
  __int64 v20; // r8
  __int64 v21; // r9
  __m128i v22; // xmm0
  __int64 v23; // rax
  __m128i v24; // xmm0
  unsigned __int64 v25; // rdx
  __int64 v26; // rax
  int v27; // ecx
  __int64 v28; // rax
  unsigned __int64 *v29; // rax
  __int64 v30; // rcx
  unsigned __int64 v31; // rsi
  __int64 v32; // rax
  __int16 v33; // dx
  __int64 v34; // rax
  bool v35; // al
  __int64 v36; // rax
  __int64 *v37; // r13
  unsigned __int16 v38; // ax
  __int64 v39; // r9
  __int64 v40; // r8
  int v41; // r9d
  unsigned __int8 *v42; // r12
  __int64 v44; // rcx
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rdx
  unsigned __int64 v48; // r14
  __int64 v49; // r15
  __int64 v50; // rax
  __int16 v51; // dx
  __int64 v52; // rax
  __int64 v53; // r9
  __int64 v54; // rdx
  __int64 v55; // rdx
  __int128 v56; // [rsp-10h] [rbp-200h]
  __int64 v57; // [rsp+0h] [rbp-1F0h]
  unsigned int v58; // [rsp+8h] [rbp-1E8h]
  __m128i v59; // [rsp+30h] [rbp-1C0h] BYREF
  __int64 v60; // [rsp+40h] [rbp-1B0h]
  __int64 v61; // [rsp+48h] [rbp-1A8h]
  __m128i v62; // [rsp+50h] [rbp-1A0h]
  unsigned int v63; // [rsp+60h] [rbp-190h] BYREF
  __int64 v64; // [rsp+68h] [rbp-188h]
  __int64 v65; // [rsp+70h] [rbp-180h] BYREF
  int v66; // [rsp+78h] [rbp-178h]
  __m128i v67; // [rsp+80h] [rbp-170h] BYREF
  __m128i v68; // [rsp+90h] [rbp-160h] BYREF
  __int128 v69; // [rsp+A0h] [rbp-150h] BYREF
  _QWORD *v70; // [rsp+B0h] [rbp-140h] BYREF
  __int64 v71; // [rsp+B8h] [rbp-138h]
  _QWORD v72[38]; // [rsp+C0h] [rbp-130h] BYREF

  v4 = *(__int16 **)(a2 + 48);
  v5 = *v4;
  v6 = *((_QWORD *)v4 + 1);
  LOWORD(v63) = v5;
  v64 = v6;
  if ( v5 )
  {
    if ( (unsigned __int16)(v5 - 176) > 0x34u )
    {
LABEL_3:
      v7 = word_4456340[(unsigned __int16)v63 - 1];
      goto LABEL_6;
    }
  }
  else if ( !sub_3007100((__int64)&v63) )
  {
    goto LABEL_5;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( (_WORD)v63 )
  {
    if ( (unsigned __int16)(v63 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    goto LABEL_3;
  }
LABEL_5:
  v7 = (unsigned int)sub_3007130((__int64)&v63, a2);
LABEL_6:
  v8 = *(_QWORD *)(**(_QWORD **)(a2 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 8LL);
  v9 = a1[1];
  v10 = *(_WORD *)v8;
  v11 = *(_QWORD *)(v8 + 8);
  v12 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  if ( v12 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v70, *a1, *(_QWORD *)(v9 + 64), v10, v11);
    v58 = (unsigned __int16)v71;
    v57 = v72[0];
  }
  else
  {
    v58 = v12(*a1, *(_QWORD *)(v9 + 64), v10, v11);
    v57 = v55;
  }
  v15 = *(_QWORD *)(a2 + 80);
  v65 = v15;
  if ( v15 )
    sub_B96E90((__int64)&v65, v15, 1);
  v66 = *(_DWORD *)(a2 + 72);
  if ( !(_WORD)v63 )
    goto LABEL_14;
  v16 = (unsigned int)(unsigned __int16)v63 - 2;
  if ( (unsigned __int16)(v63 - 2) > 7u )
  {
    v16 = (unsigned int)(unsigned __int16)v63 - 17;
    if ( (unsigned __int16)(v63 - 17) > 0x6Cu )
    {
      v16 = (unsigned int)(unsigned __int16)v63 - 176;
      if ( (unsigned __int16)(v63 - 176) > 0x1Fu )
        goto LABEL_14;
    }
  }
  v44 = *a1;
  if ( (_WORD)v63 == 1 )
  {
    if ( *(_BYTE *)(v44 + 7082) )
      goto LABEL_14;
    v45 = 1;
  }
  else
  {
    if ( !*(_QWORD *)(v44 + 8LL * (unsigned __int16)v63 + 112) )
      goto LABEL_14;
    v45 = (unsigned __int16)v63;
    v16 = 500LL * (unsigned __int16)v63;
    if ( *(_BYTE *)(v44 + v16 + 6582) )
      goto LABEL_14;
  }
  if ( (*(_BYTE *)(v44 + 500 * v45 + 6583) & 0xFB) == 0 )
  {
    v46 = sub_33D2250(a2, 0, v16, v44, v13, v14);
    v48 = v46;
    v49 = v47;
    if ( v46 )
    {
      v68.m128i_i64[0] = 0;
      v68.m128i_i32[2] = 0;
      *(_QWORD *)&v69 = 0;
      DWORD2(v69) = 0;
      v50 = *(_QWORD *)(v46 + 48) + 16LL * (unsigned int)v47;
      v51 = *(_WORD *)v50;
      v52 = *(_QWORD *)(v50 + 8);
      LOWORD(v70) = v51;
      v71 = v52;
      if ( v51 )
      {
        if ( (unsigned __int16)(v51 - 2) <= 7u
          || (unsigned __int16)(v51 - 17) <= 0x6Cu
          || (unsigned __int16)(v51 - 176) <= 0x1Fu )
        {
          goto LABEL_51;
        }
      }
      else if ( sub_3007070((__int64)&v70) )
      {
LABEL_51:
        sub_375E510((__int64)a1, v48, v49, (__int64)&v68, (__int64)&v69);
LABEL_52:
        v42 = sub_3406EB0((_QWORD *)a1[1], 0xA9u, (__int64)&v65, v63, v64, v53, *(_OWORD *)&v68, v69);
        goto LABEL_37;
      }
      sub_375E6F0((__int64)a1, v48, v49, (__int64)&v68, (__int64)&v69);
      goto LABEL_52;
    }
  }
LABEL_14:
  v17 = (unsigned int)(2 * v7);
  v70 = v72;
  v71 = 0x1000000000LL;
  if ( v17 > 0x10 )
    sub_C8D5F0((__int64)&v70, v72, v17, 0x10u, v13, v14);
  if ( (_DWORD)v7 )
  {
    v18 = 5 * v7;
    v19 = 0;
    v61 = 8 * v18;
    while ( 1 )
    {
      v68.m128i_i64[0] = 0;
      v28 = *(_QWORD *)(a2 + 40);
      v68.m128i_i32[2] = 0;
      v67.m128i_i32[2] = 0;
      v29 = (unsigned __int64 *)(v19 + v28);
      v67.m128i_i64[0] = 0;
      v30 = v29[1];
      v31 = *v29;
      v32 = *(_QWORD *)(*v29 + 48) + 16LL * *((unsigned int *)v29 + 2);
      v33 = *(_WORD *)v32;
      v34 = *(_QWORD *)(v32 + 8);
      LOWORD(v69) = v33;
      *((_QWORD *)&v69 + 1) = v34;
      if ( v33 )
      {
        if ( (unsigned __int16)(v33 - 2) > 7u
          && (unsigned __int16)(v33 - 17) > 0x6Cu
          && (unsigned __int16)(v33 - 176) > 0x1Fu )
        {
LABEL_31:
          sub_375E6F0((__int64)a1, v31, v30, (__int64)&v67, (__int64)&v68);
          goto LABEL_22;
        }
      }
      else
      {
        v60 = v30;
        v59.m128i_i64[0] = v31;
        v35 = sub_3007070((__int64)&v69);
        v31 = v59.m128i_i64[0];
        v30 = v60;
        if ( !v35 )
          goto LABEL_31;
      }
      sub_375E510((__int64)a1, v31, v30, (__int64)&v67, (__int64)&v68);
LABEL_22:
      if ( *(_BYTE *)sub_2E79000(*(__int64 **)(a1[1] + 40)) )
      {
        v22 = _mm_load_si128(&v67);
        v67.m128i_i64[0] = v68.m128i_i64[0];
        v62 = v22;
        v67.m128i_i32[2] = v68.m128i_i32[2];
        v68.m128i_i64[0] = v22.m128i_i64[0];
        v68.m128i_i32[2] = v22.m128i_i32[2];
      }
      v23 = (unsigned int)v71;
      v24 = _mm_load_si128(&v67);
      v25 = (unsigned int)v71 + 1LL;
      if ( v25 > HIDWORD(v71) )
      {
        v59 = v24;
        sub_C8D5F0((__int64)&v70, v72, v25, 0x10u, v20, v21);
        v23 = (unsigned int)v71;
        v24 = _mm_load_si128(&v59);
      }
      *(__m128i *)&v70[2 * v23] = v24;
      a3 = _mm_load_si128(&v68);
      LODWORD(v71) = v71 + 1;
      v26 = (unsigned int)v71;
      if ( (unsigned __int64)(unsigned int)v71 + 1 > HIDWORD(v71) )
      {
        v59 = a3;
        sub_C8D5F0((__int64)&v70, v72, (unsigned int)v71 + 1LL, 0x10u, v20, v21);
        v26 = (unsigned int)v71;
        a3 = _mm_load_si128(&v59);
      }
      v19 += 40;
      *(__m128i *)&v70[2 * v26] = a3;
      v27 = v71 + 1;
      LODWORD(v71) = v71 + 1;
      if ( v19 == v61 )
        goto LABEL_33;
    }
  }
  v27 = v71;
LABEL_33:
  v36 = a1[1];
  LODWORD(v61) = v27;
  v37 = *(__int64 **)(v36 + 64);
  v38 = sub_2D43050(v58, v27);
  v40 = 0;
  if ( !v38 )
  {
    v38 = sub_3009400(v37, v58, v57, (unsigned int)v61, 0);
    v40 = v54;
  }
  *((_QWORD *)&v56 + 1) = (unsigned int)v71;
  *(_QWORD *)&v56 = v70;
  sub_33FC220((_QWORD *)a1[1], 156, (__int64)&v65, v38, v40, v39, v56);
  v42 = sub_33FAF80(a1[1], 234, (__int64)&v65, v63, v64, v41, a3);
  if ( v70 != v72 )
    _libc_free((unsigned __int64)v70);
LABEL_37:
  if ( v65 )
    sub_B91220((__int64)&v65, v65);
  return v42;
}
