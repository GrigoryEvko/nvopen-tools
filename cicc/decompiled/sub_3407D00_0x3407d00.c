// Function: sub_3407D00
// Address: 0x3407d00
//
unsigned __int8 *__fastcall sub_3407D00(_QWORD *a1, __int64 a2, unsigned __int64 a3, char a4, __m128i a5)
{
  _QWORD *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r13
  _QWORD *v10; // r12
  unsigned __int16 *v11; // rax
  int v12; // r9d
  __int64 v13; // r10
  bool v14; // al
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r15
  bool v19; // al
  unsigned __int8 *v20; // r13
  __int64 v22; // r11
  __int64 (__fastcall *v23)(__int64, __int64, unsigned int, __int64); // rax
  unsigned __int16 v24; // cx
  __int64 v25; // r10
  unsigned __int16 v26; // r9
  __int64 v27; // rsi
  unsigned __int8 *v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rsi
  unsigned __int8 *v31; // r10
  __int64 v32; // r11
  __int64 v33; // rdx
  __int64 v34; // rax
  unsigned __int64 v35; // r15
  __int64 v36; // rdx
  char v37; // si
  __int64 v38; // rcx
  __int64 v39; // rdx
  __int64 v40; // rax
  unsigned __int64 v41; // rdx
  __int32 v42; // eax
  __int64 v43; // rdx
  __int128 v44; // [rsp-20h] [rbp-F0h]
  __int128 v45; // [rsp-10h] [rbp-E0h]
  __int64 v46; // [rsp+0h] [rbp-D0h]
  unsigned __int16 v47; // [rsp+0h] [rbp-D0h]
  unsigned __int8 *v48; // [rsp+0h] [rbp-D0h]
  __int64 v49; // [rsp+8h] [rbp-C8h]
  __int64 v50; // [rsp+18h] [rbp-B8h]
  __int64 v51; // [rsp+18h] [rbp-B8h]
  unsigned __int16 v52; // [rsp+18h] [rbp-B8h]
  int v53; // [rsp+2Ch] [rbp-A4h] BYREF
  __m128i v54; // [rsp+30h] [rbp-A0h] BYREF
  __m128i v55; // [rsp+40h] [rbp-90h] BYREF
  __int64 v56; // [rsp+50h] [rbp-80h] BYREF
  int v57; // [rsp+58h] [rbp-78h]
  __int64 v58; // [rsp+60h] [rbp-70h] BYREF
  __int64 v59; // [rsp+68h] [rbp-68h]
  __int64 v60; // [rsp+70h] [rbp-60h]
  __int64 v61; // [rsp+78h] [rbp-58h]
  __int64 v62; // [rsp+80h] [rbp-50h] BYREF
  __int64 v63; // [rsp+88h] [rbp-48h]
  __int64 v64; // [rsp+90h] [rbp-40h]

  v7 = sub_33F2320(a1, a2, a3, &v53);
  if ( !v7 )
    return 0;
  v9 = v8;
  v10 = v7;
  v11 = (unsigned __int16 *)(v7[6] + 16LL * (unsigned int)v8);
  v12 = *v11;
  v13 = *((_QWORD *)v11 + 1);
  LOWORD(v62) = v12;
  v63 = v13;
  if ( (_WORD)v12 )
  {
    if ( (unsigned __int16)(v12 - 17) > 0xD3u )
    {
      v54.m128i_i16[0] = v12;
      v54.m128i_i64[1] = v13;
      v55 = _mm_loadu_si128(&v54);
      if ( !a4 )
        goto LABEL_20;
      v18 = a1[2];
      goto LABEL_12;
    }
    v13 = 0;
    LOWORD(v12) = word_4456580[v12 - 1];
  }
  else
  {
    v46 = v13;
    v14 = sub_30070B0((__int64)&v62);
    v13 = v46;
    if ( !v14 )
    {
      v54.m128i_i64[1] = v46;
      v54.m128i_i16[0] = 0;
      a5 = _mm_loadu_si128(&v54);
      v55 = a5;
      if ( !a4 )
        goto LABEL_20;
      v18 = a1[2];
      goto LABEL_6;
    }
    LOWORD(v12) = sub_3009970((__int64)&v62, a2, v15, v16, v17);
    v13 = v33;
  }
  v54.m128i_i16[0] = v12;
  v54.m128i_i64[1] = v13;
  v55 = _mm_loadu_si128(&v54);
  if ( !a4 )
    goto LABEL_20;
  v18 = a1[2];
  if ( !(_WORD)v12 )
  {
LABEL_6:
    v50 = v13;
    v19 = sub_3007070((__int64)&v54);
    v13 = v50;
    if ( !v19 )
      return 0;
    LOWORD(v12) = 0;
    goto LABEL_16;
  }
LABEL_12:
  if ( *(_QWORD *)(v18 + 8LL * (unsigned __int16)v12 + 112) )
    goto LABEL_20;
  if ( (unsigned __int16)(v12 - 2) > 7u && (unsigned __int16)(v12 - 17) > 0x6Cu && (unsigned __int16)(v12 - 176) > 0x1Fu )
    return 0;
LABEL_16:
  v22 = a1[8];
  v47 = v12;
  v51 = v13;
  v23 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v18 + 592LL);
  if ( v23 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v62, v18, v22, v55.m128i_i64[0], v55.m128i_i64[1]);
    v24 = v63;
    v25 = v51;
    v26 = v47;
    v55.m128i_i16[0] = v63;
    v55.m128i_i64[1] = v64;
  }
  else
  {
    v42 = v23(v18, v22, v55.m128i_u32[0], v55.m128i_i64[1]);
    v26 = v47;
    v25 = v51;
    v55.m128i_i32[0] = v42;
    v24 = v42;
    v55.m128i_i64[1] = v43;
  }
  if ( v26 == v24 )
  {
    if ( v24 || v25 == v55.m128i_i64[1] )
      goto LABEL_20;
    v59 = v25;
    LOWORD(v58) = 0;
    goto LABEL_34;
  }
  LOWORD(v58) = v26;
  v59 = v25;
  if ( !v26 )
  {
LABEL_34:
    v52 = v24;
    v34 = sub_3007260((__int64)&v58);
    v24 = v52;
    v62 = v34;
    v35 = v34;
    v63 = v36;
    v37 = v36;
    goto LABEL_35;
  }
  if ( v26 == 1 || (unsigned __int16)(v26 - 504) <= 7u )
    goto LABEL_51;
  v35 = *(_QWORD *)&byte_444C4A0[16 * v26 - 16];
  v37 = byte_444C4A0[16 * v26 - 8];
LABEL_35:
  if ( !v24 )
  {
    v38 = sub_3007260((__int64)&v55);
    v40 = v39;
    v60 = v38;
    v41 = v38;
    v61 = v40;
    goto LABEL_37;
  }
  if ( v24 == 1 || (unsigned __int16)(v24 - 504) <= 7u )
LABEL_51:
    BUG();
  v41 = *(_QWORD *)&byte_444C4A0[16 * v24 - 16];
  LOBYTE(v40) = byte_444C4A0[16 * v24 - 8];
LABEL_37:
  if ( (!(_BYTE)v40 || v37) && v41 < v35 )
    return 0;
LABEL_20:
  v27 = *(_QWORD *)(a2 + 80);
  v58 = v27;
  if ( v27 )
    sub_B96E90((__int64)&v58, v27, 1);
  LODWORD(v59) = *(_DWORD *)(a2 + 72);
  v28 = sub_3400EE0((__int64)a1, v53, (__int64)&v58, 0, a5);
  v30 = *(_QWORD *)(a2 + 80);
  v31 = v28;
  v32 = v29;
  v56 = v30;
  if ( v30 )
  {
    v49 = v29;
    v48 = v28;
    sub_B96E90((__int64)&v56, v30, 1);
    v31 = v48;
    v32 = v49;
  }
  *((_QWORD *)&v45 + 1) = v32;
  *(_QWORD *)&v45 = v31;
  *((_QWORD *)&v44 + 1) = v9;
  *(_QWORD *)&v44 = v10;
  v57 = *(_DWORD *)(a2 + 72);
  v20 = sub_3406EB0(a1, 0x9Eu, (__int64)&v56, v55.m128i_u32[0], v55.m128i_i64[1], (__int64)&v56, v44, v45);
  if ( v56 )
    sub_B91220((__int64)&v56, v56);
  if ( v58 )
    sub_B91220((__int64)&v58, v58);
  return v20;
}
