// Function: sub_3847520
// Address: 0x3847520
//
__m128i *__fastcall sub_3847520(__int64 *a1, __int64 a2, __m128i a3)
{
  __int64 v4; // rsi
  unsigned __int64 *v5; // r15
  __int16 *v6; // rax
  unsigned __int16 v7; // r14
  unsigned __int64 v8; // r8
  __int64 v9; // rax
  unsigned __int16 v10; // si
  __int64 v11; // r10
  __int64 v12; // rdx
  __int64 (__fastcall *v13)(__int64, __int64, unsigned int, __int64); // rax
  unsigned int v14; // r14d
  __int64 *v15; // r15
  unsigned __int16 v16; // ax
  __int64 v17; // r8
  __int64 v18; // rdx
  __int64 v19; // rsi
  __int64 v20; // r8
  int v21; // r9d
  __int64 v22; // rax
  __int64 v23; // r14
  __int64 v24; // rcx
  __int64 v25; // rsi
  __int64 v26; // rdx
  __int64 v27; // r9
  int v28; // r9d
  __m128i *v29; // r12
  bool v30; // al
  bool v32; // al
  __int64 v33; // rdx
  unsigned __int16 *v34; // rdx
  __int16 v35; // ax
  __int64 v36; // rcx
  unsigned __int16 *v37; // r14
  __int64 v38; // rdx
  unsigned int v39; // eax
  __int64 v40; // rdx
  __int64 v41; // rdx
  __int128 v42; // [rsp-20h] [rbp-120h]
  __int64 v43; // [rsp+0h] [rbp-100h]
  __int64 v44; // [rsp+8h] [rbp-F8h]
  __int64 v45; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v46; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v47; // [rsp+18h] [rbp-E8h]
  int v48; // [rsp+18h] [rbp-E8h]
  __int64 v49; // [rsp+20h] [rbp-E0h] BYREF
  int v50; // [rsp+28h] [rbp-D8h]
  __int64 v51; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v52; // [rsp+38h] [rbp-C8h]
  _QWORD *v53; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v54; // [rsp+48h] [rbp-B8h]
  _QWORD v55[22]; // [rsp+50h] [rbp-B0h] BYREF

  v4 = *(_QWORD *)(a2 + 80);
  v49 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v49, v4, 1);
  v5 = *(unsigned __int64 **)(a2 + 40);
  v50 = *(_DWORD *)(a2 + 72);
  v6 = *(__int16 **)(a2 + 48);
  v7 = *v6;
  v8 = *((_QWORD *)v6 + 1);
  LOWORD(v51) = v7;
  v52 = v8;
  if ( v7 )
  {
    if ( (unsigned __int16)(v7 - 17) > 0xD3u )
      goto LABEL_20;
  }
  else
  {
    v47 = v8;
    v32 = sub_30070B0((__int64)&v51);
    v8 = v47;
    if ( !v32 )
      goto LABEL_20;
  }
  v9 = *(_QWORD *)(*v5 + 48) + 16LL * *((unsigned int *)v5 + 2);
  v10 = *(_WORD *)v9;
  v11 = *(_QWORD *)(v9 + 8);
  LOWORD(v53) = v10;
  v54 = v11;
  if ( v10 )
  {
    if ( (unsigned __int16)(v10 - 2) <= 7u
      || (unsigned __int16)(v10 - 17) <= 0x6Cu
      || (unsigned __int16)(v10 - 176) <= 0x1Fu )
    {
      goto LABEL_9;
    }
LABEL_20:
    v29 = sub_375AC00((__int64)a1, *v5, v5[1], v7, v8);
    goto LABEL_21;
  }
  v44 = v11;
  v46 = v8;
  v30 = sub_3007070((__int64)&v53);
  v8 = v46;
  v11 = v44;
  if ( !v30 )
    goto LABEL_20;
LABEL_9:
  v12 = a1[1];
  v13 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  if ( v13 == sub_2D56A50 )
  {
    HIWORD(v14) = 0;
    sub_2FE6CC0((__int64)&v53, *a1, *(_QWORD *)(v12 + 64), v10, v11);
    LOWORD(v14) = v54;
    v45 = v55[0];
  }
  else
  {
    v39 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD, __int64, unsigned __int64))v13)(
            *a1,
            *(_QWORD *)(v12 + 64),
            v10,
            v11,
            v8);
    v45 = v40;
    v14 = v39;
  }
  v15 = *(__int64 **)(a1[1] + 64);
  v16 = sub_2D43050(v14, 2);
  v17 = 0;
  if ( !v16 )
  {
    v16 = sub_3009400(v15, v14, v45, 2, 0);
    v17 = v38;
  }
  v18 = a1[1];
  v19 = *a1;
  LOWORD(v51) = v16;
  v52 = v17;
  sub_2FE6CC0((__int64)&v53, v19, *(_QWORD *)(v18 + 64), v16, v17);
  if ( !(_BYTE)v53 )
  {
    LODWORD(v22) = (unsigned __int16)v51;
    v23 = 2;
    v24 = 2;
    goto LABEL_15;
  }
  v34 = *(unsigned __int16 **)(a2 + 48);
  v35 = *v34;
  v36 = *((_QWORD *)v34 + 1);
  v37 = v34;
  LOWORD(v53) = v35;
  v54 = v36;
  if ( !v35 )
  {
    if ( !sub_3007100((__int64)&v53) )
      goto LABEL_32;
    goto LABEL_34;
  }
  if ( (unsigned __int16)(v35 - 176) <= 0x34u )
  {
LABEL_34:
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( (_WORD)v53 )
    {
      if ( (unsigned __int16)((_WORD)v53 - 176) <= 0x34u )
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
      v34 = *(unsigned __int16 **)(a2 + 48);
      goto LABEL_29;
    }
    v37 = *(unsigned __int16 **)(a2 + 48);
LABEL_32:
    v24 = (unsigned int)sub_3007130((__int64)&v53, v19);
    goto LABEL_33;
  }
LABEL_29:
  v37 = v34;
  v24 = word_4456340[(unsigned __int16)v53 - 1];
LABEL_33:
  LODWORD(v22) = *v37;
  v41 = *((_QWORD *)v37 + 1);
  v23 = (unsigned int)v24;
  LOWORD(v51) = v22;
  v52 = v41;
LABEL_15:
  v53 = v55;
  v54 = 0x800000000LL;
  if ( (_WORD)v22 )
  {
    v25 = 0;
    LOWORD(v22) = word_4456580[(int)v22 - 1];
  }
  else
  {
    v48 = v24;
    v22 = sub_3009970((__int64)&v51, v19, 0x800000000LL, v24, v20);
    LODWORD(v24) = v48;
    v43 = v22;
    v25 = v33;
  }
  v26 = v43;
  LOWORD(v26) = v22;
  sub_3847390(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), v24, (__int64)&v53, v21, v26, v25);
  *((_QWORD *)&v42 + 1) = v23;
  *(_QWORD *)&v42 = v53;
  sub_33FC220((_QWORD *)a1[1], 156, (__int64)&v49, v51, v52, v27, v42);
  v29 = (__m128i *)sub_33FAF80(
                     a1[1],
                     234,
                     (__int64)&v49,
                     **(unsigned __int16 **)(a2 + 48),
                     *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
                     v28,
                     a3);
  if ( v53 != v55 )
    _libc_free((unsigned __int64)v53);
LABEL_21:
  if ( v49 )
    sub_B91220((__int64)&v49, v49);
  return v29;
}
