// Function: sub_303B360
// Address: 0x303b360
//
__int64 __fastcall sub_303B360(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v7; // rsi
  unsigned __int16 *v8; // rdx
  int v9; // r9d
  __int64 v10; // rbx
  __int128 v11; // xmm1
  unsigned int v12; // r15d
  __int128 v13; // rax
  __int128 v14; // rax
  int v15; // r9d
  __int128 v16; // rax
  int v17; // r9d
  __int64 v18; // rdx
  __int64 v19; // rax
  int v20; // r9d
  __int16 v21; // r14
  int v22; // eax
  __int64 v23; // rdx
  int v24; // r14d
  __int128 v25; // rax
  __int128 v26; // rax
  __int128 v27; // rax
  int v28; // r9d
  unsigned int v29; // edx
  int v30; // r9d
  unsigned int v31; // edx
  int v32; // r9d
  __int64 v33; // rdx
  __int128 v34; // rax
  __int128 v35; // rax
  int v36; // r9d
  __int64 v37; // r14
  int v39; // ecx
  __int16 v40; // ax
  int v41; // edx
  int v42; // edx
  __int128 v43; // [rsp-20h] [rbp-F0h]
  __int64 *v44; // [rsp+8h] [rbp-C8h]
  __int64 (__fastcall *v45)(__int64, __int64, __int64 *, __int64, __int64); // [rsp+10h] [rbp-C0h]
  __int128 v46; // [rsp+10h] [rbp-C0h]
  __int128 v47; // [rsp+10h] [rbp-C0h]
  unsigned int v48; // [rsp+10h] [rbp-C0h]
  unsigned __int16 v49; // [rsp+20h] [rbp-B0h]
  int v50; // [rsp+20h] [rbp-B0h]
  int v51; // [rsp+28h] [rbp-A8h]
  __int128 v52; // [rsp+30h] [rbp-A0h]
  __int128 v53; // [rsp+50h] [rbp-80h]
  __int128 v54; // [rsp+50h] [rbp-80h]
  unsigned __int64 v55; // [rsp+58h] [rbp-78h]
  __int64 v56; // [rsp+70h] [rbp-60h]
  __int64 v57; // [rsp+80h] [rbp-50h] BYREF
  int v58; // [rsp+88h] [rbp-48h]
  unsigned __int16 v59; // [rsp+90h] [rbp-40h] BYREF
  __int64 v60; // [rsp+98h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 80);
  v57 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v57, v7, 1);
  v8 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * a3);
  v58 = *(_DWORD *)(a2 + 72);
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  v11 = (__int128)_mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  v12 = (unsigned __int16)v9;
  v49 = *v8;
  *(_QWORD *)&v13 = sub_33FAF80(a4, 245, (unsigned int)&v57, (unsigned __int16)v9, v10, v9, v11);
  v52 = v13;
  *(_QWORD *)&v14 = sub_33FE730(a4, &v57, v12, v10, 0, 0.5);
  *(_QWORD *)&v16 = sub_3406EB0(a4, 96, (unsigned int)&v57, v12, v10, v15, v52, v14);
  *(_QWORD *)&v53 = sub_33FAF80(a4, 269, (unsigned int)&v57, v12, v10, v17, v16);
  *((_QWORD *)&v53 + 1) = v18;
  v45 = *(__int64 (__fastcall **)(__int64, __int64, __int64 *, __int64, __int64))(*(_QWORD *)a1 + 528LL);
  v44 = *(__int64 **)(a4 + 64);
  v19 = sub_2E79000(*(__int64 **)(a4 + 40));
  if ( v45 != sub_30368A0 )
  {
    v51 = v45(a1, v19, v44, v12, v10);
    v21 = v51;
    v20 = v42;
    goto LABEL_7;
  }
  v60 = v10;
  v59 = v49;
  if ( v49 )
  {
    if ( (unsigned __int16)(v49 - 17) > 0xD3u )
      goto LABEL_6;
    if ( (unsigned __int16)(v49 - 176) > 0x34u )
      goto LABEL_12;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v59) )
    {
LABEL_6:
      v20 = 0;
      v21 = 2;
      goto LABEL_7;
    }
    if ( !sub_3007100((__int64)&v59) )
      goto LABEL_17;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( v59 )
  {
    if ( (unsigned __int16)(v59 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
LABEL_12:
    v39 = word_4456340[v59 - 1];
    goto LABEL_13;
  }
LABEL_17:
  v39 = sub_3007130((__int64)&v59, 269);
LABEL_13:
  v48 = v39;
  v40 = sub_2D43050(2, v39);
  v20 = 0;
  v21 = v40;
  if ( !v40 )
  {
    v21 = sub_3009400(v44, 2, 0, v48, 0);
    v20 = v41;
  }
LABEL_7:
  v50 = v20;
  *(_QWORD *)&v46 = sub_33FE730(a4, &v57, v12, v10, 0, 0.5);
  HIWORD(v22) = HIWORD(v51);
  *((_QWORD *)&v46 + 1) = v23;
  LOWORD(v22) = v21;
  v24 = v22;
  *(_QWORD *)&v25 = sub_33ED040(a4, 4);
  *(_QWORD *)&v26 = sub_340F900(a4, 208, (unsigned int)&v57, v24, v50, v50, v52, v46, v25);
  v47 = v26;
  *(_QWORD *)&v27 = sub_33FE730(a4, &v57, v12, v10, 0, 0.0);
  v56 = sub_340F900(a4, 205, (unsigned int)&v57, v12, v10, v28, v47, v27, v53);
  v55 = v29 | *((_QWORD *)&v53 + 1) & 0xFFFFFFFF00000000LL;
  *((_QWORD *)&v43 + 1) = v55;
  *(_QWORD *)&v43 = v56;
  *(_QWORD *)&v54 = sub_3406EB0(a4, 152, (unsigned int)&v57, v12, v10, v30, v43, v11);
  *((_QWORD *)&v54 + 1) = v31 | v55 & 0xFFFFFFFF00000000LL;
  sub_33FAF80(a4, 269, (unsigned int)&v57, v12, v10, v32, v11);
  *(_QWORD *)&v47 = sub_33FE730(a4, &v57, v12, v10, 0, 4.503599627370496e15);
  *((_QWORD *)&v47 + 1) = v33;
  *(_QWORD *)&v34 = sub_33ED040(a4, 2);
  *(_QWORD *)&v35 = sub_340F900(a4, 208, (unsigned int)&v57, v24, v50, v50, v52, v47, v34);
  v37 = sub_340F900(a4, 205, (unsigned int)&v57, v12, v10, v36, v35, v11, v54);
  if ( v57 )
    sub_B91220((__int64)&v57, v57);
  return v37;
}
