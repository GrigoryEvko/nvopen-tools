// Function: sub_3848350
// Address: 0x3848350
//
unsigned __int8 *__fastcall sub_3848350(__int64 *a1, __int64 a2, __m128i a3)
{
  __int16 *v4; // rdx
  __int16 v5; // ax
  __int64 v6; // rdx
  __int64 v7; // rsi
  __int64 v8; // r9
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // r14
  unsigned __int64 v12; // r13
  __int64 v13; // rax
  unsigned __int16 v14; // si
  __int64 v15; // r8
  __int64 (__fastcall *v16)(__int64, __int64, unsigned int, __int64); // rax
  int v17; // r9d
  int v18; // ecx
  int v19; // eax
  unsigned int v20; // r9d
  unsigned int v21; // esi
  __int128 v22; // rax
  __int64 v23; // r12
  __int16 v24; // ax
  __int64 v25; // r9
  __int64 v26; // rax
  __int64 v27; // r13
  __int64 v28; // r12
  unsigned __int16 *v29; // rax
  unsigned int v30; // edx
  __int64 v31; // r14
  unsigned __int64 v32; // r13
  __int64 v33; // r9
  unsigned int v34; // edx
  __int128 v35; // rax
  __int64 v36; // r9
  unsigned int v37; // edx
  __int64 v38; // r9
  int v39; // r9d
  unsigned __int8 *v40; // r12
  __int64 v42; // rdx
  int v43; // eax
  __int64 v44; // rdx
  __int128 v45; // [rsp-30h] [rbp-140h]
  __int128 v46; // [rsp-20h] [rbp-130h]
  __int128 v47; // [rsp-10h] [rbp-120h]
  __int128 v48; // [rsp-10h] [rbp-120h]
  __int128 v49; // [rsp-10h] [rbp-120h]
  __int64 v50; // [rsp+0h] [rbp-110h]
  __int64 *v51; // [rsp+8h] [rbp-108h]
  _QWORD *v52; // [rsp+8h] [rbp-108h]
  unsigned int v53; // [rsp+10h] [rbp-100h]
  unsigned __int8 *v54; // [rsp+10h] [rbp-100h]
  __int16 v55; // [rsp+1Ah] [rbp-F6h]
  __int64 v56; // [rsp+20h] [rbp-F0h]
  int v57; // [rsp+30h] [rbp-E0h]
  unsigned int v58; // [rsp+30h] [rbp-E0h]
  __int128 v59; // [rsp+30h] [rbp-E0h]
  unsigned __int8 *v60; // [rsp+50h] [rbp-C0h]
  unsigned int v61; // [rsp+80h] [rbp-90h] BYREF
  __int64 v62; // [rsp+88h] [rbp-88h]
  __int64 v63; // [rsp+90h] [rbp-80h] BYREF
  int v64; // [rsp+98h] [rbp-78h]
  __m128i v65; // [rsp+A0h] [rbp-70h] BYREF
  __int128 v66; // [rsp+B0h] [rbp-60h] BYREF
  __int16 v67; // [rsp+C0h] [rbp-50h] BYREF
  __int64 v68; // [rsp+C8h] [rbp-48h]
  __int64 v69; // [rsp+D0h] [rbp-40h]

  v4 = *(__int16 **)(a2 + 48);
  v5 = *v4;
  v6 = *((_QWORD *)v4 + 1);
  LOWORD(v61) = v5;
  v62 = v6;
  if ( v5 )
  {
    if ( (unsigned __int16)(v5 - 176) > 0x34u )
    {
LABEL_3:
      v57 = word_4456340[(unsigned __int16)v61 - 1];
      goto LABEL_6;
    }
  }
  else if ( !sub_3007100((__int64)&v61) )
  {
    goto LABEL_5;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( (_WORD)v61 )
  {
    if ( (unsigned __int16)(v61 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    goto LABEL_3;
  }
LABEL_5:
  v57 = sub_3007130((__int64)&v61, a2);
LABEL_6:
  v7 = *(_QWORD *)(a2 + 80);
  v63 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v63, v7, 1);
  v8 = *a1;
  v9 = a1[1];
  v64 = *(_DWORD *)(a2 + 72);
  v10 = *(_QWORD *)(a2 + 40);
  v11 = *(unsigned int *)(v10 + 48);
  v12 = *(_QWORD *)(v10 + 40);
  v13 = 16 * v11 + *(_QWORD *)(v12 + 48);
  v14 = *(_WORD *)v13;
  v15 = *(_QWORD *)(v13 + 8);
  v16 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v8 + 592LL);
  if ( v16 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v67, v8, *(_QWORD *)(v9 + 64), v14, v15);
    v17 = (unsigned __int16)v68;
    v50 = v69;
  }
  else
  {
    v43 = v16(v8, *(_QWORD *)(v9 + 64), v14, v15);
    v50 = v44;
    v17 = v43;
  }
  v18 = v57;
  v58 = v17;
  v53 = 2 * v18;
  v51 = *(__int64 **)(a1[1] + 64);
  LOWORD(v19) = sub_2D43050(v17, 2 * v18);
  v20 = v58;
  v56 = 0;
  if ( !(_WORD)v19 )
  {
    v19 = sub_3009400(v51, v58, v50, v53, 0);
    v55 = HIWORD(v19);
    v56 = v42;
  }
  HIWORD(v21) = v55;
  LOWORD(v21) = v19;
  *(_QWORD *)&v22 = sub_33FAF80(a1[1], 234, (__int64)&v63, v21, v56, v20, a3);
  v23 = *(_QWORD *)(v12 + 48) + 16 * v11;
  v65.m128i_i32[2] = 0;
  DWORD2(v66) = 0;
  v59 = v22;
  *((_QWORD *)&v22 + 1) = *(_QWORD *)(v23 + 8);
  v65.m128i_i64[0] = 0;
  *(_QWORD *)&v66 = 0;
  v24 = *(_WORD *)v23;
  v68 = *((_QWORD *)&v22 + 1);
  v67 = v24;
  if ( v24 )
  {
    if ( (unsigned __int16)(v24 - 2) <= 7u
      || (unsigned __int16)(v24 - 17) <= 0x6Cu
      || (unsigned __int16)(v24 - 176) <= 0x1Fu )
    {
      goto LABEL_16;
    }
  }
  else if ( sub_3007070((__int64)&v67) )
  {
LABEL_16:
    sub_375E510((__int64)a1, v12, v11, (__int64)&v65, (__int64)&v66);
    goto LABEL_17;
  }
  sub_375E6F0((__int64)a1, v12, v11, (__int64)&v65, (__int64)&v66);
LABEL_17:
  if ( *(_BYTE *)sub_2E79000(*(__int64 **)(a1[1] + 40)) )
  {
    a3 = _mm_loadu_si128(&v65);
    v65.m128i_i64[0] = v66;
    v65.m128i_i32[2] = DWORD2(v66);
    *(_QWORD *)&v66 = a3.m128i_i64[0];
    DWORD2(v66) = a3.m128i_i32[2];
  }
  v26 = *(_QWORD *)(a2 + 40);
  v27 = *(_QWORD *)(v26 + 88);
  v28 = *(_QWORD *)(v26 + 80);
  v29 = (unsigned __int16 *)(*(_QWORD *)(v28 + 48) + 16LL * *(unsigned int *)(v26 + 88));
  *((_QWORD *)&v47 + 1) = v27;
  *(_QWORD *)&v47 = v28;
  *((_QWORD *)&v46 + 1) = v27;
  *(_QWORD *)&v46 = v28;
  v54 = sub_3406EB0((_QWORD *)a1[1], 0x38u, (__int64)&v63, *v29, *((_QWORD *)v29 + 1), v25, v46, v47);
  v31 = 16LL * v30;
  v32 = v30 | v27 & 0xFFFFFFFF00000000LL;
  *((_QWORD *)&v48 + 1) = v32;
  *(_QWORD *)&v48 = v54;
  *(_QWORD *)&v59 = sub_340F900((_QWORD *)a1[1], 0x9Du, (__int64)&v63, v21, v56, v33, v59, *(_OWORD *)&v65, v48);
  v52 = (_QWORD *)a1[1];
  *((_QWORD *)&v59 + 1) = v34 | *((_QWORD *)&v59 + 1) & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v35 = sub_3400BD0(
                      (__int64)v52,
                      1,
                      (__int64)&v63,
                      *(unsigned __int16 *)(v31 + *((_QWORD *)v54 + 6)),
                      *(_QWORD *)(v31 + *((_QWORD *)v54 + 6) + 8),
                      0,
                      a3,
                      0);
  *((_QWORD *)&v45 + 1) = v32;
  *(_QWORD *)&v45 = v54;
  v60 = sub_3406EB0(
          v52,
          0x38u,
          (__int64)&v63,
          *(unsigned __int16 *)(*((_QWORD *)v54 + 6) + v31),
          *(_QWORD *)(*((_QWORD *)v54 + 6) + v31 + 8),
          v36,
          v45,
          v35);
  *((_QWORD *)&v49 + 1) = v37 | v32 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v49 = v60;
  sub_340F900((_QWORD *)a1[1], 0x9Du, (__int64)&v63, v21, v56, v38, v59, v66, v49);
  v40 = sub_33FAF80(a1[1], 234, (__int64)&v63, v61, v62, v39, a3);
  if ( v63 )
    sub_B91220((__int64)&v63, v63);
  return v40;
}
