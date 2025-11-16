// Function: sub_3784B90
// Address: 0x3784b90
//
unsigned __int8 *__fastcall sub_3784B90(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rsi
  __m128i v5; // xmm0
  __int64 v6; // rcx
  __int64 v7; // rax
  __int64 v8; // rax
  int v9; // eax
  _QWORD *v10; // r15
  __int64 v11; // rax
  __int16 v12; // dx
  __int64 v13; // rax
  __m128i v14; // xmm4
  _QWORD *v15; // rsi
  __int64 v16; // rax
  __int16 v17; // dx
  __int64 v18; // rax
  __m128i v19; // xmm6
  _QWORD *v20; // rsi
  __int64 v21; // rax
  __int16 v22; // dx
  __int64 v23; // rax
  __m128i v24; // xmm0
  __int64 v25; // r9
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // r15
  __int64 v29; // r14
  __int64 v30; // r9
  __int128 v31; // rax
  __int64 v32; // r9
  unsigned __int8 *v33; // r14
  __int128 v35; // [rsp-30h] [rbp-1A0h]
  __int128 v36; // [rsp-20h] [rbp-190h]
  __int128 v37; // [rsp+10h] [rbp-160h]
  __int64 v38; // [rsp+20h] [rbp-150h]
  unsigned int v39; // [rsp+28h] [rbp-148h]
  __int64 v40; // [rsp+30h] [rbp-140h]
  unsigned int v41; // [rsp+38h] [rbp-138h]
  __int128 v42; // [rsp+40h] [rbp-130h]
  __int128 v43; // [rsp+50h] [rbp-120h]
  __int128 v44; // [rsp+60h] [rbp-110h]
  __int128 v45; // [rsp+70h] [rbp-100h]
  __m128i v46; // [rsp+80h] [rbp-F0h] BYREF
  __m128i v47; // [rsp+90h] [rbp-E0h] BYREF
  __m128i v48; // [rsp+A0h] [rbp-D0h] BYREF
  __int64 v49; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v50; // [rsp+B8h] [rbp-B8h]
  __int64 v51; // [rsp+C0h] [rbp-B0h] BYREF
  int v52; // [rsp+C8h] [rbp-A8h]
  __int64 v53; // [rsp+D0h] [rbp-A0h] BYREF
  int v54; // [rsp+D8h] [rbp-98h]
  __int64 v55; // [rsp+E0h] [rbp-90h] BYREF
  int v56; // [rsp+E8h] [rbp-88h]
  __m128i v57; // [rsp+F0h] [rbp-80h] BYREF
  __m128i v58; // [rsp+100h] [rbp-70h] BYREF
  __int64 v59; // [rsp+110h] [rbp-60h] BYREF
  __int64 v60; // [rsp+118h] [rbp-58h]
  __m128i v61; // [rsp+120h] [rbp-50h] BYREF
  __m128i v62; // [rsp+130h] [rbp-40h] BYREF

  v3 = *(_QWORD *)(a2 + 40);
  v4 = *(_QWORD *)(a2 + 80);
  v5 = _mm_loadu_si128((const __m128i *)v3);
  v6 = *(_QWORD *)(v3 + 40);
  v47 = _mm_loadu_si128((const __m128i *)(v3 + 40));
  v46 = v5;
  v7 = *(_QWORD *)(v6 + 48) + 16LL * v47.m128i_u32[2];
  v48 = _mm_loadu_si128((const __m128i *)(v3 + 80));
  LOWORD(v6) = *(_WORD *)v7;
  v8 = *(_QWORD *)(v7 + 8);
  v51 = v4;
  LOWORD(v49) = v6;
  v50 = v8;
  if ( v4 )
  {
    sub_B96E90((__int64)&v51, v4, 1);
    v3 = *(_QWORD *)(a2 + 40);
  }
  v9 = *(_DWORD *)(a2 + 72);
  v54 = 0;
  v56 = 0;
  v52 = v9;
  v53 = 0;
  v55 = 0;
  sub_375E8D0(a1, *(_QWORD *)v3, *(_QWORD *)(v3 + 8), (__int64)&v53, (__int64)&v55);
  sub_33D0340((__int64)&v61, *(_QWORD *)(a1 + 8), &v49);
  v57.m128i_i64[1] = 0;
  v58.m128i_i64[1] = 0;
  v10 = *(_QWORD **)(a1 + 8);
  v41 = v61.m128i_i32[0];
  v38 = v62.m128i_i64[1];
  v57.m128i_i16[0] = 0;
  v58.m128i_i16[0] = 0;
  v40 = v61.m128i_i64[1];
  v11 = *(_QWORD *)(v47.m128i_i64[0] + 48) + 16LL * v47.m128i_u32[2];
  v39 = v62.m128i_i32[0];
  v12 = *(_WORD *)v11;
  v13 = *(_QWORD *)(v11 + 8);
  LOWORD(v59) = v12;
  v60 = v13;
  sub_33D0340((__int64)&v61, (__int64)v10, &v59);
  v14 = _mm_loadu_si128(&v62);
  v57 = _mm_loadu_si128(&v61);
  v58 = v14;
  sub_3408290(
    (__int64)&v61,
    v10,
    (__int128 *)v47.m128i_i8,
    (__int64)&v51,
    (unsigned int *)&v57,
    (unsigned int *)&v58,
    v5);
  v57.m128i_i16[0] = 0;
  v58.m128i_i16[0] = 0;
  *((_QWORD *)&v45 + 1) = v61.m128i_u32[2];
  v57.m128i_i64[1] = 0;
  v15 = *(_QWORD **)(a1 + 8);
  v58.m128i_i64[1] = 0;
  *(_QWORD *)&v45 = v61.m128i_i64[0];
  *((_QWORD *)&v44 + 1) = v62.m128i_u32[2];
  v16 = *(_QWORD *)(v48.m128i_i64[0] + 48) + 16LL * v48.m128i_u32[2];
  v17 = *(_WORD *)v16;
  v18 = *(_QWORD *)(v16 + 8);
  *(_QWORD *)&v44 = v62.m128i_i64[0];
  LOWORD(v59) = v17;
  v60 = v18;
  sub_33D0340((__int64)&v61, (__int64)v15, &v59);
  v19 = _mm_loadu_si128(&v62);
  v57 = _mm_loadu_si128(&v61);
  v58 = v19;
  sub_3408290(
    (__int64)&v61,
    v15,
    (__int128 *)v48.m128i_i8,
    (__int64)&v51,
    (unsigned int *)&v57,
    (unsigned int *)&v58,
    v5);
  v57.m128i_i16[0] = 0;
  v58.m128i_i16[0] = 0;
  v20 = *(_QWORD **)(a1 + 8);
  v57.m128i_i64[1] = 0;
  *((_QWORD *)&v43 + 1) = v61.m128i_u32[2];
  v58.m128i_i64[1] = 0;
  *(_QWORD *)&v43 = v61.m128i_i64[0];
  *((_QWORD *)&v42 + 1) = v62.m128i_u32[2];
  *(_QWORD *)&v42 = v62.m128i_i64[0];
  v21 = *(_QWORD *)(v46.m128i_i64[0] + 48) + 16LL * v46.m128i_u32[2];
  v22 = *(_WORD *)v21;
  v23 = *(_QWORD *)(v21 + 8);
  LOWORD(v59) = v22;
  v60 = v23;
  sub_33D0340((__int64)&v61, (__int64)v20, &v59);
  v24 = _mm_loadu_si128(&v62);
  v57 = _mm_loadu_si128(&v61);
  v58 = v24;
  sub_3408290(
    (__int64)&v61,
    v20,
    (__int128 *)v46.m128i_i8,
    (__int64)&v51,
    (unsigned int *)&v57,
    (unsigned int *)&v58,
    v24);
  *(_QWORD *)&v37 = v62.m128i_i64[0];
  *((_QWORD *)&v35 + 1) = v61.m128i_u32[2];
  *(_QWORD *)&v35 = v61.m128i_i64[0];
  *((_QWORD *)&v37 + 1) = v62.m128i_u32[2];
  v26 = sub_340F900(*(_QWORD **)(a1 + 8), 0xCEu, (__int64)&v51, v41, v40, v25, v35, v45, v43);
  v28 = v27;
  v29 = v26;
  *(_QWORD *)&v31 = sub_340F900(*(_QWORD **)(a1 + 8), 0xCEu, (__int64)&v51, v39, v38, v30, v37, v44, v42);
  *((_QWORD *)&v36 + 1) = v28;
  *(_QWORD *)&v36 = v29;
  v33 = sub_3406EB0(*(_QWORD **)(a1 + 8), 0x9Fu, (__int64)&v51, (unsigned int)v49, v50, v32, v36, v31);
  if ( v51 )
    sub_B91220((__int64)&v51, v51);
  return v33;
}
