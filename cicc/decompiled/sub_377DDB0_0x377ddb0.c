// Function: sub_377DDB0
// Address: 0x377ddb0
//
void __fastcall sub_377DDB0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rsi
  __int64 v8; // rsi
  __int16 *v9; // rax
  __int16 v10; // dx
  _QWORD *v11; // rdi
  __m128i v12; // kr00_16
  char v13; // bl
  __m128i v14; // kr10_16
  __int128 v15; // xmm0
  unsigned __int16 *v16; // rax
  __int64 v17; // r8
  unsigned int v18; // ecx
  __int64 v19; // rdx
  __int16 v20; // dx
  __int64 v21; // rsi
  __int64 v22; // rax
  __int16 v23; // cx
  __m128i v24; // xmm5
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // rsi
  __int64 v28; // rdx
  _QWORD *v29; // rsi
  __int16 v30; // cx
  __int64 v31; // rdx
  __m128i v32; // xmm7
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rdx
  __m128i v37; // rax
  __int64 v38; // rcx
  int v39; // edx
  __int64 v40; // rdx
  int v41; // edx
  unsigned __int8 *v42; // rax
  unsigned int v43; // edx
  __int128 v44; // [rsp-10h] [rbp-1F0h]
  __int128 v45; // [rsp+0h] [rbp-1E0h]
  __int64 *v46; // [rsp+18h] [rbp-1C8h]
  __int16 v47; // [rsp+44h] [rbp-19Ch]
  __int128 v48; // [rsp+50h] [rbp-190h]
  __int64 v49; // [rsp+50h] [rbp-190h]
  unsigned __int64 v50; // [rsp+60h] [rbp-180h]
  __m128i *v52; // [rsp+90h] [rbp-150h]
  int v53; // [rsp+A8h] [rbp-138h]
  __int64 v54; // [rsp+B0h] [rbp-130h] BYREF
  int v55; // [rsp+B8h] [rbp-128h]
  __m128i v56; // [rsp+C0h] [rbp-120h] BYREF
  __int64 v57[2]; // [rsp+D0h] [rbp-110h] BYREF
  __m128i v58; // [rsp+E0h] [rbp-100h] BYREF
  __m128i v59; // [rsp+F0h] [rbp-F0h] BYREF
  __int128 v60; // [rsp+100h] [rbp-E0h] BYREF
  __int64 v61; // [rsp+110h] [rbp-D0h]
  __int64 v62; // [rsp+118h] [rbp-C8h]
  __int64 v63; // [rsp+120h] [rbp-C0h]
  __int64 v64; // [rsp+128h] [rbp-B8h]
  __int64 v65; // [rsp+130h] [rbp-B0h]
  __int64 v66; // [rsp+138h] [rbp-A8h]
  __m128i v67; // [rsp+140h] [rbp-A0h] BYREF
  __m128i v68; // [rsp+150h] [rbp-90h] BYREF
  __int64 v69[2]; // [rsp+160h] [rbp-80h] BYREF
  _OWORD v70[2]; // [rsp+170h] [rbp-70h] BYREF
  __m128i v71; // [rsp+190h] [rbp-50h] BYREF
  __m128i v72; // [rsp+1A0h] [rbp-40h] BYREF

  v7 = *(_QWORD *)(a2 + 80);
  v54 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v54, v7, 1);
  v8 = *(_QWORD *)(a1 + 8);
  v55 = *(_DWORD *)(a2 + 72);
  v9 = *(__int16 **)(a2 + 48);
  v10 = *v9;
  *((_QWORD *)&v70[0] + 1) = *((_QWORD *)v9 + 1);
  LOWORD(v70[0]) = v10;
  sub_33D0340((__int64)&v71, v8, (__int64 *)v70);
  v11 = *(_QWORD **)(a1 + 8);
  v12 = v71;
  v13 = (*(_BYTE *)(a2 + 33) >> 2) & 3;
  v14 = v72;
  v15 = (__int128)_mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  v56 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + 40LL));
  v16 = (unsigned __int16 *)(*(_QWORD *)(v56.m128i_i64[0] + 48) + 16LL * v56.m128i_u32[2]);
  v17 = *((_QWORD *)v16 + 1);
  v18 = *v16;
  v71.m128i_i64[0] = 0;
  v71.m128i_i32[2] = 0;
  *(_QWORD *)&v48 = sub_33F17F0(v11, 51, (__int64)&v71, v18, v17);
  *((_QWORD *)&v48 + 1) = v19;
  if ( v71.m128i_i64[0] )
    sub_B91220((__int64)&v71, v71.m128i_i64[0]);
  v20 = *(_WORD *)(a2 + 96);
  v21 = *(_QWORD *)(a1 + 8);
  v57[1] = *(_QWORD *)(a2 + 104);
  v22 = *(_QWORD *)(a2 + 112);
  LOWORD(v57[0]) = v20;
  v23 = *(_WORD *)(v22 + 32);
  v70[0] = _mm_loadu_si128((const __m128i *)(v22 + 40));
  v47 = v23;
  v70[1] = _mm_loadu_si128((const __m128i *)(v22 + 56));
  sub_33D0340((__int64)&v71, v21, v57);
  v24 = _mm_loadu_si128(&v72);
  v58 = _mm_loadu_si128(&v71);
  v59 = v24;
  if ( v71.m128i_i16[0] )
  {
    if ( v71.m128i_i16[0] == 1 || (unsigned __int16)(v71.m128i_i16[0] - 504) <= 7u )
      goto LABEL_25;
    v33 = *(_QWORD *)&byte_444C4A0[16 * v71.m128i_u16[0] - 16];
    if ( !v33 )
    {
LABEL_7:
      v26 = *(_QWORD *)(a1 + 8);
      v27 = *(_QWORD *)a1;
      *(_QWORD *)&v60 = 0;
      DWORD2(v60) = 0;
      sub_3460140((__int64)&v71, (__m128i)v15, v27, a2, v26);
      v67.m128i_i64[1] = 0;
      DWORD2(v60) = v71.m128i_i32[2];
      v28 = *(_QWORD *)(v71.m128i_i64[0] + 48) + 16LL * v71.m128i_u32[2];
      v49 = v72.m128i_u32[2];
      v29 = *(_QWORD **)(a1 + 8);
      v67.m128i_i16[0] = 0;
      *(_QWORD *)&v60 = v71.m128i_i64[0];
      v68.m128i_i16[0] = 0;
      v68.m128i_i64[1] = 0;
      v30 = *(_WORD *)v28;
      v31 = *(_QWORD *)(v28 + 8);
      v50 = v72.m128i_i64[0];
      LOWORD(v69[0]) = v30;
      v69[1] = v31;
      sub_33D0340((__int64)&v71, (__int64)v29, v69);
      v32 = _mm_loadu_si128(&v72);
      v67 = _mm_loadu_si128(&v71);
      v68 = v32;
      sub_3408290((__int64)&v71, v29, &v60, (__int64)&v54, (unsigned int *)&v67, (unsigned int *)&v68, (__m128i)v15);
      *(_QWORD *)a3 = v71.m128i_i64[0];
      *(_DWORD *)(a3 + 8) = v71.m128i_i32[2];
      *(_QWORD *)a4 = v72.m128i_i64[0];
      *(_DWORD *)(a4 + 8) = v72.m128i_i32[2];
      sub_3760E70(a1, a2, 1, v50, v49);
      if ( v54 )
        sub_B91220((__int64)&v54, v54);
      return;
    }
  }
  else
  {
    v61 = sub_3007260((__int64)&v58);
    v62 = v25;
    if ( !v61 )
      goto LABEL_7;
    v33 = sub_3007260((__int64)&v58);
    v63 = v33;
    v64 = v34;
  }
  if ( (v33 & 7) != 0 )
    goto LABEL_7;
  if ( v59.m128i_i16[0] )
  {
    if ( v59.m128i_i16[0] != 1 && (unsigned __int16)(v59.m128i_i16[0] - 504) > 7u )
    {
      v35 = *(_QWORD *)&byte_444C4A0[16 * v59.m128i_u16[0] - 16];
      goto LABEL_14;
    }
LABEL_25:
    BUG();
  }
  v35 = sub_3007260((__int64)&v59);
  v65 = v35;
  v66 = v36;
LABEL_14:
  if ( !v35 )
    goto LABEL_7;
  v46 = *(__int64 **)(a1 + 8);
  v37.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v59);
  v71 = v37;
  if ( (v37.m128i_i8[0] & 7) != 0 )
    goto LABEL_7;
  *(_QWORD *)a3 = sub_33EA290(
                    v46,
                    0,
                    v13,
                    v12.m128i_u32[0],
                    v12.m128i_i64[1],
                    (__int64)&v54,
                    v15,
                    v56.m128i_i64[0],
                    v56.m128i_i64[1],
                    v48,
                    *(_OWORD *)*(_QWORD *)(a2 + 112),
                    *(_QWORD *)(*(_QWORD *)(a2 + 112) + 16LL),
                    v58.m128i_i64[0],
                    v58.m128i_i64[1],
                    *(_BYTE *)(*(_QWORD *)(a2 + 112) + 34LL),
                    v47,
                    (__int64)v70,
                    0);
  v38 = v58.m128i_i64[1];
  v53 = v39;
  v40 = v58.m128i_u32[0];
  *(_DWORD *)(a3 + 8) = v53;
  v71 = 0u;
  v72.m128i_i32[0] = 0;
  v72.m128i_i8[4] = 0;
  sub_3777490(a1, a2, v40, v38, (__int64)&v71, (unsigned int *)&v56, (__m128i)v15, 0);
  v52 = sub_33EA290(
          *(__int64 **)(a1 + 8),
          0,
          v13,
          v14.m128i_u32[0],
          v14.m128i_i64[1],
          (__int64)&v54,
          v15,
          v56.m128i_i64[0],
          v56.m128i_i64[1],
          v48,
          *(_OWORD *)&v71,
          v72.m128i_i64[0],
          v59.m128i_i64[0],
          v59.m128i_i64[1],
          *(_BYTE *)(*(_QWORD *)(a2 + 112) + 34LL),
          v47,
          (__int64)v70,
          0);
  *(_QWORD *)a4 = v52;
  *(_DWORD *)(a4 + 8) = v41;
  *((_QWORD *)&v45 + 1) = 1;
  *(_QWORD *)&v45 = v52;
  *((_QWORD *)&v44 + 1) = 1;
  *(_QWORD *)&v44 = *(_QWORD *)a3;
  v42 = sub_3406EB0(*(_QWORD **)(a1 + 8), 2u, (__int64)&v54, 1, 0, *(_QWORD *)(a1 + 8), v44, v45);
  sub_3760E70(a1, a2, 1, (unsigned __int64)v42, v43 | *((_QWORD *)&v15 + 1) & 0xFFFFFFFF00000000LL);
  if ( v54 )
    sub_B91220((__int64)&v54, v54);
}
