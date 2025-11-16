// Function: sub_2026D80
// Address: 0x2026d80
//
void __fastcall sub_2026D80(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rsi
  __int64 v7; // rsi
  __int64 v8; // rax
  __int8 v9; // dl
  __int64 v10; // rax
  _QWORD *v11; // rdi
  __m128i v12; // kr00_16
  __int64 v13; // rax
  __int128 v14; // xmm0
  __m128i v15; // xmm1
  unsigned __int8 *v16; // rax
  __int64 v17; // r8
  unsigned int v18; // ecx
  __int64 v19; // r9
  __int64 v20; // rdx
  __int64 v21; // rax
  char v22; // dl
  __int64 v23; // rsi
  __int64 v24; // rax
  __int16 v25; // cx
  __m128i v26; // xmm2
  unsigned __int16 v27; // di
  __m128i v28; // xmm3
  _QWORD *v29; // rdi
  __int64 v30; // rax
  char v31; // di
  int v32; // edx
  int v33; // r14d
  __int64 v34; // r14
  __int128 v35; // rax
  __int64 *v36; // rax
  _QWORD *v37; // rdi
  unsigned int v38; // edx
  unsigned __int64 v39; // rax
  __int64 v40; // rdx
  unsigned __int64 v41; // rcx
  __int32 v42; // eax
  int v43; // edx
  __int64 *v44; // rax
  unsigned int v45; // edx
  const __m128i *v46; // r9
  __int8 v47; // si
  __int64 v48; // r14
  __int64 v49; // rax
  __int128 v50; // [rsp-10h] [rbp-180h]
  __int64 v51; // [rsp+10h] [rbp-160h]
  unsigned int v52; // [rsp+18h] [rbp-158h]
  __int64 *v53; // [rsp+18h] [rbp-158h]
  unsigned __int16 v55; // [rsp+3Ch] [rbp-134h]
  __int128 v56; // [rsp+40h] [rbp-130h]
  __int64 v58; // [rsp+58h] [rbp-118h]
  unsigned int v59; // [rsp+60h] [rbp-110h]
  unsigned int v60; // [rsp+64h] [rbp-10Ch]
  __int64 v61; // [rsp+68h] [rbp-108h]
  __int64 v62; // [rsp+80h] [rbp-F0h]
  __int64 v63; // [rsp+88h] [rbp-E8h]
  __int64 v64; // [rsp+A0h] [rbp-D0h]
  __int64 v65; // [rsp+D0h] [rbp-A0h] BYREF
  int v66; // [rsp+D8h] [rbp-98h]
  _QWORD v67[2]; // [rsp+E0h] [rbp-90h] BYREF
  __int128 v68; // [rsp+F0h] [rbp-80h] BYREF
  __m128i v69; // [rsp+100h] [rbp-70h] BYREF
  __int64 v70; // [rsp+110h] [rbp-60h]
  __int128 v71; // [rsp+120h] [rbp-50h] BYREF
  __m128i v72; // [rsp+130h] [rbp-40h] BYREF

  v6 = *(_QWORD *)(a2 + 72);
  v65 = v6;
  if ( v6 )
    sub_1623A60((__int64)&v65, v6, 2);
  v7 = *(_QWORD *)(a1 + 8);
  v66 = *(_DWORD *)(a2 + 64);
  v8 = *(_QWORD *)(a2 + 40);
  v9 = *(_BYTE *)v8;
  v10 = *(_QWORD *)(v8 + 8);
  v69.m128i_i8[0] = v9;
  v69.m128i_i64[1] = v10;
  sub_1D19A30((__int64)&v71, v7, &v69);
  v11 = *(_QWORD **)(a1 + 8);
  v52 = v71;
  v51 = *((_QWORD *)&v71 + 1);
  v12 = v72;
  v59 = (*(_BYTE *)(a2 + 27) >> 2) & 3;
  v13 = *(_QWORD *)(a2 + 32);
  v14 = (__int128)_mm_loadu_si128((const __m128i *)v13);
  v15 = _mm_loadu_si128((const __m128i *)(v13 + 40));
  v61 = *(_QWORD *)(v13 + 40);
  v58 = 16LL * *(unsigned int *)(v13 + 48);
  v16 = (unsigned __int8 *)(*(_QWORD *)(v61 + 40) + v58);
  v17 = *((_QWORD *)v16 + 1);
  v18 = *v16;
  *(_QWORD *)&v71 = 0;
  DWORD2(v71) = 0;
  *(_QWORD *)&v56 = sub_1D2B300(v11, 0x30u, (__int64)&v71, v18, v17, v19);
  *((_QWORD *)&v56 + 1) = v20;
  if ( (_QWORD)v71 )
    sub_161E7C0((__int64)&v71, v71);
  v21 = *(_QWORD *)(a2 + 96);
  v22 = *(_BYTE *)(a2 + 88);
  LOBYTE(v68) = 0;
  v23 = *(_QWORD *)(a1 + 8);
  *((_QWORD *)&v68 + 1) = 0;
  v67[1] = v21;
  v24 = *(_QWORD *)(a2 + 104);
  LOBYTE(v67[0]) = v22;
  v25 = *(_WORD *)(v24 + 34);
  v26 = _mm_loadu_si128((const __m128i *)(v24 + 40));
  v27 = *(_WORD *)(v24 + 32);
  v69 = v26;
  v55 = v27;
  v60 = (unsigned int)(1 << v25) >> 1;
  v70 = *(_QWORD *)(v24 + 56);
  sub_1D19A30((__int64)&v71, v23, v67);
  v28 = _mm_loadu_si128(&v72);
  v29 = *(_QWORD **)(a1 + 8);
  v68 = v71;
  v30 = sub_1D264C0(
          v29,
          0,
          v59,
          v52,
          v51,
          (__int64)&v65,
          v14,
          v15.m128i_i64[0],
          v15.m128i_i64[1],
          v56,
          *(_OWORD *)*(_QWORD *)(a2 + 104),
          *(_QWORD *)(*(_QWORD *)(a2 + 104) + 16LL),
          v71,
          *((__int64 *)&v71 + 1),
          v60,
          v55,
          (__int64)&v69,
          0);
  v31 = v68;
  *(_QWORD *)a3 = v30;
  *(_DWORD *)(a3 + 8) = v32;
  if ( v31 )
    v33 = sub_2021900(v31);
  else
    v33 = sub_1F58D40((__int64)&v68);
  v34 = (unsigned int)(v33 + 7) >> 3;
  v53 = *(__int64 **)(a1 + 8);
  *(_QWORD *)&v35 = sub_1D38BB0(
                      (__int64)v53,
                      v34,
                      (__int64)&v65,
                      *(unsigned __int8 *)(*(_QWORD *)(v61 + 40) + v58),
                      *(const void ***)(*(_QWORD *)(v61 + 40) + v58 + 8),
                      0,
                      (__m128i)v14,
                      *(double *)v15.m128i_i64,
                      v26,
                      0);
  v36 = sub_1D332F0(
          v53,
          52,
          (__int64)&v65,
          *(unsigned __int8 *)(*(_QWORD *)(v61 + 40) + v58),
          *(const void ***)(*(_QWORD *)(v61 + 40) + v58 + 8),
          3u,
          *(double *)&v14,
          *(double *)v15.m128i_i64,
          v26,
          v15.m128i_i64[0],
          v15.m128i_u64[1],
          v35);
  v37 = *(_QWORD **)(a1 + 8);
  v62 = (__int64)v36;
  v39 = v38 | v15.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  v40 = *(_QWORD *)(a2 + 104);
  v63 = v39;
  v41 = *(_QWORD *)v40 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v41 )
  {
    v47 = *(_BYTE *)(v40 + 16);
    v48 = *(_QWORD *)(v40 + 8) + v34;
    if ( (*(_QWORD *)v40 & 4) != 0 )
    {
      *((_QWORD *)&v71 + 1) = v48;
      v72.m128i_i8[0] = v47;
      *(_QWORD *)&v71 = v41 | 4;
      v72.m128i_i32[1] = *(_DWORD *)(v41 + 12);
    }
    else
    {
      *(_QWORD *)&v71 = *(_QWORD *)v40 & 0xFFFFFFFFFFFFFFF8LL;
      *((_QWORD *)&v71 + 1) = v48;
      v72.m128i_i8[0] = v47;
      v49 = *(_QWORD *)v41;
      if ( *(_BYTE *)(*(_QWORD *)v41 + 8LL) == 16 )
        v49 = **(_QWORD **)(v49 + 16);
      v72.m128i_i32[1] = *(_DWORD *)(v49 + 8) >> 8;
    }
  }
  else
  {
    v42 = *(_DWORD *)(v40 + 20);
    v72.m128i_i32[0] = 0;
    v71 = 0u;
    v72.m128i_i32[1] = v42;
  }
  v64 = sub_1D264C0(
          v37,
          0,
          v59,
          v12.m128i_u32[0],
          v12.m128i_i64[1],
          (__int64)&v65,
          v14,
          v62,
          v63,
          v56,
          v71,
          v72.m128i_i64[0],
          v28.m128i_i64[0],
          v28.m128i_i64[1],
          v60,
          v55,
          (__int64)&v69,
          0);
  *(_QWORD *)a4 = v64;
  *(_DWORD *)(a4 + 8) = v43;
  *((_QWORD *)&v50 + 1) = 1;
  *(_QWORD *)&v50 = v64;
  v44 = sub_1D332F0(
          *(__int64 **)(a1 + 8),
          2,
          (__int64)&v65,
          1,
          0,
          0,
          *(double *)&v14,
          *(double *)v15.m128i_i64,
          v26,
          *(_QWORD *)a3,
          1u,
          v50);
  sub_2013400(a1, a2, 1, (__int64)v44, (__m128i *)(v45 | *((_QWORD *)&v14 + 1) & 0xFFFFFFFF00000000LL), v46);
  if ( v65 )
    sub_161E7C0((__int64)&v65, v65);
}
