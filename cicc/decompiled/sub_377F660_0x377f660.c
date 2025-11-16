// Function: sub_377F660
// Address: 0x377f660
//
void __fastcall sub_377F660(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rsi
  int v6; // eax
  __int64 v7; // rsi
  __int16 *v8; // rax
  __int16 v9; // dx
  __int64 v10; // rax
  __m128i v11; // xmm0
  __int16 v12; // dx
  __int64 v13; // rsi
  __int64 v14; // rax
  const __m128i *v15; // rax
  __m128i v16; // xmm1
  __m128i v17; // xmm2
  __m128i v18; // xmm3
  unsigned __int16 *v19; // rax
  _QWORD *v20; // rsi
  __int64 v21; // rax
  __int16 v22; // dx
  __int64 v23; // rax
  __m128i v24; // xmm5
  __int64 v25; // rcx
  __m128i *v26; // rax
  bool v27; // zf
  int v28; // edx
  unsigned __int8 *v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rdx
  unsigned __int16 *v32; // rax
  unsigned int v33; // r15d
  __int128 v34; // rax
  __int128 v35; // rax
  unsigned __int8 *v36; // rax
  __int64 v37; // r8
  __int64 v38; // rdx
  char v39; // cl
  __m128i v40; // rax
  unsigned __int64 v41; // rax
  bool v42; // al
  __int64 v43; // r9
  _QWORD *v44; // r15
  int v45; // eax
  const __m128i *v46; // rax
  int v47; // edx
  __int128 v48; // [rsp-40h] [rbp-1D0h]
  __int128 v49; // [rsp-20h] [rbp-1B0h]
  __int128 v50; // [rsp-20h] [rbp-1B0h]
  __int128 v51; // [rsp-10h] [rbp-1A0h]
  __int64 v52; // [rsp+0h] [rbp-190h]
  char v53; // [rsp+0h] [rbp-190h]
  char v54; // [rsp+0h] [rbp-190h]
  unsigned __int8 v55; // [rsp+0h] [rbp-190h]
  _QWORD *v56; // [rsp+18h] [rbp-178h]
  __int64 v57; // [rsp+18h] [rbp-178h]
  __int64 v58; // [rsp+18h] [rbp-178h]
  int v59; // [rsp+18h] [rbp-178h]
  __int64 v60; // [rsp+20h] [rbp-170h]
  unsigned int v61; // [rsp+28h] [rbp-168h]
  __int64 v62; // [rsp+40h] [rbp-150h]
  __int64 v63; // [rsp+40h] [rbp-150h]
  __int64 v64; // [rsp+40h] [rbp-150h]
  __int64 v65; // [rsp+48h] [rbp-148h]
  __int64 v66; // [rsp+48h] [rbp-148h]
  __int128 v69; // [rsp+60h] [rbp-130h]
  char v70; // [rsp+9Fh] [rbp-F1h] BYREF
  __int64 v71; // [rsp+A0h] [rbp-F0h] BYREF
  int v72; // [rsp+A8h] [rbp-E8h]
  __m128i v73; // [rsp+B0h] [rbp-E0h] BYREF
  __m128i v74; // [rsp+C0h] [rbp-D0h] BYREF
  __m128i v75; // [rsp+D0h] [rbp-C0h] BYREF
  __int128 v76; // [rsp+E0h] [rbp-B0h] BYREF
  __int128 v77; // [rsp+F0h] [rbp-A0h] BYREF
  __m128i v78; // [rsp+100h] [rbp-90h] BYREF
  __m128i v79; // [rsp+110h] [rbp-80h] BYREF
  __int64 v80; // [rsp+120h] [rbp-70h] BYREF
  __int64 v81; // [rsp+128h] [rbp-68h]
  __int64 v82; // [rsp+130h] [rbp-60h]
  __m128i v83; // [rsp+140h] [rbp-50h] BYREF
  __m128i v84; // [rsp+150h] [rbp-40h] BYREF

  v5 = *(_QWORD *)(a2 + 80);
  v71 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v71, v5, 1);
  v6 = *(_DWORD *)(a2 + 72);
  v7 = *(_QWORD *)(a1 + 8);
  v73.m128i_i16[0] = 0;
  v72 = v6;
  v8 = *(__int16 **)(a2 + 48);
  v73.m128i_i64[1] = 0;
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  LOWORD(v80) = v9;
  v81 = v10;
  sub_33D0340((__int64)&v83, v7, &v80);
  v11 = _mm_loadu_si128(&v83);
  v12 = *(_WORD *)(a2 + 96);
  v74.m128i_i16[0] = 0;
  v61 = v84.m128i_i32[0];
  v13 = *(_QWORD *)(a1 + 8);
  LOWORD(v80) = v12;
  v60 = v84.m128i_i64[1];
  v14 = *(_QWORD *)(a2 + 104);
  v74.m128i_i64[1] = 0;
  v81 = v14;
  v70 = 0;
  v73 = v11;
  sub_33D04E0((__int64)&v83, v13, (unsigned __int16 *)&v80, (unsigned __int16 *)&v73, &v70);
  v15 = *(const __m128i **)(a2 + 40);
  v16 = _mm_loadu_si128(&v83);
  *(_QWORD *)&v76 = 0;
  DWORD2(v76) = 0;
  v17 = _mm_loadu_si128(&v84);
  v18 = _mm_loadu_si128(v15 + 10);
  v74 = v16;
  *(_QWORD *)&v77 = 0;
  v75 = v18;
  DWORD2(v77) = 0;
  if ( *(_DWORD *)(v18.m128i_i64[0] + 24) == 208 )
  {
    sub_377EF80((__int64 *)a1, v18.m128i_i64[0], (__int64)&v76, (__int64)&v77, v11);
  }
  else
  {
    v19 = (unsigned __int16 *)(*(_QWORD *)(v18.m128i_i64[0] + 48) + 16LL * v75.m128i_u32[2]);
    sub_2FE6CC0((__int64)&v83, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 64LL), *v19, *((_QWORD *)v19 + 1));
    if ( v83.m128i_i8[0] == 6 )
    {
      sub_375E8D0(a1, v75.m128i_u64[0], v75.m128i_i64[1], (__int64)&v76, (__int64)&v77);
    }
    else
    {
      v79.m128i_i64[1] = 0;
      v20 = *(_QWORD **)(a1 + 8);
      v78.m128i_i16[0] = 0;
      v79.m128i_i16[0] = 0;
      v78.m128i_i64[1] = 0;
      v21 = *(_QWORD *)(v75.m128i_i64[0] + 48) + 16LL * v75.m128i_u32[2];
      v22 = *(_WORD *)v21;
      v23 = *(_QWORD *)(v21 + 8);
      LOWORD(v80) = v22;
      v81 = v23;
      sub_33D0340((__int64)&v83, (__int64)v20, &v80);
      v24 = _mm_loadu_si128(&v84);
      v78 = _mm_loadu_si128(&v83);
      v79 = v24;
      sub_3408290(
        (__int64)&v83,
        v20,
        (__int128 *)v75.m128i_i8,
        (__int64)&v71,
        (unsigned int *)&v78,
        (unsigned int *)&v79,
        v11);
      *(_QWORD *)&v76 = v83.m128i_i64[0];
      DWORD2(v76) = v83.m128i_i32[2];
      *(_QWORD *)&v77 = v84.m128i_i64[0];
      DWORD2(v77) = v84.m128i_i32[2];
    }
  }
  sub_3408380(
    &v83,
    *(_QWORD **)(a1 + 8),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 200LL),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 208LL),
    **(unsigned __int16 **)(a2 + 48),
    *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
    v11,
    (__int64)&v71);
  v62 = v83.m128i_i64[0];
  *(_QWORD *)&v69 = v84.m128i_i64[0];
  v65 = v83.m128i_u32[2];
  v25 = *(_QWORD *)(a2 + 40);
  *((_QWORD *)&v69 + 1) = v84.m128i_u32[2];
  *((_QWORD *)&v48 + 1) = v83.m128i_u32[2];
  *(_QWORD *)&v48 = v83.m128i_i64[0];
  v26 = sub_33E8960(
          *(__int64 **)(a1 + 8),
          (*(_WORD *)(a2 + 32) >> 7) & 7,
          (*(_BYTE *)(a2 + 33) >> 2) & 3,
          v73.m128i_u32[0],
          v73.m128i_i64[1],
          (__int64)&v71,
          *(_OWORD *)v25,
          *(_QWORD *)(v25 + 40),
          *(_QWORD *)(v25 + 48),
          *(_OWORD *)(v25 + 80),
          *(_OWORD *)(v25 + 120),
          v76,
          v48,
          v74.m128i_i64[0],
          v74.m128i_i64[1],
          *(const __m128i **)(a2 + 112),
          (*(_BYTE *)(a2 + 33) & 0x10) != 0);
  v27 = v70 == 0;
  *(_QWORD *)a3 = v26;
  *(_DWORD *)(a3 + 8) = v28;
  if ( !v27 )
  {
    *(_QWORD *)a4 = v26;
    *(_DWORD *)(a4 + 8) = *(_DWORD *)(a3 + 8);
    goto LABEL_8;
  }
  v31 = *(_QWORD *)(a2 + 40);
  v52 = v62;
  v56 = *(_QWORD **)(a1 + 8);
  v32 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(v31 + 40) + 48LL) + 16LL * *(unsigned int *)(v31 + 48));
  v33 = *v32;
  v63 = *((_QWORD *)v32 + 1);
  *(_QWORD *)&v34 = sub_33FB160(
                      (__int64)v56,
                      *(_QWORD *)(v31 + 120),
                      *(_QWORD *)(v31 + 128),
                      (__int64)&v71,
                      *v32,
                      v63,
                      v11);
  *((_QWORD *)&v50 + 1) = v65;
  *(_QWORD *)&v50 = v52;
  *(_QWORD *)&v35 = sub_3406EB0(v56, 0x3Au, (__int64)&v71, v33, v63, v63, v50, v34);
  v36 = sub_3406EB0(
          *(_QWORD **)(a1 + 8),
          0x38u,
          (__int64)&v71,
          v33,
          v63,
          v63,
          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
          v35);
  v37 = *(_QWORD *)(a2 + 112);
  v64 = (__int64)v36;
  v66 = v38;
  v39 = *(_BYTE *)(v37 + 34);
  if ( v74.m128i_i16[0] )
  {
    if ( (unsigned __int16)(v74.m128i_i16[0] - 176) <= 0x34u )
      goto LABEL_15;
  }
  else
  {
    v54 = *(_BYTE *)(v37 + 34);
    v58 = *(_QWORD *)(a2 + 112);
    v42 = sub_3007100((__int64)&v74);
    v37 = v58;
    v39 = v54;
    if ( v42 )
    {
LABEL_15:
      v53 = v39;
      v57 = v37;
      v40.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v74);
      v37 = v57;
      v83 = v40;
      v39 = -1;
      v41 = -(((unsigned __int64)v40.m128i_i64[0] >> 3) | (1LL << v53))
          & (((unsigned __int64)v40.m128i_i64[0] >> 3) | (1LL << v53));
      if ( v41 )
      {
        _BitScanReverse64(&v41, v41);
        v39 = 63 - (v41 ^ 0x3F);
      }
    }
  }
  v55 = v39;
  v43 = *(_QWORD *)(v37 + 72);
  v44 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 40LL);
  v83 = _mm_loadu_si128((const __m128i *)(v37 + 40));
  v59 = v43;
  v84 = _mm_loadu_si128((const __m128i *)(v37 + 56));
  v45 = sub_2EAC1E0(v37);
  v81 = 0;
  LODWORD(v82) = v45;
  BYTE4(v82) = 0;
  v80 = 0;
  v46 = (const __m128i *)sub_2E7BD70(v44, 1u, -1, v55, (int)&v83, v59, 0, v82, 1u, 0, 0);
  *(_QWORD *)a4 = sub_33E8960(
                    *(__int64 **)(a1 + 8),
                    (*(_WORD *)(a2 + 32) >> 7) & 7,
                    (*(_BYTE *)(a2 + 33) >> 2) & 3,
                    v61,
                    v60,
                    (__int64)&v71,
                    *(_OWORD *)*(_QWORD *)(a2 + 40),
                    v64,
                    v66,
                    *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL),
                    *(_OWORD *)(*(_QWORD *)(a2 + 40) + 120LL),
                    v77,
                    v69,
                    v17.m128i_i64[0],
                    v17.m128i_i64[1],
                    v46,
                    (*(_BYTE *)(a2 + 33) & 0x10) != 0);
  *(_DWORD *)(a4 + 8) = v47;
LABEL_8:
  *((_QWORD *)&v51 + 1) = 1;
  *(_QWORD *)&v51 = *(_QWORD *)a4;
  *((_QWORD *)&v49 + 1) = 1;
  *(_QWORD *)&v49 = *(_QWORD *)a3;
  v29 = sub_3406EB0(*(_QWORD **)(a1 + 8), 2u, (__int64)&v71, 1, 0, *(_QWORD *)(a1 + 8), v49, v51);
  sub_3760E70(a1, a2, 1, (unsigned __int64)v29, v30);
  if ( v71 )
    sub_B91220((__int64)&v71, v71);
}
