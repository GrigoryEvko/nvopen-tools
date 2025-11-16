// Function: sub_3788820
// Address: 0x3788820
//
__m128i *__fastcall sub_3788820(__int64 a1, __int64 a2, int a3)
{
  __int64 v5; // rax
  __m128i v6; // xmm0
  __m128i v7; // xmm1
  __m128i v8; // xmm2
  __m128i v9; // xmm3
  unsigned __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rsi
  int v13; // eax
  __int64 v14; // rsi
  unsigned __int16 *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r8
  _QWORD *v18; // rsi
  __int64 v19; // rax
  __int16 v20; // dx
  __int64 v21; // rax
  __m128i v22; // xmm1
  unsigned __int16 *v23; // rax
  _QWORD *v24; // rsi
  __int64 v25; // rax
  __int16 v26; // dx
  __int64 v27; // rax
  __m128i v28; // xmm3
  __int64 v29; // rax
  __int16 v30; // dx
  __int64 v31; // rsi
  __int64 v32; // rax
  __int16 v33; // dx
  __int64 v34; // rax
  __m128i v35; // xmm5
  _QWORD *v36; // rdi
  __int64 v37; // rax
  __int64 v38; // r9
  const __m128i *v39; // rax
  __m128i *result; // rax
  unsigned int v41; // edx
  unsigned __int8 *v42; // rax
  __m128i *v43; // r10
  __int64 v44; // r11
  unsigned int v45; // edx
  bool v46; // al
  __m128i v47; // rax
  unsigned __int8 v48; // di
  unsigned __int64 v49; // rax
  int v50; // eax
  __int64 v51; // r13
  __m128i *v52; // r10
  __int64 v53; // r11
  __int64 v54; // r9
  _QWORD *v55; // rdi
  const __m128i *v56; // rax
  unsigned int v57; // edx
  __int64 v58; // r9
  __m128i v59; // rax
  unsigned __int64 v60; // rax
  unsigned __int64 v61; // rcx
  int v62; // edx
  char v63; // si
  __int64 v64; // rdi
  __int128 v65; // [rsp-10h] [rbp-1B0h]
  __m128i *v66; // [rsp+0h] [rbp-1A0h]
  __m128i *v67; // [rsp+0h] [rbp-1A0h]
  __int128 v68; // [rsp+0h] [rbp-1A0h]
  __m128i *v69; // [rsp+0h] [rbp-1A0h]
  __int64 v70; // [rsp+8h] [rbp-198h]
  __int64 v71; // [rsp+8h] [rbp-198h]
  __int64 v72; // [rsp+8h] [rbp-198h]
  __m128i v73; // [rsp+10h] [rbp-190h]
  unsigned __int64 v74; // [rsp+28h] [rbp-178h]
  __int128 v75; // [rsp+30h] [rbp-170h]
  __m128i v77; // [rsp+50h] [rbp-150h]
  __m128i *v78; // [rsp+50h] [rbp-150h]
  unsigned __int8 v80; // [rsp+67h] [rbp-139h]
  __m128i *v81; // [rsp+70h] [rbp-130h]
  char v82; // [rsp+9Fh] [rbp-101h] BYREF
  __m128i v83; // [rsp+A0h] [rbp-100h] BYREF
  __m128i v84; // [rsp+B0h] [rbp-F0h] BYREF
  __int64 v85; // [rsp+C0h] [rbp-E0h] BYREF
  int v86; // [rsp+C8h] [rbp-D8h]
  __int64 v87; // [rsp+D0h] [rbp-D0h] BYREF
  __int64 v88; // [rsp+D8h] [rbp-C8h]
  __int64 v89; // [rsp+E0h] [rbp-C0h] BYREF
  __int64 v90; // [rsp+E8h] [rbp-B8h]
  __int128 v91; // [rsp+F0h] [rbp-B0h] BYREF
  __int128 v92; // [rsp+100h] [rbp-A0h] BYREF
  __m128i v93; // [rsp+110h] [rbp-90h] BYREF
  __m128i v94; // [rsp+120h] [rbp-80h] BYREF
  __int128 v95; // [rsp+130h] [rbp-70h] BYREF
  __int64 v96; // [rsp+140h] [rbp-60h]
  __m128i v97; // [rsp+150h] [rbp-50h] BYREF
  __m128i v98; // [rsp+160h] [rbp-40h] BYREF

  v5 = *(_QWORD *)(a2 + 40);
  v6 = _mm_loadu_si128((const __m128i *)(v5 + 80));
  v7 = _mm_loadu_si128((const __m128i *)(v5 + 120));
  v8 = _mm_loadu_si128((const __m128i *)(v5 + 160));
  v9 = _mm_loadu_si128((const __m128i *)(v5 + 40));
  v74 = *(_QWORD *)v5;
  v10 = *(_QWORD *)(v5 + 8);
  v11 = *(_QWORD *)(a2 + 112);
  v77 = v6;
  v12 = *(_QWORD *)(a2 + 80);
  v75 = (__int128)v7;
  LOBYTE(v11) = *(_BYTE *)(v11 + 34);
  v85 = v12;
  v80 = v11;
  v83 = v8;
  v84 = v9;
  if ( v12 )
    sub_B96E90((__int64)&v85, v12, 1);
  v13 = *(_DWORD *)(a2 + 72);
  v14 = *(_QWORD *)a1;
  LODWORD(v88) = 0;
  v86 = v13;
  LODWORD(v90) = 0;
  v87 = 0;
  v15 = (unsigned __int16 *)(*(_QWORD *)(v84.m128i_i64[0] + 48) + 16LL * v84.m128i_u32[2]);
  v16 = *(_QWORD *)(a1 + 8);
  v17 = *((_QWORD *)v15 + 1);
  v89 = 0;
  sub_2FE6CC0((__int64)&v97, v14, *(_QWORD *)(v16 + 64), *v15, v17);
  if ( v97.m128i_i8[0] == 6 )
  {
    sub_375E8D0(a1, v84.m128i_u64[0], v84.m128i_i64[1], (__int64)&v87, (__int64)&v89);
  }
  else
  {
    v94.m128i_i16[0] = 0;
    v93.m128i_i16[0] = 0;
    v18 = *(_QWORD **)(a1 + 8);
    v93.m128i_i64[1] = 0;
    v94.m128i_i64[1] = 0;
    v19 = *(_QWORD *)(v84.m128i_i64[0] + 48) + 16LL * v84.m128i_u32[2];
    v20 = *(_WORD *)v19;
    v21 = *(_QWORD *)(v19 + 8);
    LOWORD(v95) = v20;
    *((_QWORD *)&v95 + 1) = v21;
    sub_33D0340((__int64)&v97, (__int64)v18, (__int64 *)&v95);
    v6 = _mm_loadu_si128(&v97);
    v22 = _mm_loadu_si128(&v98);
    v93 = v6;
    v94 = v22;
    sub_3408290(
      (__int64)&v97,
      v18,
      (__int128 *)v84.m128i_i8,
      (__int64)&v85,
      (unsigned int *)&v93,
      (unsigned int *)&v94,
      v6);
    v87 = v97.m128i_i64[0];
    LODWORD(v88) = v97.m128i_i32[2];
    v89 = v98.m128i_i64[0];
    LODWORD(v90) = v98.m128i_i32[2];
  }
  *(_QWORD *)&v91 = 0;
  DWORD2(v91) = 0;
  *(_QWORD *)&v92 = 0;
  DWORD2(v92) = 0;
  if ( a3 == 1 && *(_DWORD *)(v83.m128i_i64[0] + 24) == 208 )
  {
    sub_377EF80((__int64 *)a1, v83.m128i_i64[0], (__int64)&v91, (__int64)&v92, v6);
  }
  else
  {
    v23 = (unsigned __int16 *)(*(_QWORD *)(v83.m128i_i64[0] + 48) + 16LL * v83.m128i_u32[2]);
    sub_2FE6CC0((__int64)&v97, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 64LL), *v23, *((_QWORD *)v23 + 1));
    if ( v97.m128i_i8[0] == 6 )
    {
      sub_375E8D0(a1, v83.m128i_u64[0], v83.m128i_i64[1], (__int64)&v91, (__int64)&v92);
    }
    else
    {
      v24 = *(_QWORD **)(a1 + 8);
      v93.m128i_i16[0] = 0;
      v94.m128i_i16[0] = 0;
      v93.m128i_i64[1] = 0;
      v94.m128i_i64[1] = 0;
      v25 = *(_QWORD *)(v83.m128i_i64[0] + 48) + 16LL * v83.m128i_u32[2];
      v26 = *(_WORD *)v25;
      v27 = *(_QWORD *)(v25 + 8);
      LOWORD(v95) = v26;
      *((_QWORD *)&v95 + 1) = v27;
      sub_33D0340((__int64)&v97, (__int64)v24, (__int64 *)&v95);
      v28 = _mm_loadu_si128(&v98);
      v93 = _mm_loadu_si128(&v97);
      v94 = v28;
      sub_3408290(
        (__int64)&v97,
        v24,
        (__int128 *)v83.m128i_i8,
        (__int64)&v85,
        (unsigned int *)&v93,
        (unsigned int *)&v94,
        v6);
      *(_QWORD *)&v91 = v97.m128i_i64[0];
      DWORD2(v91) = v97.m128i_i32[2];
      *(_QWORD *)&v92 = v98.m128i_i64[0];
      DWORD2(v92) = v98.m128i_i32[2];
    }
  }
  v29 = *(_QWORD *)(a2 + 104);
  v30 = *(_WORD *)(a2 + 96);
  v94.m128i_i64[1] = 0;
  v31 = *(_QWORD *)(a1 + 8);
  v93.m128i_i64[1] = v29;
  v93.m128i_i16[0] = v30;
  v94.m128i_i16[0] = 0;
  v82 = 0;
  v32 = *(_QWORD *)(v87 + 48) + 16LL * (unsigned int)v88;
  v33 = *(_WORD *)v32;
  v34 = *(_QWORD *)(v32 + 8);
  LOWORD(v95) = v33;
  *((_QWORD *)&v95 + 1) = v34;
  sub_33D04E0((__int64)&v97, v31, (unsigned __int16 *)&v93, (unsigned __int16 *)&v95, &v82);
  v35 = _mm_loadu_si128(&v98);
  v94 = _mm_loadu_si128(&v97);
  v73 = v35;
  v36 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 40LL);
  v37 = *(_QWORD *)(a2 + 112);
  v38 = *(_QWORD *)(v37 + 72);
  v97 = _mm_loadu_si128((const __m128i *)(v37 + 40));
  v98 = _mm_loadu_si128((const __m128i *)(v37 + 56));
  v39 = (const __m128i *)sub_2E7BD70(v36, 2u, -1, v80, (int)&v97, v38, *(_OWORD *)v37, *(_QWORD *)(v37 + 16), 1u, 0, 0);
  result = sub_33F65D0(
             *(__int64 **)(a1 + 8),
             v74,
             v10,
             (__int64)&v85,
             v87,
             v88,
             v77.m128i_i64[0],
             v77.m128i_i64[1],
             v75,
             v91,
             v94.m128i_i64[0],
             v94.m128i_i64[1],
             v39,
             (*(_WORD *)(a2 + 32) >> 7) & 7,
             (*(_BYTE *)(a2 + 33) & 4) != 0,
             (*(_BYTE *)(a2 + 33) & 8) != 0);
  if ( !v82 )
  {
    v70 = v41;
    v66 = result;
    v42 = sub_3465590(
            v6,
            *(_QWORD *)a1,
            v77.m128i_i64[0],
            v77.m128i_i64[1],
            v91,
            DWORD2(v91),
            (__int64)&v85,
            v94.m128i_u16[0],
            v94.m128i_i64[1],
            *(_QWORD *)(a1 + 8),
            (*(_BYTE *)(a2 + 33) & 8) != 0);
    BYTE4(v96) = 0;
    v43 = v66;
    v44 = v70;
    v77.m128i_i64[0] = (__int64)v42;
    v95 = 0u;
    v77.m128i_i64[1] = v45 | v77.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    LODWORD(v96) = 0;
    if ( v94.m128i_i16[0] )
    {
      if ( (unsigned __int16)(v94.m128i_i16[0] - 176) <= 0x34u )
        goto LABEL_16;
    }
    else
    {
      v46 = sub_3007100((__int64)&v94);
      v43 = v66;
      v44 = v70;
      if ( v46 )
      {
LABEL_16:
        v67 = v43;
        v71 = v44;
        v47.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v94);
        v48 = v80;
        v80 = -1;
        v97 = v47;
        v49 = -(((unsigned __int64)v47.m128i_i64[0] >> 3) | (1LL << v48))
            & (((unsigned __int64)v47.m128i_i64[0] >> 3) | (1LL << v48));
        if ( v49 )
        {
          _BitScanReverse64(&v49, v49);
          v80 = 63 - (v49 ^ 0x3F);
        }
        v50 = sub_2EAC1E0(*(_QWORD *)(a2 + 112));
        v51 = *(_QWORD *)(a2 + 112);
        v52 = v67;
        LODWORD(v96) = v50;
        v53 = v71;
LABEL_19:
        v54 = *(_QWORD *)(v51 + 72);
        *(_QWORD *)&v68 = v52;
        v55 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 40LL);
        v97 = _mm_loadu_si128((const __m128i *)(v51 + 40));
        *((_QWORD *)&v68 + 1) = v53;
        v98 = _mm_loadu_si128((const __m128i *)(v51 + 56));
        v56 = (const __m128i *)sub_2E7BD70(v55, 2u, -1, v80, (int)&v97, v54, v95, v96, 1u, 0, 0);
        v81 = sub_33F65D0(
                *(__int64 **)(a1 + 8),
                v74,
                v10,
                (__int64)&v85,
                v89,
                v90,
                v77.m128i_i64[0],
                v77.m128i_i64[1],
                v75,
                v92,
                v73.m128i_i64[0],
                v73.m128i_i64[1],
                v56,
                (*(_WORD *)(a2 + 32) >> 7) & 7,
                (*(_BYTE *)(a2 + 33) & 4) != 0,
                (*(_BYTE *)(a2 + 33) & 8) != 0);
        *((_QWORD *)&v65 + 1) = v57;
        *(_QWORD *)&v65 = v81;
        result = (__m128i *)sub_3406EB0(*(_QWORD **)(a1 + 8), 2u, (__int64)&v85, 1, 0, v58, v68, v65);
        goto LABEL_10;
      }
    }
    v51 = *(_QWORD *)(a2 + 112);
    v69 = v43;
    v72 = v44;
    v59.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v94);
    v52 = v69;
    v53 = v72;
    v97 = v59;
    v60 = *(_QWORD *)(v51 + 8) + ((unsigned __int64)(v59.m128i_i64[0] + 7) >> 3);
    v61 = *(_QWORD *)v51 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v61 )
    {
      v63 = *(_BYTE *)(v51 + 20);
      if ( (*(_QWORD *)v51 & 4) != 0 )
      {
        v62 = *(_DWORD *)(v61 + 12);
        v61 |= 4u;
      }
      else
      {
        v64 = *(_QWORD *)(v61 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v64 + 8) - 17 <= 1 )
          v64 = **(_QWORD **)(v64 + 16);
        v62 = *(_DWORD *)(v64 + 8) >> 8;
      }
    }
    else
    {
      v62 = *(_DWORD *)(v51 + 16);
      v63 = 0;
    }
    *(_QWORD *)&v95 = v61;
    *((_QWORD *)&v95 + 1) = v60;
    LODWORD(v96) = v62;
    BYTE4(v96) = v63;
    goto LABEL_19;
  }
LABEL_10:
  if ( v85 )
  {
    v78 = result;
    sub_B91220((__int64)&v85, v85);
    return v78;
  }
  return result;
}
