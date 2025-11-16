// Function: sub_3787680
// Address: 0x3787680
//
__m128i *__fastcall sub_3787680(__int64 a1, __int64 a2, int a3)
{
  __int64 v5; // rax
  __m128i v6; // xmm0
  __int128 v7; // xmm1
  __m128i v8; // xmm2
  __m128i v9; // xmm3
  __m128i v10; // xmm4
  unsigned __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // rsi
  int v14; // eax
  __int64 v15; // rsi
  unsigned __int16 *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r8
  _QWORD *v19; // rsi
  __int64 v20; // rax
  __int16 v21; // dx
  __int64 v22; // rax
  __m128i v23; // xmm2
  unsigned __int16 *v24; // rax
  _QWORD *v25; // rsi
  __int64 v26; // rax
  __int16 v27; // dx
  __int64 v28; // rax
  __m128i v29; // xmm4
  __int64 v30; // rax
  __int16 v31; // dx
  __int64 v32; // rsi
  __int64 v33; // rax
  __int16 v34; // dx
  __int64 v35; // rax
  unsigned __int16 *v36; // rax
  __int64 v37; // r14
  __int64 v38; // r15
  _QWORD *v39; // rdi
  __int64 v40; // rax
  __int64 v41; // r9
  __m128i v42; // xmm0
  const __m128i *v43; // rax
  __m128i *v44; // r14
  unsigned int v45; // edx
  __int64 v46; // r15
  unsigned __int8 *v48; // rax
  unsigned int v49; // edx
  __m128i v50; // rax
  unsigned __int8 v51; // di
  unsigned __int64 v52; // rax
  int v53; // eax
  __int64 v54; // r8
  __int64 v55; // r9
  _QWORD *v56; // rdi
  const __m128i *v57; // rax
  unsigned int v58; // edx
  __int64 v59; // r9
  __m128i v60; // rax
  unsigned __int64 v61; // rax
  unsigned __int64 v62; // rcx
  int v63; // edx
  char v64; // si
  __int64 v65; // rdi
  __int128 v66; // [rsp-40h] [rbp-1F0h]
  __int128 v67; // [rsp-20h] [rbp-1D0h]
  __int128 v68; // [rsp-10h] [rbp-1C0h]
  __int64 v69; // [rsp+8h] [rbp-1A8h]
  __m128i v70; // [rsp+10h] [rbp-1A0h]
  unsigned __int64 v71; // [rsp+28h] [rbp-188h]
  __m128i v73; // [rsp+50h] [rbp-160h]
  __int128 v75; // [rsp+60h] [rbp-150h]
  unsigned __int8 v76; // [rsp+77h] [rbp-139h]
  __m128i *v77; // [rsp+80h] [rbp-130h]
  char v78; // [rsp+AFh] [rbp-101h] BYREF
  __m128i v79; // [rsp+B0h] [rbp-100h] BYREF
  __m128i v80; // [rsp+C0h] [rbp-F0h] BYREF
  __int64 v81; // [rsp+D0h] [rbp-E0h] BYREF
  int v82; // [rsp+D8h] [rbp-D8h]
  __int64 v83; // [rsp+E0h] [rbp-D0h] BYREF
  __int64 v84; // [rsp+E8h] [rbp-C8h]
  __int64 v85; // [rsp+F0h] [rbp-C0h] BYREF
  __int64 v86; // [rsp+F8h] [rbp-B8h]
  __int128 v87; // [rsp+100h] [rbp-B0h] BYREF
  __int128 v88; // [rsp+110h] [rbp-A0h] BYREF
  __m128i v89; // [rsp+120h] [rbp-90h] BYREF
  __m128i v90; // [rsp+130h] [rbp-80h] BYREF
  __int128 v91; // [rsp+140h] [rbp-70h] BYREF
  __int64 v92; // [rsp+150h] [rbp-60h]
  __m128i v93; // [rsp+160h] [rbp-50h] BYREF
  __m128i v94; // [rsp+170h] [rbp-40h] BYREF

  v5 = *(_QWORD *)(a2 + 40);
  v6 = _mm_loadu_si128((const __m128i *)(v5 + 80));
  v7 = (__int128)_mm_loadu_si128((const __m128i *)(v5 + 120));
  v8 = _mm_loadu_si128((const __m128i *)(v5 + 160));
  v9 = _mm_loadu_si128((const __m128i *)(v5 + 200));
  v10 = _mm_loadu_si128((const __m128i *)(v5 + 40));
  v71 = *(_QWORD *)v5;
  v11 = *(_QWORD *)(v5 + 8);
  v12 = *(_QWORD *)(a2 + 112);
  v73 = v6;
  v13 = *(_QWORD *)(a2 + 80);
  LOBYTE(v12) = *(_BYTE *)(v12 + 34);
  v81 = v13;
  v76 = v12;
  v79 = v8;
  v80 = v10;
  if ( v13 )
    sub_B96E90((__int64)&v81, v13, 1);
  v14 = *(_DWORD *)(a2 + 72);
  v15 = *(_QWORD *)a1;
  LODWORD(v84) = 0;
  v82 = v14;
  LODWORD(v86) = 0;
  v83 = 0;
  v16 = (unsigned __int16 *)(*(_QWORD *)(v80.m128i_i64[0] + 48) + 16LL * v80.m128i_u32[2]);
  v17 = *(_QWORD *)(a1 + 8);
  v18 = *((_QWORD *)v16 + 1);
  v85 = 0;
  sub_2FE6CC0((__int64)&v93, v15, *(_QWORD *)(v17 + 64), *v16, v18);
  if ( v93.m128i_i8[0] == 6 )
  {
    sub_375E8D0(a1, v80.m128i_u64[0], v80.m128i_i64[1], (__int64)&v83, (__int64)&v85);
  }
  else
  {
    v90.m128i_i16[0] = 0;
    v89.m128i_i16[0] = 0;
    v19 = *(_QWORD **)(a1 + 8);
    v89.m128i_i64[1] = 0;
    v90.m128i_i64[1] = 0;
    v20 = *(_QWORD *)(v80.m128i_i64[0] + 48) + 16LL * v80.m128i_u32[2];
    v21 = *(_WORD *)v20;
    v22 = *(_QWORD *)(v20 + 8);
    LOWORD(v91) = v21;
    *((_QWORD *)&v91 + 1) = v22;
    sub_33D0340((__int64)&v93, (__int64)v19, (__int64 *)&v91);
    v23 = _mm_loadu_si128(&v94);
    v89 = _mm_loadu_si128(&v93);
    v90 = v23;
    sub_3408290(
      (__int64)&v93,
      v19,
      (__int128 *)v80.m128i_i8,
      (__int64)&v81,
      (unsigned int *)&v89,
      (unsigned int *)&v90,
      v6);
    v83 = v93.m128i_i64[0];
    LODWORD(v84) = v93.m128i_i32[2];
    v85 = v94.m128i_i64[0];
    LODWORD(v86) = v94.m128i_i32[2];
  }
  *(_QWORD *)&v87 = 0;
  DWORD2(v87) = 0;
  *(_QWORD *)&v88 = 0;
  DWORD2(v88) = 0;
  if ( a3 == 1 && *(_DWORD *)(v79.m128i_i64[0] + 24) == 208 )
  {
    sub_377EF80((__int64 *)a1, v79.m128i_i64[0], (__int64)&v87, (__int64)&v88, v6);
  }
  else
  {
    v24 = (unsigned __int16 *)(*(_QWORD *)(v79.m128i_i64[0] + 48) + 16LL * v79.m128i_u32[2]);
    sub_2FE6CC0((__int64)&v93, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 64LL), *v24, *((_QWORD *)v24 + 1));
    if ( v93.m128i_i8[0] == 6 )
    {
      sub_375E8D0(a1, v79.m128i_u64[0], v79.m128i_i64[1], (__int64)&v87, (__int64)&v88);
    }
    else
    {
      v25 = *(_QWORD **)(a1 + 8);
      v89.m128i_i16[0] = 0;
      v90.m128i_i16[0] = 0;
      v89.m128i_i64[1] = 0;
      v90.m128i_i64[1] = 0;
      v26 = *(_QWORD *)(v79.m128i_i64[0] + 48) + 16LL * v79.m128i_u32[2];
      v27 = *(_WORD *)v26;
      v28 = *(_QWORD *)(v26 + 8);
      LOWORD(v91) = v27;
      *((_QWORD *)&v91 + 1) = v28;
      sub_33D0340((__int64)&v93, (__int64)v25, (__int64 *)&v91);
      v29 = _mm_loadu_si128(&v94);
      v89 = _mm_loadu_si128(&v93);
      v90 = v29;
      sub_3408290(
        (__int64)&v93,
        v25,
        (__int128 *)v79.m128i_i8,
        (__int64)&v81,
        (unsigned int *)&v89,
        (unsigned int *)&v90,
        v6);
      *(_QWORD *)&v87 = v93.m128i_i64[0];
      DWORD2(v87) = v93.m128i_i32[2];
      *(_QWORD *)&v88 = v94.m128i_i64[0];
      DWORD2(v88) = v94.m128i_i32[2];
    }
  }
  v30 = *(_QWORD *)(a2 + 104);
  v31 = *(_WORD *)(a2 + 96);
  v90.m128i_i64[1] = 0;
  v32 = *(_QWORD *)(a1 + 8);
  v89.m128i_i64[1] = v30;
  v89.m128i_i16[0] = v31;
  v90.m128i_i16[0] = 0;
  v78 = 0;
  v33 = *(_QWORD *)(v83 + 48) + 16LL * (unsigned int)v84;
  v34 = *(_WORD *)v33;
  v35 = *(_QWORD *)(v33 + 8);
  LOWORD(v91) = v34;
  *((_QWORD *)&v91 + 1) = v35;
  sub_33D04E0((__int64)&v93, v32, (unsigned __int16 *)&v89, (unsigned __int16 *)&v91, &v78);
  v90 = _mm_loadu_si128(&v93);
  v36 = (unsigned __int16 *)(*(_QWORD *)(v80.m128i_i64[0] + 48) + 16LL * v80.m128i_u32[2]);
  v70 = _mm_loadu_si128(&v94);
  sub_3408380(
    &v93,
    *(_QWORD **)(a1 + 8),
    v9.m128i_i64[0],
    v9.m128i_i64[1],
    *v36,
    *((_QWORD *)v36 + 1),
    v6,
    (__int64)&v81);
  v37 = v93.m128i_i64[0];
  v38 = v93.m128i_u32[2];
  *(_QWORD *)&v75 = v94.m128i_i64[0];
  *((_QWORD *)&v75 + 1) = v94.m128i_u32[2];
  v39 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 40LL);
  v40 = *(_QWORD *)(a2 + 112);
  v41 = *(_QWORD *)(v40 + 72);
  v93 = _mm_loadu_si128((const __m128i *)(v40 + 40));
  v42 = _mm_loadu_si128((const __m128i *)(v40 + 56));
  v94 = v42;
  v43 = (const __m128i *)sub_2E7BD70(v39, 2u, -1, v76, (int)&v93, v41, *(_OWORD *)v40, *(_QWORD *)(v40 + 16), 1u, 0, 0);
  *((_QWORD *)&v66 + 1) = v38;
  *(_QWORD *)&v66 = v37;
  v44 = sub_33F51B0(
          *(__int64 **)(a1 + 8),
          v71,
          v11,
          (__int64)&v81,
          v83,
          v84,
          v73.m128i_i64[0],
          v73.m128i_i64[1],
          v7,
          v87,
          v66,
          v90.m128i_i64[0],
          v90.m128i_i64[1],
          v43,
          (*(_WORD *)(a2 + 32) >> 7) & 7,
          (*(_BYTE *)(a2 + 33) & 4) != 0,
          (*(_BYTE *)(a2 + 33) & 8) != 0);
  v46 = v45;
  if ( !v78 )
  {
    v48 = sub_3465590(
            v42,
            *(_QWORD *)a1,
            v73.m128i_i64[0],
            v73.m128i_i64[1],
            v87,
            DWORD2(v87),
            (__int64)&v81,
            v90.m128i_u16[0],
            v90.m128i_i64[1],
            *(_QWORD *)(a1 + 8),
            (*(_BYTE *)(a2 + 33) & 8) != 0);
    LODWORD(v92) = 0;
    v73.m128i_i64[0] = (__int64)v48;
    v91 = 0u;
    v73.m128i_i64[1] = v49 | v73.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    BYTE4(v92) = 0;
    if ( v90.m128i_i16[0] )
    {
      if ( (unsigned __int16)(v90.m128i_i16[0] - 176) <= 0x34u )
        goto LABEL_16;
    }
    else if ( sub_3007100((__int64)&v90) )
    {
LABEL_16:
      v50.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v90);
      v51 = v76;
      v76 = -1;
      v93 = v50;
      v52 = -(((unsigned __int64)v50.m128i_i64[0] >> 3) | (1LL << v51))
          & (((unsigned __int64)v50.m128i_i64[0] >> 3) | (1LL << v51));
      if ( v52 )
      {
        _BitScanReverse64(&v52, v52);
        v76 = 63 - (v52 ^ 0x3F);
      }
      v53 = sub_2EAC1E0(*(_QWORD *)(a2 + 112));
      v54 = *(_QWORD *)(a2 + 112);
      LODWORD(v92) = v53;
LABEL_19:
      v55 = *(_QWORD *)(v54 + 72);
      v56 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 40LL);
      v93 = _mm_loadu_si128((const __m128i *)(v54 + 40));
      v94 = _mm_loadu_si128((const __m128i *)(v54 + 56));
      v57 = (const __m128i *)sub_2E7BD70(v56, 2u, -1, v76, (int)&v93, v55, v91, v92, 1u, 0, 0);
      v77 = sub_33F51B0(
              *(__int64 **)(a1 + 8),
              v71,
              v11,
              (__int64)&v81,
              v85,
              v86,
              v73.m128i_i64[0],
              v73.m128i_i64[1],
              v7,
              v88,
              v75,
              v70.m128i_i64[0],
              v70.m128i_i64[1],
              v57,
              (*(_WORD *)(a2 + 32) >> 7) & 7,
              (*(_BYTE *)(a2 + 33) & 4) != 0,
              (*(_BYTE *)(a2 + 33) & 8) != 0);
      *((_QWORD *)&v68 + 1) = v58;
      *(_QWORD *)&v68 = v77;
      *((_QWORD *)&v67 + 1) = v46;
      *(_QWORD *)&v67 = v44;
      v44 = (__m128i *)sub_3406EB0(*(_QWORD **)(a1 + 8), 2u, (__int64)&v81, 1, 0, v59, v67, v68);
      goto LABEL_10;
    }
    v69 = *(_QWORD *)(a2 + 112);
    v60.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v90);
    v54 = v69;
    v93 = v60;
    v61 = *(_QWORD *)(v69 + 8) + ((unsigned __int64)(v60.m128i_i64[0] + 7) >> 3);
    v62 = *(_QWORD *)v69 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v62 )
    {
      v64 = *(_BYTE *)(v69 + 20);
      if ( (*(_QWORD *)v69 & 4) != 0 )
      {
        v63 = *(_DWORD *)(v62 + 12);
        v62 |= 4u;
      }
      else
      {
        v65 = *(_QWORD *)(v62 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v65 + 8) - 17 <= 1 )
          v65 = **(_QWORD **)(v65 + 16);
        v63 = *(_DWORD *)(v65 + 8) >> 8;
      }
    }
    else
    {
      v63 = *(_DWORD *)(v69 + 16);
      v64 = 0;
    }
    *(_QWORD *)&v91 = v62;
    *((_QWORD *)&v91 + 1) = v61;
    LODWORD(v92) = v63;
    BYTE4(v92) = v64;
    goto LABEL_19;
  }
LABEL_10:
  if ( v81 )
    sub_B91220((__int64)&v81, v81);
  return v44;
}
