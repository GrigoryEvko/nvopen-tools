// Function: sub_377FDD0
// Address: 0x377fdd0
//
void __fastcall sub_377FDD0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rsi
  __int64 v7; // rsi
  __int16 *v8; // rax
  __int16 v9; // dx
  __int64 v10; // rax
  __m128i v11; // xmm0
  __int64 v12; // rsi
  unsigned __int8 v13; // di
  unsigned __int16 v14; // dx
  __int64 v15; // rax
  __int128 v16; // xmm1
  __m128i v17; // xmm2
  __int128 v18; // xmm3
  __m128i v19; // xmm4
  __int64 v20; // r14
  __int64 v21; // r15
  __int64 v22; // rax
  __m128i v23; // xmm5
  __m128i v24; // xmm6
  bool v25; // zf
  unsigned __int16 *v26; // rax
  _QWORD *v27; // rsi
  __int64 v28; // rax
  __int16 v29; // dx
  __int64 v30; // rax
  __m128i v31; // xmm2
  __int64 v32; // r14
  __int64 v33; // r15
  _QWORD *v34; // rdi
  __int64 v35; // rax
  __int64 v36; // r9
  __m128i v37; // xmm0
  const __m128i *v38; // rax
  __m128i *v39; // rax
  int v40; // edx
  unsigned __int8 *v41; // rax
  unsigned int v42; // edx
  unsigned __int8 *v43; // rax
  unsigned int v44; // edx
  __int64 v45; // r14
  __m128i v46; // rax
  unsigned __int64 v47; // rax
  unsigned __int64 v48; // rcx
  int v49; // edx
  char v50; // si
  int v51; // eax
  __int64 v52; // r9
  _QWORD *v53; // rdi
  const __m128i *v54; // rax
  int v55; // edx
  __int64 v56; // rdi
  __int128 v57; // [rsp-30h] [rbp-200h]
  __int128 v58; // [rsp-20h] [rbp-1F0h]
  __int128 v59; // [rsp-10h] [rbp-1E0h]
  __int64 v60; // [rsp+0h] [rbp-1D0h]
  __int64 v61; // [rsp+8h] [rbp-1C8h]
  unsigned __int8 v62; // [rsp+33h] [rbp-19Dh]
  char v63; // [rsp+34h] [rbp-19Ch]
  __m128i v65; // [rsp+40h] [rbp-190h]
  __int128 v67; // [rsp+50h] [rbp-180h]
  char v69; // [rsp+CFh] [rbp-101h] BYREF
  __m128i v70; // [rsp+D0h] [rbp-100h] BYREF
  __int64 v71; // [rsp+E0h] [rbp-F0h] BYREF
  int v72; // [rsp+E8h] [rbp-E8h]
  __m128i v73; // [rsp+F0h] [rbp-E0h] BYREF
  unsigned __int16 v74; // [rsp+100h] [rbp-D0h] BYREF
  __int64 v75; // [rsp+108h] [rbp-C8h]
  __m128i v76; // [rsp+110h] [rbp-C0h] BYREF
  __int128 v77; // [rsp+120h] [rbp-B0h] BYREF
  __int128 v78; // [rsp+130h] [rbp-A0h] BYREF
  __m128i v79; // [rsp+140h] [rbp-90h] BYREF
  __m128i v80; // [rsp+150h] [rbp-80h] BYREF
  __int128 v81; // [rsp+160h] [rbp-70h] BYREF
  __int64 v82; // [rsp+170h] [rbp-60h]
  __m128i v83; // [rsp+180h] [rbp-50h] BYREF
  __m128i v84; // [rsp+190h] [rbp-40h] BYREF

  v6 = *(_QWORD *)(a2 + 80);
  v70.m128i_i16[0] = 0;
  v70.m128i_i64[1] = 0;
  v71 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v71, v6, 1);
  v7 = *(_QWORD *)(a1 + 8);
  v72 = *(_DWORD *)(a2 + 72);
  v8 = *(__int16 **)(a2 + 48);
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  LOWORD(v81) = v9;
  *((_QWORD *)&v81 + 1) = v10;
  sub_33D0340((__int64)&v83, v7, (__int64 *)&v81);
  v11 = _mm_loadu_si128(&v83);
  v12 = *(_QWORD *)(a1 + 8);
  v61 = v84.m128i_i64[0];
  v13 = *(_BYTE *)(*(_QWORD *)(a2 + 112) + 34LL);
  v14 = *(_WORD *)(a2 + 96);
  v70 = v11;
  v60 = v84.m128i_i64[1];
  v62 = v13;
  v63 = (*(_BYTE *)(a2 + 33) >> 2) & 3;
  v15 = *(_QWORD *)(a2 + 40);
  v16 = (__int128)_mm_loadu_si128((const __m128i *)v15);
  v17 = _mm_loadu_si128((const __m128i *)(v15 + 40));
  v74 = v14;
  v18 = (__int128)_mm_loadu_si128((const __m128i *)(v15 + 80));
  v19 = _mm_loadu_si128((const __m128i *)(v15 + 120));
  v76.m128i_i16[0] = 0;
  v20 = *(_QWORD *)(v15 + 160);
  v21 = *(_QWORD *)(v15 + 168);
  v76.m128i_i64[1] = 0;
  v22 = *(_QWORD *)(a2 + 104);
  v69 = 0;
  v75 = v22;
  v65 = v17;
  v73 = v19;
  sub_33D04E0((__int64)&v83, v12, &v74, (unsigned __int16 *)&v70, &v69);
  v23 = _mm_loadu_si128(&v83);
  *(_QWORD *)&v77 = 0;
  v24 = _mm_loadu_si128(&v84);
  DWORD2(v77) = 0;
  v25 = *(_DWORD *)(v73.m128i_i64[0] + 24) == 208;
  *(_QWORD *)&v78 = 0;
  DWORD2(v78) = 0;
  v76 = v23;
  if ( v25 )
  {
    sub_377EF80((__int64 *)a1, v73.m128i_i64[0], (__int64)&v77, (__int64)&v78, v11);
  }
  else
  {
    v26 = (unsigned __int16 *)(*(_QWORD *)(v73.m128i_i64[0] + 48) + 16LL * v73.m128i_u32[2]);
    sub_2FE6CC0((__int64)&v83, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 64LL), *v26, *((_QWORD *)v26 + 1));
    if ( v83.m128i_i8[0] == 6 )
    {
      sub_375E8D0(a1, v73.m128i_u64[0], v73.m128i_i64[1], (__int64)&v77, (__int64)&v78);
    }
    else
    {
      v80.m128i_i64[1] = 0;
      v27 = *(_QWORD **)(a1 + 8);
      v79.m128i_i16[0] = 0;
      v80.m128i_i16[0] = 0;
      v79.m128i_i64[1] = 0;
      v28 = *(_QWORD *)(v73.m128i_i64[0] + 48) + 16LL * v73.m128i_u32[2];
      v29 = *(_WORD *)v28;
      v30 = *(_QWORD *)(v28 + 8);
      LOWORD(v81) = v29;
      *((_QWORD *)&v81 + 1) = v30;
      sub_33D0340((__int64)&v83, (__int64)v27, (__int64 *)&v81);
      v31 = _mm_loadu_si128(&v84);
      v79 = _mm_loadu_si128(&v83);
      v80 = v31;
      sub_3408290(
        (__int64)&v83,
        v27,
        (__int128 *)v73.m128i_i8,
        (__int64)&v71,
        (unsigned int *)&v79,
        (unsigned int *)&v80,
        v11);
      *(_QWORD *)&v77 = v83.m128i_i64[0];
      DWORD2(v77) = v83.m128i_i32[2];
      *(_QWORD *)&v78 = v84.m128i_i64[0];
      DWORD2(v78) = v84.m128i_i32[2];
    }
  }
  sub_3408380(
    &v83,
    *(_QWORD **)(a1 + 8),
    v20,
    v21,
    **(unsigned __int16 **)(a2 + 48),
    *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
    v11,
    (__int64)&v71);
  v32 = v83.m128i_i64[0];
  v33 = v83.m128i_u32[2];
  *(_QWORD *)&v67 = v84.m128i_i64[0];
  *((_QWORD *)&v67 + 1) = v84.m128i_u32[2];
  v34 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 40LL);
  v35 = *(_QWORD *)(a2 + 112);
  v36 = *(_QWORD *)(v35 + 72);
  v83 = _mm_loadu_si128((const __m128i *)(v35 + 40));
  v37 = _mm_loadu_si128((const __m128i *)(v35 + 56));
  v84 = v37;
  v38 = (const __m128i *)sub_2E7BD70(v34, 1u, -1, v62, (int)&v83, v36, *(_OWORD *)v35, *(_QWORD *)(v35 + 16), 1u, 0, 0);
  *((_QWORD *)&v57 + 1) = v33;
  *(_QWORD *)&v57 = v32;
  v39 = sub_33E9660(
          *(__int64 **)(a1 + 8),
          (*(_WORD *)(a2 + 32) >> 7) & 7,
          v63,
          v70.m128i_u32[0],
          v70.m128i_i64[1],
          (__int64)&v71,
          v16,
          v65.m128i_i64[0],
          v65.m128i_i64[1],
          v18,
          v77,
          v57,
          v76.m128i_i64[0],
          v76.m128i_i64[1],
          v38,
          (*(_BYTE *)(a2 + 33) & 0x10) != 0);
  v25 = v69 == 0;
  *(_QWORD *)a3 = v39;
  *(_DWORD *)(a3 + 8) = v40;
  if ( !v25 )
  {
    *(_QWORD *)a4 = v39;
    *(_DWORD *)(a4 + 8) = *(_DWORD *)(a3 + 8);
    goto LABEL_8;
  }
  v43 = sub_3465590(
          v37,
          *(_QWORD *)a1,
          v65.m128i_i64[0],
          v65.m128i_i64[1],
          v77,
          DWORD2(v77),
          (__int64)&v71,
          v76.m128i_u16[0],
          v76.m128i_i64[1],
          *(_QWORD *)(a1 + 8),
          (*(_BYTE *)(a2 + 33) & 0x10) != 0);
  LODWORD(v82) = 0;
  v65.m128i_i64[0] = (__int64)v43;
  v81 = 0u;
  v65.m128i_i64[1] = v44 | v65.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  BYTE4(v82) = 0;
  if ( v76.m128i_i16[0] )
  {
    if ( (unsigned __int16)(v76.m128i_i16[0] - 176) > 0x34u )
    {
LABEL_15:
      v45 = *(_QWORD *)(a2 + 112);
      v46.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v76);
      v83 = v46;
      v47 = *(_QWORD *)(v45 + 8) + ((unsigned __int64)(v46.m128i_i64[0] + 7) >> 3);
      v48 = *(_QWORD *)v45 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v48 )
      {
        v50 = *(_BYTE *)(v45 + 20);
        if ( (*(_QWORD *)v45 & 4) != 0 )
        {
          v49 = *(_DWORD *)(v48 + 12);
          v48 |= 4u;
        }
        else
        {
          v56 = *(_QWORD *)(v48 + 8);
          if ( (unsigned int)*(unsigned __int8 *)(v56 + 8) - 17 <= 1 )
            v56 = **(_QWORD **)(v56 + 16);
          v49 = *(_DWORD *)(v56 + 8) >> 8;
        }
      }
      else
      {
        v49 = *(_DWORD *)(v45 + 16);
        v50 = 0;
      }
      *(_QWORD *)&v81 = v48;
      *((_QWORD *)&v81 + 1) = v47;
      LODWORD(v82) = v49;
      BYTE4(v82) = v50;
      goto LABEL_20;
    }
  }
  else if ( !sub_3007100((__int64)&v76) )
  {
    goto LABEL_15;
  }
  v51 = sub_2EAC1E0(*(_QWORD *)(a2 + 112));
  v45 = *(_QWORD *)(a2 + 112);
  LODWORD(v82) = v51;
LABEL_20:
  v52 = *(_QWORD *)(v45 + 72);
  v53 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 40LL);
  v83 = _mm_loadu_si128((const __m128i *)(v45 + 40));
  v84 = _mm_loadu_si128((const __m128i *)(v45 + 56));
  v54 = (const __m128i *)sub_2E7BD70(v53, 1u, -1, v62, (int)&v83, v52, v81, v82, 1u, 0, 0);
  *(_QWORD *)a4 = sub_33E9660(
                    *(__int64 **)(a1 + 8),
                    (*(_WORD *)(a2 + 32) >> 7) & 7,
                    v63,
                    v61,
                    v60,
                    (__int64)&v71,
                    v16,
                    v65.m128i_i64[0],
                    v65.m128i_i64[1],
                    v18,
                    v78,
                    v67,
                    v24.m128i_i64[0],
                    v24.m128i_i64[1],
                    v54,
                    (*(_BYTE *)(a2 + 33) & 0x10) != 0);
  *(_DWORD *)(a4 + 8) = v55;
LABEL_8:
  *((_QWORD *)&v59 + 1) = 1;
  *(_QWORD *)&v59 = *(_QWORD *)a4;
  *((_QWORD *)&v58 + 1) = 1;
  *(_QWORD *)&v58 = *(_QWORD *)a3;
  v41 = sub_3406EB0(*(_QWORD **)(a1 + 8), 2u, (__int64)&v71, 1, 0, *(_QWORD *)(a1 + 8), v58, v59);
  sub_3760E70(a1, a2, 1, (unsigned __int64)v41, v42 | *((_QWORD *)&v16 + 1) & 0xFFFFFFFF00000000LL);
  if ( v71 )
    sub_B91220((__int64)&v71, v71);
}
