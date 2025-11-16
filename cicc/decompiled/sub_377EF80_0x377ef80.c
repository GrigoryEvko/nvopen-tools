// Function: sub_377EF80
// Address: 0x377ef80
//
void __fastcall sub_377EF80(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5)
{
  __int64 v6; // rsi
  __int64 v7; // rsi
  __int16 *v8; // rax
  __int16 v9; // dx
  __int64 v10; // rax
  __int64 v11; // rsi
  unsigned int *v12; // rax
  unsigned __int16 *v13; // rax
  __int64 v14; // rsi
  __int64 v15; // r11
  unsigned int *v16; // r10
  int v17; // eax
  __int64 v18; // rdx
  __int16 v19; // cx
  __m128i v20; // xmm1
  unsigned __int16 *v21; // rax
  __int64 v22; // rsi
  __int64 v23; // r11
  int v24; // eax
  __int64 v25; // rcx
  __int64 v26; // rax
  __int16 v27; // dx
  __int64 v28; // rax
  __m128i v29; // xmm3
  __int64 v30; // r9
  bool v31; // zf
  __int64 v32; // rax
  __int64 v33; // r14
  __int64 v34; // r15
  int v35; // edx
  __int64 v36; // r9
  int v37; // edx
  int v38; // edx
  __int64 v39; // r9
  int v40; // edx
  __int128 v41; // [rsp-20h] [rbp-1A0h]
  __int128 v42; // [rsp-10h] [rbp-190h]
  __int128 *v43; // [rsp+0h] [rbp-180h]
  __int64 v44; // [rsp+0h] [rbp-180h]
  __int128 v45; // [rsp+0h] [rbp-180h]
  __int64 v46; // [rsp+18h] [rbp-168h]
  __int64 v47; // [rsp+20h] [rbp-160h]
  __int64 v48; // [rsp+28h] [rbp-158h]
  __int64 v49; // [rsp+30h] [rbp-150h]
  __int64 v51; // [rsp+40h] [rbp-140h]
  _QWORD *v52; // [rsp+40h] [rbp-140h]
  __int64 v53; // [rsp+40h] [rbp-140h]
  _QWORD *v54; // [rsp+40h] [rbp-140h]
  __int128 v55; // [rsp+40h] [rbp-140h]
  __int64 v57; // [rsp+A0h] [rbp-E0h] BYREF
  int v58; // [rsp+A8h] [rbp-D8h]
  __int128 v59; // [rsp+B0h] [rbp-D0h] BYREF
  __int128 v60; // [rsp+C0h] [rbp-C0h] BYREF
  __int128 v61; // [rsp+D0h] [rbp-B0h] BYREF
  __int128 v62; // [rsp+E0h] [rbp-A0h] BYREF
  __int64 v63; // [rsp+F0h] [rbp-90h] BYREF
  int v64; // [rsp+F8h] [rbp-88h]
  __m128i v65; // [rsp+100h] [rbp-80h] BYREF
  __m128i v66; // [rsp+110h] [rbp-70h] BYREF
  __int64 v67; // [rsp+120h] [rbp-60h] BYREF
  __int64 v68; // [rsp+128h] [rbp-58h]
  __m128i v69; // [rsp+130h] [rbp-50h] BYREF
  __m128i v70; // [rsp+140h] [rbp-40h] BYREF

  v6 = *(_QWORD *)(a2 + 80);
  v57 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v57, v6, 1);
  v7 = a1[1];
  v58 = *(_DWORD *)(a2 + 72);
  v8 = *(__int16 **)(a2 + 48);
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  LOWORD(v67) = v9;
  v68 = v10;
  sub_33D0340((__int64)&v69, v7, &v67);
  v11 = *a1;
  DWORD2(v59) = 0;
  v49 = v69.m128i_i64[0];
  DWORD2(v60) = 0;
  v48 = v69.m128i_i64[1];
  DWORD2(v61) = 0;
  v47 = v70.m128i_i64[0];
  DWORD2(v62) = 0;
  v46 = v70.m128i_i64[1];
  v12 = *(unsigned int **)(a2 + 40);
  *(_QWORD *)&v59 = 0;
  *(_QWORD *)&v60 = 0;
  *(_QWORD *)&v61 = 0;
  *(_QWORD *)&v62 = 0;
  v13 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v12 + 48LL) + 16LL * v12[2]);
  sub_2FE6CC0((__int64)&v69, v11, *(_QWORD *)(a1[1] + 64), *v13, *((_QWORD *)v13 + 1));
  if ( v69.m128i_i8[0] == 6 )
  {
    sub_375E8D0(
      (__int64)a1,
      **(_QWORD **)(a2 + 40),
      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
      (__int64)&v59,
      (__int64)&v60);
  }
  else
  {
    v14 = *(_QWORD *)(a2 + 80);
    v15 = a1[1];
    v63 = v14;
    if ( v14 )
    {
      v51 = v15;
      sub_B96E90((__int64)&v63, v14, 1);
      v15 = v51;
    }
    v16 = *(unsigned int **)(a2 + 40);
    v17 = *(_DWORD *)(a2 + 72);
    v66.m128i_i16[0] = 0;
    v64 = v17;
    v65.m128i_i16[0] = 0;
    v65.m128i_i64[1] = 0;
    v66.m128i_i64[1] = 0;
    v43 = (__int128 *)v16;
    v52 = (_QWORD *)v15;
    v18 = *(_QWORD *)(*(_QWORD *)v16 + 48LL) + 16LL * v16[2];
    v19 = *(_WORD *)v18;
    v68 = *(_QWORD *)(v18 + 8);
    LOWORD(v67) = v19;
    sub_33D0340((__int64)&v69, v15, &v67);
    a5 = _mm_loadu_si128(&v69);
    v20 = _mm_loadu_si128(&v70);
    v65 = a5;
    v66 = v20;
    sub_3408290((__int64)&v69, v52, v43, (__int64)&v63, (unsigned int *)&v65, (unsigned int *)&v66, a5);
    if ( v63 )
      sub_B91220((__int64)&v63, v63);
    *(_QWORD *)&v59 = v69.m128i_i64[0];
    DWORD2(v59) = v69.m128i_i32[2];
    *(_QWORD *)&v60 = v70.m128i_i64[0];
    DWORD2(v60) = v70.m128i_i32[2];
  }
  v21 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL) + 48LL)
                           + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 48LL));
  sub_2FE6CC0((__int64)&v69, *a1, *(_QWORD *)(a1[1] + 64), *v21, *((_QWORD *)v21 + 1));
  if ( v69.m128i_i8[0] == 6 )
  {
    sub_375E8D0(
      (__int64)a1,
      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
      (__int64)&v61,
      (__int64)&v62);
    v32 = *(_QWORD *)(a2 + 40);
    if ( *(_DWORD *)(a2 + 24) != 208 )
    {
LABEL_15:
      sub_3777990(&v69, a1, *(_QWORD *)(v32 + 120), *(_QWORD *)(v32 + 128), a5);
      *(_QWORD *)&v55 = v69.m128i_i64[0];
      *(_QWORD *)&v45 = v70.m128i_i64[0];
      *((_QWORD *)&v55 + 1) = v69.m128i_u32[2];
      *((_QWORD *)&v45 + 1) = v70.m128i_u32[2];
      sub_3408380(
        &v69,
        (_QWORD *)a1[1],
        *(_QWORD *)(*(_QWORD *)(a2 + 40) + 160LL),
        *(_QWORD *)(*(_QWORD *)(a2 + 40) + 168LL),
        **(unsigned __int16 **)(a2 + 48),
        *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
        a5,
        (__int64)&v57);
      v33 = v70.m128i_i64[0];
      *((_QWORD *)&v41 + 1) = v69.m128i_u32[2];
      *(_QWORD *)&v41 = v69.m128i_i64[0];
      v34 = v70.m128i_u32[2];
      *(_QWORD *)a3 = sub_33FC1D0(
                        (_QWORD *)a1[1],
                        *(unsigned int *)(a2 + 24),
                        (__int64)&v57,
                        v49,
                        v48,
                        a1[1],
                        v59,
                        v61,
                        *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL),
                        v55,
                        v41);
      *(_DWORD *)(a3 + 8) = v35;
      *((_QWORD *)&v42 + 1) = v34;
      *(_QWORD *)&v42 = v33;
      *(_QWORD *)a4 = sub_33FC1D0(
                        (_QWORD *)a1[1],
                        *(unsigned int *)(a2 + 24),
                        (__int64)&v57,
                        v47,
                        v46,
                        v36,
                        v60,
                        v62,
                        *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL),
                        v45,
                        v42);
      *(_DWORD *)(a4 + 8) = v37;
      goto LABEL_16;
    }
  }
  else
  {
    v22 = *(_QWORD *)(a2 + 80);
    v23 = a1[1];
    v63 = v22;
    if ( v22 )
    {
      v53 = v23;
      sub_B96E90((__int64)&v63, v22, 1);
      v23 = v53;
    }
    v24 = *(_DWORD *)(a2 + 72);
    v25 = *(_QWORD *)(a2 + 40);
    v65.m128i_i64[1] = 0;
    v64 = v24;
    v65.m128i_i16[0] = 0;
    v66.m128i_i16[0] = 0;
    v66.m128i_i64[1] = 0;
    v44 = v25;
    v54 = (_QWORD *)v23;
    v26 = *(_QWORD *)(*(_QWORD *)(v25 + 40) + 48LL) + 16LL * *(unsigned int *)(v25 + 48);
    v27 = *(_WORD *)v26;
    v28 = *(_QWORD *)(v26 + 8);
    LOWORD(v67) = v27;
    v68 = v28;
    sub_33D0340((__int64)&v69, v23, &v67);
    v29 = _mm_loadu_si128(&v70);
    v65 = _mm_loadu_si128(&v69);
    v66 = v29;
    sub_3408290(
      (__int64)&v69,
      v54,
      (__int128 *)(v44 + 40),
      (__int64)&v63,
      (unsigned int *)&v65,
      (unsigned int *)&v66,
      a5);
    if ( v63 )
      sub_B91220((__int64)&v63, v63);
    v31 = *(_DWORD *)(a2 + 24) == 208;
    *(_QWORD *)&v61 = v69.m128i_i64[0];
    DWORD2(v61) = v69.m128i_i32[2];
    *(_QWORD *)&v62 = v70.m128i_i64[0];
    DWORD2(v62) = v70.m128i_i32[2];
    v32 = *(_QWORD *)(a2 + 40);
    if ( !v31 )
      goto LABEL_15;
  }
  *(_QWORD *)a3 = sub_340F900((_QWORD *)a1[1], 0xD0u, (__int64)&v57, v49, v48, v30, v59, v61, *(_OWORD *)(v32 + 80));
  *(_DWORD *)(a3 + 8) = v38;
  *(_QWORD *)a4 = sub_340F900(
                    (_QWORD *)a1[1],
                    *(_DWORD *)(a2 + 24),
                    (__int64)&v57,
                    v47,
                    v46,
                    v39,
                    v60,
                    v62,
                    *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL));
  *(_DWORD *)(a4 + 8) = v40;
LABEL_16:
  if ( v57 )
    sub_B91220((__int64)&v57, v57);
}
