// Function: sub_377C060
// Address: 0x377c060
//
void __fastcall sub_377C060(__int64 *a1, unsigned __int64 a2, int a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rsi
  __int64 v8; // rsi
  __int16 *v9; // rax
  __int16 v10; // cx
  __int64 v11; // rdx
  __m128i v12; // kr00_16
  __m128i v13; // kr10_16
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rsi
  _QWORD *v18; // r14
  unsigned int *v19; // r11
  int v20; // eax
  __int64 v21; // rax
  __int16 v22; // dx
  __int64 v23; // rax
  __m128i v24; // xmm0
  __m128i v25; // xmm1
  __int64 v26; // rsi
  _QWORD *v27; // r14
  __int64 v28; // rcx
  int v29; // eax
  __int64 v30; // rax
  __int16 v31; // dx
  __int64 v32; // rax
  __m128i v33; // xmm3
  unsigned int v34; // r14d
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rbx
  __int64 v40; // r9
  __int64 v41; // rdi
  __int64 v42; // rbx
  unsigned __int16 *v43; // rdx
  __int64 v44; // r14
  __int64 v45; // r9
  unsigned __int8 *v46; // rax
  __int64 v47; // rdx
  __int128 v48; // [rsp-20h] [rbp-1A0h]
  __int128 v49; // [rsp-10h] [rbp-190h]
  __int128 *v50; // [rsp+0h] [rbp-180h]
  __int64 v51; // [rsp+8h] [rbp-178h]
  __int64 v52; // [rsp+20h] [rbp-160h]
  __int64 v53; // [rsp+28h] [rbp-158h]
  __int64 v54; // [rsp+30h] [rbp-150h]
  __int64 v55; // [rsp+38h] [rbp-148h]
  unsigned int *v56; // [rsp+38h] [rbp-148h]
  unsigned int *v57; // [rsp+50h] [rbp-130h]
  __int64 v58; // [rsp+58h] [rbp-128h]
  unsigned __int8 *v59; // [rsp+58h] [rbp-128h]
  unsigned __int8 *v62; // [rsp+68h] [rbp-118h]
  unsigned __int16 v64; // [rsp+74h] [rbp-10Ch]
  __int64 v65; // [rsp+80h] [rbp-100h] BYREF
  int v66; // [rsp+88h] [rbp-F8h]
  __int64 v67; // [rsp+90h] [rbp-F0h] BYREF
  __int64 v68; // [rsp+98h] [rbp-E8h]
  __int64 v69[2]; // [rsp+A0h] [rbp-E0h] BYREF
  __int128 v70; // [rsp+B0h] [rbp-D0h] BYREF
  __int128 v71; // [rsp+C0h] [rbp-C0h] BYREF
  __int128 v72; // [rsp+D0h] [rbp-B0h] BYREF
  __int128 v73; // [rsp+E0h] [rbp-A0h] BYREF
  __int64 v74; // [rsp+F0h] [rbp-90h] BYREF
  int v75; // [rsp+F8h] [rbp-88h]
  __m128i v76; // [rsp+100h] [rbp-80h] BYREF
  __m128i v77; // [rsp+110h] [rbp-70h] BYREF
  __int64 v78; // [rsp+120h] [rbp-60h] BYREF
  __int64 v79; // [rsp+128h] [rbp-58h]
  __m128i v80; // [rsp+130h] [rbp-50h] BYREF
  __m128i v81; // [rsp+140h] [rbp-40h] BYREF

  v7 = *(_QWORD *)(a2 + 80);
  v65 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v65, v7, 1);
  v8 = a1[1];
  v66 = *(_DWORD *)(a2 + 72);
  v9 = *(__int16 **)(a2 + 48);
  v10 = *v9;
  v68 = *((_QWORD *)v9 + 1);
  v11 = *((_QWORD *)v9 + 3);
  LOWORD(v9) = v9[8];
  LOWORD(v67) = v10;
  v69[1] = v11;
  LOWORD(v69[0]) = (_WORD)v9;
  sub_33D0340((__int64)&v80, v8, &v67);
  v12 = v80;
  v13 = v81;
  sub_33D0340((__int64)&v80, a1[1], v69);
  v14 = *a1;
  *(_QWORD *)&v70 = 0;
  v55 = v80.m128i_i64[0];
  DWORD2(v70) = 0;
  v54 = v80.m128i_i64[1];
  *(_QWORD *)&v71 = 0;
  v53 = v81.m128i_i64[0];
  DWORD2(v71) = 0;
  v52 = v81.m128i_i64[1];
  v15 = a1[1];
  *(_QWORD *)&v72 = 0;
  DWORD2(v72) = 0;
  v16 = *(_QWORD *)(v15 + 64);
  *(_QWORD *)&v73 = 0;
  DWORD2(v73) = 0;
  sub_2FE6CC0((__int64)&v80, v14, v16, (unsigned __int16)v67, v68);
  if ( v80.m128i_i8[0] == 6 )
  {
    sub_375E8D0(
      (__int64)a1,
      **(_QWORD **)(a2 + 40),
      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
      (__int64)&v70,
      (__int64)&v71);
    sub_375E8D0(
      (__int64)a1,
      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
      (__int64)&v72,
      (__int64)&v73);
  }
  else
  {
    v17 = *(_QWORD *)(a2 + 80);
    v18 = (_QWORD *)a1[1];
    v74 = v17;
    if ( v17 )
      sub_B96E90((__int64)&v74, v17, 1);
    v19 = *(unsigned int **)(a2 + 40);
    v20 = *(_DWORD *)(a2 + 72);
    v76.m128i_i16[0] = 0;
    v77.m128i_i16[0] = 0;
    v75 = v20;
    v76.m128i_i64[1] = 0;
    v77.m128i_i64[1] = 0;
    v50 = (__int128 *)v19;
    v21 = *(_QWORD *)(*(_QWORD *)v19 + 48LL) + 16LL * v19[2];
    v22 = *(_WORD *)v21;
    v23 = *(_QWORD *)(v21 + 8);
    LOWORD(v78) = v22;
    v79 = v23;
    sub_33D0340((__int64)&v80, (__int64)v18, &v78);
    v24 = _mm_loadu_si128(&v80);
    v25 = _mm_loadu_si128(&v81);
    v76 = v24;
    v77 = v25;
    sub_3408290((__int64)&v80, v18, v50, (__int64)&v74, (unsigned int *)&v76, (unsigned int *)&v77, v24);
    if ( v74 )
      sub_B91220((__int64)&v74, v74);
    v26 = *(_QWORD *)(a2 + 80);
    v27 = (_QWORD *)a1[1];
    *(_QWORD *)&v70 = v80.m128i_i64[0];
    v74 = v26;
    DWORD2(v70) = v80.m128i_i32[2];
    *(_QWORD *)&v71 = v81.m128i_i64[0];
    DWORD2(v71) = v81.m128i_i32[2];
    if ( v26 )
      sub_B96E90((__int64)&v74, v26, 1);
    v28 = *(_QWORD *)(a2 + 40);
    v29 = *(_DWORD *)(a2 + 72);
    v76.m128i_i16[0] = 0;
    v77.m128i_i16[0] = 0;
    v75 = v29;
    v76.m128i_i64[1] = 0;
    v77.m128i_i64[1] = 0;
    v51 = v28;
    v30 = *(_QWORD *)(*(_QWORD *)(v28 + 40) + 48LL) + 16LL * *(unsigned int *)(v28 + 48);
    v31 = *(_WORD *)v30;
    v32 = *(_QWORD *)(v30 + 8);
    LOWORD(v78) = v31;
    v79 = v32;
    sub_33D0340((__int64)&v80, (__int64)v27, &v78);
    v33 = _mm_loadu_si128(&v81);
    v76 = _mm_loadu_si128(&v80);
    v77 = v33;
    sub_3408290(
      (__int64)&v80,
      v27,
      (__int128 *)(v51 + 40),
      (__int64)&v74,
      (unsigned int *)&v76,
      (unsigned int *)&v77,
      v24);
    if ( v74 )
      sub_B91220((__int64)&v74, v74);
    *(_QWORD *)&v72 = v80.m128i_i64[0];
    DWORD2(v72) = v80.m128i_i32[2];
    *(_QWORD *)&v73 = v81.m128i_i64[0];
    DWORD2(v73) = v81.m128i_i32[2];
  }
  v34 = *(_DWORD *)(a2 + 24);
  v35 = sub_33E5110((__int64 *)a1[1], v12.m128i_i64[0], v12.m128i_i64[1], v55, v54);
  v58 = v36;
  v56 = (unsigned int *)v35;
  v37 = sub_33E5110((__int64 *)a1[1], v13.m128i_i64[0], v13.m128i_i64[1], v53, v52);
  v39 = v38;
  v57 = (unsigned int *)v37;
  v59 = sub_3411F20((_QWORD *)a1[1], v34, (__int64)&v65, v56, v58, (__int64)v56, v70, v72);
  v41 = a4;
  v62 = sub_3411F20((_QWORD *)a1[1], v34, (__int64)&v65, v57, v39, v40, v71, v73);
  *((_DWORD *)v59 + 7) = *(_DWORD *)(a2 + 28);
  *((_DWORD *)v62 + 7) = *(_DWORD *)(a2 + 28);
  *(_QWORD *)v41 = v59;
  *(_DWORD *)(v41 + 8) = a3;
  *(_QWORD *)a5 = v62;
  *(_DWORD *)(a5 + 8) = a3;
  v42 = (unsigned int)(1 - a3);
  v43 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16 * v42);
  v44 = *((_QWORD *)v43 + 1);
  v64 = *v43;
  sub_2FE6CC0((__int64)&v80, *a1, *(_QWORD *)(a1[1] + 64), *v43, v44);
  if ( v80.m128i_i8[0] == 6 )
  {
    sub_3760810((__int64)a1, a2, v42, (unsigned __int64)v59, v42, v45, (unsigned __int64)v62, v42);
  }
  else
  {
    *((_QWORD *)&v49 + 1) = v42;
    *(_QWORD *)&v49 = v62;
    *((_QWORD *)&v48 + 1) = v42;
    *(_QWORD *)&v48 = v59;
    v46 = sub_3406EB0((_QWORD *)a1[1], 0x9Fu, (__int64)&v65, v64, v44, a1[1], v48, v49);
    sub_3760E70((__int64)a1, a2, v42, (unsigned __int64)v46, v47);
  }
  if ( v65 )
    sub_B91220((__int64)&v65, v65);
}
