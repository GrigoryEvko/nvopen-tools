// Function: sub_381D320
// Address: 0x381d320
//
void __fastcall sub_381D320(__int64 a1, unsigned __int64 a2, _DWORD *a3, __int64 a4)
{
  unsigned int v4; // r15d
  __int64 *v7; // rax
  __int64 v8; // rsi
  __m128i v9; // xmm0
  __int64 v10; // rdi
  __int64 v11; // rbx
  int v12; // eax
  __int64 v13; // rdx
  int v14; // eax
  __int64 v15; // rbx
  __int64 v16; // r12
  __int64 v17; // r14
  __int64 v18; // r9
  __int64 v19; // rdx
  unsigned int *v20; // rax
  _QWORD *v21; // rdi
  __int64 v22; // rsi
  unsigned int *v23; // r15
  __m128i v24; // xmm3
  __m128i v25; // xmm4
  __m128i v26; // xmm5
  __int64 v27; // rdx
  __int64 v28; // r9
  unsigned __int8 *v29; // rax
  int v30; // edx
  __int64 v31; // r9
  unsigned __int8 *v32; // rax
  __int64 v33; // r8
  unsigned __int64 v34; // rcx
  int v35; // edx
  __int64 (__fastcall *v36)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v37; // rdx
  _QWORD *v38; // r15
  unsigned int v39; // ebx
  __int128 v40; // rax
  __int64 v41; // r9
  unsigned int v42; // edx
  __int64 v43; // rdx
  __int64 v44; // r9
  _QWORD *v45; // r15
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rbx
  __int128 v49; // rax
  __int64 v50; // rax
  unsigned int v51; // edx
  unsigned __int16 *v52; // rax
  __int128 v53; // rax
  _QWORD *v54; // r15
  __int64 v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rbx
  __int128 v58; // rax
  __int64 v59; // r9
  __int128 v60; // [rsp-20h] [rbp-190h]
  __int128 v61; // [rsp-10h] [rbp-180h]
  __int64 v62; // [rsp+0h] [rbp-170h]
  unsigned int v63; // [rsp+8h] [rbp-168h]
  unsigned int v64; // [rsp+Ch] [rbp-164h]
  unsigned __int64 v65; // [rsp+10h] [rbp-160h]
  unsigned int v68; // [rsp+30h] [rbp-140h]
  __int128 v69; // [rsp+30h] [rbp-140h]
  __m128i v70; // [rsp+40h] [rbp-130h]
  __int128 v71; // [rsp+50h] [rbp-120h]
  __int128 v72; // [rsp+50h] [rbp-120h]
  unsigned int v73; // [rsp+60h] [rbp-110h]
  unsigned __int16 v74; // [rsp+70h] [rbp-100h]
  __int64 v75; // [rsp+70h] [rbp-100h]
  __int64 v76; // [rsp+70h] [rbp-100h]
  unsigned int v77; // [rsp+70h] [rbp-100h]
  __int128 v78; // [rsp+70h] [rbp-100h]
  __int64 v79; // [rsp+A0h] [rbp-D0h] BYREF
  int v80; // [rsp+A8h] [rbp-C8h]
  __m128i v81; // [rsp+B0h] [rbp-C0h] BYREF
  __m128i v82; // [rsp+C0h] [rbp-B0h] BYREF
  __m128i v83; // [rsp+D0h] [rbp-A0h] BYREF
  __m128i v84; // [rsp+E0h] [rbp-90h] BYREF
  _OWORD v85[2]; // [rsp+F0h] [rbp-80h] BYREF
  __m128i v86; // [rsp+110h] [rbp-60h] BYREF
  __m128i v87; // [rsp+120h] [rbp-50h]
  unsigned __int8 *v88; // [rsp+130h] [rbp-40h]
  int v89; // [rsp+138h] [rbp-38h]

  v7 = *(__int64 **)(a2 + 40);
  v8 = *(_QWORD *)(a2 + 80);
  v9 = _mm_loadu_si128((const __m128i *)v7);
  v10 = *v7;
  v79 = v8;
  v11 = *((unsigned int *)v7 + 2);
  v70 = _mm_loadu_si128((const __m128i *)(v7 + 5));
  if ( v8 )
    sub_B96E90((__int64)&v79, v8, 1);
  v80 = *(_DWORD *)(a2 + 72);
  v12 = *(_DWORD *)(a2 + 24);
  if ( v12 == 77 )
  {
    v64 = 12;
    v63 = 56;
    v68 = 72;
  }
  else
  {
    if ( v12 != 79 )
LABEL_29:
      BUG();
    v64 = 10;
    v63 = 57;
    v68 = 73;
  }
  v62 = 16 * v11;
  v65 = a2;
  v13 = 16 * v11 + *(_QWORD *)(v10 + 48);
  LOWORD(v14) = *(_WORD *)v13;
  v15 = *(_QWORD *)a1;
  v16 = *(_QWORD *)(v13 + 8);
  v17 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 64LL);
  while ( 1 )
  {
    LOWORD(v4) = v14;
    v74 = v14;
    sub_2FE6CC0((__int64)&v86, v15, v17, v4, v16);
    if ( !v86.m128i_i8[0] )
      break;
    if ( v86.m128i_i8[0] != 2 )
      goto LABEL_29;
    v36 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v15 + 592LL);
    if ( v36 == sub_2D56A50 )
    {
      sub_2FE6CC0((__int64)&v86, v15, v17, v4, v16);
      LOWORD(v14) = v86.m128i_i16[4];
      v16 = v87.m128i_i64[0];
    }
    else
    {
      v14 = v36(v15, v17, v4, v16);
      HIWORD(v4) = HIWORD(v14);
      v16 = v43;
    }
  }
  v19 = 1;
  if ( v74 == 1 || v74 && (v19 = v74, *(_QWORD *)(v15 + 8LL * v74 + 112)) )
  {
    if ( (*(_BYTE *)(v68 + 500 * v19 + v15 + 6414) & 0xFB) == 0 )
    {
      v81.m128i_i64[0] = 0;
      v81.m128i_i32[2] = 0;
      v82.m128i_i64[0] = 0;
      v82.m128i_i32[2] = 0;
      v83.m128i_i64[0] = 0;
      v83.m128i_i32[2] = 0;
      v84.m128i_i64[0] = 0;
      v84.m128i_i32[2] = 0;
      sub_375E510(a1, v9.m128i_u64[0], v9.m128i_i64[1], (__int64)&v81, (__int64)&v82);
      sub_375E510(a1, v70.m128i_u64[0], v70.m128i_i64[1], (__int64)&v83, (__int64)&v84);
      v20 = (unsigned int *)sub_33E5110(
                              *(__int64 **)(a1 + 8),
                              *(unsigned __int16 *)(*(_QWORD *)(v81.m128i_i64[0] + 48) + 16LL * v81.m128i_u32[2]),
                              *(_QWORD *)(*(_QWORD *)(v81.m128i_i64[0] + 48) + 16LL * v81.m128i_u32[2] + 8),
                              *(unsigned __int16 *)(*(_QWORD *)(v65 + 48) + 16LL),
                              *(_QWORD *)(*(_QWORD *)(v65 + 48) + 24LL));
      v21 = *(_QWORD **)(a1 + 8);
      v22 = *(unsigned int *)(v65 + 24);
      v88 = 0;
      v23 = v20;
      *((_QWORD *)&v61 + 1) = 2;
      *(_QWORD *)&v61 = v85;
      v24 = _mm_loadu_si128(&v83);
      v25 = _mm_loadu_si128(&v82);
      v26 = _mm_loadu_si128(&v84);
      v85[0] = _mm_loadu_si128(&v81);
      v75 = v27;
      v85[1] = v24;
      v86 = v25;
      v87 = v26;
      v89 = 0;
      v29 = sub_3411630(v21, v22, (__int64)&v79, v20, v27, v28, v61);
      v89 = 1;
      *(_QWORD *)a3 = v29;
      v88 = v29;
      a3[2] = v30;
      *((_QWORD *)&v60 + 1) = 3;
      *(_QWORD *)&v60 = &v86;
      v32 = sub_3411630(*(_QWORD **)(a1 + 8), v68, (__int64)&v79, v23, v75, v31, v60);
      v33 = 1;
      v34 = (unsigned __int64)v32;
      *(_QWORD *)a4 = v32;
      *(_DWORD *)(a4 + 8) = v35;
      goto LABEL_11;
    }
  }
  *(_QWORD *)&v69 = sub_3406EB0(
                      *(_QWORD **)(a1 + 8),
                      v63,
                      (__int64)&v79,
                      *(unsigned __int16 *)(*(_QWORD *)(v10 + 48) + v62),
                      *(_QWORD *)(*(_QWORD *)(v10 + 48) + v62 + 8),
                      v18,
                      *(_OWORD *)&v9,
                      *(_OWORD *)&v70);
  *((_QWORD *)&v69 + 1) = v37;
  sub_375BC20((__int64 *)a1, v69, v37, (__int64)a3, a4, v9);
  if ( *(_DWORD *)(v65 + 24) != 77 )
    goto LABEL_20;
  if ( sub_33CF4D0(v70.m128i_i64[0]) )
  {
    v52 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)a3 + 48LL) + 16LL * (unsigned int)a3[2]);
    *(_QWORD *)&v53 = sub_3406EB0(
                        *(_QWORD **)(a1 + 8),
                        0xBBu,
                        (__int64)&v79,
                        *v52,
                        *((_QWORD *)v52 + 1),
                        v44,
                        *(_OWORD *)a3,
                        *(_OWORD *)a4);
    v54 = *(_QWORD **)(a1 + 8);
    v78 = v53;
    *(_QWORD *)&v53 = *(_QWORD *)(*(_QWORD *)a3 + 48LL) + 16LL * (unsigned int)a3[2];
    *(_QWORD *)&v72 = sub_3400BD0(
                        (__int64)v54,
                        0,
                        (__int64)&v79,
                        *(unsigned __int16 *)v53,
                        *(_QWORD *)(v53 + 8),
                        0,
                        v9,
                        0);
    v55 = *(_QWORD *)(v65 + 48);
    *((_QWORD *)&v72 + 1) = v56;
    v57 = *(_QWORD *)(v55 + 24);
    v73 = *(unsigned __int16 *)(v55 + 16);
    *(_QWORD *)&v58 = sub_33ED040(v54, 0x11u);
    v50 = sub_340F900(v54, 0xD0u, (__int64)&v79, v73, v57, v59, v78, v72, v58);
  }
  else
  {
    if ( *(_DWORD *)(v65 + 24) != 77 || !sub_33CF460(v70.m128i_i64[0]) )
    {
LABEL_20:
      v38 = *(_QWORD **)(a1 + 8);
      v39 = *(unsigned __int16 *)(*(_QWORD *)(v65 + 48) + 16LL);
      v76 = *(_QWORD *)(*(_QWORD *)(v65 + 48) + 24LL);
      *(_QWORD *)&v40 = sub_33ED040(v38, v64);
      v34 = sub_340F900(v38, 0xD0u, (__int64)&v79, v39, v76, v41, v69, *(_OWORD *)&v9, v40);
      v33 = v42;
      goto LABEL_11;
    }
    v45 = *(_QWORD **)(a1 + 8);
    *(_QWORD *)&v71 = sub_3400BD0(
                        (__int64)v45,
                        0,
                        (__int64)&v79,
                        *(unsigned __int16 *)(*(_QWORD *)(v10 + 48) + v62),
                        *(_QWORD *)(*(_QWORD *)(v10 + 48) + v62 + 8),
                        0,
                        v9,
                        0);
    v46 = *(_QWORD *)(v65 + 48);
    *((_QWORD *)&v71 + 1) = v47;
    v48 = *(_QWORD *)(v46 + 24);
    v77 = *(unsigned __int16 *)(v46 + 16);
    *(_QWORD *)&v49 = sub_33ED040(v45, 0x16u);
    v50 = sub_340F900(v45, 0xD0u, (__int64)&v79, v77, v48, *((__int64 *)&v71 + 1), *(_OWORD *)&v9, v71, v49);
  }
  v34 = v50;
  v33 = v51;
LABEL_11:
  sub_3760E70(a1, v65, 1, v34, v33);
  if ( v79 )
    sub_B91220((__int64)&v79, v79);
}
