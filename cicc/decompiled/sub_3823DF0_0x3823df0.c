// Function: sub_3823DF0
// Address: 0x3823df0
//
void __fastcall sub_3823DF0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r13d
  __int64 *v6; // rax
  __int64 v7; // rsi
  __m128i v8; // xmm0
  __int64 v9; // rcx
  __int64 v10; // r15
  __int64 v11; // r14
  __int64 v12; // rdx
  unsigned __int16 v13; // r12
  __int64 v14; // rbx
  __int64 v15; // r15
  __int64 v16; // r14
  __int64 v17; // r9
  __int64 v18; // rdx
  __int128 v19; // rax
  __int64 v20; // rcx
  unsigned __int16 *v21; // r15
  __int64 v22; // r14
  unsigned int v23; // r13d
  __int64 v24; // r9
  __int128 v25; // rax
  __int64 v26; // r9
  __int128 v27; // kr00_16
  __int128 v28; // rax
  __int64 v29; // r9
  __int64 v30; // rax
  unsigned int v31; // edx
  unsigned __int8 *v32; // r14
  __int64 v33; // rdx
  __int64 v34; // r15
  __int128 v35; // rax
  unsigned int v36; // edx
  unsigned __int64 v37; // rcx
  __int64 (__fastcall *v38)(__int64, __int64, unsigned int, __int64); // rax
  unsigned int *v39; // rax
  _QWORD *v40; // rdi
  unsigned int *v41; // r15
  __int64 v42; // rdx
  __int64 v43; // r14
  __m128i v44; // xmm2
  __int64 v45; // r9
  unsigned __int8 *v46; // rax
  __m128i v47; // xmm5
  __m128i v48; // xmm4
  int v49; // edx
  _QWORD *v50; // rdi
  __int64 v51; // r9
  unsigned __int8 *v52; // rax
  int v53; // edx
  __int64 v54; // rdx
  unsigned int v55; // edx
  __int128 v56; // [rsp-30h] [rbp-190h]
  __int128 v57; // [rsp-20h] [rbp-180h]
  __int128 v58; // [rsp-10h] [rbp-170h]
  __int64 v59; // [rsp+8h] [rbp-158h]
  unsigned int v60; // [rsp+14h] [rbp-14Ch]
  unsigned __int64 v61; // [rsp+18h] [rbp-148h]
  __int128 v65; // [rsp+30h] [rbp-130h]
  __int64 v66; // [rsp+30h] [rbp-130h]
  __m128i v67; // [rsp+40h] [rbp-120h]
  unsigned int v68; // [rsp+40h] [rbp-120h]
  __int128 v69; // [rsp+50h] [rbp-110h]
  int v70; // [rsp+60h] [rbp-100h]
  _QWORD *v71; // [rsp+60h] [rbp-100h]
  __int64 v72; // [rsp+68h] [rbp-F8h]
  __int128 v73; // [rsp+70h] [rbp-F0h]
  unsigned __int16 v74; // [rsp+70h] [rbp-F0h]
  __int64 v75; // [rsp+B0h] [rbp-B0h] BYREF
  int v76; // [rsp+B8h] [rbp-A8h]
  __m128i v77; // [rsp+C0h] [rbp-A0h] BYREF
  __m128i v78; // [rsp+D0h] [rbp-90h] BYREF
  __m128i v79; // [rsp+E0h] [rbp-80h] BYREF
  __m128i v80; // [rsp+F0h] [rbp-70h] BYREF
  __m128i v81; // [rsp+100h] [rbp-60h] BYREF
  __m128i v82; // [rsp+110h] [rbp-50h]
  unsigned __int8 *v83; // [rsp+120h] [rbp-40h]
  int v84; // [rsp+128h] [rbp-38h]

  v6 = *(__int64 **)(a2 + 40);
  v7 = *(_QWORD *)(a2 + 80);
  v8 = _mm_loadu_si128((const __m128i *)v6);
  v9 = *v6;
  v10 = *((unsigned int *)v6 + 2);
  v75 = v7;
  v72 = v9;
  v67 = _mm_loadu_si128((const __m128i *)(v6 + 5));
  if ( v7 )
    sub_B96E90((__int64)&v75, v7, 1);
  v11 = *(_QWORD *)a1;
  v76 = *(_DWORD *)(a2 + 72);
  v61 = a2;
  v70 = *(_DWORD *)(a2 + 24);
  v59 = 16 * v10;
  v60 = (v70 != 76) + 74;
  v12 = *(_QWORD *)(v72 + 48) + 16 * v10;
  v13 = *(_WORD *)v12;
  v14 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 64LL);
  v15 = v11;
  v16 = *(_QWORD *)(v12 + 8);
  while ( 1 )
  {
    LOWORD(v4) = v13;
    sub_2FE6CC0((__int64)&v81, v15, v14, v4, v16);
    if ( !v81.m128i_i8[0] )
      break;
    if ( v81.m128i_i8[0] != 2 )
      BUG();
    v38 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v15 + 592LL);
    if ( v38 == sub_2D56A50 )
    {
      sub_2FE6CC0((__int64)&v81, v15, v14, v4, v16);
      v13 = v81.m128i_u16[4];
      v16 = v82.m128i_i64[0];
    }
    else
    {
      v4 = v38(v15, v14, v4, v16);
      v13 = v4;
      v16 = v54;
    }
  }
  v18 = 1;
  if ( (v13 == 1 || v13 && (v18 = v13, *(_QWORD *)(v15 + 8LL * v13 + 112)))
    && (v17 = v15 + 500 * v18, (*(_BYTE *)(v60 + v17 + 6414) & 0xFB) == 0) )
  {
    v77.m128i_i64[0] = 0;
    v77.m128i_i32[2] = 0;
    v78.m128i_i64[0] = 0;
    v78.m128i_i32[2] = 0;
    v79.m128i_i64[0] = 0;
    v79.m128i_i32[2] = 0;
    v80.m128i_i64[0] = 0;
    v80.m128i_i32[2] = 0;
    sub_375E510(a1, v8.m128i_u64[0], v8.m128i_i64[1], (__int64)&v77, (__int64)&v78);
    sub_375E510(a1, v67.m128i_u64[0], v67.m128i_i64[1], (__int64)&v79, (__int64)&v80);
    v39 = (unsigned int *)sub_33E5110(
                            *(__int64 **)(a1 + 8),
                            *(unsigned __int16 *)(*(_QWORD *)(v77.m128i_i64[0] + 48) + 16LL * v77.m128i_u32[2]),
                            *(_QWORD *)(*(_QWORD *)(v77.m128i_i64[0] + 48) + 16LL * v77.m128i_u32[2] + 8),
                            *(unsigned __int16 *)(*(_QWORD *)(v61 + 48) + 16LL),
                            *(_QWORD *)(*(_QWORD *)(v61 + 48) + 24LL));
    v40 = *(_QWORD **)(a1 + 8);
    v41 = v39;
    v43 = v42;
    *((_QWORD *)&v58 + 1) = 2;
    *(_QWORD *)&v58 = &v81;
    v44 = _mm_loadu_si128(&v77);
    v82 = _mm_loadu_si128(&v79);
    v81 = v44;
    v46 = sub_3411630(v40, 2 * (unsigned int)(v70 != 76) + 77, (__int64)&v75, v39, v42, v45, v58);
    v47 = _mm_loadu_si128(&v80);
    v48 = _mm_loadu_si128(&v78);
    *(_QWORD *)a3 = v46;
    v83 = v46;
    *(_DWORD *)(a3 + 8) = v49;
    v50 = *(_QWORD **)(a1 + 8);
    *((_QWORD *)&v57 + 1) = 3;
    *(_QWORD *)&v57 = &v81;
    v84 = 1;
    v81 = v48;
    v82 = v47;
    v52 = sub_3411630(v50, v60, (__int64)&v75, v41, v43, v51, v57);
    LODWORD(v50) = v53;
    v37 = (unsigned __int64)v52;
    v36 = 1;
    *(_QWORD *)a4 = v52;
    *(_DWORD *)(a4 + 8) = (_DWORD)v50;
  }
  else
  {
    HIWORD(v23) = 0;
    *(_QWORD *)&v19 = sub_3406EB0(
                        *(_QWORD **)(a1 + 8),
                        (unsigned int)(*(_DWORD *)(v61 + 24) != 76) + 56,
                        (__int64)&v75,
                        *(unsigned __int16 *)(v59 + *(_QWORD *)(v72 + 48)),
                        *(_QWORD *)(v59 + *(_QWORD *)(v72 + 48) + 8),
                        v17,
                        *(_OWORD *)&v8,
                        *(_OWORD *)&v67);
    v20 = a3;
    v65 = v19;
    sub_375BC20((__int64 *)a1, v19, *((__int64 *)&v19 + 1), v20, a4, v8);
    v21 = (unsigned __int16 *)(*(_QWORD *)(v72 + 48) + v59);
    v22 = *((_QWORD *)v21 + 1);
    LOWORD(v23) = *v21;
    *(_QWORD *)&v25 = sub_3406EB0(
                        *(_QWORD **)(a1 + 8),
                        0xBCu,
                        (__int64)&v75,
                        *v21,
                        v22,
                        v24,
                        *(_OWORD *)&v8,
                        *(_OWORD *)&v67);
    v73 = v25;
    v27 = v65;
    if ( v70 == 76 )
    {
      v27 = v65;
      *(_QWORD *)&v73 = sub_34074A0(*(_QWORD **)(a1 + 8), (__int64)&v75, v25, *((__int64 *)&v25 + 1), v23, v22, v8);
      *((_QWORD *)&v73 + 1) = v55 | *((_QWORD *)&v73 + 1) & 0xFFFFFFFF00000000LL;
    }
    *(_QWORD *)&v28 = sub_3406EB0(*(_QWORD **)(a1 + 8), 0xBCu, (__int64)&v75, v23, v22, v26, *(_OWORD *)&v8, v27);
    *(_QWORD *)&v69 = sub_3406EB0(*(_QWORD **)(a1 + 8), 0xBAu, (__int64)&v75, v23, v22, v29, v73, v28);
    v30 = *(_QWORD *)(v61 + 48);
    v68 = v31;
    v74 = *(_WORD *)(v30 + 16);
    v66 = *(_QWORD *)(v30 + 24);
    v71 = *(_QWORD **)(a1 + 8);
    v32 = sub_3400BD0((__int64)v71, 0, (__int64)&v75, v23, v22, 0, v8, 0);
    v34 = v33;
    *((_QWORD *)&v69 + 1) = v68;
    *(_QWORD *)&v35 = sub_33ED040(v71, 0x14u);
    *((_QWORD *)&v56 + 1) = v34;
    *(_QWORD *)&v56 = v32;
    v37 = sub_340F900(v71, 0xD0u, (__int64)&v75, v74, v66, v68, v69, v56, v35);
  }
  sub_3760E70(a1, v61, 1, v37, v36);
  if ( v75 )
    sub_B91220((__int64)&v75, v75);
}
