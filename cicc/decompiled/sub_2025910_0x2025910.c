// Function: sub_2025910
// Address: 0x2025910
//
void __fastcall sub_2025910(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __m128i a5, __m128 a6, __m128i a7)
{
  __int64 *v9; // rax
  unsigned int v10; // ebx
  __int64 v11; // rsi
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // rsi
  char *v15; // rax
  char v16; // dl
  __int64 v17; // r9
  __int8 v18; // al
  __int64 v19; // rax
  __int64 v20; // r15
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned __int64 *v23; // rax
  unsigned __int64 v24; // r14
  __int64 v25; // rbx
  __int64 v26; // rax
  __int64 v27; // rax
  char v28; // dl
  __int64 v29; // r8
  bool v30; // al
  unsigned int v31; // eax
  __int64 v32; // rsi
  __int64 v33; // r11
  __int64 v34; // r10
  int v35; // eax
  unsigned int *v36; // r10
  __int64 v37; // rdx
  __int64 v38; // rdx
  char v39; // cl
  __m128i v40; // xmm3
  _QWORD *v41; // rsi
  __int64 v42; // rcx
  __int64 *v43; // rdi
  __int64 *v44; // rax
  int v45; // edx
  __int64 v46; // rdx
  __int64 v47; // r9
  int v48; // edx
  unsigned int v49; // edx
  const __m128i *v50; // r9
  __int128 v51; // [rsp-20h] [rbp-260h]
  __int128 v52; // [rsp-10h] [rbp-250h]
  __int128 v53; // [rsp-10h] [rbp-250h]
  __int64 v56; // [rsp+20h] [rbp-220h]
  __int8 v57; // [rsp+2Eh] [rbp-212h]
  char v58; // [rsp+2Fh] [rbp-211h]
  __int64 v59; // [rsp+30h] [rbp-210h]
  __int64 v60; // [rsp+50h] [rbp-1F0h]
  unsigned int v61; // [rsp+68h] [rbp-1D8h]
  __int64 v62; // [rsp+78h] [rbp-1C8h]
  __int64 v63; // [rsp+78h] [rbp-1C8h]
  __int64 v64; // [rsp+80h] [rbp-1C0h]
  __int64 *v65; // [rsp+80h] [rbp-1C0h]
  __int64 v66; // [rsp+88h] [rbp-1B8h]
  __int64 *v67; // [rsp+90h] [rbp-1B0h]
  __int64 *v68; // [rsp+A0h] [rbp-1A0h]
  int v69; // [rsp+B8h] [rbp-188h]
  __int64 v70; // [rsp+C0h] [rbp-180h] BYREF
  int v71; // [rsp+C8h] [rbp-178h]
  __m128i v72; // [rsp+D0h] [rbp-170h] BYREF
  __m128i v73; // [rsp+E0h] [rbp-160h] BYREF
  char v74[8]; // [rsp+F0h] [rbp-150h] BYREF
  __int64 v75; // [rsp+F8h] [rbp-148h]
  __int64 v76; // [rsp+100h] [rbp-140h] BYREF
  int v77; // [rsp+108h] [rbp-138h]
  __m128i v78; // [rsp+110h] [rbp-130h] BYREF
  __m128i v79; // [rsp+120h] [rbp-120h] BYREF
  __int64 v80; // [rsp+130h] [rbp-110h] BYREF
  __int64 v81; // [rsp+138h] [rbp-108h]
  char v82; // [rsp+140h] [rbp-100h]
  __int64 v83; // [rsp+148h] [rbp-F8h]
  __m128i v84; // [rsp+150h] [rbp-F0h] BYREF
  __m128i v85; // [rsp+160h] [rbp-E0h] BYREF
  _QWORD *v86; // [rsp+170h] [rbp-D0h] BYREF
  __int64 v87; // [rsp+178h] [rbp-C8h]
  _QWORD v88[8]; // [rsp+180h] [rbp-C0h] BYREF
  __int64 *v89; // [rsp+1C0h] [rbp-80h] BYREF
  __int64 v90; // [rsp+1C8h] [rbp-78h]
  __int64 v91; // [rsp+1D0h] [rbp-70h] BYREF
  __int64 v92; // [rsp+1D8h] [rbp-68h]

  v9 = *(__int64 **)(a2 + 32);
  v10 = *(_DWORD *)(a2 + 56);
  v11 = *(_QWORD *)(a2 + 72);
  v12 = *v9;
  v13 = v9[1];
  v70 = v11;
  v60 = v13;
  if ( v11 )
    sub_1623A60((__int64)&v70, v11, 2);
  v14 = *(_QWORD *)(a1 + 8);
  v71 = *(_DWORD *)(a2 + 64);
  v15 = *(char **)(a2 + 40);
  v16 = *v15;
  v87 = *((_QWORD *)v15 + 1);
  LOBYTE(v86) = v16;
  sub_1D19A30((__int64)&v89, v14, &v86);
  v88[0] = v12;
  v58 = (char)v89;
  v88[1] = v60;
  v59 = v90;
  v18 = v91;
  v91 = v12;
  v57 = v18;
  v19 = v92;
  v92 = v60;
  v56 = v19;
  v86 = v88;
  v89 = &v91;
  v87 = 0x400000001LL;
  v90 = 0x400000001LL;
  if ( v10 > 1 )
  {
    v20 = 40;
    v66 = 40LL * (v10 - 2) + 80;
    while ( 1 )
    {
      v23 = (unsigned __int64 *)(v20 + *(_QWORD *)(a2 + 32));
      v24 = *v23;
      v25 = v23[1];
      v26 = *((unsigned int *)v23 + 2);
      v72.m128i_i64[0] = v24;
      v27 = *(_QWORD *)(v24 + 40) + 16 * v26;
      v72.m128i_i64[1] = v25;
      v73.m128i_i64[0] = v24;
      v73.m128i_i64[1] = v25;
      v28 = *(_BYTE *)v27;
      v29 = *(_QWORD *)(v27 + 8);
      v74[0] = v28;
      v75 = v29;
      if ( v28 )
      {
        if ( (unsigned __int8)(v28 - 14) > 0x5Fu )
          goto LABEL_6;
      }
      else
      {
        v62 = v29;
        v30 = sub_1F58D20((__int64)v74);
        v28 = 0;
        v29 = v62;
        if ( !v30 )
          goto LABEL_6;
      }
      v31 = v61;
      LOBYTE(v31) = v28;
      sub_1F40D10((__int64)&v84, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), v31, v29);
      if ( v84.m128i_i8[0] != 6 )
      {
        v32 = *(_QWORD *)(a2 + 72);
        v33 = *(_QWORD *)(a1 + 8);
        v76 = v32;
        if ( v32 )
        {
          v64 = v33;
          sub_1623A60((__int64)&v76, v32, 2);
          v33 = v64;
        }
        v34 = *(_QWORD *)(a2 + 32);
        v35 = *(_DWORD *)(a2 + 64);
        v78.m128i_i64[1] = 0;
        v78.m128i_i8[0] = 0;
        v36 = (unsigned int *)(v20 + v34);
        v77 = v35;
        v79.m128i_i64[1] = 0;
        v37 = v36[2];
        v79.m128i_i8[0] = 0;
        v63 = (__int64)v36;
        v38 = *(_QWORD *)(*(_QWORD *)v36 + 40LL) + 16 * v37;
        v65 = (__int64 *)v33;
        v39 = *(_BYTE *)v38;
        v81 = *(_QWORD *)(v38 + 8);
        LOBYTE(v80) = v39;
        sub_1D19A30((__int64)&v84, v33, &v80);
        a7 = _mm_load_si128(&v84);
        v40 = _mm_load_si128(&v85);
        v78 = a7;
        v79 = v40;
        sub_1D40600(
          (__int64)&v84,
          v65,
          v63,
          (__int64)&v76,
          (const void ***)&v78,
          (const void ***)&v79,
          a5,
          *(double *)a6.m128_u64,
          a7);
        if ( v76 )
          sub_161E7C0((__int64)&v76, v76);
        v72.m128i_i64[0] = v84.m128i_i64[0];
        v72.m128i_i32[2] = v84.m128i_i32[2];
        v73.m128i_i64[0] = v85.m128i_i64[0];
        v73.m128i_i32[2] = v85.m128i_i32[2];
        v21 = (unsigned int)v87;
        if ( (unsigned int)v87 < HIDWORD(v87) )
          goto LABEL_7;
LABEL_18:
        sub_16CD150((__int64)&v86, v88, 0, 16, v29, v17);
        v21 = (unsigned int)v87;
        goto LABEL_7;
      }
      sub_2017DE0(a1, v24, v25, &v72, &v73);
LABEL_6:
      v21 = (unsigned int)v87;
      if ( (unsigned int)v87 >= HIDWORD(v87) )
        goto LABEL_18;
LABEL_7:
      a5 = _mm_load_si128(&v72);
      *(__m128i *)&v86[2 * v21] = a5;
      v22 = (unsigned int)v90;
      LODWORD(v87) = v87 + 1;
      if ( (unsigned int)v90 >= HIDWORD(v90) )
      {
        sub_16CD150((__int64)&v89, &v91, 0, 16, v29, v17);
        v22 = (unsigned int)v90;
      }
      a6 = (__m128)_mm_load_si128(&v73);
      v20 += 40;
      *(__m128 *)&v89[2 * v22] = a6;
      LODWORD(v90) = v90 + 1;
      if ( v66 == v20 )
      {
        v41 = v86;
        v42 = (unsigned int)v87;
        goto LABEL_21;
      }
    }
  }
  v41 = v88;
  v42 = 1;
LABEL_21:
  v43 = *(__int64 **)(a1 + 8);
  *((_QWORD *)&v52 + 1) = v42;
  v82 = 1;
  LOBYTE(v80) = v58;
  v83 = 0;
  v81 = v59;
  v85.m128i_i8[0] = 1;
  v84.m128i_i8[0] = v57;
  v85.m128i_i64[1] = 0;
  v84.m128i_i64[1] = v56;
  *(_QWORD *)&v52 = v41;
  v44 = sub_1D373B0(
          v43,
          *(unsigned __int16 *)(a2 + 24),
          (__int64)&v70,
          (unsigned __int8 *)&v80,
          2,
          *(double *)a5.m128i_i64,
          *(double *)a6.m128_u64,
          a7,
          v17,
          v52);
  v69 = v45;
  v46 = (unsigned int)v90;
  *(_QWORD *)a3 = v44;
  *(_DWORD *)(a3 + 8) = v69;
  *((_QWORD *)&v51 + 1) = v46;
  *(_QWORD *)&v51 = v89;
  v68 = sub_1D373B0(
          *(__int64 **)(a1 + 8),
          *(unsigned __int16 *)(a2 + 24),
          (__int64)&v70,
          (unsigned __int8 *)&v84,
          2,
          *(double *)a5.m128i_i64,
          *(double *)a6.m128_u64,
          a7,
          v47,
          v51);
  *(_QWORD *)a4 = v68;
  *(_DWORD *)(a4 + 8) = v48;
  *((_QWORD *)&v53 + 1) = 1;
  *(_QWORD *)&v53 = v68;
  v67 = sub_1D332F0(
          *(__int64 **)(a1 + 8),
          2,
          (__int64)&v70,
          1,
          0,
          0,
          *(double *)a5.m128i_i64,
          *(double *)a6.m128_u64,
          a7,
          *(_QWORD *)a3,
          1u,
          v53);
  sub_2013400(a1, a2, 1, (__int64)v67, (__m128i *)(v49 | v60 & 0xFFFFFFFF00000000LL), v50);
  if ( v89 != &v91 )
    _libc_free((unsigned __int64)v89);
  if ( v86 != v88 )
    _libc_free((unsigned __int64)v86);
  if ( v70 )
    sub_161E7C0((__int64)&v70, v70);
}
