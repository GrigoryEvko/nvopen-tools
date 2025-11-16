// Function: sub_212E720
// Address: 0x212e720
//
void __fastcall sub_212E720(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 a4,
        double a5,
        double a6,
        __m128i a7,
        __int64 a8,
        unsigned int a9)
{
  __int64 *v11; // rax
  __int64 v12; // rsi
  __int64 v13; // rbx
  __m128 v14; // xmm0
  __m128i v15; // xmm1
  __int64 v16; // rbx
  __int64 v17; // r14
  __int64 v18; // rdx
  unsigned __int8 v19; // r15
  __int64 v20; // rbx
  __int64 v21; // r14
  __int64 v22; // r12
  unsigned int v23; // r9d
  int v24; // ecx
  __int64 *v25; // r14
  unsigned __int64 v26; // rdx
  __int16 *v27; // r15
  __int64 *v28; // rbx
  __int64 v29; // rdx
  __int64 v30; // r9
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 *v33; // rcx
  const __m128i *v34; // r9
  unsigned int v35; // edx
  __int64 v36; // r8
  unsigned int v37; // edi
  const void ***v38; // rax
  __int64 v39; // rsi
  __int64 *v40; // rdi
  int v41; // edx
  int v42; // r14d
  __m128i v43; // xmm2
  __m128i v44; // xmm3
  __m128i v45; // xmm4
  __m128i v46; // xmm5
  __int64 v47; // r9
  int v48; // edx
  __int64 *v49; // rdi
  __int64 v50; // r9
  __int64 *v51; // rax
  int v52; // edx
  unsigned int v53; // eax
  __int128 v54; // [rsp-20h] [rbp-180h]
  __int128 v55; // [rsp-10h] [rbp-170h]
  unsigned int v56; // [rsp+8h] [rbp-158h]
  unsigned int v57; // [rsp+10h] [rbp-150h]
  __int64 v58; // [rsp+18h] [rbp-148h]
  unsigned __int64 v61; // [rsp+40h] [rbp-120h]
  unsigned int v62; // [rsp+48h] [rbp-118h]
  const void **v63; // [rsp+48h] [rbp-118h]
  __int64 v64; // [rsp+60h] [rbp-100h]
  unsigned __int64 v65; // [rsp+60h] [rbp-100h]
  const void ***v66; // [rsp+60h] [rbp-100h]
  __int64 *v67; // [rsp+80h] [rbp-E0h]
  __int64 v68; // [rsp+90h] [rbp-D0h] BYREF
  int v69; // [rsp+98h] [rbp-C8h]
  __m128i v70; // [rsp+A0h] [rbp-C0h] BYREF
  __m128i v71; // [rsp+B0h] [rbp-B0h] BYREF
  __m128i v72; // [rsp+C0h] [rbp-A0h] BYREF
  __m128i v73; // [rsp+D0h] [rbp-90h] BYREF
  _OWORD v74[2]; // [rsp+E0h] [rbp-80h] BYREF
  __m128i v75; // [rsp+100h] [rbp-60h] BYREF
  __m128i v76; // [rsp+110h] [rbp-50h]
  __int64 *v77; // [rsp+120h] [rbp-40h]
  int v78; // [rsp+128h] [rbp-38h]

  v11 = *(__int64 **)(a2 + 32);
  v12 = *(_QWORD *)(a2 + 72);
  v13 = *v11;
  v14 = (__m128)_mm_loadu_si128((const __m128i *)v11);
  v15 = _mm_loadu_si128((const __m128i *)(v11 + 5));
  v68 = v12;
  v64 = v13;
  v16 = *((unsigned int *)v11 + 2);
  if ( v12 )
  {
    v62 = a9;
    sub_1623A60((__int64)&v68, v12, 2);
    a9 = v62;
  }
  v17 = *(_QWORD *)a1;
  v58 = 16 * v16;
  v69 = *(_DWORD *)(a2 + 64);
  v61 = a2;
  v18 = 16 * v16 + *(_QWORD *)(v64 + 40);
  v19 = *(_BYTE *)v18;
  v20 = v17;
  v21 = *(_QWORD *)(v18 + 8);
  v22 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL);
  while ( 1 )
  {
    LOBYTE(a9) = v19;
    sub_1F40D10((__int64)&v75, v20, v22, a9, v21);
    if ( !v75.m128i_i8[0] )
      break;
    v53 = v57;
    v56 = v23;
    LOBYTE(v53) = v19;
    sub_1F40D10((__int64)&v75, v20, v22, v53, v21);
    v19 = v75.m128i_u8[8];
    v21 = v76.m128i_i64[0];
    a9 = v56;
  }
  v24 = *(unsigned __int16 *)(v61 + 24);
  if ( v19 == 1 )
  {
    v37 = 1;
  }
  else
  {
    if ( !v19 )
      goto LABEL_7;
    v37 = v19;
    if ( !*(_QWORD *)(v20 + 8LL * v19 + 120) )
      goto LABEL_7;
  }
  if ( (*(_BYTE *)((unsigned int)(v24 != 52) + 68 + v20 + 259LL * v37 + 2422) & 0xFB) != 0 )
  {
LABEL_7:
    v25 = sub_1D332F0(
            *(__int64 **)(a1 + 8),
            (unsigned int)(v24 != 71) + 52,
            (__int64)&v68,
            *(unsigned __int8 *)(*(_QWORD *)(v64 + 40) + v58),
            *(const void ***)(*(_QWORD *)(v64 + 40) + v58 + 8),
            0,
            *(double *)v14.m128_u64,
            *(double *)v15.m128i_i64,
            a7,
            v14.m128_i64[0],
            v14.m128_u64[1],
            *(_OWORD *)&v15);
    v27 = (__int16 *)v26;
    sub_200E870(a1, (__int64)v25, v26, a3, (_QWORD *)a4, (__m128i)v14, *(double *)v15.m128i_i64, a7);
    v28 = *(__int64 **)(a1 + 8);
    v63 = *(const void ***)(*(_QWORD *)(v61 + 40) + 24LL);
    v65 = *(unsigned __int8 *)(*(_QWORD *)(v61 + 40) + 16LL);
    v31 = sub_1D28D50(v28, 2 * (unsigned int)(*(_WORD *)(v61 + 24) == 71) + 10, v29, v65, (__int64)v63, v30);
    v33 = sub_1D3A900(
            v28,
            0x89u,
            (__int64)&v68,
            v65,
            v63,
            0,
            v14,
            *(double *)v15.m128i_i64,
            a7,
            (unsigned __int64)v25,
            v27,
            *(_OWORD *)&v14,
            v31,
            v32);
    v36 = v35;
    goto LABEL_8;
  }
  v70.m128i_i64[0] = 0;
  v70.m128i_i32[2] = 0;
  v71.m128i_i64[0] = 0;
  v71.m128i_i32[2] = 0;
  v72.m128i_i64[0] = 0;
  v72.m128i_i32[2] = 0;
  v73.m128i_i64[0] = 0;
  v73.m128i_i32[2] = 0;
  sub_20174B0(a1, v14.m128_u64[0], v14.m128_i64[1], &v70, &v71);
  sub_20174B0(a1, v15.m128i_u64[0], v15.m128i_i64[1], &v72, &v73);
  v38 = (const void ***)sub_1D252B0(
                          *(_QWORD *)(a1 + 8),
                          *(unsigned __int8 *)(*(_QWORD *)(v70.m128i_i64[0] + 40) + 16LL * v70.m128i_u32[2]),
                          *(_QWORD *)(*(_QWORD *)(v70.m128i_i64[0] + 40) + 16LL * v70.m128i_u32[2] + 8),
                          *(unsigned __int8 *)(*(_QWORD *)(v61 + 40) + 16LL),
                          *(_QWORD *)(*(_QWORD *)(v61 + 40) + 24LL));
  v39 = *(unsigned __int16 *)(v61 + 24);
  v40 = *(__int64 **)(a1 + 8);
  v77 = 0;
  v42 = v41;
  *((_QWORD *)&v55 + 1) = 2;
  v43 = _mm_loadu_si128(&v70);
  *(_QWORD *)&v55 = v74;
  v44 = _mm_loadu_si128(&v72);
  v45 = _mm_loadu_si128(&v71);
  v74[0] = v43;
  v46 = _mm_loadu_si128(&v73);
  v66 = v38;
  v74[1] = v44;
  v75 = v45;
  v76 = v46;
  v78 = 0;
  v67 = sub_1D36D80(v40, v39, (__int64)&v68, v38, v41, *(double *)v14.m128_u64, *(double *)v15.m128i_i64, v43, v47, v55);
  *(_QWORD *)a3 = v67;
  v77 = v67;
  *(_DWORD *)(a3 + 8) = v48;
  v49 = *(__int64 **)(a1 + 8);
  *((_QWORD *)&v54 + 1) = 3;
  *(_QWORD *)&v54 = &v75;
  v78 = 1;
  v51 = sub_1D36D80(
          v49,
          (unsigned int)((_DWORD)v39 != 71) + 68,
          (__int64)&v68,
          v66,
          v42,
          *(double *)v14.m128_u64,
          *(double *)v15.m128i_i64,
          v43,
          v50,
          v54);
  v36 = 1;
  v33 = v51;
  *(_QWORD *)a4 = v51;
  *(_DWORD *)(a4 + 8) = v52;
LABEL_8:
  sub_2013400(a1, v61, 1, (__int64)v33, (__m128i *)v36, v34);
  if ( v68 )
    sub_161E7C0((__int64)&v68, v68);
}
