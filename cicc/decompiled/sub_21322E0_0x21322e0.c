// Function: sub_21322E0
// Address: 0x21322e0
//
void __fastcall sub_21322E0(__int64 a1, unsigned __int64 a2, __int64 a3, _QWORD *a4, double a5, double a6, __m128i a7)
{
  unsigned __int64 v7; // r10
  __int64 *v11; // rax
  __int64 v12; // rsi
  __m128 v13; // xmm0
  __m128i v14; // xmm1
  __int64 v15; // r15
  __int64 v16; // r11
  __int64 *v17; // rdi
  unsigned __int64 v18; // rdx
  __int64 v19; // rax
  const void **v20; // rbx
  __int128 v21; // rax
  __int64 *v22; // r14
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 *v28; // rax
  __int64 *v29; // r15
  __int64 v30; // rdx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 *v35; // rax
  bool v36; // zf
  __int64 v37; // rcx
  __int64 *v38; // r14
  unsigned int v39; // edx
  __int64 v40; // r15
  __int64 v41; // r9
  __int64 v42; // rax
  __int64 v43; // rdx
  __int128 v44; // rax
  __int64 *v45; // r14
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // r9
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 *v51; // rax
  __int64 v52; // rcx
  unsigned int v53; // edx
  __int16 *v54; // r15
  __int64 v55; // rax
  __int64 v56; // rdx
  __int128 v57; // rax
  __int64 *v58; // rax
  __m128i *v59; // rdx
  const __m128i *v60; // r9
  __int128 v61; // [rsp-20h] [rbp-D0h]
  unsigned __int64 v62; // [rsp+8h] [rbp-A8h]
  unsigned __int64 v63; // [rsp+18h] [rbp-98h]
  __int64 *v64; // [rsp+18h] [rbp-98h]
  __int64 *v65; // [rsp+20h] [rbp-90h]
  __int16 *v66; // [rsp+28h] [rbp-88h]
  __int64 v67; // [rsp+30h] [rbp-80h]
  __int128 v68; // [rsp+30h] [rbp-80h]
  unsigned __int64 v69; // [rsp+40h] [rbp-70h]
  __int64 v70; // [rsp+40h] [rbp-70h]
  __int128 v71; // [rsp+40h] [rbp-70h]
  __int128 v72; // [rsp+50h] [rbp-60h]
  __int16 *v73; // [rsp+58h] [rbp-58h]
  unsigned __int64 v74; // [rsp+60h] [rbp-50h]
  unsigned int v75; // [rsp+68h] [rbp-48h]
  unsigned __int64 v76; // [rsp+68h] [rbp-48h]
  __int64 v77; // [rsp+70h] [rbp-40h] BYREF
  int v78; // [rsp+78h] [rbp-38h]

  v7 = a2;
  v11 = *(__int64 **)(a2 + 32);
  v12 = *(_QWORD *)(a2 + 72);
  v13 = (__m128)_mm_loadu_si128((const __m128i *)v11);
  v14 = _mm_loadu_si128((const __m128i *)(v11 + 5));
  v15 = *v11;
  v16 = *((unsigned int *)v11 + 2);
  v77 = v12;
  if ( v12 )
  {
    v69 = v7;
    v75 = v16;
    sub_1623A60((__int64)&v77, v12, 2);
    v7 = v69;
    v16 = v75;
  }
  v17 = *(__int64 **)(a1 + 8);
  v70 = 16 * v16;
  v78 = *(_DWORD *)(v7 + 64);
  v76 = v7;
  v65 = sub_1D332F0(
          v17,
          (unsigned int)(*(_WORD *)(v7 + 24) != 70) + 52,
          (__int64)&v77,
          *(unsigned __int8 *)(*(_QWORD *)(v15 + 40) + 16 * v16),
          *(const void ***)(*(_QWORD *)(v15 + 40) + 16 * v16 + 8),
          0,
          *(double *)v13.m128_u64,
          *(double *)v14.m128i_i64,
          a7,
          v13.m128_i64[0],
          v13.m128_u64[1],
          *(_OWORD *)&v14);
  v66 = (__int16 *)v18;
  sub_200E870(a1, (__int64)v65, v18, a3, a4, (__m128i)v13, *(double *)v14.m128i_i64, a7);
  v19 = *(_QWORD *)(v76 + 40);
  v63 = v76;
  LOBYTE(v76) = *(_BYTE *)(v19 + 16);
  v20 = *(const void ***)(v19 + 24);
  *(_QWORD *)&v21 = sub_1D38BB0(
                      *(_QWORD *)(a1 + 8),
                      0,
                      (__int64)&v77,
                      *(unsigned __int8 *)(*(_QWORD *)(v15 + 40) + v70),
                      *(const void ***)(*(_QWORD *)(v15 + 40) + v70 + 8),
                      0,
                      (__m128i)v13,
                      *(double *)v14.m128i_i64,
                      a7,
                      0);
  v22 = *(__int64 **)(a1 + 8);
  v71 = v21;
  v26 = sub_1D28D50(v22, 0x13u, *((__int64 *)&v21 + 1), v23, v24, v25);
  v28 = sub_1D3A900(
          v22,
          0x89u,
          (__int64)&v77,
          (unsigned __int8)v76,
          v20,
          0,
          v13,
          *(double *)v14.m128i_i64,
          a7,
          v13.m128_u64[0],
          (__int16 *)v13.m128_u64[1],
          v71,
          v26,
          v27);
  v29 = *(__int64 **)(a1 + 8);
  LODWORD(v22) = v30;
  v74 = (unsigned __int64)v28;
  v33 = sub_1D28D50(v29, 0x13u, v30, (unsigned __int8)v76, v31, v32);
  v35 = sub_1D3A900(
          v29,
          0x89u,
          (__int64)&v77,
          (unsigned __int8)v76,
          v20,
          0,
          v13,
          *(double *)v14.m128i_i64,
          a7,
          v14.m128i_u64[0],
          (__int16 *)v14.m128i_i64[1],
          v71,
          v33,
          v34);
  v36 = *(_WORD *)(v63 + 24) == 70;
  v62 = v63;
  v64 = *(__int64 **)(a1 + 8);
  v37 = (unsigned int)v22;
  v38 = v35;
  v67 = v37;
  v73 = (__int16 *)v37;
  v40 = v39;
  v42 = sub_1D28D50(v64, 5 * (unsigned int)!v36 + 17, v39, v37, v37, v41);
  *((_QWORD *)&v61 + 1) = v40;
  *(_QWORD *)&v61 = v38;
  *(_QWORD *)&v44 = sub_1D3A900(
                      v64,
                      0x89u,
                      (__int64)&v77,
                      (unsigned __int8)v76,
                      v20,
                      0,
                      v13,
                      *(double *)v14.m128i_i64,
                      a7,
                      v74,
                      v73,
                      v61,
                      v42,
                      v43);
  v45 = *(__int64 **)(a1 + 8);
  v72 = v44;
  v49 = sub_1D28D50(v45, 0x13u, *((__int64 *)&v44 + 1), v46, v47, v48);
  v51 = sub_1D3A900(
          v45,
          0x89u,
          (__int64)&v77,
          (unsigned __int8)v76,
          v20,
          0,
          v13,
          *(double *)v14.m128i_i64,
          a7,
          (unsigned __int64)v65,
          v66,
          v71,
          v49,
          v50);
  v52 = v67;
  *(_QWORD *)&v68 = v51;
  *(_QWORD *)&v71 = *(_QWORD *)(a1 + 8);
  *((_QWORD *)&v68 + 1) = v53;
  v54 = (__int16 *)v52;
  v55 = sub_1D28D50((_QWORD *)v71, 0x16u, v53, v52, (__int64)v51, 0);
  *(_QWORD *)&v57 = sub_1D3A900(
                      (__int64 *)v71,
                      0x89u,
                      (__int64)&v77,
                      (unsigned __int8)v76,
                      v20,
                      0,
                      v13,
                      *(double *)v14.m128i_i64,
                      a7,
                      v74,
                      v54,
                      v68,
                      v55,
                      v56);
  v58 = sub_1D332F0(
          *(__int64 **)(a1 + 8),
          118,
          (__int64)&v77,
          (unsigned __int8)v76,
          v20,
          0,
          *(double *)v13.m128_u64,
          *(double *)v14.m128i_i64,
          a7,
          v72,
          *((unsigned __int64 *)&v72 + 1),
          v57);
  sub_2013400(a1, v62, 1, (__int64)v58, v59, v60);
  if ( v77 )
    sub_161E7C0((__int64)&v77, v77);
}
