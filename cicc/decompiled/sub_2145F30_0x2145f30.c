// Function: sub_2145F30
// Address: 0x2145f30
//
__int64 __fastcall sub_2145F30(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  char *v6; // rdx
  char v7; // al
  const void **v8; // rdx
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // r15
  unsigned __int64 v14; // r13
  unsigned int v15; // eax
  unsigned int v16; // esi
  __int64 v17; // rax
  __int64 v18; // rax
  char v19; // cl
  __int64 v20; // rax
  __int16 *v21; // rdx
  bool v22; // al
  __int64 v23; // rax
  unsigned __int64 v24; // r15
  __int64 v25; // r14
  const void ***v26; // rax
  unsigned int v27; // edx
  __int64 v28; // r13
  unsigned __int64 v29; // r15
  unsigned int v30; // edx
  __int128 v31; // rax
  unsigned int v32; // edx
  unsigned int v33; // edx
  __int64 v34; // r12
  const void **v36; // rdx
  __int128 v37; // [rsp-10h] [rbp-130h]
  __int128 v38; // [rsp-10h] [rbp-130h]
  _QWORD *v39; // [rsp+8h] [rbp-118h]
  __int64 v40; // [rsp+10h] [rbp-110h]
  unsigned int v41; // [rsp+18h] [rbp-108h]
  __int64 *v42; // [rsp+18h] [rbp-108h]
  __int64 v43; // [rsp+20h] [rbp-100h]
  __int64 *v44; // [rsp+20h] [rbp-100h]
  unsigned int v45; // [rsp+28h] [rbp-F8h]
  const void **v46; // [rsp+30h] [rbp-F0h]
  int v47; // [rsp+40h] [rbp-E0h]
  unsigned int v48; // [rsp+40h] [rbp-E0h]
  unsigned __int64 v49; // [rsp+40h] [rbp-E0h]
  __int64 *v50; // [rsp+40h] [rbp-E0h]
  __int16 *v51; // [rsp+48h] [rbp-D8h]
  unsigned __int64 v52; // [rsp+48h] [rbp-D8h]
  __int64 *v53; // [rsp+50h] [rbp-D0h]
  __int64 *v54; // [rsp+60h] [rbp-C0h]
  unsigned int v55; // [rsp+90h] [rbp-90h] BYREF
  const void **v56; // [rsp+98h] [rbp-88h]
  __int64 v57; // [rsp+A0h] [rbp-80h] BYREF
  int v58; // [rsp+A8h] [rbp-78h]
  __m128i v59; // [rsp+B0h] [rbp-70h] BYREF
  __int128 v60; // [rsp+C0h] [rbp-60h] BYREF
  _BYTE v61[8]; // [rsp+D0h] [rbp-50h] BYREF
  __int64 v62; // [rsp+D8h] [rbp-48h]
  __int64 v63; // [rsp+E0h] [rbp-40h]

  v6 = *(char **)(a2 + 40);
  v7 = *v6;
  v8 = (const void **)*((_QWORD *)v6 + 1);
  LOBYTE(v55) = v7;
  v56 = v8;
  if ( v7 )
    v47 = word_4310E40[(unsigned __int8)(v7 - 14)];
  else
    v47 = sub_1F58D30((__int64)&v55);
  v9 = *(_QWORD *)(a2 + 72);
  v57 = v9;
  if ( v9 )
    sub_1623A60((__int64)&v57, v9, 2);
  v10 = *(_QWORD *)(a1 + 8);
  v11 = *(_QWORD *)a1;
  v58 = *(_DWORD *)(a2 + 64);
  v12 = *(_QWORD *)(a2 + 32);
  v13 = *(unsigned int *)(v12 + 48);
  v14 = *(_QWORD *)(v12 + 40);
  v43 = 16 * v13;
  sub_1F40D10(
    (__int64)v61,
    v11,
    *(_QWORD *)(v10 + 48),
    *(unsigned __int8 *)(*(_QWORD *)(v14 + 40) + v43),
    *(_QWORD *)(*(_QWORD *)(v14 + 40) + v43 + 8));
  v40 = v63;
  LODWORD(v11) = 2 * v47;
  v41 = 2 * v47;
  v48 = (unsigned __int8)v62;
  v39 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 48LL);
  LOBYTE(v15) = sub_1D15020(v62, v11);
  v46 = 0;
  if ( !(_BYTE)v15 )
  {
    v15 = sub_1F593D0(v39, v48, v40, v41);
    v45 = v15;
    v46 = v36;
  }
  v16 = v45;
  LOBYTE(v16) = v15;
  v17 = sub_1D309E0(
          *(__int64 **)(a1 + 8),
          158,
          (__int64)&v57,
          v16,
          v46,
          0,
          *(double *)a3.m128i_i64,
          a4,
          *(double *)a5.m128i_i64,
          *(_OWORD *)*(_QWORD *)(a2 + 32));
  v59.m128i_i32[2] = 0;
  v49 = v17;
  v18 = *(_QWORD *)(v14 + 40) + v43;
  DWORD2(v60) = 0;
  v59.m128i_i64[0] = 0;
  *(_QWORD *)&v60 = 0;
  v19 = *(_BYTE *)v18;
  v20 = *(_QWORD *)(v18 + 8);
  v51 = v21;
  v61[0] = v19;
  v62 = v20;
  if ( v19 )
    v22 = (unsigned __int8)(v19 - 14) <= 0x47u || (unsigned __int8)(v19 - 2) <= 5u;
  else
    v22 = sub_1F58CF0((__int64)v61);
  if ( v22 )
    sub_20174B0(a1, v14, v13, &v59, &v60);
  else
    sub_2016B80(a1, v14, v13, &v59, &v60);
  if ( *(_BYTE *)sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL)) )
  {
    a3 = _mm_loadu_si128(&v59);
    v59.m128i_i64[0] = v60;
    v59.m128i_i32[2] = DWORD2(v60);
    *(_QWORD *)&v60 = a3.m128i_i64[0];
    DWORD2(v60) = a3.m128i_i32[2];
  }
  v23 = *(_QWORD *)(a2 + 32);
  v24 = *(_QWORD *)(v23 + 88);
  v25 = *(_QWORD *)(v23 + 80);
  v26 = (const void ***)(*(_QWORD *)(v25 + 40) + 16LL * *(unsigned int *)(v23 + 88));
  *((_QWORD *)&v37 + 1) = v24;
  *(_QWORD *)&v37 = v25;
  v44 = sub_1D332F0(
          *(__int64 **)(a1 + 8),
          52,
          (__int64)&v57,
          *(unsigned __int8 *)v26,
          v26[1],
          0,
          *(double *)a3.m128i_i64,
          a4,
          a5,
          v25,
          v24,
          v37);
  v28 = 16LL * v27;
  v29 = v27 | v24 & 0xFFFFFFFF00000000LL;
  v50 = sub_1D3A900(
          *(__int64 **)(a1 + 8),
          0x69u,
          (__int64)&v57,
          v16,
          v46,
          0,
          (__m128)a3,
          a4,
          a5,
          v49,
          v51,
          *(_OWORD *)&v59,
          (__int64)v44,
          v29);
  v52 = v30 | (unsigned __int64)v51 & 0xFFFFFFFF00000000LL;
  v42 = *(__int64 **)(a1 + 8);
  *(_QWORD *)&v31 = sub_1D38BB0(
                      (__int64)v42,
                      1,
                      (__int64)&v57,
                      *(unsigned __int8 *)(v28 + v44[5]),
                      *(const void ***)(v28 + v44[5] + 8),
                      0,
                      a3,
                      a4,
                      a5,
                      0);
  v54 = sub_1D332F0(
          v42,
          52,
          (__int64)&v57,
          *(unsigned __int8 *)(v44[5] + v28),
          *(const void ***)(v44[5] + v28 + 8),
          0,
          *(double *)a3.m128i_i64,
          a4,
          a5,
          (__int64)v44,
          v29,
          v31);
  v53 = sub_1D3A900(
          *(__int64 **)(a1 + 8),
          0x69u,
          (__int64)&v57,
          v16,
          v46,
          0,
          (__m128)a3,
          a4,
          a5,
          (unsigned __int64)v50,
          (__int16 *)v52,
          v60,
          (__int64)v54,
          v32 | v29 & 0xFFFFFFFF00000000LL);
  *((_QWORD *)&v38 + 1) = v33 | v52 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v38 = v53;
  v34 = sub_1D309E0(
          *(__int64 **)(a1 + 8),
          158,
          (__int64)&v57,
          v55,
          v56,
          0,
          *(double *)a3.m128i_i64,
          a4,
          *(double *)a5.m128i_i64,
          v38);
  if ( v57 )
    sub_161E7C0((__int64)&v57, v57);
  return v34;
}
