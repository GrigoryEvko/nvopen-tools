// Function: sub_2144300
// Address: 0x2144300
//
void __fastcall sub_2144300(__int64 a1, unsigned __int64 a2, __m128i *a3, __int64 a4, __m128i a5)
{
  __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rsi
  unsigned __int8 *v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rax
  __m128i v14; // xmm1
  __int64 v15; // r15
  int v16; // r11d
  _QWORD *v17; // rdi
  __int64 v18; // rax
  __m128i v19; // xmm2
  unsigned __int16 v20; // dx
  __int64 v21; // rax
  char v22; // di
  __int32 v23; // edx
  unsigned int v24; // r10d
  __int64 v25; // r11
  __int128 v26; // rax
  __int64 *v27; // rax
  _QWORD *v28; // rdi
  unsigned int v29; // edx
  unsigned __int64 v30; // rax
  __int64 v31; // rdx
  unsigned __int16 v32; // si
  unsigned __int64 v33; // r8
  int v34; // edx
  __int64 v35; // rax
  int v36; // edx
  __int64 *v37; // r15
  unsigned int v38; // edx
  const __m128i *v39; // r9
  __m128i v40; // xmm0
  __int64 v41; // r10
  char v42; // r9
  __int64 v43; // rdx
  bool v44; // zf
  __int128 v45; // [rsp-10h] [rbp-150h]
  char v46; // [rsp+Fh] [rbp-131h]
  __int64 v47; // [rsp+10h] [rbp-130h]
  __int64 v48; // [rsp+18h] [rbp-128h]
  unsigned int v49; // [rsp+28h] [rbp-118h]
  __int64 *v50; // [rsp+28h] [rbp-118h]
  int v51; // [rsp+30h] [rbp-110h]
  __int64 v52; // [rsp+38h] [rbp-108h]
  __int64 v54; // [rsp+48h] [rbp-F8h]
  __int64 v55; // [rsp+50h] [rbp-F0h]
  __m128i *v56; // [rsp+50h] [rbp-F0h]
  __int64 v57; // [rsp+B0h] [rbp-90h] BYREF
  int v58; // [rsp+B8h] [rbp-88h]
  unsigned int v59; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v60; // [rsp+C8h] [rbp-78h]
  __m128i v61; // [rsp+D0h] [rbp-70h] BYREF
  __int64 v62; // [rsp+E0h] [rbp-60h]
  __int128 v63; // [rsp+F0h] [rbp-50h] BYREF
  __int64 v64; // [rsp+100h] [rbp-40h]

  v8 = *(_QWORD *)(a2 + 72);
  v57 = v8;
  if ( v8 )
    sub_1623A60((__int64)&v57, v8, 2);
  v9 = *(_QWORD *)(a1 + 8);
  v10 = *(_QWORD *)a1;
  v58 = *(_DWORD *)(a2 + 64);
  v11 = *(unsigned __int8 **)(a2 + 40);
  v46 = *v11;
  sub_1F40D10((__int64)&v63, v10, *(_QWORD *)(v9 + 48), *v11, *((_QWORD *)v11 + 1));
  v12 = *(_QWORD *)(a2 + 104);
  LOBYTE(v59) = BYTE8(v63);
  v60 = v64;
  v13 = *(_QWORD *)(a2 + 32);
  v14 = _mm_loadu_si128((const __m128i *)(v13 + 40));
  v15 = *(_QWORD *)v13;
  v54 = *(_QWORD *)(v13 + 8);
  v52 = *(_QWORD *)(v13 + 40);
  v49 = *(_DWORD *)(v13 + 48);
  v16 = sub_1E34390(v12);
  v17 = *(_QWORD **)(a1 + 8);
  v51 = v16;
  v18 = *(_QWORD *)(a2 + 104);
  v19 = _mm_loadu_si128((const __m128i *)(v18 + 40));
  v62 = *(_QWORD *)(v18 + 56);
  v20 = *(_WORD *)(v18 + 32);
  v61 = v19;
  v21 = sub_1D2B730(
          v17,
          v59,
          v60,
          (__int64)&v57,
          v15,
          v54,
          v14.m128i_i64[0],
          v14.m128i_i64[1],
          *(_OWORD *)v18,
          *(_QWORD *)(v18 + 16),
          v16,
          v20,
          (__int64)&v61,
          0);
  v22 = v59;
  a3->m128i_i64[0] = v21;
  a3->m128i_i32[2] = v23;
  if ( v22 )
    v24 = sub_2143AC0(v22);
  else
    v24 = sub_1F58D40((__int64)&v59);
  v48 = v24 >> 3;
  v25 = 16LL * v49;
  v50 = *(__int64 **)(a1 + 8);
  v47 = v25;
  *(_QWORD *)&v26 = sub_1D38BB0(
                      (__int64)v50,
                      v48,
                      (__int64)&v57,
                      *(unsigned __int8 *)(*(_QWORD *)(v52 + 40) + v25),
                      *(const void ***)(*(_QWORD *)(v52 + 40) + v25 + 8),
                      0,
                      a5,
                      *(double *)v14.m128i_i64,
                      v19,
                      0);
  v27 = sub_1D332F0(
          v50,
          52,
          (__int64)&v57,
          *(unsigned __int8 *)(*(_QWORD *)(v52 + 40) + v47),
          *(const void ***)(*(_QWORD *)(v52 + 40) + v47 + 8),
          0,
          *(double *)a5.m128i_i64,
          *(double *)v14.m128i_i64,
          v19,
          v14.m128i_i64[0],
          v14.m128i_u64[1],
          v26);
  v28 = *(_QWORD **)(a1 + 8);
  v55 = (__int64)v27;
  v30 = v29 | v14.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  v31 = *(_QWORD *)(a2 + 104);
  v32 = *(_WORD *)(v31 + 32);
  v33 = *(_QWORD *)v31 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v33 )
  {
    v41 = *(_QWORD *)(v31 + 8) + v48;
    v42 = *(_BYTE *)(v31 + 16);
    if ( (*(_QWORD *)v31 & 4) != 0 )
    {
      *((_QWORD *)&v63 + 1) = *(_QWORD *)(v31 + 8) + v48;
      LOBYTE(v64) = v42;
      *(_QWORD *)&v63 = v33 | 4;
      HIDWORD(v64) = *(_DWORD *)(v33 + 12);
    }
    else
    {
      v43 = *(_QWORD *)v33;
      *(_QWORD *)&v63 = v33;
      *((_QWORD *)&v63 + 1) = v41;
      v44 = *(_BYTE *)(v43 + 8) == 16;
      LOBYTE(v64) = v42;
      if ( v44 )
        v43 = **(_QWORD **)(v43 + 16);
      HIDWORD(v64) = *(_DWORD *)(v43 + 8) >> 8;
    }
  }
  else
  {
    v34 = *(_DWORD *)(v31 + 20);
    LODWORD(v64) = 0;
    v63 = 0u;
    HIDWORD(v64) = v34;
  }
  v35 = sub_1D2B730(
          v28,
          v59,
          v60,
          (__int64)&v57,
          v15,
          v54,
          v55,
          v30,
          v63,
          v64,
          -(v48 | v51) & (v48 | v51),
          v32,
          (__int64)&v61,
          0);
  *(_QWORD *)a4 = v35;
  *(_DWORD *)(a4 + 8) = v36;
  *((_QWORD *)&v45 + 1) = 1;
  *(_QWORD *)&v45 = v35;
  v37 = sub_1D332F0(
          *(__int64 **)(a1 + 8),
          2,
          (__int64)&v57,
          1,
          0,
          0,
          *(double *)a5.m128i_i64,
          *(double *)v14.m128i_i64,
          v19,
          a3->m128i_i64[0],
          1u,
          v45);
  v56 = (__m128i *)(v38 | v54 & 0xFFFFFFFF00000000LL);
  if ( *(_BYTE *)sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL)) == 1 || v46 == 13 )
  {
    v40 = _mm_loadu_si128(a3);
    a3->m128i_i64[0] = *(_QWORD *)a4;
    a3->m128i_i32[2] = *(_DWORD *)(a4 + 8);
    *(_QWORD *)a4 = v40.m128i_i64[0];
    *(_DWORD *)(a4 + 8) = v40.m128i_i32[2];
  }
  sub_2013400(a1, a2, 1, (__int64)v37, v56, v39);
  if ( v57 )
    sub_161E7C0((__int64)&v57, v57);
}
