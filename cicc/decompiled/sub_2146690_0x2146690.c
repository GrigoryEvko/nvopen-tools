// Function: sub_2146690
// Address: 0x2146690
//
__int64 *__fastcall sub_2146690(__int64 *a1, __int64 a2, __m128i a3)
{
  __int64 v5; // rsi
  __int64 v6; // rsi
  unsigned __int8 *v7; // rax
  char v8; // r15
  __int64 v9; // rdi
  const __m128i *v10; // rax
  __m128i v11; // xmm1
  __int64 v12; // rax
  __m128i v13; // xmm2
  __int64 v14; // rax
  unsigned int v15; // eax
  __int64 v16; // rax
  unsigned __int64 v17; // rsi
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // rax
  char v21; // dl
  __int64 v22; // rax
  bool v23; // al
  __int64 v24; // rax
  __int64 *v25; // rdi
  __int32 v26; // edx
  __int64 v27; // r15
  __int128 v28; // rax
  __int64 *v29; // rax
  _QWORD *v30; // rdi
  unsigned int v31; // edx
  unsigned __int64 v32; // r10
  __int64 v33; // rdx
  unsigned int v34; // ecx
  unsigned __int64 v35; // r8
  int v36; // edx
  __int64 *v37; // rdi
  int v38; // edx
  __int64 *v39; // r14
  __int64 v41; // r14
  char v42; // r9
  __int64 v43; // rdx
  bool v44; // zf
  __int64 v45; // [rsp+10h] [rbp-130h]
  unsigned int v46; // [rsp+20h] [rbp-120h]
  __int64 v47; // [rsp+28h] [rbp-118h]
  __int64 v48; // [rsp+30h] [rbp-110h]
  unsigned int v49; // [rsp+38h] [rbp-108h]
  __int64 v50; // [rsp+90h] [rbp-B0h] BYREF
  int v51; // [rsp+98h] [rbp-A8h]
  char v52[8]; // [rsp+A0h] [rbp-A0h] BYREF
  __int64 v53; // [rsp+A8h] [rbp-98h]
  __m128i v54; // [rsp+B0h] [rbp-90h] BYREF
  __int128 v55; // [rsp+C0h] [rbp-80h] BYREF
  __m128i v56; // [rsp+D0h] [rbp-70h] BYREF
  __int64 v57; // [rsp+E0h] [rbp-60h]
  __int128 v58; // [rsp+F0h] [rbp-50h] BYREF
  __int64 v59; // [rsp+100h] [rbp-40h]

  v5 = *(_QWORD *)(a2 + 72);
  v50 = v5;
  if ( v5 )
    sub_1623A60((__int64)&v50, v5, 2);
  v6 = *a1;
  v51 = *(_DWORD *)(a2 + 64);
  v7 = (unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL) + 40LL)
                         + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 48LL));
  v8 = *v7;
  sub_1F40D10((__int64)&v58, v6, *(_QWORD *)(a1[1] + 48), *v7, *((_QWORD *)v7 + 1));
  v9 = *(_QWORD *)(a2 + 104);
  v52[0] = BYTE8(v58);
  v53 = v59;
  v10 = *(const __m128i **)(a2 + 32);
  v11 = _mm_loadu_si128(v10 + 5);
  v48 = v10->m128i_i64[0];
  v47 = v10->m128i_i64[1];
  v49 = sub_1E34390(v9);
  v12 = *(_QWORD *)(a2 + 104);
  v13 = _mm_loadu_si128((const __m128i *)(v12 + 40));
  v14 = *(_QWORD *)(v12 + 56);
  v56 = v13;
  v57 = v14;
  if ( v52[0] )
    v15 = sub_2143AC0(v52[0]);
  else
    v15 = sub_1F58D40((__int64)v52);
  v54.m128i_i32[2] = 0;
  v46 = v15 >> 3;
  v16 = *(_QWORD *)(a2 + 32);
  DWORD2(v55) = 0;
  v17 = *(_QWORD *)(v16 + 40);
  v18 = *(_QWORD *)(v16 + 48);
  v54.m128i_i64[0] = 0;
  v19 = *(unsigned int *)(v16 + 48);
  *(_QWORD *)&v55 = 0;
  v20 = *(_QWORD *)(v17 + 40) + 16 * v19;
  v21 = *(_BYTE *)v20;
  v22 = *(_QWORD *)(v20 + 8);
  LOBYTE(v58) = v21;
  *((_QWORD *)&v58 + 1) = v22;
  if ( v21 )
  {
    v23 = (unsigned __int8)(v21 - 14) <= 0x47u || (unsigned __int8)(v21 - 2) <= 5u;
  }
  else
  {
    v45 = v18;
    v23 = sub_1F58CF0((__int64)&v58);
    v18 = v45;
  }
  if ( v23 )
    sub_20174B0((__int64)a1, v17, v18, &v54, &v55);
  else
    sub_2016B80((__int64)a1, v17, v18, &v54, &v55);
  if ( *(_BYTE *)sub_1E0A0C0(*(_QWORD *)(a1[1] + 32)) == 1 || v8 == 13 )
  {
    a3 = _mm_loadu_si128(&v54);
    v54.m128i_i64[0] = v55;
    v54.m128i_i32[2] = DWORD2(v55);
    *(_QWORD *)&v55 = a3.m128i_i64[0];
    DWORD2(v55) = a3.m128i_i32[2];
  }
  v24 = sub_1D2BF40(
          (_QWORD *)a1[1],
          v48,
          v47,
          (__int64)&v50,
          v54.m128i_i64[0],
          v54.m128i_i64[1],
          v11.m128i_i64[0],
          v11.m128i_i64[1],
          *(_OWORD *)*(_QWORD *)(a2 + 104),
          *(_QWORD *)(*(_QWORD *)(a2 + 104) + 16LL),
          v49,
          *(unsigned __int16 *)(*(_QWORD *)(a2 + 104) + 32LL),
          (__int64)&v56);
  v25 = (__int64 *)a1[1];
  v54.m128i_i64[0] = v24;
  v54.m128i_i32[2] = v26;
  v27 = 16LL * v11.m128i_u32[2];
  *(_QWORD *)&v28 = sub_1D38BB0(
                      (__int64)v25,
                      v46,
                      (__int64)&v50,
                      *(unsigned __int8 *)(v27 + *(_QWORD *)(v11.m128i_i64[0] + 40)),
                      *(const void ***)(v27 + *(_QWORD *)(v11.m128i_i64[0] + 40) + 8),
                      0,
                      a3,
                      *(double *)v11.m128i_i64,
                      v13,
                      0);
  v29 = sub_1D332F0(
          v25,
          52,
          (__int64)&v50,
          *(unsigned __int8 *)(*(_QWORD *)(v11.m128i_i64[0] + 40) + v27),
          *(const void ***)(*(_QWORD *)(v11.m128i_i64[0] + 40) + v27 + 8),
          3u,
          *(double *)a3.m128i_i64,
          *(double *)v11.m128i_i64,
          v13,
          v11.m128i_i64[0],
          v11.m128i_u32[2],
          v28);
  v30 = (_QWORD *)a1[1];
  v32 = v31 | v11.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  v33 = *(_QWORD *)(a2 + 104);
  v34 = *(unsigned __int16 *)(v33 + 32);
  v35 = *(_QWORD *)v33 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v35 )
  {
    v41 = *(_QWORD *)(v33 + 8) + v46;
    v42 = *(_BYTE *)(v33 + 16);
    if ( (*(_QWORD *)v33 & 4) != 0 )
    {
      *((_QWORD *)&v58 + 1) = *(_QWORD *)(v33 + 8) + v46;
      LOBYTE(v59) = v42;
      *(_QWORD *)&v58 = v35 | 4;
      HIDWORD(v59) = *(_DWORD *)(v35 + 12);
    }
    else
    {
      v43 = *(_QWORD *)v35;
      *(_QWORD *)&v58 = v35;
      *((_QWORD *)&v58 + 1) = v41;
      v44 = *(_BYTE *)(v43 + 8) == 16;
      LOBYTE(v59) = v42;
      if ( v44 )
        v43 = **(_QWORD **)(v43 + 16);
      HIDWORD(v59) = *(_DWORD *)(v43 + 8) >> 8;
    }
  }
  else
  {
    v36 = *(_DWORD *)(v33 + 20);
    LODWORD(v59) = 0;
    v58 = 0u;
    HIDWORD(v59) = v36;
  }
  *(_QWORD *)&v55 = sub_1D2BF40(
                      v30,
                      v48,
                      v47,
                      (__int64)&v50,
                      v55,
                      *((__int64 *)&v55 + 1),
                      (__int64)v29,
                      v32,
                      v58,
                      v59,
                      -(v46 | v49) & (v46 | v49),
                      v34,
                      (__int64)&v56);
  v37 = (__int64 *)a1[1];
  DWORD2(v55) = v38;
  v39 = sub_1D332F0(
          v37,
          2,
          (__int64)&v50,
          1,
          0,
          0,
          *(double *)a3.m128i_i64,
          *(double *)v11.m128i_i64,
          v13,
          v54.m128i_i64[0],
          v54.m128i_u64[1],
          v55);
  if ( v50 )
    sub_161E7C0((__int64)&v50, v50);
  return v39;
}
