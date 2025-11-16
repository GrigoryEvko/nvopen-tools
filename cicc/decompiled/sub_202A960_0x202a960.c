// Function: sub_202A960
// Address: 0x202a960
//
__int64 __fastcall sub_202A960(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  unsigned __int64 *v6; // rax
  __int64 v7; // rdx
  __int32 v8; // edx
  int v9; // edx
  __int64 *v10; // r13
  __int64 *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rsi
  __int64 *v14; // r10
  unsigned __int8 *v15; // rax
  __int64 v16; // r11
  unsigned int v17; // r12d
  const void **v18; // r15
  __int64 v19; // r12
  __int128 v21; // [rsp-10h] [rbp-B0h]
  __int64 *v22; // [rsp+0h] [rbp-A0h]
  __int64 v23; // [rsp+8h] [rbp-98h]
  __m128i v24; // [rsp+40h] [rbp-60h] BYREF
  __int64 v25; // [rsp+50h] [rbp-50h] BYREF
  __int64 v26; // [rsp+58h] [rbp-48h]
  __int64 v27; // [rsp+60h] [rbp-40h] BYREF
  int v28; // [rsp+68h] [rbp-38h]

  v6 = *(unsigned __int64 **)(a2 + 32);
  v24.m128i_i32[2] = 0;
  LODWORD(v26) = 0;
  v24.m128i_i64[0] = 0;
  v7 = v6[1];
  v25 = 0;
  sub_2017DE0(a1, *v6, v7, &v24, &v25);
  v24.m128i_i64[0] = sub_200D2A0(
                       a1,
                       v24.m128i_i64[0],
                       v24.m128i_i64[1],
                       *(double *)a3.m128i_i64,
                       a4,
                       *(double *)a5.m128i_i64);
  v24.m128i_i32[2] = v8;
  v25 = sub_200D2A0(a1, v25, v26, *(double *)a3.m128i_i64, a4, *(double *)a5.m128i_i64);
  LODWORD(v26) = v9;
  if ( *(_BYTE *)sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL)) )
  {
    a3 = _mm_loadu_si128(&v24);
    v24.m128i_i64[0] = v25;
    v24.m128i_i32[2] = v26;
    v25 = a3.m128i_i64[0];
    LODWORD(v26) = a3.m128i_i32[2];
  }
  v10 = *(__int64 **)(a1 + 8);
  v11 = sub_200DAC0(a1, v24.m128i_i64[0], v24.m128i_i64[1], v25, v26, a3, a4, a5);
  v13 = *(_QWORD *)(a2 + 72);
  v14 = v11;
  v15 = *(unsigned __int8 **)(a2 + 40);
  v16 = v12;
  v17 = *v15;
  v18 = (const void **)*((_QWORD *)v15 + 1);
  v27 = v13;
  if ( v13 )
  {
    v23 = v12;
    v22 = v14;
    sub_1623A60((__int64)&v27, v13, 2);
    v14 = v22;
    v16 = v23;
  }
  *((_QWORD *)&v21 + 1) = v16;
  *(_QWORD *)&v21 = v14;
  v28 = *(_DWORD *)(a2 + 64);
  v19 = sub_1D309E0(v10, 158, (__int64)&v27, v17, v18, 0, *(double *)a3.m128i_i64, a4, *(double *)a5.m128i_i64, v21);
  if ( v27 )
    sub_161E7C0((__int64)&v27, v27);
  return v19;
}
