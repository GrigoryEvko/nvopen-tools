// Function: sub_2028A10
// Address: 0x2028a10
//
unsigned __int64 __fastcall sub_2028A10(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __m128i a5,
        __m128i a6,
        __m128i a7)
{
  __int64 v10; // rsi
  __int64 v11; // rsi
  char *v12; // rax
  char v13; // dl
  __int64 v14; // rax
  __m128i v15; // kr00_16
  __m128i v16; // kr10_16
  unsigned __int8 *v17; // rax
  __int64 v18; // rsi
  __int64 v19; // r11
  unsigned int *v20; // r10
  int v21; // eax
  __int64 v22; // rdx
  __int64 v23; // rdx
  char v24; // cl
  __int64 v25; // rsi
  __int64 *v26; // rdi
  int v27; // edx
  unsigned int v28; // edx
  unsigned __int64 result; // rax
  int v30; // edx
  __int64 v31; // [rsp+0h] [rbp-120h]
  __int64 v32; // [rsp+8h] [rbp-118h]
  __int64 *v33; // [rsp+8h] [rbp-118h]
  __int64 v35; // [rsp+80h] [rbp-A0h] BYREF
  int v36; // [rsp+88h] [rbp-98h]
  __int64 v37; // [rsp+90h] [rbp-90h] BYREF
  int v38; // [rsp+98h] [rbp-88h]
  __m128i v39; // [rsp+A0h] [rbp-80h] BYREF
  __m128 v40; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v41; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v42; // [rsp+C8h] [rbp-58h]
  __m128i v43; // [rsp+D0h] [rbp-50h] BYREF
  __m128i v44; // [rsp+E0h] [rbp-40h] BYREF

  v10 = *(_QWORD *)(a2 + 72);
  v35 = v10;
  if ( v10 )
    sub_1623A60((__int64)&v35, v10, 2);
  v11 = *(_QWORD *)(a1 + 8);
  v36 = *(_DWORD *)(a2 + 64);
  v12 = *(char **)(a2 + 40);
  v13 = *v12;
  v14 = *((_QWORD *)v12 + 1);
  LOBYTE(v41) = v13;
  v42 = v14;
  sub_1D19A30((__int64)&v43, v11, &v41);
  v15 = v43;
  v16 = v44;
  v17 = (unsigned __int8 *)(*(_QWORD *)(**(_QWORD **)(a2 + 32) + 40LL)
                          + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 8LL));
  sub_1F40D10((__int64)&v43, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), *v17, *((_QWORD *)v17 + 1));
  if ( v43.m128i_i8[0] == 6 )
  {
    sub_2017DE0(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL), (_DWORD *)a3, (_DWORD *)a4);
  }
  else
  {
    v18 = *(_QWORD *)(a2 + 72);
    v19 = *(_QWORD *)(a1 + 8);
    v37 = v18;
    if ( v18 )
    {
      v32 = v19;
      sub_1623A60((__int64)&v37, v18, 2);
      v19 = v32;
    }
    v20 = *(unsigned int **)(a2 + 32);
    v21 = *(_DWORD *)(a2 + 64);
    v39.m128i_i8[0] = 0;
    v39.m128i_i64[1] = 0;
    v38 = v21;
    v40.m128_u64[1] = 0;
    v22 = v20[2];
    v40.m128_i8[0] = 0;
    v31 = (__int64)v20;
    v23 = *(_QWORD *)(*(_QWORD *)v20 + 40LL) + 16 * v22;
    v33 = (__int64 *)v19;
    v24 = *(_BYTE *)v23;
    v42 = *(_QWORD *)(v23 + 8);
    LOBYTE(v41) = v24;
    sub_1D19A30((__int64)&v43, v19, &v41);
    a5 = _mm_loadu_si128(&v43);
    a6 = _mm_loadu_si128(&v44);
    v39 = a5;
    v40 = (__m128)a6;
    sub_1D40600(
      (__int64)&v43,
      v33,
      v31,
      (__int64)&v37,
      (const void ***)&v39,
      (const void ***)&v40,
      a5,
      *(double *)a6.m128i_i64,
      a7);
    if ( v37 )
      sub_161E7C0((__int64)&v37, v37);
    *(_QWORD *)a3 = v43.m128i_i64[0];
    *(_DWORD *)(a3 + 8) = v43.m128i_i32[2];
    *(_QWORD *)a4 = v44.m128i_i64[0];
    *(_DWORD *)(a4 + 8) = v44.m128i_i32[2];
  }
  v25 = *(unsigned __int16 *)(a2 + 24);
  v26 = *(__int64 **)(a1 + 8);
  if ( (_DWORD)v25 == 154 )
  {
    *(_QWORD *)a3 = sub_1D332F0(
                      v26,
                      v25,
                      (__int64)&v35,
                      v15.m128i_i64[0],
                      (const void **)v15.m128i_i64[1],
                      0,
                      *(double *)a5.m128i_i64,
                      *(double *)a6.m128i_i64,
                      a7,
                      *(_QWORD *)a3,
                      *(_QWORD *)(a3 + 8),
                      *(_OWORD *)(*(_QWORD *)(a2 + 32) + 40LL));
    *(_DWORD *)(a3 + 8) = v30;
    *(_QWORD *)a4 = sub_1D332F0(
                      *(__int64 **)(a1 + 8),
                      *(unsigned __int16 *)(a2 + 24),
                      (__int64)&v35,
                      v16.m128i_i64[0],
                      (const void **)v16.m128i_i64[1],
                      0,
                      *(double *)a5.m128i_i64,
                      *(double *)a6.m128i_i64,
                      a7,
                      *(_QWORD *)a4,
                      *(_QWORD *)(a4 + 8),
                      *(_OWORD *)(*(_QWORD *)(a2 + 32) + 40LL));
  }
  else
  {
    *(_QWORD *)a3 = sub_1D309E0(
                      v26,
                      v25,
                      (__int64)&v35,
                      v15.m128i_i64[0],
                      (const void **)v15.m128i_i64[1],
                      0,
                      *(double *)a5.m128i_i64,
                      *(double *)a6.m128i_i64,
                      *(double *)a7.m128i_i64,
                      *(_OWORD *)a3);
    *(_DWORD *)(a3 + 8) = v27;
    *(_QWORD *)a4 = sub_1D309E0(
                      *(__int64 **)(a1 + 8),
                      *(unsigned __int16 *)(a2 + 24),
                      (__int64)&v35,
                      v16.m128i_i64[0],
                      (const void **)v16.m128i_i64[1],
                      0,
                      *(double *)a5.m128i_i64,
                      *(double *)a6.m128i_i64,
                      *(double *)a7.m128i_i64,
                      *(_OWORD *)a4);
  }
  result = v28;
  *(_DWORD *)(a4 + 8) = v28;
  if ( v35 )
    return sub_161E7C0((__int64)&v35, v35);
  return result;
}
