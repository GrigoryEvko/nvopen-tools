// Function: sub_B39900
// Address: 0xb39900
//
__int64 __fastcall sub_B39900(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        unsigned int a6,
        const void *a7,
        __int64 a8,
        __int128 a9,
        __int64 a10,
        __int128 a11,
        __int64 a12,
        char *a13,
        __int64 a14,
        __int64 a15)
{
  __int64 v16; // rbx
  __int64 v17; // rdi
  __int64 v18; // r15
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  unsigned __int64 v22; // rsi
  __int64 v23; // rax
  _QWORD *v24; // rbx
  _QWORD *v25; // r15
  __int64 v26; // r12
  __int64 v27; // rdi
  __int64 v28; // rax
  __int64 v29; // rbx
  __int64 *v30; // rax
  unsigned __int64 v31; // rax
  __int64 v32; // rdi
  char *v36; // [rsp+8h] [rbp-C8h]
  __m128i v39; // [rsp+20h] [rbp-B0h]
  char v40; // [rsp+30h] [rbp-A0h]
  __m128i v41; // [rsp+40h] [rbp-90h]
  char v42; // [rsp+50h] [rbp-80h]
  _QWORD v43[4]; // [rsp+60h] [rbp-70h] BYREF
  _QWORD *v44; // [rsp+80h] [rbp-50h] BYREF
  _QWORD *v45; // [rsp+88h] [rbp-48h]
  __int64 v46; // [rsp+90h] [rbp-40h]

  v40 = a10;
  v36 = a13;
  v16 = a14;
  v42 = a12;
  v39 = _mm_loadu_si128((const __m128i *)&a9);
  v41 = _mm_loadu_si128((const __m128i *)&a11);
  v17 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 48) + 72LL) + 40LL);
  v44 = *(_QWORD **)(a5 + 8);
  v18 = sub_B6E160(v17, 151, &v44, 1);
  sub_B37F20((__int64)v43, a1, a2, a3, a5, a6, a7, a8);
  sub_B39750(
    (__int64)&v44,
    v36,
    v16,
    v19,
    v20,
    v21,
    v39.m128i_i64[0],
    v39.m128i_i64[1],
    v40,
    v41.m128i_i64[0],
    v41.m128i_i64[1],
    v42);
  v22 = 0;
  if ( v18 )
    v22 = *(_QWORD *)(v18 + 24);
  v23 = sub_B33530(
          (unsigned int **)a1,
          v22,
          v18,
          v43[0],
          (__int64)(v43[1] - v43[0]) >> 3,
          a15,
          (__int64)v44,
          0x6DB6DB6DB6DB6DB7LL * (v45 - v44),
          0);
  v24 = v45;
  v25 = v44;
  v26 = v23;
  if ( v45 != v44 )
  {
    do
    {
      v27 = v25[4];
      if ( v27 )
        j_j___libc_free_0(v27, v25[6] - v27);
      if ( (_QWORD *)*v25 != v25 + 2 )
        j_j___libc_free_0(*v25, v25[2] + 1LL);
      v25 += 7;
    }
    while ( v24 != v25 );
    v25 = v44;
  }
  if ( v25 )
    j_j___libc_free_0(v25, v46 - (_QWORD)v25);
  v28 = sub_A77D30(*(__int64 **)(a1 + 72), 82, a4);
  LODWORD(v44) = 2;
  v29 = v28;
  v30 = (__int64 *)sub_BD5C60(v26, 82);
  v31 = sub_A7B660((__int64 *)(v26 + 72), v30, &v44, 1, v29);
  v32 = v43[0];
  *(_QWORD *)(v26 + 72) = v31;
  if ( v32 )
    j_j___libc_free_0(v32, v43[2] - v32);
  return v26;
}
