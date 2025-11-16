// Function: sub_B39B30
// Address: 0xb39b30
//
__int64 __fastcall sub_B39B30(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        int a6,
        __int64 a7,
        unsigned int a8,
        const void *a9,
        __int64 a10,
        __int128 a11,
        __int64 a12,
        __int128 a13,
        __int64 a14,
        char *a15,
        __int64 a16,
        __int64 a17)
{
  __int64 v19; // rdi
  __int64 v20; // r13
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  unsigned __int64 v24; // r10
  __int64 v25; // rax
  _QWORD *v26; // r14
  _QWORD *v27; // r13
  __int64 v28; // r15
  __int64 v29; // rdi
  __int64 v30; // rax
  __int64 v31; // rbx
  __int64 *v32; // rax
  unsigned __int64 v33; // rax
  __int64 v34; // rdi
  char *v37; // [rsp+10h] [rbp-D0h]
  __int64 v39; // [rsp+20h] [rbp-C0h]
  __m128i v41; // [rsp+30h] [rbp-B0h]
  char v42; // [rsp+40h] [rbp-A0h]
  __m128i v43; // [rsp+50h] [rbp-90h]
  char v44; // [rsp+60h] [rbp-80h]
  __int64 v45[4]; // [rsp+70h] [rbp-70h] BYREF
  _QWORD *v46; // [rsp+90h] [rbp-50h] BYREF
  _QWORD *v47; // [rsp+98h] [rbp-48h]
  __int64 v48; // [rsp+A0h] [rbp-40h]

  v39 = a16;
  v37 = a15;
  v42 = a12;
  v41 = _mm_loadu_si128((const __m128i *)&a11);
  v44 = a14;
  v43 = _mm_loadu_si128((const __m128i *)&a13);
  v19 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 48) + 72LL) + 40LL);
  v46 = *(_QWORD **)(a5 + 8);
  v20 = sub_B6E160(v19, 151, &v46, 1);
  sub_B37F20((__int64)v45, a1, a2, a3, a5, a8, a9, a10);
  sub_B39750(
    (__int64)&v46,
    v37,
    v39,
    v21,
    v22,
    v23,
    v41.m128i_i64[0],
    v41.m128i_i64[1],
    v42,
    v43.m128i_i64[0],
    v43.m128i_i64[1],
    v44);
  v24 = 0;
  if ( v20 )
    v24 = *(_QWORD *)(v20 + 24);
  v25 = sub_B33310(
          (unsigned int **)a1,
          v24,
          v20,
          a6,
          a7,
          a17,
          v45[0],
          (v45[1] - v45[0]) >> 3,
          (__int64)v46,
          0x6DB6DB6DB6DB6DB7LL * (v47 - v46));
  v26 = v47;
  v27 = v46;
  v28 = v25;
  if ( v47 != v46 )
  {
    do
    {
      v29 = v27[4];
      if ( v29 )
        j_j___libc_free_0(v29, v27[6] - v29);
      if ( (_QWORD *)*v27 != v27 + 2 )
        j_j___libc_free_0(*v27, v27[2] + 1LL);
      v27 += 7;
    }
    while ( v26 != v27 );
    v27 = v46;
  }
  if ( v27 )
    j_j___libc_free_0(v27, v48 - (_QWORD)v27);
  v30 = sub_A77D30(*(__int64 **)(a1 + 72), 82, a4);
  LODWORD(v46) = 2;
  v31 = v30;
  v32 = (__int64 *)sub_BD5C60(v28, 82);
  v33 = sub_A7B660((__int64 *)(v28 + 72), v32, &v46, 1, v31);
  v34 = v45[0];
  *(_QWORD *)(v28 + 72) = v33;
  if ( v34 )
    j_j___libc_free_0(v34, v45[2] - v34);
  return v28;
}
