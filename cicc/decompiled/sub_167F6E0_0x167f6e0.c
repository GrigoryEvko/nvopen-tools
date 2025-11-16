// Function: sub_167F6E0
// Address: 0x167f6e0
//
__int64 __fastcall sub_167F6E0(
        _QWORD *a1,
        _QWORD ***a2,
        int a3,
        __m128i *a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  void (__fastcall *v14)(__m128i *, __m128i *, __int64); // rax
  __m128i v15; // xmm0
  __int64 v16; // rdx
  __m128i v17; // xmm1
  __int64 v18; // rax
  _QWORD **v19; // rax
  double v20; // xmm4_8
  double v21; // xmm5_8
  unsigned int v22; // eax
  _QWORD **v23; // r14
  unsigned int v24; // r13d
  _QWORD *v25; // rbx
  _QWORD *v26; // r12
  __int64 v27; // rsi
  __int64 *v28; // r12
  __int64 v30; // r14
  __int64 v31; // rax
  __int64 *v32; // rbx
  __int64 v33; // r15
  _QWORD **v34; // [rsp+8h] [rbp-C8h] BYREF
  __m128i v35; // [rsp+10h] [rbp-C0h] BYREF
  void (__fastcall *v36)(__m128i *, __m128i *, __int64); // [rsp+20h] [rbp-B0h]
  __int64 v37; // [rsp+28h] [rbp-A8h]
  __int64 v38[6]; // [rsp+30h] [rbp-A0h] BYREF
  __int64 *v39; // [rsp+60h] [rbp-70h]
  int v40; // [rsp+70h] [rbp-60h]
  _QWORD *v41; // [rsp+80h] [rbp-50h]
  unsigned int v42; // [rsp+90h] [rbp-40h]

  sub_167C560((__int64)v38, a1);
  v14 = (void (__fastcall *)(__m128i *, __m128i *, __int64))a4[1].m128i_i64[0];
  v15 = _mm_loadu_si128(a4);
  v16 = v37;
  v17 = _mm_loadu_si128(&v35);
  a4[1].m128i_i64[0] = 0;
  v36 = v14;
  v18 = a4[1].m128i_i64[1];
  *a4 = v17;
  a4[1].m128i_i64[1] = v16;
  v37 = v18;
  v19 = *a2;
  *a2 = 0;
  v34 = v19;
  v35 = v15;
  v22 = sub_167DAB0(
          v38,
          (__int64 *)&v34,
          a3,
          &v35,
          *(double *)v15.m128i_i64,
          *(double *)v17.m128i_i64,
          a7,
          a8,
          v20,
          v21,
          a11,
          a12);
  v23 = v34;
  v24 = v22;
  if ( v34 )
  {
    sub_1633490(v34);
    j_j___libc_free_0(v23, 736);
  }
  if ( v36 )
    v36(&v35, &v35, 3);
  if ( v42 )
  {
    v25 = v41;
    v26 = &v41[2 * v42];
    do
    {
      if ( *v25 != -4 && *v25 != -8 )
      {
        v27 = v25[1];
        if ( v27 )
          sub_161E7C0((__int64)(v25 + 1), v27);
      }
      v25 += 2;
    }
    while ( v26 != v25 );
  }
  j___libc_free_0(v41);
  if ( v40 )
  {
    v30 = sub_16704E0();
    v31 = sub_16704F0();
    v32 = v39;
    v33 = v31;
    v28 = &v39[v40];
    if ( v39 == v28 )
      goto LABEL_14;
    do
    {
      if ( !sub_1670560(*v32, v30) )
        sub_1670560(*v32, v33);
      ++v32;
    }
    while ( v28 != v32 );
  }
  v28 = v39;
LABEL_14:
  j___libc_free_0(v28);
  j___libc_free_0(v38[2]);
  return v24;
}
