// Function: sub_1AEE6A0
// Address: 0x1aee6a0
//
__int64 __fastcall sub_1AEE6A0(
        __int64 a1,
        char a2,
        char a3,
        __int64 a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  __int64 v13; // rbx
  unsigned __int64 v14; // rax
  unsigned int v15; // eax
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // r15
  unsigned int i; // r13d
  __int64 v19; // r12
  const __m128i *v20; // rsi
  __int64 v21; // r13
  _QWORD *v22; // rdi
  __int64 v23; // rbx
  unsigned int v24; // r14d
  _QWORD *v25; // r13
  __int64 v26; // rax
  double v27; // xmm4_8
  double v28; // xmm5_8
  _QWORD *v29; // r15
  unsigned __int64 *v30; // rcx
  unsigned __int64 v31; // rdx
  double v32; // xmm4_8
  double v33; // xmm5_8
  _QWORD *v35; // r15
  __m128i *v36; // rax
  __m128i *v37; // r13
  __m128i *v38; // r15
  __int64 v39; // rsi
  __int64 v40; // rsi
  unsigned __int8 *v41; // rsi
  int v44; // [rsp+14h] [rbp-7Ch]
  const __m128i *v46; // [rsp+20h] [rbp-70h] BYREF
  __m128i *v47; // [rsp+28h] [rbp-68h]
  const __m128i *v48; // [rsp+30h] [rbp-60h]
  __m128i v49; // [rsp+40h] [rbp-50h] BYREF
  __int16 v50; // [rsp+50h] [rbp-40h]

  v13 = *(_QWORD *)(a1 + 40);
  v46 = 0;
  v47 = 0;
  v48 = 0;
  if ( a4 )
  {
    v14 = sub_157EBA0(v13);
    v15 = sub_15F4D60(v14);
    sub_1953AE0(&v46, v15);
  }
  v16 = sub_157EBA0(v13);
  if ( v16 )
  {
    v44 = sub_15F4D60(v16);
    v17 = sub_157EBA0(v13);
    if ( v44 )
    {
      for ( i = 0; i != v44; ++i )
      {
        v19 = sub_15F4DF0(v17, i);
        sub_157F2D0(v19, v13, a3);
        if ( a4 )
        {
          v49.m128i_i64[0] = v13;
          v20 = v47;
          v49.m128i_i64[1] = v19 | 4;
          if ( v47 == v48 )
          {
            sub_17F2860(&v46, v47, &v49);
          }
          else
          {
            if ( v47 )
            {
              a5 = (__m128)_mm_loadu_si128(&v49);
              *v47 = (__m128i)a5;
              v20 = v47;
            }
            v47 = (__m128i *)&v20[1];
          }
        }
      }
    }
  }
  if ( a2 )
  {
    v35 = (_QWORD *)sub_15E26F0(*(__int64 **)(*(_QWORD *)(v13 + 56) + 40LL), 205, 0, 0);
    v50 = 257;
    v36 = (__m128i *)sub_1648A60(72, 1u);
    v37 = v36;
    if ( v36 )
      sub_15F5ED0((__int64)v36, v35, (__int64)&v49, a1);
    v38 = v37 + 3;
    v39 = *(_QWORD *)(a1 + 48);
    v49.m128i_i64[0] = v39;
    if ( v39 )
    {
      sub_1623A60((__int64)&v49, v39, 2);
      if ( v38 == &v49 )
      {
        if ( v49.m128i_i64[0] )
          sub_161E7C0((__int64)&v49, v49.m128i_i64[0]);
        goto LABEL_13;
      }
      v40 = v37[3].m128i_i64[0];
      if ( !v40 )
        goto LABEL_37;
    }
    else
    {
      if ( v38 == &v49 )
        goto LABEL_13;
      v40 = v37[3].m128i_i64[0];
      if ( !v40 )
        goto LABEL_13;
    }
    sub_161E7C0((__int64)v37[3].m128i_i64, v40);
LABEL_37:
    v41 = (unsigned __int8 *)v49.m128i_i64[0];
    v37[3].m128i_i64[0] = v49.m128i_i64[0];
    if ( v41 )
      sub_1623210((__int64)&v49, v41, (__int64)v37[3].m128i_i64);
  }
LABEL_13:
  v21 = sub_16498A0(a1);
  v22 = sub_1648A60(56, 0);
  if ( v22 )
    sub_15F82A0((__int64)v22, v21, a1);
  v23 = v13 + 40;
  v24 = 0;
  v25 = (_QWORD *)(a1 + 24);
  if ( a1 + 24 != v23 )
  {
    while ( 1 )
    {
      if ( *(v25 - 2) )
      {
        v26 = sub_1599EF0((__int64 **)*(v25 - 3));
        sub_164D160((__int64)(v25 - 3), v26, a5, a6, a7, a8, v27, v28, a11, a12);
      }
      v29 = (_QWORD *)v25[1];
      ++v24;
      sub_157EA20(v23, (__int64)(v25 - 3));
      v30 = (unsigned __int64 *)v25[1];
      v31 = *v25 & 0xFFFFFFFFFFFFFFF8LL;
      *v30 = v31 | *v30 & 7;
      *(_QWORD *)(v31 + 8) = v30;
      *v25 &= 7uLL;
      v25[1] = 0;
      sub_164BEC0((__int64)(v25 - 3), (__int64)(v25 - 3), v31, (__int64)v30, a5, a6, a7, a8, v32, v33, a11, a12);
      if ( (_QWORD *)v23 == v29 )
        break;
      if ( !v29 )
        BUG();
      v25 = v29;
    }
  }
  if ( a4 )
    sub_15CD9D0(a4, v46->m128i_i64, v47 - v46);
  if ( v46 )
    j_j___libc_free_0(v46, (char *)v48 - (char *)v46);
  return v24;
}
