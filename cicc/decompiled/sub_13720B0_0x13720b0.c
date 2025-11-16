// Function: sub_13720B0
// Address: 0x13720b0
//
__int64 __fastcall sub_13720B0(_QWORD *a1)
{
  __int64 v2; // rax
  __int16 v3; // cx
  __int64 v4; // r14
  unsigned __int64 v5; // rdx
  __int64 v6; // r15
  __int64 v7; // r13
  unsigned __int64 v8; // rax
  __int16 v9; // dx
  __int16 v10; // dx
  __int64 v11; // rdx
  unsigned __int64 v12; // r14
  unsigned __int64 v13; // r15
  __int64 v14; // rax
  __int64 v15; // r12
  __int64 v16; // rdx
  __int16 v17; // ax
  __int64 v18; // rax
  __int16 v19; // r13
  int v20; // r10d
  int v21; // r10d
  __int64 v22; // rax
  __int64 *v23; // r15
  __int64 *v24; // r13
  __int64 v25; // r12
  __int64 v26; // rax
  __int64 *v27; // rax
  __int64 v28; // rdi
  __int64 v29; // rsi
  __int64 result; // rax
  __int64 *i; // r12
  __int64 *v32; // rdi
  __m128i *v33; // r12
  __int64 v34; // r15
  __int64 v35; // rax
  __m128i *v36; // r12
  __m128i *v37; // rdi
  unsigned __int64 v38; // rsi
  int v39; // esi
  unsigned int v40; // edx
  __int16 v41; // dx
  __m128i v42; // xmm0
  __int64 v43; // [rsp+8h] [rbp-D8h]
  __int64 v44; // [rsp+10h] [rbp-D0h]
  __int64 v45; // [rsp+18h] [rbp-C8h]
  int v46; // [rsp+20h] [rbp-C0h]
  __int32 v47; // [rsp+20h] [rbp-C0h]
  __int64 v48; // [rsp+28h] [rbp-B8h]
  __int64 v49; // [rsp+28h] [rbp-B8h]
  __int64 v50; // [rsp+28h] [rbp-B8h]
  __int64 v51; // [rsp+60h] [rbp-80h] BYREF
  __int16 v52; // [rsp+68h] [rbp-78h]
  unsigned __int64 v53; // [rsp+70h] [rbp-70h] BYREF
  __int64 v54; // [rsp+78h] [rbp-68h]
  __int64 v55; // [rsp+80h] [rbp-60h] BYREF
  __int16 v56; // [rsp+88h] [rbp-58h]
  __m128i v57; // [rsp+90h] [rbp-50h] BYREF
  __int64 *v58; // [rsp+A0h] [rbp-40h]
  __int64 v59; // [rsp+A8h] [rbp-38h]

  v52 = 0x3FFF;
  v2 = a1[8];
  v51 = -1;
  v53 = 0;
  LOWORD(v54) = 0;
  if ( a1[9] != v2 )
  {
    v3 = 0x3FFF;
    v4 = 0;
    v5 = -1;
    while ( 1 )
    {
      v6 = 24 * v4 + a1[1];
      if ( (int)sub_1371720(*(_QWORD *)v6, *(_WORD *)(v6 + 8), v5, v3) < 0 )
      {
        v51 = *(_QWORD *)v6;
        v52 = *(_WORD *)(v6 + 8);
      }
      v7 = a1[1] + 24 * v4;
      if ( (int)sub_1371720(v53, v54, *(_QWORD *)v7, *(_WORD *)(v7 + 8)) < 0 )
      {
        v53 = *(_QWORD *)v7;
        LOWORD(v54) = *(_WORD *)(v7 + 8);
      }
      if ( ++v4 >= 0xAAAAAAAAAAAAAAABLL * ((__int64)(a1[9] - a1[8]) >> 3) )
        break;
      v5 = v51;
      v3 = v52;
    }
  }
  v8 = sub_1371CE0(v53, v54, (__int64)&v51);
  if ( !v8 )
  {
    v55 = 0;
    v56 = 0;
LABEL_11:
    v55 = sub_1371CE0(1, 64, (__int64)&v53);
    v56 = v10;
    goto LABEL_12;
  }
  _BitScanReverse64(&v38, v8);
  v39 = v38 ^ 0x3F;
  v40 = 63 - v39 + v9;
  if ( v8 != 1LL << (63 - (unsigned __int8)v39) )
    v40 += (v8 >> (62 - (unsigned __int8)v39)) & 1;
  v55 = 0;
  v56 = 0;
  if ( v40 > 0x3D )
    goto LABEL_11;
  v57.m128i_i64[0] = v51;
  v57.m128i_i16[4] = v52;
  v57.m128i_i64[0] = sub_1371CE0(1, 0, (__int64)&v57);
  v57.m128i_i16[4] = v41;
  v42 = _mm_loadu_si128(&v57);
  v55 = v57.m128i_i64[0];
  v56 = v42.m128i_i16[4];
  sub_1371BB0((__int64)&v55, 3);
LABEL_12:
  v11 = a1[1];
  if ( a1[2] == v11 )
  {
    v44 = a1[1];
  }
  else
  {
    v12 = 0;
    do
    {
      v15 = 24 * v12;
      v16 = 24 * v12 + v11;
      v17 = *(_WORD *)(v16 + 8);
      v57.m128i_i64[0] = *(_QWORD *)v16;
      v57.m128i_i16[4] = v17;
      v18 = sub_1371E10((__int64)&v57, (unsigned __int64 *)&v55);
      v19 = *(_WORD *)(v18 + 8);
      v13 = *(_QWORD *)v18;
      v20 = sub_1371720(*(_QWORD *)v18, v19, 1u, 0);
      v14 = 1;
      if ( v20 >= 0 )
      {
        v21 = sub_1371720(v13, v19, 0xFFFFFFFFFFFFFFFFLL, 0);
        v14 = -1;
        if ( v21 < 0 )
        {
          if ( v19 <= 0 )
          {
            if ( v19 )
              v13 >>= -(char)v19;
          }
          else
          {
            v13 <<= v19;
          }
          v14 = 1;
          if ( v13 )
            v14 = v13;
        }
      }
      ++v12;
      *(_QWORD *)(a1[1] + v15 + 16) = v14;
      v11 = a1[1];
      v44 = a1[2];
    }
    while ( 0xAAAAAAAAAAAAAAABLL * ((v44 - v11) >> 3) > v12 );
  }
  v22 = a1[3];
  v23 = (__int64 *)a1[5];
  v24 = a1 + 5;
  a1[3] = 0;
  v43 = v22;
  a1[2] = 0;
  a1[1] = 0;
  v57.m128i_i64[0] = 0;
  v58 = &v57.m128i_i64[1];
  v57.m128i_i64[1] = (__int64)&v57.m128i_i64[1];
  v59 = 0;
  if ( v23 == a1 + 5 )
  {
    v27 = &v57.m128i_i64[1];
  }
  else
  {
    do
    {
      v45 = v11;
      v25 = v23[4];
      v46 = *((_DWORD *)v23 + 4);
      v48 = v23[3];
      v26 = sub_22077B0(40);
      *(_QWORD *)(v26 + 32) = v25;
      *(_DWORD *)(v26 + 16) = v46;
      *(_QWORD *)(v26 + 24) = v48;
      sub_2208C80(v26, &v57.m128i_u64[1]);
      ++v59;
      v23 = (__int64 *)*v23;
      v11 = v45;
    }
    while ( v23 != v24 );
    v27 = (__int64 *)v57.m128i_i64[1];
  }
  v49 = v11;
  v57.m128i_i64[0] = (__int64)v27;
  sub_1371990((__int64)a1);
  v28 = a1[1];
  v29 = a1[3];
  a1[2] = v44;
  result = v43;
  a1[1] = v49;
  a1[3] = v43;
  if ( v28 )
    result = j_j___libc_free_0(v28, v29 - v28);
  for ( i = (__int64 *)a1[5]; i != v24; result = j_j___libc_free_0(v32, 40) )
  {
    v32 = i;
    i = (__int64 *)*i;
  }
  v33 = (__m128i *)v57.m128i_i64[1];
  a1[6] = v24;
  a1[5] = v24;
  a1[7] = 0;
  if ( v33 == (__m128i *)&v57.m128i_u64[1] )
  {
    a1[4] = v24;
  }
  else
  {
    do
    {
      v34 = v33[2].m128i_i64[0];
      v47 = v33[1].m128i_i32[0];
      v50 = v33[1].m128i_i64[1];
      v35 = sub_22077B0(40);
      *(_QWORD *)(v35 + 32) = v34;
      *(_DWORD *)(v35 + 16) = v47;
      *(_QWORD *)(v35 + 24) = v50;
      sub_2208C80(v35, a1 + 5);
      ++a1[7];
      v33 = (__m128i *)v33->m128i_i64[0];
    }
    while ( v33 != (__m128i *)&v57.m128i_u64[1] );
    result = a1[5];
    v36 = (__m128i *)v57.m128i_i64[1];
    for ( a1[4] = result; v36 != (__m128i *)&v57.m128i_u64[1]; result = j_j___libc_free_0(v37, 40) )
    {
      v37 = v36;
      v36 = (__m128i *)v36->m128i_i64[0];
    }
  }
  return result;
}
