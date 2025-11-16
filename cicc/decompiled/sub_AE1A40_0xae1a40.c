// Function: sub_AE1A40
// Address: 0xae1a40
//
__int64 __fastcall sub_AE1A40(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  unsigned __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rsi
  __int64 v9; // rdi
  __int64 v10; // rdx
  unsigned __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rsi
  __int64 v14; // rdi
  __int64 v15; // rdx
  __m128i v16; // xmm0
  __m128i v17; // xmm1
  __m128i v18; // xmm2
  __m128i v19; // xmm3
  __m128i v20; // xmm4
  __m128i v21; // xmm5
  _BYTE v23[8]; // [rsp+0h] [rbp-A0h] BYREF
  __m128i v24; // [rsp+8h] [rbp-98h] BYREF
  __m128i v25; // [rsp+18h] [rbp-88h] BYREF
  __m128i v26; // [rsp+28h] [rbp-78h] BYREF
  _BYTE v27[8]; // [rsp+40h] [rbp-60h] BYREF
  __m128i v28; // [rsp+48h] [rbp-58h] BYREF
  __m128i v29; // [rsp+58h] [rbp-48h] BYREF
  __m128i v30; // [rsp+68h] [rbp-38h] BYREF

  v23[0] = a4;
  v25.m128i_i64[0] = a2;
  v25.m128i_i64[1] = a3;
  v24 = 0u;
  v26.m128i_i64[0] = (__int64)v23;
  v26.m128i_i64[1] = 1;
  v6 = sub_C931B0(&v25, v23, 1, 0);
  if ( v6 == -1 )
  {
    v6 = v25.m128i_u64[1];
    v8 = v25.m128i_i64[0];
    v9 = 0;
    v10 = 0;
  }
  else
  {
    v7 = v6 + 1;
    v8 = v25.m128i_i64[0];
    if ( v6 + 1 > v25.m128i_i64[1] )
    {
      v7 = v25.m128i_i64[1];
      v9 = 0;
    }
    else
    {
      v9 = v25.m128i_i64[1] - v7;
    }
    v10 = v25.m128i_i64[0] + v7;
    if ( v6 > v25.m128i_i64[1] )
      v6 = v25.m128i_u64[1];
  }
  v27[0] = a4;
  v24.m128i_i64[0] = v8;
  v25.m128i_i64[0] = v10;
  v25.m128i_i64[1] = v9;
  v24.m128i_i64[1] = v6;
  v28 = 0u;
  v29 = 0u;
  v30.m128i_i64[0] = (__int64)v27;
  v30.m128i_i64[1] = 1;
  v11 = sub_C931B0(&v29, v27, 1, 0);
  if ( v11 == -1 )
  {
    v11 = v29.m128i_u64[1];
    v13 = v29.m128i_i64[0];
    v14 = 0;
    v15 = 0;
  }
  else
  {
    v12 = v11 + 1;
    v13 = v29.m128i_i64[0];
    if ( v11 + 1 > v29.m128i_i64[1] )
    {
      v12 = v29.m128i_i64[1];
      v14 = 0;
    }
    else
    {
      v14 = v29.m128i_i64[1] - v12;
    }
    v15 = v29.m128i_i64[0] + v12;
    if ( v11 > v29.m128i_i64[1] )
      v11 = v29.m128i_u64[1];
  }
  v28.m128i_i64[1] = v11;
  v16 = _mm_loadu_si128(&v24);
  v17 = _mm_loadu_si128(&v25);
  v28.m128i_i64[0] = v13;
  v18 = _mm_loadu_si128(&v26);
  v29.m128i_i64[0] = v15;
  v29.m128i_i64[1] = v14;
  *(_BYTE *)a1 = v23[0];
  *(__m128i *)(a1 + 8) = v16;
  *(__m128i *)(a1 + 24) = v17;
  *(__m128i *)(a1 + 40) = v18;
  if ( (_BYTE *)v26.m128i_i64[0] == v23 )
  {
    *(_QWORD *)(a1 + 40) = a1;
    *(_QWORD *)(a1 + 48) = 1;
  }
  v19 = _mm_loadu_si128(&v28);
  v20 = _mm_loadu_si128(&v29);
  v21 = _mm_loadu_si128(&v30);
  *(_BYTE *)(a1 + 56) = v27[0];
  *(__m128i *)(a1 + 64) = v19;
  *(__m128i *)(a1 + 80) = v20;
  *(__m128i *)(a1 + 96) = v21;
  if ( (_BYTE *)v30.m128i_i64[0] == v27 )
  {
    *(_QWORD *)(a1 + 104) = 1;
    *(_QWORD *)(a1 + 96) = a1 + 56;
  }
  return a1;
}
