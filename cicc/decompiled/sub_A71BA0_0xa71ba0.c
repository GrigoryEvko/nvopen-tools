// Function: sub_A71BA0
// Address: 0xa71ba0
//
__int64 __fastcall sub_A71BA0(__int64 a1, _BYTE *a2, __int64 *a3, _BYTE *a4)
{
  __int64 v4; // r14
  __int64 v6; // rax
  bool v7; // zf
  __m128i v8; // xmm0
  __m128i v9; // xmm1
  __int64 v11; // rax
  char v12; // al
  __m128i *v13; // rcx
  char v14; // dl
  __m128i v15; // xmm2
  __m128i v16; // xmm3
  __int64 v17; // [rsp+0h] [rbp-120h] BYREF
  __int64 v18; // [rsp+8h] [rbp-118h] BYREF
  __m128i v19; // [rsp+10h] [rbp-110h] BYREF
  __m128i v20; // [rsp+20h] [rbp-100h] BYREF
  __int64 v21; // [rsp+30h] [rbp-F0h]
  __m128i v22; // [rsp+40h] [rbp-E0h] BYREF
  __m128i v23; // [rsp+50h] [rbp-D0h]
  __int64 v24; // [rsp+60h] [rbp-C0h]
  __m128i v25; // [rsp+70h] [rbp-B0h]
  char v26; // [rsp+90h] [rbp-90h]
  char v27; // [rsp+91h] [rbp-8Fh]
  __m128i v28; // [rsp+A0h] [rbp-80h] BYREF
  __m128i v29; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v30; // [rsp+C0h] [rbp-60h]
  __m128i v31; // [rsp+D0h] [rbp-50h] BYREF
  __m128i v32; // [rsp+E0h] [rbp-40h]
  __int64 v33; // [rsp+F0h] [rbp-30h]

  if ( *a2 )
  {
    v6 = sub_A71B80(a3);
    v7 = *a4 == 0;
    v18 = v6;
    v28.m128i_i64[0] = (__int64)"=";
    v29.m128i_i64[0] = (__int64)&v18;
    LOWORD(v30) = 2819;
    if ( v7 )
    {
      v8 = _mm_loadu_si128(&v28);
      v9 = _mm_loadu_si128(&v29);
      v33 = v30;
      v31 = v8;
      v32 = v9;
    }
    else
    {
      v31.m128i_i64[0] = (__int64)a4;
      v32.m128i_i64[0] = (__int64)&v28;
      LOWORD(v33) = 515;
    }
  }
  else
  {
    v27 = 1;
    v25.m128i_i64[0] = (__int64)")";
    v26 = 3;
    v11 = sub_A71B80(a3);
    v7 = *a4 == 0;
    v17 = v11;
    v19.m128i_i64[0] = (__int64)"(";
    v20.m128i_i64[0] = (__int64)&v17;
    LOWORD(v21) = 2819;
    if ( v7 )
    {
      v15 = _mm_loadu_si128(&v19);
      v16 = _mm_loadu_si128(&v20);
      v24 = v21;
      v12 = v26;
      v22 = v15;
      v23 = v16;
    }
    else
    {
      v22.m128i_i64[0] = (__int64)a4;
      v23.m128i_i64[0] = (__int64)&v19;
      v12 = v26;
      LOWORD(v24) = 515;
    }
    v13 = &v22;
    v14 = 2;
    if ( BYTE1(v24) == 1 )
    {
      v4 = v22.m128i_i64[1];
      v13 = (__m128i *)v22.m128i_i64[0];
      v14 = 3;
    }
    v31.m128i_i64[0] = (__int64)v13;
    v31.m128i_i64[1] = v4;
    v32 = v25;
    LOBYTE(v33) = v14;
    BYTE1(v33) = v12;
  }
  sub_CA0F50(a1, &v31);
  return a1;
}
