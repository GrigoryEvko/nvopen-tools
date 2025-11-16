// Function: sub_CA3860
// Address: 0xca3860
//
__m128i *__fastcall sub_CA3860(__m128i *a1, __int64 a2, __m128i *a3)
{
  bool v4; // zf
  __int32 v5; // eax
  __int64 v6; // rdx
  __m128i v8; // xmm3
  __int64 v9; // rax
  _OWORD v10[2]; // [rsp+10h] [rbp-200h] BYREF
  __int128 v11; // [rsp+30h] [rbp-1E0h]
  __int128 v12; // [rsp+40h] [rbp-1D0h]
  __int64 v13; // [rsp+50h] [rbp-1C0h]
  __m128i v14; // [rsp+60h] [rbp-1B0h] BYREF
  __m128i v15; // [rsp+70h] [rbp-1A0h] BYREF
  __m128i v16; // [rsp+80h] [rbp-190h] BYREF
  __int64 v17; // [rsp+90h] [rbp-180h]
  __int64 v18; // [rsp+98h] [rbp-178h]
  __int64 v19; // [rsp+A0h] [rbp-170h]
  __int64 v20; // [rsp+A8h] [rbp-168h]
  __int8 v21; // [rsp+B0h] [rbp-160h]
  __m128i v22; // [rsp+C0h] [rbp-150h] BYREF
  __int64 v23; // [rsp+D0h] [rbp-140h]
  _BYTE v24[312]; // [rsp+D8h] [rbp-138h] BYREF

  v4 = *(_BYTE *)(a2 + 328) == 0;
  v11 = 0;
  v22.m128i_i64[0] = (__int64)v24;
  v22.m128i_i64[1] = 0;
  v23 = 256;
  v13 = 0;
  HIDWORD(v11) = 0xFFFF;
  memset(v10, 0, sizeof(v10));
  v12 = 0;
  if ( v4 || (*(_BYTE *)(a2 + 320) & 1) != 0 )
  {
    v16.m128i_i64[0] = a3[2].m128i_i64[0];
    v14 = _mm_loadu_si128(a3);
    v15 = _mm_loadu_si128(a3 + 1);
  }
  else
  {
    sub_CA0EC0((__int64)a3, (__int64)&v22);
    v16.m128i_i16[0] = 261;
    v14 = *(__m128i *)(a2 + 168);
    sub_C846B0((__int64)&v14, (unsigned __int8 **)&v22);
    v16.m128i_i16[0] = 261;
    v14 = v22;
  }
  v5 = sub_C826E0((__int64)&v14, (__int64)v10, 1);
  if ( v5 )
  {
    a1[5].m128i_i8[8] |= 1u;
    a1->m128i_i32[0] = v5;
    a1->m128i_i64[1] = v6;
  }
  else
  {
    sub_CA37D0((__int64)&v14, (__int64)v10, (void **)a3);
    a1[5].m128i_i8[8] &= ~1u;
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    if ( (__m128i *)v14.m128i_i64[0] == &v15 )
    {
      a1[1] = _mm_load_si128(&v15);
    }
    else
    {
      a1->m128i_i64[0] = v14.m128i_i64[0];
      a1[1].m128i_i64[0] = v15.m128i_i64[0];
    }
    v8 = _mm_load_si128(&v16);
    a1->m128i_i64[1] = v14.m128i_i64[1];
    v9 = v17;
    a1[2] = v8;
    a1[3].m128i_i64[0] = v9;
    a1[3].m128i_i64[1] = v18;
    a1[4].m128i_i64[0] = v19;
    a1[4].m128i_i64[1] = v20;
    a1[5].m128i_i8[0] = v21;
  }
  if ( (_BYTE *)v22.m128i_i64[0] != v24 )
    _libc_free(v22.m128i_i64[0], v10);
  return a1;
}
