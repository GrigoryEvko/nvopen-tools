// Function: sub_1803770
// Address: 0x1803770
//
__int64 __fastcall sub_1803770(__m128i *a1, __int64 *a2)
{
  __int64 v3; // rax
  int v4; // eax
  _QWORD *v5; // rdi
  unsigned __int64 *v6; // rdi
  __int64 v7; // rdx
  size_t v8; // rcx
  __int64 v9; // rsi
  unsigned __int64 *v10; // rdi
  __m128i v11; // xmm0
  size_t v13; // rdx
  __m128i v14; // [rsp+0h] [rbp-A0h] BYREF
  __int64 v15; // [rsp+10h] [rbp-90h]
  _QWORD *v16; // [rsp+20h] [rbp-80h] BYREF
  __int16 v17; // [rsp+30h] [rbp-70h]
  unsigned __int64 *v18; // [rsp+40h] [rbp-60h] BYREF
  size_t n; // [rsp+48h] [rbp-58h]
  _QWORD src[10]; // [rsp+50h] [rbp-50h] BYREF

  sub_1803290((__int64)&a1[45].m128i_i64[1], (__int64)a2);
  a1[10].m128i_i64[0] = *a2;
  v3 = sub_1632FA0((__int64)a2);
  v4 = sub_15A9520(v3, 0);
  v5 = (_QWORD *)a1[10].m128i_i64[0];
  a1[14].m128i_i32[0] = 8 * v4;
  v16 = a2 + 30;
  a1[14].m128i_i64[1] = sub_1644C60(v5, 8 * v4);
  v17 = 260;
  sub_16E1010((__int64)&v18, (__int64)&v16);
  v6 = (unsigned __int64 *)a1[10].m128i_i64[1];
  if ( v18 == src )
  {
    v13 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)v6 = src[0];
      else
        memcpy(v6, src, n);
      v13 = n;
      v6 = (unsigned __int64 *)a1[10].m128i_i64[1];
    }
    a1[11].m128i_i64[0] = v13;
    *((_BYTE *)v6 + v13) = 0;
    v6 = v18;
  }
  else
  {
    v7 = src[0];
    v8 = n;
    if ( v6 == &a1[11].m128i_u64[1] )
    {
      a1[10].m128i_i64[1] = (__int64)v18;
      a1[11].m128i_i64[0] = v8;
      a1[11].m128i_i64[1] = v7;
    }
    else
    {
      v9 = a1[11].m128i_i64[1];
      a1[10].m128i_i64[1] = (__int64)v18;
      a1[11].m128i_i64[0] = v8;
      a1[11].m128i_i64[1] = v7;
      if ( v6 )
      {
        v18 = v6;
        src[0] = v9;
        goto LABEL_5;
      }
    }
    v18 = src;
    v6 = src;
  }
LABEL_5:
  n = 0;
  *(_BYTE *)v6 = 0;
  v10 = v18;
  a1[12].m128i_i64[1] = src[2];
  a1[13].m128i_i64[0] = src[3];
  a1[13].m128i_i64[1] = src[4];
  if ( v10 != src )
    j_j___libc_free_0(v10, src[0] + 1LL);
  sub_1802D00((__int64)&v14, &a1[10].m128i_i32[2], a1[14].m128i_i32[0], a1[14].m128i_i8[4]);
  v11 = _mm_loadu_si128(&v14);
  a1[16].m128i_i64[0] = v15;
  a1[15] = v11;
  return 1;
}
