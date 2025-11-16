// Function: sub_CA3C80
// Address: 0xca3c80
//
__m128i *__fastcall sub_CA3C80(__int64 a1, void **a2)
{
  _BYTE *v4; // rdi
  __int64 v5; // rcx
  __int64 v6; // rdx
  __int64 v7; // rsi
  __m128i *result; // rax
  __m128i v9; // xmm0
  _QWORD *v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rdx
  _QWORD v13[2]; // [rsp+0h] [rbp-F0h] BYREF
  _QWORD v14[2]; // [rsp+10h] [rbp-E0h] BYREF
  __m128i v15; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v16; // [rsp+30h] [rbp-C0h]
  __int64 v17; // [rsp+38h] [rbp-B8h]
  __int64 v18; // [rsp+40h] [rbp-B0h]
  __int64 v19; // [rsp+48h] [rbp-A8h]
  char v20; // [rsp+50h] [rbp-A0h]
  __m128i v21; // [rsp+60h] [rbp-90h] BYREF
  _QWORD src[9]; // [rsp+70h] [rbp-80h] BYREF
  char v23; // [rsp+B8h] [rbp-38h]

  sub_CA0F50(v21.m128i_i64, a2);
  v4 = *(_BYTE **)(a1 + 104);
  if ( (_QWORD *)v21.m128i_i64[0] == src )
  {
    v12 = v21.m128i_i64[1];
    if ( v21.m128i_i64[1] )
    {
      if ( v21.m128i_i64[1] == 1 )
        *v4 = src[0];
      else
        memcpy(v4, src, v21.m128i_u64[1]);
      v12 = v21.m128i_i64[1];
      v4 = *(_BYTE **)(a1 + 104);
    }
    *(_QWORD *)(a1 + 112) = v12;
    v4[v12] = 0;
    v4 = (_BYTE *)v21.m128i_i64[0];
  }
  else
  {
    v5 = v21.m128i_i64[1];
    v6 = src[0];
    if ( v4 == (_BYTE *)(a1 + 120) )
    {
      *(_QWORD *)(a1 + 104) = v21.m128i_i64[0];
      *(_QWORD *)(a1 + 112) = v5;
      *(_QWORD *)(a1 + 120) = v6;
    }
    else
    {
      v7 = *(_QWORD *)(a1 + 120);
      *(_QWORD *)(a1 + 104) = v21.m128i_i64[0];
      *(_QWORD *)(a1 + 112) = v5;
      *(_QWORD *)(a1 + 120) = v6;
      if ( v4 )
      {
        v21.m128i_i64[0] = (__int64)v4;
        src[0] = v7;
        goto LABEL_5;
      }
    }
    v21.m128i_i64[0] = (__int64)src;
    v4 = src;
  }
LABEL_5:
  v21.m128i_i64[1] = 0;
  *v4 = 0;
  if ( (_QWORD *)v21.m128i_i64[0] != src )
    j_j___libc_free_0(v21.m128i_i64[0], src[0] + 1LL);
  result = sub_CA3AE0(&v21, a1);
  if ( (v23 & 1) == 0 )
  {
    sub_CA3780((__int64)v13, (__int64)&v21, a2);
    sub_2240D70(a1 + 16, v13);
    v9 = _mm_loadu_si128(&v15);
    v10 = (_QWORD *)v13[0];
    *(_QWORD *)(a1 + 64) = v16;
    v11 = v17;
    *(__m128i *)(a1 + 48) = v9;
    *(_QWORD *)(a1 + 72) = v11;
    *(_QWORD *)(a1 + 80) = v18;
    *(_QWORD *)(a1 + 88) = v19;
    *(_BYTE *)(a1 + 96) = v20;
    result = (__m128i *)v14;
    if ( v10 != v14 )
      result = (__m128i *)j_j___libc_free_0(v10, v14[0] + 1LL);
    if ( (v23 & 1) == 0 && (_QWORD *)v21.m128i_i64[0] != src )
      return (__m128i *)j_j___libc_free_0(v21.m128i_i64[0], src[0] + 1LL);
  }
  return result;
}
