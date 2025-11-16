// Function: sub_2C75F20
// Address: 0x2c75f20
//
__int64 __fastcall sub_2C75F20(__int64 a1, __int64 *a2)
{
  __int64 v3; // rsi
  __int64 v4; // rsi
  _QWORD *v5; // rdi
  __int64 v6; // r8
  size_t v7; // rdx
  __int64 v9; // [rsp+10h] [rbp-80h] BYREF
  __int64 v10; // [rsp+18h] [rbp-78h] BYREF
  void *dest; // [rsp+20h] [rbp-70h] BYREF
  size_t v12; // [rsp+28h] [rbp-68h]
  __m128i v13; // [rsp+30h] [rbp-60h] BYREF
  _QWORD *v14; // [rsp+40h] [rbp-50h] BYREF
  size_t n; // [rsp+48h] [rbp-48h]
  _QWORD src[8]; // [rsp+50h] [rbp-40h] BYREF

  v3 = *a2;
  v9 = v3;
  if ( !v3 || (sub_B96E90((__int64)&v9, v3, 1), !v9) )
  {
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)a1 = a1 + 16;
    *(_BYTE *)(a1 + 16) = 0;
    return a1;
  }
  sub_2C75AD0((__int64)&dest, (__int64)&v9);
  while ( 1 )
  {
    if ( !(unsigned __int8)sub_2C75D80((__int64)dest, v12) )
    {
      *(_QWORD *)a1 = a1 + 16;
      if ( dest == &v13 )
      {
        *(__m128i *)(a1 + 16) = _mm_load_si128(&v13);
      }
      else
      {
        *(_QWORD *)a1 = dest;
        *(_QWORD *)(a1 + 16) = v13.m128i_i64[0];
      }
      *(_QWORD *)(a1 + 8) = v12;
      goto LABEL_28;
    }
    v4 = sub_B10D40((__int64)&v9);
    if ( !v4 )
      break;
    sub_B10CB0(&v10, v4);
    sub_2C75AD0((__int64)&v14, (__int64)&v10);
    v5 = dest;
    if ( v14 == src )
    {
      v7 = n;
      if ( !n )
        goto LABEL_20;
      if ( n != 1 )
      {
        memcpy(dest, src, n);
        v7 = n;
        v5 = dest;
LABEL_20:
        v12 = v7;
        *((_BYTE *)v5 + v7) = 0;
        v5 = v14;
        goto LABEL_10;
      }
      *(_BYTE *)dest = src[0];
      v12 = n;
      *((_BYTE *)dest + n) = 0;
      v5 = v14;
    }
    else
    {
      if ( dest == &v13 )
      {
        dest = v14;
        v12 = n;
        v13.m128i_i64[0] = src[0];
      }
      else
      {
        v6 = v13.m128i_i64[0];
        dest = v14;
        v12 = n;
        v13.m128i_i64[0] = src[0];
        if ( v5 )
        {
          v14 = v5;
          src[0] = v6;
          goto LABEL_10;
        }
      }
      v14 = src;
      v5 = src;
    }
LABEL_10:
    n = 0;
    *(_BYTE *)v5 = 0;
    if ( v14 != src )
      j_j___libc_free_0((unsigned __int64)v14);
    if ( v9 )
      sub_B91220((__int64)&v9, v9);
    v9 = v10;
    if ( v10 )
    {
      sub_B96E90((__int64)&v9, v10, 1);
      if ( v10 )
        sub_B91220((__int64)&v10, v10);
    }
  }
  sub_2C75AD0(a1, (__int64)&v9);
  if ( dest != &v13 )
    j_j___libc_free_0((unsigned __int64)dest);
LABEL_28:
  if ( v9 )
    sub_B91220((__int64)&v9, v9);
  return a1;
}
