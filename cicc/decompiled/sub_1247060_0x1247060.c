// Function: sub_1247060
// Address: 0x1247060
//
__int64 __fastcall sub_1247060(__int64 a1)
{
  int v1; // eax
  const __m128i *v2; // rsi
  bool v3; // zf
  unsigned int v4; // r14d
  __int64 v5; // rsi
  __m128i *v6; // r13
  const __m128i *i; // r12
  __int64 v8; // rdx
  int v10; // [rsp+4h] [rbp-9Ch] BYREF
  __int64 v11; // [rsp+8h] [rbp-98h] BYREF
  const __m128i *v12; // [rsp+10h] [rbp-90h] BYREF
  __m128i *v13; // [rsp+18h] [rbp-88h]
  const __m128i *v14; // [rsp+20h] [rbp-80h]
  __m128i v15; // [rsp+30h] [rbp-70h] BYREF
  _BYTE v16[96]; // [rsp+40h] [rbp-60h] BYREF

  v1 = sub_1205200(a1 + 176);
  v14 = 0;
  *(_DWORD *)(a1 + 240) = v1;
  v12 = 0;
  v13 = 0;
  if ( v1 == 511 )
  {
    while ( 1 )
    {
      v4 = sub_122E830(a1, &v10, &v11);
      if ( (_BYTE)v4 )
        break;
      v2 = v13;
      v15.m128i_i32[0] = v10;
      v15.m128i_i64[1] = v11;
      if ( v13 == v14 )
      {
        sub_1216720(&v12, v13, &v15);
        if ( *(_DWORD *)(a1 + 240) != 511 )
          goto LABEL_9;
      }
      else
      {
        if ( v13 )
        {
          *v13 = _mm_loadu_si128(&v15);
          v2 = v13;
        }
        v3 = *(_DWORD *)(a1 + 240) == 511;
        v13 = (__m128i *)&v2[1];
        if ( !v3 )
          goto LABEL_9;
      }
    }
  }
  else
  {
LABEL_9:
    v5 = (__int64)&v11;
    v10 = -1;
    v15.m128i_i64[0] = (__int64)v16;
    v15.m128i_i64[1] = 0xC00000000LL;
    v4 = sub_1245840((_QWORD **)a1, &v11, 0, &v10, (__int64)&v15);
    if ( !(_BYTE)v4 )
    {
      v6 = v13;
      for ( i = v12; v6 != i; ++i )
      {
        v8 = i->m128i_i64[1];
        v5 = i->m128i_u32[0];
        sub_B994D0(v11, v5, v8);
      }
    }
    if ( (_BYTE *)v15.m128i_i64[0] != v16 )
      _libc_free(v15.m128i_i64[0], v5);
  }
  if ( v12 )
    j_j___libc_free_0(v12, (char *)v14 - (char *)v12);
  return v4;
}
