// Function: sub_2579B00
// Address: 0x2579b00
//
char __fastcall sub_2579B00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  const __m128i *v8; // rbx
  __int64 v9; // rsi
  unsigned __int64 v10; // rdi
  __int64 v11; // rcx
  __m128i *v12; // r8
  signed __int64 v13; // rdx
  __int64 v14; // r9
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // rax
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  __m128i *v21; // rax
  __int64 v22; // r9
  __int64 v23; // rax
  __int64 v24; // rdi
  __int8 *v25; // rbx
  _BYTE v26[80]; // [rsp+0h] [rbp-50h] BYREF

  v6 = a1 + 24;
  v8 = (const __m128i *)a2;
  if ( !*(_DWORD *)(a1 + 40) )
  {
    v9 = *(unsigned int *)(a1 + 64);
    v10 = *(_QWORD *)(a1 + 56);
    v11 = *(unsigned int *)(a1 + 64);
    v12 = (__m128i *)(v10 + 24 * v9);
    v13 = 0xAAAAAAAAAAAAAAABLL * ((24 * v9) >> 3);
    if ( v13 >> 2 )
    {
      v14 = 3 * (v13 >> 2);
      v15 = v8->m128i_i64[0];
      v16 = v10;
      a6 = v10 + 32 * v14;
      while ( *(_QWORD *)v16 != v15
           || *(_QWORD *)(v16 + 8) != v8->m128i_i64[1]
           || *(_BYTE *)(v16 + 16) != v8[1].m128i_i8[0] )
      {
        if ( v15 == *(_QWORD *)(v16 + 24)
          && *(_QWORD *)(v16 + 32) == v8->m128i_i64[1]
          && *(_BYTE *)(v16 + 40) == v8[1].m128i_i8[0] )
        {
          if ( v12 != (__m128i *)(v16 + 24) )
            goto LABEL_12;
          goto LABEL_31;
        }
        if ( v15 == *(_QWORD *)(v16 + 48)
          && *(_QWORD *)(v16 + 56) == v8->m128i_i64[1]
          && *(_BYTE *)(v16 + 64) == v8[1].m128i_i8[0] )
        {
          if ( v12 != (__m128i *)(v16 + 48) )
            goto LABEL_12;
          goto LABEL_31;
        }
        if ( v15 == *(_QWORD *)(v16 + 72)
          && *(_QWORD *)(v16 + 80) == v8->m128i_i64[1]
          && *(_BYTE *)(v16 + 88) == v8[1].m128i_i8[0] )
        {
          if ( v12 != (__m128i *)(v16 + 72) )
            goto LABEL_12;
          goto LABEL_31;
        }
        v16 += 96LL;
        if ( a6 == v16 )
        {
          a6 = 0xAAAAAAAAAAAAAAABLL;
          v13 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)v12->m128i_i64 - v16) >> 3);
          goto LABEL_28;
        }
      }
LABEL_11:
      if ( v12 != (__m128i *)v16 )
        goto LABEL_12;
      goto LABEL_31;
    }
    v16 = v10;
LABEL_28:
    switch ( v13 )
    {
      case 2LL:
        v15 = v8->m128i_i64[0];
        break;
      case 3LL:
        v15 = v8->m128i_i64[0];
        if ( *(_QWORD *)v16 == v8->m128i_i64[0] )
        {
          a6 = v8->m128i_i64[1];
          if ( *(_QWORD *)(v16 + 8) == a6 )
          {
            a6 = v8[1].m128i_u8[0];
            if ( *(_BYTE *)(v16 + 16) == (_BYTE)a6 )
              goto LABEL_11;
          }
        }
        v16 += 24LL;
        break;
      case 1LL:
        v15 = v8->m128i_i64[0];
LABEL_39:
        if ( v15 == *(_QWORD *)v16 && *(_QWORD *)(v16 + 8) == v8->m128i_i64[1] )
        {
          v15 = v8[1].m128i_u8[0];
          if ( *(_BYTE *)(v16 + 16) == (_BYTE)v15 )
            goto LABEL_11;
        }
LABEL_31:
        v15 = v9 + 1;
        if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 68) )
        {
          v22 = a1 + 56;
          v9 = a1 + 72;
          if ( v10 > (unsigned __int64)v8 || v12 <= v8 )
          {
            sub_C8D5F0(a1 + 56, (const void *)v9, v15, 0x18u, (__int64)v12, v22);
            v15 = 3LL * *(unsigned int *)(a1 + 64);
            v12 = (__m128i *)(*(_QWORD *)(a1 + 56) + 24LL * *(unsigned int *)(a1 + 64));
          }
          else
          {
            sub_C8D5F0(a1 + 56, (const void *)v9, v15, 0x18u, (__int64)v12, v22);
            v23 = *(_QWORD *)(a1 + 56);
            v15 = 3LL * *(unsigned int *)(a1 + 64);
            v8 = (const __m128i *)((char *)v8 + v23 - v10);
            v12 = (__m128i *)(v23 + 24LL * *(unsigned int *)(a1 + 64));
          }
        }
        *v12 = _mm_loadu_si128(v8);
        v12[1].m128i_i64[0] = v8[1].m128i_i64[0];
        v11 = (unsigned int)(*(_DWORD *)(a1 + 64) + 1);
        *(_DWORD *)(a1 + 64) = v11;
        if ( (unsigned int)v11 > 8 )
        {
          sub_25798B0(v6);
          v11 = *(unsigned int *)(a1 + 64);
        }
        goto LABEL_12;
      default:
        goto LABEL_31;
    }
    if ( v15 == *(_QWORD *)v16 )
    {
      a6 = v8->m128i_i64[1];
      if ( *(_QWORD *)(v16 + 8) == a6 )
      {
        a6 = v8[1].m128i_u8[0];
        if ( *(_BYTE *)(v16 + 16) == (_BYTE)a6 )
          goto LABEL_11;
      }
    }
    v16 += 24LL;
    goto LABEL_39;
  }
  v9 = a1 + 24;
  sub_2579770((__int64)v26, v6, a2);
  if ( v26[32] )
  {
    v19 = *(unsigned int *)(a1 + 64);
    v20 = *(_QWORD *)(a1 + 56);
    v12 = (__m128i *)(v19 + 1);
    if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 68) )
    {
      v24 = a1 + 56;
      v9 = a1 + 72;
      if ( v20 > (unsigned __int64)v8 || (unsigned __int64)v8 >= v20 + 24 * v19 )
      {
        sub_C8D5F0(v24, (const void *)v9, (unsigned __int64)v12, 0x18u, (__int64)v12, a6);
        v20 = *(_QWORD *)(a1 + 56);
        v19 = *(unsigned int *)(a1 + 64);
      }
      else
      {
        v25 = &v8->m128i_i8[-v20];
        sub_C8D5F0(v24, (const void *)v9, (unsigned __int64)v12, 0x18u, (__int64)v12, a6);
        v20 = *(_QWORD *)(a1 + 56);
        v19 = *(unsigned int *)(a1 + 64);
        v8 = (const __m128i *)&v25[v20];
      }
    }
    v21 = (__m128i *)(v20 + 24 * v19);
    *v21 = _mm_loadu_si128(v8);
    v15 = v8[1].m128i_u64[0];
    v21[1].m128i_i64[0] = v15;
    v11 = (unsigned int)(*(_DWORD *)(a1 + 64) + 1);
    *(_DWORD *)(a1 + 64) = v11;
  }
  else
  {
    v11 = *(unsigned int *)(a1 + 64);
  }
LABEL_12:
  if ( unk_4CDFC44 <= (unsigned int)v11 )
    return (*(__int64 (__fastcall **)(__int64, __int64, unsigned __int64, __int64, __m128i *, __int64))(*(_QWORD *)a1 + 40LL))(
             a1,
             v9,
             v15,
             v11,
             v12,
             a6);
  *(_BYTE *)(a1 + 264) &= (_DWORD)v11 == 0;
  return (_DWORD)v11 == 0;
}
