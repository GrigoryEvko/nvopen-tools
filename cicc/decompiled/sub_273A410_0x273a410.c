// Function: sub_273A410
// Address: 0x273a410
//
char *__fastcall sub_273A410(const __m128i *src, const __m128i *a2, __int64 a3, __int64 a4, __m128i *a5)
{
  const __m128i *v7; // r13
  __int64 v8; // r12
  __m128i v10; // xmm3
  unsigned __int32 v11; // eax
  int v12; // eax
  __int32 v13; // ecx
  __m128i v14; // xmm0
  signed __int64 v15; // r8
  __int8 *v16; // rbx
  unsigned __int64 v18; // r8
  unsigned __int64 v19; // rsi
  bool v20; // al

  v7 = src;
  v8 = a3;
  if ( src != a2 && a3 != a4 )
  {
    do
    {
      v11 = v7[3].m128i_u32[0];
      if ( *(_DWORD *)(v8 + 48) == v11 )
      {
        v12 = *(_DWORD *)(v8 + 56);
        v13 = v7[3].m128i_i32[2];
        if ( v12 )
        {
          if ( !v13 )
            goto LABEL_4;
          v18 = *(_QWORD *)v8;
          if ( v12 == 3 )
            v18 = sub_2739680(*(_QWORD *)v8);
          v19 = v7->m128i_i64[0];
          if ( v13 == 3 )
            v19 = sub_2739680(v7->m128i_i64[0]);
          if ( !sub_B445A0(v18, v19) )
          {
LABEL_4:
            v10 = _mm_loadu_si128(v7);
            v7 += 4;
            a5 += 4;
            a5[-4] = v10;
            a5[-3] = _mm_loadu_si128(v7 - 3);
            a5[-2] = _mm_loadu_si128(v7 - 2);
            a5[-1].m128i_i64[0] = v7[-1].m128i_i64[0];
            a5[-1].m128i_i32[2] = v7[-1].m128i_i32[2];
            if ( v7 == a2 )
              break;
            continue;
          }
        }
        else if ( !v13 )
        {
          v20 = 0;
          if ( **(_BYTE **)(v8 + 8) != 17 )
            v20 = **(_BYTE **)(v8 + 16) != 17;
          if ( *(_BYTE *)v7->m128i_i64[1] == 17
            || (unsigned __int8)v20 >= (unsigned __int8)(*(_BYTE *)v7[1].m128i_i64[0] != 17) )
          {
            goto LABEL_4;
          }
        }
      }
      else if ( *(_DWORD *)(v8 + 48) >= v11 )
      {
        goto LABEL_4;
      }
      v14 = _mm_loadu_si128((const __m128i *)v8);
      a5 += 4;
      v8 += 64;
      a5[-4] = v14;
      a5[-3] = _mm_loadu_si128((const __m128i *)(v8 - 48));
      a5[-2] = _mm_loadu_si128((const __m128i *)(v8 - 32));
      a5[-1].m128i_i64[0] = *(_QWORD *)(v8 - 16);
      a5[-1].m128i_i32[2] = *(_DWORD *)(v8 - 8);
      if ( v7 == a2 )
        break;
    }
    while ( v8 != a4 );
  }
  v15 = (char *)a2 - (char *)v7;
  if ( a2 != v7 )
  {
    memmove(a5, v7, (char *)a2 - (char *)v7);
    v15 = (char *)a2 - (char *)v7;
  }
  v16 = &a5->m128i_i8[v15];
  if ( a4 != v8 )
    memmove(v16, (const void *)v8, a4 - v8);
  return &v16[a4 - v8];
}
