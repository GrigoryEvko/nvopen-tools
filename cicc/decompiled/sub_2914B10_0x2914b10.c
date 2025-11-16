// Function: sub_2914B10
// Address: 0x2914b10
//
void __fastcall sub_2914B10(__m128i *a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5)
{
  signed __int64 v5; // r12
  char *v6; // r15
  __int64 *v7; // r11
  signed __int64 v8; // rbx
  __int64 v9; // r13
  const __m128i *v10; // rax
  const __m128i *v11; // r11
  const __m128i *v12; // r10
  const __m128i *v13; // r9
  __int64 v14; // r14
  __m128i *v15; // rax
  __m128i *v16; // r9
  const __m128i *v17; // rax
  __m128i v18; // xmm0
  __int64 v19; // rdx
  __int64 v20; // rdx
  __m128i *v22; // [rsp+8h] [rbp-68h]
  unsigned __int64 v23; // [rsp+10h] [rbp-60h]
  const __m128i *v24; // [rsp+18h] [rbp-58h]
  unsigned __int64 *v25; // [rsp+18h] [rbp-58h]

  if ( a4 )
  {
    v5 = a5;
    if ( a5 )
    {
      v6 = (char *)a1;
      v7 = (__int64 *)a2;
      v8 = a4;
      if ( a5 + a4 == 2 )
      {
        v15 = a1;
        v16 = a2;
LABEL_12:
        if ( v16->m128i_i64[0] >= (unsigned __int64)v15->m128i_i64[0] )
        {
          if ( v16->m128i_i64[0] > (unsigned __int64)v15->m128i_i64[0] )
            return;
          v20 = (v16[1].m128i_i64[0] >> 2) & 1;
          if ( (_BYTE)v20 == ((v15[1].m128i_i64[0] >> 2) & 1) )
          {
            if ( v16->m128i_i64[1] <= (unsigned __int64)v15->m128i_i64[1] )
              return;
          }
          else if ( (_BYTE)v20 )
          {
            return;
          }
        }
        v18 = _mm_loadu_si128(v15);
        v19 = v15[1].m128i_i64[0];
        *v15 = _mm_loadu_si128(v16);
        v15[1].m128i_i64[0] = v16[1].m128i_i64[0];
        v16[1].m128i_i64[0] = v19;
        *v16 = v18;
        return;
      }
      if ( a4 <= a5 )
        goto LABEL_10;
LABEL_5:
      v9 = v8 / 2;
      v10 = (const __m128i *)sub_2913650(
                               v7,
                               a3,
                               (unsigned __int64 *)&v6[8 * (v8 / 2)
                                                     + 8 * ((v8 + ((unsigned __int64)v8 >> 63)) & 0xFFFFFFFFFFFFFFFELL)]);
      v12 = (const __m128i *)&v6[8 * (v8 / 2) + 8 * ((v8 + ((unsigned __int64)v8 >> 63)) & 0xFFFFFFFFFFFFFFFELL)];
      v13 = v10;
      v14 = 0xAAAAAAAAAAAAAAABLL * (((char *)v10 - (char *)v11) >> 3);
      while ( 1 )
      {
        v22 = (__m128i *)v13;
        v24 = v12;
        v5 -= v14;
        v23 = sub_29133D0(v12, v11, v13);
        sub_2914B10(v6, v24, v23, v9, v14);
        v8 -= v9;
        if ( !v8 )
          break;
        v15 = (__m128i *)v23;
        v16 = v22;
        if ( !v5 )
          break;
        if ( v5 + v8 == 2 )
          goto LABEL_12;
        v6 = (char *)v23;
        v7 = (__int64 *)v22;
        if ( v8 > v5 )
          goto LABEL_5;
LABEL_10:
        v14 = v5 / 2;
        v25 = (unsigned __int64 *)&v7[v5 / 2 + ((v5 + ((unsigned __int64)v5 >> 63)) & 0xFFFFFFFFFFFFFFFELL)];
        v17 = (const __m128i *)sub_29135A0(v6, (__int64)v7, v25);
        v13 = (const __m128i *)v25;
        v12 = v17;
        v9 = 0xAAAAAAAAAAAAAAABLL * (((char *)v17 - v6) >> 3);
      }
    }
  }
}
