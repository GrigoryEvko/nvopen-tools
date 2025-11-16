// Function: sub_1A1B460
// Address: 0x1a1b460
//
void __fastcall sub_1A1B460(char *a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5)
{
  signed __int64 v5; // rbx
  char *v6; // r15
  __int64 *v7; // r11
  signed __int64 v8; // r12
  __int64 v9; // r13
  const __m128i *v10; // rax
  const __m128i *v11; // r11
  const __m128i *v12; // r10
  const __m128i *v13; // r9
  __int64 v14; // r14
  __int64 *v15; // rax
  __m128i *v16; // r9
  const __m128i *v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rdi
  __int64 v20; // rdx
  unsigned __int64 v21; // rdx
  __m128i *v23; // [rsp+8h] [rbp-48h]
  unsigned __int64 v24; // [rsp+10h] [rbp-40h]
  const __m128i *v25; // [rsp+18h] [rbp-38h]
  unsigned __int64 *v26; // [rsp+18h] [rbp-38h]

  if ( a5 )
  {
    v5 = a4;
    if ( a4 )
    {
      v6 = a1;
      v7 = (__int64 *)a2;
      v8 = a5;
      if ( a5 + a4 == 2 )
      {
        v15 = (__int64 *)a1;
        v16 = a2;
LABEL_12:
        v18 = *v15;
        if ( v16->m128i_i64[0] < (unsigned __int64)*v15 )
        {
          v19 = v15[2];
          v21 = v15[1];
        }
        else
        {
          if ( v16->m128i_i64[0] > (unsigned __int64)*v15 )
            return;
          v19 = v15[2];
          v20 = (v16[1].m128i_i64[0] >> 2) & 1;
          if ( (_BYTE)v20 == ((v19 >> 2) & 1) )
          {
            v21 = v15[1];
            if ( v16->m128i_i64[1] <= v21 )
              return;
          }
          else
          {
            if ( (_BYTE)v20 )
              return;
            v21 = v15[1];
          }
        }
        *(__m128i *)v15 = _mm_loadu_si128(v16);
        v15[2] = v16[1].m128i_i64[0];
        v16->m128i_i64[0] = v18;
        v16->m128i_i64[1] = v21;
        v16[1].m128i_i64[0] = v19;
        return;
      }
      if ( a5 >= a4 )
        goto LABEL_10;
LABEL_5:
      v9 = v5 / 2;
      v10 = (const __m128i *)sub_1A1AD80(
                               v7,
                               a3,
                               (unsigned __int64 *)&v6[8 * (v5 / 2)
                                                     + 8 * ((v5 + ((unsigned __int64)v5 >> 63)) & 0xFFFFFFFFFFFFFFFELL)]);
      v12 = (const __m128i *)&v6[8 * (v5 / 2) + 8 * ((v5 + ((unsigned __int64)v5 >> 63)) & 0xFFFFFFFFFFFFFFFELL)];
      v13 = v10;
      v14 = 0xAAAAAAAAAAAAAAABLL * (((char *)v10 - (char *)v11) >> 3);
      while ( 1 )
      {
        v23 = (__m128i *)v13;
        v25 = v12;
        v8 -= v14;
        v24 = sub_1A1A4F0(v12, v11, v13);
        sub_1A1B460(v6, v25, v24, v9, v14);
        v5 -= v9;
        if ( !v5 )
          break;
        v15 = (__int64 *)v24;
        v16 = v23;
        if ( !v8 )
          break;
        if ( v8 + v5 == 2 )
          goto LABEL_12;
        v6 = (char *)v24;
        v7 = (__int64 *)v23;
        if ( v8 < v5 )
          goto LABEL_5;
LABEL_10:
        v14 = v8 / 2;
        v26 = (unsigned __int64 *)&v7[v8 / 2 + ((v8 + ((unsigned __int64)v8 >> 63)) & 0xFFFFFFFFFFFFFFFELL)];
        v17 = (const __m128i *)sub_1A1ACD0(v6, (__int64)v7, v26);
        v13 = (const __m128i *)v26;
        v12 = v17;
        v9 = 0xAAAAAAAAAAAAAAABLL * (((char *)v17 - v6) >> 3);
      }
    }
  }
}
