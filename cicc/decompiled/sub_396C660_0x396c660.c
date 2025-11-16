// Function: sub_396C660
// Address: 0x396c660
//
void __fastcall sub_396C660(char *a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5)
{
  signed __int64 v5; // r12
  char *v6; // r15
  __m128i *v7; // r9
  signed __int64 v8; // rbx
  __int64 v9; // r13
  const __m128i *v10; // r9
  const __m128i *v11; // r10
  const __m128i *v12; // r11
  __int64 v13; // r14
  unsigned __int64 v14; // rax
  __m128i *v15; // r10
  __int32 v16; // edx
  __int64 v17; // rsi
  __int64 v18; // rcx
  __m128i *v20; // [rsp+8h] [rbp-48h]
  unsigned __int64 v21; // [rsp+10h] [rbp-40h]
  const __m128i *v22; // [rsp+18h] [rbp-38h]

  if ( a4 )
  {
    v5 = a5;
    if ( a5 )
    {
      v6 = a1;
      v7 = a2;
      v8 = a4;
      if ( a4 + a5 == 2 )
      {
        v14 = (unsigned __int64)a1;
        v15 = a2;
LABEL_12:
        v16 = *(_DWORD *)v14;
        if ( v15->m128i_i32[0] < *(_DWORD *)v14 )
        {
          v17 = *(_QWORD *)(v14 + 8);
          v18 = *(_QWORD *)(v14 + 16);
          *(__m128i *)v14 = _mm_loadu_si128(v15);
          *(_QWORD *)(v14 + 16) = v15[1].m128i_i64[0];
          v15->m128i_i32[0] = v16;
          v15->m128i_i64[1] = v17;
          v15[1].m128i_i64[0] = v18;
        }
      }
      else
      {
        if ( a4 <= a5 )
          goto LABEL_10;
LABEL_5:
        v9 = v8 / 2;
        v11 = (const __m128i *)sub_396C180(
                                 v7,
                                 a3,
                                 &v6[8 * (v8 / 2) + 8 * ((v8 + ((unsigned __int64)v8 >> 63)) & 0xFFFFFFFFFFFFFFFELL)]);
        v13 = 0xAAAAAAAAAAAAAAABLL * (((char *)v11 - (char *)v10) >> 3);
        while ( 1 )
        {
          v20 = (__m128i *)v11;
          v22 = v12;
          v5 -= v13;
          v21 = sub_396B9E0(v12, v10, v11);
          sub_396C660(v6, v22, v21, v9, v13);
          v8 -= v9;
          if ( !v8 )
            break;
          v14 = v21;
          v15 = v20;
          if ( !v5 )
            break;
          if ( v5 + v8 == 2 )
            goto LABEL_12;
          v6 = (char *)v21;
          v7 = v20;
          if ( v8 > v5 )
            goto LABEL_5;
LABEL_10:
          v13 = v5 / 2;
          v12 = (const __m128i *)sub_396C120(
                                   v6,
                                   (__int64)v7,
                                   &v7->m128i_i32[2 * (v5 / 2)
                                                + 2 * ((v5 + ((unsigned __int64)v5 >> 63)) & 0xFFFFFFFFFFFFFFFELL)]);
          v9 = 0xAAAAAAAAAAAAAAABLL * (((char *)v12 - v6) >> 3);
        }
      }
    }
  }
}
