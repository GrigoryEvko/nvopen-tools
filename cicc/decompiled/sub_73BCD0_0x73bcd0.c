// Function: sub_73BCD0
// Address: 0x73bcd0
//
void __fastcall sub_73BCD0(const __m128i *a1, __m128i *a2, int a3)
{
  __int8 v3; // r12
  __int64 v4; // rbx
  __int64 i; // r12
  __int8 v6; // al
  unsigned __int8 v7; // bl
  __m128i *v8; // r12
  const __m128i *v9; // rax
  __m128i *v10; // r12
  const __m128i *v11; // rax
  const __m128i *v12; // r13
  int v13; // ebx
  __m128i *v14; // r14
  __int8 v15; // al
  __int64 v16; // rdi
  _QWORD *v17; // rbx
  char v18; // dl
  __m128i *v20; // [rsp+10h] [rbp-40h]
  _QWORD *m128i_i64; // [rsp+18h] [rbp-38h]
  int v22; // [rsp+18h] [rbp-38h]

  v3 = a1[8].m128i_i8[12];
  v4 = a2[7].m128i_i64[0];
  if ( v3 == 7 )
  {
    v10 = (__m128i *)a2[10].m128i_i64[1];
    *a2 = _mm_loadu_si128(a1);
    a2[1] = _mm_loadu_si128(a1 + 1);
    a2[2] = _mm_loadu_si128(a1 + 2);
    a2[3] = _mm_loadu_si128(a1 + 3);
    a2[4] = _mm_loadu_si128(a1 + 4);
    a2[5] = _mm_loadu_si128(a1 + 5);
    a2[6] = _mm_loadu_si128(a1 + 6);
    a2[7] = _mm_loadu_si128(a1 + 7);
    a2[8] = _mm_loadu_si128(a1 + 8);
    a2[9] = _mm_loadu_si128(a1 + 9);
    a2[10] = _mm_loadu_si128(a1 + 10);
    a2[11] = _mm_loadu_si128(a1 + 11);
    sub_7258B0((__int64)a2);
    a2[5].m128i_i8[8] &= ~8u;
    a2[7].m128i_i64[0] = v4;
    a2[7].m128i_i64[1] = 0;
    a2[4].m128i_i64[1] = 0;
    v11 = (const __m128i *)a1[10].m128i_i64[1];
    *v10 = _mm_loadu_si128(v11);
    v10[1] = _mm_loadu_si128(v11 + 1);
    v10[2] = _mm_loadu_si128(v11 + 2);
    v10[3] = _mm_loadu_si128(v11 + 3);
    a2[10].m128i_i64[1] = (__int64)v10;
    v12 = *(const __m128i **)a1[10].m128i_i64[1];
    if ( v12 )
    {
      v20 = 0;
      v13 = 0;
      v14 = 0;
      while ( 1 )
      {
        m128i_i64 = v14->m128i_i64;
        v14 = (__m128i *)sub_72B0C0(v12->m128i_i64[1], &dword_4F077C8);
        *v14 = _mm_loadu_si128(v12);
        v14[1] = _mm_loadu_si128(v12 + 1);
        v14[2] = _mm_loadu_si128(v12 + 2);
        v14[3] = _mm_loadu_si128(v12 + 3);
        v14[4] = _mm_loadu_si128(v12 + 4);
        v14[5].m128i_i64[0] = v12[5].m128i_i64[0];
        v15 = v12[2].m128i_i8[0];
        if ( (v15 & 4) != 0 )
        {
          if ( a3 )
          {
            if ( (v15 & 0x10) != 0 )
            {
              v14[2].m128i_i64[1] = 0;
            }
            else
            {
              v16 = v12[2].m128i_i64[1];
              if ( v16 )
                v14[2].m128i_i64[1] = (__int64)sub_73BB50(v16);
            }
          }
          else
          {
            v14[2].m128i_i8[0] &= 0xEBu;
            v14[2].m128i_i64[1] = 0;
            v14[3].m128i_i64[0] = 0;
          }
        }
        v14[4].m128i_i64[0] = sub_5CF190((const __m128i *)v12[4].m128i_i64[0]);
        if ( v20 )
          *m128i_i64 = v14;
        else
          v20 = v14;
        if ( v13 == -1 )
          break;
        v12 = (const __m128i *)v12->m128i_i64[0];
        ++v13;
        if ( !v12 )
          goto LABEL_28;
      }
      v14->m128i_i64[0] = 0;
    }
    else
    {
      v20 = 0;
    }
LABEL_28:
    v10->m128i_i64[0] = (__int64)v20;
    for ( i = a2[10].m128i_i64[0]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v7 = 1;
  }
  else
  {
    if ( v3 == 12 )
    {
      v8 = (__m128i *)a2[10].m128i_i64[1];
      *a2 = _mm_loadu_si128(a1);
      a2[1] = _mm_loadu_si128(a1 + 1);
      a2[2] = _mm_loadu_si128(a1 + 2);
      a2[3] = _mm_loadu_si128(a1 + 3);
      a2[4] = _mm_loadu_si128(a1 + 4);
      a2[5] = _mm_loadu_si128(a1 + 5);
      a2[6] = _mm_loadu_si128(a1 + 6);
      a2[7] = _mm_loadu_si128(a1 + 7);
      a2[8] = _mm_loadu_si128(a1 + 8);
      a2[9] = _mm_loadu_si128(a1 + 9);
      a2[10] = _mm_loadu_si128(a1 + 10);
      a2[11] = _mm_loadu_si128(a1 + 11);
      sub_7258B0((__int64)a2);
      a2[5].m128i_i8[8] &= ~8u;
      a2[7].m128i_i64[0] = v4;
      a2[7].m128i_i64[1] = 0;
      a2[4].m128i_i64[1] = 0;
      v9 = (const __m128i *)a1[10].m128i_i64[1];
      *v8 = _mm_loadu_si128(v9);
      v8[1] = _mm_loadu_si128(v9 + 1);
      v8[2] = _mm_loadu_si128(v9 + 2);
      v8[3] = _mm_loadu_si128(v9 + 3);
      v8[4].m128i_i64[0] = v9[4].m128i_i64[0];
      return;
    }
    *a2 = _mm_loadu_si128(a1);
    a2[1] = _mm_loadu_si128(a1 + 1);
    a2[2] = _mm_loadu_si128(a1 + 2);
    a2[3] = _mm_loadu_si128(a1 + 3);
    a2[4] = _mm_loadu_si128(a1 + 4);
    a2[5] = _mm_loadu_si128(a1 + 5);
    a2[6] = _mm_loadu_si128(a1 + 6);
    a2[7] = _mm_loadu_si128(a1 + 7);
    a2[8] = _mm_loadu_si128(a1 + 8);
    a2[9] = _mm_loadu_si128(a1 + 9);
    a2[10] = _mm_loadu_si128(a1 + 10);
    a2[11] = _mm_loadu_si128(a1 + 11);
    sub_7258B0((__int64)a2);
    a2[5].m128i_i8[8] &= ~8u;
    a2[7].m128i_i64[0] = v4;
    a2[7].m128i_i64[1] = 0;
    a2[4].m128i_i64[1] = 0;
    if ( v3 != 8 )
      return;
    for ( i = sub_8D40F0(a2); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    if ( (a1[10].m128i_i8[9] & 0x10) != 0 )
    {
      v22 = dword_4F04C5C;
      v17 = sub_72D900((__int64)a1);
      v18 = *((_BYTE *)v17 + 32);
      dword_4F04C5C = dword_4F04C58;
      *((_QWORD *)sub_733470((__int64)a2, 0, v18, (_QWORD *)((char *)v17 + 36)) + 3) = v17;
      dword_4F04C5C = v22;
    }
    v6 = a2[10].m128i_i8[9];
    if ( (v6 & 4) != 0 )
    {
      sub_72DA30((__int64)a2, (__int64)a1, 4);
      v6 = a2[10].m128i_i8[9];
    }
    v7 = 2;
    if ( (v6 & 8) != 0 )
      sub_72DA30((__int64)a2, (__int64)a1, 5);
  }
  if ( (unsigned int)sub_8D23B0(i) )
  {
    if ( (unsigned __int8)(*(_BYTE *)(i + 140) - 9) <= 2u )
      sub_880320(i, v7, a2, 6, &dword_4F077C8);
  }
}
