// Function: sub_2444670
// Address: 0x2444670
//
void __fastcall sub_2444670(__int64 *a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  __int64 v6; // r11
  __int64 v7; // r9
  __int64 v8; // rbx
  __int64 v9; // r13
  __m128i *v10; // r9
  __m128i *v11; // r10
  __int64 v12; // r11
  __m128i *v13; // r15
  __int64 v14; // r14
  __int64 *v15; // rax
  unsigned __int64 v16; // rdx
  __int64 v17; // rcx
  __m128i *v19; // [rsp+8h] [rbp-48h]
  __int64 v20; // [rsp+10h] [rbp-40h]
  __m128i *v21; // [rsp+18h] [rbp-38h]

  if ( a4 )
  {
    v5 = a5;
    if ( a5 )
    {
      v6 = (__int64)a1;
      v7 = (__int64)a2;
      v8 = a4;
      if ( a4 + a5 == 2 )
      {
        v15 = a1;
        v13 = a2;
LABEL_12:
        v16 = v15[1];
        if ( v13->m128i_i64[1] > v16 )
        {
          v17 = *v15;
          *(__m128i *)v15 = _mm_loadu_si128(v13);
          v13->m128i_i64[0] = v17;
          v13->m128i_i64[1] = v16;
        }
      }
      else
      {
        if ( a4 <= a5 )
          goto LABEL_10;
LABEL_5:
        v9 = v8 / 2;
        v13 = (__m128i *)sub_2444260(v7, a3, v6 + 16 * (v8 / 2));
        v14 = v13 - v10;
        while ( 1 )
        {
          v20 = v12;
          v21 = v11;
          v5 -= v14;
          v19 = sub_2444480(v11, v10, v13);
          sub_2444670(v20, v21, v19, v9, v14);
          v8 -= v9;
          if ( !v8 )
            break;
          v15 = (__int64 *)v19;
          if ( !v5 )
            break;
          if ( v5 + v8 == 2 )
            goto LABEL_12;
          v6 = (__int64)v19;
          v7 = (__int64)v13;
          if ( v8 > v5 )
            goto LABEL_5;
LABEL_10:
          v14 = v5 / 2;
          v13 = (__m128i *)(v7 + 16 * (v5 / 2));
          v11 = (__m128i *)sub_2444210(v6, v7, (__int64)v13);
          v9 = ((__int64)v11->m128i_i64 - v12) >> 4;
        }
      }
    }
  }
}
