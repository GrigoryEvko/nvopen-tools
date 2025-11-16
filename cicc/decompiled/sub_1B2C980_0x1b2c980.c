// Function: sub_1B2C980
// Address: 0x1b2c980
//
void __fastcall sub_1B2C980(__int64 a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  signed __int64 v6; // r14
  __int64 v7; // r11
  __int64 v8; // r10
  signed __int64 v9; // r13
  __int64 v10; // rcx
  __m128i *v11; // r12
  __int64 v12; // rax
  const __m128i *v13; // r10
  __int64 v14; // r11
  __m128i *v15; // rbx
  __int64 v16; // r15
  __int64 v17; // rdx
  __int64 v18; // rax
  __int32 v19; // r8d
  __int64 v20; // r9
  __int64 v21; // rdi
  __int64 v22; // rsi
  __int64 v23; // rcx
  __int8 v24; // al
  __int64 v26; // [rsp+10h] [rbp-50h]
  __m128i *v27; // [rsp+10h] [rbp-50h]
  const __m128i *v28; // [rsp+18h] [rbp-48h]
  __int64 v29; // [rsp+18h] [rbp-48h]
  const __m128i *v30; // [rsp+18h] [rbp-48h]
  __int64 v31; // [rsp+20h] [rbp-40h]
  __int64 v32; // [rsp+20h] [rbp-40h]
  __int64 v33; // [rsp+20h] [rbp-40h]
  __int64 v34[7]; // [rsp+28h] [rbp-38h] BYREF

  v34[0] = a6;
  if ( a4 )
  {
    v6 = a5;
    if ( a5 )
    {
      v7 = a1;
      v8 = (__int64)a2;
      v9 = a4;
      if ( a5 + a4 == 2 )
      {
        v15 = a2;
        v17 = a1;
LABEL_12:
        v33 = v17;
        if ( sub_1B2B020(v34, (__int64)v15, v17) )
        {
          v19 = *(_DWORD *)(v33 + 8);
          v20 = *(_QWORD *)v33;
          v21 = *(_QWORD *)(v33 + 16);
          *(__m128i *)v33 = _mm_loadu_si128(v15);
          v22 = *(_QWORD *)(v33 + 24);
          v23 = *(_QWORD *)(v33 + 32);
          v24 = *(_BYTE *)(v33 + 40);
          *(__m128i *)(v33 + 16) = _mm_loadu_si128(v15 + 1);
          *(__m128i *)(v33 + 32) = _mm_loadu_si128(v15 + 2);
          v15->m128i_i64[0] = v20;
          v15->m128i_i32[2] = v19;
          v15[1].m128i_i64[0] = v21;
          v15[1].m128i_i64[1] = v22;
          v15[2].m128i_i64[0] = v23;
          v15[2].m128i_i8[8] = v24;
        }
      }
      else
      {
        v10 = v34[0];
        if ( v9 <= a5 )
          goto LABEL_10;
LABEL_5:
        v26 = v7;
        v28 = (const __m128i *)v8;
        v31 = v9 / 2;
        v11 = (__m128i *)(v7 + 16 * (v9 / 2 + ((v9 + ((unsigned __int64)v9 >> 63)) & 0xFFFFFFFFFFFFFFFELL)));
        v12 = sub_1B2C840(v8, a3, (__int64)v11, v10);
        v13 = v28;
        v14 = v26;
        v15 = (__m128i *)v12;
        v16 = 0xAAAAAAAAAAAAAAABLL * ((v12 - (__int64)v28) >> 4);
        while ( 1 )
        {
          v29 = v14;
          v27 = sub_1B29FB0(v11, v13, v15);
          v6 -= v16;
          sub_1B2C980(v29, v11, v27, v31, v16, v34[0]);
          v9 -= v31;
          if ( !v9 )
            break;
          v17 = (__int64)v27;
          if ( !v6 )
            break;
          if ( v6 + v9 == 2 )
            goto LABEL_12;
          v10 = v34[0];
          v8 = (__int64)v15;
          v7 = (__int64)v27;
          if ( v9 > v6 )
            goto LABEL_5;
LABEL_10:
          v30 = (const __m128i *)v8;
          v32 = v7;
          v16 = v6 / 2;
          v15 = (__m128i *)(v8 + 16 * (v6 / 2 + ((v6 + ((unsigned __int64)v6 >> 63)) & 0xFFFFFFFFFFFFFFFELL)));
          v18 = sub_1B2C8E0(v7, v8, (__int64)v15, v10);
          v14 = v32;
          v13 = v30;
          v11 = (__m128i *)v18;
          v31 = 0xAAAAAAAAAAAAAAABLL * ((v18 - v32) >> 4);
        }
      }
    }
  }
}
