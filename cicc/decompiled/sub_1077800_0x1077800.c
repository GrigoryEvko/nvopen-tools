// Function: sub_1077800
// Address: 0x1077800
//
void __fastcall sub_1077800(char *src, __m128i *a2, const __m128i *a3)
{
  __int64 v3; // rbx
  char *v4; // r12
  char *v5; // rdi
  __int64 v6; // r12
  __int64 v7; // rbx
  char *v8; // r8
  const __m128i *v9; // r15
  __int64 v10; // r14
  __int64 v11; // r12
  const __m128i *v12; // rdi
  signed __int64 v13; // rax
  char *v14; // r8
  const __m128i *v15; // r14
  __int64 v16; // r15
  __int64 v17; // rbx
  const __m128i *v18; // rdi
  signed __int64 v19; // rax
  __int64 v20; // [rsp+0h] [rbp-60h]
  signed __int64 v21; // [rsp+0h] [rbp-60h]
  __int64 v24; // [rsp+18h] [rbp-48h]
  const __m128i *v25; // [rsp+28h] [rbp-38h]

  v3 = (char *)a2 - src;
  v25 = (const __m128i *)((char *)a3 + (char *)a2 - src);
  v24 = 0xCCCCCCCCCCCCCCCDLL * (((char *)a2 - src) >> 3);
  if ( (char *)a2 - src <= 240 )
  {
    sub_10776B0(src, a2->m128i_i8);
  }
  else
  {
    v4 = src;
    do
    {
      v5 = v4;
      v4 += 280;
      sub_10776B0(v5, v4);
    }
    while ( (char *)a2 - v4 > 240 );
    sub_10776B0(v4, a2->m128i_i8);
    if ( v3 > 280 )
    {
      v6 = 7;
      while ( 1 )
      {
        v7 = 2 * v6;
        if ( v24 < 2 * v6 )
        {
          v8 = (char *)a3;
          v13 = v24;
          v9 = (const __m128i *)src;
        }
        else
        {
          v8 = (char *)a3;
          v9 = (const __m128i *)src;
          v20 = v6;
          v10 = 80 * v6;
          v11 = 40 * v6;
          do
          {
            v12 = v9;
            v9 = (const __m128i *)((char *)v9 + v10);
            v8 = sub_1076E90(
                   v12,
                   (const __m128i *)((char *)v9 + v11 - v10),
                   (const __m128i *)((char *)v9 + v11 - v10),
                   v9,
                   v8);
            v13 = 0xCCCCCCCCCCCCCCCDLL * (((char *)a2 - (char *)v9) >> 3);
          }
          while ( v7 <= v13 );
          v6 = v20;
        }
        if ( v6 <= v13 )
          v13 = v6;
        v6 *= 4;
        sub_1076E90(v9, (const __m128i *)((char *)v9 + 40 * v13), (const __m128i *)((char *)v9 + 40 * v13), a2, v8);
        v14 = src;
        if ( v24 < v6 )
          break;
        v21 = v7;
        v15 = a3;
        v16 = 40 * v6;
        v17 = 40 * v7;
        do
        {
          v18 = v15;
          v15 = (const __m128i *)((char *)v15 + v16);
          v14 = sub_10775C0(
                  v18,
                  (const __m128i *)((char *)v15 + v17 - v16),
                  (const __m128i *)((char *)v15 + v17 - v16),
                  v15,
                  v14);
          v19 = 0xCCCCCCCCCCCCCCCDLL * (((char *)v25 - (char *)v15) >> 3);
        }
        while ( v6 <= v19 );
        if ( v19 > v21 )
          v19 = v21;
        sub_10775C0(v15, (const __m128i *)((char *)v15 + 40 * v19), (const __m128i *)((char *)v15 + 40 * v19), v25, v14);
        if ( v24 <= v6 )
          return;
      }
      if ( v24 <= v7 )
        v7 = v24;
      sub_10775C0(a3, (const __m128i *)((char *)a3 + 40 * v7), (const __m128i *)((char *)a3 + 40 * v7), v25, src);
    }
  }
}
