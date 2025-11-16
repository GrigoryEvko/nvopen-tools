// Function: sub_C7F500
// Address: 0xc7f500
//
__int64 __fastcall sub_C7F500(__int64 a1, unsigned __int64 a2, int a3, unsigned __int64 a4, char a5)
{
  unsigned int v6; // edx
  unsigned __int64 v7; // rcx
  __int64 v8; // r8
  unsigned int v9; // ecx
  __m128i v10; // xmm0
  __int8 *v11; // rax
  char v12; // cl
  unsigned __int64 v13; // rdx
  __m128i si128; // xmm0
  __m128i v16; // xmm0
  unsigned int v17; // r8d
  __m128i v18; // [rsp+10h] [rbp-80h] BYREF
  __m128i v19; // [rsp+20h] [rbp-70h]
  __m128i v20; // [rsp+30h] [rbp-60h]
  __m128i v21; // [rsp+40h] [rbp-50h]
  __m128i v22; // [rsp+50h] [rbp-40h]
  __m128i v23; // [rsp+60h] [rbp-30h]
  __m128i v24; // [rsp+70h] [rbp-20h]
  __m128i v25; // [rsp+80h] [rbp-10h]

  v6 = 0;
  if ( a5 )
  {
    v6 = 128;
    if ( a4 <= 0x7F )
      v6 = a4;
  }
  if ( !a2 )
  {
    v9 = a3 & 0xFFFFFFFD;
    if ( (unsigned int)(a3 - 2) > 1 )
      goto LABEL_7;
    goto LABEL_15;
  }
  _BitScanReverse64(&v7, a2);
  LODWORD(v8) = (int)(67 - (v7 ^ 0x3F)) >> 2;
  v9 = a3 & 0xFFFFFFFD;
  if ( (unsigned int)(a3 - 2) <= 1 )
  {
    if ( (_DWORD)v8 != 1 )
    {
      v17 = v8 + 2;
      if ( v17 >= v6 )
        v6 = v17;
      goto LABEL_17;
    }
LABEL_15:
    if ( v6 < 3 )
      v6 = 3;
LABEL_17:
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F66F70);
    v18 = si128;
    v19 = si128;
    v18.m128i_i8[1] = 120;
    v20 = si128;
    v21 = si128;
    v22 = si128;
    v23 = si128;
    v24 = si128;
    v25 = si128;
    goto LABEL_10;
  }
  if ( (_DWORD)v8 == 1 )
  {
LABEL_7:
    v10 = _mm_load_si128((const __m128i *)&xmmword_3F66F70);
    if ( !v6 )
      v6 = 1;
    v18 = v10;
    v19 = v10;
    v20 = v10;
    v21 = v10;
    v22 = v10;
    v23 = v10;
    v24 = v10;
    v25 = v10;
LABEL_10:
    v8 = v6;
    v11 = &v18.m128i_i8[v6];
    if ( !a2 )
      return sub_CB6200(a1, &v18, v8);
    goto LABEL_11;
  }
  v16 = _mm_load_si128((const __m128i *)&xmmword_3F66F70);
  if ( v6 >= (unsigned int)v8 )
    LODWORD(v8) = v6;
  v18 = v16;
  v8 = (unsigned int)v8;
  v19 = v16;
  v20 = v16;
  v11 = &v18.m128i_i8[(unsigned int)v8];
  v21 = v16;
  v22 = v16;
  v23 = v16;
  v24 = v16;
  v25 = v16;
LABEL_11:
  v12 = v9 != 0 ? 0x20 : 0;
  do
  {
    --v11;
    v13 = a2 & 0xF;
    a2 >>= 4;
    *v11 = v12 | a0123456789abcd_10[v13];
  }
  while ( a2 );
  return sub_CB6200(a1, &v18, v8);
}
