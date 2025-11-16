// Function: sub_16F4F70
// Address: 0x16f4f70
//
__int64 __fastcall sub_16F4F70(__int64 a1, unsigned __int64 a2, int a3, __int64 a4)
{
  unsigned int v5; // edx
  unsigned __int64 v7; // rcx
  unsigned __int64 v8; // rax
  unsigned int v9; // edi
  size_t v10; // r9
  __m128i v11; // xmm0
  __int8 *v12; // rcx
  __int8 v13; // al
  __m128i si128; // xmm0
  __m128i v16; // xmm0
  unsigned int v17; // r9d
  __m128i v18; // [rsp+0h] [rbp-80h] BYREF
  __m128i v19; // [rsp+10h] [rbp-70h]
  __m128i v20; // [rsp+20h] [rbp-60h]
  __m128i v21; // [rsp+30h] [rbp-50h]
  __m128i v22; // [rsp+40h] [rbp-40h]
  __m128i v23; // [rsp+50h] [rbp-30h]
  __m128i v24; // [rsp+60h] [rbp-20h]
  __m128i v25; // [rsp+70h] [rbp-10h]

  v5 = 0;
  if ( *(_BYTE *)(a4 + 8) )
  {
    v7 = *(_QWORD *)a4;
    v5 = 128;
    if ( v7 <= 0x7F )
      v5 = v7;
  }
  if ( !a2 )
  {
    v9 = a3 & 0xFFFFFFFD;
    if ( (unsigned int)(a3 - 2) > 1 )
      goto LABEL_7;
    goto LABEL_16;
  }
  _BitScanReverse64(&v8, a2);
  v9 = a3 & 0xFFFFFFFD;
  v10 = (unsigned __int64)(67LL - (int)(v8 ^ 0x3F)) >> 2;
  if ( (unsigned int)(a3 - 2) <= 1 )
  {
    if ( v10 != 1 )
    {
      v17 = v10 + 2;
      if ( v17 >= v5 )
        v5 = v17;
      goto LABEL_18;
    }
LABEL_16:
    if ( v5 < 3 )
      v5 = 3;
LABEL_18:
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
  if ( v10 == 1 )
  {
LABEL_7:
    v11 = _mm_load_si128((const __m128i *)&xmmword_3F66F70);
    if ( !v5 )
      v5 = 1;
    v18 = v11;
    v19 = v11;
    v20 = v11;
    v21 = v11;
    v22 = v11;
    v23 = v11;
    v24 = v11;
    v25 = v11;
LABEL_10:
    v10 = v5;
    v12 = &v18.m128i_i8[v5];
    if ( !a2 )
      return sub_16E7EE0(a1, v18.m128i_i8, v10);
    goto LABEL_11;
  }
  v16 = _mm_load_si128((const __m128i *)&xmmword_3F66F70);
  if ( v5 >= (unsigned int)v10 )
    LODWORD(v10) = v5;
  v18 = v16;
  v10 = (unsigned int)v10;
  v19 = v16;
  v20 = v16;
  v12 = &v18.m128i_i8[(unsigned int)v10];
  v21 = v16;
  v22 = v16;
  v23 = v16;
  v24 = v16;
  v25 = v16;
  do
  {
LABEL_11:
    --v12;
    v13 = (v9 == 0 ? 55 : 87) + (a2 & 0xF);
    if ( (a2 & 0xF) <= 9 )
      v13 = (a2 & 0xF) + 48;
    a2 >>= 4;
    *v12 = v13;
  }
  while ( a2 );
  return sub_16E7EE0(a1, v18.m128i_i8, v10);
}
