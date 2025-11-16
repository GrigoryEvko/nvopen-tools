// Function: sub_2C47B10
// Address: 0x2c47b10
//
__int64 __fastcall sub_2C47B10(__int64 *a1, __int64 a2, int a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v6; // rbx
  __int64 v7; // r14
  __int64 v8; // r9
  _BYTE *v9; // rcx
  int v10; // eax
  _BYTE *v11; // r8
  __int64 v12; // rax
  __int64 v13; // rdx
  _BYTE *v14; // rdi
  __int64 v15; // r9
  __int64 v16; // rsi
  _QWORD *v17; // rax
  int v18; // edx
  int v19; // ecx
  unsigned int v20; // r14d
  _QWORD *v21; // rax
  __m128i *v22; // rdx
  __m128i si128; // xmm0
  _BYTE *v25; // [rsp+10h] [rbp-70h] BYREF
  __int64 v26; // [rsp+18h] [rbp-68h]
  _BYTE v27[96]; // [rsp+20h] [rbp-60h] BYREF

  v6 = *(unsigned int *)(a2 + 56);
  v7 = *(_QWORD *)(a2 + 48);
  v25 = v27;
  v26 = 0x600000000LL;
  v8 = 8 * v6;
  if ( v6 > 6 )
  {
    sub_C8D5F0((__int64)&v25, v27, v6, 8u, a5, v8);
    v11 = v25;
    v10 = v26;
    v8 = 8 * v6;
    v9 = &v25[8 * (unsigned int)v26];
  }
  else
  {
    v9 = v27;
    v10 = 0;
    v11 = v27;
  }
  if ( v8 )
  {
    v12 = 0;
    do
    {
      *(_QWORD *)&v9[8 * v12] = *(_QWORD *)(v7 + 8 * v12);
      ++v12;
    }
    while ( (__int64)(v6 - v12) > 0 );
    v11 = v25;
    v10 = v26;
  }
  v13 = *a1;
  LODWORD(v26) = v6 + v10;
  v14 = &v11[8 * (unsigned int)(v6 + v10)];
  if ( v14 == v11 )
    goto LABEL_14;
  v15 = v13 + 96;
  v16 = 0;
  v17 = v11;
  if ( v13 )
    v16 = v13 + 96;
  v18 = 0;
  do
  {
    v19 = *v17++ == v16;
    v18 += v19;
  }
  while ( v14 != (_BYTE *)v17 );
  if ( v18 != 1 || (v20 = 1, *(_QWORD *)&v11[8 * a3] != v15) )
  {
LABEL_14:
    v21 = sub_CB72A0();
    v22 = (__m128i *)v21[4];
    if ( v21[3] - (_QWORD)v22 <= 0x33u )
    {
      v20 = 0;
      sub_CB6200((__int64)v21, "EVL is used as non-last operand in EVL-based recipe\n", 0x34u);
      v11 = v25;
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_43A1420);
      v22[3].m128i_i32[0] = 174420073;
      v20 = 0;
      *v22 = si128;
      v11 = v25;
      v22[1] = _mm_load_si128((const __m128i *)&xmmword_43A1430);
      v22[2] = _mm_load_si128((const __m128i *)&xmmword_43A1440);
      v21[4] += 52LL;
    }
  }
  if ( v11 != v27 )
    _libc_free((unsigned __int64)v11);
  return v20;
}
