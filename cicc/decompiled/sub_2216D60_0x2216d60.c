// Function: sub_2216D60
// Address: 0x2216d60
//
int __fastcall sub_2216D60(__int64 a1)
{
  __int64 i; // rax
  _BYTE *(__fastcall *v2)(__int64, __m128i *, char *, void *); // rax
  __m128i v3; // xmm1
  __m128i v4; // xmm2
  __m128i v5; // xmm3
  __m128i v6; // xmm4
  __m128i v7; // xmm5
  __m128i v8; // xmm6
  __m128i v9; // xmm7
  __m128i v10; // xmm0
  __m128i v11; // xmm1
  __m128i v12; // xmm2
  __m128i v13; // xmm3
  __m128i v14; // xmm4
  __m128i v15; // xmm5
  __m128i v16; // xmm6
  __m128i v17; // xmm7
  int result; // eax
  __m128i vars0; // [rsp+0h] [rbp+0h] BYREF
  __m128i vars10; // [rsp+10h] [rbp+10h] BYREF
  __m128i vars20; // [rsp+20h] [rbp+20h] BYREF
  __m128i vars30; // [rsp+30h] [rbp+30h] BYREF
  __m128i vars40; // [rsp+40h] [rbp+40h] BYREF
  __m128i vars50; // [rsp+50h] [rbp+50h] BYREF
  __m128i vars60; // [rsp+60h] [rbp+60h] BYREF
  __m128i vars70; // [rsp+70h] [rbp+70h] BYREF
  __m128i vars80; // [rsp+80h] [rbp+80h] BYREF
  __m128i vars90; // [rsp+90h] [rbp+90h] BYREF
  __m128i varsA0; // [rsp+A0h] [rbp+A0h] BYREF
  __m128i varsB0; // [rsp+B0h] [rbp+B0h] BYREF
  __m128i varsC0; // [rsp+C0h] [rbp+C0h] BYREF
  __m128i varsD0; // [rsp+D0h] [rbp+D0h] BYREF
  __m128i varsE0; // [rsp+E0h] [rbp+E0h] BYREF
  __m128i varsF0; // [rsp+F0h] [rbp+F0h] BYREF
  char vars100; // [rsp+100h] [rbp+100h] BYREF

  for ( i = 0; i != 256; ++i )
    vars0.m128i_i8[i] = i;
  v2 = *(_BYTE *(__fastcall **)(__int64, __m128i *, char *, void *))(*(_QWORD *)a1 + 56LL);
  if ( (char *)v2 == (char *)sub_2216D40 )
  {
    v3 = _mm_loadu_si128(&vars10);
    v4 = _mm_loadu_si128(&vars20);
    v5 = _mm_loadu_si128(&vars30);
    v6 = _mm_loadu_si128(&vars40);
    v7 = _mm_loadu_si128(&vars50);
    *(__m128i *)(a1 + 57) = _mm_loadu_si128(&vars0);
    v8 = _mm_loadu_si128(&vars60);
    v9 = _mm_loadu_si128(&vars70);
    *(__m128i *)(a1 + 73) = v3;
    v10 = _mm_loadu_si128(&vars80);
    *(__m128i *)(a1 + 89) = v4;
    v11 = _mm_loadu_si128(&vars90);
    v12 = _mm_loadu_si128(&varsA0);
    *(__m128i *)(a1 + 105) = v5;
    v13 = _mm_loadu_si128(&varsB0);
    *(__m128i *)(a1 + 121) = v6;
    v14 = _mm_loadu_si128(&varsC0);
    *(__m128i *)(a1 + 137) = v7;
    v15 = _mm_loadu_si128(&varsD0);
    *(__m128i *)(a1 + 153) = v8;
    v16 = _mm_loadu_si128(&varsE0);
    *(__m128i *)(a1 + 169) = v9;
    v17 = _mm_loadu_si128(&varsF0);
    *(__m128i *)(a1 + 185) = v10;
    *(__m128i *)(a1 + 201) = v11;
    *(__m128i *)(a1 + 217) = v12;
    *(__m128i *)(a1 + 233) = v13;
    *(__m128i *)(a1 + 249) = v14;
    *(__m128i *)(a1 + 265) = v15;
    *(__m128i *)(a1 + 281) = v16;
    *(__m128i *)(a1 + 297) = v17;
  }
  else
  {
    v2(a1, &vars0, &vars100, (void *)(a1 + 57));
  }
  *(_BYTE *)(a1 + 56) = 1;
  result = memcmp(&vars0, (const void *)(a1 + 57), 0x100u);
  if ( result )
    *(_BYTE *)(a1 + 56) = 2;
  return result;
}
