// Function: sub_3527650
// Address: 0x3527650
//
__int64 __fastcall sub_3527650(__int64 a1, __int64 *a2)
{
  const char *v2; // rax
  __int64 v3; // rdx
  _QWORD *v4; // rax
  __m128i *v5; // rdx
  __m128i si128; // xmm0
  void *v7; // r13
  unsigned __int8 *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdi
  _BYTE *v11; // rax
  __int64 v13[4]; // [rsp+0h] [rbp-20h] BYREF

  if ( !qword_503D1D0
    || (v2 = sub_2E791E0(a2),
        v13[1] = v3,
        v13[0] = (__int64)v2,
        sub_C931B0(v13, (_WORD *)qword_503D1C8, qword_503D1D0, 0) != -1) )
  {
    v4 = sub_CB72A0();
    v5 = (__m128i *)v4[4];
    if ( v4[3] - (_QWORD)v5 <= 0x20u )
    {
      sub_CB6200((__int64)v4, "Writing Machine CFG for function ", 0x21u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_44E4C10);
      v5[2].m128i_i8[0] = 32;
      *v5 = si128;
      v5[1] = _mm_load_si128((const __m128i *)&xmmword_44E4C20);
      v4[4] += 33LL;
    }
    v7 = sub_CB72A0();
    v8 = (unsigned __int8 *)sub_2E791E0(a2);
    v10 = sub_CB5DB0((__int64)v7, v8, v9, 0);
    v11 = *(_BYTE **)(v10 + 32);
    if ( (unsigned __int64)v11 >= *(_QWORD *)(v10 + 24) )
    {
      sub_CB5D20(v10, 10);
    }
    else
    {
      *(_QWORD *)(v10 + 32) = v11 + 1;
      *v11 = 10;
    }
    sub_3527060(a2);
  }
  return 0;
}
