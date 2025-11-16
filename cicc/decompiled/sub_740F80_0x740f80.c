// Function: sub_740F80
// Address: 0x740f80
//
__m128i *__fastcall sub_740F80(char *s)
{
  size_t v2; // r13
  const __m128i *v3; // rdi
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  unsigned int v8; // eax
  _UNKNOWN *__ptr32 *v9; // r8
  unsigned __int64 v10; // rcx
  __m128i **v11; // r14
  __m128i *v12; // r12
  __m128i *v13; // rbx
  bool v14; // zf
  __m128i *v15; // rax
  char *v16; // rax
  const __m128i *v18; // [rsp+8h] [rbp-28h] BYREF

  v18 = (const __m128i *)sub_724DC0();
  v2 = strlen(s) + 1;
  sub_724C70((__int64)v18, 2);
  v3 = v18;
  v18[8].m128i_i64[0] = (__int64)sub_73CA60(v2);
  v3[11].m128i_i64[0] = v2;
  v3[11].m128i_i64[1] = (__int64)s;
  v8 = sub_72DB90((__int64)v3, 2, v4, v5, v6, v7);
  v10 = (18957679 * (unsigned __int64)v8) >> 32;
  v11 = (__m128i **)(qword_4F07AE0 + 8LL * (v8 % 0x7F7));
  v12 = *v11;
  if ( *v11 )
  {
    v13 = 0;
    while ( 1 )
    {
      v14 = (unsigned int)sub_739430((__int64)v12, (__int64)v18, 1u, v10, v9) == 0;
      v15 = (__m128i *)v12[7].m128i_i64[1];
      if ( !v14 )
        break;
      v13 = v12;
      if ( !v15 )
        goto LABEL_8;
      v12 = (__m128i *)v12[7].m128i_i64[1];
    }
    if ( v13 )
    {
      v13[7].m128i_i64[1] = (__int64)v15;
      v15 = *v11;
    }
  }
  else
  {
LABEL_8:
    v12 = (__m128i *)sub_724D80(2);
    sub_72A510(v18, v12);
    v16 = (char *)sub_724830(v2);
    v12[11].m128i_i64[1] = (__int64)v16;
    strcpy(v16, (const char *)v18[11].m128i_i64[1]);
    sub_73B910((__int64)v12);
    v15 = *v11;
  }
  v12[7].m128i_i64[1] = (__int64)v15;
  *v11 = v12;
  sub_724E30((__int64)&v18);
  return v12;
}
