// Function: sub_7E2190
// Address: 0x7e2190
//
__m128i *__fastcall sub_7E2190(char *src, int a2, __int64 a3, __int8 a4)
{
  char *v5; // r13
  __m128i *v6; // r12
  char v7; // al
  size_t v9; // rax
  char *v10; // rax

  v5 = src;
  v6 = sub_735FB0(a3, a4, 0);
  if ( !a2 )
  {
    v9 = strlen(src);
    v10 = (char *)sub_7E1510(v9 + 1);
    v5 = strcpy(v10, src);
  }
  v6[10].m128i_i8[14] |= 4u;
  v7 = 3;
  v6->m128i_i64[1] = (__int64)v5;
  if ( (unsigned __int8)a4 > 1u )
    v7 = v5 != 0 && a4 == 2;
  v6[5].m128i_i8[8] = (16 * v7) | v6[5].m128i_i8[8] & 0x8F;
  return v6;
}
