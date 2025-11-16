// Function: sub_622500
// Address: 0x622500
//
char *__fastcall sub_622500(const __m128i *a1, int a2)
{
  int v2; // edx
  __int64 v3; // rax
  __int64 v4; // r15
  char *v5; // r13
  unsigned __int64 v6; // rdi
  __int64 v7; // rax
  char *v8; // r13
  __int64 v9; // r9
  __int64 v10; // r11
  char *v11; // r10
  int v12; // ebx
  __int64 v13; // r12
  char *v14; // rdi
  unsigned __int64 v15; // rcx
  char *v16; // rsi
  int v18; // [rsp+Ch] [rbp-C4h]
  int v19; // [rsp+24h] [rbp-ACh] BYREF
  unsigned __int64 v20; // [rsp+28h] [rbp-A8h] BYREF
  __m128i v21; // [rsp+30h] [rbp-A0h] BYREF
  __int16 v22[8]; // [rsp+40h] [rbp-90h] BYREF
  __m128i v23[8]; // [rsp+50h] [rbp-80h] BYREF

  if ( !dword_4CFDDE8 )
  {
    dword_4CFDDE8 = 1;
    v2 = 2;
    v3 = 10;
    do
    {
      ++v2;
      v3 *= 10;
    }
    while ( v2 != 19 );
    dword_4CFDDD8 = 18;
    qword_4CFDDE0 = v3;
  }
  v21 = _mm_loadu_si128(a1);
  if ( v21.m128i_i16[0] < 0 && a2 )
  {
    sub_621710(v21.m128i_i16, (_BOOL4 *)&v19);
    v18 = 1;
  }
  else
  {
    v18 = 0;
  }
  v4 = 7;
  sub_620D80(v23, qword_4CFDDE0);
  while ( (int)sub_621000(v21.m128i_i16, 0, v23[0].m128i_i16, 0) > 0 )
  {
    sub_621760(&v21, v23, v21.m128i_i16, v22, 0, (_BOOL4 *)&v19);
    sub_620E00(v22, 0, (__int64 *)&v20, &v19);
    v23[1].m128i_i64[v4--] = v20;
  }
  v5 = &byte_4CFDE00;
  sub_620E00(&v21, 0, (__int64 *)&v20, &v19);
  v6 = v20;
  v23[1].m128i_i64[(int)v4] = v20;
  if ( v18 )
  {
    byte_4CFDE00 = 45;
    v5 = (char *)&unk_4CFDE01;
  }
  if ( v6 > 9 )
  {
    v7 = (int)sub_622470(v6, v5);
  }
  else
  {
    v5[1] = 0;
    v7 = 1;
    *v5 = v6 + 48;
  }
  v8 = &v5[v7];
  LODWORD(v9) = v4 + 1;
  if ( (_DWORD)v4 != 7 )
  {
    v10 = dword_4CFDDD8;
    v9 = (int)v9;
    v11 = v8;
    v12 = dword_4CFDDD8;
    v13 = dword_4CFDDD8 - 1LL;
    v14 = &v8[dword_4CFDDD8 - 2LL - (unsigned int)(dword_4CFDDD8 - 1)];
    do
    {
      v15 = v23[1].m128i_u64[v9];
      v16 = &v11[v13];
      if ( v12 )
      {
        do
        {
          *v16-- = v15 % 0xA + 48;
          v15 /= 0xAu;
        }
        while ( v16 != v14 );
      }
      ++v9;
      v11 += v10;
      v14 += v10;
    }
    while ( (_DWORD)v9 != 8 );
    v8 += ((unsigned int)(6 - v4) + 1LL) * v10;
  }
  *v8 = 0;
  return &byte_4CFDE00;
}
