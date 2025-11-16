// Function: sub_11F3A90
// Address: 0x11f3a90
//
unsigned __int64 __fastcall sub_11F3A90(double *a1, __int64 a2, char *a3, __int64 a4)
{
  __int64 v5; // rsi
  __int64 v7; // rdi
  int v8; // r13d
  __int64 v10; // rax
  char v11; // al
  __m128i v12; // [rsp+0h] [rbp-40h] BYREF
  __m128i v13[3]; // [rsp+10h] [rbp-30h] BYREF

  v5 = a4;
  v7 = (__int64)a3;
  if ( a4 )
  {
    v11 = *a3;
    if ( *a3 == 80 || v11 == 112 )
    {
      v5 = a4 - 1;
      v7 = (__int64)(a3 + 1);
      v8 = 3;
      goto LABEL_3;
    }
    switch ( v11 )
    {
      case 'F':
      case 'f':
        v5 = a4 - 1;
        v7 = (__int64)(a3 + 1);
        break;
      case 'E':
        v5 = a4 - 1;
        v7 = (__int64)(a3 + 1);
        v8 = 1;
        goto LABEL_3;
      case 'e':
        v5 = a4 - 1;
        v7 = (__int64)(a3 + 1);
        v8 = 0;
        goto LABEL_3;
    }
  }
  v8 = 2;
LABEL_3:
  v12 = 0;
  if ( !v5 || sub_C93C90(v7, v5, 0xAu, (unsigned __int64 *)v13) )
  {
    v13[0] = _mm_loadu_si128(&v12);
    v12.m128i_i64[0] = sub_C7F6B0(v8);
  }
  else
  {
    v10 = v13[0].m128i_i64[0];
    v12.m128i_i8[8] = 1;
    if ( v13[0].m128i_i64[0] >= 0x63uLL )
      v10 = 99;
    v12.m128i_i64[0] = v10;
    v13[0] = _mm_loadu_si128(&v12);
  }
  v12.m128i_i8[8] = 1;
  return sub_C7F6E0(a2, v8, v12.m128i_i64[0], 1, *a1);
}
