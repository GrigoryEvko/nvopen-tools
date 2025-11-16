// Function: sub_7AFEF0
// Address: 0x7afef0
//
_BOOL8 __fastcall sub_7AFEF0(char *a1, __int64 *a2, __int64 a3, char a4)
{
  __int64 v4; // rbx
  _BOOL8 result; // rax
  char v6; // dl
  const char *v7; // r12
  size_t v8; // rsi
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  __m128i v15; // [rsp+0h] [rbp-50h] BYREF
  __m128i v16; // [rsp+10h] [rbp-40h]
  __m128i v17; // [rsp+20h] [rbp-30h]
  __m128i v18; // [rsp+30h] [rbp-20h]

  sub_7AFD10(a1, a2, a3, a4);
  v4 = *a2;
  result = 0;
  if ( *a2 )
  {
    v6 = *(_BYTE *)(v4 + 8);
    result = 1;
    if ( (v6 & 2) == 0 )
    {
      result = 0;
      if ( (v6 & 1) != 0 && (*(_BYTE *)(v4 + 8) & 0xC) != 0 )
      {
        v15 = _mm_loadu_si128(xmmword_4F06660);
        v16 = _mm_loadu_si128(&xmmword_4F06660[1]);
        v17 = _mm_loadu_si128(&xmmword_4F06660[2]);
        v18 = _mm_loadu_si128(&xmmword_4F06660[3]);
        v7 = *(const char **)(v4 + 16);
        v8 = strlen(v7);
        v9 = sub_87A100(v7, v8, &v15);
        v14 = sub_81B700(
                v9,
                v8,
                v10,
                v11,
                v12,
                v13,
                v15.m128i_i64[0],
                v15.m128i_i64[1],
                v16.m128i_i64[0],
                v16.m128i_i64[1],
                v17.m128i_i64[0],
                v17.m128i_i64[1],
                v18.m128i_i64[0],
                v18.m128i_i64[1]);
        if ( (*(_BYTE *)(v4 + 8) & 8) != 0 )
          return v14 != 0;
        else
          return v14 == 0;
      }
    }
  }
  return result;
}
