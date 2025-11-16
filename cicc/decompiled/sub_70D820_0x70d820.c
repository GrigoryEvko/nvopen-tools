// Function: sub_70D820
// Address: 0x70d820
//
__int64 __fastcall sub_70D820(__m128i *a1, int a2, int a3, const __m128i *a4, unsigned __int64 a5, char a6, _BOOL4 *a7)
{
  int v8; // r14d
  int v9; // eax
  __int64 result; // rax
  __m128i *v11; // rdi
  __m128i v15[4]; // [rsp+20h] [rbp-40h] BYREF

  v8 = sub_620E90((__int64)a4);
  sub_620DE0(v15, a5);
  v9 = sub_620E90((__int64)a4);
  result = sub_621F20(v15, a4 + 11, v9, a7);
  if ( !*a7 )
  {
    v11 = a1 + 11;
    if ( a3 )
      result = sub_621670(v11, a2, v15[0].m128i_i16, v8, a7);
    else
      result = sub_621340((unsigned __int16 *)v11, a2, v15[0].m128i_i16, v8, a7);
    if ( !a2 && (a6 & 1) != 0 )
      *a7 = 0;
  }
  return result;
}
