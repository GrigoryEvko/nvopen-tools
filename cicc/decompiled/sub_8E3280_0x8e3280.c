// Function: sub_8E3280
// Address: 0x8e3280
//
__int64 __fastcall sub_8E3280(const __m128i *a1, __int64 a2, __m128i **a3)
{
  __m128i *v4; // rax
  unsigned int v5; // r8d

  v4 = sub_8E3250(a1);
  v5 = 0;
  *a3 = v4;
  if ( v4 != a1 && (v5 = 1, a1) && v4 && dword_4F07588 )
    return (a1[2].m128i_i64[0] == 0) | (unsigned __int8)(v4[2].m128i_i64[0] != a1[2].m128i_i64[0]);
  else
    return v5;
}
