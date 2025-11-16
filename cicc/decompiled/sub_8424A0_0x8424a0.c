// Function: sub_8424A0
// Address: 0x8424a0
//
void __fastcall sub_8424A0(__m128i *a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5)
{
  if ( dword_4F077C4 == 2 )
  {
    if ( a1[1].m128i_i8[0] == 5 )
      sub_8422F0(a1, a2);
    else
      sub_831820(a1, (__int64)a2, a3, a4, a5);
    if ( a1[1].m128i_i8[1] != 2 )
      goto LABEL_3;
  }
  else if ( a1[1].m128i_i8[1] != 2 )
  {
LABEL_3:
    sub_6F69D0(a1, 0);
    return;
  }
  if ( word_4D04898 )
    sub_82BB50(a1);
}
