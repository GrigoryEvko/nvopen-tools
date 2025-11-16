// Function: sub_6F6860
// Address: 0x6f6860
//
void __fastcall sub_6F6860(__m128i *a1, __int64 a2, __int64 *a3, _DWORD *a4)
{
  if ( a3 )
    *a3 = 0;
  if ( a4 )
    *a4 = 0;
  if ( a1[1].m128i_i8[0] == 3 && (a1[1].m128i_i8[3] & 8) != 0 )
    sub_6F63B0(a1, a2, a3, a4);
}
