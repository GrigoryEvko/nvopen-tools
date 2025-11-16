// Function: sub_14857A0
// Address: 0x14857a0
//
__int64 __fastcall sub_14857A0(_QWORD *a1, __int64 a2, __int64 a3, __m128i a4, __m128i a5)
{
  if ( *(_WORD *)(a2 + 24) == 5 && (*(_BYTE *)(a2 + 26) & 2) != 0 )
    return sub_14851E0(a1, a2, a3, a4, a5);
  else
    return sub_1483CF0(a1, a2, a3, a4, a5);
}
