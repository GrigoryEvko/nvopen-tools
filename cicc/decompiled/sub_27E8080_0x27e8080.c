// Function: sub_27E8080
// Address: 0x27e8080
//
__int64 __fastcall sub_27E8080(__int64 *a1, __int64 a2)
{
  unsigned __int16 v2; // ax

  v2 = *(_WORD *)(a2 + 2);
  if ( ((v2 >> 7) & 6) != 0 || (v2 & 1) != 0 )
    return 0;
  else
    return sub_27E6CB0(a1, a2);
}
