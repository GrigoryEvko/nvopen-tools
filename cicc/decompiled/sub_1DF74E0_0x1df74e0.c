// Function: sub_1DF74E0
// Address: 0x1df74e0
//
__int64 __fastcall sub_1DF74E0(__int64 a1, __int64 a2)
{
  __int16 v2; // ax

  v2 = *(_WORD *)(a2 + 46);
  if ( (v2 & 4) != 0 || (v2 & 8) == 0 )
    return (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 8LL) >> 26) & 1LL;
  else
    return sub_1E15D00(a2, 0x4000000, 2);
}
