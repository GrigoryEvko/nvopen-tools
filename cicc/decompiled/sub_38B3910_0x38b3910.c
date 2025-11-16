// Function: sub_38B3910
// Address: 0x38b3910
//
__int64 __fastcall sub_38B3910(__int64 a1, __int64 a2)
{
  if ( (unsigned __int8)sub_388AF10(a1, 341, "expected 'summary' here") )
    return 1;
  if ( (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here") )
    return 1;
  if ( (unsigned __int8)sub_388AF10(a1, 12, "expected '(' here") )
    return 1;
  if ( (unsigned __int8)sub_388ED60(a1, a2) )
    return 1;
  if ( *(_DWORD *)(a1 + 64) == 4
    && (*(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8), (unsigned __int8)sub_38B33D0(a1, (_QWORD *)(a2 + 40))) )
  {
    return 1;
  }
  else
  {
    return sub_388AF10(a1, 13, "expected ')' here");
  }
}
