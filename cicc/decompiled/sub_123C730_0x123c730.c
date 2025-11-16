// Function: sub_123C730
// Address: 0x123c730
//
__int64 __fastcall sub_123C730(__int64 a1, __int64 a2)
{
  if ( (unsigned __int8)sub_120AFE0(a1, 463, "expected 'summary' here") )
    return 1;
  if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here") )
    return 1;
  if ( (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here") )
    return 1;
  if ( (unsigned __int8)sub_1210940(a1, a2) )
    return 1;
  if ( *(_DWORD *)(a1 + 240) == 4
    && (*(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176), (unsigned __int8)sub_123C1F0(a1, (_QWORD *)(a2 + 40))) )
  {
    return 1;
  }
  else
  {
    return sub_120AFE0(a1, 13, "expected ')' here");
  }
}
