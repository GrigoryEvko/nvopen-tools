// Function: sub_1239370
// Address: 0x1239370
//
__int64 __fastcall sub_1239370(__int64 a1, __int64 *a2, _QWORD *a3, __int32 a4)
{
  if ( (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here") )
    return 1;
  if ( (unsigned __int8)sub_1239130(a1, a2, a3, a4) )
    return 1;
  if ( *(_DWORD *)(a1 + 240) == 4
    && (*(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176), (unsigned __int8)sub_1212CC0(a1, (__int64)(a2 + 2))) )
  {
    return 1;
  }
  else
  {
    return sub_120AFE0(a1, 13, "expected ')' here");
  }
}
