// Function: sub_1241A90
// Address: 0x1241a90
//
__int64 __fastcall sub_1241A90(__int64 a1, __int64 a2)
{
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' in allocs")
    || (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' in allocs") )
  {
    return 1;
  }
  else
  {
    return sub_1241420(a1, a2);
  }
}
