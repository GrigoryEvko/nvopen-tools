// Function: sub_12425E0
// Address: 0x12425e0
//
__int64 __fastcall sub_12425E0(__int64 a1, __int64 *a2)
{
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' in callsites")
    || (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' in callsites") )
  {
    return 1;
  }
  else
  {
    return sub_1241B00(a1, a2);
  }
}
