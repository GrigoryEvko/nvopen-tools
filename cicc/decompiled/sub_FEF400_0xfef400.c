// Function: sub_FEF400
// Address: 0xfef400
//
bool __fastcall sub_FEF400(__int64 a1, __int64 *a2)
{
  bool result; // al

  result = sub_FEF380(a1, a2);
  if ( !result )
    return sub_FEF3D0(a1, a2);
  return result;
}
