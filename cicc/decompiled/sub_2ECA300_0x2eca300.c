// Function: sub_2ECA300
// Address: 0x2eca300
//
bool __fastcall sub_2ECA300(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  bool result; // al

  result = a2 == a4 || a1 == a3;
  if ( !result )
  {
    result = a1 > a3 && a2 < a4;
    if ( !result )
    {
      result = a2 > a4 && a1 > a3 && a1 < a4;
      if ( !result )
        return a3 < a4 && a1 < a3 && a2 > a3;
    }
  }
  return result;
}
