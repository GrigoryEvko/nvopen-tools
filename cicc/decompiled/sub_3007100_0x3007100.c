// Function: sub_3007100
// Address: 0x3007100
//
bool __fastcall sub_3007100(__int64 a1)
{
  bool result; // al

  result = sub_30070B0(a1);
  if ( result )
    return *(_BYTE *)(*(_QWORD *)(a1 + 8) + 8LL) == 18;
  return result;
}
