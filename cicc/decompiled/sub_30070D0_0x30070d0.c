// Function: sub_30070D0
// Address: 0x30070d0
//
bool __fastcall sub_30070D0(__int64 a1)
{
  bool result; // al

  result = sub_30070B0(a1);
  if ( result )
    return *(_BYTE *)(*(_QWORD *)(a1 + 8) + 8LL) == 17;
  return result;
}
