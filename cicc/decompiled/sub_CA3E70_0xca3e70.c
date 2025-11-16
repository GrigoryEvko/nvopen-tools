// Function: sub_CA3E70
// Address: 0xca3e70
//
bool __fastcall sub_CA3E70(__int64 a1)
{
  bool result; // al

  result = sub_CA3AD0(a1);
  if ( result )
    return *(_DWORD *)(a1 + 72) != 1;
  return result;
}
