// Function: sub_2E85490
// Address: 0x2e85490
//
bool __fastcall sub_2E85490(_DWORD *a1, __int64 a2)
{
  bool result; // al

  result = 0;
  if ( !*(_BYTE *)a2 )
    return *a1 == *(_DWORD *)(a2 + 8);
  return result;
}
