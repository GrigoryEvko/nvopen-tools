// Function: sub_385F840
// Address: 0x385f840
//
bool __fastcall sub_385F840(__int64 a1)
{
  bool result; // al

  result = sub_385F830(a1);
  if ( !result )
    return *(_DWORD *)(a1 + 8) == 1;
  return result;
}
