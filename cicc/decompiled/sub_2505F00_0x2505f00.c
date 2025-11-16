// Function: sub_2505F00
// Address: 0x2505f00
//
bool __fastcall sub_2505F00(__int64 a1, __int64 a2)
{
  bool result; // al

  result = 1;
  if ( *(_BYTE *)a2 == 85 )
    return (*(_WORD *)(a2 + 2) & 3) != 2;
  return result;
}
