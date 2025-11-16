// Function: sub_B491E0
// Address: 0xb491e0
//
bool __fastcall sub_B491E0(__int64 a1)
{
  unsigned __int8 v1; // dl
  bool result; // al

  v1 = **(_BYTE **)(a1 - 32);
  result = 0;
  if ( v1 > 0x15u )
    return v1 != 25;
  return result;
}
