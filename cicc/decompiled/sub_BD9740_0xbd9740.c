// Function: sub_BD9740
// Address: 0xbd9740
//
bool __fastcall sub_BD9740(_BYTE *a1)
{
  bool result; // al

  result = 1;
  if ( a1 )
    return *a1 <= 0x24u && ((1LL << *a1) & 0x140000F000LL) != 0;
  return result;
}
