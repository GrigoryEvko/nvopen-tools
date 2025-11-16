// Function: sub_2EEE7A0
// Address: 0x2eee7a0
//
bool __fastcall sub_2EEE7A0(__int64 a1, __int64 a2)
{
  bool result; // al

  result = 1;
  if ( (((unsigned __int8)a2 ^ (unsigned __int8)a1) & 7) == 0 )
    return ((a2 ^ a1) & 0xFFFFFFFFFFFFFFF8LL) != 0;
  return result;
}
