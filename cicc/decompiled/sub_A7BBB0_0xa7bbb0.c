// Function: sub_A7BBB0
// Address: 0xa7bbb0
//
bool __fastcall sub_A7BBB0(__int64 a1, size_t a2, const void *a3, size_t a4)
{
  bool result; // al

  result = 0;
  if ( a4 <= a2 )
  {
    result = 1;
    if ( a4 )
      return memcmp((const void *)(a2 - a4 + a1), a3, a4) == 0;
  }
  return result;
}
