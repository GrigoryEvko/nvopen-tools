// Function: sub_9691B0
// Address: 0x9691b0
//
bool __fastcall sub_9691B0(const void *a1, size_t a2, const void *a3, __int64 a4)
{
  bool result; // al

  result = 0;
  if ( a2 == a4 )
  {
    result = 1;
    if ( a2 )
      return memcmp(a1, a3, a2) == 0;
  }
  return result;
}
