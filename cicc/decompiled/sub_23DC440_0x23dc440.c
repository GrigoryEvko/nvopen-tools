// Function: sub_23DC440
// Address: 0x23dc440
//
bool __fastcall sub_23DC440(const void *a1, size_t a2, const void *a3, size_t a4)
{
  bool result; // al

  result = 0;
  if ( a4 <= a2 )
  {
    result = 1;
    if ( a4 )
      return memcmp(a1, a3, a4) == 0;
  }
  return result;
}
