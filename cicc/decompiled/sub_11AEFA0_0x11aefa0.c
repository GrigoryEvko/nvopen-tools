// Function: sub_11AEFA0
// Address: 0x11aefa0
//
bool __fastcall sub_11AEFA0(const void *a1, __int64 a2, const void *a3, __int64 a4)
{
  bool result; // al

  result = 0;
  if ( a2 == a4 )
  {
    result = 1;
    if ( 4 * a2 )
      return memcmp(a1, a3, 4 * a2) == 0;
  }
  return result;
}
