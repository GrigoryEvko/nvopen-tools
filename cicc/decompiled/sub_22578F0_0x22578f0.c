// Function: sub_22578F0
// Address: 0x22578f0
//
bool __fastcall sub_22578F0(__int64 a1)
{
  const char *v1; // rdi
  bool result; // al

  v1 = *(const char **)(a1 + 8);
  result = 1;
  if ( v1 != "NSt8ios_base7failureE" )
  {
    result = 0;
    if ( *v1 != 42 )
      return strcmp(v1, "NSt8ios_base7failureE") == 0;
  }
  return result;
}
