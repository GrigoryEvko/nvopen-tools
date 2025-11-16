// Function: sub_BCB420
// Address: 0xbcb420
//
bool __fastcall sub_BCB420(__int64 a1)
{
  bool result; // al
  __int64 v2; // rdx
  size_t v3; // rdx

  result = 0;
  v2 = 8LL * *(unsigned int *)(a1 + 12);
  if ( v2 )
  {
    result = 1;
    v3 = v2 - 8;
    if ( v3 )
      return memcmp((const void *)(*(_QWORD *)(a1 + 16) + 8LL), *(const void **)(a1 + 16), v3) == 0;
  }
  return result;
}
