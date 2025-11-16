// Function: sub_199CBA0
// Address: 0x199cba0
//
bool __fastcall sub_199CBA0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  bool result; // al
  size_t v4; // rdx

  v2 = *(unsigned int *)(a1 + 8);
  result = 0;
  if ( *(_DWORD *)(a2 + 8) == (_DWORD)v2 )
  {
    v4 = 8 * v2;
    result = 1;
    if ( v4 )
      return memcmp(*(const void **)a1, *(const void **)a2, v4) == 0;
  }
  return result;
}
