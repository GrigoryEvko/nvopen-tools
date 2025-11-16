// Function: sub_190A670
// Address: 0x190a670
//
bool __fastcall sub_190A670(__int64 a1, __int64 a2)
{
  bool result; // al
  __int64 v3; // rdx
  size_t v4; // rdx

  result = 0;
  if ( *(_DWORD *)a1 == *(_DWORD *)a2 )
  {
    result = 1;
    if ( *(_DWORD *)a1 <= 0xFFFFFFFD )
    {
      result = 0;
      if ( *(_QWORD *)(a1 + 8) == *(_QWORD *)(a2 + 8) )
      {
        v3 = *(unsigned int *)(a1 + 32);
        if ( v3 == *(_DWORD *)(a2 + 32) )
        {
          v4 = 4 * v3;
          result = 1;
          if ( v4 )
            return memcmp(*(const void **)(a1 + 24), *(const void **)(a2 + 24), v4) == 0;
        }
      }
    }
  }
  return result;
}
