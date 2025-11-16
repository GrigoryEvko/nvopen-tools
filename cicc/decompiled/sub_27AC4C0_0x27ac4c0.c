// Function: sub_27AC4C0
// Address: 0x27ac4c0
//
bool __fastcall sub_27AC4C0(__int64 a1, __int64 a2)
{
  bool result; // al
  __int64 v3; // rdx
  size_t v4; // rdx

  result = 0;
  if ( *(_DWORD *)(a1 + 12) == *(_DWORD *)(a2 + 12) && *(_QWORD *)(a1 + 40) == *(_QWORD *)(a2 + 40) )
  {
    v3 = *(unsigned int *)(a1 + 36);
    if ( (_DWORD)v3 == *(_DWORD *)(a2 + 36) )
    {
      v4 = 8 * v3;
      result = 1;
      if ( v4 )
        return memcmp(*(const void **)(a1 + 24), *(const void **)(a2 + 24), v4) == 0;
    }
  }
  return result;
}
