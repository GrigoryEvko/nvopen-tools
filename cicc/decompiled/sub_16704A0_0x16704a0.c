// Function: sub_16704A0
// Address: 0x16704a0
//
bool __fastcall sub_16704A0(__int64 a1, __int64 a2)
{
  bool result; // al
  __int64 v3; // rdx
  size_t v4; // rdx

  result = 0;
  if ( *(_BYTE *)(a1 + 16) == *(_BYTE *)(a2 + 16) )
  {
    v3 = *(_QWORD *)(a1 + 8);
    if ( v3 == *(_QWORD *)(a2 + 8) )
    {
      v4 = 8 * v3;
      result = 1;
      if ( v4 )
        return memcmp(*(const void **)a1, *(const void **)a2, v4) == 0;
    }
  }
  return result;
}
