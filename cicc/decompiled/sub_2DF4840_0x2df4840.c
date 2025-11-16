// Function: sub_2DF4840
// Address: 0x2df4840
//
bool __fastcall sub_2DF4840(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // dl
  bool result; // al
  size_t v4; // rdx

  v2 = *(_BYTE *)(a2 + 8) & 0x3F;
  if ( v2 != (*(_BYTE *)(a1 + 8) & 0x3F) )
    return 0;
  result = *(_BYTE *)(a2 + 8) >> 7 == *(_BYTE *)(a1 + 8) >> 7
        && ((*(_BYTE *)(a2 + 8) & 0x40) != 0) == ((*(_BYTE *)(a1 + 8) & 0x40) != 0);
  if ( !result || *(_QWORD *)(a1 + 16) != *(_QWORD *)(a2 + 16) )
    return 0;
  v4 = 4LL * v2;
  if ( v4 )
    return memcmp(*(const void **)a1, *(const void **)a2, v4) == 0;
  return result;
}
