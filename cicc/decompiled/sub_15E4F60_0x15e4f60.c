// Function: sub_15E4F60
// Address: 0x15e4f60
//
bool __fastcall sub_15E4F60(__int64 a1)
{
  char v1; // dl
  bool result; // al

  v1 = *(_BYTE *)(a1 + 16);
  if ( v1 == 3 )
    return (*(_DWORD *)(a1 + 20) & 0xFFFFFFF) == 0;
  result = 0;
  if ( !v1 && a1 + 72 == (*(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFF8LL) )
    return (*(_DWORD *)(a1 + 32) & 0x400000) == 0;
  return result;
}
