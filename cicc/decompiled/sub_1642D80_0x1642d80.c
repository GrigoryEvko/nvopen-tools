// Function: sub_1642D80
// Address: 0x1642d80
//
bool __fastcall sub_1642D80(__int64 a1)
{
  bool result; // al
  unsigned __int64 v2; // rdi
  __int64 v3; // rdx

  result = 0;
  v2 = a1 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v2 )
  {
    if ( *(_BYTE *)(v2 + 16) == 78 )
    {
      v3 = *(_QWORD *)(v2 - 24);
      if ( !*(_BYTE *)(v3 + 16) && (*(_BYTE *)(v3 + 33) & 0x20) != 0 )
        return *(_DWORD *)(v3 + 36) == 76;
    }
  }
  return result;
}
