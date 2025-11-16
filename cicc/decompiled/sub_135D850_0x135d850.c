// Function: sub_135D850
// Address: 0x135d850
//
bool __fastcall sub_135D850(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdi
  bool result; // al
  __int64 v4; // rdx

  v2 = a1 & 0xFFFFFFFFFFFFFFF8LL;
  result = 0;
  if ( *(_BYTE *)(v2 + 16) == 78 )
  {
    v4 = *(_QWORD *)(v2 - 24);
    if ( !*(_BYTE *)(v4 + 16) )
    {
      result = (v2 != 0) & (*(_BYTE *)(v4 + 33) >> 5);
      if ( result )
        return *(_DWORD *)(v4 + 36) == a2;
    }
  }
  return result;
}
