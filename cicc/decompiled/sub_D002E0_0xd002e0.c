// Function: sub_D002E0
// Address: 0xd002e0
//
bool __fastcall sub_D002E0(__int64 a1, int a2)
{
  bool result; // al
  __int64 v3; // rdx

  result = 0;
  if ( *(_BYTE *)a1 == 85 )
  {
    v3 = *(_QWORD *)(a1 - 32);
    if ( v3 )
    {
      if ( !*(_BYTE *)v3 && *(_QWORD *)(v3 + 24) == *(_QWORD *)(a1 + 80) && (*(_BYTE *)(v3 + 33) & 0x20) != 0 )
        return *(_DWORD *)(v3 + 36) == a2;
    }
  }
  return result;
}
