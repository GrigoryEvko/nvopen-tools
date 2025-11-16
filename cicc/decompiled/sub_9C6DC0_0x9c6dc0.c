// Function: sub_9C6DC0
// Address: 0x9c6dc0
//
bool __fastcall sub_9C6DC0(__int64 a1)
{
  bool result; // al
  __int64 v2; // rdx

  result = 0;
  if ( *(_BYTE *)a1 == 85 )
  {
    v2 = *(_QWORD *)(a1 - 32);
    if ( v2 )
    {
      if ( !*(_BYTE *)v2 && *(_QWORD *)(v2 + 24) == *(_QWORD *)(a1 + 80) && (*(_BYTE *)(v2 + 33) & 0x20) != 0 )
        return (unsigned int)(*(_DWORD *)(v2 + 36) - 68) <= 3;
    }
  }
  return result;
}
