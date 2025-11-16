// Function: sub_AA4410
// Address: 0xaa4410
//
bool __fastcall sub_AA4410(_BYTE *a1, __int64 a2)
{
  bool result; // al
  __int64 v3; // rcx

  result = 1;
  if ( *(_BYTE *)a2 == 85 )
  {
    v3 = *(_QWORD *)(a2 - 32);
    if ( v3 )
    {
      if ( *(_BYTE *)v3
        || *(_QWORD *)(v3 + 24) != *(_QWORD *)(a2 + 80)
        || (*(_BYTE *)(v3 + 33) & 0x20) == 0
        || (result = 0, (unsigned int)(*(_DWORD *)(v3 + 36) - 68) > 3) )
      {
        result = 1;
        if ( *a1 && !*(_BYTE *)v3 && *(_QWORD *)(v3 + 24) == *(_QWORD *)(a2 + 80) && (*(_BYTE *)(v3 + 33) & 0x20) != 0 )
          return *(_DWORD *)(v3 + 36) != 291;
      }
    }
  }
  return result;
}
