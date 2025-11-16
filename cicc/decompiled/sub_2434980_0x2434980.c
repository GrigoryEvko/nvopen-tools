// Function: sub_2434980
// Address: 0x2434980
//
bool __fastcall sub_2434980(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rdx
  bool result; // al
  __int64 v4; // rcx

  v2 = *(_QWORD *)(a2 + 24);
  result = a1[1] != v2 && *a1 != v2;
  if ( result && *(_BYTE *)v2 == 85 )
  {
    v4 = *(_QWORD *)(v2 - 32);
    if ( v4 )
    {
      if ( !*(_BYTE *)v4 && *(_QWORD *)(v4 + 24) == *(_QWORD *)(v2 + 80) && (*(_BYTE *)(v4 + 33) & 0x20) != 0 )
        return (unsigned int)(*(_DWORD *)(v4 + 36) - 210) > 1;
    }
  }
  return result;
}
