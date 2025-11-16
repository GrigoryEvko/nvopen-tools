// Function: sub_D22300
// Address: 0xd22300
//
bool __fastcall sub_D22300(__int64 a1)
{
  bool result; // al
  __int64 v2; // rdx

  result = 0;
  if ( *(_BYTE *)a1 == 85 )
  {
    v2 = *(_QWORD *)(a1 - 32);
    if ( v2 )
    {
      if ( !*(_BYTE *)v2 && *(_QWORD *)(v2 + 24) == *(_QWORD *)(a1 + 80) )
        return *(_DWORD *)(v2 + 36) == 169;
    }
  }
  return result;
}
