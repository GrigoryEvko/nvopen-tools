// Function: sub_1024060
// Address: 0x1024060
//
bool __fastcall sub_1024060(_DWORD *a1, __int64 a2)
{
  bool result; // al
  __int64 v3; // rdx

  result = 0;
  if ( *(_BYTE *)a2 == 85 )
  {
    v3 = *(_QWORD *)(a2 - 32);
    if ( v3 )
    {
      if ( !*(_BYTE *)v3 && *(_QWORD *)(v3 + 24) == *(_QWORD *)(a2 + 80) )
        return *(_DWORD *)(v3 + 36) == *a1;
    }
  }
  return result;
}
