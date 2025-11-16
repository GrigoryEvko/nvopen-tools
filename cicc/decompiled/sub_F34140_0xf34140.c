// Function: sub_F34140
// Address: 0xf34140
//
bool __fastcall sub_F34140(__int64 a1, __int64 a2)
{
  bool result; // al
  char v3; // dl

  result = 0;
  if ( *(_QWORD *)a1 == *(_QWORD *)a2 )
  {
    v3 = *(_BYTE *)(a1 + 24);
    if ( v3 == *(_BYTE *)(a2 + 24) )
      return (!v3 || *(_QWORD *)(a1 + 8) == *(_QWORD *)(a2 + 8) && *(_QWORD *)(a1 + 16) == *(_QWORD *)(a2 + 16))
          && *(_QWORD *)(a1 + 32) == *(_QWORD *)(a2 + 32);
  }
  return result;
}
