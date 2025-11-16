// Function: sub_173DB60
// Address: 0x173db60
//
bool __fastcall sub_173DB60(__int64 a1)
{
  unsigned __int8 v1; // dl
  char v2; // al
  bool result; // al
  char v4; // al
  unsigned __int8 v5; // dl

  v1 = *(_BYTE *)(a1 + 16);
  if ( v1 <= 0x17u )
  {
    result = 0;
    if ( v1 == 5 )
    {
      v4 = *(_BYTE *)(*(_QWORD *)a1 + 8LL);
      if ( v4 == 16 )
        v4 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)a1 + 16LL) + 8LL);
      v5 = v4 - 1;
      result = 1;
      if ( v5 > 5u )
        return *(_WORD *)(a1 + 18) == 52;
    }
  }
  else
  {
    v2 = *(_BYTE *)(*(_QWORD *)a1 + 8LL);
    if ( v2 == 16 )
      v2 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)a1 + 16LL) + 8LL);
    return v1 == 76 || (unsigned __int8)(v2 - 1) <= 5u;
  }
  return result;
}
