// Function: sub_20C7570
// Address: 0x20c7570
//
bool __fastcall sub_20C7570(__int64 a1, __int64 a2, __int64 a3)
{
  bool result; // al
  char v4; // al
  unsigned __int8 v6; // al
  unsigned __int8 v7; // al

  result = 1;
  if ( a1 != a2 )
  {
    v4 = *(_BYTE *)(a1 + 8);
    if ( v4 == 15 )
      return *(_BYTE *)(a2 + 8) == 15;
    else
      return v4 == 16
          && *(_BYTE *)(a2 + 8) == 16
          && (v6 = sub_1F59570(a1)) != 0
          && *(_QWORD *)(a3 + 8LL * v6 + 120)
          && (v7 = sub_1F59570(a2)) != 0
          && *(_QWORD *)(a3 + 8LL * v7 + 120) != 0;
  }
  return result;
}
