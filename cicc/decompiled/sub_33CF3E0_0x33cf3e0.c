// Function: sub_33CF3E0
// Address: 0x33cf3e0
//
bool __fastcall sub_33CF3E0(__int64 a1)
{
  bool result; // al
  __int64 v2; // rbx
  char v3; // dl
  __int64 v4; // rbx

  result = *(_DWORD *)(a1 + 24) == 12 || *(_DWORD *)(a1 + 24) == 36;
  if ( result )
  {
    v2 = *(_QWORD *)(a1 + 96);
    if ( *(void **)(v2 + 24) != sub_C33340() )
    {
      v3 = *(_BYTE *)(v2 + 44);
      result = 0;
      v4 = v2 + 24;
      if ( (v3 & 7) != 3 )
        return result;
      return ((*(_BYTE *)(v4 + 20) >> 3) ^ 1) & 1;
    }
    v4 = *(_QWORD *)(v2 + 32);
    result = 0;
    if ( (*(_BYTE *)(v4 + 20) & 7) == 3 )
      return ((*(_BYTE *)(v4 + 20) >> 3) ^ 1) & 1;
  }
  return result;
}
