// Function: sub_1196680
// Address: 0x1196680
//
__int64 __fastcall sub_1196680(_BYTE *a1, _BYTE *a2)
{
  unsigned int v2; // eax
  unsigned int v3; // r8d
  __int64 v6; // rsi
  char v7; // al
  unsigned int v8; // r8d
  _QWORD *v9[5]; // [rsp-28h] [rbp-28h] BYREF

  v2 = (unsigned __int8)*a2 - 29;
  if ( v2 <= 0x1D )
  {
    v3 = 1;
    if ( v2 <= 0x1B )
    {
      if ( *a2 == 42 )
      {
        LOBYTE(v3) = *a1 == 54;
        return v3;
      }
      return 0;
    }
    return v3;
  }
  if ( *a2 != 59 )
    return 0;
  v3 = 1;
  if ( (unsigned __int8)(*a1 - 54) > 1u )
    return v3;
  v6 = *((_QWORD *)a2 - 8);
  v9[0] = 0;
  v7 = sub_995B10(v9, v6);
  v8 = 0;
  if ( !v7 )
    return (unsigned int)sub_995B10(v9, *((_QWORD *)a2 - 4)) ^ 1;
  return v8;
}
