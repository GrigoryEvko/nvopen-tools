// Function: sub_7F5130
// Address: 0x7f5130
//
__int64 __fastcall sub_7F5130(__int64 a1)
{
  char v1; // al
  unsigned __int8 v2; // cl
  unsigned int v3; // r8d
  __int64 v5; // rcx

  v1 = *(_BYTE *)(a1 + 24);
  while ( v1 == 1 )
  {
    v2 = *(_BYTE *)(a1 + 56);
    if ( v2 == 50 )
    {
      v5 = *(_QWORD *)(a1 + 72);
      v1 = *(_BYTE *)(*(_QWORD *)(v5 + 16) + 24LL);
      if ( *(_BYTE *)(v5 + 24) == 2 )
      {
        a1 = *(_QWORD *)(v5 + 16);
      }
      else if ( v1 == 2 )
      {
        v1 = *(_BYTE *)(v5 + 24);
        a1 = *(_QWORD *)(a1 + 72);
      }
      else
      {
        v1 = *(_BYTE *)(a1 + 24);
      }
    }
    else
    {
      if ( v2 > 0x32u )
      {
        if ( (unsigned __int8)(v2 - 94) > 1u )
          return 0;
      }
      else if ( v2 > 8u || ((1LL << v2) & 0x121) == 0 )
      {
        return 0;
      }
      a1 = *(_QWORD *)(a1 + 72);
      v1 = *(_BYTE *)(a1 + 24);
    }
  }
  v3 = 0;
  if ( v1 == 3 )
    return *(_BYTE *)(*(_QWORD *)(a1 + 56) + 172LL) & 1;
  return v3;
}
