// Function: sub_641850
// Address: 0x641850
//
__int64 __fastcall sub_641850(__int64 a1)
{
  char v1; // al
  unsigned int v2; // r8d
  __int64 v3; // rdx
  char v5; // al
  const char *v6; // rsi

  while ( 1 )
  {
    v1 = *(_BYTE *)(a1 + 140);
    if ( v1 != 12 )
      break;
    a1 = *(_QWORD *)(a1 + 160);
  }
  v2 = 0;
  if ( (unsigned __int8)(v1 - 9) <= 2u )
  {
    if ( unk_4D049B0 )
    {
      v3 = *(_QWORD *)(a1 + 40);
      if ( v3 )
      {
        if ( *(_BYTE *)(v3 + 28) == 3 && *(_QWORD *)(v3 + 32) == *(_QWORD *)(unk_4D049B0 + 88LL) )
        {
          v5 = *(_BYTE *)(a1 + 89);
          v6 = 0;
          if ( (v5 & 0x40) == 0 )
          {
            if ( (v5 & 8) != 0 )
              v6 = *(const char **)(a1 + 24);
            else
              v6 = *(const char **)(a1 + 8);
          }
          return strcmp(v6, "__infovec") == 0;
        }
      }
    }
  }
  return v2;
}
