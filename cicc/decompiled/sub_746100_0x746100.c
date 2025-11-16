// Function: sub_746100
// Address: 0x746100
//
_BOOL8 __fastcall sub_746100(__int64 a1, __int64 a2)
{
  _BOOL8 result; // rax
  __int64 v3; // rax
  char v4; // dl
  __int64 v5; // rax
  char v6; // dl
  __int64 v7; // rdx

  result = 0;
  if ( !*(_BYTE *)(a2 + 145) )
  {
    if ( !*(_BYTE *)(a2 + 153) )
      goto LABEL_22;
    if ( *(_BYTE *)(a1 + 140) == 12 )
    {
      if ( *(_QWORD *)(a1 + 8) )
      {
        v3 = a1;
        do
        {
          v3 = *(_QWORD *)(v3 + 160);
          v4 = *(_BYTE *)(v3 + 140);
        }
        while ( v4 == 12 );
        result = 0;
        if ( v4 != 21 )
        {
          v5 = a1;
          do
          {
            v5 = *(_QWORD *)(v5 + 160);
            v6 = *(_BYTE *)(v5 + 140);
          }
          while ( v6 == 12 );
          result = 0;
          if ( v6 )
          {
            v7 = *(_QWORD *)(a1 + 40);
            if ( !v7 || *(_BYTE *)(v7 + 28) != 3 || **(_QWORD ***)(v7 + 32) != qword_4D049B8 )
            {
LABEL_22:
              if ( *(_BYTE *)(a1 + 184) != 10 || (result = 1, (*(_BYTE *)(a1 + 186) & 0x20) != 0) )
              {
                result = 0;
                if ( (*(_BYTE *)(a1 + 89) & 4) != 0 )
                  return (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL) + 177LL) & 0x10) != 0;
              }
            }
          }
        }
      }
    }
  }
  return result;
}
